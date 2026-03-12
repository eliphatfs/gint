import torch
import unittest
from gint import TensorInterface, bytecode, cdiv
from gint.host.frontend import *


# ---------------------------------------------------------------------------
# Helper bytecode-generation functions (called at trace time, not on-device)
# ---------------------------------------------------------------------------

def _swz(i0, i1, i2, i3):
    """Permute TOS in-place."""
    fperm_w(i0, i1, i2, i3)


def _mat2mul(v1_reg, v2_reg):
    """
    Stack: [] -> [result]
    result = v1 * swz(v2,0303) + swz(v1,1032) * swz(v2,2121)
    """
    fload_reg(v1_reg)
    fload_reg(v2_reg); _swz(0, 3, 0, 3)
    fmul()                                    # [v1*swz(v2,0303)]
    fload_reg(v1_reg); _swz(1, 0, 3, 2)
    fload_reg(v2_reg); _swz(2, 1, 2, 1)
    fmul()                                    # [term1, swz(v1)*swz(v2)]
    fadd()


def _mat2adjmul(v1_reg, v2_reg):
    """
    Stack: [] -> [result]
    result = swz(v1,3300)*v2 - swz(v1,1122)*swz(v2,2301)   (= adj(v1)*v2)
    """
    fload_reg(v1_reg); _swz(3, 3, 0, 0)
    fload_reg(v2_reg)
    fmul()                                    # [term1]
    fload_reg(v1_reg); _swz(1, 1, 2, 2)
    fload_reg(v2_reg); _swz(2, 3, 0, 1)
    fmul()                                    # [term1, term2]
    frsub()                                   # term1 - term2  (frsub = second - top)


def _mat2muladj(v1_reg, v2_reg):
    """
    Stack: [] -> [result]
    result = v1*swz(v2,3030) - swz(v1,1032)*swz(v2,2121)   (= v1*adj(v2))
    """
    fload_reg(v1_reg)
    fload_reg(v2_reg); _swz(3, 0, 3, 0)
    fmul()                                    # [term1]
    fload_reg(v1_reg); _swz(1, 0, 3, 2)
    fload_reg(v2_reg); _swz(2, 1, 2, 1)
    fmul()                                    # [term1, term2]
    frsub()                                   # term1 - term2


@bytecode
def inv4x4_kernel(a: TensorInterface, c: TensorInterface, REGW: int, WARP: int):
    """
    4x4 matrix inverse via Cramer's rule with block-matrix SIMD decomposition.
    Reference: https://lxjk.github.io/2017/09/03/Fast-4x4-Matrix-Inverse-with-SSE-SIMD-Explained.html

    Register map:
      reg0=A  reg1=B  reg2=C  reg3=D     (2x2 sub-blocks)
      reg4=detSub=(|A|,|B|,|C|,|D|)
      reg5=D_C=adj(D)*C   reg6=A_B=adj(A)*B
      reg7=rDetM=(1,-1,-1,1)/det(M)
    Max stack depth: 4  (pool slots 0-3, safe with all 8 regs occupying pool[4..11])
    """
    B, D = a.shape
    assert D == 16
    a_block = make_block_2d(a, [B, 16], [a.strides[0], a.strides[1]], [cdiv(B, 32), 1], [32, 0])
    c_block = make_block_2d(c, [B, 16], [c.strides[0], c.strides[1]], [cdiv(B, 32), 1], [32, 0])

    # ------------------------------------------------------------------
    # Load matrix rows: reg0..reg3 = row0..row3
    # ------------------------------------------------------------------
    for i in range(4):
        fldg_2dw(4 * i, a_block)
        fstore_reg(i)

    # ------------------------------------------------------------------
    # detSub = (|A|, |B|, |C|, |D|) -- four 2x2 determinants packed
    #
    # detSub = VecShuffle(r0,r2, 0,2,0,2) * VecShuffle(r1,r3, 1,3,1,3)
    #        - VecShuffle(r0,r2, 1,3,1,3) * VecShuffle(r1,r3, 0,2,0,2)
    # ------------------------------------------------------------------
    fload_reg(0); fload_reg(2); fshuf2(0, 2, 0, 2)   # (m00,m02,m20,m22)
    fload_reg(1); fload_reg(3); fshuf2(1, 3, 1, 3)   # (m11,m13,m31,m33)
    fmul()                                            # term1
    fload_reg(0); fload_reg(2); fshuf2(1, 3, 1, 3)   # (m01,m03,m21,m23)
    fload_reg(1); fload_reg(3); fshuf2(0, 2, 0, 2)   # (m10,m12,m30,m32)
    fmul()                                            # term1, term2
    frsub()                                           # detSub = term1 - term2
    fstore_reg(4)

    # ------------------------------------------------------------------
    # Extract 2x2 blocks A,B,C,D using VecShuffle_0101 / VecShuffle_2323
    #   A = (r0[0],r0[1],r1[0],r1[1])  B = (r0[2],r0[3],r1[2],r1[3])
    #   C = (r2[0],r2[1],r3[0],r3[1])  D = (r2[2],r2[3],r3[2],r3[3])
    # Compute A on stack then B: store B first (reg1) then A (reg0),
    # similarly C then D, eliminating the temp-rename moves.
    # ------------------------------------------------------------------
    fload_reg(0); fload_reg(1); fshuf2(0, 1, 0, 1)   # [A]
    fload_reg(0); fload_reg(1); fshuf2(2, 3, 2, 3)   # [A, B]
    fstore_reg(1)                                      # reg1 = B
    fstore_reg(0)                                      # reg0 = A
    fload_reg(2); fload_reg(3); fshuf2(0, 1, 0, 1)   # [C]
    fload_reg(2); fload_reg(3); fshuf2(2, 3, 2, 3)   # [C, D]
    fstore_reg(3)                                      # reg3 = D
    fstore_reg(2)                                      # reg2 = C

    # ------------------------------------------------------------------
    # D_C = adj(D)*C  -> reg5
    # A_B = adj(A)*B  -> reg6
    # ------------------------------------------------------------------
    _mat2adjmul(3, 2); fstore_reg(5)   # D_C
    _mat2adjmul(0, 1); fstore_reg(6)   # A_B

    # ------------------------------------------------------------------
    # detM = detA*detD + detB*detC - tr(A_B * swz(D_C, 0,2,1,3))
    # ------------------------------------------------------------------

    # detA * detD  (depth stays <= 4)
    fload_reg(4); fperm_w(0, 0, 0, 0)         # [detA]
    fload_reg(4); fperm_w(3, 3, 3, 3)         # [detA, detD]
    fmul()                                     # [detA*detD]

    # detB * detC
    fload_reg(4); fperm_w(1, 1, 1, 1)         # [detA*detD, detB]
    fload_reg(4); fperm_w(2, 2, 2, 2)         # [detA*detD, detB, detC]
    fmul()                                     # [detA*detD, detB*detC]
    fadd()                                     # [term12]

    # trace(A_B * swz(D_C, 0,2,1,3)) via horizontal sum
    fload_reg(6)                               # [term12, A_B]
    fload_reg(5); _swz(0, 2, 1, 3)            # [term12, A_B, swz(D_C,0213)]
    fmul()                                     # [term12, prod]
    dup(); _swz(2, 3, 0, 1); fadd()           # [term12, halfsum]
    dup(); _swz(1, 0, 3, 2); fadd()           # [term12, trace_bcast]
    frsub()                                    # [detM = term12 - trace]

    # rDetM = adjSignMask / detM,  adjSignMask = (1,-1,-1,1)
    # Build adjSignMask: push (1,1,1,1) and (-1,-1,-1,-1), shuffle+swizzle
    fpush(1.0); fpush(-1.0)
    fshuf2(0, 1, 0, 1)                         # (1,1,-1,-1)
    _swz(0, 3, 3, 0)                           # (1,-1,-1,1)
    swap()                                     # [(1,-1,-1,1), detM]
    frdiv()                                    # [rDetM = adjSignMask/detM]
    fstore_reg(7)

    # ------------------------------------------------------------------
    # Cofactor blocks scaled by rDetM.
    #
    # Register clobber conflict: X_ must write reg0 (A), W_ must write
    # reg1 (B), but Z_ reads both A and B; Z_ writes reg3 (D) but W_
    # reads D.  Solve by computing pairs that share inputs together:
    #
    #   Pair 1 (use A=reg0, B=reg1, D_C=reg5):
    #     Z_ = detC*B - Mat2MulAdj(A, D_C)  -> reg0
    #     X_ = detD*A - Mat2Mul(B, D_C)     -> reg1
    #   Pair 2 (use C=reg2, D=reg3, A_B=reg6):
    #     W_ = detA*D - Mat2Mul(C, A_B)     -> reg2
    #     Y_ = detB*C - Mat2MulAdj(D, A_B)  -> reg3
    #
    # Stack discipline: compute mat2mul/mat2muladj FIRST (base depth 0,
    # peak +3), then load det×block (at base depth 1, peak +3 → max 4).
    # With all 8 regs in use, max safe depth = 4 (pool[0..3]).
    # The second computation in each pair runs at base depth 1 (first
    # result on stack), reaching max depth 4 — exactly safe.
    # ------------------------------------------------------------------

    # --- Pair 1: Z_ then X_ (both use A, B, D_C) ---
    # Z_ = detC*B - Mat2MulAdj(A, D_C)
    _mat2muladj(0, 5)                          # [mat2muladj(A,D_C)]  depth 1
    fload_reg(4); fperm_w(2, 2, 2, 2)         # [mat, detC]          depth 2
    fload_reg(1); fmul()                       # [mat, detC*B]        depth 2
    fsub()                                     # [Z_]                 depth 1
    fload_reg(7); fmul()                       # [Z_scaled]           depth 1

    # X_ = detD*A - Mat2Mul(B, D_C)   (Z_scaled at depth 1, safe peak = 4)
    _mat2mul(1, 5)                             # [Z_, mat2mul(B,D_C)] depth 2
    fload_reg(4); fperm_w(3, 3, 3, 3)         # [Z_, mat, detD]      depth 3
    fload_reg(0); fmul()                       # [Z_, mat, detD*A]    depth 3
    fsub()                                     # [Z_, X_]             depth 2
    fload_reg(7); fmul()                       # [Z_, X_scaled]       depth 2
    fstore_reg(1)                              # reg1 = X_            depth 1
    fstore_reg(0)                              # reg0 = Z_            depth 0

    # --- Pair 2: W_ then Y_ (both use C, D, A_B) ---
    # W_ = detA*D - Mat2Mul(C, A_B)
    _mat2mul(2, 6)                             # [mat2mul(C,A_B)]     depth 1
    fload_reg(4); fperm_w(0, 0, 0, 0)         # [mat, detA]          depth 2
    fload_reg(3); fmul()                       # [mat, detA*D]        depth 2
    fsub()                                     # [W_]                 depth 1
    fload_reg(7); fmul()                       # [W_scaled]           depth 1

    # Y_ = detB*C - Mat2MulAdj(D, A_B)   (W_scaled at depth 1)
    _mat2muladj(3, 6)                          # [W_, mat2muladj(D,A_B)] depth 2
    fload_reg(4); fperm_w(1, 1, 1, 1)         # [W_, mat, detB]      depth 3
    fload_reg(2); fmul()                       # [W_, mat, detB*C]    depth 3
    fsub()                                     # [W_, Y_]             depth 2
    fload_reg(7); fmul()                       # [W_, Y_scaled]       depth 2
    fstore_reg(3)                              # reg3 = Y_            depth 1
    fstore_reg(2)                              # reg2 = W_            depth 0

    # ------------------------------------------------------------------
    # Final output rows via VecShuffle:
    #   out_row0 = VecShuffle(X_, Y_, 3,1,3,1)
    #   out_row1 = VecShuffle(X_, Y_, 2,0,2,0)
    #   out_row2 = VecShuffle(Z_, W_, 3,1,3,1)
    #   out_row3 = VecShuffle(Z_, W_, 2,0,2,0)
    # reg map: reg0=Z_, reg1=X_, reg2=W_, reg3=Y_
    # ------------------------------------------------------------------
    fload_reg(1); fload_reg(3); dup2()
    fshuf2(3, 1, 3, 1); fstg_2dw(0,  c_block)
    fshuf2(2, 0, 2, 0); fstg_2dw(4,  c_block)
    fload_reg(0); fload_reg(2); dup2()
    fshuf2(3, 1, 3, 1); fstg_2dw(8,  c_block)
    fshuf2(2, 0, 2, 0); fstg_2dw(12, c_block)

    halt()


class TestInv4x4(unittest.TestCase):

    def _run(self, a: torch.Tensor):
        B = a.shape[0]
        c = torch.zeros(B, 16, device='cuda', dtype=torch.float32)
        inv4x4_kernel(a.view(B, 16), c, grid_dim=cdiv(B, 32))
        c_ref = torch.linalg.inv(a).reshape(B, 16)
        torch.testing.assert_close(c, c_ref, atol=1e-3, rtol=1e-3)

    def test_identity(self):
        for B in [1, 32, 64]:
            a = torch.eye(4, device='cuda', dtype=torch.float32).unsqueeze(0).expand(B, -1, -1).contiguous()
            self._run(a)

    def test_diagonal(self):
        for B in [1, 32]:
            diag = torch.rand(B, 4, device='cuda', dtype=torch.float32) + 0.5
            a = torch.diag_embed(diag)
            self._run(a)

    def test_known_matrix(self):
        for B in [1, 32, 1000]:
            torch.manual_seed(42)
            a = torch.randn(B, 4, 4, device='cuda', dtype=torch.float32)
            a = a + 4.0 * torch.eye(4, device='cuda', dtype=torch.float32).unsqueeze(0)
            self._run(a)

    def test_batch_sizes(self):
        for B in [1, 32, 65535]:
            torch.manual_seed(7)
            a = torch.randn(B, 4, 4, device='cuda', dtype=torch.float32)
            a = a + 4.0 * torch.eye(4, device='cuda', dtype=torch.float32).unsqueeze(0)
            self._run(a)

    def test_profile_inv4x4(self):
        torch.manual_seed(42)
        B = 65535
        a = torch.randn(B, 4, 4, device='cuda', dtype=torch.float32)
        a = a + 4.0 * torch.eye(4, device='cuda', dtype=torch.float32).unsqueeze(0)
        c = torch.zeros(B, 16, device='cuda', dtype=torch.float32)
        for _ in range(2):
            inv4x4_kernel(a.view(B, 16), c, grid_dim=cdiv(B, 32))
            torch.linalg.inv(a)


if __name__ == '__main__':
    unittest.main()
