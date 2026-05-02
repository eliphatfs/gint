"""Bytecode kernels and Python wrappers for batched small (N <= 4) matmul
and matrix inverse.

The 4x4 kernels mirror the hand-written reference implementations in
``tests/test_bmm4x4.py`` / ``tests/test_inv4x4.py``. For N < 4 the input is
padded to 4x4 (zeros for matmul, identity-block for inverse — both preserve
the top-left N×N block of the result) and the same 4x4 kernel runs. This
trades a constant amount of extra compute for one shared codegen path.

The conductor uses these via FX rewrite (``gint.conductor.special_ops``).
The kernels are ``SugarProgram`` instances so they go through the standard
per-shape executor cache; building bytecode is amortized across calls.
"""

import math

import torch

from .frontend import (
    make_block_2d,
    fldg_2dw, fstg_2dw,
    fmul, fadd, fsub, frsub, fdiv, fpush4,
    fperm_w, fshuf2,
    dup, dup2, swap, pop, halt,
    dup_broadcast_w, fload_reg, fstore_reg,
)
from .sugar import bytecode
from .utils import cdiv


# ---------------------------------------------------------------------------
# 4x4 batched matmul
# ---------------------------------------------------------------------------

@bytecode
def bmm4x4_kernel(a, b, c, REGW: int, WARP: int):
    """Batched 4x4 matmul: c = a @ b. Tensors are (B, 16) row-major."""
    B, _ = a.shape
    a_block = make_block_2d(a, [B, 16], [a.strides[0], a.strides[1]],
                            [cdiv(B, 32), 1], [32, 0])
    b_block = make_block_2d(b, [B, 16], [b.strides[0], b.strides[1]],
                            [cdiv(B, 32), 1], [32, 0])
    c_block = make_block_2d(c, [B, 16], [c.strides[0], c.strides[1]],
                            [cdiv(B, 32), 1], [32, 0])
    for i in range(4):
        fldg_2dw(4 * i, a_block)            # A[b, i, 0:4]
        for k in range(4):
            dup_broadcast_w(k)              # A[b, i, k]
            fldg_2dw(4 * k, b_block)        # B[b, k, 0:4]
            fmul()                          # partial C[b, i, 0:4]
            swap()
        pop()
        for _ in range(3):
            fadd()
        fstg_2dw(4 * i, c_block)
    halt()


# ---------------------------------------------------------------------------
# 4x4 batched inverse (Cramer's rule with 2x2 block decomposition).
# Reference: https://lxjk.github.io/2017/09/03/Fast-4x4-Matrix-Inverse-with-SSE-SIMD-Explained.html
# ---------------------------------------------------------------------------

def _swz(i0, i1, i2, i3):
    fperm_w(i0, i1, i2, i3)


def _mat2mul(v1_reg, v2_reg):
    fload_reg(v1_reg)
    fload_reg(v2_reg); _swz(0, 3, 0, 3)
    fmul()
    fload_reg(v1_reg); _swz(1, 0, 3, 2)
    fload_reg(v2_reg); _swz(2, 1, 2, 1)
    fmul()
    fadd()


def _mat2adjmul(v1_reg, v2_reg):
    fload_reg(v1_reg); _swz(3, 3, 0, 0)
    fload_reg(v2_reg)
    fmul()
    fload_reg(v1_reg); _swz(1, 1, 2, 2)
    fload_reg(v2_reg); _swz(2, 3, 0, 1)
    fmul()
    frsub()


def _mat2muladj(v1_reg, v2_reg):
    fload_reg(v1_reg)
    fload_reg(v2_reg); _swz(3, 0, 3, 0)
    fmul()
    fload_reg(v1_reg); _swz(1, 0, 3, 2)
    fload_reg(v2_reg); _swz(2, 1, 2, 1)
    fmul()
    frsub()


@bytecode
def inv4x4_kernel(a, c, REGW: int, WARP: int):
    """Batched 4x4 inverse: c = inv(a). Tensors are (B, 16) row-major."""
    B, _ = a.shape
    a_block = make_block_2d(a, [B, 16], [a.strides[0], a.strides[1]],
                            [cdiv(B, 32), 1], [32, 0])
    c_block = make_block_2d(c, [B, 16], [c.strides[0], c.strides[1]],
                            [cdiv(B, 32), 1], [32, 0])

    # reg0..reg3 = rows 0..3 of A
    for i in range(4):
        fldg_2dw(4 * i, a_block)
        fstore_reg(i)

    # detSub = (|A|, |B|, |C|, |D|), four 2x2 determinants packed as a 4-vec
    fload_reg(0); fload_reg(2); fshuf2(0, 2, 0, 2)
    fload_reg(1); fload_reg(3); fshuf2(1, 3, 1, 3)
    fmul()
    fload_reg(0); fload_reg(2); fshuf2(1, 3, 1, 3)
    fload_reg(1); fload_reg(3); fshuf2(0, 2, 0, 2)
    fmul()
    frsub()
    fstore_reg(4)

    # Extract 2x2 blocks A,B,C,D into reg0..reg3
    fload_reg(0); fload_reg(1); dup2()
    fshuf2(0, 1, 0, 1); fstore_reg(0)   # A
    fshuf2(2, 3, 2, 3); fstore_reg(1)   # B
    fload_reg(2); fload_reg(3); dup2()
    fshuf2(0, 1, 0, 1); fstore_reg(2)   # C
    fshuf2(2, 3, 2, 3); fstore_reg(3)   # D

    # D_C = adj(D) * C  -> reg5; A_B = adj(A) * B -> reg6
    _mat2adjmul(3, 2); fstore_reg(5)
    _mat2adjmul(0, 1); fstore_reg(6)

    # detM = detA*detD + detB*detC - tr(A_B * swz(D_C, 0,2,1,3))
    fload_reg(4); fperm_w(0, 0, 1, 1)
    fload_reg(4); fperm_w(3, 3, 2, 2)
    fmul()
    dup_broadcast_w(0)
    swap()
    fperm_w(3, 3, 3, 3)
    fadd()
    fload_reg(6)
    fload_reg(5); _swz(0, 2, 1, 3)
    fmul()
    dup(); _swz(2, 3, 0, 1); fadd()
    dup(); _swz(1, 0, 3, 2); fadd()
    frsub()

    # rDetM = (1, -1, -1, 1) / detM
    fpush4(0x01ffff01)
    fdiv()
    fstore_reg(7)

    # Pair 1: Z_ then X_ (use A, B, D_C)
    _mat2muladj(0, 5)
    fload_reg(4); fperm_w(2, 2, 2, 2)
    fload_reg(1); fmul()
    fsub()
    fload_reg(7); fmul()
    _mat2mul(1, 5)
    fload_reg(4); fperm_w(3, 3, 3, 3)
    fload_reg(0); fmul()
    fsub()
    fload_reg(7); fmul()
    fstore_reg(1)
    fstore_reg(0)

    # Pair 2: W_ then Y_ (use C, D, A_B)
    _mat2mul(2, 6)
    fload_reg(4); fperm_w(0, 0, 0, 0)
    fload_reg(3); fmul()
    fsub()
    fload_reg(7); fmul()
    _mat2muladj(3, 6)
    fload_reg(4); fperm_w(1, 1, 1, 1)
    fload_reg(2); fmul()
    fsub()
    fload_reg(7); fmul()
    fstore_reg(3)
    fstore_reg(2)

    # Final output rows via VecShuffle
    # reg map: reg0=Z_, reg1=X_, reg2=W_, reg3=Y_
    fload_reg(1); fload_reg(3); dup2()
    fshuf2(3, 1, 3, 1); fstg_2dw(0,  c_block)
    fshuf2(2, 0, 2, 0); fstg_2dw(4,  c_block)
    fload_reg(0); fload_reg(2); dup2()
    fshuf2(3, 1, 3, 1); fstg_2dw(8,  c_block)
    fshuf2(2, 0, 2, 0); fstg_2dw(12, c_block)
    halt()


# ---------------------------------------------------------------------------
# Python wrappers (handle arbitrary batch shape and N <= 4 via padding)
# ---------------------------------------------------------------------------

def _batch_flatten(t: torch.Tensor):
    """Return ``(B, batch_shape, N)`` for a tensor with shape ``(..., N, N)``."""
    n2 = t.shape[-1]
    n1 = t.shape[-2]
    assert n1 == n2, f"matrix must be square, got {(n1, n2)}"
    batch_shape = t.shape[:-2]
    B = 1
    for d in batch_shape:
        B *= int(d)
    return B, batch_shape, int(n1)


def gint_bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Batched ``a @ b`` for matrices of size N <= 4 (N square, equal in a/b).

    Supports arbitrary batch shape (anything before the trailing two dims).
    For N == 1 falls back to elementwise multiply (no kernel launch needed).
    For N < 4 pads the inputs with zeros and runs the 4x4 kernel — zero
    padding is identity for ``A @ B`` (the top-left N×N of the result is
    the true ``a @ b``).
    """
    B, batch_shape, N = _batch_flatten(a)
    Bb, _, Nb = _batch_flatten(b)
    assert B == Bb and N == Nb, "bmm operand shape mismatch"
    assert N <= 4, f"gint_bmm only supports N <= 4, got N={N}"

    if N == 1:
        # 1×1 matmul ≡ elementwise multiply; reshape to keep trailing (1,1).
        return (a * b).contiguous()

    a_flat = a.reshape(B, N, N).contiguous()
    b_flat = b.reshape(B, N, N).contiguous()

    if N == 4:
        c_flat = torch.empty(B, 16, dtype=a.dtype, device=a.device)
        bmm4x4_kernel(a_flat.view(B, 16), b_flat.view(B, 16), c_flat,
                      grid_dim=cdiv(B, 32))
        return c_flat.view(*batch_shape, 4, 4)

    # N in {2, 3}: pad to 4×4 with zeros, run 4×4 kernel, slice back.
    a_pad = a_flat.new_zeros(B, 4, 4)
    b_pad = b_flat.new_zeros(B, 4, 4)
    a_pad[:, :N, :N] = a_flat
    b_pad[:, :N, :N] = b_flat
    c_pad = torch.empty(B, 16, dtype=a.dtype, device=a.device)
    bmm4x4_kernel(a_pad.view(B, 16), b_pad.view(B, 16), c_pad,
                  grid_dim=cdiv(B, 32))
    out = c_pad.view(B, 4, 4)[:, :N, :N].contiguous()
    return out.view(*batch_shape, N, N)


def gint_inv(a: torch.Tensor) -> torch.Tensor:
    """Batched ``inv(a)`` for matrices of size N <= 4 (square).

    For N == 1 falls back to elementwise reciprocal.
    For N < 4 pads with an identity block (zeros off-diagonal, ones on the
    extra diagonal entries) so the padded matrix is block-diagonal
    ``diag(A, I_{4-N})``; its inverse is ``diag(inv(A), I_{4-N})`` and the
    top-left N×N is ``inv(A)``.
    """
    B, batch_shape, N = _batch_flatten(a)
    assert N <= 4, f"gint_inv only supports N <= 4, got N={N}"

    if N == 1:
        return a.reciprocal().contiguous()

    a_flat = a.reshape(B, N, N).contiguous()

    if N == 4:
        c_flat = torch.empty(B, 16, dtype=a.dtype, device=a.device)
        inv4x4_kernel(a_flat.view(B, 16), c_flat, grid_dim=cdiv(B, 32))
        return c_flat.view(*batch_shape, 4, 4)

    # N in {2, 3}: embed A into the top-left of an identity-padded 4×4.
    a_pad = torch.eye(4, dtype=a.dtype, device=a.device).expand(B, 4, 4).contiguous()
    a_pad[:, :N, :N] = a_flat
    c_pad = torch.empty(B, 16, dtype=a.dtype, device=a.device)
    inv4x4_kernel(a_pad.view(B, 16), c_pad, grid_dim=cdiv(B, 32))
    out = c_pad.view(B, 4, 4)[:, :N, :N].contiguous()
    return out.view(*batch_shape, N, N)
