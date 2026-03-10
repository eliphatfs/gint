import torch
import unittest
from gint import TensorInterface, bytecode, cdiv
from gint.host.frontend import *


@bytecode
def inv4x4_kernel(a: TensorInterface, c: TensorInterface, REGW: int, WARP: int):
    B, D = a.shape
    assert D == 16
    a_block = make_block_2d(a, [B, 16], [a.strides[0], a.strides[1]], [cdiv(B, 32), 1], [32, 0])
    c_block = make_block_2d(c, [B, 16], [c.strides[0], c.strides[1]], [cdiv(B, 32), 1], [32, 0])

    # Load A rows into regs 0-3
    for i in range(4):
        fldg_2dw(4 * i, a_block)
        fstore_reg(i)

    # Initialize identity rows in regs 4-7
    # fpush4 interprets 4 bytes (little-endian) as int8 values converted to float
    # byte value 1 -> 1.0, byte value 0 -> 0.0
    fpush4(0x00000001)  # [1, 0, 0, 0]
    fstore_reg(4)
    fpush4(0x00000100)  # [0, 1, 0, 0]
    fstore_reg(5)
    fpush4(0x00010000)  # [0, 0, 1, 0]
    fstore_reg(6)
    fpush4(0x01000000)  # [0, 0, 0, 1]
    fstore_reg(7)

    # Gauss-Jordan elimination: 4 pivot steps
    for p in range(4):
        # Scale pivot row: multiply A_pivot and I_pivot by 1/A[p][p]
        fload_reg(p)            # [A_p]
        dup_broadcast_w(p)      # [A_p, A[p][p]]
        frcp()                  # [A_p, rcp]
        dupx1()                 # [rcp, A_p, rcp]
        fmul()                  # [rcp, A_p * rcp]
        fstore_reg(p)           # [rcp]
        fload_reg(4 + p)        # [rcp, I_p]
        fmul()                  # [rcp * I_p]
        fstore_reg(4 + p)       # []

        # Eliminate column p from all other rows
        for r in range(4):
            if r == p:
                continue
            # Stack sequence uses A_r (still on stack) for both I and A updates
            fload_reg(r)            # [A_r]
            dup_broadcast_w(p)      # [A_r, coeff]   coeff = A_r[p] (original)
            fload_reg(4 + p)        # [A_r, coeff, I_pivot]
            fmul()                  # [A_r, coeff * I_pivot]
            fload_reg(4 + r)        # [A_r, coeff*I_pivot, I_r]
            fsub()                  # [A_r, I_r - coeff*I_pivot]
            fstore_reg(4 + r)       # [A_r]
            dup_broadcast_w(p)      # [A_r, coeff]   A_r unchanged, coeff recomputed
            fload_reg(p)            # [A_r, coeff, A_pivot]
            fmul()                  # [A_r, coeff * A_pivot]
            frsub()                 # [A_r - coeff*A_pivot]
            fstore_reg(r)           # []

    # Store result rows (identity side = A^{-1})
    for i in range(4):
        fload_reg(4 + i)
        fstg_2dw(4 * i, c_block)

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
            # Ensure invertibility by adding scaled identity
            a = a + 4.0 * torch.eye(4, device='cuda', dtype=torch.float32).unsqueeze(0)
            self._run(a)

    def test_batch_sizes(self):
        for B in [1, 32, 65535]:
            torch.manual_seed(7)
            a = torch.randn(B, 4, 4, device='cuda', dtype=torch.float32)
            a = a + 4.0 * torch.eye(4, device='cuda', dtype=torch.float32).unsqueeze(0)
            self._run(a)


if __name__ == '__main__':
    unittest.main()
