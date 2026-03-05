import torch
import unittest
from gint import TensorInterface, bytecode, cdiv
from gint.host.frontend import *

@bytecode
def bmm4x4_kernel(a: TensorInterface, b: TensorInterface, c: TensorInterface, REGW: int, WARP: int):
    B, D = a.shape
    
    # A block for column load: stride_1=batch, stride_2=row_stride.
    # We set stride_1 to batch stride * 32 because each warp handles 32 batch elements.
    # Wait, the frontend make_block_2d already handles the cdiv(B, 32) part if we set grid_dim.
    # Actually, we want Lane t to handle Batch t.
    # So stride_1 should be a.strides[0].
    assert a.shape == b.shape == c.shape
    assert D == 16
    a_block = make_block_2d(a, [B, 16], [a.strides[0], a.strides[1]], [cdiv(B, 32), 1], [32, 0])
    b_block = make_block_2d(b, [B, 16], [b.strides[0], b.strides[1]], [cdiv(B, 32), 1], [32, 0])
    c_block = make_block_2d(c, [B, 16], [c.strides[0], c.strides[1]], [cdiv(B, 32), 1], [32, 0])

    for i in range(4):
        
        # C[b, i, 0:4]
        fldg_2dw(4 * i, a_block) # A[b, i, 0:4]
        for k in range(4):
            dup_broadcast_w(k)  # A[b, i, k]
            fldg_2dw(4 * k, b_block) # B[b, k, 0:4]
            fmul()  # C[b, i, 0:4] += .
            swap()
        pop()  # remove A[b, i, 0:4], remaining 4 adds
        for k in range(3):
            fadd()
        fstg_2dw(4 * i, c_block)

    halt()


class TestBMM4x4(unittest.TestCase):
    def test_bmm4x4(self):
        for B in [1, 32, 64, 1000, 1000000]:
            torch.manual_seed(42)
            a = torch.randn(B, 4, 4, device='cuda', dtype=torch.float32)
            b = torch.randn(B, 4, 4, device='cuda', dtype=torch.float32)
            c = torch.zeros(B, 4, 4, device='cuda', dtype=torch.float32)
            
            bmm4x4_kernel(a.view(B, 16), b.view(B, 16), c.view(B, 16), grid_dim=cdiv(B, 32))
            
            c_ref = torch.bmm(a, b)
            torch.testing.assert_close(c, c_ref, atol=1e-5, rtol=1e-5)

    def test_profile_bmm4x4(self):
        torch.manual_seed(42)
        for it in range(2):
            B = 65535
            a = torch.empty(B, 4, 4, device='cuda', dtype=torch.float32)
            b = torch.empty(B, 4, 4, device='cuda', dtype=torch.float32)
            c = torch.empty(B, 4, 4, device='cuda', dtype=torch.float32)
        
            bmm4x4_kernel(a.view(B, 16), b.view(B, 16), c.view(B, 16), grid_dim=cdiv(B, 32))
            torch.bmm(a, b)

if __name__ == '__main__':
    unittest.main()
