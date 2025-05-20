import torch
import unittest
from gint import ProgramTensorInfo, TensorInterface, bytecode, cdiv
from gint.host.frontend import *


@bytecode
def basic_expr1(a: TensorInterface, b: TensorInterface, c: TensorInterface, ILP: int):
    assert a.shape == b.shape == c.shape
    for arg in (a, b, c):
        assert arg.typestr == 'f4'
    B, C = a.shape
    ldtinfos()
    for i in range(0, C, 32):
        ldg_f1_float(i, a)
        movf(0, 1)
        ldg_f1_float(i, b)
        add_f0_f1()
        add_f0_f1()
        div_f0_f1()
        sub_f0_f1()
        neg_f0()
        stg_f0_float(i, c)
    halt()
    return [ProgramTensorInfo(arg.elm_size, arg.strides[1], C, [arg.strides[0]], [B], [0]) for arg in (a, b, c)]


class TestFrontendExpression(unittest.TestCase):
    
    def test_expr_1(self):
        for s in [1, 4, 6, 31, 32, 1000, 200000]:
            for p in [1, 16, 18, 32, 64, 256]:
                torch.manual_seed(42)
                a = torch.randn(s, p, device='cuda', dtype=torch.float32)
                b = torch.randn(s, p, device='cuda', dtype=torch.float32)
                c = torch.empty(s, p, device='cuda', dtype=torch.float32)
                basic_expr1(a, b, c, grid_dim=cdiv(s, ILP))
                c_ref = -((a + b + b) / b - b)
                torch.testing.assert_close(c, c_ref)
