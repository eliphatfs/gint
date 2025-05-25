import torch
import unittest
from gint import ProgramTensorInfo, TensorInterface, bytecode, cdiv
from gint.host.frontend import *


@bytecode
def basic_expr1(a: TensorInterface, b: TensorInterface, c: TensorInterface, ILP: int, WARP: int):
    assert a.shape == b.shape == c.shape
    for arg in (a, b, c):
        assert arg.typestr == 'f4'
    B, C = a.shape
    ldtinfos()
    for i in range(0, C, WARP):
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


@bytecode
def vector_expr2(x: TensorInterface, y: TensorInterface, ILP: int, WARP: int, BLOCK: int):
    # implements cubic easing function x ** 2 * (3 - 2 * x)
    assert x.shape == y.shape
    for arg in (x, y):
        assert arg.typestr == 'f4'
    C, = x.shape
    block = BLOCK
    ldtinfos()
    for i in range(0, block, WARP):
        ldg_f1_float(i, x)
        immf(0, 3.0)  # 3, x, 0, 0
        immf(2, -2.0)  # 3, x, -2, 0
        fma_f0_f1_f2()  # 3 - 2x, x, -2, 0
        movf(3, 0)  # 3 - 2x, x, -2, 3 - 2x
        movf(0, 1)  # x, x, -2, 3 - 2x
        mul_f0_f1()  # x ** 2, x, -2, 3 - 2x
        movf(1, 3)  # x ** 2, 3 - 2x
        mul_f0_f1()
        stg_f0_float(i, y)
    halt()
    return [ProgramTensorInfo(4, arg.strides[0], C, [arg.strides[0] * block], [cdiv(C, block)], [block]) for arg in (x, y)]


class TestFrontendExpression(unittest.TestCase):
    
    def test_expr_1(self):
        for s in [1, 4, 6, 31, 32, 1000, 200000]:
            for p in [1, 16, 18, 32, 64, 256]:
                torch.manual_seed(42)
                a = torch.rand(s, p, device='cuda', dtype=torch.float32) + 1e-6
                b = torch.rand(s, p, device='cuda', dtype=torch.float32) + 1e-6
                c = torch.empty(s, p, device='cuda', dtype=torch.float32)
                basic_expr1(a, b, c, grid_dim=cdiv(s, ILP))
                c_ref = -((a + b + b) / b - b)
                torch.testing.assert_close(c, c_ref)


    def test_expr_2(self):
        for z in [1, 5, 16, 17, 31, 32, 33, 1023, 1024, 10000000]:
            torch.manual_seed(42)
            x = torch.rand(z, device='cuda', dtype=torch.float32)
            y = torch.empty(z, device='cuda', dtype=torch.float32)
            vector_expr2(x, y, grid_dim=cdiv(cdiv(z, 256), ILP), BLOCK=256)
            y_ref = x ** 2 * (3 - 2 * x)
            torch.testing.assert_close(y, y_ref)
