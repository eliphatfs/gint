import torch
import unittest
from gint import ProgramTensorInfo, TensorInterface, bytecode, cdiv
from gint.host.frontend import *


@bytecode
def vector_expr2(x: TensorInterface, y: TensorInterface, ILP: int, WARP: int, BLOCK: int):
    # implements cubic easing function x ** 2 * (3 - 2 * x)
    assert x.shape == y.shape
    for arg in (x, y):
        assert arg.typestr == 'f4'
    C, = x.shape
    block = BLOCK
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


def cubic_ease(x: torch.Tensor):
    orig_shape = x.shape
    x = x.view(-1)
    y = torch.empty_like(x)
    vector_expr2(x, y, grid_dim=cdiv(cdiv(len(x), 256), ILP), BLOCK=256, cuda_stream=torch.cuda.current_stream().cuda_stream)
    y_ref = x ** 2 * (3 - 2 * x)
    return y_ref.view(orig_shape), y.view(orig_shape)


class TestCudaGraphCapture(unittest.TestCase):
    
    def test_cubic_ease(self):
        z = [12938, 3]
        torch.manual_seed(42)
        x1 = torch.rand(z, device='cuda', dtype=torch.float32)
        x2 = torch.rand(z, device='cuda', dtype=torch.float32)
        graph = torch.cuda.make_graphed_callables(cubic_ease, (x2,))
        y_ref1g, y_1g = graph(x1)
        y_ref2g, y_2g = graph(x2)
        
        y_ref1 = x1 ** 2 * (3 - 2 * x1)
        y_ref2 = x2 ** 2 * (3 - 2 * x2)
        
        torch.testing.assert_close(y_ref2g, y_ref2)
        torch.testing.assert_close(y_2g, y_ref2)
        torch.testing.assert_close(y_ref1g, y_ref1)
        torch.testing.assert_close(y_1g, y_ref1)
        
        y1r, y1 = cubic_ease(x1)
        torch.testing.assert_close(y1, y_ref1)
        torch.testing.assert_close(y1r, y_ref1)
        y2r, y2 = cubic_ease(x2)
        torch.testing.assert_close(y2, y_ref2)
        torch.testing.assert_close(y2r, y_ref2)
