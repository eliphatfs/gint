import torch
import unittest
from gint import TensorInterface, bytecode, cdiv
from gint.host.frontend import *
from gint.host.executor import get_executor


@bytecode
def vec_add(a: TensorInterface, b: TensorInterface, c: TensorInterface, REGW: int, WARP: int):
    N, = a.shape
    a, b, c = [make_block_1d(arg, WARP * REGW, arg.strides[0], cdiv(N, WARP * REGW), WARP * REGW) for arg in (a, b, c)]
    fldg_1d(0, a)
    fldg_1d(0, b)
    fadd()
    fstg_1d(0, c)
    halt()


@bytecode
def vec_mul(a: TensorInterface, b: TensorInterface, c: TensorInterface, REGW: int, WARP: int):
    N, = a.shape
    a, b, c = [make_block_1d(arg, WARP * REGW, arg.strides[0], cdiv(N, WARP * REGW), WARP * REGW) for arg in (a, b, c)]
    fldg_1d(0, a)
    fldg_1d(0, b)
    fmul()
    fstg_1d(0, c)
    halt()


class TestIndirectDispatch(unittest.TestCase):

    def test_indirect_add_mul(self):
        N = 128
        REGW = 4
        executor = get_executor()
        WARP = executor.warp_size()
        warps_per_program = cdiv(N, WARP * REGW)

        # Two sets of inputs/outputs
        a1 = torch.randn(N, device='cuda', dtype=torch.float32)
        b1 = torch.randn(N, device='cuda', dtype=torch.float32)
        c1 = torch.zeros(N, device='cuda', dtype=torch.float32)

        a2 = torch.randn(N, device='cuda', dtype=torch.float32)
        b2 = torch.randn(N, device='cuda', dtype=torch.float32)
        c2 = torch.zeros(N, device='cuda', dtype=torch.float32)

        # indices: first warps_per_program warps run add, next run mul
        grid_dim = warps_per_program * 2
        indices = [0] * warps_per_program + [1] * warps_per_program

        executor.execute_indirect(
            programs=[vec_add, vec_mul],
            args_list=[(a1, b1, c1), (a2, b2, c2)],
            indices=indices,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(c1, a1 + b1)
        torch.testing.assert_close(c2, a2 * b2)


if __name__ == '__main__':
    unittest.main()
