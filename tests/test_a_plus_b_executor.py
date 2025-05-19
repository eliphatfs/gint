import numpy
import torch
import unittest
from gint.kernel.interpreter.main import ILP
from gint.host.executor import BaseExecutableProgram, ProgramData, ProgramTensorInfo, TensorInterface
from gint.host.utils import cdiv


class BatchAddProgram(BaseExecutableProgram):
    
    def get_program(self, a: TensorInterface, b: TensorInterface, c: TensorInterface) -> ProgramData:
        assert a.shape == b.shape == c.shape
        for arg in (a, b, c):
            assert arg.typestr == 'f4'
        B, C = a.shape
        
        bc = [1, 0]  # load tensor infos
        for i in range(0, C, 32):
            bc.extend([
                2, 16 * i,  # ldg f1 a[i: i + 32]
                8, 0,  # mov f0 f1
                2, 16 * i + 1,  # ldg f1 b[i: i + 32]
                4, 0,  # fadd f0 f1
                3, 16 * i + 2,  # stg f0 c[i: i + 32]
            ])
        bc.extend([0, 0])  # halt
        return ProgramData(
            numpy.array(bc, dtype=numpy.int32),
            [ProgramTensorInfo(4, arg.strides[1], C, [arg.strides[0]], [B], [0]) for arg in (a, b, c)]
        )


class VectorAddProgram(BaseExecutableProgram):
    
    def get_program(self, a: TensorInterface, b: TensorInterface, c: TensorInterface) -> ProgramData:
        assert a.shape == b.shape == c.shape
        for arg in (a, b, c):
            assert arg.typestr == 'f4'
        C, = a.shape
        
        bc = [1, 0]  # load tensor infos
        block = 256
        for i in range(0, block, 32):
            bc.extend([
                2, 16 * i,  # ldg f1 a[i: i + 32]
                8, 0,  # mov f0 f1
                2, 16 * i + 1,  # ldg f1 b[i: i + 32]
                4, 0,  # fadd f0 f1
                3, 16 * i + 2,  # stg f0 c[i: i + 32]
            ])
        bc.extend([0, 0])  # halt
        
        return ProgramData(
            numpy.array(bc, dtype=numpy.int32),
            [ProgramTensorInfo(4, arg.strides[0], C, [arg.strides[0] * block], [cdiv(C, block)], [block]) for arg in (a, b, c)]
        )


class TestInterpretAPB(unittest.TestCase):
    
    def test_batch_a_plus_b(self):
        batch_add_prog = BatchAddProgram()
        for s in [1, 4, 6, 31, 32, 1000, 200000]:
            for p in [1, 16, 18, 32, 64, 256]:
                torch.manual_seed(42)
                a = torch.randn(s, p, device='cuda', dtype=torch.float32)
                b = torch.randn(s, p, device='cuda', dtype=torch.float32)
                c = torch.empty(s, p, device='cuda', dtype=torch.float32)
                batch_add_prog(a, b, c, grid_dim=(s + ILP - 1) // ILP)
                c_ref = a + b
                torch.testing.assert_close(c, c_ref)

    def test_vector_a_plus_b(self):
        vector_add_prog = VectorAddProgram()
        for s in [1, 32, 33, 256, 12345, 100000, 100000000]:
            torch.manual_seed(42)
            a = torch.randn(s, device='cuda', dtype=torch.float32)
            b = torch.randn(s, device='cuda', dtype=torch.float32)
            c = torch.empty(s, device='cuda', dtype=torch.float32)
            vector_add_prog(a, b, c, grid_dim=cdiv(cdiv(s, 256), ILP))
            c_ref = a + b
            torch.testing.assert_close(c, c_ref)
    
    def test_vector_a_plus_b_prof(self):
        vector_add_prog = VectorAddProgram()
        for s in [1, 32, 33, 256, 12345, 100000, 100000000]:
            torch.manual_seed(42)
            a = torch.randn(s, dtype=torch.float32).cuda()
            b = torch.randn(s, dtype=torch.float32).cuda()
            c = torch.empty(s, device='cuda', dtype=torch.float32)
            vector_add_prog(a, b, c, grid_dim=cdiv(cdiv(s, 256), ILP))
            _ = a + b
