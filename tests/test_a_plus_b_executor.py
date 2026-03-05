import numpy
import torch
import unittest
from gint.kernel.interpreter.main import REG_WIDTH
from gint.host.executor import BaseExecutableProgram, ProgramData, ProgramTensorInfo, TensorInterface
from gint.host.utils import cdiv


class BatchAddProgram(BaseExecutableProgram):
    
    def get_program(self, a: TensorInterface, b: TensorInterface, c: TensorInterface) -> ProgramData:
        assert a.shape == b.shape == c.shape
        for arg in (a, b, c):
            assert arg.typestr == 'f4'
        B, C = a.shape
        
        bc = []
        for i in range(0, C, 32):
            bc.extend([
                70, 16 * i,  # ldg2dt f1 a[i: i + 32]
                70, 16 * i + 1,  # ldg2dt f1 b[i: i + 32]
                2, 0,  # fadd f0 f1
                71, 16 * i + 2,  # stg2dt f0 c[i: i + 32]
            ])
        bc.extend([0, 0])  # halt
        return ProgramData(
            numpy.array(bc, dtype=numpy.int32),
            [ProgramTensorInfo(
                4, [], [], [C, arg.strides[1]], [B, arg.strides[0]],
                [1, cdiv(B, self.REGW)], [0, self.REGW]
            ) for arg in (a, b, c)]
        )


class VectorAddProgram(BaseExecutableProgram):
    
    def get_program(self, a: TensorInterface, b: TensorInterface, c: TensorInterface) -> ProgramData:
        assert a.shape == b.shape == c.shape
        for arg in (a, b, c):
            assert arg.typestr == 'f4'
        C, = a.shape
        
        bc = []
        block = 256
        for i in range(0, block, 32 * self.REGW):
            bc.extend([
                15, 16 * i,  # ldg1d f1 a[i: i + 128]
                15, 16 * i + 1,  # ldg1d f1 b[i: i + 128]
                2, 0,  # fadd f0 f1
                16, 16 * i + 2,  # stg1d f0 c[i: i + 128]
            ])
        bc.extend([0, 0])  # halt
        
        return ProgramData(
            numpy.array(bc, dtype=numpy.int32),
            [ProgramTensorInfo(
                4, [], [], [C, arg.strides[0]], [1, 0],
                [cdiv(C, block), 1], [block, 0]
            ) for arg in (a, b, c)]
        )


class TestInterpretAPB(unittest.TestCase):
    
    def test_batch_a_plus_b(self):
        batch_add_prog = BatchAddProgram()
        for s in [1, 4, 6, 31, 32, 1000, 9999, 200000]:
            for p in [1, 16, 18, 32, 64, 256]:
                torch.manual_seed(42)
                a = torch.randn(s, p, device='cuda', dtype=torch.float32)
                b = torch.randn(s, p, device='cuda', dtype=torch.float32)
                c = torch.empty(s, p, device='cuda', dtype=torch.float32)
                batch_add_prog(a, b, c, grid_dim=(s + REG_WIDTH - 1) // REG_WIDTH)
                c_ref = a + b
                torch.testing.assert_close(c, c_ref)

    def test_vector_a_plus_b(self):
        vector_add_prog = VectorAddProgram()
        for s in [1, 32, 33, 256, 12345, 100000, 100000000]:
            torch.manual_seed(42)
            a = torch.randn(s, device='cuda', dtype=torch.float32)
            b = torch.randn(s, device='cuda', dtype=torch.float32)
            c = torch.zeros(s, device='cuda', dtype=torch.float32)
            vector_add_prog(a, b, c, grid_dim=cdiv(s, 256))
            c_ref = a + b
            torch.testing.assert_close(c, c_ref)
    
    def test_vector_a_plus_b_prof(self):
        vector_add_prog = VectorAddProgram()
        for s in [1, 32, 33, 256, 12345, 100000, 100000000]:
            torch.manual_seed(42)
            a = torch.randn(s, dtype=torch.float32).cuda()
            b = torch.randn(s, dtype=torch.float32).cuda()
            c = torch.empty(s, device='cuda', dtype=torch.float32)
            vector_add_prog(a, b, c, grid_dim=cdiv(s, 256))
            _ = a + b
