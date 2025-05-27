import numpy
import torch
import unittest
from typing import Callable
from gint.kernel.interpreter.main import ILP
from gint.host.executor import BaseExecutableProgram, ProgramData, ProgramTensorInfo, TensorInterface
from gint.host.utils import cdiv


class MinimalBinaryArithProgram(BaseExecutableProgram):
    
    def __init__(self, op: str) -> None:
        super().__init__()
        self.op = op
        self.op_map = {
            'add': 4,
            'mul': 17,
            'sub': 19,
            'rsub': 20,
            'div': 21,
            'rdiv': 22,
        }
    
    def get_program(self, a: TensorInterface, b: TensorInterface, c: TensorInterface) -> ProgramData:
        assert a.shape == b.shape == c.shape
        for arg in (a, b, c):
            assert arg.typestr == 'f4'
        B, C = a.shape
        
        bc = [
            2, 0,  # ldg f1 a
            8, 0,  # mov f0 f1
            2, 1,  # ldg f1 b
            self.op_map[self.op], 0,  # the arith f0 = f0 () f1
            3, 2,  # stg f0 c
            0, 0,  # halt
        ]
        return ProgramData(
            numpy.array(bc, dtype=numpy.int32),
            [ProgramTensorInfo(4, arg.strides[1], C, [arg.strides[0]], [B], [0]) for arg in (a, b, c)]
        )


class MinimalFMAProgram(BaseExecutableProgram):
    
    def get_program(self, a: TensorInterface, b: TensorInterface, c: TensorInterface, d: TensorInterface) -> ProgramData:
        assert a.shape == b.shape == c.shape == d.shape
        for arg in (a, b, c, d):
            assert arg.typestr == 'f4'
        B, C = a.shape
        
        bc = [
            2, 0,  # ldg f1 a
            8, 0,  # mov f0 f1
            2, 2,  # ldg f1 c
            9, 0,  # mov f2 f1
            2, 1,  # ldg f1 b
            18, 0,  # fma f0 f1 f2
            3, 3,  # stg f0 d
            0, 0,  # halt
        ]
        return ProgramData(
            numpy.array(bc, dtype=numpy.int32),
            [ProgramTensorInfo(4, arg.strides[1], C, [arg.strides[0]], [B], [0]) for arg in (a, b, c, d)]
        )


class MinimalInplaceNegProgram(BaseExecutableProgram):
    
    def get_program(self, arg: TensorInterface) -> ProgramData:
        assert arg.typestr == 'f4'
        B, C = arg.shape
        
        bc = [
            2, 0,  # ldg f1 a
            8, 0,  # mov f0 f1
            23, 0,  # neg f0
            3, 0,  # stg f0 a
            0, 0,  # halt
        ]
        return ProgramData(
            numpy.array(bc, dtype=numpy.int32),
            [ProgramTensorInfo(4, arg.strides[1], C, [arg.strides[0]], [B], [0])]
        )


class TestBasicNonBinaryArithmetics(unittest.TestCase):
    
    def test_inplace_neg(self):
        prog = MinimalInplaceNegProgram()
        for s in [1, 4, 6, 31, 32, 1000, 200000]:
            torch.manual_seed(42)
            a = torch.rand(s, 32, device='cuda', dtype=torch.float32) + 1e-6
            a_ref = -a
            prog(a, grid_dim=cdiv(s, ILP))
            torch.testing.assert_close(a, a_ref)
    
    def test_fma(self):
        prog = MinimalFMAProgram()
        for s in [1, 4, 6, 31, 32, 1000, 200000]:
            torch.manual_seed(42)
            a = torch.rand(s, 32, device='cuda', dtype=torch.float32) + 1e-6
            b = torch.rand(s, 32, device='cuda', dtype=torch.float32) + 1e-6
            c = torch.rand(s, 32, device='cuda', dtype=torch.float32) + 1e-6
            d = torch.empty(s, 32, device='cuda', dtype=torch.float32)
            prog(a, b, c, d, grid_dim=cdiv(s, ILP))
            d_ref = a + b * c
            torch.testing.assert_close(d, d_ref)


class TestBasicBinaryArithmetics(unittest.TestCase):
    
    def _case(self, op: str, ref_lambda: Callable):
        prog = MinimalBinaryArithProgram(op)
        for s in [1, 4, 6, 31, 32, 1000, 200000]:
            torch.manual_seed(42)
            a = torch.rand(s, 32, device='cuda', dtype=torch.float32) + 1e-6
            b = torch.rand(s, 32, device='cuda', dtype=torch.float32) + 1e-6
            c = torch.empty(s, 32, device='cuda', dtype=torch.float32)
            prog(a, b, c, grid_dim=cdiv(s, ILP))
            c_ref = ref_lambda(a, b)
            torch.testing.assert_close(c, c_ref)
    
    def test_add(self):
        self._case('add', (lambda a, b: a + b))
    
    def test_mul(self):
        self._case('mul', (lambda a, b: a * b))
    
    def test_sub(self):
        self._case('sub', (lambda a, b: a - b))
    
    def test_rsub(self):
        self._case('rsub', (lambda a, b: b - a))
    
    def test_div(self):
        self._case('div', (lambda a, b: a / b))
    
    def test_rdiv(self):
        self._case('rdiv', (lambda a, b: b / a))
