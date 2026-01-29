import numpy
import torch
import unittest
from gint.kernel.interpreter.main import REG_WIDTH
from gint.host.executor import BaseExecutableProgram, ProgramData, ProgramTensorInfo, TensorInterface
from gint.host.utils import cdiv


class IntegerArithProgram(BaseExecutableProgram):
    def __init__(self, op_code: int) -> None:
        super().__init__()
        self.op_code = op_code
    
    def get_program(self, a: TensorInterface, b: TensorInterface, c: TensorInterface) -> ProgramData:
        B, C = a.shape
        bc = [
            15, 1,  # ldg b
            15, 0,  # ldg a
            self.op_code, 0,  # iarith
            16, 2,  # stg c
            0, 0,  # halt
        ]
        return ProgramData(
            numpy.array(bc, dtype=numpy.int32),
            [ProgramTensorInfo(4, arg.strides[1], C, [arg.strides[0]], [B], [0]) for arg in (a, b, c)]
        )


class IndirectMemoryProgram(BaseExecutableProgram):
    def get_program(self, data: TensorInterface, indices: TensorInterface, out: TensorInterface) -> ProgramData:
        # data: tensor to access indirectly (tensor 0)
        # indices: indices to use (tensor 1)
        # out: where to store results (tensor 2)
        B, C = out.shape
        bc = [
            15, 1,  # ldg indices (onto stack)
            66, 0,  # LoadGlobalF32Indirect from tensor 0
            16, 2,  # stg out
            0, 0,  # halt
        ]
        return ProgramData(
            numpy.array(bc, dtype=numpy.int32),
            [
                ProgramTensorInfo(4, data.strides[1], data.shape[1], [data.strides[0]], [data.shape[0]], [0]),
                ProgramTensorInfo(4, indices.strides[1], C, [indices.strides[0]], [B], [0]),
                ProgramTensorInfo(4, out.strides[1], C, [out.strides[0]], [B], [0]),
            ]
        )


class PackedImmProgram(BaseExecutableProgram):
    def __init__(self, op_code: int, packed_val: int) -> None:
        super().__init__()
        self.op_code = op_code
        self.packed_val = packed_val
    
    def get_program(self, out: TensorInterface) -> ProgramData:
        # We'll use indirect store to verify all 4 slots.
        # Tensor 0: out
        # Indices: [0, 1, 2, 3] packed in i32
        indices_packed = numpy.array([0, 1, 2, 3], dtype=numpy.int8).view(numpy.int32)[0]
        
        B, C = out.shape
        bc = [
            self.op_code, self.packed_val,     # Push values [v0, v1, v2, v3]
            69, int(indices_packed),           # Push indices [0, 1, 2, 3] (LoadImm4I)
            67, 0,                             # StoreGlobalF32Indirect to tensor 0
            0, 0,                              # halt
        ]
        return ProgramData(
            numpy.array(bc, dtype=numpy.int32),
            [ProgramTensorInfo(4, out.strides[1], C, [out.strides[0]], [B], [0])]
        )


class TestNewInstructions(unittest.TestCase):
    def test_integer_arith(self):
        # IAdd: 56
        prog = IntegerArithProgram(56)
        s = 1024
        a_int = torch.randint(0, 100, (s, 32), dtype=torch.int32, device='cuda')
        b_int = torch.randint(0, 100, (s, 32), dtype=torch.int32, device='cuda')
        c_int = torch.empty((s, 32), dtype=torch.int32, device='cuda')
        
        # Bitcast to float for gint interface
        a = a_int.view(torch.float32)
        b = b_int.view(torch.float32)
        c = c_int.view(torch.float32)
        
        prog(a, b, c, grid_dim=cdiv(s, REG_WIDTH))
        
        torch.testing.assert_close(c_int, a_int + b_int)

    def test_bitwise_ops(self):
        # IAnd: 63
        prog = IntegerArithProgram(63)
        s = 512
        a_int = torch.randint(0, 0xFF, (s, 32), dtype=torch.int32, device='cuda')
        b_int = torch.randint(0, 0xFF, (s, 32), dtype=torch.int32, device='cuda')
        c_int = torch.empty((s, 32), dtype=torch.int32, device='cuda')
        
        a = a_int.view(torch.float32)
        b = b_int.view(torch.float32)
        c = c_int.view(torch.float32)
        
        prog(a, b, c, grid_dim=cdiv(s, REG_WIDTH))
        torch.testing.assert_close(c_int, a_int & b_int)

    def test_indirect_load(self):
        data = torch.rand(100, device='cuda', dtype=torch.float32)
        indices_int = torch.randint(0, 100, (10, 32), dtype=torch.int32, device='cuda')
        out = torch.empty((10, 32), device='cuda', dtype=torch.float32)
        
        prog = IndirectMemoryProgram()
        data_2d = data.view(1, 100) 
        
        prog(data_2d, indices_int.view(torch.float32), out, grid_dim=cdiv(10, REG_WIDTH))
        
        ref = data[indices_int.long()]
        torch.testing.assert_close(out, ref)

    def test_load_imm4f(self):
        # LoadImm4F: 68
        packed = numpy.array([1, -2, 3, -4], dtype=numpy.int8).view(numpy.int32)[0]
        prog = PackedImmProgram(68, int(packed))
        
        out = torch.empty((1, 4), device='cuda', dtype=torch.float32)
        prog(out, grid_dim=1)
        
        ref = torch.tensor([[1.0, -2.0, 3.0, -4.0]], device='cuda', dtype=torch.float32)
        torch.testing.assert_close(out, ref)

    def test_load_imm4i(self):
        # LoadImm4I: 69
        packed = numpy.array([10, 20, 30, 40], dtype=numpy.int8).view(numpy.int32)[0]
        prog = PackedImmProgram(69, int(packed))
        
        out_int = torch.empty((1, 4), device='cuda', dtype=torch.int32)
        out = out_int.view(torch.float32)
        prog(out, grid_dim=1)
        
        ref = torch.tensor([[10, 20, 30, 40]], device='cuda', dtype=torch.int32)
        torch.testing.assert_close(out_int, ref)

if __name__ == '__main__':
    unittest.main()
