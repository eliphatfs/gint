import torch
import unittest
from gint import ProgramTensorInfo, TensorInterface, bytecode, cdiv
from gint.host.frontend import *


@bytecode
def basic_expr1(a: TensorInterface, b: TensorInterface, c: TensorInterface, REGW: int, WARP: int):
    assert a.shape == b.shape == c.shape
    for arg in (a, b, c):
        assert arg.typestr == 'f4'
    B, C = a.shape
    for i in range(0, C, WARP):
        fldg(i, b)
        dup()  # b, b
        dup2()  # b, b, b, b
        fldg(i, a)  # b, b, b, b, a
        fadd()  # b, b, b, a + b
        fadd()  # b, b, a + b + b
        fdiv()  # b, (a + b + b) / b
        fsub()  # (a + b + b) / b - b
        fneg()  # -((a + b + b) / b - b)
        fstg(i, c)
    halt()
    return [ProgramTensorInfo(arg.elm_size, arg.strides[1], C, [arg.strides[0]], [B], [0]) for arg in (a, b, c)]


@bytecode
def vector_expr2(x: TensorInterface, y: TensorInterface, REGW: int, WARP: int, BLOCK: int):
    # implements cubic easing function x ** 2 * (3 - 2 * x)
    assert x.shape == y.shape
    for arg in (x, y):
        assert arg.typestr == 'f4'
    C, = x.shape
    block = BLOCK
    for i in range(0, block, WARP):
        fldg(i, x)  # x
        dup()  # x, x
        dup()  # x, x, x
        fmaimm(-2.0, 3.0)  # x, x, 3 - 2x
        fmul()  # x, x * (3 - 2x)
        fmul()  # x ** 2 * (3 - 2 * x)
        fstg(i, y)
    halt()
    return [ProgramTensorInfo(4, arg.strides[0], C, [arg.strides[0] * block], [cdiv(C, block)], [block]) for arg in (x, y)]


@bytecode
def meaningless_execute_everything(x: TensorInterface, x_f16: TensorInterface, x_bf16: TensorInterface, x_u8: TensorInterface, REGW: int, WARP: int):
    nop()  # 0
    fldg(0, x)  # 1
    dup()  # 2
    dup2()  # 4
    fstg(0, x)  # 3
    fadd()  # 2
    dup2()  # 4
    fmul()  # 3
    fsub()  # 2
    frsub()  # 1
    dup()  # 2
    dup2()  # 4
    fdiv()  # 3
    frdiv()  # 2
    fneg()  # 2
    pop2()  # 0
    fpush(1.0)  # 1
    fpush(1.0)  # 2
    fpush(1.0)  # 3
    warp_allreduce_fmax()
    warp_allreduce_fmin()
    warp_allreduce_fsum()
    warp_allreduce_fprod()  # 3
    frem()  # 2
    pop()  # 1
    fsqrt()  # 1
    fpush(1.0)
    fsin()
    fcos()
    ftan()
    fpush(1.0)  # 2
    fasin()
    fpush(1.0)  # 3
    facos()
    fpush(1.0)  # 4
    fatan()
    pop2()  # 2
    pop2()  # 0
    fpush(1.0)  # 1
    fexp()
    fexp2()
    fpush(1.0)  # 2
    flog()
    flog2()
    pop2()
    fpush(1.0)  # 1
    ferf()
    fpush(1.0)  # 2
    frsqrt()
    pop2()  # 0
    fpush(1.0)
    fpush(2.0)
    fatan2()
    pop()
    fpush(1.0)  # 1
    dup()  # 2
    dupx1()  # 3
    dupx2()  # 4
    fpow()  # 3
    fge()  # 2
    fgt()  # 1
    dup()
    dup2()  # 4
    dup2()  # 6
    dup2()  # 8
    fle()  # 7
    flt()  # 6
    fne()  # 5
    feq()  # 4
    fapprox(0.1)  # 3
    fselect()  # 1
    fldg_f16(0, x_f16)  # 2
    fstg_f16(0, x_f16)  # 1
    fldg_bf16(0, x_bf16)  # 2
    fstg_bf16(0, x_bf16)  # 1
    fldg_u8(0, x_u8)  # 2
    faddimm(1.0)
    fmulimm(1.0)
    fmaimm(1.0, 1.0)
    pop2()
    halt()
    return [
        ProgramTensorInfo(4, x.strides[0], 32, [x.strides[0]], [0], [1]),
        ProgramTensorInfo(2, x_f16.strides[0], 32, [x_f16.strides[0]], [0], [1]),
        ProgramTensorInfo(2, x_bf16.strides[0], 32, [x_bf16.strides[0]], [0], [1]),
        ProgramTensorInfo(1, x_u8.strides[0], 32, [x_u8.strides[0]], [0], [1]),
    ]


@bytecode
def indirect_arith_test(a, b, c, data, indices, out_load, out_store, REGW: int, WARP: int):
    # Integer arith: a + b -> c
    fldg(0, a)
    fldg(0, b)
    iadd()
    fstg(0, c)
    
    # Indirect load: data[indices] -> out_load
    fldg(0, indices)
    fldg_ind(data)
    fstg(0, out_load)
    
    # Indirect store: out_load -> indices -> out_store
    fldg(0, out_load)
    fldg(0, indices)
    fstg_ind(out_store)
    
    halt()
    return [ProgramTensorInfo(4, arg.strides[-1] if arg.ndim > 0 else 0, arg.shape[-1] if arg.ndim > 0 else 1, list(arg.strides[:-1]), list(arg.shape[:-1]), [0]*max(0, arg.ndim-1)) for arg in (a, b, c, data, indices, out_load, out_store)]


@bytecode
def packed_imm_test(packed_f_out, packed_i_out, REGW: int, WARP: int):
    # Packed Imm F -> packed_f_out
    fpush4(0x04030201) # [1.0, 2.0, 3.0, 4.0]
    ipush4(0x03020100) # [0, 1, 2, 3] indices
    fstg_ind(packed_f_out)
    
    # Packed Imm I -> packed_i_out
    ipush4(0x40302010) # [16, 32, 48, 64]
    ipush4(0x03020100) # [0, 1, 2, 3] indices
    fstg_ind(packed_i_out)
    
    halt()
    return [ProgramTensorInfo(4, arg.strides[-1] if arg.ndim > 0 else 0, arg.shape[-1] if arg.ndim > 0 else 1, list(arg.strides[:-1]), list(arg.shape[:-1]), [0]*max(0, arg.ndim-1)) for arg in (packed_f_out, packed_i_out)]


class TestFrontendExpression(unittest.TestCase):
    
    def test_expr_1(self):
        # ... (lines 173-182)
        for s in [1, 4, 6, 31, 32, 1000, 200000]:
            for p in [1, 16, 18, 32, 64, 256]:
                torch.manual_seed(42)
                a = torch.rand(s, p, device='cuda', dtype=torch.float32) + 1e-6
                b = torch.rand(s, p, device='cuda', dtype=torch.float32) + 1e-6
                c = torch.empty(s, p, device='cuda', dtype=torch.float32)
                basic_expr1(a, b, c, grid_dim=cdiv(s, REG_WIDTH))
                c_ref = -((a + b + b) / b - b)
                torch.testing.assert_close(c, c_ref)


    def test_expr_2(self):
        # ... (lines 185-191)
        for z in [1, 5, 16, 17, 31, 32, 33, 1023, 1024, 10000000]:
            torch.manual_seed(42)
            x = torch.rand(z, device='cuda', dtype=torch.float32)
            y = torch.empty(z, device='cuda', dtype=torch.float32)
            vector_expr2(x, y, grid_dim=cdiv(cdiv(z, 256), REG_WIDTH), BLOCK=256)
            y_ref = x ** 2 * (3 - 2 * x)
            torch.testing.assert_close(y, y_ref)

    def test_meaningless_everything(self):
        torch.manual_seed(42)
        x = torch.rand(32, device='cuda', dtype=torch.float32)
        x_f16 = torch.rand(32, device='cuda', dtype=torch.float16)
        x_bf16 = torch.rand(32, device='cuda', dtype=torch.bfloat16)
        x_u8 = (torch.rand(32, device='cuda', dtype=torch.bfloat16) * 255).byte()
        meaningless_execute_everything(x, x_f16, x_bf16, x_u8, grid_dim=1)

    def test_indirect_and_integer(self):
        a_int = torch.randint(0, 100, (1, 32), dtype=torch.int32, device='cuda')
        b_int = torch.randint(0, 100, (1, 32), dtype=torch.int32, device='cuda')
        c_int = torch.empty((1, 32), dtype=torch.int32, device='cuda')
        data = torch.rand(100, device='cuda', dtype=torch.float32)
        indices = torch.randint(0, 100, (1, 32), dtype=torch.int32, device='cuda')
        out_load = torch.empty((1, 32), device='cuda', dtype=torch.float32)
        out_store = torch.zeros(100, device='cuda', dtype=torch.float32)
        
        indirect_arith_test(
            a_int.view(torch.float32), 
            b_int.view(torch.float32), 
            c_int.view(torch.float32), 
            data.view(1, 100), 
            indices.view(torch.float32), 
            out_load, 
            out_store.view(1, 100),
            grid_dim=1
        )
        
        torch.testing.assert_close(c_int, a_int + b_int)
        torch.testing.assert_close(out_load, data[indices.long()])
        
        # Verify indirect store
        ref_store = torch.zeros(100, device='cuda', dtype=torch.float32)
        ref_store[indices.long()] = out_load
        torch.testing.assert_close(out_store, ref_store)
        
    def test_packed_imm(self):
        packed_f_out = torch.zeros(100, device='cuda', dtype=torch.float32)
        packed_i_out = torch.zeros(100, device='cuda', dtype=torch.int32)
        
        packed_imm_test(
            packed_f_out.view(1, 100),
            packed_i_out.view(torch.float32).view(1, 100),
            grid_dim=1
        )
        
        # Verify packed imm f
        ref_packed_f = torch.zeros(100, device='cuda', dtype=torch.float32)
        ref_packed_f[0:4] = torch.tensor([1.0, 2.0, 3.0, 4.0], device='cuda')
        torch.testing.assert_close(packed_f_out, ref_packed_f)
        
        # Verify packed imm i
        ref_packed_i = torch.zeros(100, device='cuda', dtype=torch.int32)
        ref_packed_i[0:4] = torch.tensor([16, 32, 48, 64], device='cuda', dtype=torch.int32)
        torch.testing.assert_close(packed_i_out, ref_packed_i)
