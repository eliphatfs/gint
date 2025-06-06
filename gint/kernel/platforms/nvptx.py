import uuid
from llvmlite import ir

from .common import *
from .platform import PlatformIRBuilder


class NVPTXIRBuilder(PlatformIRBuilder):
    
    @classmethod
    def create_kernel_module(cls, fn_type: ir.FunctionType, fn_name: str):
        mod = ir.Module('gint_device_module')
        mod.triple = "nvptx64-nvidia-cuda"
        mod.data_layout = "e-p:64:64:64-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-f32:32:32-f64:64:64-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64"
        geval = ir.Function(mod, fn_type, fn_name)
        geval.calling_convention = "ptx_kernel"
        entry_bb = geval.append_basic_block("entry")
        return NVPTXIRBuilder(entry_bb)
    
    def smem_addrspace(self) -> int:
        return 3

    def warp_sync(self) -> None:
        self.intrinsic('llvm.nvvm.bar.warp.sync', void, [i32(-1)])
    
    def read_sreg(self, name: str) -> ir.Value:
        return self.intrinsic(f'llvm.nvvm.read.ptx.sreg.{name}', i32, [])
    
    def thread_idx_x(self) -> ir.Value:
        return self.read_sreg("tid.x")
    
    def thread_idx_y(self) -> ir.Value:
        return self.read_sreg("tid.y")
    
    def logical_program_idx(self) -> ir.Value:
        return self.add(self.mul(self.read_sreg("ctaid.x"), self.read_sreg('ntid.y')), self.read_sreg('tid.y'))

    def warp_size(self) -> ir.Value:
        return i32(32)
    
    def lane_id(self) -> ir.Value:
        return self.read_sreg("laneid")
    
    def printf(self, fmt: str, *args: ir.Value) -> ir.Value:
        fmt_lit = self.string_literal(fmt)
        
        wrapper_func_ty = ir.FunctionType(i32, [], True)
        wrapper_func = ir.Function(self.module, wrapper_func_ty, '_printf_wrapper_' + uuid.uuid4().hex)
        
        ret = self.call(wrapper_func, args)
        block_ptr = self.block
        
        self.position_at_end(wrapper_func.append_basic_block("entry"))
        va_list_ptr = self.alloca(p_i8, name="va_list_storage")
        self.intrinsic("llvm.va_start", void, [va_list_ptr])
        
        vprintf = self.intrinsic_fn('vprintf', i32, [p_i8, p_i8])
        va_list = self.load(va_list_ptr)
        call_result = self.call(vprintf, [fmt_lit, va_list])
        self.intrinsic("llvm.va_end", void, [va_list_ptr])
        self.ret(call_result)
        
        self.position_at_end(block_ptr)
        return ret

    def warp_broadcast_lane(self, value: ir.NamedValue, lane: ir.Value) -> ir.Value:
        if value.type == i32:
            return self.intrinsic(
                "llvm.nvvm.shfl.sync.idx.i32",
                i32, [i32(-1), value, lane, i32(31)]
            )
        if value.type == f32:
            return self.intrinsic(
                "llvm.nvvm.shfl.sync.idx.f32",
                f32, [i32(-1), value, lane, i32(31)]
            )
        raise TypeError("Expected value type f32 or i32, got", value.type)

    def warp_allreduce_f32(self, value: ir.Value, op: EReducePrimitiveOp) -> ir.Value:
        red = value
        for mask in [16, 8, 4, 2, 1]:
            comm = self.intrinsic(
                "llvm.nvvm.shfl.sync.bfly.f32",
                f32, [i32(-1), red, i32(mask), i32(31)]
            )
            if op == EReducePrimitiveOp.Sum:
                red = self.fadd(red, comm)
            elif op == EReducePrimitiveOp.Max:
                red = self.intrinsic('llvm.maximum.f32', f32, [red, comm])
            elif op == EReducePrimitiveOp.Min:
                red = self.intrinsic('llvm.minimum.f32', f32, [red, comm])
            elif op == EReducePrimitiveOp.Prod:
                red = self.fmul(red, comm)
        return red
    
    def special_unary(self, value: ir.Value, op: EUnarySpecialOp) -> ir.Value:
        intrinsic_map = {
            EUnarySpecialOp.Sqrt: '__nv_sqrtf',
            EUnarySpecialOp.Sin: '__nv_fast_sinf',
            EUnarySpecialOp.Cos: '__nv_fast_cosf',
            EUnarySpecialOp.Tan: '__nv_fast_tanf',
            EUnarySpecialOp.ArcSin: '__nv_asinf',
            EUnarySpecialOp.ArcCos: '__nv_acosf',
            EUnarySpecialOp.ArcTan: '__nv_atanf',
            EUnarySpecialOp.Exp: '__nv_expf',
            EUnarySpecialOp.Exp2: '__nv_exp2f',
            EUnarySpecialOp.Log: '__nv_logf',
            EUnarySpecialOp.Log2: '__nv_log2f',
            EUnarySpecialOp.RSqrt: '__nv_rsqrtf',
            EUnarySpecialOp.Erf: '__nv_erff',
        }
        return self.intrinsic(intrinsic_map[op], f32, [value])
    
    def special_binary(self, a: ir.Value, b: ir.Value, op: EBinarySpecialOp) -> ir.Value:
        intrinsic_map = {
            EBinarySpecialOp.ArcTan2: '__nv_atan2f',
            EBinarySpecialOp.Pow: '__nv_powf',
        }
        return self.intrinsic(intrinsic_map[op], f32, [a, b])
