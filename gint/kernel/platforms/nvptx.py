import uuid
from llvmlite import ir
from .common import *
from .platform import PlatformIRBuilder


class NVPTXIRBuilder(PlatformIRBuilder):
    
    def read_sreg(self, name: str) -> ir.Value:
        return self.intrinsic(f'llvm.nvvm.read.ptx.sreg.{name}', i32, [])
    
    def thread_idx_x(self) -> ir.Value:
        return self.read_sreg("tid.x")
    
    def block_idx_x(self) -> ir.Value:
        return self.read_sreg("ctaid.x")

    def warp_size(self) -> ir.Value:
        return i32(32)
    
    def lane_id(self) -> ir.Value:
        return self.read_sreg("laneid")
    
    def printf(self, fmt: str, *args: tuple[ir.Value, ...]) -> ir.Value:
        fmt_lit = self.string_literal(fmt)
        
        wrapper_func_ty = ir.FunctionType(i32, [], True)
        wrapper_func = ir.Function(self.module, wrapper_func_ty, '_printf_wrapper_' + uuid.uuid4().hex)
        
        ret = self.call(wrapper_func, args)
        block_ptr = self.block
        
        self.position_at_end(wrapper_func.append_basic_block("entry"))
        va_list_ptr = self.alloca(p_i8, name="va_list_storage")
        self.intrinsic("llvm.va_start", void, [va_list_ptr])
        vprintf_type = ir.FunctionType(i32, [p_i8, p_i8])
        vprintf = ir.Function(self.module, vprintf_type, 'vprintf')
        va_list = self.load(va_list_ptr)
        call_result = self.call(vprintf, [fmt_lit, va_list])
        self.intrinsic("llvm.va_end", void, [va_list_ptr])
        self.ret(call_result)
        
        self.position_at_end(block_ptr)
        return ret
