import uuid
from typing import *
from llvmlite import ir
from .common import *


class PlatformIRBuilder(ir.IRBuilder):
    
    def thread_idx_x(self) -> ir.Value:
        raise NotImplementedError
    
    def block_idx_x(self) -> ir.Value:
        raise NotImplementedError

    def warp_size(self) -> ir.Value:
        raise NotImplementedError

    def lane_id(self) -> ir.Value:
        raise NotImplementedError
    
    def printf(self, fmt: str, *args: tuple[ir.Value, ...]) -> ir.Value:
        raise NotImplementedError
    
    def warp_broadcast_lane(self, value: ir.Value, lane: ir.Value) -> ir.Value:
        raise NotImplementedError
    
    def arg(self, idx: int) -> ir.Value:
        fn: ir.Function = self.function
        return fn.args[idx]
    
    def intrinsic(self, name: str, res_ty: ir.Type, args: list[ir.Value]):
        return self.call(self.intrinsic_fn(name, res_ty, [a.type for a in args]), args)

    def intrinsic_fn(self, name: str, res_ty: ir.Type, args_tys: list[ir.Type]):
        mod: ir.Module = self.module
        if name in mod.globals:
            return mod.globals[name]
        return ir.Function(self.module, ir.FunctionType(res_ty, args_tys), name)

    def string_literal(self, string: str, name: Optional[str] = None) -> ir.Constant:
        """
        Creates a string constant in the LLVM module.

        Args:
            string: The Python string to convert to a constant.
            name: The name for the global variable. By default, a underscore with a UUID will be generated.

        Returns:
            An ir.Constant representing the string constant.
        """
        string_with_null = string + '\0'
        string_bytes = string_with_null.encode('utf-8')

        string_type = ir.ArrayType(i8, len(string_bytes))
        
        if name is None:
            name = '_' + uuid.uuid4().hex

        global_var = ir.GlobalVariable(self.module, string_type, name=name)
        global_var.linkage = 'private'
        global_var.global_constant = True
        global_var.initializer = ir.Constant(string_type, bytearray(string_bytes))
        global_var.unnamed_addr = True

        string_pointer = global_var.gep([i32(0), i32(0)])

        return string_pointer

    def emit(self):
        assert self.module is not None, "module is None. please point to a block before calling emit."
        return str(self.module).encode()

    def warp_broadcast_lane_b64_as_2xi32(self, value: ir.NamedValue, lane: ir.Value) -> ir.Value:
        ptr_ty = None
        if value.type.is_pointer:
            ptr_ty = value.type
            value = self.ptrtoint(value, i64)
        i32x2 = self.bitcast(value, ir.VectorType(i32, 2))
        e0 = self.extract_element(i32x2, i32(0))
        e1 = self.extract_element(i32x2, i32(1))
        be0 = self.warp_broadcast_lane(e0, lane)
        be1 = self.warp_broadcast_lane(e1, lane)
        bi32x2 = self.insert_element(i32x2, be0, i32(0))
        bi32x2 = self.insert_element(bi32x2, be1, i32(1))
        ret = self.bitcast(bi32x2, value.type)
        if ptr_ty is not None:
            ret = self.inttoptr(ret, ptr_ty)
        return ret
