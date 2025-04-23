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
    
    def intrinsic(self, name: str, res_ty: ir.Type, args: list[ir.Value]):
        return self.call(self.intrinsic_fn(name, res_ty, [a.type for a in args]), args)

    def intrinsic_fn(self, name: str, res_ty: ir.Type, args_tys: list[ir.Type]):
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
