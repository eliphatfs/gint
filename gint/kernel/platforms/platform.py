from llvmlite import ir


class PlatformIRBuilder(ir.IRBuilder):
    
    def thread_idx_x(self) -> ir.Value:
        raise NotImplementedError
    
    def block_idx_x(self) -> ir.Value:
        raise NotImplementedError

    def warp_size(self) -> ir.Value:
        raise NotImplementedError

    def lane_id(self) -> ir.Value:
        raise NotImplementedError
    
    def intrinsic(self, name: str, res_ty: ir.Type, args: list[ir.Value]):
        return self.call(self.intrinsic_fn(name, res_ty, [a.type for a in args]), args)

    def intrinsic_fn(self, name: str, res_ty: ir.Type, args_tys: list[ir.Type]):
        return ir.Function(self.module, ir.FunctionType(res_ty, args_tys), name)
