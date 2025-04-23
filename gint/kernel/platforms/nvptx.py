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
        return ir.Constant(i32, 32)
    
    def lane_id(self) -> ir.Value:
        return self.read_sreg("laneid")
