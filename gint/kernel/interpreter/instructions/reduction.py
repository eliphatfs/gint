from ..state import InterpreterState, InterpreterStateSpec
from ...platforms.platform import PlatformIRBuilder
from ..instruction import Instruction
from ...platforms.common import *


class _WarpAllReduceBase(Instruction):
    op: EReducePrimitiveOp
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        state[ispec.rf0] = list([
            LL.warp_allreduce_f32(x, self.op)
            for x in state[ispec.rf0]
        ])


class WarpAllReduceSum(_WarpAllReduceBase):
    op = EReducePrimitiveOp.Sum


class WarpAllReduceMax(_WarpAllReduceBase):
    op = EReducePrimitiveOp.Max


class WarpAllReduceMin(_WarpAllReduceBase):
    op = EReducePrimitiveOp.Min


class WarpAllReduceProd(_WarpAllReduceBase):
    op = EReducePrimitiveOp.Prod
