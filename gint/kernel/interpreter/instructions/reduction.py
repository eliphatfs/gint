from ..state import StackMachineState
from ...platforms.platform import PlatformIRBuilder
from ..instruction import DefaultControlInstruction
from ...platforms.common import *


class _WarpAllReduceBase(DefaultControlInstruction):
    op: EReducePrimitiveOp
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        reg = state.peek()
        state.pop().push([
            LL.warp_allreduce_f32(x, self.op) for x in reg
        ])


class WarpAllReduceSum(_WarpAllReduceBase):
    op = EReducePrimitiveOp.Sum


class WarpAllReduceMax(_WarpAllReduceBase):
    op = EReducePrimitiveOp.Max


class WarpAllReduceMin(_WarpAllReduceBase):
    op = EReducePrimitiveOp.Min


class WarpAllReduceProd(_WarpAllReduceBase):
    op = EReducePrimitiveOp.Prod
