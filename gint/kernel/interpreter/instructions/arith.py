from ..state import StackMachineState
from ...platforms.platform import PlatformIRBuilder
from ..instruction import DefaultControlInstruction
from ...platforms.common import *


def emit_set_arith(state: StackMachineState, fn, *rss: int):
    assert set(rss) == set(range(len(rss)))
    res = [fn(*args) for args in zip(*[state.peek(rs) for rs in rss])]
    for _ in rss:
        state.pop()
    state.push(res)


class FAdd(DefaultControlInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_arith(state, LL.fadd, 0, 1)


class FMul(DefaultControlInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_arith(state, LL.fmul, 0, 1)


class FMA(DefaultControlInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_arith(state, LL.fma, 0, 1, 2)


class FSub(DefaultControlInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_arith(state, LL.fsub, 0, 1)


class FRSub(DefaultControlInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_arith(state, LL.fsub, 1, 0)


class FNeg(DefaultControlInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_arith(state, LL.fneg, 0)


class FDiv(DefaultControlInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_arith(state, LL.fdiv, 0, 1)


class FRDiv(DefaultControlInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_arith(state, LL.fdiv, 1, 0)


class FRem(DefaultControlInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_arith(state, LL.frem, 0, 1)
