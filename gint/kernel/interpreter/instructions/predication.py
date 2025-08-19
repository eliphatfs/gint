from functools import partial
from ..state import StackMachineState
from ...platforms.platform import PlatformIRBuilder
from ..instruction import DefaultControlInstruction, DefaultControlOperandInstruction
from ...platforms.common import *


def emit_set_cmp(state: StackMachineState, LL: PlatformIRBuilder, fn, *rss: int):
    assert set(rss) == set(range(len(rss)))
    res = [LL.uitofp(fn(*args), f32) for args in zip(*[state.peek(rs) for rs in rss])]
    for _ in rss:
        state.pop()
    state.push(res)


class FGt(DefaultControlInstruction):
        
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_cmp(state, LL, partial(LL.fcmp_ordered, '>'), 0, 1)


class FLt(DefaultControlInstruction):
        
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_cmp(state, LL, partial(LL.fcmp_ordered, '<'), 0, 1)


class FGe(DefaultControlInstruction):
        
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_cmp(state, LL, partial(LL.fcmp_ordered, '>='), 0, 1)


class FLe(DefaultControlInstruction):
        
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_cmp(state, LL, partial(LL.fcmp_ordered, '<='), 0, 1)


class FEq(DefaultControlInstruction):
        
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_cmp(state, LL, partial(LL.fcmp_ordered, '=='), 0, 1)


class FNe(DefaultControlInstruction):
        
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_cmp(state, LL, partial(LL.fcmp_unordered, '!='), 0, 1)


class FApprox(DefaultControlOperandInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        eps = LL.bitcast(self.op, f32)
        s0 = state.peek()
        s1 = state.peek(1)
        state.pop().pop().push([
            LL.uitofp(LL.fcmp_ordered('<', LL.intrinsic('llvm.fabs.f32', f32, [LL.fsub(f0, f1)]), eps), f32)
            for f0, f1 in zip(s0, s1)
        ])


class Select(DefaultControlInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        s0 = state.peek()
        s1 = state.peek(1)
        s2 = state.peek(2)
        state.pop().pop().pop().push([
            LL.select(LL.fcmp_ordered('>', b0, f32(0.0)), f0, f1)
            for b0, f0, f1 in zip(s0, s1, s2)
        ])
