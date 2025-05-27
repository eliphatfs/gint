from functools import partial
from ..state import InterpreterState, InterpreterStateSpec, RegisterSetSpec
from ...platforms.platform import PlatformIRBuilder
from ..instruction import Instruction
from ...platforms.common import *


def emit_set_cmp(state: InterpreterState, LL: PlatformIRBuilder, fn, ispec: InterpreterStateSpec, *rss: RegisterSetSpec):
    state[ispec.rf0] = [LL.uitofp(fn(*args), f32) for args in zip(*[state[rs] for rs in rss])]


class FGt(Instruction):
        
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        emit_set_cmp(state, LL, partial(LL.fcmp_ordered, '>'), ispec, ispec.rf1, ispec.rf2)


class FLt(Instruction):
        
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        emit_set_cmp(state, LL, partial(LL.fcmp_ordered, '<'), ispec, ispec.rf1, ispec.rf2)


class FGe(Instruction):
        
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        emit_set_cmp(state, LL, partial(LL.fcmp_ordered, '>='), ispec, ispec.rf1, ispec.rf2)


class FLe(Instruction):
        
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        emit_set_cmp(state, LL, partial(LL.fcmp_ordered, '<='), ispec, ispec.rf1, ispec.rf2)


class FEq(Instruction):
        
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        emit_set_cmp(state, LL, partial(LL.fcmp_ordered, '=='), ispec, ispec.rf1, ispec.rf2)


class FNe(Instruction):
        
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        emit_set_cmp(state, LL, partial(LL.fcmp_unordered, '!='), ispec, ispec.rf1, ispec.rf2)


class FApprox(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        eps = LL.bitcast(state.operand, f32)
        state[ispec.rf0] = [
            LL.uitofp(LL.fcmp_ordered('<', LL.intrinsic('llvm.fabs.f32', f32, [LL.fsub(f0, f1)]), eps), f32)
            for f0, f1 in zip(state[ispec.rf1], state[ispec.rf2])
        ]


class Select(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        state[ispec.rf0] = [
            LL.select(LL.fcmp_ordered('>', b0, f32(0.0)), f0, f1)
            for b0, f0, f1 in zip(state[ispec.rf0], state[ispec.rf1], state[ispec.rf2])
        ]
