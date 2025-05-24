from ..state import InterpreterState, InterpreterStateSpec, RegisterSetSpec
from ...platforms.platform import PlatformIRBuilder
from ..instruction import Instruction
from ...platforms.common import *


def emit_set_arith(state: InterpreterState, fn, ispec: InterpreterStateSpec, *rss: RegisterSetSpec):
    state[ispec.rf0] = [fn(*args) for args in zip(*[state[rs] for rs in rss])]


class FAdd(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        emit_set_arith(state, LL.fadd, ispec, ispec.rf0, ispec.rf1)


class FMul(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        emit_set_arith(state, LL.fmul, ispec, ispec.rf0, ispec.rf1)


class FMA(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        emit_set_arith(state, LL.fma, ispec, ispec.rf1, ispec.rf2, ispec.rf0)


class FSub(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        emit_set_arith(state, LL.fsub, ispec, ispec.rf0, ispec.rf1)


class FRSub(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        emit_set_arith(state, LL.fsub, ispec, ispec.rf1, ispec.rf0)


class FNeg(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        emit_set_arith(state, LL.fneg, ispec, ispec.rf0)


class FDiv(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        emit_set_arith(state, LL.fdiv, ispec, ispec.rf0, ispec.rf1)


class FRDiv(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        emit_set_arith(state, LL.fdiv, ispec, ispec.rf1, ispec.rf0)


class FRem(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        emit_set_arith(state, LL.frem, ispec, ispec.rf0, ispec.rf1)
