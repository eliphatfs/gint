from ..state import InterpreterState, InterpreterStateSpec
from ...platforms.platform import PlatformIRBuilder
from ..instruction import Instruction
from ...platforms.common import *


class LoadF0Imm(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        state[ispec.rf0] = [LL.bitcast(state.operand, f32)] * ispec.ilp


class LoadF1Imm(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        state[ispec.rf1] = [LL.bitcast(state.operand, f32)] * ispec.ilp


class LoadF2Imm(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        state[ispec.rf2] = [LL.bitcast(state.operand, f32)] * ispec.ilp


class LoadF3Imm(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        state[ispec.rf3] = [LL.bitcast(state.operand, f32)] * ispec.ilp


class FAddImm(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        operand = LL.bitcast(state.operand, f32)
        state[ispec.rf0] = [LL.fadd(x, operand) for x in state[ispec.rf0]]


class FMulImm(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        operand = LL.bitcast(state.operand, f32)
        state[ispec.rf0] = [LL.fmul(x, operand) for x in state[ispec.rf0]]


class FMAImm(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        operand = LL.bitcast(state.operand, ir.VectorType(f16, 2))
        mul = LL.fpext(LL.extract_element(operand, i32(0)), f32)
        add = LL.fpext(LL.extract_element(operand, i32(1)), f32)
        state[ispec.rf0] = [LL.fma(x, mul, add) for x in state[ispec.rf0]]
