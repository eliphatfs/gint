from ..state import StackMachineState
from ...platforms.platform import PlatformIRBuilder
from ..instruction import DefaultControlOperandInstruction
from ...platforms.common import *


class LoadImm(DefaultControlOperandInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        state.push([LL.bitcast(self.op, f32)] * state.reg_width)


class FAddImm(DefaultControlOperandInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        spx = state.peek()
        operand = LL.bitcast(self.op, f32)
        state.pop().push([LL.fadd(x, operand) for x in spx])


class FMulImm(DefaultControlOperandInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        spx = state.peek()
        operand = LL.bitcast(self.op, f32)
        state.pop().push([LL.fmul(x, operand) for x in spx])


class FMAImm(DefaultControlOperandInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        operand = LL.bitcast(self.op, ir.VectorType(f16, 2))
        mul = LL.fpext(LL.extract_element(operand, i32(0)), f32)
        add = LL.fpext(LL.extract_element(operand, i32(1)), f32)
        spx = state.peek()
        state.pop().push([LL.fma(x, mul, add) for x in spx])
