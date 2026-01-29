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


class LoadImm4F(DefaultControlOperandInstruction):
    """Load 4 int8 packed in i32 operand, cast each to f32"""
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        vals = []
        for i in range(4):
            # Extract i-th byte (little endian: byte 0 is LSB)
            v_i8 = LL.trunc(LL.lshr(self.op, i32(8 * i)), i8)
            v_f32 = LL.sitofp(v_i8, f32)
            vals.append(v_f32)
        state.push(vals)


class LoadImm4I(DefaultControlOperandInstruction):
    """Load 4 int8 packed in i32 operand, cast each to i32 and bitcast to f32"""
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        vals = []
        for i in range(4):
            # Extract i-th byte
            v_i8 = LL.trunc(LL.lshr(self.op, i32(8 * i)), i8)
            v_i32 = LL.sext(v_i8, i32)
            v_f32 = LL.bitcast(v_i32, f32)
            vals.append(v_f32)
        state.push(vals)
