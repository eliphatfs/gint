from ..state import StackMachineState
from ...platforms.platform import PlatformIRBuilder
from ..instruction import DefaultControlInstruction, DefaultControlOperandInstruction
from ...platforms.common import *


class Pop(DefaultControlInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        state.pop()


class Pop2(DefaultControlInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        state.pop().pop()


class Dup(DefaultControlInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        state.push(state.peek())


class DupX1(DefaultControlInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        v1 = state.peek()
        v2 = state.peek(1)
        state.pop().pop().push(v1).push(v2).push(v1)


class DupX2(DefaultControlInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        v1 = state.peek()
        v2 = state.peek(1)
        v3 = state.peek(2)
        state.pop().pop().pop().push(v1).push(v3).push(v2).push(v1)


class Dup2(DefaultControlInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        v1 = state.peek()
        v2 = state.peek(1)
        state.push(v2).push(v1)


class Swap(DefaultControlInstruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        v1 = state.peek()
        v2 = state.peek(1)
        state.pop().pop().push(v1).push(v2)


def binary_cond_tree(LL: PlatformIRBuilder, v: list[ir.Value], low: int, high: int, op: ir.Value) -> ir.Value:
    if low >= high - 1:
        return v[low]
    mid = (low + high) // 2
    return LL.select(
        LL.icmp_signed('<', op, i32(mid)),
        binary_cond_tree(LL, v, low, mid, op),
        binary_cond_tree(LL, v, mid, high, op)
    )


class DupBroadcastW(DefaultControlOperandInstruction):

    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        v = state.peek()
        op = self.op
        state.push([binary_cond_tree(LL, v, 0, len(v), op)] * state.reg_width)


class FPermW(DefaultControlOperandInstruction):
    """
    Permute the 4 elements of the top-of-stack vector.
    Operand is an i32 encoding 4 i8 indices (i8x4 layout, little-endian):
      bits  7.. 0 -> index for output lane 0
      bits 15.. 8 -> index for output lane 1
      bits 23..16 -> index for output lane 2
      bits 31..24 -> index for output lane 3
    Each index selects which input lane (0–3) feeds the corresponding output lane.
    """

    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        v = state.peek()
        op = self.op  # i32

        # Reinterpret the i32 operand as a <4 x i8> vector to extract per-lane indices.
        vec_i8x4 = ir.VectorType(i8, 4)
        op_vec = LL.bitcast(op, vec_i8x4)

        result = []
        for lane in range(state.reg_width):
            idx_i8 = LL.extract_element(op_vec, i32(lane))
            idx = LL.zext(idx_i8, i32)
            # Build a binary selection tree over the 4 input elements.
            selected = binary_cond_tree(LL, v, 0, len(v), idx)
            result.append(selected)

        state.pop().push(result)


class FShuf2(DefaultControlOperandInstruction):
    """
    Shuffle lanes from two top-of-stack vectors into one, like VecShuffle(vec1, vec2, x,y,z,w):
      return (vec1[x], vec1[y], vec2[z], vec2[w])
    vec2 is on top of the stack, vec1 is below it.
    Operand is an i32 encoding 4 i8 indices (i8x4 layout, little-endian):
      bits  7.. 0 -> index into vec1 for output lane 0  (x)
      bits 15.. 8 -> index into vec1 for output lane 1  (y)
      bits 23..16 -> index into vec2 for output lane 2  (z)
      bits 31..24 -> index into vec2 for output lane 3  (w)
    """

    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        vec2 = state.peek()
        vec1 = state.peek(1)
        op = self.op  # i32

        vec_i8x4 = ir.VectorType(i8, 4)
        op_vec = LL.bitcast(op, vec_i8x4)

        result = []
        for lane in range(state.reg_width):
            idx_i8 = LL.extract_element(op_vec, i32(lane))
            idx = LL.zext(idx_i8, i32)
            src = vec1 if lane < 2 else vec2
            selected = binary_cond_tree(LL, src, 0, len(src), idx)
            result.append(selected)

        state.pop().pop().push(result)


class DupBroadcastT(DefaultControlOperandInstruction):
    """
    Currently disabled because leads to mysterious increase in register per thread
    """

    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        v = state.peek()
        op = self.op
        state.push([LL.warp_broadcast_lane(v[i], op) for i in range(state.reg_width)])
