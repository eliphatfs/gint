from ..state import StackMachineState
from ...platforms.platform import PlatformIRBuilder
from ..instruction import DefaultControlInstruction
from ...platforms.common import *


def emit_set_arith_i32(LL: PlatformIRBuilder, state: StackMachineState, fn, *rss: int):
    assert set(rss) == set(range(len(rss)))
    
    def cast_to_i32(v):
        return LL.bitcast(v, i32)
    
    def cast_to_f32(v):
        return LL.bitcast(v, f32)

    args_list = [state.peek(rs) for rs in rss]
    # args_list is list of lists (each inner list is REG_WIDTH elements)
    
    res_i32 = [fn(*[cast_to_i32(arg) for arg in args]) for args in zip(*args_list)]
    res_f32 = [cast_to_f32(v) for v in res_i32]
    
    for _ in rss:
        state.pop()
    state.push(res_f32)


class IAdd(DefaultControlInstruction):
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_arith_i32(LL, state, LL.add, 0, 1)


class IMul(DefaultControlInstruction):
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_arith_i32(LL, state, LL.mul, 0, 1)


class ISub(DefaultControlInstruction):
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_arith_i32(LL, state, LL.sub, 0, 1)


class IDiv(DefaultControlInstruction):
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_arith_i32(LL, state, LL.sdiv, 0, 1)


class IRem(DefaultControlInstruction):
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_arith_i32(LL, state, LL.srem, 0, 1)


class IShl(DefaultControlInstruction):
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_arith_i32(LL, state, LL.shl, 0, 1)


class IShr(DefaultControlInstruction):
    """Logical shift right"""
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_arith_i32(LL, state, LL.lshr, 0, 1)


class IAnd(DefaultControlInstruction):
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_arith_i32(LL, state, LL.and_, 0, 1)


class IOr(DefaultControlInstruction):
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_arith_i32(LL, state, LL.or_, 0, 1)


class IXor(DefaultControlInstruction):
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        emit_set_arith_i32(LL, state, LL.xor, 0, 1)
