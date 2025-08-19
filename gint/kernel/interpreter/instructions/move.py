from ..state import StackMachineState
from ...platforms.platform import PlatformIRBuilder
from ..instruction import DefaultControlInstruction
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
