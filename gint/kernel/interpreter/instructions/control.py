from ..instruction import *
from ..state import StackMachineState
from ...platforms.platform import PlatformIRBuilder


class Halt(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        LL.ret_void()
    
    def attrs(self):
        return EInsnAttrs.NoReturn


class Nop(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        pass
