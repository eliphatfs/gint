from ..instruction import *
from ..state import InterpreterState, InterpreterStateSpec
from ...platforms.platform import PlatformIRBuilder


class Halt(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        LL.ret_void()
    
    def attrs(self):
        return EInsnAttrs.NoReturn


class Nop(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        pass
