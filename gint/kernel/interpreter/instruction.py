from enum import Flag
from .state import InterpreterState, InterpreterStateSpec
from ..platforms.platform import PlatformIRBuilder


class EInsnAttrs(Flag):
    Nothing = 0
    NoReturn = 1
    Unlikely = 2
    Operand = 4


class Instruction:
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        raise NotImplementedError

    def attrs(self):
        return EInsnAttrs.Nothing
