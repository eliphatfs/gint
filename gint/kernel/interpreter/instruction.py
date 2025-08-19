from enum import Flag
from .state import StackMachineState
from ..platforms.platform import PlatformIRBuilder
from ..platforms.common import *


class EInsnAttrs(Flag):
    Nothing = 0
    NoReturn = 1
    Unlikely = 2


class Instruction:
    
    def __init__(self, LL: PlatformIRBuilder, state: StackMachineState):
        self.LL = LL
        self.state = state
    
    def operand(self):
        LL = self.LL
        pc = self.state.pc
        return LL.load(LL.gep(pc, [i32(1)], inbounds=True))
    
    def update_pc(self, delta = None):
        """
        relative update pc
        delta: defaults to i32(2); other ir.Values accepted
        """
        LL = self.LL
        state = self.state
        if delta is None:
            delta = i32(2)
        upd_pc = LL.gep(state.pc, [delta], inbounds=True)
        upd_opcode = LL.load(upd_pc)
        state.pc = upd_pc
        state.opcode = upd_opcode

    def emit_self(self):
        return self.emit(self.LL, self.state)
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        raise NotImplementedError

    def attrs(self):
        return EInsnAttrs.Nothing
