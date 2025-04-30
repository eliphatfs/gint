from ..state import InterpreterState, InterpreterStateSpec
from ...platforms.platform import PlatformIRBuilder
from ..instruction import Instruction
from ...platforms.common import *


# Mov A B: read B write A

class MovF1F0(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        state[ispec.rf1] = list(state[ispec.rf0])


class MovF2F0(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        state[ispec.rf2] = list(state[ispec.rf0])


class MovF3F0(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        state[ispec.rf3] = list(state[ispec.rf0])


class MovF0F1(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        state[ispec.rf0] = list(state[ispec.rf1])


class MovF2F1(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        state[ispec.rf2] = list(state[ispec.rf1])


class MovF3F1(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        state[ispec.rf3] = list(state[ispec.rf1])


class MovF0F2(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        state[ispec.rf0] = list(state[ispec.rf2])


class MovF1F2(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        state[ispec.rf1] = list(state[ispec.rf2])


class MovF3F2(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        state[ispec.rf3] = list(state[ispec.rf2])


class MovF0F3(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        state[ispec.rf0] = list(state[ispec.rf3])


class MovF1F3(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        state[ispec.rf1] = list(state[ispec.rf3])


class MovF2F3(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        state[ispec.rf2] = list(state[ispec.rf3])


all_moves = [
    MovF1F0(),
    MovF2F0(),
    MovF3F0(),
    MovF0F1(),
    MovF2F1(),
    MovF3F1(),
    MovF0F2(),
    MovF1F2(),
    MovF3F2(),
    MovF0F3(),
    MovF1F3(),
    MovF2F3(),
]
