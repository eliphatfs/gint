from llvmlite import ir
from ..state import InterpreterState, InterpreterStateSpec
from ...platforms.platform import PlatformIRBuilder
from ..instruction import Instruction
from ...platforms.common import *


class FAddTo(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        a0, a1, a2, a3 = state[ispec.rf0]
        b0, b1, b2, b3 = state[ispec.rf1]
        state[ispec.rf0] = [
            LL.fadd(a0, b0),
            LL.fadd(a1, b1),
            LL.fadd(a2, b2),
            LL.fadd(a3, b3),
        ]
