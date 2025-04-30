from ..state import InterpreterState, InterpreterStateSpec
from ...platforms.platform import PlatformIRBuilder
from ..instruction import Instruction
from ...platforms.common import *


class FAddTo(Instruction):
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        state[ispec.rf0] = [
            LL.fadd(x, y) for x, y in zip(state[ispec.rf0], state[ispec.rf1])
        ]
