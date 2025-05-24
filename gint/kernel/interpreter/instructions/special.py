from ..state import InterpreterState, InterpreterStateSpec, RegisterSetSpec
from ...platforms.platform import PlatformIRBuilder
from ..instruction import Instruction
from ...platforms.common import *

# portable special functions only for now
# do we add platform-functions like rsqrt?

def emit_set_special(state: InterpreterState, LL: PlatformIRBuilder, ispec: InterpreterStateSpec, intrinsic: str, *rss: RegisterSetSpec):
    state[ispec.rf0] = [LL.intrinsic(intrinsic, f32, list(args)) for args in zip(*[state[rs] for rs in rss])]


class _UnarySpecialBase(Instruction):
    intrinsic: str
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        emit_set_special(state, LL, ispec, self.intrinsic, ispec.rf0)


class _BinarySpecialBase(Instruction):
    intrinsic: str
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        emit_set_special(state, LL, ispec, self.intrinsic, ispec.rf0, ispec.rf1)


class FSqrt(_UnarySpecialBase):
    intrinsic = 'llvm.sqrt.f32'


class FSin(_UnarySpecialBase):
    intrinsic = 'llvm.sin.f32'


class FCos(_UnarySpecialBase):
    intrinsic = 'llvm.cos.f32'


class FTan(_UnarySpecialBase):
    intrinsic = 'llvm.tan.f32'


class FArcSin(_UnarySpecialBase):
    intrinsic = 'llvm.asin.f32'


class FArcCos(_UnarySpecialBase):
    intrinsic = 'llvm.acos.f32'


class FArcTan(_UnarySpecialBase):
    intrinsic = 'llvm.atan.f32'


class FArcTan2(_BinarySpecialBase):
    intrinsic = 'llvm.atan2.f32'


class FPow(_BinarySpecialBase):
    intrinsic = 'llvm.pow.f32'


class FExp(_UnarySpecialBase):
    intrinsic = 'llvm.exp.f32'


class FExp2(_UnarySpecialBase):
    intrinsic = 'llvm.exp2.f32'


class FLog(_UnarySpecialBase):
    intrinsic = 'llvm.log.f32'


class FLog2(_UnarySpecialBase):
    intrinsic = 'llvm.log2.f32'


class FAbs(_UnarySpecialBase):
    intrinsic = 'llvm.fabs.f32'


class FFloor(_UnarySpecialBase):
    intrinsic = 'llvm.floor.f32'


class FCeil(_UnarySpecialBase):
    intrinsic = 'llvm.ceil.f32'


class FTrunc(_UnarySpecialBase):
    intrinsic = 'llvm.trunc.f32'


class FRound(_UnarySpecialBase):
    intrinsic = 'llvm.round.f32'
