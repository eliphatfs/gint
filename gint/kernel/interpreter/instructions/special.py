from ..state import InterpreterState, InterpreterStateSpec, RegisterSetSpec
from ...platforms.platform import PlatformIRBuilder
from ..instruction import Instruction, EInsnAttrs
from ...platforms.common import *

# portable special functions only for now
# do we add platform-functions like rsqrt?

def emit_set_special_unary(state: InterpreterState, LL: PlatformIRBuilder, ispec: InterpreterStateSpec, op: EUnarySpecialOp, *rss: RegisterSetSpec):
    state[ispec.rf0] = [LL.special_unary(x, op) for x, in zip(*[state[rs] for rs in rss])]


def emit_set_special_binary(state: InterpreterState, LL: PlatformIRBuilder, ispec: InterpreterStateSpec, op: EBinarySpecialOp, *rss: RegisterSetSpec):
    state[ispec.rf0] = [LL.special_binary(a, b, op) for a, b in zip(*[state[rs] for rs in rss])]


class _UnarySpecialBase(Instruction):
    op: EUnarySpecialOp
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        emit_set_special_unary(state, LL, ispec, self.op, ispec.rf0)
    
    def attrs(self):
        return EInsnAttrs.Unlikely


class _BinarySpecialBase(Instruction):
    op: EBinarySpecialOp
    
    def emit(self, LL: PlatformIRBuilder, state: InterpreterState, ispec: InterpreterStateSpec):
        emit_set_special_binary(state, LL, ispec, self.op, ispec.rf0, ispec.rf1)
    
    def attrs(self):
        return EInsnAttrs.Unlikely


class FSqrt(_UnarySpecialBase):
    op = EUnarySpecialOp.Sqrt


class FSin(_UnarySpecialBase):
    op = EUnarySpecialOp.Sin


class FCos(_UnarySpecialBase):
    op = EUnarySpecialOp.Cos


class FTan(_UnarySpecialBase):
    op = EUnarySpecialOp.Tan


class FArcSin(_UnarySpecialBase):
    op = EUnarySpecialOp.ArcSin


class FArcCos(_UnarySpecialBase):
    op = EUnarySpecialOp.ArcCos


class FArcTan(_UnarySpecialBase):
    op = EUnarySpecialOp.ArcTan


class FArcTan2(_BinarySpecialBase):
    op = EBinarySpecialOp.ArcTan2


class FPow(_BinarySpecialBase):
    op = EBinarySpecialOp.Pow


class FExp(_UnarySpecialBase):
    op = EUnarySpecialOp.Exp


class FExp2(_UnarySpecialBase):
    op = EUnarySpecialOp.Exp2


class FLog(_UnarySpecialBase):
    op = EUnarySpecialOp.Log


class FLog2(_UnarySpecialBase):
    op = EUnarySpecialOp.Log2


class FRSqrt(_UnarySpecialBase):
    op = EUnarySpecialOp.RSqrt


class FErf(_UnarySpecialBase):
    op = EUnarySpecialOp.Erf
