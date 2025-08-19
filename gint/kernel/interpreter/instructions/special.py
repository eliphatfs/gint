from ..state import StackMachineState
from ...platforms.platform import PlatformIRBuilder
from ..instruction import DefaultControlInstruction, EInsnAttrs
from ...platforms.common import *


class _UnarySpecialBase(DefaultControlInstruction):
    op: EUnarySpecialOp
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        reg = state.peek()
        state.pop().push([LL.special_unary(x, self.op) for x in reg])
    
    def attrs(self):
        return EInsnAttrs.Unlikely


class _BinarySpecialBase(DefaultControlInstruction):
    op: EBinarySpecialOp
    
    def emit(self, LL: PlatformIRBuilder, state: StackMachineState):
        r0 = state.peek()
        r1 = state.peek(1)
        state.pop().pop().push([LL.special_binary(a, b, self.op) for a, b in zip(r0, r1)])
    
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
