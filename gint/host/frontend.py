import numpy
import functools
from contextvars import ContextVar
from typing import Optional, Sequence
from .executor import TensorInterface
from ..kernel.interpreter.main import *


class FrontendState(object):
    
    def __init__(self, fn_args: Sequence[TensorInterface]) -> None:
        self.bc: list[list[int]] = []
        self.fn_args = fn_args
        self.fn_args_map = {id(x): i for i, x in enumerate(fn_args)}


_frontend_state: ContextVar[Optional[FrontendState]] = ContextVar("_frontend_state", default=None)


def _bc(func):
    @functools.wraps(func)
    def _api_wrapper(*args):
        f = _frontend_state.get()
        if f is None:
            return func(*args)
        else:
            argmap = f.fn_args_map
            result = func(
                *(argmap.get(id(x), x) for x in args)
            )
            insns_local = INSNS
            f.bc.append([insns_local.get(x, x) for x in result])
            return result

    return _api_wrapper


@_bc
def halt():
    return [Halt, 0]

@_bc
def nop():
    return [Nop, 0]

@_bc
def fadd():
    return [FAdd, 0]

@_bc
def fmul():
    return [FMul, 0]

@_bc
def fma():
    return [FMA, 0]

@_bc
def fsub():
    return [FSub, 0]

@_bc
def frsub():
    return [FRSub, 0]

@_bc
def fneg():
    return [FNeg, 0]

@_bc
def fdiv():
    return [FDiv, 0]

@_bc
def frdiv():
    return [FRDiv, 0]

@_bc
def frem():
    return [FRem, 0]

@_bc
def fpush(val: float):
    return [LoadImm, numpy.float32(val).view(numpy.int32)]

@_bc
def faddimm(val: float):
    return [FAddImm, numpy.float32(val).view(numpy.int32)]

@_bc
def fmulimm(val: float):
    return [FMulImm, numpy.float32(val).view(numpy.int32)]

@_bc
def fmaimm(mul: float, add: float):
    """
    note: the immediate values have only half precision
    """
    return [FMAImm, numpy.array([mul, add], dtype=numpy.float16).view(numpy.int32).item()]

@_bc
def fldg(offset, arg_i):
    return [LoadGlobalF32, 16 * offset + arg_i]

@_bc
def fldg_f16(offset, arg_i):
    return [LoadGlobalF16, 16 * offset + arg_i]

@_bc
def fldg_bf16(offset, arg_i):
    return [LoadGlobalBF16, 16 * offset + arg_i]

@_bc
def fldg_u8(offset, arg_i):
    return [LoadGlobalU8, 16 * offset + arg_i]

@_bc
def fstg(offset, arg_i):
    return [StoreGlobalF32, 16 * offset + arg_i]

@_bc
def fstg_f16(offset, arg_i):
    return [StoreGlobalF16, 16 * offset + arg_i]

@_bc
def fstg_bf16(offset, arg_i):
    return [StoreGlobalBF16, 16 * offset + arg_i]

@_bc
def pop():
    return [Pop, 0]

@_bc
def pop2():
    return [Pop2, 0]

@_bc
def dup():
    return [Dup, 0]

@_bc
def dupx1():
    return [DupX1, 0]

@_bc
def dupx2():
    return [DupX2, 0]

@_bc
def dup2():
    return [Dup2, 0]

@_bc
def fgt():
    return [FGt, 0]

@_bc
def flt():
    return [FLt, 0]

@_bc
def fge():
    return [FGe, 0]

@_bc
def fle():
    return [FLe, 0]

@_bc
def feq():
    return [FEq, 0]

@_bc
def fne():
    return [FNe, 0]

@_bc
def fapprox(eps: float):
    return [FApprox, numpy.float32(eps).view(numpy.int32)]

@_bc
def fselect():
    return [Select, 0]

@_bc
def warp_allreduce_fsum():
    return [WarpAllReduceSum, 0]

@_bc
def warp_allreduce_fmin():
    return [WarpAllReduceMin, 0]

@_bc
def warp_allreduce_fmax():
    return [WarpAllReduceMax, 0]

@_bc
def warp_allreduce_fprod():
    return [WarpAllReduceProd, 0]

@_bc
def fsqrt():
    return [FSqrt, 0]

@_bc
def fsin():
    return [FSin, 0]

@_bc
def fcos():
    return [FCos, 0]

@_bc
def ftan():
    return [FTan, 0]

@_bc
def fasin():
    return [FArcSin, 0]

@_bc
def facos():
    return [FArcCos, 0]

@_bc
def fatan():
    return [FArcTan, 0]

@_bc
def fatan2():
    return [FArcTan2, 0]

@_bc
def fpow():
    return [FPow, 0]

@_bc
def fexp():
    return [FExp, 0]

@_bc
def fexp2():
    return [FExp2, 0]

@_bc
def flog():
    return [FLog, 0]

@_bc
def flog2():
    return [FLog2, 0]

@_bc
def frsqrt():
    return [FRSqrt, 0]

@_bc
def ferf():
    return [FErf, 0]

@_bc
def iadd():
    return [IAdd, 0]

@_bc
def imul():
    return [IMul, 0]

@_bc
def isub():
    return [ISub, 0]

@_bc
def idiv():
    return [IDiv, 0]

@_bc
def irem():
    return [IRem, 0]

@_bc
def ishl():
    return [IShl, 0]

@_bc
def ishr():
    return [IShr, 0]

@_bc
def iand():
    return [IAnd, 0]

@_bc
def ior():
    return [IOr, 0]

@_bc
def ixor():
    return [IXor, 0]

@_bc
def fldg_ind(arg_i):
    return [LoadGlobalF32Indirect, arg_i]

@_bc
def fstg_ind(arg_i):
    return [StoreGlobalF32Indirect, arg_i]

@_bc
def fpush4(packed_val: int):
    return [LoadImm4F, packed_val]

@_bc
def ipush4(packed_val: int):
    return [LoadImm4I, packed_val]

@_bc
def swap():
    return [Swap, 0]
