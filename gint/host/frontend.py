import numpy
import functools
from contextvars import ContextVar
from typing import Optional, Sequence
from .executor import TensorInterface, ProgramTensorInfo
from ..kernel.interpreter.main import *


class FrontendState(object):
    
    def __init__(self, fn_args: Sequence[TensorInterface]) -> None:
        self.bc: list[list[int]] = []
        self.tis: list[ProgramTensorInfo] = []
        self.tis_arg_ids: list[int] = []
        self.fn_args = fn_args
        self.fn_args_map = {id(x): i for i, x in enumerate(fn_args)}
        self.tensor_info_map = {}


_frontend_state: ContextVar[Optional[FrontendState]] = ContextVar("_frontend_state", default=None)


def _bc(func):
    @functools.wraps(func)
    def _api_wrapper(*args):
        f = _frontend_state.get()
        if f is None:
            return func(*args)
        else:
            argmap = f.tensor_info_map
            result = func(
                *(argmap.get(id(x), x) for x in args)
            )
            insns_local = INSNS
            f.bc.append([insns_local.get(x, x) for x in result])
            return result

    return _api_wrapper


def _ti(func):
    @functools.wraps(func)
    def _api_wrapper(*args, **kwargs):
        f = _frontend_state.get()
        if f is None:
            return func(*args, **kwargs)
        else:
            # args[0] is the TensorInterface (or its id)
            arg = args[0]
            if not isinstance(arg, int):
                arg = id(arg)
            result = func(*args, **kwargs)
            f.tensor_info_map[id(result)] = len(f.tis)
            f.tis.append(result)
            f.tis_arg_ids.append(f.fn_args_map[arg])
            return result

    return _api_wrapper


@_ti
def make_block_1d(
    t: TensorInterface,
    block1d_shape: int,
    block1d_stride: int,
    block1d_grid_dim: int = 1,
    block1d_grid_step: int = 0,
    batch_strides: list[int] = [],
    batch_shape: list[int] = [],
) -> ProgramTensorInfo:
    return ProgramTensorInfo(
        elm_size=t.elm_size,
        batch_strides=batch_strides,
        batch_shape=batch_shape,
        block_shape_stride_1=[block1d_shape, block1d_stride],
        # dummy for 2D fields
        block_shape_stride_2=[1, 0],
        block_grid_dims=[block1d_grid_dim, 1],
        block_grid_steps=[block1d_grid_step, 0]
    )


@_ti
def make_block_2d(
    t: TensorInterface,
    block2d_shape: list[int],  # in order t, w (32x4)
    block2d_stride: list[int],
    block2d_grid_dims: list[int] = [1, 1],
    block2d_grid_steps: list[int] = [0, 0],
    batch_strides: list[int] = [],
    batch_shape: list[int] = [],
) -> ProgramTensorInfo:
    return ProgramTensorInfo(
        elm_size=t.elm_size,
        batch_strides=batch_strides,
        batch_shape=batch_shape,
        block_shape_stride_1=[block2d_shape[0], block2d_stride[0]],
        block_shape_stride_2=[block2d_shape[1], block2d_stride[1]],
        block_grid_dims=block2d_grid_dims,
        block_grid_steps=block2d_grid_steps
    )


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
def fldg_1d(offset, arg_i):
    return [LoadGlobal1DF32, 16 * offset + arg_i]

@_bc
def fldg_1d_f16(offset, arg_i):
    return [LoadGlobal1DF16, 16 * offset + arg_i]

@_bc
def fldg_1d_bf16(offset, arg_i):
    return [LoadGlobal1DBF16, 16 * offset + arg_i]

@_bc
def fldg_1d_u8(offset, arg_i):
    return [LoadGlobal1DU8, 16 * offset + arg_i]

@_bc
def fstg_1d(offset, arg_i):
    return [StoreGlobal1DF32, 16 * offset + arg_i]

@_bc
def fstg_1d_f16(offset, arg_i):
    return [StoreGlobal1DF16, 16 * offset + arg_i]

@_bc
def fstg_1d_bf16(offset, arg_i):
    return [StoreGlobal1DBF16, 16 * offset + arg_i]

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
def fldg_1d_ind(arg_i):
    return [LoadGlobal1DF32Indirect, arg_i]

@_bc
def fstg_1d_ind(arg_i):
    return [StoreGlobal1DF32Indirect, arg_i]

@_bc
def fpush4(packed_val: int):
    return [LoadImm4F, packed_val]

@_bc
def ipush4(packed_val: int):
    return [LoadImm4I, packed_val]

@_bc
def swap():
    return [Swap, 0]

@_bc
def fldg_2dt(offset, arg_i):
    return [LoadGlobal2DTF32, 16 * offset + arg_i]

@_bc
def fldg_2dt_f16(offset, arg_i):
    return [LoadGlobal2DTF16, 16 * offset + arg_i]

@_bc
def fldg_2dt_bf16(offset, arg_i):
    return [LoadGlobal2DTBF16, 16 * offset + arg_i]

@_bc
def fldg_2dt_u8(offset, arg_i):
    return [LoadGlobal2DTU8, 16 * offset + arg_i]

@_bc
def fstg_2dt(offset, arg_i):
    return [StoreGlobal2DTF32, 16 * offset + arg_i]

@_bc
def fstg_2dt_f16(offset, arg_i):
    return [StoreGlobal2DTF16, 16 * offset + arg_i]

@_bc
def fstg_2dt_bf16(offset, arg_i):
    return [StoreGlobal2DTBF16, 16 * offset + arg_i]

@_bc
def fldg_2dw(offset, arg_i):
    return [LoadGlobal2DWF32, 16 * offset + arg_i]

@_bc
def fldg_2dw_f16(offset, arg_i):
    return [LoadGlobal2DWF16, 16 * offset + arg_i]

@_bc
def fldg_2dw_bf16(offset, arg_i):
    return [LoadGlobal2DWBF16, 16 * offset + arg_i]

@_bc
def fldg_2dw_u8(offset, arg_i):
    return [LoadGlobal2DWU8, 16 * offset + arg_i]

@_bc
def fstg_2dw(offset, arg_i):
    return [StoreGlobal2DWF32, 16 * offset + arg_i]

@_bc
def fstg_2dw_f16(offset, arg_i):
    return [StoreGlobal2DWF16, 16 * offset + arg_i]

@_bc
def fstg_2dw_bf16(offset, arg_i):
    return [StoreGlobal2DWBF16, 16 * offset + arg_i]

@_bc
def adv_block_2d(offset, arg_i):
    return [AdvanceBlock2D, 16 * offset + arg_i]

@_bc
def adv_base(offset, arg_i):
    return [AdvanceBase, 16 * offset + arg_i]

@_bc
def dup_broadcast_w(w):
    return [DupBroadcastW, w]
