import numpy
import functools
from threading import Lock
from typing import Optional, Sequence
from .executor import TensorInterface
from ..kernel.interpreter.main import ILP


class FrontendState(object):
    
    def __init__(self, fn_args: Sequence[TensorInterface]) -> None:
        self.bc: list[list[int]] = []
        self.fn_args = fn_args
        self.fn_args_map = {id(x): i for i, x in enumerate(fn_args)}


_flock = Lock()
_f: list[Optional[FrontendState]] = [None]


def _bc(func):
    @functools.wraps(func)
    def _api_wrapper(*args):
        f = _f[0]
        if f is None:
            return func(*args)
        else:
            argmap = f.fn_args_map
            result = func(
                *(argmap.get(id(x), x) for x in args)
            )
            f.bc.append(result)
            return result

    return _api_wrapper


@_bc
def halt():
    return [0, 0]


@_bc
def ldtinfos():
    return [1, 0]


@_bc
def ldg_f1_float(offset, arg_i):
    return [2, 16 * offset + arg_i]


@_bc
def stg_f0_float(offset, arg_i):
    return [3, 16 * offset + arg_i]


@_bc
def add_f0_f1():
    return [4, 0]


movf_map = {}
x = 5
for src in range(4):
    for dst in range(4):
        if src != dst:
            movf_map[dst, src] = x
            x += 1
del src, dst, x


@_bc
def movf(dst: int, src: int):
    return [movf_map[dst, src], 0]


@_bc
def mul_f0_f1():
    return [17, 0]


@_bc
def fma_f0_f1_f2():  # f0 = f0 + f1 * f2
    return [18, 0]


@_bc
def sub_f0_f1():
    return [19, 0]


@_bc
def rsub_f0_f1():
    return [20, 0]


@_bc
def div_f0_f1():
    return [21, 0]


@_bc
def rdiv_f0_f1():
    return [22, 0]


@_bc
def neg_f0():
    return [23, 0]


@_bc
def immf(dst: int, val: float):
    return [24 + dst, numpy.float32(val).view(numpy.int32)]


@_bc
def warp_allreduce_sum_f0():
    return [28, 0]


@_bc
def warp_allreduce_max_f0():
    return [29, 0]


@_bc
def warp_allreduce_min_f0():
    return [30, 0]


@_bc
def warp_allreduce_prod_f0():
    return [31, 0]


@_bc
def frem_f0_f1():
    return [32, 0]


@_bc
def fsqrt_f0():
    return [33, 0]


@_bc
def fsin_f0():
    return [34, 0]


@_bc
def fcos_f0():
    return [35, 0]


@_bc
def ftan_f0():
    return [36, 0]


@_bc
def fasin_f0():
    return [37, 0]


@_bc
def facos_f0():
    return [38, 0]


@_bc
def fatan_f0():
    return [39, 0]


@_bc
def fatan2_f0():
    return [40, 0]


@_bc
def fpow_f0():
    return [41, 0]


@_bc
def fexp_f0():
    return [42, 0]


@_bc
def fexp2_f0():
    return [43, 0]


@_bc
def flog_f0():
    return [44, 0]


@_bc
def flog2_f0():
    return [45, 0]


@_bc
def frsqrt_f0():
    return [46, 0]


@_bc
def ferf_f0():
    return [47, 0]


@_bc
def ldg_f1_half(offset, arg_i):
    return [48, 16 * offset + arg_i]


@_bc
def stg_f0_half(offset, arg_i):
    return [49, 16 * offset + arg_i]


@_bc
def ldg_f1_bf16(offset, arg_i):
    return [50, 16 * offset + arg_i]


@_bc
def stg_f0_bf16(offset, arg_i):
    return [51, 16 * offset + arg_i]


@_bc
def ldg_f1_u8(offset, arg_i):
    return [52, 16 * offset + arg_i]
