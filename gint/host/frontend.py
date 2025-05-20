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
