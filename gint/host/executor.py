import sys
import numpy
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, Union, Optional, Sequence

from ..kernel.interpreter.main import REG_WIDTH


class ConstraintTerm(object):
    def __init__(self, wt_thread, wt_offset, wt_width):
        self.wt_thread = wt_thread
        self.wt_offset = wt_offset
        self.wt_width = wt_width

    def __add__(self, other: "ConstraintTerm"):
        return ConstraintTerm(self.wt_thread + other.wt_thread, self.wt_offset + other.wt_offset, self.wt_width + other.wt_width)

    def __mul__(self, other: int):
        return ConstraintTerm(self.wt_thread * other, self.wt_offset * other, self.wt_width * other)

    def __lt__(self, other: int):
        return Constraint(self, other)

    def __le__(self, other: int):
        return Constraint(self, other + 1)


class Constraint(object):
    def __init__(self, term: ConstraintTerm, size: int):
        self.term = term
        self.size = size


ThreadIdx = ConstraintTerm(1, 0, 0)
WidthIdx = ConstraintTerm(0, 0, 1)
Offset = ConstraintTerm(0, 1, 0)


@dataclass
class ProgramTensorInfo:
    elm_size: int

    t_stride: int
    w_stride: int
    o_stride: int

    b_strides: list[int] = field(default_factory=list)
    b_sizes: list[int] = field(default_factory=list)
    
    b2t_stride: int = 0
    b2t_size: int = 1
    b2w_stride: int = REG_WIDTH
    b2w_size: int = 1

    constraints: list[Constraint] = field(default_factory=list)


@dataclass
class ProgramData:
    program: numpy.ndarray[numpy.int32]
    input_infos: list[ProgramTensorInfo]


@dataclass
class TensorInterface:
    typechr: str
    elm_size: int
    base_ptr: int
    shape: tuple[int, ...]
    strides: tuple[int, ...]  # in elements, not bytes
    assert sys.byteorder == 'little', 'gint does not support big endian yet'

    @property
    def ndim(self):
        return len(self.shape)
    
    @property
    def typestr(self):
        return f'{self.typechr}{self.elm_size}'
    
    @property
    def __cuda_array_interface__(self):
        return {
            'version': 2,
            'data': (self.base_ptr, False),
            'shape': self.shape,
            'typestr': f'<{self.typechr}{self.elm_size}',
            'strides': tuple(x * self.elm_size for x in self.strides)
        }
    
    @classmethod
    def from_cuda_array_interface(cls, cai_supported: Union[dict, Any]):
        cai: Optional[dict] = getattr(cai_supported, '__cuda_array_interface__', None)
        if cai is None:
            if isinstance(cai_supported, dict):
                cai = cai_supported
            else:
                raise TypeError('__cuda_array_interface__ not found for cuda array interface object.')
        if 'typestr' not in cai:
            raise TypeError("Invalid __cuda_array_interface__: missing `typestr`")
        typestr = cai["typestr"]
        shape = cai["shape"]
        cai_version = cai["version"]
        ptr, ro = cai['data']
        assert cai_version >= 2, cai_version
        endian, typechr, nbytes = typestr[0], typestr[1], int(typestr[2:])
        assert endian in '<|', 'gint does not support big endian yet'
        # < little, | irrelevant
        assert not ro, 'gint does not support readonly'
        strides_bytes = cai.get('strides')
        if strides_bytes:
            strides = tuple(x // nbytes for x in strides_bytes)
        else:
            # C-contiguous
            strides = []
            prod = 1
            for s in reversed(shape):
                strides.append(prod)
                prod *= s
            strides = tuple(reversed(strides))
        assert len(shape) == len(strides), (shape, strides, "shape and strides must have the same ndim")
        return TensorInterface(typechr, nbytes, ptr, shape, strides)


def _convert_arg(arg):
    if isinstance(arg, TensorInterface):
        return arg
    else:
        return TensorInterface.from_cuda_array_interface(arg)


class BaseExecutableProgram(object):
    
    REGW = REG_WIDTH
    
    def get_program(self, *args: TensorInterface, **extra_kwargs) -> ProgramData:
        raise NotImplementedError()
    
    def cache_policy(self, *args: TensorInterface, **extra_kwargs) -> Hashable:
        return tuple((x.shape, x.strides, x.typechr, x.elm_size) for x in args) + tuple(sorted(extra_kwargs.items()))
    
    def executor_warp_size(self) -> int:
        return get_executor().warp_size()

    def __call__(self, *args, grid_dim: int, **extra_kwargs):
        get_executor().execute(self, [_convert_arg(x) for x in args], grid_dim, **extra_kwargs)


class BaseExecutor(object):
    
    def warp_size(self) -> int:
        raise NotImplementedError
    
    def execute(self, program: BaseExecutableProgram, args: Sequence[TensorInterface], grid_dim: int, **extra_kwargs):
        raise NotImplementedError


executor: Optional[BaseExecutor] = None


def get_executor():
    global executor
    if executor is None:
        # initialize
        from .cuda.executor_impl import CudaExecutor
        executor = CudaExecutor()
    return executor
