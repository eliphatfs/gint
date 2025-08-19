import sys
import numpy
from dataclasses import dataclass
from collections.abc import Hashable
from typing import Any, Union, Optional, Sequence

from ..kernel.interpreter.main import REG_WIDTH


@dataclass
class ProgramTensorInfo:
    elm_size: int
    
    thread_stride: int
    thread_size: int
    
    block_strides: list[int]
    block_sizes: list[int]
    block_thread_offset_strides: list[int]


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
