import sys
import numpy
from dataclasses import dataclass
from collections.abc import Hashable
from typing import Any, Union, Optional, Sequence


@dataclass
class ProgramTensorInfo:
    elm_size: int
    
    thread_stride: int
    thread_size: int
    
    block_strides: list[int]
    block_sizes: list[int]


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
    
    def __cuda_array_interface__(self):
        return {
            'version': 2,
            'data': (self.base_ptr, False),
            'shape': self.shape,
            'typestr': f'<{self.typechr}{self.elm_size}',
            'strides': self.strides
        }
    
    @classmethod
    def from_cuda_array_interface(cls, cai_supported: Union[dict, Any]):
        if hasattr(cai_supported, '__cuda_array_interface__'):
            cai: dict = cai_supported.__cuda_array_interface__
        elif not isinstance(cai_supported, dict):
            raise TypeError('__cuda_array_interface__ not found for cuda array interface object.')
        else:
            cai: dict = cai_supported
        if 'typestr' not in cai:
            raise TypeError("Invalid __cuda_array_interface__: missing `typestr`")
        typestr = cai["typestr"]
        shape = cai["shape"]
        cai_version = cai["version"]
        ptr, ro = cai['data']
        assert cai_version >= 2, cai_version
        endian, typechr, nbytes = typestr[0], typestr[1], int(typestr[2:])
        assert endian == '<', 'gint does not support big endian yet'
        assert sys.byteorder == 'little', 'gint does not support big endian yet'
        assert not ro, 'gint does not support readonly'
        strides_bytes = cai.get('strides')
        if strides_bytes:
            strides = [x // nbytes for x in strides_bytes]
        else:
            # C-contiguous
            strides = []
            prod = 1
            for s in reversed(shape):
                strides.append(prod)
                prod *= s
            strides = list(reversed(strides))
        return TensorInterface(typechr, nbytes, ptr, shape, strides)


class BaseExecutableProgram(object):
    
    def get_program(self, *args: TensorInterface) -> ProgramData:
        raise NotImplementedError()
    
    def cache_policy(self, *args: TensorInterface) -> Hashable:
        raise NotImplementedError

    def __call__(self, *args: TensorInterface, grid_dim: int):
        get_executor().execute(self, args, grid_dim)


class BaseExecutor(object):
    
    def execute(self, program: BaseExecutableProgram, args: Sequence[TensorInterface], grid_dim: int):
        raise NotImplementedError


executor: Optional[BaseExecutor] = None


def get_executor():
    global executor
    if executor is None:
        # initialize
        from .cuda.executor_impl import CudaExecutor
        executor = CudaExecutor()
    return executor
