import os
import sys
import numpy
from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Any, Union, Optional, Sequence

from ..kernel.interpreter.main import REG_WIDTH, VARIANTS, DEFAULT_VARIANT
from .analyzer import analyze_bytecode, BytecodeStats


@dataclass
class ProgramTensorInfo:
    elm_size: int

    batch_strides: list[int]
    batch_shape: list[int]

    block_shape_stride_1: list[int]
    block_shape_stride_2: list[int]
    block_grid_dims: list[int]
    block_grid_steps: list[int]

@dataclass
class ProgramData:
    program: numpy.ndarray[numpy.int32]
    input_infos: list[ProgramTensorInfo]
    input_indices: Optional[list[int]] = None
    _stats: Optional[BytecodeStats] = field(default=None, repr=False, compare=False)

    @property
    def stats(self) -> BytecodeStats:
        if self._stats is None:
            self._stats = analyze_bytecode(self.program)
        return self._stats


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


def select_variant(bytecode_or_pd) -> str:
    """Pick the smallest kernel variant whose limits fit the given bytecode.

    Accepts either a raw bytecode array or a ``ProgramData`` (in which case
    the cached ``stats`` is reused — analyze_bytecode walks the whole stream
    and is the dominant cost on the cache-miss path of execute()).

    Variants are ordered by ascending pool size; the first one that satisfies
    the program's measured stack depth and register usage wins. Falls back
    to DEFAULT_VARIANT if none fits (which currently shouldn't happen since
    the largest variant covers the full instruction set).
    """
    stats = (bytecode_or_pd.stats if isinstance(bytecode_or_pd, ProgramData)
             else analyze_bytecode(bytecode_or_pd))
    candidates = sorted(VARIANTS.items(), key=lambda kv: kv[1][0])  # by pool_size
    for name, (pool_size, num_regs, max_stack) in candidates:
        if stats.max_stack <= max_stack and (stats.max_reg_idx + 1) <= num_regs:
            return name
    return DEFAULT_VARIANT


def _variant_max(a: str, b: str) -> str:
    """Return the larger of two variants by pool size."""
    return a if VARIANTS[a][0] >= VARIANTS[b][0] else b


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


# Make every gint kernel safe to call from inside upstream ``torch.compile``.
# Without this, dynamo recurses into the bytecode-recording and executor
# stack — ContextVar set/reset, opaque cuda-bindings — and blows up with
# internal errors like ``'NoneType' object has no attribute 'make_guard'``.
# ``torch.compiler.disable`` graph-breaks at the call site: dynamo compiles
# up to the call, runs it eagerly, then resumes tracing. It is a near no-op
# (~1us/call) outside an active trace.
try:
    import torch as _torch
    BaseExecutableProgram.__call__ = _torch.compiler.disable(
        BaseExecutableProgram.__call__
    )
except (ImportError, AttributeError):
    pass


class BaseExecutor(object):

    def warp_size(self) -> int:
        raise NotImplementedError

    def execute(self, program: BaseExecutableProgram, args: Sequence[TensorInterface], grid_dim: int, **extra_kwargs):
        raise NotImplementedError

    def execute_indirect(self, programs: Sequence[BaseExecutableProgram], args_list: Sequence[Sequence[TensorInterface]], indices: Sequence[int], **extra_kwargs):
        raise NotImplementedError


executor: Optional[BaseExecutor] = None


def get_executor():
    global executor
    if executor is None:
        backend = os.environ.get('GINT_BACKEND', '').lower()
        if backend == 'hip':
            from .hip.executor_impl import HipExecutor
            executor = HipExecutor()
        elif backend == 'cuda':
            from .cuda.executor_impl import CudaExecutor
            executor = CudaExecutor()
        else:
            # Auto-detect: try CUDA first (backward compatible), then HIP
            try:
                from .cuda.executor_impl import CudaExecutor
                executor = CudaExecutor()
            except Exception:
                from .hip.executor_impl import HipExecutor
                executor = HipExecutor()
    return executor
