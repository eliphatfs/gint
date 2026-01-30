import numpy
import functools
from typing import Callable, Protocol, Hashable, Optional
from .frontend import FrontendState, _frontend_state
from .executor import BaseExecutableProgram, ProgramData, ProgramTensorInfo, TensorInterface


class CachePolicyCallable(Protocol):
    def __call__(self, *args: TensorInterface, **extra_kwargs) -> Hashable: ...


class SugarCallable(Protocol):
    def __call__(self, *args: TensorInterface, REGW: int, WARP: int, **extra_kwargs) -> list[ProgramTensorInfo]: ...


class SugarDeviceCallable(Protocol):
    def __call__(self, *args: TensorInterface, grid_dim: int, **extra_kwargs) -> None: ...


class SugarProgram(BaseExecutableProgram):
    
    def __init__(self, func: Callable, cache_policy_fn: CachePolicyCallable) -> None:
        super().__init__()
        self.func = func
        self.cache_policy_fn = cache_policy_fn
    
    def cache_policy(self, *args: TensorInterface, **extra_kwargs) -> Hashable:
        if self.cache_policy_fn is not None:
            return self.cache_policy_fn(*args, **extra_kwargs)
        else:
            return super().cache_policy(*args, **extra_kwargs)
    
    def get_program(self, *args: TensorInterface, **extra_kwargs) -> ProgramData:
        bc, tis = self.func(*args, REGW=self.REGW, WARP=self.executor_warp_size(), **extra_kwargs)
        return ProgramData(numpy.array(bc, dtype=numpy.int32).reshape(-1), tis)
    

def _bytecode(func: SugarCallable, cache_policy: CachePolicyCallable) -> SugarDeviceCallable:
    
    @functools.wraps(func)
    def sugar_wrapper(*args, REGW: int, WARP: int, **extra_kwargs):
        token = _frontend_state.set(FrontendState(args))
        try:
            tis = func(*args, REGW=REGW, WARP=WARP, **extra_kwargs)
            f = _frontend_state.get()
            return f.bc, tis
        finally:
            _frontend_state.reset(token)
    
    return SugarProgram(sugar_wrapper, cache_policy)



def bytecode(func: Optional[SugarCallable] = None, cache_policy: Optional[CachePolicyCallable] = None) -> SugarDeviceCallable:
    if func is None:
        return functools.partial(_bytecode, cache_policy=cache_policy)
    else:
        return _bytecode(func, cache_policy)
