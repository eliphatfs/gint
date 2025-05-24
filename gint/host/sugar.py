import numpy
import functools
from typing import Callable, Protocol, Hashable, Optional
from .frontend import FrontendState, _f, _flock
from .executor import BaseExecutableProgram, ProgramData, ProgramTensorInfo, TensorInterface


class CachePolicyCallable(Protocol):
    def __call__(self, *args: TensorInterface) -> Hashable: ...


class SugarCallable(Protocol):
    def __call__(self, *args: TensorInterface, ILP: int, WARP: int) -> list[ProgramTensorInfo]: ...


class SugarDeviceCallable(Protocol):
    def __call__(self, *args: TensorInterface, grid_dim: int) -> None: ...


class SugarProgram(BaseExecutableProgram):
    
    def __init__(self, func: Callable, cache_policy_fn: CachePolicyCallable) -> None:
        super().__init__()
        self.func = func
        self.cache_policy_fn = cache_policy_fn
    
    def cache_policy(self, *args: TensorInterface) -> Hashable:
        if self.cache_policy_fn is not None:
            return self.cache_policy_fn(*args)
        else:
            return super().cache_policy(*args)
    
    def get_program(self, *args: TensorInterface) -> ProgramData:
        bc, tis = self.func(*args, ILP=self.ILP, WARP=self.executor_warp_size())
        return ProgramData(numpy.array(bc, dtype=numpy.int32).reshape(-1), tis)
    

def _bytecode(func: SugarCallable, cache_policy: CachePolicyCallable) -> SugarDeviceCallable:
    
    @functools.wraps(func)
    def sugar_wrapper(*args, ILP: int, WARP: int):
        global _f
        with _flock:
            try:
                _f[0] = FrontendState(args)
                tis = func(*args, ILP=ILP, WARP=WARP)
                return _f[0].bc, tis
            finally:
                _f[0] = None
    
    return SugarProgram(sugar_wrapper, cache_policy)



def bytecode(func: Optional[SugarCallable] = None, cache_policy: Optional[CachePolicyCallable] = None) -> SugarDeviceCallable:
    if func is None:
        return functools.partial(_bytecode, cache_policy=cache_policy)
    else:
        return _bytecode(func, cache_policy)
