import atexit
import ctypes
from hip import hip
from typing import Callable, List, Tuple, Union


class HipDriverError(RuntimeError):
    """Custom exception for HIP Driver API errors."""
    pass


def check_hip_error(maybe_err, extra_message: Union[bytes, bytearray, str] = ""):
    """Checks for HIP errors and raises an exception if one occurs."""
    err = maybe_err
    if isinstance(err, tuple) and isinstance(err[0], hip.hipError_t):
        err = err[0]
    if isinstance(err, hip.hipError_t):
        if err != hip.hipError_t.hipSuccess:
            if isinstance(extra_message, (bytes, bytearray)):
                extra_message = extra_message.decode(errors='ignore')
            if extra_message:
                print(extra_message)
            raise HipDriverError(f"HIP Driver API Error: {err.name} ({err.value})")
    return maybe_err


class DriverContext(object):
    def __init__(self, device_ordinal: int) -> None:
        self.cleanup_stack: List[Callable[[], None]] = []
        check_hip_error(hip.hipInit(0))
        check_hip_error(hip.hipSetDevice(device_ordinal))
        _, self.device = check_hip_error(hip.hipGetDevice())
        # Get a context handle for caching
        _, self.context = check_hip_error(hip.hipCtxGetCurrent())

    def deferred(self, cleanup_fn: Callable[[], None]):
        self.cleanup_stack.append(cleanup_fn)

    def __enter__(self):
        return self

    def __exit__(self, ty, value, tb):
        for cleanup in reversed(self.cleanup_stack):
            cleanup()


class ExistingDriverContext(DriverContext):
    def __init__(self) -> None:
        self.cleanup_stack: List[Callable[[], None]] = []
        _, self.device = check_hip_error(hip.hipGetDevice())
        _, self.context = check_hip_error(hip.hipCtxGetCurrent())

    def __exit__(self, ty, value, tb):
        for cleanup in reversed(self.cleanup_stack):
            cleanup()


_existing_driver_ctx_cache: dict[int, ExistingDriverContext] = {}


def current_context():
    _, ctx = hip.hipCtxGetCurrent()
    ctx_id = int(ctx)
    if ctx_id == 0:
        # No context yet — HIP may not be initialized
        raise RuntimeError(
            "HIP is not initialized. Please initialize HIP (for example, create any torch cuda tensor) or use `with DriverContext`."
        )
    if ctx_id not in _existing_driver_ctx_cache:
        _existing_driver_ctx_cache[ctx_id] = ExistingDriverContext()
        atexit.register(lambda: _existing_driver_ctx_cache[ctx_id].__exit__(None, None, None))
    return _existing_driver_ctx_cache[ctx_id]


def hipfb_load(driver_ctx: DriverContext, hipfb: bytes, fn_name: bytes) -> hip.hipFunction_t:
    """Load a HIP fat binary (or HSACO) and extract a kernel function."""
    err, module = hip.hipModuleLoadData(hipfb)
    check_hip_error(err)
    driver_ctx.deferred(lambda: check_hip_error(hip.hipModuleUnload(module)))

    err, func = hip.hipModuleGetFunction(module, fn_name)
    check_hip_error(err)
    return func


class CTypesWrapper:
    def __init__(self, c: ctypes._SimpleCData) -> None:
        self.c = c

    def getPtr(self) -> int:
        return ctypes.addressof(self.c)


def prepare_arg(arg: Union[int, float, ctypes._SimpleCData, hip.hipDeviceptr_t]):
    if isinstance(arg, int):
        return CTypesWrapper(ctypes.c_int(arg))
    if isinstance(arg, float):
        return CTypesWrapper(ctypes.c_float(arg))
    if isinstance(arg, ctypes._SimpleCData):
        return CTypesWrapper(arg)
    if isinstance(arg, hip.hipDeviceptr_t):
        return arg
    raise TypeError("Unrecognized argument type for kernel", type(arg))


def launch_kernel(
    kernel: hip.hipFunction_t,
    *args: Union[int, float, ctypes._SimpleCData, hip.hipDeviceptr_t],
    grid_dim: Union[Tuple[int, int, int], int],
    block_dim: Union[Tuple[int, int, int], int],
    smem_bytes: int = 0,
    stream: hip.hipStream_t = hip.hipStream_t(0),
    sync: bool = False
):
    """
    | int args will be represented as native int in C.
    | float args will be represented as single-precision float in C.
    | hipDeviceptr_t will be passed as-is.
    | ctypes scalars will be passed as-is.
    """
    if isinstance(grid_dim, int):
        grid_dim = (grid_dim, 1, 1)
    if isinstance(block_dim, int):
        block_dim = (block_dim, 1, 1)
    assert len(grid_dim) == 3, "Expected scalar or 3 dims in `grid_dim`, got %d" % len(grid_dim)
    assert len(block_dim) == 3, "Expected scalar or 3 dims in `block_dim`, got %d" % len(block_dim)
    preped_args = [prepare_arg(arg) for arg in args]
    kernel_args = (ctypes.c_void_p * len(args))()
    for i, parg in enumerate(preped_args):
        kernel_args[i] = parg.getPtr()
    extra = 0
    err = hip.hipModuleLaunchKernel(
        kernel,
        *grid_dim,
        *block_dim,
        smem_bytes,
        stream,
        kernel_args,
        extra
    )
    check_hip_error(err)
    if sync:
        check_hip_error(hip.hipStreamSynchronize(stream))
