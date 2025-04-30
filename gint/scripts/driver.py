import ctypes
from cuda import cuda
from typing import Callable, List, Tuple, Union


class CudaDriverError(RuntimeError):
    """Custom exception for CUDA Driver API errors."""
    pass


def check_cuda_error(maybe_err, extra_message: Union[bytes, bytearray, str] = ""):
    """Checks for CUDA errors and raises an exception if one occurs."""
    err = maybe_err
    if isinstance(err, tuple) and isinstance(err[0], cuda.CUresult):
        err = err[0]
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            # [0] is CUresult
            err_name = cuda.cuGetErrorName(err)[1]
            err_string = cuda.cuGetErrorString(err)[1]
            if isinstance(extra_message, (bytes, bytearray)):
                extra_message = extra_message.decode(errors='ignore')
            print(extra_message)
            raise CudaDriverError(f"CUDA Driver API Error: {err_name} ({err_string})")
    return maybe_err


class DriverContext(object):
    context: cuda.CUcontext
    device: cuda.CUdevice
    
    def __init__(self, device_ordinal: int) -> None:
        self.context = None
        self.cleanup_stack: List[Callable[[], None]] = []
        check_cuda_error(cuda.cuInit(0))
        err, device = cuda.cuDeviceGet(device_ordinal)
        check_cuda_error(err)
        err, context = cuda.cuCtxCreate(0, device)
        check_cuda_error(err)
        self.device = device
        self.context = context
    
    def deferred(self, cleanup_fn: Callable[[], None]):
        self.cleanup_stack.append(cleanup_fn)
    
    def __enter__(self):
        return self
    
    def __exit__(self, ty, value, tb):
        for cleanup in reversed(self.cleanup_stack):
            cleanup()
        if self.context is not None:
            err = cuda.cuCtxDestroy(self.context)
            check_cuda_error(err)


class ExistingDriverContext(DriverContext):
    
    def __init__(self, device_ordinal: int) -> None:
        self.cleanup_stack: List[Callable[[], None]] = []
        err, device = cuda.cuDeviceGet(device_ordinal)
        check_cuda_error(err)
        err, context = cuda.cuCtxGetCurrent()
        check_cuda_error(err)
        self.device = device
        self.context = context
    
    def __exit__(self, ty, value, tb):
        for cleanup in reversed(self.cleanup_stack):
            cleanup()
        

def read_ptx(filename):
    with open(filename, "rb") as f:
        return f.read()


def ptx_link(driver_ctx: DriverContext, ptx_source_bytes: bytes, fn_name: bytes, verbose: int = 0) ->cuda.CUfunction:
    log_size = 16384
    info_log = bytearray(log_size)
    error_log = bytearray(log_size)
    # Using ctypes for walltime as it's an output parameter in C
    walltime = ctypes.c_float(0.0)

    # Option enums from cuda.driver.CUjit_option
    options = [
        cuda.CUjit_option.CU_JIT_WALL_TIME,
        cuda.CUjit_option.CU_JIT_INFO_LOG_BUFFER,
        cuda.CUjit_option.CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        cuda.CUjit_option.CU_JIT_ERROR_LOG_BUFFER,
        cuda.CUjit_option.CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        cuda.CUjit_option.CU_JIT_LOG_VERBOSE,
    ]

    option_vals = [
        ctypes.addressof(walltime), # Pointer to the float
        info_log,                  # Pass bytearray directly (check bindings doc)
        log_size,                  # Pass size as int
        error_log,                 # Pass bytearray directly
        log_size,                  # Pass size as int
        1                          # Verbose level as int
    ]

    err, state = cuda.cuLinkCreate(len(options), options, option_vals)
    check_cuda_error(err)
    driver_ctx.deferred(lambda: check_cuda_error(cuda.cuLinkDestroy(state)))

    err = cuda.cuLinkAddData(
        state,
        cuda.CUjitInputType.CU_JIT_INPUT_PTX,
        ptx_source_bytes,
        len(ptx_source_bytes),
        b"ptx_data",
        0, None, None
    )
    check_cuda_error(err, error_log)

    err, cubin_out, cubin_size_out = cuda.cuLinkComplete(state)
    check_cuda_error(err)

    if verbose >= 1:
        print(f"CUDA Link Completed in {walltime.value:.4f} ms.")
    if verbose >= 2:
        info_log_str = info_log.decode(errors='ignore').strip()
        if info_log_str:
            print(f"Linker Info Log:\n{info_log_str}")

    err, module = cuda.cuModuleLoadData(cubin_out) # Pass the cubin handle directly
    check_cuda_error(err)
    driver_ctx.deferred(lambda: check_cuda_error(cuda.cuModuleUnload(module)))

    err, func = cuda.cuModuleGetFunction(module, fn_name)
    check_cuda_error(err)
    return func


class CTypesWrapper:
    def __init__(self, c: ctypes._SimpleCData) -> None:
        self.c = c
    
    def getPtr(self) -> int:
        return ctypes.addressof(self.c)


def prepare_arg(arg: Union[int, float, ctypes._SimpleCData, ctypes.Structure, cuda.CUdeviceptr]):
    if isinstance(arg, int):
        return CTypesWrapper(ctypes.c_int(arg))
    if isinstance(arg, float):
        return CTypesWrapper(ctypes.c_float(arg))
    if isinstance(arg, ctypes._SimpleCData):
        return CTypesWrapper(arg)
    if isinstance(arg, ctypes.Structure):
        return CTypesWrapper(arg)
    if isinstance(arg, cuda.CUdeviceptr):
        return arg
    raise TypeError("Unrecognized argument type for kernel", type(arg))


def launch_kernel(
    kernel: Union[cuda.CUfunction, cuda.CUkernel],
    *args: Union[int, float, ctypes._SimpleCData, cuda.CUdeviceptr],
    grid_dim: Union[Tuple[int, int, int], int],
    block_dim: Union[Tuple[int, int, int], int],
    smem_bytes: int = 0,
    stream: cuda.CUstream = cuda.CUstream(0),
    sync: bool = False
):
    """
    | int args will be represented as native int in C.
    | float args will be represented as single-precision float in C.
    | CUdeviceptr will be passed as-is.
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
    err = cuda.cuLaunchKernel(kernel, *grid_dim, *block_dim, smem_bytes, stream, kernel_args, extra)
    check_cuda_error(err)
    if sync:
        check_cuda_error(cuda.cuStreamSynchronize(stream))


def main():
    with DriverContext(0) as driver_ctx:
        context = driver_ctx.context
        print(f"Created context: {context}")

        ptx_file = "test.ptx"
        kernel_name = b"geval"
        ptx_content = read_ptx(ptx_file)

        function = ptx_link(driver_ctx, ptx_content, kernel_name, verbose=2)

        launch_kernel(
            function, 42,
            grid_dim=2,
            block_dim=64,
            sync=True
        )


if __name__ == "__main__":
    main()
