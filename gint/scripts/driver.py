from cuda import cuda
import sys
import numpy as np
import ctypes


def check_cuda_error(err):
    """Checks for CUDA errors and raises an exception if one occurs."""
    if isinstance(err, tuple) and isinstance(err[0], cuda.CUresult):
        err = err[0]
    if isinstance(err, cuda.CUresult):
        if err != cuda.CUresult.CUDA_SUCCESS:
            # [0] is CUresult
            err_name = cuda.cuGetErrorName(err)[1]
            err_string = cuda.cuGetErrorString(err)[1]
            raise CudaDriverError(f"CUDA Driver API Error: {err_name} ({err_string})")


class CudaDriverError(RuntimeError):
    """Custom exception for CUDA Driver API errors."""
    pass


def read_ptx(filename):
    with open(filename, "rb") as f:
        return f.read()


def ptx_link(ptx_source_bytes, fn_name):
    lState = None # Initialize to None for finally block

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

    try:
        err, lState = cuda.cuLinkCreate(len(options), options, option_vals)
        check_cuda_error(err)

        err = cuda.cuLinkAddData(
            lState,
            cuda.CUjitInputType.CU_JIT_INPUT_PTX,
            ptx_source_bytes,
            len(ptx_source_bytes),
            b"ptx_data",
            0, None, None
        )

        if err != cuda.CUresult.CUDA_SUCCESS:
            print(f"PTX Linker Error during cuLinkAddData:\n{error_log.decode(errors='ignore').strip()}", file=sys.stderr)
            check_cuda_error(err)

        err, cubin_out, cubin_size_out = cuda.cuLinkComplete(lState)
        check_cuda_error(err)

        print(f"CUDA Link Completed in {walltime.value:.4f} ms.")
        info_log_str = info_log.decode(errors='ignore').strip()
        if info_log_str:
             print(f"Linker Info Log:\n{info_log_str}")

        err, module = cuda.cuModuleLoadData(cubin_out) # Pass the cubin handle directly
        check_cuda_error(err)

        err, func = cuda.cuModuleGetFunction(module, fn_name)
        check_cuda_error(err)
        return module, func
    finally:
        if lState is not None:
            err = cuda.cuLinkDestroy(lState)
            check_cuda_error(err)


def main():
    context = None
    module = None

    try:
        check_cuda_error(cuda.cuInit(0))
        err, device = cuda.cuDeviceGet(0)
        check_cuda_error(err)
        err, context = cuda.cuCtxCreate(0, device)
        check_cuda_error(err)
        print(f"Created context: {context}")

        ptx_file = "test.ptx"
        kernel_name = b"geval"
        ptx_content = read_ptx(ptx_file)

        module, function = ptx_link(ptx_content, kernel_name)

        x = ctypes.c_int(42)

        kernel_args = (ctypes.c_void_p * 1)()
        kernel_args[0] = ctypes.addressof(x)

        # --- Launch Kernel ---
        gridDimX = 2
        gridDimY = 1
        gridDimZ = 1
        blockDimX = 64
        blockDimY = 1
        blockDimZ = 1
        sharedMemBytes = 0
        stream = cuda.CUstream(0)

        err = cuda.cuLaunchKernel(
            function,
            gridDimX, gridDimY, gridDimZ,
            blockDimX, blockDimY, blockDimZ,
            sharedMemBytes,
            stream,  # Stream handle
            kernel_args,  # Arguments (list of pointers/values)
            0
        )
        check_cuda_error(err)

        check_cuda_error(cuda.cuStreamSynchronize(stream))
    finally:
        if module is not None:
            err = cuda.cuModuleUnload(module)
            check_cuda_error(err)

        if context is not None:
            err = cuda.cuCtxDestroy(context)
            check_cuda_error(err)


if __name__ == "__main__":
    main()
