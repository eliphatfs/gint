import os
import lzma
import ctypes
import numpy
import cuda.bindings.driver as cuda
from typing import Optional, Sequence
from ...kernel.interpreter.main import SMEM_PER_WARP, VARIANTS, variant_kernel_name
from ...kernel.interpreter.structs import HTensorInfo
from ..executor import BaseExecutor, BaseExecutableProgram, TensorInterface, _convert_arg, select_variant, _variant_max
from .driver import current_context, fatbin_load, launch_kernel, check_cuda_error
from ..utils import cdiv, fill_tensor_info as _fill_tensor_info


class CudaExecutor(BaseExecutor):

    def __init__(self) -> None:
        self.func_cache = {}
        with lzma.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gint.fatbin.xz")) as fi:
            self.fatbin = fi.read()

    def warp_size(self) -> int:
        return 32

    def geval_func_handle(self, variant: str):
        dctx = current_context()
        key = (dctx, variant)
        if key not in self.func_cache:
            sym = variant_kernel_name(variant).encode()
            cufunc = fatbin_load(dctx, self.fatbin, sym)
            concurrencies = []
            for num_warps in [1, 2, 4]:
                _, blocks = check_cuda_error(cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(cufunc, num_warps * 32, SMEM_PER_WARP * num_warps))
                concurrencies.append((num_warps, blocks))
            _, device = check_cuda_error(cuda.cuCtxGetDevice())
            _, num_sm = check_cuda_error(cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device))
            self.func_cache[key] = dctx, cufunc, concurrencies, num_sm
        return self.func_cache[key]

    def heuristic_best_warps(self, num_blocks, concurrencies, num_sm):
        best_warps = 1
        best_time = 1e30
        for num_warps, concurrency in concurrencies:
            launched_blocks = cdiv(num_blocks, num_warps)
            waves = cdiv(launched_blocks, concurrency * num_sm)
            this_time = waves * (concurrency * num_warps) ** 0.5
            if this_time < best_time:
                best_warps = num_warps
                best_time = this_time
        return best_warps

    def execute(self, program: BaseExecutableProgram, args: Sequence[TensorInterface], grid_dim: int, cuda_stream: int = 0, **extra_kwargs):
        if not hasattr(program, '_cu'):
            program._cu = {}

        pcu: dict = program._cu
        pcp = program.cache_policy(*args, **extra_kwargs)
        cacheline = pcu.get(pcp)

        if cacheline is None:
            cacheline = pcu[pcp] = self._build_cacheline(
                program, args, grid_dim, **extra_kwargs)

        (dinfo, ti, indices, cufunc, kernel_args,
         gx, gy, gz, bx, by, bz, smem, _keepalive) = cacheline

        for i, tidx in enumerate(indices):
            ti.base_ptr[i] = args[tidx].base_ptr
        custream = cuda.CUstream(cuda_stream)
        if check_cuda_error(cuda.cuStreamIsCapturing(custream))[1] == cuda.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
            check_cuda_error(cuda.cuMemcpyHtoD(dinfo, ctypes.addressof(ti), ctypes.sizeof(ti)))
        else:
            check_cuda_error(cuda.cuMemcpyHtoDAsync(dinfo, ctypes.addressof(ti), ctypes.sizeof(ti), custream))
        check_cuda_error(cuda.cuLaunchKernel(
            cufunc, gx, gy, gz, bx, by, bz, smem, custream, kernel_args, 0))

    def _build_cacheline(self, program, args, grid_dim, **extra_kwargs):
        """Build a cacheline for one (program, shape) combination. All
        kernel args (dcode, dinfo, nargs, grid_dim, flag) and launch
        params (block dim, smem) are constants per cache key, so we
        pre-build the ``(c_void_p * 5)()`` array cuLaunchKernel needs and
        the c_int wrappers it points into. The runtime hot path then
        just rewrites tensor base ptrs and calls cuLaunchKernel directly,
        skipping the per-call ``prepare_arg`` + ``CTypesWrapper`` chain
        in ``launch_kernel``.

        Holds references to the c_int wrappers in ``_keepalive`` so the
        ctypes addresses inside ``kernel_args`` stay valid.
        """
        pd = program.get_program(*args, **extra_kwargs)
        if pd.input_indices is None:
            pd.input_indices = list(range(len(pd.input_infos)))
        assert len(pd.input_indices) == len(pd.input_infos), "Input indices must match input infos"
        if len(args) != len(pd.input_indices):
            raise ValueError("Number of arguments don't match the program.")

        variant = select_variant(pd)

        _, dcode = check_cuda_error(cuda.cuMemAlloc(len(pd.program) * 4))
        check_cuda_error(cuda.cuMemcpyHtoD(dcode, pd.program, len(pd.program) * 4))
        _, hinfo = check_cuda_error(cuda.cuMemAllocHost(ctypes.sizeof(HTensorInfo)))
        _, dinfo = check_cuda_error(cuda.cuMemAlloc(ctypes.sizeof(HTensorInfo)))
        dctx_local = current_context()
        dctx_local.deferred(lambda: check_cuda_error(cuda.cuMemFree(dcode)))
        dctx_local.deferred(lambda: check_cuda_error(cuda.cuMemFree(dinfo)))
        dctx_local.deferred(lambda: check_cuda_error(cuda.cuMemFreeHost(hinfo)))

        ti = HTensorInfo.from_address(int(hinfo))
        _fill_tensor_info(ti, pd.input_infos)

        _, cufunc, concurrencies, num_sm = self.geval_func_handle(variant)

        # Pre-build kernel_args. Layout matches the geval(...) signature:
        # (i32* code, TensorInfo* tinfo, i32 num_tensors, i32 grid_dim, i32 flag)
        c_nargs = ctypes.c_int(len(args))
        c_grid_dim = ctypes.c_int(grid_dim)
        c_flag = ctypes.c_int(0)
        kernel_args = (ctypes.c_void_p * 5)()
        kernel_args[0] = dcode.getPtr()
        kernel_args[1] = dinfo.getPtr()
        kernel_args[2] = ctypes.addressof(c_nargs)
        kernel_args[3] = ctypes.addressof(c_grid_dim)
        kernel_args[4] = ctypes.addressof(c_flag)
        # Keep dcode/dinfo + the c_int wrappers alive — kernel_args holds
        # raw addresses into them; if any go out of scope cuLaunchKernel
        # will read freed memory.
        keepalive = (dcode, dinfo, c_nargs, c_grid_dim, c_flag)

        best_warps = self.heuristic_best_warps(grid_dim, concurrencies, num_sm)
        gx = cdiv(grid_dim, best_warps)
        smem = SMEM_PER_WARP * best_warps

        return (dinfo, ti, pd.input_indices, cufunc, kernel_args,
                gx, 1, 1, 32, best_warps, 1, smem, keepalive)

    def execute_indirect(
        self,
        programs: Sequence[BaseExecutableProgram],
        args_list: Sequence[Sequence],
        indices: Sequence[int],
        cuda_stream: int = 0,
    ):
        assert len(programs) == len(args_list), "programs and args_list must have the same length"
        grid_dim = len(indices)
        dctx = current_context()

        # Convert args and build per-program device resources
        converted_args_list = [[_convert_arg(a) for a in args] for args in args_list]
        dcode_list = []
        dinfo_list = []
        ntensors_list = []
        batch_variant: Optional[str] = None
        for prog, args in zip(programs, converted_args_list):
            pd = prog.get_program(*args)
            if pd.input_indices is None:
                pd.input_indices = list(range(len(pd.input_infos)))
            v = select_variant(pd)
            batch_variant = v if batch_variant is None else _variant_max(batch_variant, v)

            # Upload bytecode
            _, dcode = check_cuda_error(cuda.cuMemAlloc(len(pd.program) * 4))
            check_cuda_error(cuda.cuMemcpyHtoD(dcode, pd.program, len(pd.program) * 4))
            dctx.deferred(lambda dc=dcode: check_cuda_error(cuda.cuMemFree(dc)))

            # Allocate and fill tensor info
            _, hinfo = check_cuda_error(cuda.cuMemAllocHost(ctypes.sizeof(HTensorInfo)))
            _, dinfo = check_cuda_error(cuda.cuMemAlloc(ctypes.sizeof(HTensorInfo)))
            dctx.deferred(lambda di=dinfo: check_cuda_error(cuda.cuMemFree(di)))
            dctx.deferred(lambda hi=hinfo: check_cuda_error(cuda.cuMemFreeHost(hi)))
            ti = HTensorInfo.from_address(int(hinfo))
            _fill_tensor_info(ti, pd.input_infos)
            for i, tidx in enumerate(pd.input_indices):
                ti.base_ptr[i] = args[tidx].base_ptr
            check_cuda_error(cuda.cuMemcpyHtoD(dinfo, ctypes.addressof(ti), ctypes.sizeof(ti)))

            dcode_list.append(dcode)
            dinfo_list.append(dinfo)
            ntensors_list.append(len(pd.input_infos))

        _, cufunc, concurrencies, num_sm = self.geval_func_handle(batch_variant)

        # Build per-warp pointer tables
        code_ptrs = (ctypes.c_int64 * grid_dim)()
        tinfo_ptrs = (ctypes.c_int64 * grid_dim)()
        for i, pidx in enumerate(indices):
            code_ptrs[i] = int(dcode_list[pidx])
            tinfo_ptrs[i] = int(dinfo_list[pidx])

        _, code_ptrs_dev = check_cuda_error(cuda.cuMemAlloc(grid_dim * 8))
        _, tinfo_ptrs_dev = check_cuda_error(cuda.cuMemAlloc(grid_dim * 8))
        dctx.deferred(lambda: check_cuda_error(cuda.cuMemFree(code_ptrs_dev)))
        dctx.deferred(lambda: check_cuda_error(cuda.cuMemFree(tinfo_ptrs_dev)))
        check_cuda_error(cuda.cuMemcpyHtoD(code_ptrs_dev, code_ptrs, grid_dim * 8))
        check_cuda_error(cuda.cuMemcpyHtoD(tinfo_ptrs_dev, tinfo_ptrs, grid_dim * 8))

        max_ntensors = max(ntensors_list)
        best_warps = self.heuristic_best_warps(grid_dim, concurrencies, num_sm)
        launch_kernel(
            cufunc, code_ptrs_dev, tinfo_ptrs_dev, max_ntensors, grid_dim, 1,
            grid_dim=cdiv(grid_dim, best_warps), block_dim=(32, best_warps, 1),
            smem_bytes=SMEM_PER_WARP * best_warps, stream=cuda.CUstream(cuda_stream)
        )
