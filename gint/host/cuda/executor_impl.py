import os
import lzma
import ctypes
import numpy
import cuda.bindings.driver as cuda
from typing import Sequence
from ...kernel.interpreter.main import SMEM_PER_WARP
from ...kernel.interpreter.structs import HTensorInfo
from ..executor import BaseExecutor, BaseExecutableProgram, TensorInterface, _convert_arg
from .driver import current_context, fatbin_load, launch_kernel, check_cuda_error
from ..utils import cdiv


def _fill_tensor_info(ti: HTensorInfo, input_infos, slot_offset: int = 0):
    for i, t in enumerate(input_infos):
        si = slot_offset + i
        ti.elm_size[si] = t.elm_size
        rev_bs = t.batch_strides[::-1]
        rev_bsz = t.batch_shape[::-1]
        assert len(rev_bs) == len(rev_bsz) <= 4, "At most 4 block axes supported!"
        for j in range(4):
            if j < len(rev_bsz):
                ti.batch_strides[si][j] = rev_bs[j]
                ti.batch_shape[si][j] = rev_bsz[j]
            else:
                ti.batch_strides[si][j] = 0
                ti.batch_shape[si][j] = 1
        ti.block_shape_stride_1[si][0] = t.block_shape_stride_1[0]
        ti.block_shape_stride_1[si][1] = t.block_shape_stride_1[1]
        ti.block_shape_stride_2[si][0] = t.block_shape_stride_2[0]
        ti.block_shape_stride_2[si][1] = t.block_shape_stride_2[1]
        ti.block_grid_dims[si][0] = t.block_grid_dims[0]
        ti.block_grid_dims[si][1] = t.block_grid_dims[1]
        ti.block_grid_steps[si][0] = t.block_grid_steps[0]
        ti.block_grid_steps[si][1] = t.block_grid_steps[1]


class CudaExecutor(BaseExecutor):

    def __init__(self) -> None:
        self.func_cache = {}
        with lzma.open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "gint.fatbin.xz")) as fi:
            self.fatbin = fi.read()

    def warp_size(self) -> int:
        return 32

    def geval_func_handle(self):
        dctx = current_context()
        if dctx not in self.func_cache:
            cufunc = fatbin_load(dctx, self.fatbin, b'geval')
            concurrencies = []
            for num_warps in [1, 2, 4]:
                _, blocks = check_cuda_error(cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(cufunc, num_warps * 32, SMEM_PER_WARP * num_warps))
                concurrencies.append((num_warps, blocks))
            _, device = check_cuda_error(cuda.cuCtxGetDevice())
            _, num_sm = check_cuda_error(cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device))
            self.func_cache[dctx] = dctx, cufunc, concurrencies, num_sm
        return self.func_cache[dctx]

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
        dctx, cufunc, concurrencies, num_sm = self.geval_func_handle()

        if not hasattr(program, '_cu'):
            program._cu = {}

        pcu: dict = program._cu
        pcp = program.cache_policy(*args, **extra_kwargs)
        cacheline = pcu.get(pcp)

        if cacheline is None:
            pd = program.get_program(*args, **extra_kwargs)
            if pd.input_indices is None:
                pd.input_indices = list(range(len(pd.input_infos)))
            assert len(pd.input_indices) == len(pd.input_infos), "Input indices must match input infos"
            _, dcode = check_cuda_error(cuda.cuMemAlloc(len(pd.program) * 4))
            check_cuda_error(cuda.cuMemcpyHtoD(dcode, pd.program, len(pd.program) * 4))
            _, hinfo = check_cuda_error(cuda.cuMemAllocHost(ctypes.sizeof(HTensorInfo)))
            _, dinfo = check_cuda_error(cuda.cuMemAlloc(ctypes.sizeof(HTensorInfo)))
            dctx.deferred(lambda: check_cuda_error(cuda.cuMemFree(dcode)))
            dctx.deferred(lambda: check_cuda_error(cuda.cuMemFree(dinfo)))
            dctx.deferred(lambda: check_cuda_error(cuda.cuMemFreeHost(hinfo)))
            pcu[pcp] = cacheline = dcode, dinfo, HTensorInfo.from_address(int(hinfo)), len(args), pd.input_indices
            ti = HTensorInfo.from_address(int(hinfo))
            _fill_tensor_info(ti, pd.input_infos)

        dcode, dinfo, ti, nargs, indices = cacheline
        if len(args) != nargs:
            raise ValueError("Number of arguments don't match the program.")

        for i, tidx in enumerate(indices):
            ti.base_ptr[i] = args[tidx].base_ptr
        custream = cuda.CUstream(cuda_stream)
        if check_cuda_error(cuda.cuStreamIsCapturing(custream))[1] == cuda.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE:
            check_cuda_error(cuda.cuMemcpyHtoD(dinfo, ctypes.addressof(ti), ctypes.sizeof(ti)))
        else:
            check_cuda_error(cuda.cuMemcpyHtoDAsync(dinfo, ctypes.addressof(ti), ctypes.sizeof(ti), custream))
        best_warps = self.heuristic_best_warps(grid_dim, concurrencies, num_sm)
        launch_kernel(cufunc, dcode, dinfo, nargs, grid_dim, 0, grid_dim=cdiv(grid_dim, best_warps), block_dim=(32, best_warps, 1), smem_bytes=SMEM_PER_WARP * best_warps, stream=custream)

    def execute_indirect(
        self,
        programs: Sequence[BaseExecutableProgram],
        args_list: Sequence[Sequence],
        indices: Sequence[int],
        cuda_stream: int = 0,
    ):
        assert len(programs) == len(args_list), "programs and args_list must have the same length"
        grid_dim = len(indices)
        dctx, cufunc, concurrencies, num_sm = self.geval_func_handle()

        # Convert args and build per-program device resources
        converted_args_list = [[_convert_arg(a) for a in args] for args in args_list]
        dcode_list = []
        dinfo_list = []
        ntensors_list = []
        for prog, args in zip(programs, converted_args_list):
            pd = prog.get_program(*args)
            if pd.input_indices is None:
                pd.input_indices = list(range(len(pd.input_infos)))

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
