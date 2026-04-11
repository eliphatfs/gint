import os
import lzma
import zipfile
import ctypes
import numpy
from hip import hip
from typing import Sequence
from ...kernel.interpreter.main import SMEM_PER_WARP
from ...kernel.interpreter.structs import HTensorInfo
from ..executor import BaseExecutor, BaseExecutableProgram, TensorInterface, _convert_arg
from .driver import current_context, hipfb_load, launch_kernel, check_hip_error, get_gfx_name
from ..utils import cdiv, fill_tensor_info as _fill_tensor_info


# Map GFX names to fallback generic targets
_GFX_GENERIC_FALLBACK = {}
# RDNA3 + RDNA3.5 → gfx11-generic
for _i in range(1100, 1154):
    _GFX_GENERIC_FALLBACK[f"gfx{_i}"] = "gfx11-generic"
# RDNA4 → gfx12-generic
for _i in range(1200, 1202):
    _GFX_GENERIC_FALLBACK[f"gfx{_i}"] = "gfx12-generic"


class HipExecutor(BaseExecutor):

    def __init__(self) -> None:
        self.func_cache = {}
        self._zip_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gint_amdgcn.zip")

    def _load_hsaco_for_gfx(self, gfx: str) -> bytes:
        """Load the HSACO for the given GFX name from the zip archive."""
        with zipfile.ZipFile(self._zip_path, 'r') as zf:
            names = zf.namelist()
            # Try exact match first, then generic fallback
            for candidate in [gfx, _GFX_GENERIC_FALLBACK.get(gfx)]:
                if candidate is None:
                    continue
                entry = f"gint_{candidate}.hsaco.xz"
                if entry in names:
                    return lzma.decompress(zf.read(entry))
        raise RuntimeError(
            f"No HSACO found for {gfx} in {self._zip_path}. "
            f"Available: {names}"
        )

    def warp_size(self) -> int:
        return 32

    def geval_func_handle(self):
        dctx = current_context()
        if dctx not in self.func_cache:
            gfx = get_gfx_name(dctx.device)
            hsaco = self._load_hsaco_for_gfx(gfx)
            hipfunc = hipfb_load(dctx, hsaco, b'geval')
            concurrencies = []
            for num_warps in [1, 2, 4]:
                _, blocks = check_hip_error(hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(hipfunc, num_warps * 32, SMEM_PER_WARP * num_warps))
                concurrencies.append((num_warps, blocks))
            _, device = check_hip_error(hip.hipGetDevice())
            _, num_cu = check_hip_error(hip.hipDeviceGetAttribute(hip.hipDeviceAttribute_t.hipDeviceAttributeMultiprocessorCount, device))
            self.func_cache[dctx] = dctx, hipfunc, concurrencies, num_cu
        return self.func_cache[dctx]

    def heuristic_best_warps(self, num_blocks, concurrencies, num_cu):
        best_warps = 1
        best_time = 1e30
        for num_warps, concurrency in concurrencies:
            launched_blocks = cdiv(num_blocks, num_warps)
            waves = cdiv(launched_blocks, concurrency * num_cu)
            this_time = waves * (concurrency * num_warps) ** 0.5
            if this_time < best_time:
                best_warps = num_warps
                best_time = this_time
        return best_warps

    def execute(self, program: BaseExecutableProgram, args: Sequence[TensorInterface], grid_dim: int, cuda_stream: int = 0, **extra_kwargs):
        dctx, hipfunc, concurrencies, num_cu = self.geval_func_handle()

        if not hasattr(program, '_hip'):
            program._hip = {}

        phip: dict = program._hip
        pcp = program.cache_policy(*args, **extra_kwargs)
        cacheline = phip.get(pcp)

        if cacheline is None:
            pd = program.get_program(*args, **extra_kwargs)
            if pd.input_indices is None:
                pd.input_indices = list(range(len(pd.input_infos)))
            assert len(pd.input_indices) == len(pd.input_infos), "Input indices must match input infos"
            _, dcode = check_hip_error(hip.hipMalloc(len(pd.program) * 4))
            check_hip_error(hip.hipMemcpyHtoD(dcode, pd.program, len(pd.program) * 4))
            _, hinfo = check_hip_error(hip.hipHostMalloc(ctypes.sizeof(HTensorInfo), 0))
            _, dinfo = check_hip_error(hip.hipMalloc(ctypes.sizeof(HTensorInfo)))
            dctx.deferred(lambda: check_hip_error(hip.hipFree(dcode)))
            dctx.deferred(lambda: check_hip_error(hip.hipFree(dinfo)))
            dctx.deferred(lambda: check_hip_error(hip.hipHostFree(hinfo)))
            ti = HTensorInfo.from_address(int(hinfo))
            _fill_tensor_info(ti, pd.input_infos)
            phip[pcp] = cacheline = dcode, dinfo, ti, len(args), pd.input_indices

        dcode, dinfo, ti, nargs, indices = cacheline
        if len(args) != nargs:
            raise ValueError("Number of arguments don't match the program.")

        for i, tidx in enumerate(indices):
            ti.base_ptr[i] = args[tidx].base_ptr
        hipstream = hip.hipStream_t(cuda_stream)
        if check_hip_error(hip.hipStreamIsCapturing(hipstream))[1] == hip.hipStreamCaptureStatus.hipStreamCaptureStatusActive:
            check_hip_error(hip.hipMemcpyHtoD(dinfo, ctypes.addressof(ti), ctypes.sizeof(ti)))
        else:
            check_hip_error(hip.hipMemcpyHtoDAsync(dinfo, ctypes.addressof(ti), ctypes.sizeof(ti), hipstream))
        best_warps = self.heuristic_best_warps(grid_dim, concurrencies, num_cu)
        launch_kernel(hipfunc, dcode, dinfo, nargs, grid_dim, 0, grid_dim=cdiv(grid_dim, best_warps), block_dim=(32, best_warps, 1), smem_bytes=SMEM_PER_WARP * best_warps, stream=hipstream)

    def execute_indirect(
        self,
        programs: Sequence[BaseExecutableProgram],
        args_list: Sequence[Sequence],
        indices: Sequence[int],
        cuda_stream: int = 0,
    ):
        assert len(programs) == len(args_list), "programs and args_list must have the same length"
        grid_dim = len(indices)
        dctx, hipfunc, concurrencies, num_cu = self.geval_func_handle()

        converted_args_list = [[_convert_arg(a) for a in args] for args in args_list]
        dcode_list = []
        dinfo_list = []
        ntensors_list = []
        for prog, args in zip(programs, converted_args_list):
            pd = prog.get_program(*args)
            if pd.input_indices is None:
                pd.input_indices = list(range(len(pd.input_infos)))

            _, dcode = check_hip_error(hip.hipMalloc(len(pd.program) * 4))
            check_hip_error(hip.hipMemcpyHtoD(dcode, pd.program, len(pd.program) * 4))
            dctx.deferred(lambda dc=dcode: check_hip_error(hip.hipFree(dc)))

            _, hinfo = check_hip_error(hip.hipHostMalloc(ctypes.sizeof(HTensorInfo), 0))
            _, dinfo = check_hip_error(hip.hipMalloc(ctypes.sizeof(HTensorInfo)))
            dctx.deferred(lambda di=dinfo: check_hip_error(hip.hipFree(di)))
            dctx.deferred(lambda hi=hinfo: check_hip_error(hip.hipHostFree(hi)))
            ti = HTensorInfo.from_address(int(hinfo))
            _fill_tensor_info(ti, pd.input_infos)
            for i, tidx in enumerate(pd.input_indices):
                ti.base_ptr[i] = args[tidx].base_ptr
            check_hip_error(hip.hipMemcpyHtoD(dinfo, ctypes.addressof(ti), ctypes.sizeof(ti)))

            dcode_list.append(dcode)
            dinfo_list.append(dinfo)
            ntensors_list.append(len(pd.input_infos))

        code_ptrs = (ctypes.c_int64 * grid_dim)()
        tinfo_ptrs = (ctypes.c_int64 * grid_dim)()
        for i, pidx in enumerate(indices):
            code_ptrs[i] = int(dcode_list[pidx])
            tinfo_ptrs[i] = int(dinfo_list[pidx])

        _, code_ptrs_dev = check_hip_error(hip.hipMalloc(grid_dim * 8))
        _, tinfo_ptrs_dev = check_hip_error(hip.hipMalloc(grid_dim * 8))
        dctx.deferred(lambda: check_hip_error(hip.hipFree(code_ptrs_dev)))
        dctx.deferred(lambda: check_hip_error(hip.hipFree(tinfo_ptrs_dev)))
        check_hip_error(hip.hipMemcpyHtoD(code_ptrs_dev, code_ptrs, grid_dim * 8))
        check_hip_error(hip.hipMemcpyHtoD(tinfo_ptrs_dev, tinfo_ptrs, grid_dim * 8))

        max_ntensors = max(ntensors_list)
        best_warps = self.heuristic_best_warps(grid_dim, concurrencies, num_cu)
        launch_kernel(
            hipfunc, code_ptrs_dev, tinfo_ptrs_dev, max_ntensors, grid_dim, 1,
            grid_dim=cdiv(grid_dim, best_warps), block_dim=(32, best_warps, 1),
            smem_bytes=SMEM_PER_WARP * best_warps, stream=hip.hipStream_t(cuda_stream)
        )
