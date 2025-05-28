import os
import lzma
import ctypes
import cuda.bindings.driver as cuda
from typing import Sequence
from ...kernel.interpreter.structs import HTensorInfo
from ..executor import BaseExecutor, BaseExecutableProgram, TensorInterface
from .driver import current_context, fatbin_load, launch_kernel, check_cuda_error
from ..utils import cdiv


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
                _, blocks = check_cuda_error(cuda.cuOccupancyMaxActiveBlocksPerMultiprocessor(cufunc, num_warps * 32, 0))
                concurrency = blocks * num_warps
                concurrencies.append((num_warps, concurrency))
            _, device = check_cuda_error(cuda.cuCtxGetDevice())
            _, num_sm = check_cuda_error(cuda.cuDeviceGetAttribute(cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device))
            self.func_cache[dctx] = dctx, cufunc, concurrencies, num_sm
        return self.func_cache[dctx]
    
    def heuristic_best_warps(self, num_blocks, concurrencies, num_sm):
        best_warps = 1
        best_time = 1e30
        for num_warps, concurrency in concurrencies:
            if num_blocks % num_warps == 0:
                launched_blocks = num_blocks // num_warps
                waves = cdiv(launched_blocks, concurrency * num_sm)
                this_time = waves * num_warps / (concurrency ** 0.5)
                if this_time < best_time:
                    best_warps = num_warps
                    best_time = this_time
        return best_warps

    def execute(self, program: BaseExecutableProgram, args: Sequence[TensorInterface], grid_dim: int, **extra_kwargs):
        dctx, cufunc, concurrencies, num_sm = self.geval_func_handle()
        
        if not hasattr(program, '_cu'):
            program._cu = {}
        
        pcu: dict = program._cu
        pcp = program.cache_policy(*args, **extra_kwargs)
        cacheline = pcu.get(pcp)
        
        if cacheline is None:
            pd = program.get_program(*args, **extra_kwargs)
            _, dcode = check_cuda_error(cuda.cuMemAlloc(len(pd.program) * 4))
            check_cuda_error(cuda.cuMemcpyHtoD(dcode, pd.program, len(pd.program) * 4))
            _, hinfo = check_cuda_error(cuda.cuMemAllocHost(ctypes.sizeof(HTensorInfo)))
            _, dinfo = check_cuda_error(cuda.cuMemAlloc(ctypes.sizeof(HTensorInfo)))
            dctx.deferred(lambda: check_cuda_error(cuda.cuMemFree(dcode)))
            dctx.deferred(lambda: check_cuda_error(cuda.cuMemFree(dinfo)))
            dctx.deferred(lambda: check_cuda_error(cuda.cuMemFreeHost(hinfo)))
            pcu[pcp] = cacheline = dcode, dinfo, HTensorInfo.from_address(int(hinfo)), len(pd.input_infos)
            ti = HTensorInfo.from_address(int(hinfo))
            
            for i, t in enumerate(pd.input_infos):
                rev_bs = t.block_strides[::-1]
                rev_bsz = t.block_sizes[::-1]
                rev_btofss = t.block_thread_offset_strides[::-1]
                for j in range(4):
                    if j < len(rev_bsz):
                        ti.b_stride[i][j] = rev_bs[j]
                        ti.b_size[i][j] = rev_bsz[j]
                        ti.bt_ofs_stride[i][j] = rev_btofss[j]
                    else:
                        ti.b_stride[i][j] = 0
                        ti.b_size[i][j] = 1
                        ti.bt_ofs_stride[i][j] = 0
                ti.t_stride[i] = t.thread_stride
                ti.t_size[i] = t.thread_size
                ti.elm_size[i] = t.elm_size
        
        dcode, dinfo, ti, nargs = cacheline
        if len(args) != nargs:
            raise ValueError("Tensor info should match arguments one-one")
        
        for i, t in enumerate(args):
            ti.base_ptr[i] = t.base_ptr
        check_cuda_error(cuda.cuMemcpyHtoDAsync(dinfo, ctypes.addressof(ti), ctypes.sizeof(ti), 0))
        best_warps = self.heuristic_best_warps(grid_dim, concurrencies, num_sm)
        launch_kernel(cufunc, dcode, dinfo, nargs, grid_dim=(grid_dim // best_warps, 1, 1), block_dim=(32, best_warps, 1))
