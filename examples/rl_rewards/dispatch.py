"""Pre-built indirect-dispatch helper for the rl_rewards benchmark.

``CudaExecutor.execute_indirect`` is a one-shot API: every call retraces
each program's bytecode in Python, allocates a fresh device buffer for
every code/tinfo blob, uploads them, and only then launches.  That makes
it convenient but it folds dispatch setup into every call, which would
swamp the kernel time we actually want to measure.

This file factors that work into a ``PreparedIndirectDispatch`` object:

* the constructor traces once, allocates the device buffers, fills the
  per-warp pointer tables, and pre-bakes the ``(c_void_p * 5)`` argument
  array that ``cuLaunchKernel`` reads;
* ``launch()`` issues exactly one ``cuLaunchKernel`` — no allocations,
  no tracing, no host-to-device copies.

This is the pattern a real RL-style consumer would use: build the
dispatch once at episode start (or whenever programs/tensor metadata
change), then launch repeatedly each step.
"""
import ctypes
from typing import Sequence

import cuda.bindings.driver as cuda

from gint.host.cuda.executor_impl import CudaExecutor
from gint.host.cuda.driver import current_context, check_cuda_error
from gint.host.executor import (
    BaseExecutableProgram, _convert_arg, select_variant, _variant_max,
)
from gint.host.utils import cdiv, fill_tensor_info as _fill_tensor_info
from gint.kernel.interpreter.main import SMEM_PER_WARP
from gint.kernel.interpreter.structs import HTensorInfo


class PreparedIndirectDispatch:
    """Cached state for a single ``execute_indirect`` configuration.

    Mirrors the body of ``CudaExecutor.execute_indirect`` up to (but not
    including) the actual kernel launch.  Once built, ``launch()`` only
    issues a ``cuLaunchKernel`` call against the cached pointer tables
    and tensor-info buffers.
    """

    def __init__(
        self,
        executor: CudaExecutor,
        programs: Sequence[BaseExecutableProgram],
        args_list: Sequence[Sequence],
        indices: Sequence[int],
    ):
        assert len(programs) == len(args_list)
        self._executor = executor
        self._dctx = current_context()
        grid_dim = len(indices)
        self._grid_dim = grid_dim

        converted = [[_convert_arg(a) for a in args] for args in args_list]

        # --- per-program: trace bytecode once, upload code + tinfo
        dcode_list, dinfo_list, ntensors_list = [], [], []
        batch_variant = None
        for prog, args in zip(programs, converted):
            pd = prog.get_program(*args)
            if pd.input_indices is None:
                pd.input_indices = list(range(len(pd.input_infos)))
            v = select_variant(pd)
            batch_variant = v if batch_variant is None else _variant_max(batch_variant, v)

            _, dcode = check_cuda_error(cuda.cuMemAlloc(len(pd.program) * 4))
            check_cuda_error(cuda.cuMemcpyHtoD(dcode, pd.program, len(pd.program) * 4))
            self._dctx.deferred(lambda dc=dcode: check_cuda_error(cuda.cuMemFree(dc)))

            _, hinfo = check_cuda_error(cuda.cuMemAllocHost(ctypes.sizeof(HTensorInfo)))
            _, dinfo = check_cuda_error(cuda.cuMemAlloc(ctypes.sizeof(HTensorInfo)))
            self._dctx.deferred(lambda di=dinfo: check_cuda_error(cuda.cuMemFree(di)))
            self._dctx.deferred(lambda hi=hinfo: check_cuda_error(cuda.cuMemFreeHost(hi)))

            ti = HTensorInfo.from_address(int(hinfo))
            _fill_tensor_info(ti, pd.input_infos)
            for i, tidx in enumerate(pd.input_indices):
                ti.base_ptr[i] = args[tidx].base_ptr
            check_cuda_error(cuda.cuMemcpyHtoD(dinfo, ctypes.addressof(ti), ctypes.sizeof(ti)))

            dcode_list.append(dcode)
            dinfo_list.append(dinfo)
            ntensors_list.append(len(pd.input_infos))

        # --- per-warp pointer tables (the indirect-dispatch payload)
        code_ptrs = (ctypes.c_int64 * grid_dim)()
        tinfo_ptrs = (ctypes.c_int64 * grid_dim)()
        for i, pidx in enumerate(indices):
            code_ptrs[i] = int(dcode_list[pidx])
            tinfo_ptrs[i] = int(dinfo_list[pidx])

        _, code_ptrs_dev = check_cuda_error(cuda.cuMemAlloc(grid_dim * 8))
        _, tinfo_ptrs_dev = check_cuda_error(cuda.cuMemAlloc(grid_dim * 8))
        self._dctx.deferred(lambda: check_cuda_error(cuda.cuMemFree(code_ptrs_dev)))
        self._dctx.deferred(lambda: check_cuda_error(cuda.cuMemFree(tinfo_ptrs_dev)))
        check_cuda_error(cuda.cuMemcpyHtoD(code_ptrs_dev, code_ptrs, grid_dim * 8))
        check_cuda_error(cuda.cuMemcpyHtoD(tinfo_ptrs_dev, tinfo_ptrs, grid_dim * 8))

        # --- pre-baked launch parameters
        max_ntensors = max(ntensors_list)
        _, cufunc, concurrencies, num_sm = executor.geval_func_handle(batch_variant)
        best_warps = executor.heuristic_best_warps(grid_dim, concurrencies, num_sm)

        c_nargs = ctypes.c_int(max_ntensors)
        c_grid_dim = ctypes.c_int(grid_dim)
        c_flag = ctypes.c_int(1)   # indirect mode (>0)
        kernel_args = (ctypes.c_void_p * 5)()
        kernel_args[0] = code_ptrs_dev.getPtr()
        kernel_args[1] = tinfo_ptrs_dev.getPtr()
        kernel_args[2] = ctypes.addressof(c_nargs)
        kernel_args[3] = ctypes.addressof(c_grid_dim)
        kernel_args[4] = ctypes.addressof(c_flag)

        # The c_int wrappers and pointer tables must outlive every
        # launch — kernel_args holds raw addresses into them.
        self._keepalive = (code_ptrs_dev, tinfo_ptrs_dev, dcode_list,
                           dinfo_list, c_nargs, c_grid_dim, c_flag,
                           code_ptrs, tinfo_ptrs)
        self._kernel_args = kernel_args
        self._cufunc = cufunc
        self._gx = cdiv(grid_dim, best_warps)
        self._block_dim = (32, best_warps, 1)
        self._smem = SMEM_PER_WARP * best_warps

    def launch(self, cuda_stream: int = 0):
        """Issue exactly one cuLaunchKernel against the cached state."""
        custream = cuda.CUstream(cuda_stream)
        check_cuda_error(cuda.cuLaunchKernel(
            self._cufunc,
            self._gx, 1, 1,
            *self._block_dim,
            self._smem,
            custream,
            self._kernel_args,
            0,
        ))
