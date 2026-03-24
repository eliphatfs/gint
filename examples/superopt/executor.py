"""Batch GPU execution for superoptimizer candidates."""

import struct
import ctypes
import numpy as np
import torch
import cuda.bindings.driver as cuda

from gint.host.executor import (
    BaseExecutableProgram, ProgramData, ProgramTensorInfo,
    TensorInterface, get_executor, _convert_arg,
)
from gint.host.cuda.executor_impl import _fill_tensor_info
from gint.host.cuda.driver import check_cuda_error, launch_kernel
from gint.kernel.interpreter.main import SMEM_PER_WARP
from gint.kernel.interpreter.structs import HTensorInfo
from gint.host.utils import cdiv

_TI_SIZE = ctypes.sizeof(HTensorInfo)


class RawProgram(BaseExecutableProgram):
    """A program built from a raw bytecode numpy array."""

    def __init__(self, bytecode_np, tensor_infos, input_indices=None):
        super().__init__()
        self._pd = ProgramData(bytecode_np, tensor_infos, input_indices)

    def get_program(self, *args, **kwargs):
        return self._pd

    def cache_policy(self, *args, **kwargs):
        return id(self)           # never cache — each instance is unique


def _make_tensor_info(test_size):
    """ProgramTensorInfo for a 1-D contiguous f32 tensor of *test_size* elements."""
    return ProgramTensorInfo(
        elm_size=4,
        batch_strides=[],
        batch_shape=[],
        block_shape_stride_1=[test_size, 1],
        block_shape_stride_2=[1, 0],
        block_grid_dims=[1, 1],
        block_grid_steps=[test_size, 0],
    )


def run_reference(bytecode_np, input_tensors, output_tensor, n_inputs):
    """Run a single program through the normal executor path."""
    test_size = output_tensor.shape[0]
    ti = _make_tensor_info(test_size)
    tis = [ti] * (n_inputs + 1)
    indices = list(range(n_inputs + 1))
    prog = RawProgram(bytecode_np, tis, indices)
    args = [_convert_arg(t) for t in input_tensors] + [_convert_arg(output_tensor)]
    get_executor().execute(prog, args, grid_dim=1)
    torch.cuda.synchronize()


class BatchRunner:
    """Run many short programs sharing the same tensor layout efficiently.

    All candidates must have:
      - the same number of input tensors (n_inputs)
      - the same program word count (prog_words)
      - the same output tensor size (test_size)

    Internally this does:
      1.  ONE cuMemAlloc + cuMemcpyHtoD for all bytecodes (concatenated)
      2.  ONE cuMemAlloc + cuMemcpyHtoD for all HTensorInfo structs
      3.  ONE cuMemAlloc + cuMemcpyHtoD for each pointer table (2 tables)
      4.  ONE kernel launch (indirect mode, 1 warp per candidate)
      5.  Explicit cuMemFree for everything (no deferred accumulation)
    """

    def __init__(self, n_inputs, test_size=128):
        self.n_inputs = n_inputs
        self.test_size = test_size
        self.n_tensors = n_inputs + 1     # inputs + 1 output

        # Kernel handle
        executor = get_executor()
        self._dctx, self._cufunc, self._conc, self._nsm = executor.geval_func_handle()
        self._heuristic = executor.heuristic_best_warps

        # Build template HTensorInfo (all tensor slots share the same layout)
        ti = _make_tensor_info(test_size)
        tis = [ti] * self.n_tensors
        template = HTensorInfo()
        _fill_tensor_info(template, tis)
        self._template_bytes = bytes(
            (ctypes.c_char * _TI_SIZE).from_address(ctypes.addressof(template))
        )
        # Byte offset of base_ptr[output_slot] within HTensorInfo
        self._output_ptr_off = n_inputs * 8          # base_ptr is int64 array at offset 0

    # ------------------------------------------------------------------

    def run(self, bytecodes_np, input_tensors, output_buffer):
        """Execute a batch of candidate programs.

        Parameters
        ----------
        bytecodes_np : numpy (N, prog_words) int32
            Complete programs (prefix + body + suffix).
        input_tensors : list[torch.Tensor]
            Shared input tensors (length = n_inputs).
        output_buffer : torch.Tensor, shape (>= N, test_size)
            Pre-allocated device buffer; rows 0..N-1 receive outputs.

        Returns
        -------
        torch.Tensor
            View ``output_buffer[:N]``.
        """
        N, prog_words = bytecodes_np.shape
        if N == 0:
            return output_buffer[:0]

        allocs = []        # track device pointers for cleanup
        try:
            # 1. Upload all bytecodes as one contiguous block ---------------
            flat = np.ascontiguousarray(bytecodes_np.reshape(-1))
            _, dcode = check_cuda_error(cuda.cuMemAlloc(flat.nbytes))
            allocs.append(dcode)
            check_cuda_error(cuda.cuMemcpyHtoD(dcode, flat, flat.nbytes))

            # 2. Build per-candidate tensor infos --------------------------
            ti_data = bytearray(self._template_bytes) * N
            out_base = output_buffer.data_ptr()
            out_stride = self.test_size * 4           # bytes per output row
            inp_ptrs = [t.data_ptr() for t in input_tensors]
            for i in range(N):
                off = i * _TI_SIZE
                # Patch input base pointers (shared for all candidates)
                for j, ptr in enumerate(inp_ptrs):
                    struct.pack_into("<q", ti_data, off + j * 8, ptr)
                # Patch output base pointer
                struct.pack_into(
                    "<q", ti_data,
                    off + self._output_ptr_off,
                    out_base + i * out_stride,
                )

            ti_np = np.frombuffer(ti_data, dtype=np.uint8)
            _, dinfos = check_cuda_error(cuda.cuMemAlloc(len(ti_data)))
            allocs.append(dinfos)
            check_cuda_error(cuda.cuMemcpyHtoD(dinfos, ti_np, len(ti_data)))

            # 3. Build pointer tables (int64 device addresses) -------------
            prog_bytes = prog_words * 4
            code_ptrs  = np.array(
                [int(dcode) + i * prog_bytes for i in range(N)], dtype=np.int64
            )
            tinfo_ptrs = np.array(
                [int(dinfos) + i * _TI_SIZE for i in range(N)], dtype=np.int64
            )

            _, code_dev = check_cuda_error(cuda.cuMemAlloc(code_ptrs.nbytes))
            allocs.append(code_dev)
            check_cuda_error(cuda.cuMemcpyHtoD(code_dev, code_ptrs, code_ptrs.nbytes))

            _, tinfo_dev = check_cuda_error(cuda.cuMemAlloc(tinfo_ptrs.nbytes))
            allocs.append(tinfo_dev)
            check_cuda_error(cuda.cuMemcpyHtoD(tinfo_dev, tinfo_ptrs, tinfo_ptrs.nbytes))

            # 4. Launch kernel (indirect mode, flag=1) ---------------------
            grid_dim = N
            best_warps = self._heuristic(grid_dim, self._conc, self._nsm)
            launch_kernel(
                self._cufunc,
                code_dev, tinfo_dev, self.n_tensors, grid_dim, 1,
                grid_dim=cdiv(grid_dim, best_warps),
                block_dim=(32, best_warps, 1),
                smem_bytes=SMEM_PER_WARP * best_warps,
                stream=cuda.CUstream(0),
            )
            torch.cuda.synchronize()

        finally:
            # 5. Cleanup ---------------------------------------------------
            for ptr in allocs:
                check_cuda_error(cuda.cuMemFree(ptr))

        return output_buffer[:N]
