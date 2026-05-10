"""GPU-eager batch runner.

Same wire format as `executor.BatchRunner` but builds bytecodes, HTensorInfo
arrays, and the indirect pointer tables on the GPU via torch ops instead of
on the CPU via Python loops + struct.pack_into. Eliminates the per-candidate
Python loop and avoids cuMemAlloc/cuMemcpyHtoD/cuMemFree per batch (torch's
caching allocator handles the storage).

Inputs:
    - op_table: (n_ops, 2) int32 — column 0 opcode, column 1 operand.
    - indices:  (N, body_len) int32 — body is a sequence of op indices.

The prefix (load inputs) and suffix (store output + halt) are constant per
arity and are broadcast into the bytecode buffer.
"""

import ctypes
import numpy as np
import torch
import cuda.bindings.driver as cuda

from gint.host.executor import get_executor
from gint.host.cuda.executor_impl import _fill_tensor_info
from gint.host.cuda.driver import launch_kernel
from gint.kernel.interpreter.main import SMEM_PER_WARP
from gint.kernel.interpreter.structs import HTensorInfo
from gint.host.utils import cdiv

from .executor import _make_tensor_info, _TI_SIZE, SUPEROPT_VARIANT
from .candidates import make_prefix, make_suffix


class BatchRunnerGPU:
    """Batch runner that builds bytecodes/tinfos/pointer tables on GPU.

    Construction
    ------------
    n_inputs   : int          — number of input tensors per candidate.
    op_table   : (n_ops, 2) int32 numpy or list of (opcode, operand) — search ops.
    test_size  : int          — output tensor length per candidate (default 128).
    """

    def __init__(self, n_inputs, op_table, test_size=128):
        self.n_inputs = n_inputs
        self.test_size = test_size
        self.n_tensors = n_inputs + 1
        self.prefix_words = 2 * n_inputs
        self.suffix_words = 4

        # Op table on GPU: (n_ops, 2) int32, col0=opcode, col1=operand
        if isinstance(op_table, np.ndarray):
            op_arr = op_table.astype(np.int32, copy=False)
        else:
            op_arr = np.array(
                [(op.opcode, op.operand) for op in op_table], dtype=np.int32
            )
        self.op_table_gpu = torch.from_numpy(op_arr).contiguous().cuda()
        self.n_ops = self.op_table_gpu.shape[0]

        # Constant prefix/suffix words on GPU
        self.prefix_gpu = torch.tensor(
            make_prefix(n_inputs), dtype=torch.int32, device="cuda"
        )
        self.suffix_gpu = torch.tensor(
            make_suffix(n_inputs), dtype=torch.int32, device="cuda"
        )

        # Kernel handle
        executor = get_executor()
        self._dctx, self._cufunc, self._conc, self._nsm = executor.geval_func_handle(SUPEROPT_VARIANT)
        self._heuristic = executor.heuristic_best_warps

        # Build template HTensorInfo bytes (76 int64s, all tensor slots share layout)
        ti_template_struct = HTensorInfo()
        tis = [_make_tensor_info(test_size)] * self.n_tensors
        _fill_tensor_info(ti_template_struct, tis)
        template_bytes = bytes(
            (ctypes.c_char * _TI_SIZE).from_address(ctypes.addressof(ti_template_struct))
        )
        # As int64 view, length 76. base_ptr slots are the first n_tensors entries.
        assert _TI_SIZE % 8 == 0
        self._ti_words = _TI_SIZE // 8                     # 76
        self._template_i64 = torch.frombuffer(
            bytearray(template_bytes), dtype=torch.int64
        ).clone().cuda()                                   # (76,)

    # ------------------------------------------------------------------

    def run(self, indices, input_tensors, output_buffer):
        """Execute a batch of candidate programs, given body op-indices.

        Parameters
        ----------
        indices : (N, body_len) int32 tensor (cpu or cuda).
        input_tensors : list[torch.Tensor]   shared inputs.
        output_buffer : torch.Tensor (>=N, test_size) — receives outputs.

        Returns
        -------
        torch.Tensor view of output_buffer[:N].
        """
        if indices.numel() == 0:
            return output_buffer[:0]
        if not isinstance(indices, torch.Tensor):
            indices = torch.from_numpy(np.asarray(indices, dtype=np.int32))
        if indices.device.type != "cuda":
            indices = indices.to("cuda", non_blocking=True)
        if indices.dtype != torch.int32 and indices.dtype != torch.int64:
            indices = indices.to(torch.int32)

        N, body_len = indices.shape
        body_words = body_len * 2
        total_words = self.prefix_words + body_words + self.suffix_words

        # ---- 1. Bytecodes (N, total_words) int32 on GPU --------------
        bytecodes = torch.empty((N, total_words), dtype=torch.int32, device="cuda")
        if self.prefix_words:
            bytecodes[:, : self.prefix_words] = self.prefix_gpu
        # Body: gather (opcode, operand) pairs from op_table at indices, layout
        # body[i, 2*j]   = op_table[indices[i, j], 0]   (opcode)
        # body[i, 2*j+1] = op_table[indices[i, j], 1]   (operand)
        ops_at_idx = self.op_table_gpu[indices.long()]      # (N, body_len, 2) int32
        bytecodes[:, self.prefix_words : self.prefix_words + body_words] = (
            ops_at_idx.reshape(N, body_words)
        )
        bytecodes[:, -self.suffix_words :] = self.suffix_gpu

        # ---- 2. Tensor infos (N, 76) int64 on GPU --------------------
        # Start from template, repeat to (N, 76), patch input + output base_ptr.
        # Input ptrs are constant per-batch; we patch them once into the
        # template before the repeat to avoid an N-row write.
        template = self._template_i64.clone()               # (76,) int64
        if self.n_inputs:
            inp_ptr_t = torch.tensor(
                [t.data_ptr() for t in input_tensors],
                dtype=torch.int64, device="cuda",
            )
            template[: self.n_inputs] = inp_ptr_t
        tinfos = template.unsqueeze(0).repeat(N, 1)         # (N, 76) int64

        out_base = output_buffer.data_ptr()
        out_stride = self.test_size * 4                     # bytes
        out_ptrs = (
            torch.arange(N, dtype=torch.int64, device="cuda") * out_stride + out_base
        )
        tinfos[:, self.n_inputs] = out_ptrs

        # ---- 3. Pointer tables (int64 device addresses) --------------
        prog_bytes = total_words * 4
        bytecodes_base = bytecodes.data_ptr()
        tinfos_base = tinfos.data_ptr()
        code_ptrs = (
            torch.arange(N, dtype=torch.int64, device="cuda") * prog_bytes
            + bytecodes_base
        )
        tinfo_ptrs = (
            torch.arange(N, dtype=torch.int64, device="cuda") * _TI_SIZE
            + tinfos_base
        )

        # ---- 4. Launch (indirect mode, flag=1) -----------------------
        grid_dim = N
        best_warps = self._heuristic(grid_dim, self._conc, self._nsm)
        launch_kernel(
            self._cufunc,
            cuda.CUdeviceptr(int(code_ptrs.data_ptr())),
            cuda.CUdeviceptr(int(tinfo_ptrs.data_ptr())),
            self.n_tensors, grid_dim, 1,
            grid_dim=cdiv(grid_dim, best_warps),
            block_dim=(32, best_warps, 1),
            smem_bytes=SMEM_PER_WARP * best_warps,
            stream=cuda.CUstream(0),
        )

        # NB: bytecodes / tinfos / pointer tables must outlive the launch.
        # Since the next call enqueues onto the same stream they will be
        # serialized, but we sync before returning so callers can read
        # output_buffer immediately (matches BatchRunner.run semantics).
        torch.cuda.synchronize()
        return output_buffer[:N]
