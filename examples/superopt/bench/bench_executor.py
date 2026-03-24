"""Experiment 2: Per-batch GPU executor breakdown.

Instruments the BatchRunner.run() method to measure time spent in each
sub-phase of a single batch execution:
  - build_ti:    Building per-candidate HTensorInfo (struct.pack_into loop)
  - build_ptrs:  Creating pointer arrays (numpy)
  - alloc_code:  cuMemAlloc + cuMemcpyHtoD for bytecodes
  - copy_ti:     cuMemAlloc + cuMemcpyHtoD for tensor infos
  - copy_ptrs:   cuMemAlloc + cuMemcpyHtoD for pointer tables
  - launch:      Kernel launch + synchronize
  - free:        cuMemFree for all allocations

Target: relu (small target, focus on executor overhead per batch)
"""

import time
import struct
import ctypes
import json
import numpy as np
import torch
import cuda.bindings.driver as cuda

from ..opcodes import build_search_ops
from ..candidates import enumerate_exact_length, sequences_to_bytecodes, make_reference_bytecode
from ..executor import BatchRunner, run_reference, _make_tensor_info, _TI_SIZE
from ..targets import get_target
from gint.host.executor import get_executor, _convert_arg
from gint.host.cuda.executor_impl import _fill_tensor_info
from gint.host.cuda.driver import check_cuda_error, launch_kernel
from gint.kernel.interpreter.main import SMEM_PER_WARP
from gint.kernel.interpreter.structs import HTensorInfo
from gint.host.utils import cdiv

TEST_SIZE = 128
BATCH_SIZE = 4096


def _make_test_inputs(arity, seed=42):
    torch.manual_seed(seed)
    inputs = []
    for _ in range(arity):
        x = torch.empty(TEST_SIZE, device="cuda", dtype=torch.float32)
        x[:100] = torch.randn(100, device="cuda")
        x[100:108] = torch.tensor(
            [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 100.0], device="cuda")
        x[108:116] = torch.tensor(
            [1e-6, -1e-6, 1e6, -1e6, 0.1, -0.1, 3.14, -3.14], device="cuda")
        x[116:] = torch.randn(TEST_SIZE - 116, device="cuda") * 10
        inputs.append(x)
    return inputs


def _instrumented_run(runner, bytecodes_np, input_tensors, output_buffer):
    """BatchRunner.run() with per-phase timing instrumentation."""
    N, prog_words = bytecodes_np.shape
    if N == 0:
        return output_buffer[:0], {}

    timings = {}
    allocs = []

    try:
        # 1. Upload bytecodes
        t0 = time.perf_counter()
        flat = np.ascontiguousarray(bytecodes_np.reshape(-1))
        _, dcode = check_cuda_error(cuda.cuMemAlloc(flat.nbytes))
        allocs.append(dcode)
        check_cuda_error(cuda.cuMemcpyHtoD(dcode, flat, flat.nbytes))
        timings["alloc_code_ms"] = (time.perf_counter() - t0) * 1000

        # 2. Build per-candidate tensor infos (the struct.pack_into loop)
        t0 = time.perf_counter()
        ti_data = bytearray(runner._template_bytes) * N
        out_base = output_buffer.data_ptr()
        out_stride = runner.test_size * 4
        inp_ptrs = [t.data_ptr() for t in input_tensors]
        for i in range(N):
            off = i * _TI_SIZE
            for j, ptr in enumerate(inp_ptrs):
                struct.pack_into("<q", ti_data, off + j * 8, ptr)
            struct.pack_into(
                "<q", ti_data, off + runner._output_ptr_off,
                out_base + i * out_stride)
        timings["build_ti_ms"] = (time.perf_counter() - t0) * 1000

        # 3. Upload tensor infos
        t0 = time.perf_counter()
        ti_np = np.frombuffer(ti_data, dtype=np.uint8)
        _, dinfos = check_cuda_error(cuda.cuMemAlloc(len(ti_data)))
        allocs.append(dinfos)
        check_cuda_error(cuda.cuMemcpyHtoD(dinfos, ti_np, len(ti_data)))
        timings["copy_ti_ms"] = (time.perf_counter() - t0) * 1000

        # 4. Build pointer tables
        t0 = time.perf_counter()
        prog_bytes = prog_words * 4
        code_ptrs = np.array(
            [int(dcode) + i * prog_bytes for i in range(N)], dtype=np.int64)
        tinfo_ptrs = np.array(
            [int(dinfos) + i * _TI_SIZE for i in range(N)], dtype=np.int64)
        timings["build_ptrs_ms"] = (time.perf_counter() - t0) * 1000

        # 5. Upload pointer tables
        t0 = time.perf_counter()
        _, code_dev = check_cuda_error(cuda.cuMemAlloc(code_ptrs.nbytes))
        allocs.append(code_dev)
        check_cuda_error(cuda.cuMemcpyHtoD(code_dev, code_ptrs, code_ptrs.nbytes))
        _, tinfo_dev = check_cuda_error(cuda.cuMemAlloc(tinfo_ptrs.nbytes))
        allocs.append(tinfo_dev)
        check_cuda_error(cuda.cuMemcpyHtoD(tinfo_dev, tinfo_ptrs, tinfo_ptrs.nbytes))
        timings["copy_ptrs_ms"] = (time.perf_counter() - t0) * 1000

        # 6. Launch kernel
        t0 = time.perf_counter()
        grid_dim = N
        best_warps = runner._heuristic(grid_dim, runner._conc, runner._nsm)
        launch_kernel(
            runner._cufunc,
            code_dev, tinfo_dev, runner.n_tensors, grid_dim, 1,
            grid_dim=cdiv(grid_dim, best_warps),
            block_dim=(32, best_warps, 1),
            smem_bytes=SMEM_PER_WARP * best_warps,
            stream=cuda.CUstream(0),
        )
        torch.cuda.synchronize()
        timings["launch_ms"] = (time.perf_counter() - t0) * 1000

    finally:
        # 7. Cleanup
        t0 = time.perf_counter()
        for ptr in allocs:
            check_cuda_error(cuda.cuMemFree(ptr))
        timings["free_ms"] = (time.perf_counter() - t0) * 1000

    timings["total_ms"] = sum(timings.values())
    return output_buffer[:N], timings


def run(target_name="relu", n_batches=20, verbose=True):
    """Run executor-level profiling. Returns dict of results."""
    target = get_target(target_name)
    arity = target["arity"]
    body = target["body"]

    ops = build_search_ops(include_transcendental=False)
    # Generate enough candidates for multiple batches
    search_len = max(3, len(body))
    seqs = list(enumerate_exact_length(ops, arity, 1, search_len))

    if len(seqs) < BATCH_SIZE:
        # Pad with duplicates if needed
        while len(seqs) < BATCH_SIZE:
            seqs.extend(seqs[:BATCH_SIZE - len(seqs)])

    bytecodes = sequences_to_bytecodes(seqs[:BATCH_SIZE], arity)
    inputs = _make_test_inputs(arity)
    runner = BatchRunner(arity, TEST_SIZE)

    # Warmup
    out_buf = torch.empty(BATCH_SIZE, TEST_SIZE, device="cuda", dtype=torch.float32)
    runner.run(bytecodes, inputs, out_buf)

    # Collect timings over n_batches
    all_timings = []
    for _ in range(n_batches):
        out_buf = torch.empty(BATCH_SIZE, TEST_SIZE, device="cuda", dtype=torch.float32)
        _, timings = _instrumented_run(runner, bytecodes, inputs, out_buf)
        all_timings.append(timings)

    # Average
    keys = all_timings[0].keys()
    avg = {}
    for k in keys:
        vals = [t[k] for t in all_timings]
        avg[k] = round(np.mean(vals), 3)

    results = {
        "experiment": "executor_breakdown",
        "target": target_name,
        "batch_size": BATCH_SIZE,
        "n_batches": n_batches,
        "avg_timings_ms": avg,
        "throughput_candidates_per_s": round(BATCH_SIZE / (avg["total_ms"] / 1000)),
    }

    if verbose:
        total = avg["total_ms"]
        print(f"Executor breakdown (batch_size={BATCH_SIZE}, avg over {n_batches} runs)")
        print()
        for k, v in avg.items():
            if k == "total_ms":
                continue
            pct = 100 * v / total if total > 0 else 0
            label = k.replace("_ms", "")
            print(f"  {label:16s} {v:8.3f} ms  ({pct:5.1f}%)")
        print(f"  {'TOTAL':16s} {total:8.3f} ms")
        print(f"  Throughput: {results['throughput_candidates_per_s']:,} candidates/s")

    return results


if __name__ == "__main__":
    run()
