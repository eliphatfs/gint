"""A/B benchmark: CPU bytecode/tinfo build vs torch-eager GPU build.

Measures four configurations on the same workload:

  cpu_baseline          enumerate (yields SearchOp lists)
                        + sequences_to_bytecodes (numpy)
                        + BatchRunner.run         (per-cand struct.pack_into)

  gpu_runner_only       enumerate (yields SearchOp lists)
                        + seqs->indices conversion
                        + BatchRunnerGPU.run      (torch-eager build)

  gpu_full              enumerate_exact_length_indices (emits int32 array)
                        + BatchRunnerGPU.run

Runs gelu length 5 (~14M sequences) by default. Reports per-phase wall
time, total time, and per-batch executor time.
"""

import time
import argparse
import numpy as np
import torch

from ..opcodes import build_search_ops, OP_FEXP, OP_FERF
from ..candidates import (
    enumerate_exact_length, enumerate_exact_length_indices,
    sequences_to_bytecodes, make_reference_bytecode,
)
from ..executor import BatchRunner, run_reference
from ..executor_torch import BatchRunnerGPU
from ..targets import get_target

TEST_SIZE = 128
BATCH_SIZE = 4096


def _needs_transcendental(body):
    trans_ops = {OP_FEXP, 49, 50, 51, 52, 40, 41, 42, 53, OP_FERF}
    return any(op in trans_ops for op, _ in body)


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


def _compare_outputs(out, ref, atol=1e-6, rtol=1e-5):
    ref = ref.unsqueeze(0)
    diff = (out - ref).abs()
    tol = atol + rtol * ref.abs()
    both_nan = out.isnan() & ref.isnan()
    return ((diff <= tol) | both_nan).all(dim=1)


def _setup(target_name):
    target = get_target(target_name)
    arity = target["arity"]
    body = target["body"]
    imm_vals = {0.0, 0.5, 1.0, 2.0, -1.0}
    from ..opcodes import OP_FPUSH, OP_FADDIMM, OP_FMULIMM
    for op, operand in body:
        if op in (OP_FPUSH, OP_FADDIMM, OP_FMULIMM) and operand != 0:
            val = float(np.array(operand, dtype=np.int32).view(np.float32))
            imm_vals.add(val)
    ops = build_search_ops(
        include_transcendental=_needs_transcendental(body),
        immediate_values=sorted(imm_vals),
    )
    inputs = _make_test_inputs(arity)
    ref_out = torch.empty(TEST_SIZE, device="cuda", dtype=torch.float32)
    run_reference(make_reference_bytecode(body, arity), inputs, ref_out, arity)
    return target, arity, body, ops, inputs, ref_out


def run_cpu_baseline(arity, ops, inputs, ref_out, length):
    """Original pipeline: SearchOp lists -> numpy bytecodes -> BatchRunner."""
    runner = BatchRunner(arity, TEST_SIZE)
    out_buf = torch.empty(BATCH_SIZE, TEST_SIZE, device="cuda", dtype=torch.float32)

    t_enum = t_build = t_run = t_cmp = 0.0
    n_total = n_match = 0
    t0 = time.perf_counter()
    seqs = list(enumerate_exact_length(ops, arity, 1, length))
    t_enum = time.perf_counter() - t0
    n_total = len(seqs)

    for start in range(0, n_total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_total)
        batch = seqs[start:end]
        if len(batch) < BATCH_SIZE:
            out_buf = torch.empty(len(batch), TEST_SIZE, device="cuda", dtype=torch.float32)

        t1 = time.perf_counter()
        bc = sequences_to_bytecodes(batch, arity)
        t_build += time.perf_counter() - t1

        t2 = time.perf_counter()
        runner.run(bc, inputs, out_buf)
        t_run += time.perf_counter() - t2

        t3 = time.perf_counter()
        n_match += int(_compare_outputs(out_buf[:len(batch)], ref_out).sum().item())
        t_cmp += time.perf_counter() - t3

    return dict(
        config="cpu_baseline", n_total=n_total, n_match=n_match,
        enum_s=t_enum, build_s=t_build, run_s=t_run, cmp_s=t_cmp,
        total_s=t_enum + t_build + t_run + t_cmp,
    )


def run_gpu_runner_only(arity, ops, inputs, ref_out, length):
    """Hybrid: SearchOp-list enum (Python) -> indices via dict -> BatchRunnerGPU."""
    op_to_idx = {(op.opcode, op.operand): i for i, op in enumerate(ops)}
    runner = BatchRunnerGPU(arity, ops, TEST_SIZE)
    out_buf = torch.empty(BATCH_SIZE, TEST_SIZE, device="cuda", dtype=torch.float32)

    t_enum = t_build = t_run = t_cmp = 0.0
    n_total = n_match = 0
    t0 = time.perf_counter()
    seqs = list(enumerate_exact_length(ops, arity, 1, length))
    t_enum = time.perf_counter() - t0
    n_total = len(seqs)

    for start in range(0, n_total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_total)
        batch = seqs[start:end]
        if len(batch) < BATCH_SIZE:
            out_buf = torch.empty(len(batch), TEST_SIZE, device="cuda", dtype=torch.float32)

        t1 = time.perf_counter()
        idx_np = np.empty((len(batch), length), dtype=np.int32)
        for i, seq in enumerate(batch):
            for j, op in enumerate(seq):
                idx_np[i, j] = op_to_idx[(op.opcode, op.operand)]
        idx_t = torch.from_numpy(idx_np).cuda()
        t_build += time.perf_counter() - t1

        t2 = time.perf_counter()
        runner.run(idx_t, inputs, out_buf)
        t_run += time.perf_counter() - t2

        t3 = time.perf_counter()
        n_match += int(_compare_outputs(out_buf[:len(batch)], ref_out).sum().item())
        t_cmp += time.perf_counter() - t3

    return dict(
        config="gpu_runner_only", n_total=n_total, n_match=n_match,
        enum_s=t_enum, build_s=t_build, run_s=t_run, cmp_s=t_cmp,
        total_s=t_enum + t_build + t_run + t_cmp,
    )


def run_gpu_full(arity, ops, inputs, ref_out, length):
    """All-in: indices-emitting enum -> BatchRunnerGPU."""
    runner = BatchRunnerGPU(arity, ops, TEST_SIZE)
    out_buf = torch.empty(BATCH_SIZE, TEST_SIZE, device="cuda", dtype=torch.float32)

    t_enum = t_build = t_run = t_cmp = 0.0
    n_total = n_match = 0
    t0 = time.perf_counter()
    idx_np = enumerate_exact_length_indices(ops, arity, 1, length)
    t_enum = time.perf_counter() - t0
    n_total = len(idx_np)

    for start in range(0, n_total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_total)
        batch_idx = idx_np[start:end]
        if len(batch_idx) < BATCH_SIZE:
            out_buf = torch.empty(len(batch_idx), TEST_SIZE, device="cuda", dtype=torch.float32)

        t1 = time.perf_counter()
        idx_t = torch.from_numpy(batch_idx).cuda()
        t_build += time.perf_counter() - t1

        t2 = time.perf_counter()
        runner.run(idx_t, inputs, out_buf)
        t_run += time.perf_counter() - t2

        t3 = time.perf_counter()
        n_match += int(_compare_outputs(out_buf[:len(batch_idx)], ref_out).sum().item())
        t_cmp += time.perf_counter() - t3

    return dict(
        config="gpu_full", n_total=n_total, n_match=n_match,
        enum_s=t_enum, build_s=t_build, run_s=t_run, cmp_s=t_cmp,
        total_s=t_enum + t_build + t_run + t_cmp,
    )


def _print_row(r):
    print(f"  {r['config']:18s}  "
          f"enum={r['enum_s']:7.3f}s  build={r['build_s']:7.3f}s  "
          f"run={r['run_s']:7.3f}s  cmp={r['cmp_s']:6.3f}s  "
          f"TOTAL={r['total_s']:7.3f}s  "
          f"({r['n_total']/r['total_s']/1e6:.2f} M/s)  "
          f"match={r['n_match']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="gelu")
    ap.add_argument("--length", type=int, default=5)
    ap.add_argument(
        "--configs", nargs="+", default=["cpu", "gpu_runner_only", "gpu_full"],
        choices=["cpu", "gpu_runner_only", "gpu_full"],
    )
    args = ap.parse_args()

    target, arity, body, ops, inputs, ref_out = _setup(args.target)
    print(f"Target: {args.target}  arity={arity}  body_len={len(body)}  "
          f"search_length={args.length}  n_ops={len(ops)}")
    print()

    rows = []
    if "cpu" in args.configs:
        rows.append(run_cpu_baseline(arity, ops, inputs, ref_out, args.length))
        _print_row(rows[-1])
    if "gpu_runner_only" in args.configs:
        rows.append(run_gpu_runner_only(arity, ops, inputs, ref_out, args.length))
        _print_row(rows[-1])
    if "gpu_full" in args.configs:
        rows.append(run_gpu_full(arity, ops, inputs, ref_out, args.length))
        _print_row(rows[-1])

    if len(rows) >= 2:
        base = rows[0]["total_s"]
        print()
        print("Speedup vs cpu_baseline:")
        for r in rows:
            print(f"  {r['config']:18s}  {base / r['total_s']:.2f}x")


if __name__ == "__main__":
    main()
