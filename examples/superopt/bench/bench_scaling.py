"""Experiment 4: Batch size scaling and full-search projections.

Tests how throughput varies with batch size (the number of candidate programs
evaluated per kernel launch), and projects total wall-clock time for a full
gelu search under each execution strategy.

Also measures the actual full gelu brute-force search time as ground truth.
"""

import time
import json
import numpy as np
import torch

from ..opcodes import build_search_ops, OP_FEXP, OP_FERF
from ..candidates import enumerate_exact_length, sequences_to_bytecodes, make_reference_bytecode
from ..executor import BatchRunner, run_reference
from ..targets import get_target

TEST_SIZE = 128


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


def _compare_outputs(candidate_outputs, ref_output, atol=1e-6, rtol=1e-5):
    ref = ref_output.unsqueeze(0)
    diff = (candidate_outputs - ref).abs()
    tol = atol + rtol * ref.abs()
    both_nan = candidate_outputs.isnan() & ref.isnan()
    close = (diff <= tol) | both_nan
    return close.all(dim=1)


def run(target_name="gelu", search_length=5,
        batch_sizes=None, verbose=True):
    """Run batch size scaling experiment. Returns dict of results."""
    if batch_sizes is None:
        batch_sizes = [256, 512, 1024, 2048, 4096, 8192, 16384]

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
    ref_output = torch.empty(TEST_SIZE, device="cuda", dtype=torch.float32)
    ref_bc = make_reference_bytecode(body, arity)
    run_reference(ref_bc, inputs, ref_output, arity)

    # Generate a pool of candidates for scaling test
    seqs = list(enumerate_exact_length(ops, arity, 1, search_length))
    n_total = len(seqs)
    pool_size = min(n_total, max(batch_sizes) * 4)
    pool_seqs = seqs[:pool_size]
    pool_bc = sequences_to_bytecodes(pool_seqs, arity)

    runner = BatchRunner(arity, TEST_SIZE)

    # Warmup
    out_warmup = torch.empty(
        min(1024, pool_size), TEST_SIZE, device="cuda", dtype=torch.float32)
    runner.run(pool_bc[:min(1024, pool_size)], inputs, out_warmup)

    if verbose:
        print(f"Batch size scaling: {target_name} (length {search_length})")
        print(f"  Total candidates: {n_total:,}")
        print(f"  Pool size: {pool_size:,}")
        print()

    scaling_results = []
    for bs in batch_sizes:
        if bs > pool_size:
            continue
        n_reps = max(1, pool_size // bs)
        t0 = time.perf_counter()
        for rep in range(n_reps):
            start = (rep * bs) % pool_size
            end = min(start + bs, pool_size)
            actual_bs = end - start
            if actual_bs < bs and start > 0:
                continue
            out_buf = torch.empty(
                actual_bs, TEST_SIZE, device="cuda", dtype=torch.float32)
            runner.run(pool_bc[start:end], inputs, out_buf)
            _compare_outputs(out_buf, ref_output)
        elapsed = time.perf_counter() - t0
        throughput = (n_reps * bs) / elapsed

        entry = {
            "batch_size": bs,
            "n_reps": n_reps,
            "time_s": round(elapsed, 3),
            "throughput_per_s": round(throughput),
        }
        scaling_results.append(entry)

        if verbose:
            print(f"  batch_size={bs:>6d}: {throughput:>12,.0f} candidates/s  "
                  f"({n_reps} reps, {elapsed:.3f}s)")

    # --- Full search timing ---
    if verbose:
        print()
        print(f"Full brute-force search ({n_total:,} candidates)...")

    t0 = time.perf_counter()
    default_bs = 4096
    n_matches = 0
    for start in range(0, n_total, default_bs):
        end = min(start + default_bs, n_total)
        batch_seqs = seqs[start:end]
        bytecodes = sequences_to_bytecodes(batch_seqs, arity)
        out_buf = torch.empty(
            len(batch_seqs), TEST_SIZE, device="cuda", dtype=torch.float32)
        runner.run(bytecodes, inputs, out_buf)
        hit_mask = _compare_outputs(out_buf, ref_output)
        n_matches += hit_mask.sum().item()
    t_full = time.perf_counter() - t0
    full_throughput = n_total / t_full

    if verbose:
        print(f"  Time: {t_full:.1f}s, Matches: {n_matches}")
        print(f"  Throughput: {full_throughput:,.0f} candidates/s")
        print()

    # Projections based on measured throughputs from the comparison experiment.
    # CPU throughput varies widely: ~300K/s with early NaN exit (most sequences
    # fail on the first element), but much lower on targets with fewer NaN paths.
    # Sequential GPU throughput is dominated by per-launch overhead (~12K/s).
    cpu_est_throughput = 300000   # from bench_comparison (with early NaN exit)
    seq_gpu_est_throughput = 12000  # from bench_comparison (reused runner)

    if verbose:
        cpu_est = n_total / cpu_est_throughput
        seq_est = n_total / seq_gpu_est_throughput
        print(f"  Projected times for {n_total:,} candidates:")
        print(f"    CPU interpreter:   ~{cpu_est:.0f}s  (est. {cpu_est_throughput:,}/s)")
        print(f"    Sequential GPU:    ~{seq_est:.0f}s  (est. {seq_gpu_est_throughput:,}/s)")
        print(f"    Batch GPU (actual): {t_full:.1f}s  ({full_throughput:,.0f}/s)")

    results = {
        "experiment": "batch_scaling",
        "target": target_name,
        "search_length": search_length,
        "n_candidates": n_total,
        "scaling": scaling_results,
        "full_search": {
            "time_s": round(t_full, 1),
            "throughput_per_s": round(full_throughput),
            "matches": n_matches,
        },
    }

    return results


if __name__ == "__main__":
    run()
