"""Experiment 1: Phase-level time breakdown of brute-force search.

Measures wall-clock time for each phase of the superoptimizer pipeline:
  1. Enumeration  — generating valid candidate sequences (Python DFS)
  2. Bytecode build — converting sequences to GPU bytecode arrays (numpy)
  3. GPU execution  — batch kernel launch via indirect mode
  4. Comparison     — vectorized output comparison against reference

Target: gelu (6 insns, search length 5 → ~14M candidates)
"""

import time
import json
import sys
import numpy as np
import torch

from ..opcodes import build_search_ops, OP_FEXP, OP_FERF
from ..candidates import enumerate_exact_length, sequences_to_bytecodes, make_reference_bytecode
from ..executor import BatchRunner, run_reference
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


def _compare_outputs(candidate_outputs, ref_output, atol=1e-6, rtol=1e-5):
    ref = ref_output.unsqueeze(0)
    diff = (candidate_outputs - ref).abs()
    tol = atol + rtol * ref.abs()
    both_nan = candidate_outputs.isnan() & ref.isnan()
    close = (diff <= tol) | both_nan
    return close.all(dim=1)


def run(target_name="gelu", search_length=5, verbose=True):
    """Run phase-level profiling. Returns dict of results."""
    target = get_target(target_name)
    arity = target["arity"]
    body = target["body"]

    # Build search ops
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

    # Prepare reference
    inputs = _make_test_inputs(arity)
    ref_output = torch.empty(TEST_SIZE, device="cuda", dtype=torch.float32)
    ref_bc = make_reference_bytecode(body, arity)
    run_reference(ref_bc, inputs, ref_output, arity)
    runner = BatchRunner(arity, TEST_SIZE)

    # --- Phase 1: Enumeration ---
    t0 = time.perf_counter()
    seqs = list(enumerate_exact_length(ops, arity, 1, search_length))
    t_enum = time.perf_counter() - t0
    n_total = len(seqs)

    # --- Phase 2 + 3 + 4: Process in batches ---
    t_build_total = 0.0
    t_gpu_total = 0.0
    t_cmp_total = 0.0
    n_matches = 0

    for start in range(0, n_total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_total)
        batch_seqs = seqs[start:end]

        # Phase 2: Bytecode build
        t1 = time.perf_counter()
        bytecodes = sequences_to_bytecodes(batch_seqs, arity)
        t_build_total += time.perf_counter() - t1

        # Phase 3: GPU execution
        out_buf = torch.empty(
            len(batch_seqs), TEST_SIZE, device="cuda", dtype=torch.float32)
        t2 = time.perf_counter()
        runner.run(bytecodes, inputs, out_buf)
        t_gpu_total += time.perf_counter() - t2

        # Phase 4: Comparison
        t3 = time.perf_counter()
        hit_mask = _compare_outputs(out_buf, ref_output)
        n_matches += hit_mask.sum().item()
        t_cmp_total += time.perf_counter() - t3

    t_total = t_enum + t_build_total + t_gpu_total + t_cmp_total

    results = {
        "experiment": "phase_breakdown",
        "target": target_name,
        "search_length": search_length,
        "n_candidates": n_total,
        "n_ops": len(ops),
        "batch_size": BATCH_SIZE,
        "n_matches": n_matches,
        "phases": {
            "enumeration_s": round(t_enum, 3),
            "bytecode_build_s": round(t_build_total, 3),
            "gpu_execution_s": round(t_gpu_total, 3),
            "comparison_s": round(t_cmp_total, 3),
        },
        "total_s": round(t_total, 3),
        "throughput_candidates_per_s": round(n_total / t_total),
    }

    if verbose:
        print(f"Phase breakdown: {target_name} (length {search_length})")
        print(f"  Candidates: {n_total:,}")
        print(f"  Matches: {n_matches}")
        print()
        for name, t in results["phases"].items():
            pct = 100 * t / t_total if t_total > 0 else 0
            label = name.replace("_s", "").replace("_", " ").title()
            print(f"  {label:20s} {t:8.3f}s  ({pct:5.1f}%)")
        print(f"  {'Total':20s} {t_total:8.3f}s")
        print(f"  Throughput: {results['throughput_candidates_per_s']:,} candidates/s")

    return results


if __name__ == "__main__":
    run()
