"""Experiment 3: Three-way speed comparison.

Compares candidate evaluation throughput across three execution strategies:
  1. CPU Python interpreter  — pure-Python stack machine emulator
  2. Sequential GPU          — one kernel launch per candidate (normal executor)
  3. Batch GPU (ours)        — indirect mode, 1 warp per candidate per launch

Target: relu (69K candidates at length 4, chosen for manageable CPU runtime)
"""

import time
import json
import numpy as np
import torch

from ..opcodes import (
    build_search_ops,
    OP_FADD, OP_FMUL, OP_FSUB, OP_FRSUB, OP_FNEG, OP_FDIV, OP_FRDIV, OP_FREM,
    OP_FPUSH, OP_FADDIMM, OP_FMULIMM,
    OP_FGT, OP_FLT, OP_FGE, OP_FLE, OP_FEQ, OP_FNE, OP_SELECT,
    OP_FSQRT, OP_FRSQRT, OP_FEXP, OP_FEXP2, OP_FLOG, OP_FLOG2,
    OP_FSIN, OP_FCOS, OP_FERF, OP_FRCP,
    OP_DUP, OP_POP, OP_DUPX1, OP_DUPX2, OP_SWAP,
)
from ..candidates import (
    enumerate_exact_length, sequences_to_bytecodes, make_reference_bytecode,
)
from ..executor import BatchRunner, run_reference
from ..targets import get_target

TEST_SIZE = 128
BATCH_SIZE = 4096

# CPU interpreter — all arithmetic uses np.float32 to match GPU precision.
import math
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

F = np.float32


def _cpu_eval_candidate(seq, input_val):
    """Evaluate a candidate on a single scalar input using Python stack machine.

    All arithmetic is done in np.float32 to match GPU precision.
    """
    stack = [F(input_val)]
    try:
        for op in seq:
            oc = op.opcode
            # Binary arithmetic (pop 2, push 1)
            if oc == OP_FADD:
                a, b = stack.pop(), stack.pop(); stack.append(F(a + b))
            elif oc == OP_FMUL:
                a, b = stack.pop(), stack.pop(); stack.append(F(a * b))
            elif oc == OP_FSUB:
                a, b = stack.pop(), stack.pop(); stack.append(F(a - b))
            elif oc == OP_FRSUB:
                a, b = stack.pop(), stack.pop(); stack.append(F(b - a))
            elif oc == OP_FDIV:
                a, b = stack.pop(), stack.pop()
                stack.append(F(a / b) if b != 0 else F(float('inf')))
            elif oc == OP_FRDIV:
                a, b = stack.pop(), stack.pop()
                stack.append(F(b / a) if a != 0 else F(float('inf')))
            elif oc == OP_FREM:
                a, b = stack.pop(), stack.pop()
                stack.append(F(math.fmod(float(a), float(b))) if b != 0 else F(float('nan')))
            # Unary arithmetic
            elif oc == OP_FNEG:
                stack.append(F(-stack.pop()))
            elif oc == OP_FRCP:
                v = stack.pop()
                stack.append(F(1.0 / float(v)) if v != 0 else F(float('inf')))
            # Comparisons (pop 2, push 1)
            elif oc == OP_FGT:
                a, b = stack.pop(), stack.pop()
                stack.append(F(1.0) if a > b else F(0.0))
            elif oc == OP_FLT:
                a, b = stack.pop(), stack.pop()
                stack.append(F(1.0) if a < b else F(0.0))
            elif oc == OP_FGE:
                a, b = stack.pop(), stack.pop()
                stack.append(F(1.0) if a >= b else F(0.0))
            elif oc == OP_FLE:
                a, b = stack.pop(), stack.pop()
                stack.append(F(1.0) if a <= b else F(0.0))
            elif oc == OP_FEQ:
                a, b = stack.pop(), stack.pop()
                stack.append(F(1.0) if a == b else F(0.0))
            elif oc == OP_FNE:
                a, b = stack.pop(), stack.pop()
                stack.append(F(1.0) if a != b else F(0.0))
            # Select (pop 3, push 1)
            elif oc == OP_SELECT:
                cond, t_val, f_val = stack.pop(), stack.pop(), stack.pop()
                stack.append(t_val if cond > 0 else f_val)
            # Transcendentals
            elif oc == OP_FSQRT:
                stack.append(F(math.sqrt(float(max(stack.pop(), F(0.0))))))
            elif oc == OP_FRSQRT:
                v = float(stack.pop())
                stack.append(F(1.0 / math.sqrt(v)) if v > 0 else F(float('inf')))
            elif oc == OP_FEXP:
                stack.append(F(math.exp(float(min(stack.pop(), F(88.0))))))
            elif oc == OP_FEXP2:
                stack.append(F(2.0 ** float(min(stack.pop(), F(127.0)))))
            elif oc == OP_FLOG:
                v = float(stack.pop())
                stack.append(F(math.log(v)) if v > 0 else F(float('-inf')))
            elif oc == OP_FLOG2:
                v = float(stack.pop())
                stack.append(F(math.log2(v)) if v > 0 else F(float('-inf')))
            elif oc == OP_FSIN:
                stack.append(F(math.sin(float(stack.pop()))))
            elif oc == OP_FCOS:
                stack.append(F(math.cos(float(stack.pop()))))
            elif oc == OP_FERF:
                stack.append(F(math.erf(float(stack.pop()))))
            # Immediates
            elif oc == OP_FPUSH:
                stack.append(np.array(op.operand, dtype=np.int32).view(np.float32).item())
            elif oc == OP_FADDIMM:
                val = np.array(op.operand, dtype=np.int32).view(np.float32).item()
                stack.append(F(stack.pop() + F(val)))
            elif oc == OP_FMULIMM:
                val = np.array(op.operand, dtype=np.int32).view(np.float32).item()
                stack.append(F(stack.pop() * F(val)))
            # Stack manipulation
            elif oc == OP_DUP:
                stack.append(stack[-1])
            elif oc == OP_POP:
                stack.pop()
            elif oc == OP_SWAP:
                stack[-1], stack[-2] = stack[-2], stack[-1]
            elif oc == OP_DUPX1:
                v1, v2 = stack.pop(), stack.pop()
                stack.extend([v1, v2, v1])
            elif oc == OP_DUPX2:
                v1, v2, v3 = stack.pop(), stack.pop(), stack.pop()
                stack.extend([v1, v3, v2, v1])
            else:
                return F(float('nan'))
        return stack[-1] if stack else F(float('nan'))
    except (IndexError, ValueError, OverflowError, ZeroDivisionError):
        return F(float('nan'))


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


def run(target_name="relu", search_length=4, max_cpu_candidates=None,
        max_seq_candidates=None, verbose=True):
    """Run three-way comparison. Returns dict of results."""
    target = get_target(target_name)
    arity = target["arity"]
    body = target["body"]

    ops = build_search_ops(include_transcendental=False)
    seqs = list(enumerate_exact_length(ops, arity, 1, search_length))
    n_total = len(seqs)

    # Cap candidate counts for slower methods
    cpu_n = min(n_total, max_cpu_candidates or n_total)
    seq_n = min(n_total, max_seq_candidates or min(n_total, 2000))

    inputs = _make_test_inputs(arity)
    ref_output = torch.empty(TEST_SIZE, device="cuda", dtype=torch.float32)
    ref_bc = make_reference_bytecode(body, arity)
    run_reference(ref_bc, inputs, ref_output, arity)

    if verbose:
        print(f"Three-way comparison: {target_name} (length {search_length})")
        print(f"  Total candidates: {n_total:,}")
        print(f"  CPU candidates:   {cpu_n:,}")
        print(f"  Seq-GPU cands:    {seq_n:,}")
        print()

    # --- Method 1: CPU Python interpreter ---
    cpu_inputs_np = [t.cpu().numpy() for t in inputs]
    ref_np = ref_output.cpu().numpy()

    t0 = time.perf_counter()
    cpu_matches = 0
    for seq in seqs[:cpu_n]:
        match = True
        for elem_idx in range(TEST_SIZE):
            result = float(_cpu_eval_candidate(seq, cpu_inputs_np[0][elem_idx]))
            expected = float(ref_np[elem_idx])
            r_nan = np.isnan(result)
            e_nan = np.isnan(expected)
            if r_nan and e_nan:
                continue  # both NaN → match
            if r_nan or e_nan:
                match = False  # one NaN, other not → mismatch
                break
            if abs(result - expected) > 1e-6 + 1e-5 * abs(expected):
                match = False
                break
        if match:
            cpu_matches += 1
    t_cpu = time.perf_counter() - t0
    cpu_throughput = cpu_n / t_cpu

    if verbose:
        print(f"  CPU interpreter:    {cpu_throughput:>12,.0f} candidates/s  "
              f"({cpu_n:,} in {t_cpu:.2f}s, {cpu_matches} matches)")

    # --- Method 2: Sequential GPU (one launch per candidate) ---
    runner_single = BatchRunner(arity, TEST_SIZE)
    # Warmup the single-candidate runner
    bc_w = sequences_to_bytecodes([seqs[0]], arity)
    out_w = torch.empty(1, TEST_SIZE, device="cuda", dtype=torch.float32)
    runner_single.run(bc_w, inputs, out_w)

    t0 = time.perf_counter()
    seq_matches = 0
    for seq in seqs[:seq_n]:
        bc = sequences_to_bytecodes([seq], arity)
        out = torch.empty(1, TEST_SIZE, device="cuda", dtype=torch.float32)
        runner_single.run(bc, inputs, out)
        if _compare_outputs(out, ref_output).item():
            seq_matches += 1
    t_seq = time.perf_counter() - t0
    seq_throughput = seq_n / t_seq

    if verbose:
        print(f"  Sequential GPU:     {seq_throughput:>12,.0f} candidates/s  "
              f"({seq_n:,} in {t_seq:.2f}s, {seq_matches} matches)")

    # --- Method 3: Batch GPU (ours) ---
    runner = BatchRunner(arity, TEST_SIZE)
    # Warmup
    bc_warmup = sequences_to_bytecodes(seqs[:min(BATCH_SIZE, n_total)], arity)
    out_warmup = torch.empty(bc_warmup.shape[0], TEST_SIZE, device="cuda", dtype=torch.float32)
    runner.run(bc_warmup, inputs, out_warmup)

    t0 = time.perf_counter()
    batch_matches = 0
    for start in range(0, n_total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_total)
        batch_seqs = seqs[start:end]
        bytecodes = sequences_to_bytecodes(batch_seqs, arity)
        out_buf = torch.empty(len(batch_seqs), TEST_SIZE, device="cuda", dtype=torch.float32)
        runner.run(bytecodes, inputs, out_buf)
        hit_mask = _compare_outputs(out_buf, ref_output)
        batch_matches += hit_mask.sum().item()
    t_batch = time.perf_counter() - t0
    batch_throughput = n_total / t_batch

    if verbose:
        print(f"  Batch GPU (ours):   {batch_throughput:>12,.0f} candidates/s  "
              f"({n_total:,} in {t_batch:.2f}s, {batch_matches} matches)")
        print()
        print(f"  Speedup vs CPU:     {batch_throughput / cpu_throughput:.1f}x")
        print(f"  Speedup vs seq GPU: {batch_throughput / seq_throughput:.1f}x")

    results = {
        "experiment": "three_way_comparison",
        "target": target_name,
        "search_length": search_length,
        "n_candidates_total": n_total,
        "cpu": {
            "n_evaluated": cpu_n,
            "time_s": round(t_cpu, 3),
            "throughput_per_s": round(cpu_throughput),
            "matches": cpu_matches,
        },
        "sequential_gpu": {
            "n_evaluated": seq_n,
            "time_s": round(t_seq, 3),
            "throughput_per_s": round(seq_throughput),
            "matches": seq_matches,
        },
        "batch_gpu": {
            "n_evaluated": n_total,
            "time_s": round(t_batch, 3),
            "throughput_per_s": round(batch_throughput),
            "matches": batch_matches,
        },
        "speedup_vs_cpu": round(batch_throughput / cpu_throughput, 1),
        "speedup_vs_seq_gpu": round(batch_throughput / seq_throughput, 1),
    }

    return results


if __name__ == "__main__":
    run()
