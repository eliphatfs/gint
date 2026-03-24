"""Search loop: brute-force and stochastic superoptimization."""

import time
import numpy as np
import torch

from .opcodes import build_search_ops, OP_FEXP, OP_FERF
from .candidates import (
    enumerate_exact_length, sequences_to_bytecodes, make_reference_bytecode,
    random_valid_batch,
)
from .executor import BatchRunner, run_reference
from .targets import get_target

TEST_SIZE = 128       # = WARP_SIZE(32) * REG_WIDTH(4)
BATCH_SIZE = 4096     # candidates per kernel launch


def _needs_transcendental(body):
    """Check if the target body uses any transcendental ops."""
    trans_ops = {OP_FEXP, 49, 50, 51, 52, 40, 41, 42, 53, OP_FERF}
    return any(op in trans_ops for op, _ in body)


def _make_test_inputs(arity, seed=42):
    """Create test input tensors with diverse values."""
    torch.manual_seed(seed)
    inputs = []
    for _ in range(arity):
        x = torch.empty(TEST_SIZE, device="cuda", dtype=torch.float32)
        # Mix: 100 randn + 28 hand-picked values
        x[:100] = torch.randn(100, device="cuda")
        x[100:108] = torch.tensor(
            [0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 100.0],
            device="cuda",
        )
        x[108:116] = torch.tensor(
            [1e-6, -1e-6, 1e6, -1e6, 0.1, -0.1, 3.14, -3.14],
            device="cuda",
        )
        x[116:] = torch.randn(TEST_SIZE - 116, device="cuda") * 10
        inputs.append(x)
    return inputs


def _compare_outputs(candidate_outputs, ref_output, atol=1e-6, rtol=1e-5):
    """Return boolean mask of which candidates match the reference (vectorized)."""
    ref = ref_output.unsqueeze(0)           # (1, T)
    diff = (candidate_outputs - ref).abs()  # (N, T)
    tol = atol + rtol * ref.abs()
    # Both NaN → match;  one NaN → no match
    both_nan = candidate_outputs.isnan() & ref.isnan()
    close = (diff <= tol) | both_nan
    return close.all(dim=1)                 # (N,) bool


# ------------------------------------------------------------------
# Brute-force search
# ------------------------------------------------------------------

def brute_force(target_name, max_length=None, extra_immediates=None,
                verbose=True):
    """Exhaustive search for shorter equivalent sequences.

    Searches lengths 1 .. max_length (default: len(target)-1) and returns
    the first (shortest) batch of matches.
    """
    target = get_target(target_name)
    arity = target["arity"]
    body = target["body"]
    ref_len = len(body)

    if max_length is None:
        max_length = ref_len - 1

    if verbose:
        print(f"Target: {target_name} ({ref_len} insns)")
        print(f"  {target['description']}")
        # Reverse-map opcodes to names for display
        from .opcodes import OP_FPUSH, OP_FADDIMM, OP_FMULIMM
        def _op_name(op, operand):
            imm_ops = {OP_FPUSH: "fpush", OP_FADDIMM: "faddimm", OP_FMULIMM: "fmulimm"}
            if op in imm_ops:
                val = float(np.array(operand, dtype=np.int32).view(np.float32))
                return f"{imm_ops[op]}({val:g})"
            # Find name from search ops or fallback to opcode number
            from .opcodes import build_search_ops as _bso
            for sop in _bso(include_transcendental=True):
                if sop.opcode == op and sop.operand == 0:
                    return sop.name
            return f"op{op}"
        body_names = " -> ".join(_op_name(op, operand) for op, operand in body)
        print(f"  body: {body_names}")
        print(f"Searching for shorter equivalents (max_length={max_length})")
        print()

    # Immediate values: defaults + any appearing in the target body
    imm_vals = {0.0, 0.5, 1.0, 2.0, -1.0}
    if extra_immediates:
        imm_vals.update(extra_immediates)
    # Extract immediates used in the target body
    from .opcodes import OP_FPUSH, OP_FADDIMM, OP_FMULIMM
    for op, operand in body:
        if op in (OP_FPUSH, OP_FADDIMM, OP_FMULIMM) and operand != 0:
            val = float(np.array(operand, dtype=np.int32).view(np.float32))
            imm_vals.add(val)
    imm_vals = sorted(imm_vals)

    ops = build_search_ops(
        include_transcendental=_needs_transcendental(body),
        immediate_values=imm_vals,
    )

    if verbose:
        print(f"Search space: {len(ops)} ops, immediates={imm_vals}")
        print()

    # Prepare test inputs and reference output
    inputs = _make_test_inputs(arity)
    ref_output = torch.empty(TEST_SIZE, device="cuda", dtype=torch.float32)
    ref_bc = make_reference_bytecode(body, arity)
    run_reference(ref_bc, inputs, ref_output, arity)

    runner = BatchRunner(arity, TEST_SIZE)
    t0 = time.perf_counter()

    for length in range(1, max_length + 1):
        # Enumerate all valid sequences of this length
        seqs = list(enumerate_exact_length(ops, arity, 1, length))
        n = len(seqs)

        if n == 0:
            if verbose:
                print(f"  length {length}: 0 valid sequences")
            continue

        # Convert to bytecodes and run in batches
        bytecodes = sequences_to_bytecodes(seqs, arity)
        matches = []

        for start in range(0, n, BATCH_SIZE):
            end = min(start + BATCH_SIZE, n)
            batch = bytecodes[start:end]
            out_buf = torch.empty(
                len(batch), TEST_SIZE, device="cuda", dtype=torch.float32
            )
            runner.run(batch, inputs, out_buf)
            hit_mask = _compare_outputs(out_buf, ref_output)
            for idx in hit_mask.nonzero(as_tuple=True)[0]:
                matches.append(seqs[start + idx.item()])

        elapsed = time.perf_counter() - t0

        if verbose:
            status = f"{len(matches)} match{'es' if len(matches) != 1 else ''}"
            print(f"  length {length}: {n:,} sequences, {status}"
                  f"  ({elapsed:.2f}s)")

        if matches:
            # Verify with additional test vectors
            verified = _verify(matches, arity, body, runner)
            if verbose:
                print()
                print(f"Found {len(verified)} verified shorter sequence(s):")
                for seq in verified:
                    print(f"  {' -> '.join(op.name for op in seq)}")
            return verified

    elapsed = time.perf_counter() - t0
    if verbose:
        print()
        print(f"No shorter equivalent found. "
              f"Current {ref_len}-insn sequence is optimal "
              f"(within search space).  [{elapsed:.2f}s]")
    return []


def _verify(candidates, arity, ref_body, runner, n_trials=10):
    """Re-test candidates with multiple random test-vector sets."""
    ref_bc = make_reference_bytecode(ref_body, arity)
    verified = []

    for seq in candidates:
        ok = True
        for seed in range(100, 100 + n_trials):
            inputs = _make_test_inputs(arity, seed=seed)
            ref_out = torch.empty(TEST_SIZE, device="cuda", dtype=torch.float32)
            run_reference(ref_bc, inputs, ref_out, arity)

            bc = sequences_to_bytecodes([seq], arity)
            out = torch.empty(1, TEST_SIZE, device="cuda", dtype=torch.float32)
            runner.run(bc, inputs, out)

            if not _compare_outputs(out, ref_out).all():
                ok = False
                break
        if ok:
            verified.append(seq)
    return verified


# ------------------------------------------------------------------
# Stochastic search (for longer sequences)
# ------------------------------------------------------------------

def stochastic(target_name, length, n_candidates=100000, n_generations=100,
               mutation_rate=0.3, extra_immediates=None, verbose=True):
    """Random-mutation hill climbing for a target length.

    Generates random valid candidates, then iteratively mutates the best ones.
    """
    target = get_target(target_name)
    arity = target["arity"]
    body = target["body"]

    imm_vals = {0.0, 0.5, 1.0, 2.0, -1.0}
    if extra_immediates:
        imm_vals.update(extra_immediates)
    from .opcodes import OP_FPUSH, OP_FADDIMM, OP_FMULIMM
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

    runner = BatchRunner(arity, TEST_SIZE)
    rng = np.random.default_rng(42)
    t0 = time.perf_counter()

    if verbose:
        print(f"Stochastic search: target={target_name}, length={length}, "
              f"pop={n_candidates}, gens={n_generations}")

    best_score = -1.0

    for gen in range(n_generations):
        seqs = random_valid_batch(ops, arity, 1, length, n_candidates, rng=rng)
        if not seqs:
            continue

        bytecodes = sequences_to_bytecodes(seqs, arity)
        matches = []

        for start in range(0, len(seqs), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(seqs))
            batch = bytecodes[start:end]
            out_buf = torch.empty(
                len(batch), TEST_SIZE, device="cuda", dtype=torch.float32
            )
            runner.run(batch, inputs, out_buf)

            hit_mask = _compare_outputs(out_buf, ref_output)
            for idx in hit_mask.nonzero(as_tuple=True)[0]:
                matches.append(seqs[start + idx.item()])

            # Track best score (fraction of matching elements)
            ref = ref_output.unsqueeze(0)
            diff = (out_buf - ref).abs()
            tol = 1e-6 + 1e-5 * ref.abs()
            close_frac = ((diff <= tol).float().mean(dim=1)).max().item()
            best_score = max(best_score, close_frac)

        if matches:
            verified = _verify(matches, arity, body, runner)
            if verified:
                elapsed = time.perf_counter() - t0
                if verbose:
                    print(f"  gen {gen}: FOUND {len(verified)} match(es)  "
                          f"[{elapsed:.2f}s]")
                    for seq in verified:
                        print(f"    {' -> '.join(op.name for op in seq)}")
                return verified

        if verbose and (gen + 1) % 10 == 0:
            elapsed = time.perf_counter() - t0
            print(f"  gen {gen+1}/{n_generations}: best_score={best_score:.4f}"
                  f"  [{elapsed:.2f}s]")

    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"No match found after {n_generations} generations.  [{elapsed:.2f}s]")
    return []
