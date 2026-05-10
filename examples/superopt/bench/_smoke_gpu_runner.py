"""Smoke test: verify BatchRunnerGPU output matches BatchRunner exactly."""

import numpy as np
import torch

from ..opcodes import build_search_ops
from ..candidates import (
    enumerate_exact_length, sequences_to_bytecodes, make_reference_bytecode,
)
from ..executor import BatchRunner, run_reference
from ..executor_torch import BatchRunnerGPU
from ..targets import get_target

TEST_SIZE = 128


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


def main():
    target = get_target("relu")
    arity = target["arity"]
    body_len = 4

    ops = build_search_ops(include_transcendental=False)
    op_to_idx = {(op.opcode, op.operand): i for i, op in enumerate(ops)}

    seqs = list(enumerate_exact_length(ops, arity, 1, body_len))
    print(f"enumerated {len(seqs)} sequences of length {body_len}")

    # Take first 1024 for a quick check
    N = min(1024, len(seqs))
    seqs = seqs[:N]

    # CPU path: build bytecodes and run on BatchRunner
    bc_np = sequences_to_bytecodes(seqs, arity)
    inputs = _make_test_inputs(arity)
    out_cpu = torch.empty(N, TEST_SIZE, device="cuda", dtype=torch.float32)
    cpu_runner = BatchRunner(arity, TEST_SIZE)
    cpu_runner.run(bc_np, inputs, out_cpu)

    # GPU path: build indices and run on BatchRunnerGPU
    indices_np = np.array(
        [[op_to_idx[(op.opcode, op.operand)] for op in seq] for seq in seqs],
        dtype=np.int32,
    )
    indices_t = torch.from_numpy(indices_np).cuda()
    out_gpu = torch.empty(N, TEST_SIZE, device="cuda", dtype=torch.float32)
    gpu_runner = BatchRunnerGPU(arity, ops, TEST_SIZE)
    gpu_runner.run(indices_t, inputs, out_gpu)

    # Compare exactly (same kernel, same inputs, same bytecodes — bit-identical
    # unless a NaN slips in)
    both_nan = out_cpu.isnan() & out_gpu.isnan()
    eq = (out_cpu == out_gpu) | both_nan
    n_mismatch = (~eq).sum().item()
    if n_mismatch == 0:
        print(f"OK: {N} candidates match bit-for-bit")
    else:
        first_bad = (~eq).any(dim=1).nonzero()[0, 0].item()
        print(f"MISMATCH on {n_mismatch} elements")
        print(f"  first bad row {first_bad}: seq = {[op.name for op in seqs[first_bad]]}")
        print(f"  cpu  out[:8] = {out_cpu[first_bad, :8].tolist()}")
        print(f"  gpu  out[:8] = {out_gpu[first_bad, :8].tolist()}")


if __name__ == "__main__":
    main()
