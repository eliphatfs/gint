"""Run all benchmark experiments and save results to JSON.

Usage:
    python -m examples.superopt.bench.run_all [--out results.json]
"""

import argparse
import json
import os
import sys
import time
import torch


def main():
    parser = argparse.ArgumentParser(description="Run all superoptimizer benchmarks")
    parser.add_argument("--out", default=None,
                        help="Output JSON file (default: bench/results.json)")
    parser.add_argument("--skip", nargs="*", default=[],
                        choices=["phases", "executor", "comparison", "scaling"],
                        help="Skip specific experiments")
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(os.path.dirname(__file__), "results.json")

    # Collect system info
    device_name = torch.cuda.get_device_name(0)
    cuda_version = torch.version.cuda
    pytorch_version = torch.__version__

    all_results = {
        "system": {
            "gpu": device_name,
            "cuda_version": cuda_version,
            "pytorch_version": pytorch_version,
        },
        "experiments": {},
    }

    print("=" * 60)
    print("Superoptimizer Benchmark Suite")
    print(f"  GPU: {device_name}")
    print(f"  CUDA: {cuda_version}, PyTorch: {pytorch_version}")
    print("=" * 60)
    print()

    t_total = time.perf_counter()

    # Experiment 1: Phase breakdown
    if "phases" not in args.skip:
        print("-" * 60)
        print("Experiment 1: Phase-level breakdown")
        print("-" * 60)
        from .bench_phases import run as run_phases
        result = run_phases(target_name="gelu", search_length=5)
        all_results["experiments"]["phase_breakdown"] = result
        print()

    # Experiment 2: Executor breakdown
    if "executor" not in args.skip:
        print("-" * 60)
        print("Experiment 2: Per-batch executor breakdown")
        print("-" * 60)
        from .bench_executor import run as run_executor
        result = run_executor(target_name="relu", n_batches=20)
        all_results["experiments"]["executor_breakdown"] = result
        print()

    # Experiment 3: Three-way comparison
    if "comparison" not in args.skip:
        print("-" * 60)
        print("Experiment 3: Three-way speed comparison")
        print("-" * 60)
        from .bench_comparison import run as run_comparison
        result = run_comparison(target_name="relu", search_length=4,
                                max_seq_candidates=2000)
        all_results["experiments"]["three_way_comparison"] = result
        print()

    # Experiment 4: Batch size scaling
    if "scaling" not in args.skip:
        print("-" * 60)
        print("Experiment 4: Batch size scaling")
        print("-" * 60)
        from .bench_scaling import run as run_scaling
        result = run_scaling(target_name="gelu", search_length=5)
        all_results["experiments"]["batch_scaling"] = result
        print()

    elapsed = time.perf_counter() - t_total
    all_results["total_time_s"] = round(elapsed, 1)

    print("=" * 60)
    print(f"All experiments complete. Total time: {elapsed:.1f}s")
    print(f"Results saved to: {out_path}")
    print("=" * 60)

    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    return all_results


if __name__ == "__main__":
    main()
