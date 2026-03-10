"""
Benchmark runner: measures wall-time throughput for gint, torch, and triton (where available)
implementations across multiple kernels.

Usage:
    python benchmark/bench.py [--kernels bmm4x4 inv4x4 rmsnorm] [--iters 200] [--warmup 20]
"""

import argparse
import time
import torch
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gint import cdiv
from gint.host.frontend import REG_WIDTH
from tests.test_bmm4x4 import bmm4x4_kernel
from tests.test_inv4x4 import inv4x4_kernel


# ---------------------------------------------------------------------------
# Timing helper
# ---------------------------------------------------------------------------

def measure_ms(fn, warmup: int, iters: int) -> float:
    """Returns median milliseconds per call."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return elapsed / iters * 1000.0


# ---------------------------------------------------------------------------
# Per-kernel benchmark configs
# ---------------------------------------------------------------------------

def bench_bmm4x4(iters, warmup):
    B = 65535
    torch.manual_seed(42)
    a = torch.randn(B, 4, 4, device='cuda', dtype=torch.float32)
    b = torch.randn(B, 4, 4, device='cuda', dtype=torch.float32)
    c = torch.empty(B, 4, 4, device='cuda', dtype=torch.float32)
    av, bv, cv = a.view(B, 16), b.view(B, 16), c.view(B, 16)
    grid = cdiv(B, 32)

    results = {}
    results['gint'] = measure_ms(lambda: bmm4x4_kernel(av, bv, cv, grid_dim=grid), warmup, iters)
    results['torch'] = measure_ms(lambda: torch.bmm(a, b), warmup, iters)
    return results


def bench_inv4x4(iters, warmup):
    B = 65535
    torch.manual_seed(42)
    a = torch.randn(B, 4, 4, device='cuda', dtype=torch.float32)
    a = a + 4.0 * torch.eye(4, device='cuda', dtype=torch.float32).unsqueeze(0)
    c = torch.empty(B, 16, device='cuda', dtype=torch.float32)
    av = a.view(B, 16)
    grid = cdiv(B, 32)

    results = {}
    results['gint'] = measure_ms(lambda: inv4x4_kernel(av, c, grid_dim=grid), warmup, iters)
    results['torch'] = measure_ms(lambda: torch.linalg.inv(a), warmup, iters)
    return results


def bench_rmsnorm(iters, warmup):
    try:
        from tests.test_rmsnorm import rmsnorm, generalized_rms_norm
    except ImportError:
        print("  [rmsnorm] Could not import triton, skipping.")
        return {}

    b, t, nh, h = 1, 3000, 12, 64
    torch.manual_seed(42)
    x = (torch.randn(b, t, nh, h, dtype=torch.bfloat16) * 1.5 + 0.5).cuda().transpose(1, 2)
    y = torch.empty_like(x)
    w = torch.rand(h, device='cuda', dtype=torch.bfloat16)
    grid = b * nh * cdiv(t, REG_WIDTH)

    results = {}
    results['gint'] = measure_ms(lambda: rmsnorm(x, y, w, grid_dim=grid), warmup, iters)
    results['triton'] = measure_ms(lambda: generalized_rms_norm(x, x, [3], w, 1e-5), warmup, iters)
    results['torch'] = measure_ms(lambda: torch.nn.functional.rms_norm(x, (h,), w, eps=1e-5), warmup, iters)
    return results


BENCHMARKS = {
    'bmm4x4': bench_bmm4x4,
    'inv4x4': bench_inv4x4,
    'rmsnorm': bench_rmsnorm,
}

ALL_IMPLS = ['gint', 'triton', 'torch']


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Gint benchmark suite')
    parser.add_argument('--kernels', nargs='+', default=list(BENCHMARKS.keys()),
                        choices=list(BENCHMARKS.keys()), help='Kernels to benchmark')
    parser.add_argument('--iters', type=int, default=200, help='Timed iterations')
    parser.add_argument('--warmup', type=int, default=20, help='Warmup iterations')
    args = parser.parse_args()

    print(f"\nBenchmark  (warmup={args.warmup}, iters={args.iters}, device={torch.cuda.get_device_name()})\n")

    col_w = 12
    header = f"{'kernel':<16}" + "".join(f"{impl:>{col_w}}" for impl in ALL_IMPLS)
    print(header)
    print("-" * len(header))

    for name in args.kernels:
        fn = BENCHMARKS[name]
        print(f"  {name:<14}", end="", flush=True)
        results = fn(args.iters, args.warmup)
        for impl in ALL_IMPLS:
            if impl in results:
                print(f"{results[impl]:>{col_w}.3f}ms", end="")
            else:
                print(f"{'N/A':>{col_w}}", end="")
        print()

    print()


if __name__ == '__main__':
    main()
