"""
Roofline benchmark: Horner polynomial evaluation with tunable degree.

Sweeps arithmetic intensity by varying the number of FMA operations per
element loaded.  Each element goes through a geometric-series polynomial:

    p(x) = 1 + x*(1 + x*(1 + ... + x*1))     (degree D = D FMAs)
         = 1 + x + x^2 + ... + x^D

This is non-linear in x, so the compiler cannot collapse the FMA chain into
a single multiply-add.

Arithmetic intensity = D * 2 FLOPs / (2 * 4 bytes) = D/4 FLOPs/byte.

Usage:
    python benchmark/bench_roofline.py --degree 16
    python benchmark/bench_roofline.py --degree 1,4,16,64,256
    python benchmark/bench_roofline.py --degree 16 --clear-triton-cache
"""

import argparse
import csv
import math
import os
import pathlib
import shutil
import sys
import time

import torch
import triton
import triton.language as tl

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gint import bytecode, cdiv
from gint.host.frontend import (
    make_block_1d, fldg_1d, fstg_1d,
    fpush, fadd, fmul, faddimm, halt,
    fload_reg, fstore_reg,
)

# ---------------------------------------------------------------------------
# Timing helpers (mirror bench_lowlevel.py)
# ---------------------------------------------------------------------------

def time_call_ms(fn) -> float:
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0


def median_ms(fn, warmup, iters) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return samples[len(samples) // 2] * 1000.0


def kernel_time_ms(fn, warmup=10, iters=50) -> float:
    import torch.profiler as tp
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    with tp.profile(activities=[tp.ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
    torch.cuda.synchronize()
    total_us = sum(e.self_device_time_total for e in prof.key_averages())
    return total_us / iters / 1000.0


# ---------------------------------------------------------------------------
# Priming
# ---------------------------------------------------------------------------

def prime_cuda():
    x = torch.randn(1024, device='cuda')
    y = torch.randn(1024, device='cuda')
    for _ in range(3):
        z = x + y + x
        del z
    torch.cuda.synchronize()


def prime_backends():
    """Pay gint + triton bootstrap on a smoke function (x * 2.0)."""
    x = torch.randn(64, device='cuda', dtype=torch.float32)
    _ = x * 2.0  # eager

    @triton.jit
    def _smoke_triton(in_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        a = tl.load(in_ptr + offs, mask=mask)
        tl.store(out_ptr + offs, a * 2.0, mask=mask)

    out = torch.empty_like(x)
    _smoke_triton[(triton.cdiv(x.numel(), 1024),)](x, out, x.numel(), BLOCK=1024)

    # gint smoke: simple add kernel to pay executor bootstrap
    @bytecode
    def _smoke_gint(a, b, y, REGW: int, WARP: int, N: int, M: int):
        a_ti = make_block_1d(a, N, 1, 1, 0, [N], [M])
        b_ti = make_block_1d(b, N, 1, 1, 0, [N], [M])
        y_ti = make_block_1d(y, N, 1, 1, 0, [N], [M])
        fldg_1d(0, a_ti)
        fldg_1d(0, b_ti)
        fadd()
        fstg_1d(0, y_ti)
        halt()

    s = torch.randn(128, device='cuda', dtype=torch.float32)
    t = torch.randn(128, device='cuda', dtype=torch.float32)
    u = torch.empty(128, device='cuda', dtype=torch.float32)
    _smoke_gint(s, t, u, N=128, M=1, grid_dim=1)
    torch.cuda.synchronize()


def clear_triton_cache():
    user = os.environ.get('USER') or os.environ.get('LOGNAME') or 'root'
    targets = [
        pathlib.Path.home() / ".triton" / "cache",
        pathlib.Path(f"/tmp/torchinductor_{user}"),
    ]
    for cache in targets:
        if cache.exists():
            shutil.rmtree(cache)
            print(f"Cleared {cache}")
        else:
            print(f"No cache at {cache}")


def verify(name, ref, got, atol):
    if not torch.allclose(ref, got, atol=atol, rtol=1e-3):
        diff = (ref - got).abs().max().item()
        print(f"  [warn] {name}: max abs diff = {diff:.3e}")


# ---------------------------------------------------------------------------
# Polynomial reference
# ---------------------------------------------------------------------------

def poly_ref(x: torch.Tensor, degree: int) -> torch.Tensor:
    """Evaluate p(x) = 1 + x + x^2 + ... + x^D via Horner's method."""
    acc = torch.full_like(x, 1.0)          # c_D = 1
    for _ in range(degree):
        acc = acc * x + 1.0                 # x * acc + c_k
    return acc


# ---------------------------------------------------------------------------
# Gint kernel (built dynamically per degree)
# ---------------------------------------------------------------------------

def _make_gint_poly(degree: int):
    """Return a SugarProgram that evaluates a degree-D geometric-series
    polynomial via Horner's method.

    Stack trace per iteration (acc = acc * x + 1.0):
        fload_reg(0)   # [acc, x]
        fmul()         # [acc * x]
        faddimm(1.0)   # [acc * x + 1.0]
    """

    @bytecode
    def kernel(x, y, REGW: int, WARP: int, N: int, M: int):
        x_ti = make_block_1d(x, N, 1, 1, 0, [N], [M])
        y_ti = make_block_1d(y, N, 1, 1, 0, [N], [M])

        chunk = REGW * WARP
        for off in range(0, N, chunk):
            fldg_1d(off, x_ti)           # x
            fstore_reg(0)                 # ; reg0 = x
            fpush(1.0)                    # acc = 1.0 (= c_D)
            for _ in range(degree):
                fload_reg(0)              # [acc, x]
                fmul()                    # [acc * x]
                faddimm(1.0)              # [acc * x + 1.0]
            fstg_1d(off, y_ti)            # store result
        halt()

    return kernel


# ---------------------------------------------------------------------------
# Triton kernel (built dynamically per degree)
# ---------------------------------------------------------------------------

def _make_triton_poly(degree: int):
    """Return a triton kernel for degree-D geometric-series polynomial.

    DEGREE is baked at JIT time via tl.constexpr so triton can unroll the
    loop over the degree-D Horner chain.
    """
    # Bake the degree into the function's default argument so it is visible
    # to triton's source-inspection without a closure capture.
    D = degree

    @triton.jit
    def kernel(x_ptr, y_ptr, n, DEGREE: tl.constexpr, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        x = tl.load(x_ptr + offs, mask=mask)
        acc = 1.0 + 0.0 * x            # c_D
        for _ in range(DEGREE):
            acc = x * acc + 1.0         # c_k = 1 for all k
        tl.store(y_ptr + offs, acc, mask=mask)

    return kernel, D


# ---------------------------------------------------------------------------
# Build function
# ---------------------------------------------------------------------------

def build_roofline(n=1 << 24, degree=16, enable_fp_fusion=True):
    """Build roofline benchmark for a given polynomial degree.

    Input uniform in [-0.5, 0.5] so the geometric series is bounded:
    p(x) ≤ 1/(1 - 0.5) = 2.0.

    If enable_fp_fusion is False, Triton passes --fmad=false to ptxas,
    so multiplies and adds stay as separate instructions (matching gint).
    """
    torch.manual_seed(42)
    x = torch.rand(n, device='cuda', dtype=torch.float32) - 0.5  # [-0.5, 0.5]
    y_gint = torch.empty(n, device='cuda', dtype=torch.float32)
    ref = poly_ref(x, degree)

    gint_kernel = _make_gint_poly(degree)
    triton_kernel, _D = _make_triton_poly(degree)

    chunk = 128  # REGW * WARP
    M = cdiv(n, chunk)
    N = chunk

    impls = {}
    impls['eager'] = lambda _x=x: poly_ref(_x, degree)

    def triton_call():
        out = torch.empty_like(x)
        BLOCK = 1024
        triton_kernel[(triton.cdiv(n, BLOCK),)](
            x, out, n, DEGREE=_D, BLOCK=BLOCK, enable_fp_fusion=enable_fp_fusion)
        return out
    triton_label = "triton" if enable_fp_fusion else "triton-no-fma"
    impls[triton_label] = triton_call

    def gint_call():
        gint_kernel(x, y_gint, N=N, M=M, grid_dim=M)
        return y_gint
    impls['gint'] = gint_call
    impls['gint'] = gint_call

    ai = degree / 4.0  # FLOPs / byte
    shape_str = f"n={n}, degree={degree}, arith_intensity={ai:.2f} FLOP/B"
    return impls, ref, shape_str, 1e-2, gint_call


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Roofline benchmark — Horner polynomial with tunable degree")
    p.add_argument('--degree', type=str, default='16',
                   help='Comma-separated polynomial degrees (FMAs per element). '
                        'Each degree is benchmarked separately.')
    p.add_argument('--n', type=int, default=1 << 24,
                   help='Number of elements (default: 2^24)')
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--startup-calls', type=int, default=5)
    p.add_argument('--clear-triton-cache', action='store_true')
    p.add_argument('--triton-no-fma', action='store_true',
                   help='Pass --fmad=false to ptxas (disables FMA fusion in Triton)')
    p.add_argument('--gint-only', action='store_true',
                   help='Only benchmark gint (skip triton and eager)')
    p.add_argument('--output-csv', type=str, default=None,
                   help='Write results to CSV file instead of stdout table')
    args = p.parse_args()

    degrees = [int(d.strip()) for d in args.degree.split(',')]

    if args.clear_triton_cache:
        clear_triton_cache()

    prime_cuda()
    prime_backends()

    # Prepare CSV output if requested
    csv_writer = None
    csv_file = None
    if args.output_csv:
        csv_file = open(args.output_csv, 'w', newline='')
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['degree', 'impl', 'startup_ms', 'wall_ms', 'kernel_ms'])

    K = args.startup_calls

    for degree in degrees:
        impls, ref, shape_str, atol, _verify_fn = build_roofline(
            n=args.n, degree=degree,
            enable_fp_fusion=not args.triton_no_fma)

        if not args.gint_only:
            print(f"\n{'='*70}")
            print(f"poly (geometric series) roofline benchmark  ({shape_str})")
            print(f"device={torch.cuda.get_device_name()}, dtype=fp32")
            print(f"warmup={args.warmup}, iters={args.iters}\n")

            header = f"{'impl':<10}" + "".join(
                f"{'call'+str(i+1):>10}" for i in range(K))
            header += f"{'startup':>12}{'wall':>10}{'kernel':>10}"
            print(header)
            print('-' * len(header))

        for name, fn in impls.items():
            if args.gint_only and name != 'gint':
                continue
            early = [time_call_ms(fn) for _ in range(K)]
            verify(name, ref, fn(), atol=atol)
            runtime = median_ms(fn, args.warmup, args.iters)
            kernel = kernel_time_ms(fn)
            startup = sum(early) - K * runtime

            if csv_writer:
                csv_writer.writerow([degree, name, f"{startup:.3f}", f"{runtime:.4f}", f"{kernel:.4f}"])
            elif not args.gint_only:
                early_str = "".join(f"{t:>10.3f}" for t in early)
                print(f"{name:<10}{early_str}{startup:>12.3f}{runtime:>10.4f}"
                      f"{kernel:>10.4f}")

        if not args.gint_only:
            # Report throughput
            gbytes = args.n * 2 * 4 / 1e9  # load + store, 4 bytes each
            gflops = args.n * degree * 2 / 1e9  # degree FMAs, 2 FLOPs each
            print(f"\nGB per call: {gbytes:.3f}   GFLOP per call: {gflops:.3f}")
            print("startup = sum(first K calls) - K * wall  (ms)")
            print("wall    = end-to-end median per-call time (sync around each "
                  "call)")
            print("kernel  = GPU-only kernel time per call from torch.profiler")

    if csv_file:
        csv_file.close()


if __name__ == '__main__':
    main()