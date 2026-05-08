"""Plot REGW ablation results on roofline chart with eager + triton baselines.

Usage:  python scripts/plot_ablation.py [--csv ablation_results.csv]
"""

import argparse
import csv
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmark.bench_roofline import (
    kernel_time_ms, prime_cuda, prime_backends, build_roofline, verify,
)


def get_gpu_roofline():
    name = torch.cuda.get_device_name()
    name_lower = name.lower()
    specs = {
        "4090": (1008.0, 82.6),
        "3090": (936.0, 35.6),
        "3090 ti": (1008.0, 40.0),
        "a100": (2039.0, 19.5),
        "a100-sxm4": (2039.0, 19.5),
        "h100": (3352.0, 67.0),
        "a6000": (768.0, 38.7),
        "4070": (504.0, 29.1),
        "4080": (716.0, 48.7),
        "rtx 5070 ti": (896.0, 44.4),
        "rtx 5080": (960.0, 53.7),
        "5090": (1790.0, 104.8),
    }
    for key, (bw, peak) in specs.items():
        if key in name_lower:
            return bw, peak, name
    # Try substring match
    for key, (bw, peak) in specs.items():
        if any(part in name_lower for part in key.split()):
            return bw, peak, name
    print(f"Warning: unknown GPU '{name}'. Using fallback estimate.")
    return 900.0, 30.0, name


def compute_throughput(degree, kernel_ms, n=1 << 24):
    gflops = n * degree * 2 / 1e9
    return gflops / (kernel_ms / 1000.0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='ablation_results.csv')
    parser.add_argument('--output', default=None)
    parser.add_argument('--eager', action='store_true',
                        help='Include triton and eager baselines (requires benchmarking them)')
    parser.add_argument('--regws', type=str, default=None,
                        help='Comma-separated REGW values to include (default: all in CSV)')
    args = parser.parse_args()

    bw_gbs, peak_tflops, gpu_name = get_gpu_roofline()
    peak_gflops = peak_tflops * 1000
    peak_nofma = peak_gflops / 2  # without FMA, peak is halved

    print(f"GPU: {gpu_name}")
    print(f"Memory bandwidth: {bw_gbs:.0f} GB/s")
    print(f"Peak FP32 (no-FMA): {peak_tflops/2:.1f} TFLOPS")

    # --- Read ablation CSV ---
    csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.csv)
    degrees = []
    regw_data = {}  # regw -> {degree: kernel_ms}

    with open(csv_path) as f:
        for row in csv.DictReader(f):
            regw = int(row['regw'])
            deg = int(row['degree'])
            kt = float(row['kernel_ms'])
            if regw not in regw_data:
                regw_data[regw] = {}
            regw_data[regw][deg] = kt
            if deg not in degrees:
                degrees.append(deg)

    degrees = sorted(degrees)
    ai_vals = [d / 4.0 for d in degrees]
    n = 1 << 24

    # --- Run eager + triton baselines ---
    eager_results = {}
    triton_results = {}
    if args.eager:
        print("\nRunning eager + triton baselines...")
        prime_cuda()
        prime_backends()
        for degree in degrees:
            impls, ref, _, atol, _ = build_roofline(n=n, degree=degree, enable_fp_fusion=False)
            verify("eager", ref, impls["eager"](), atol=atol)
            kt = kernel_time_ms(impls["eager"], warmup=5, iters=30)
            eager_results[degree] = compute_throughput(degree, kt)
            if "triton-no-fma" in impls:
                verify("triton", ref, impls["triton-no-fma"](), atol=atol)
                kt = kernel_time_ms(impls["triton-no-fma"], warmup=5, iters=30)
                triton_results[degree] = compute_throughput(degree, kt)
            print(f"  degree={degree:3d}  ai={degree/4:7.2f}  eager={eager_results[degree]:8.1f} GF/s"
                  f"  triton={triton_results.get(degree, 0):8.1f} GF/s")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 8))

    # Roofline curves
    ai_range = np.logspace(-1, 2.5, 200)
    mem_line = np.minimum(bw_gbs * ai_range, peak_nofma)
    ax.loglog(ai_range, mem_line, 'k--', alpha=0.5, linewidth=1.2,
              label=f"Roofline no-FMA (BW={bw_gbs:.0f} GB/s, peak={peak_tflops/2:.1f} TFLOPS)")

    # REGW colors
    regw_list = [int(r) for r in args.regws.split(',')] if args.regws else sorted(regw_data.keys())
    cmap = plt.cm.viridis
    colors = {regw: cmap(i / max(1, len(regw_list) - 1)) for i, regw in enumerate(regw_list)}

    for regw in regw_list:
        if regw not in regw_data:
            continue
        xs = [ai_vals[i] for i, d in enumerate(degrees) if d in regw_data[regw]]
        ys = [compute_throughput(d, regw_data[regw][d]) for d in degrees if d in regw_data[regw]]
        label = f"REGW={regw}" if regw != 4 else f"REGW={regw} (current)"
        ax.loglog(xs, ys, 'o-', color=colors[regw], label=label,
                  linewidth=1.5, markersize=6)

    # Baselines
    if args.eager:
        xs = [d/4.0 for d in degrees]
        ys = [eager_results[d] for d in degrees]
        ax.loglog(xs, ys, 's--', color='gray', label='PyTorch eager',
                  linewidth=1.5, markersize=6, alpha=0.8)
        if triton_results:
            ys = [triton_results[d] for d in degrees]
            ax.loglog(xs, ys, '^--', color='orange', label='Triton (no-FMA)',
                      linewidth=1.5, markersize=6, alpha=0.8)

    ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)")
    ax.set_ylabel("Throughput (GFLOP/s)")
    ax.set_title(f"REGW Ablation — {gpu_name} (Horner polynomial, fp32, no FMA)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    # Label each degree on the REGW=4 (current) line
    if 4 in regw_data:
        for i, d in enumerate(degrees):
            if d in regw_data[4]:
                tp = compute_throughput(d, regw_data[4][d])
                ax.annotate(f"D={d}", (ai_vals[i], tp),
                            textcoords="offset points", xytext=(0, 8),
                            fontsize=6, ha='center', alpha=0.8, color=colors[4])

    out_path = args.output or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                           "benchmark", "ablation_roofline.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()