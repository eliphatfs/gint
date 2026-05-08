"""Plot ablation results on roofline chart with eager baseline.

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
    }
    for key, (bw, peak) in specs.items():
        if key in name_lower:
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
    args = parser.parse_args()

    bw_gbs, peak_tflops, gpu_name = get_gpu_roofline()
    peak_gflops = peak_tflops * 1000
    peak_nofma = peak_gflops / 2

    print(f"GPU: {gpu_name}")
    print(f"Memory bandwidth: {bw_gbs:.0f} GB/s")
    print(f"Peak FP32 (no-FMA): {peak_tflops/2:.1f} TFLOPS")

    # --- Read ablation CSV ---
    degrees = []
    ai_vals = []
    mode_data = {}  # mode_label -> {degree: kernel_ms}

    with open(args.csv) as f:
        for row in csv.DictReader(f):
            mode = row['mode']
            deg = int(row['degree'])
            kt = float(row['kernel_ms'])
            if mode not in mode_data:
                mode_data[mode] = {}
            mode_data[mode][deg] = kt
            if deg not in degrees:
                degrees.append(deg)
                ai_vals.append(deg / 4.0)

    degrees = sorted(degrees)
    ai_vals = [d / 4.0 for d in degrees]
    n = 1 << 24

    # --- Run eager baseline ---
    print("\nRunning eager baseline...")
    prime_cuda()
    prime_backends()

    eager_results = {}
    for degree in degrees:
        impls, ref, _, atol, _ = build_roofline(n=n, degree=degree, enable_fp_fusion=False)
        verify("eager", ref, impls["eager"](), atol=atol)
        kt = kernel_time_ms(impls["eager"], warmup=5, iters=30)
        tp = compute_throughput(degree, kt)
        eager_results[degree] = tp
        print(f"  degree={degree:3d}  ai={degree/4:7.2f}  eager={tp:8.1f} GF/s")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(12, 8))

    ai_range = np.logspace(-1, 2.5, 200)
    mem_line = np.minimum(bw_gbs * ai_range, peak_nofma)
    ax.loglog(ai_range, mem_line, 'k--', alpha=0.5, linewidth=1.2,
              label=f"Roofline no-FMA (BW={bw_gbs:.0f} GB/s, peak={peak_tflops/2:.1f} TFLOPS)")

    colors = {
        'A': '#1f77b4',   # blue
        'B': '#ff7f0e',   # orange
        'C': '#2ca02c',   # green
        'D': '#d62728',   # red
        'E': '#9467bd',   # purple
        'eager': '#7f7f7f',  # gray
    }
    labels = {
        'A': 'A: switch (PHI)',
        'B': 'B: alloca (PHI→local mem)',
        'C': 'C: balanced tree',
        'D': 'D: optimal (freq-weighted) tree',
        'E': 'E: alloca + balanced tree',
        'eager': 'PyTorch eager',
    }

    for label in ['A', 'B', 'C', 'D', 'E']:
        if label not in mode_data:
            continue
        xs = [ai_vals[i] for i, d in enumerate(degrees)]
        ys = [compute_throughput(d, mode_data[label][d]) for d in degrees]
        ax.loglog(xs, ys, 'o-', color=colors[label], label=labels[label],
                  linewidth=1.5, markersize=6)

    # Eager
    xs = [d/4.0 for d in degrees]
    ys = [eager_results[d] for d in degrees]
    ax.loglog(xs, ys, 's--', color=colors['eager'], label=labels['eager'],
              linewidth=1.5, markersize=6, alpha=0.8)

    ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)")
    ax.set_ylabel("Throughput (GFLOP/s)")
    ax.set_title(f"Dispatch Ablation — {gpu_name} (Horner polynomial, fp32, no FMA)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    for mode in ['A', 'D']:
        if mode in mode_data:
            for i, d in enumerate(degrees):
                tp = compute_throughput(d, mode_data[mode][d])
                ax.annotate(f"D={d}", (ai_vals[i], tp),
                            textcoords="offset points", xytext=(0, 8),
                            fontsize=6, ha='center', alpha=0.6, color=colors[mode])

    out_path = args.output or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                            "benchmark", "ablation_roofline.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()