"""Sweep the roofline benchmark and plot results (no-FMA comparison).

Only measures separate mul + add operations — Triton with --fmad=false
(passes enable_fp_fusion=False) and gint (which always uses fmul + faddimm).

Usage:  python benchmark/plot_roofline.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmark.bench_roofline import (
    build_roofline, kernel_time_ms, prime_cuda, prime_backends,
    clear_triton_cache, verify,
)


def get_gpu_roofline():
    """Return (bandwidth_GBs, peak_fp32_TFLOPS) for the current GPU."""
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


def main():
    bw_gbs, peak_tflops, gpu_name = get_gpu_roofline()
    peak_gflops = peak_tflops * 1000
    peak_nofma = peak_gflops / 2  # separate mul + add, not fused

    print(f"GPU: {gpu_name}")
    print(f"Memory bandwidth: {bw_gbs:.0f} GB/s")
    print(f"Peak FP32 (FMA):  {peak_tflops:.1f} TFLOPS")
    print(f"Peak FP32 (no-FMA): {peak_tflops/2:.1f} TFLOPS")

    degrees = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    n = 1 << 24  # 16M elements

    prime_cuda()
    prime_backends()

    results = {"degree": [], "ai": [], "gint": [], "triton-no-fma": [], "eager": []}

    for degree in degrees:
        impls, ref, _, atol, _ = build_roofline(
            n=n, degree=degree, enable_fp_fusion=False)

        verify("gint", ref, impls["gint"](), atol=atol)
        verify("triton-no-fma", ref, impls["triton-no-fma"](), atol=atol)

        gflops = n * degree * 2 / 1e9  # degree FMAs, 2 FLOPs each
        ai = degree / 4.0              # FLOPs / byte

        for name in ["gint", "triton-no-fma", "eager"]:
            kt = kernel_time_ms(impls[name], warmup=5, iters=30)
            throughput = gflops / (kt / 1000.0)
            results[name].append(throughput)

        results["degree"].append(degree)
        results["ai"].append(ai)

        print(f"degree={degree:3d}  ai={ai:7.2f}  "
              f"gint={results['gint'][-1]:8.1f} GF/s  "
              f"triton-no-fma={results['triton-no-fma'][-1]:8.1f} GF/s  "
              f"eager={results['eager'][-1]:8.1f} GF/s")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 7))

    ai_range = np.logspace(-1, 2.5, 200)

    # Non-FMA roofline: memory bandwidth slope + half-peak compute ceiling
    mem_line = np.minimum(bw_gbs * ai_range, peak_nofma)
    ax.loglog(ai_range, mem_line, 'k--', alpha=0.5, linewidth=1.2,
              label=f"Roofline no-FMA (BW={bw_gbs:.0f} GB/s, "
                    f"peak={peak_tflops/2:.1f} TFLOPS)")

    colors = {"gint": "#1f77b4", "triton-no-fma": "#d62728", "eager": "#2ca02c"}
    labels = {"gint": "gint", "triton-no-fma": "Triton (no FMA)", "eager": "eager"}

    for name in ["gint", "triton-no-fma", "eager"]:
        ax.loglog(results["ai"], results[name], 'o-', color=colors[name],
                  label=labels[name], linewidth=1.5, markersize=6)

    ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)")
    ax.set_ylabel("Throughput (GFLOP/s)")
    ax.set_title(f"Roofline (no FMA) — {gpu_name} (Horner polynomial, fp32)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    for name in ["gint", "triton-no-fma"]:
        for i, d in enumerate(results["degree"]):
            ax.annotate(f"D={d}",
                        (results["ai"][i], results[name][i]),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=7, ha='center', alpha=0.7,
                        color=colors[name])

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "roofline_nofma.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()