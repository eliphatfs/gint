"""Sweep the roofline benchmark and plot results.

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
    build_roofline, median_ms, kernel_time_ms, prime_cuda, prime_backends,
    clear_triton_cache, verify,
)


def get_gpu_roofline():
    """Return (bandwidth_GBs, peak_fp32_TFLOPS) for the current GPU."""
    name = torch.cuda.get_device_name()
    name_lower = name.lower()

    # Hardcoded spec values for common GPUs (the CUDA Python API doesn't
    # expose memory clock / bus width directly).
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

    # Fallback estimate
    sm_count = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"Warning: unknown GPU '{name}'. Using fallback estimate.")
    return 900.0, 30.0, name


def main():
    bw_gbs, peak_tflops, gpu_name = get_gpu_roofline()
    print(f"GPU: {gpu_name}")
    print(f"Memory bandwidth: {bw_gbs:.0f} GB/s")
    print(f"Peak FP32:        {peak_tflops:.1f} TFLOPS")

    degrees = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    n = 1 << 24  # 16M elements

    prime_cuda()
    prime_backends()

    results = {"degree": [], "ai": [], "gint": [], "triton": [], "eager": []}

    for degree in degrees:
        impls, ref, shape_str, atol, _verify_fn = build_roofline(
            n=n, degree=degree)

        # Single verify
        verify("gint", ref, impls["gint"](), atol=atol)
        verify("triton", ref, impls["triton"](), atol=atol)

        # Measure
        gbytes = n * 2 * 4 / 1e9      # load + store, 4 bytes each
        gflops = n * degree * 2 / 1e9  # degree FMAs, 2 FLOPs each
        ai = degree / 4.0              # FLOPs / byte

        row = {"degree": degree, "ai": ai}
        for name in ["gint", "triton", "eager"]:
            kt = kernel_time_ms(impls[name], warmup=5, iters=30)  # ms
            throughput = gflops / (kt / 1000.0)  # GFLOP/s
            row[name] = throughput
            results[name].append(throughput)

        results["degree"].append(degree)
        results["ai"].append(ai)

        print(f"degree={degree:3d}  ai={ai:7.2f}  "
              f"gint={row['gint']:8.1f} GF/s  "
              f"triton={row['triton']:8.1f} GF/s  "
              f"eager={row['eager']:8.1f} GF/s")

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(10, 7))

    # Roofline lines
    ai_range = np.logspace(-1, 2.5, 200)
    # Memory bandwidth bound: GFLOP/s = BW * AI
    mem_line = np.minimum(bw_gbs * ai_range, peak_tflops * 1000)
    ax.loglog(ai_range, mem_line, 'k--', alpha=0.5, linewidth=1,
              label=f"Roofline (BW={bw_gbs:.0f} GB/s, peak={peak_tflops:.1f} TFLOPS)")

    # Peak compute horizontal line
    ax.axhline(y=peak_tflops * 1000, color='k', alpha=0.3, linewidth=0.8)

    colors = {"gint": "#1f77b4", "triton": "#ff7f0e", "eager": "#2ca02c"}

    for name, label in [("gint", "gint"), ("triton", "Triton"), ("eager", "eager")]:
        ax.loglog(results["ai"], results[name], 'o-', color=colors[name],
                  label=label, linewidth=1.5, markersize=6)

    ax.set_xlabel("Arithmetic Intensity (FLOPs / byte)")
    ax.set_ylabel("Throughput (GFLOP/s)")
    ax.set_title(f"Roofline — {gpu_name} (Horner polynomial, fp32)")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)

    # Annotate degree values at each point
    for name in ["gint", "triton"]:
        for i, d in enumerate(results["degree"]):
            ax.annotate(f"D={d}",
                        (results["ai"][i], results[name][i]),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=7, ha='center', alpha=0.7,
                        color=colors[name])

    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "roofline.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()