"""Ablation runner: builds kernels for each dispatch mode, runs benchmarks
in separate Python processes (to ensure the executor reloads the fatbin),
and collects results into a CSV file.

Usage:
    python scripts/run_ablation.py
    python scripts/run_ablation.py --modes A,B,C
    python scripts/run_ablation.py --degrees 1,4,16,64,256
"""

import argparse
import csv
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_kernel(dispatch_mode: str):
    """Build fatbin for a given dispatch mode."""
    # Step 1: Generate PTX
    subprocess.run([
        'gint-gen-llir',
        '-t', 'ptx', '--cc', '70',
        '--dispatch-mode', dispatch_mode,
        '-o', os.path.join(PROJECT_ROOT, 'artifact', 'gint.ptx'),
    ], check=True, cwd=PROJECT_ROOT)

    # Step 2: Compile to fatbin (sm_89 only -- local GPU)
    subprocess.run([
        'nvcc', '-lineinfo', '-fatbin',
        '-gencode', 'arch=compute_89,code=sm_89',
        os.path.join(PROJECT_ROOT, 'artifact', 'gint.ptx'),
        '-o', os.path.join(PROJECT_ROOT, 'artifact', 'gint.fatbin'),
    ], check=True, cwd=PROJECT_ROOT)

    # Step 3: Compress and deploy
    subprocess.run([
        'xz', '-efk',
        os.path.join(PROJECT_ROOT, 'artifact', 'gint.fatbin'),
    ], check=True, cwd=PROJECT_ROOT)
    subprocess.run([
        'cp', '-v',
        os.path.join(PROJECT_ROOT, 'artifact', 'gint.fatbin.xz'),
        os.path.join(PROJECT_ROOT, 'gint', 'host', 'cuda', 'gint.fatbin.xz'),
    ], check=True, cwd=PROJECT_ROOT)


def run_benchmark(degrees: list[int], output_csv: str):
    """Run bench_roofline.py in a subprocess (fresh executor)."""
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, 'benchmark', 'bench_roofline.py'),
        '--degree', ','.join(str(d) for d in degrees),
        '--gint-only',
        '--clear-triton-cache',
        '--triton-no-fma',
        '--output-csv', output_csv,
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)


def main():
    parser = argparse.ArgumentParser(description='Ablation study runner')
    parser.add_argument('--modes', type=str, default='A,B,C,D,E',
                        help='Comma-separated modes to run')
    parser.add_argument('--degrees', type=str, default='1,2,4,8,16,32,64,128,256',
                        help='Comma-separated polynomial degrees')
    parser.add_argument('--output', type=str, default='ablation_results.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    mode_labels = [m.strip() for m in args.modes.split(',')]
    degrees = [int(d.strip()) for d in args.degrees.split(',')]

    MODE_MAP = {
        'A': 'switch',
        'B': 'alloca-switch',
        'C': 'balanced',
        'D': 'optimal',
        'E': 'alloca-balanced',
    }

    all_rows = []
    for label in mode_labels:
        dispatch_mode = MODE_MAP[label]
        print(f"\n{'='*60}")
        print(f"Setup {label}: dispatch_mode={dispatch_mode}")
        print(f"{'='*60}")

        # Build kernel
        print(f"Building kernel for mode {label}...")
        build_kernel(dispatch_mode)
        print(f"Build complete.")

        # Run benchmark in fresh process
        tmp_csv = f"/tmp/ablation_{label}.csv"
        print(f"Running benchmark for mode {label} (degrees: {degrees})...")
        run_benchmark(degrees, tmp_csv)

        # Read results
        with open(tmp_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['mode'] = label
                all_rows.append(row)
                print(f"  deg={row['degree']:>4s} startup={row['startup_ms']:>10s} "
                      f"wall={row['wall_ms']:>10s} kernel={row['kernel_ms']:>10s}")

    if all_rows:
        with open(args.output, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['mode', 'degree', 'impl', 'startup_ms', 'wall_ms', 'kernel_ms'])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nResults written to {args.output}")
    else:
        print("\nNo results collected.")

    # Restore default switch kernel
    print("\nRestoring default (switch) kernel...")
    build_kernel('switch')
    print("Done.")


if __name__ == '__main__':
    main()