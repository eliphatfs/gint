"""Ablation runner: builds kernels for each REGW value, runs benchmarks
in separate Python processes (to ensure the executor reloads the fatbin),
and collects results into a CSV file.

Usage:
    python scripts/run_ablation.py
    python scripts/run_ablation.py --regws 1,2,4,8,16
    python scripts/run_ablation.py --degrees 1,2,4,8,16,32,64,128,256
"""

import argparse
import csv
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_kernel(regw: int):
    """Build fatbin for a given REG_WIDTH."""
    env = {**os.environ, 'REG_WIDTH': str(regw)}
    subprocess.run(
        ['bash', os.path.join(PROJECT_ROOT, 'generate.sh')],
        check=True, cwd=PROJECT_ROOT, env=env)
    # Verify the kernel was deployed
    fatbin_path = os.path.join(PROJECT_ROOT, 'gint', 'host', 'cuda', 'gint.fatbin.xz')
    if not os.path.exists(fatbin_path):
        raise FileNotFoundError(f"fatbin not found at {fatbin_path}")


def run_benchmark(regw: int, degrees: list[int], output_csv: str):
    """Run bench_roofline.py in a subprocess (fresh executor)."""
    env = {**os.environ, 'GINT_REG_WIDTH': str(regw)}
    cmd = [
        sys.executable,
        os.path.join(PROJECT_ROOT, 'benchmark', 'bench_roofline.py'),
        '--degree', ','.join(str(d) for d in degrees),
        '--gint-only',
        '--clear-triton-cache',
        '--triton-no-fma',
        '--output-csv', output_csv,
    ]
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT, env=env)


def main():
    parser = argparse.ArgumentParser(description='REGW ablation study runner')
    parser.add_argument('--regws', type=str, default='1,2,4,8,16',
                        help='Comma-separated REGW values to test')
    parser.add_argument('--degrees', type=str, default='1,2,4,8,16,32,64,128,256',
                        help='Comma-separated polynomial degrees')
    parser.add_argument('--output', type=str, default='ablation_results.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    regws = [int(r.strip()) for r in args.regws.split(',')]
    degrees = [int(d.strip()) for d in args.degrees.split(',')]

    all_rows = []
    for regw in regws:
        print(f"\n{'='*60}")
        print(f"REGW={regw}")
        print(f"{'='*60}")

        # Build kernel
        print(f"Building kernel for REGW={regw}...")
        build_kernel(regw)
        print(f"Build complete.")

        # Run benchmark in fresh process
        tmp_csv = f"/tmp/ablation_regw{regw}.csv"
        print(f"Running benchmark for REGW={regw} (degrees: {degrees})...")
        run_benchmark(regw, degrees, tmp_csv)

        # Read results
        with open(tmp_csv, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                row['regw'] = str(regw)
                all_rows.append(row)
                print(f"  deg={row['degree']:>4s}  startup={row['startup_ms']:>10s} "
                      f"wall={row['wall_ms']:>10s}  kernel={row['kernel_ms']:>10s}")

    if all_rows:
        out_path = os.path.join(PROJECT_ROOT, args.output)
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['regw', 'degree', 'impl', 'startup_ms', 'wall_ms', 'kernel_ms'])
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\nResults written to {out_path}")
    else:
        print("\nNo results collected.")

    # Restore default REGW=4 kernel
    print("\nRestoring default (REGW=4) kernel...")
    build_kernel(4)
    print("Done.")


if __name__ == '__main__':
    main()