#!/usr/bin/env python3
"""Benchmark: CPU vs CUDA distance matrix computation via Python bindings."""

import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import dtwcpp
except ImportError:
    print("dtwcpp not installed. Run: uv pip install -e .")
    sys.exit(1)


def bench(n_series, length, band=-1, n_repeats=3):
    """Benchmark a single configuration."""
    rng = np.random.default_rng(42)
    series = [list(rng.standard_normal(length)) for _ in range(n_series)]
    num_pairs = n_series * (n_series - 1) // 2

    # CPU
    times_cpu = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        dm_cpu = dtwcpp.compute_distance_matrix(series, band=band, device="cpu")
        times_cpu.append(time.perf_counter() - t0)

    # CUDA
    times_gpu = []
    dm_gpu = None
    if dtwcpp.CUDA_AVAILABLE and dtwcpp.cuda_available():
        # Warm-up
        _ = dtwcpp.compute_distance_matrix(series, band=band, device="cuda")
        for _ in range(n_repeats):
            t0 = time.perf_counter()
            dm_gpu = dtwcpp.compute_distance_matrix(series, band=band, device="cuda")
            times_gpu.append(time.perf_counter() - t0)
        # Verify correctness
        max_diff = np.max(np.abs(dm_cpu - dm_gpu))
        if max_diff > 1e-8:
            print(f"  WARNING: max GPU-CPU diff = {max_diff:.2e}")

    cpu_median = float(np.median(times_cpu))
    gpu_median = float(np.median(times_gpu)) if times_gpu else None
    speedup = cpu_median / gpu_median if gpu_median else None

    return {
        "n_series": n_series,
        "length": length,
        "band": band,
        "pairs": num_pairs,
        "cpu_median_s": cpu_median,
        "gpu_median_s": gpu_median,
        "speedup": speedup,
    }


def main():
    print("=" * 60)
    print("DTWC++ Python Benchmark: CPU vs CUDA")
    print("=" * 60)
    print(f"CUDA compiled: {dtwcpp.CUDA_AVAILABLE}")
    if dtwcpp.CUDA_AVAILABLE:
        print(f"CUDA device:   {dtwcpp.cuda_device_info()}")
        print(f"GPU available: {dtwcpp.cuda_available()}")
    print()

    configs = [
        (50, 100, -1),
        (100, 100, -1),
        (50, 500, -1),
        (100, 500, -1),
        (200, 500, -1),
        (50, 1000, -1),
        (100, 1000, -1),
        (50, 500, 50),
    ]

    results = []
    header = f"{'N':>5} {'L':>5} {'band':>5} | {'CPU (s)':>10} {'GPU (s)':>10} {'Speedup':>8}"
    print(header)
    print("-" * len(header))

    for n, l, b in configs:
        r = bench(n, l, b)
        results.append(r)
        gpu_str = f"{r['gpu_median_s']:.4f}" if r["gpu_median_s"] else "N/A"
        spd_str = f"{r['speedup']:.1f}x" if r["speedup"] else "N/A"
        print(f"{n:5d} {l:5d} {b:5d} | {r['cpu_median_s']:10.4f} {gpu_str:>10} {spd_str:>8}")

    # Save JSON
    out = Path("benchmarks/results/cuda_python_benchmark.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
