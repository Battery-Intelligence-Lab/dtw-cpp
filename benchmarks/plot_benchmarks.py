#!/usr/bin/env python3
"""
Plot DTWC++ benchmark results from Google Benchmark JSON output.

Usage:
    python plot_benchmarks.py results/bench_abc1234.json
    python plot_benchmarks.py results/bench_abc1234.json results/bench_def5678.json  # compare commits

Generates PNG plots in benchmarks/plots/ directory.
"""

import json
import sys
import os
import re
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)


def load_benchmarks(json_path):
    """Load benchmark results from a JSON file."""
    with open(json_path) as f:
        data = json.load(f)
    results = []
    for b in data["benchmarks"]:
        if "aggregate_name" in b:
            continue  # skip mean/median/stddev aggregates
        results.append({
            "name": b["name"],
            "real_time": b.get("real_time", 0),
            "cpu_time": b.get("cpu_time", 0),
            "time_unit": b.get("time_unit", "ns"),
            "items_per_second": b.get("items_per_second", 0),
        })
    label = Path(json_path).stem  # e.g. "bench_abc1234"
    context = data.get("context", {})
    return label, context, results


def parse_benchmark_name(name):
    """Parse 'BM_dtwFull/1000' or 'BM_dtwBanded/4000/50' into (suite, args)."""
    parts = name.split("/")
    suite = parts[0]
    args = [int(x) for x in parts[1:]] if len(parts) > 1 else []
    return suite, args


def to_us(time_val, unit):
    """Convert time to microseconds."""
    if unit == "ns":
        return time_val / 1000
    elif unit == "us":
        return time_val
    elif unit == "ms":
        return time_val * 1000
    return time_val


def plot_dtw_scaling(results_list, labels):
    """Plot DTW computation time vs series length for all algorithms."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for idx, (label, results) in enumerate(zip(labels, results_list)):
        color_offset = idx * 0.3

        # Group by suite
        suites = defaultdict(list)
        for r in results:
            suite, args = parse_benchmark_name(r["name"])
            if suite in ("BM_dtwFull", "BM_dtwFull_L"):
                suites[suite].append((args[0], to_us(r["real_time"], r["time_unit"])))
            elif suite == "BM_dtwBanded" and len(args) == 2:
                key = f"BM_dtwBanded(band={args[1]})"
                suites[key].append((args[0], to_us(r["real_time"], r["time_unit"])))

        # Left plot: log-log scaling
        ax = axes[0]
        markers = {"BM_dtwFull": "s", "BM_dtwFull_L": "o"}
        for suite_name, data_points in sorted(suites.items()):
            data_points.sort()
            lengths, times = zip(*data_points)
            marker = markers.get(suite_name, "^")
            suffix = f" ({label})" if len(labels) > 1 else ""
            ax.loglog(lengths, times, marker=marker, linewidth=2,
                     label=suite_name.replace("BM_", "") + suffix, markersize=8)

        ax.set_xlabel("Series Length (n)", fontsize=12)
        ax.set_ylabel("Time (us)", fontsize=12)
        ax.set_title("DTW Computation Time vs Series Length", fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which="both")

        # Right plot: banded speedup over full
        ax = axes[1]
        full_times = dict(suites.get("BM_dtwFull_L", []))
        for suite_name, data_points in sorted(suites.items()):
            if "Banded" not in suite_name:
                continue
            data_points.sort()
            lengths, times = zip(*data_points)
            speedups = [full_times.get(l, t) / t if t > 0 else 0 for l, t in zip(lengths, times)]
            suffix = f" ({label})" if len(labels) > 1 else ""
            ax.bar([f"n={l}" for l in lengths], speedups, alpha=0.7,
                   label=suite_name.replace("BM_", "") + suffix)

        ax.set_xlabel("Configuration", fontsize=12)
        ax.set_ylabel("Speedup over Full DTW", fontsize=12)
        ax.set_title("Banded DTW Speedup", fontsize=14)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, axis="y")
        ax.axhline(y=1, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    out = PLOT_DIR / "dtw_scaling.png"
    plt.savefig(out, dpi=100)
    print(f"Saved: {out}")
    plt.close()


def plot_distance_matrix(results_list, labels):
    """Plot fillDistanceMatrix performance."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, (label, results) in enumerate(zip(labels, results_list)):
        dm_results = []
        for r in results:
            suite, args = parse_benchmark_name(r["name"])
            if suite == "BM_fillDistanceMatrix":
                N, L, band = args
                time_ms = to_us(r["real_time"], r["time_unit"]) / 1000
                pairs = N * (N - 1) // 2
                dm_results.append((f"N={N}\nL={L}\nb={band}", time_ms, pairs))

        if dm_results:
            configs, times, pairs = zip(*dm_results)
            suffix = f" ({label})" if len(labels) > 1 else ""
            bars = ax.bar([f"{c}" for c in configs], times, alpha=0.7,
                         label=f"Time{suffix}")

            # Add pairs/sec annotation
            for bar, t, p in zip(bars, times, pairs):
                if t > 0:
                    rate = p / (t / 1000)  # pairs per second
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                           f"{rate:.0f}\npairs/s", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Configuration (N series, L length, b band)", fontsize=12)
    ax.set_ylabel("Time (ms)", fontsize=12)
    ax.set_title("Distance Matrix Fill Time", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = PLOT_DIR / "distance_matrix.png"
    plt.savefig(out, dpi=100)
    print(f"Saved: {out}")
    plt.close()


def plot_memory_efficiency(results_list, labels):
    """Plot memory efficiency: Full vs Full_L vs Banded."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Memory usage per algorithm for n=4000
    n = 4000
    algorithms = {
        "dtwFull\n(full matrix)": n * n * 8,  # O(n^2) doubles
        "dtwFull_L\n(rolling buffer)": n * 8,  # O(n) doubles
        "dtwBanded\n(band=50)": n * 8,  # O(n) rolling column
        "dtwBanded\n(band=100)": n * 8,  # O(n) rolling column
    }

    names = list(algorithms.keys())
    memory_kb = [v / 1024 for v in algorithms.values()]

    colors = ["#e74c3c", "#2ecc71", "#3498db", "#9b59b6"]
    bars = ax.bar(names, memory_kb, color=colors, alpha=0.8)

    for bar, mem in zip(bars, memory_kb):
        if mem > 100:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f"{mem:.0f} KB", ha="center", va="bottom", fontsize=11, fontweight="bold")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                   f"{mem:.1f} KB", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Memory per DTW call (KB)", fontsize=12)
    ax.set_title(f"DTW Scratch Memory Usage (n={n})", fontsize=14)
    ax.set_yscale("log")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    out = PLOT_DIR / "memory_efficiency.png"
    plt.savefig(out, dpi=100)
    print(f"Saved: {out}")
    plt.close()


def plot_roofline(results_list, labels, context):
    """Plot roofline model for DTW."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # System parameters (from benchmark context)
    peak_flops = context.get("mhz_per_cpu", 2500) * 1e6 * 2 * 8  # rough: freq * 2 FMA * 8 AVX width
    peak_bw = 50e9  # ~50 GB/s typical DDR4 bandwidth

    # Plot roofline
    oi = [0.01, 0.1, 0.125, 0.5, 1, 2, 5, 10, 50]
    roofline = [min(peak_flops, o * peak_bw) for o in oi]
    ax.loglog(oi, [r / 1e9 for r in roofline], "k-", linewidth=2, label="Roofline")
    ax.axvline(x=0.125, color="red", linestyle="--", alpha=0.7, label="DTW OI = 0.125 FLOP/byte")

    # Plot actual DTW performance points
    for idx, (label, results) in enumerate(zip(labels, results_list)):
        for r in results:
            suite, args = parse_benchmark_name(r["name"])
            if suite == "BM_dtwFull_L" and len(args) == 1:
                n = args[0]
                time_s = to_us(r["real_time"], r["time_unit"]) / 1e6
                flops = 5 * n * n  # ~5 FLOPs per cell
                gflops = flops / time_s / 1e9
                ax.plot(0.125, gflops, "ro", markersize=10, zorder=5)
                ax.annotate(f"n={n}", (0.125, gflops), textcoords="offset points",
                           xytext=(10, 5), fontsize=9)

    ax.set_xlabel("Operational Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Performance (GFLOP/s)", fontsize=12)
    ax.set_title("DTW Roofline Analysis", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(0.01, 50)
    plt.tight_layout()
    out = PLOT_DIR / "roofline.png"
    plt.savefig(out, dpi=100)
    print(f"Saved: {out}")
    plt.close()


def main():
    if len(sys.argv) < 2:
        # Auto-find latest result
        results_dir = Path(__file__).parent / "results"
        json_files = sorted(results_dir.glob("bench_*.json"))
        if not json_files:
            print("Usage: python plot_benchmarks.py results/bench_*.json")
            sys.exit(1)
        files = [str(json_files[-1])]
    else:
        files = sys.argv[1:]

    all_labels = []
    all_results = []
    context = {}

    for f in files:
        label, ctx, results = load_benchmarks(f)
        all_labels.append(label)
        all_results.append(results)
        if not context:
            context = ctx

    print(f"Loaded {len(files)} benchmark file(s): {', '.join(all_labels)}")
    print(f"System: {context.get('num_cpus', '?')} CPUs @ {context.get('mhz_per_cpu', '?')} MHz")
    print()

    plot_dtw_scaling(all_results, all_labels)
    plot_distance_matrix(all_results, all_labels)
    plot_memory_efficiency(all_results, all_labels)
    plot_roofline(all_results, all_labels, context)

    print(f"\nAll plots saved to: {PLOT_DIR}/")


if __name__ == "__main__":
    main()
