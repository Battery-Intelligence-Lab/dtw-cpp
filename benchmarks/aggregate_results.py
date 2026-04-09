#!/usr/bin/env python3
"""Aggregate per-dataset timing JSONs into a single benchmark results file.

Reads timing_*.json files from one or more benchmark result directories,
merges them with benchmark_meta.json hardware info, and writes a single
JSON file suitable for the docs website.

Usage:
    python benchmarks/aggregate_results.py results/ucr_benchmark_cpu_12345/
    python benchmarks/aggregate_results.py results/ucr_benchmark_cpu_* results/ucr_benchmark_gpu_* -o benchmarks/ucr_results.json

Output schema (ucr_benchmark_results.json):
{
    "version": "1.0",
    "generated": "2026-04-09T...",
    "runs": [
        {
            "run_id": "cpu_12345",
            "device": "cpu",
            "hardware": { "cpu": "...", "gpu": null, ... },
            "datasets": [
                { "dataset": "Coffee", "k": 2, "cost": 294.778, ... },
                ...
            ]
        },
        ...
    ]
}
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


def load_run(result_dir: Path) -> dict | None:
    """Load a single benchmark run from a results directory."""
    meta_path = result_dir / "benchmark_meta.json"
    if not meta_path.exists():
        print(f"WARNING: No benchmark_meta.json in {result_dir}, skipping", file=sys.stderr)
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    # Collect per-dataset timing files
    datasets = []
    for timing_file in sorted(result_dir.glob("timing_*.json")):
        try:
            with open(timing_file) as f:
                entry = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"WARNING: Malformed JSON in {timing_file}: {e}, skipping", file=sys.stderr)
            continue
        # Keep only the fields needed for the website
        datasets.append({
            "dataset": entry.get("dataset"),
            "n_series": entry.get("n_series"),
            "series_length": entry.get("series_length"),
            "k": entry.get("k"),
            "method": entry.get("method"),
            "cost": entry.get("cost"),
            "converged": entry.get("converged"),
            "iterations": entry.get("iterations"),
            "silhouette_mean": entry.get("silhouette_mean"),
            "gpu_dtw_ms": entry.get("gpu_dtw_ms"),
            "elapsed_s": entry.get("elapsed_s"),
            "status": entry.get("status"),
        })

    # Build run ID from directory name
    run_id = result_dir.name

    # Hardware info
    hardware = {
        "cpu": meta.get("cpu"),
        "cpus": meta.get("cpus"),
        "omp_threads": meta.get("omp_threads"),
        "gpu": meta.get("gpu"),
        "gpu_memory": meta.get("gpu_memory"),
        "gpu_precision": meta.get("gpu_precision"),
        "node": meta.get("node"),
    }

    return {
        "run_id": run_id,
        "device": meta.get("device", "cpu"),
        "method": meta.get("method", "kmedoids"),
        "dtype": meta.get("dtype", "float64"),
        "job_id": meta.get("job_id"),
        "timestamp": meta.get("timestamp_end"),
        "summary": {
            "total": meta.get("total", 0),
            "pass": meta.get("pass", 0),
            "fail": meta.get("fail", 0),
            "skip": meta.get("skip", 0),
            "timeout": meta.get("timeout", 0),
        },
        "hardware": hardware,
        "datasets": datasets,
    }


def merge_runs(runs: list[dict]) -> dict:
    """Merge multiple runs into a single benchmark document."""
    return {
        "version": "1.0",
        "generated": datetime.now(timezone.utc).isoformat(),
        "n_runs": len(runs),
        "runs": runs,
    }


def print_summary_table(runs: list[dict]) -> None:
    """Print a human-readable summary of all runs."""
    for run in runs:
        hw = run["hardware"]
        device_str = hw.get("gpu") or hw.get("cpu") or "unknown"
        n_pass = run["summary"]["pass"]
        n_total = run["summary"]["total"]
        print(f"\n{'='*70}")
        print(f"  Run: {run['run_id']}")
        print(f"  Device: {run['device']} — {device_str}")
        print(f"  Datasets: {n_pass}/{n_total} passed")
        print(f"{'='*70}")

        # Top 10 slowest datasets
        passed = [d for d in run["datasets"] if d["status"] == "pass"]
        passed.sort(key=lambda d: d.get("elapsed_s", 0), reverse=True)
        if passed:
            print(f"\n  {'Dataset':<35} {'N':>6} {'k':>4} {'Time(s)':>8} {'Cost':>12} {'Sil':>6}")
            print(f"  {'-'*35} {'-'*6} {'-'*4} {'-'*8} {'-'*12} {'-'*6}")
            for d in passed[:10]:
                sil = f"{d['silhouette_mean']:.3f}" if d.get("silhouette_mean") not in (None, "null") else "  n/a"
                cost = f"{d['cost']:.3f}" if d.get("cost") not in (None, "null") else "n/a"
                print(f"  {d['dataset']:<35} {d['n_series']:>6} {d['k']:>4} {d['elapsed_s']:>8.2f} {cost:>12} {sil:>6}")

        # Total time
        total_time = sum(d.get("elapsed_s") or 0 for d in run["datasets"])
        print(f"\n  Total benchmark time: {total_time:.1f}s ({total_time/60:.1f} min)")


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate DTWC++ benchmark timing JSONs into a single file"
    )
    parser.add_argument(
        "dirs",
        nargs="+",
        type=Path,
        help="One or more benchmark result directories (e.g., results/ucr_benchmark_cpu_12345/)",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("benchmarks/ucr_benchmark_results.json"),
        help="Output JSON file (default: benchmarks/ucr_benchmark_results.json)",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress summary table output",
    )
    args = parser.parse_args()

    runs = []
    for d in args.dirs:
        if not d.is_dir():
            print(f"WARNING: {d} is not a directory, skipping", file=sys.stderr)
            continue
        run = load_run(d)
        if run:
            runs.append(run)

    if not runs:
        print("ERROR: No valid benchmark runs found.", file=sys.stderr)
        sys.exit(1)

    if not args.quiet:
        print_summary_table(runs)

    result = merge_runs(runs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote {len(runs)} run(s), {sum(r['summary']['total'] for r in runs)} datasets to {args.output}")


if __name__ == "__main__":
    main()
