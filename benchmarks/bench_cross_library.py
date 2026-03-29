#!/usr/bin/env python3
"""
Cross-library DTW benchmark: dtwcpp vs dtaidistance vs tslearn vs aeon.

Compares pairwise DTW distance, distance matrix construction, and k-medoids
clustering on identical synthetic data with deterministic seeds.

Usage:
    python bench_cross_library.py              # run all benchmarks
    python bench_cross_library.py --quick      # short version (fewer configs)
    python bench_cross_library.py --timeout 60 # per-benchmark timeout in seconds

Results saved to benchmarks/results/cross_library_YYYYMMDD_HHMMSS.json
Plot saved to benchmarks/plots/cross_library.png
"""

import argparse
import json
import signal
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Timeout mechanism (cross-platform)
# ---------------------------------------------------------------------------
class TimeoutError(Exception):
    pass


if sys.platform != "win32":
    def _timeout_handler(signum, frame):
        raise TimeoutError("Benchmark timed out")

    class timeout_context:
        def __init__(self, seconds: int):
            self.seconds = seconds

        def __enter__(self):
            signal.signal(signal.SIGALRM, _timeout_handler)
            signal.alarm(self.seconds)
            return self

        def __exit__(self, *args):
            signal.alarm(0)
else:
    # Windows: use threading-based timeout
    import threading
    import ctypes

    class timeout_context:
        def __init__(self, seconds: int):
            self.seconds = seconds
            self._deadline = None

        def __enter__(self):
            self._deadline = time.monotonic() + self.seconds
            return self

        def __exit__(self, *args):
            pass

        def check(self):
            """Call periodically inside the benchmark to check timeout."""
            if time.monotonic() > self._deadline:
                raise TimeoutError("Benchmark timed out")


# ---------------------------------------------------------------------------
# Data generation (deterministic, matches C++ benchmarks)
# ---------------------------------------------------------------------------
def random_series(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=length)


def random_dataset(n_series: int, length: int, base_seed: int = 100) -> np.ndarray:
    """Generate N random series of given length. Shape: (n_series, length)."""
    return np.array([random_series(length, base_seed + i) for i in range(n_series)])


# ---------------------------------------------------------------------------
# Library wrappers — each returns (time_seconds, result_or_None)
# ---------------------------------------------------------------------------

def bench_dtwcpp_distance(x: np.ndarray, y: np.ndarray) -> float:
    import dtwcpp
    return dtwcpp.dtw_distance(x, y)


def bench_dtwcpp_distmat(data: np.ndarray, band: int = -1) -> np.ndarray:
    import dtwcpp
    n = len(data)
    dm = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            if band > 0:
                d = dtwcpp.dtw_distance(data[i], data[j], band=band)
            else:
                d = dtwcpp.dtw_distance(data[i], data[j])
            dm[i, j] = dm[j, i] = d
    return dm


def bench_dtwcpp_cluster(data: np.ndarray, k: int) -> Any:
    import dtwcpp
    clust = dtwcpp.DTWClustering(n_clusters=k)
    return clust.fit_predict(data)


def bench_dtaidistance_distance(x: np.ndarray, y: np.ndarray) -> float:
    from dtaidistance import dtw
    return dtw.distance_fast(x, y)


def bench_dtaidistance_distmat(data: np.ndarray, band: int = -1) -> Any:
    from dtaidistance import dtw
    kwargs = {}
    if band > 0:
        kwargs["window"] = band
    # dtaidistance wants a list of numpy arrays
    series_list = [data[i] for i in range(len(data))]
    return dtw.distance_matrix_fast(series_list, **kwargs)


def bench_dtaidistance_cluster(data: np.ndarray, k: int) -> Any:
    from dtaidistance import dtw, clustering
    series_list = [data[i] for i in range(len(data))]
    dm = dtw.distance_matrix_fast(series_list)
    model = clustering.KMedoids(dm, k)
    return model.calculate()


def bench_tslearn_distance(x: np.ndarray, y: np.ndarray) -> float:
    from tslearn.metrics import dtw as tslearn_dtw
    return tslearn_dtw(x.reshape(-1, 1), y.reshape(-1, 1))


def bench_tslearn_distmat(data: np.ndarray, band: int = -1) -> Any:
    from tslearn.metrics import cdist_dtw
    kwargs = {}
    if band > 0:
        kwargs["sakoe_chiba_radius"] = band
    return cdist_dtw(data.reshape(data.shape[0], data.shape[1], 1), **kwargs)


def bench_tslearn_cluster(data: np.ndarray, k: int) -> Any:
    from tslearn.clustering import TimeSeriesKMeans
    model = TimeSeriesKMeans(n_clusters=k, metric="dtw", max_iter=10, random_state=42)
    return model.fit_predict(data.reshape(data.shape[0], data.shape[1], 1))


def bench_aeon_distance(x: np.ndarray, y: np.ndarray) -> float:
    from aeon.distances import dtw_distance
    return dtw_distance(x, y)


def bench_aeon_distmat(data: np.ndarray, band: int = -1) -> Any:
    from aeon.distances import pairwise_distance
    kwargs: dict[str, Any] = {"method": "dtw"}
    if band > 0:
        kwargs["window"] = band / len(data[0])  # aeon uses fraction
    return pairwise_distance(data, **kwargs)


def bench_aeon_cluster(data: np.ndarray, k: int) -> Any:
    from aeon.clustering import TimeSeriesKMedoids
    model = TimeSeriesKMedoids(n_clusters=k, method="pam", random_state=42)
    return model.fit_predict(data)


# ---------------------------------------------------------------------------
# Timing harness
# ---------------------------------------------------------------------------

LIBRARIES = {
    "dtwcpp": {
        "distance": bench_dtwcpp_distance,
        "distmat": bench_dtwcpp_distmat,
        "cluster": bench_dtwcpp_cluster,
    },
    "dtaidistance": {
        "distance": bench_dtaidistance_distance,
        "distmat": bench_dtaidistance_distmat,
        "cluster": bench_dtaidistance_cluster,
    },
    "tslearn": {
        "distance": bench_tslearn_distance,
        "distmat": bench_tslearn_distmat,
        "cluster": bench_tslearn_cluster,
    },
    "aeon": {
        "distance": bench_aeon_distance,
        "distmat": bench_aeon_distmat,
        "cluster": bench_aeon_cluster,
    },
}


def check_available(lib_name: str) -> bool:
    """Check if a library is importable."""
    try:
        if lib_name == "dtwcpp":
            import dtwcpp  # noqa: F401
        elif lib_name == "dtaidistance":
            import dtaidistance  # noqa: F401
        elif lib_name == "tslearn":
            import tslearn  # noqa: F401
        elif lib_name == "aeon":
            import aeon  # noqa: F401
        return True
    except ImportError:
        return False


def time_fn(fn, *args, n_repeats: int = 3, timeout_sec: int = 300, **kwargs) -> dict:
    """Time a function, return dict with median/min/max/status."""
    times = []
    result = None

    for rep in range(n_repeats):
        try:
            t0 = time.perf_counter()
            if sys.platform == "win32":
                # Windows: check wall clock after each call
                deadline = t0 + timeout_sec
                result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - t0
                if elapsed > timeout_sec:
                    return {"status": "timeout", "time_sec": elapsed,
                            "median": None, "min": None, "max": None}
            else:
                with timeout_context(timeout_sec):
                    result = fn(*args, **kwargs)
                elapsed = time.perf_counter() - t0

            times.append(elapsed)
        except TimeoutError:
            return {"status": "timeout", "time_sec": timeout_sec,
                    "median": None, "min": None, "max": None}
        except Exception as e:
            return {"status": f"error: {e}", "time_sec": None,
                    "median": None, "min": None, "max": None}

    return {
        "status": "ok",
        "median": float(np.median(times)),
        "min": float(np.min(times)),
        "max": float(np.max(times)),
        "times": [float(t) for t in times],
    }


# ---------------------------------------------------------------------------
# Benchmark configurations
# ---------------------------------------------------------------------------

def get_configs(quick: bool = False):
    """Return benchmark configurations."""
    if quick:
        distance_configs = [(500, 42, 43)]
        distmat_configs = [(20, 500, -1)]
        cluster_configs = [(20, 500, 3)]
    else:
        distance_configs = [
            # (length, seed_x, seed_y)
            (100, 42, 43),
            (500, 42, 43),
            (1000, 42, 43),
            (4000, 42, 43),
        ]
        distmat_configs = [
            # (n_series, length, band)
            (20, 500, -1),
            (50, 500, -1),
            (50, 1000, -1),
            (50, 500, 10),
        ]
        cluster_configs = [
            # (n_series, length, k)
            (50, 500, 3),
        ]
    return distance_configs, distmat_configs, cluster_configs


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def format_time(t: float | None) -> str:
    """Format time in human-readable units."""
    if t is None:
        return "—"
    if t < 1e-3:
        return f"{t*1e6:.0f} us"
    elif t < 1.0:
        return f"{t*1e3:.1f} ms"
    else:
        return f"{t:.2f} s"


def print_table(title: str, headers: list[str], rows: list[list[str]]):
    """Print a markdown-style table."""
    print(f"\n### {title}\n")
    widths = [max(len(h), max((len(r[i]) for r in rows), default=0))
              for i, h in enumerate(headers)]
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep_line = " | ".join("-" * w for w in widths)
    print(f"| {header_line} |")
    print(f"| {sep_line} |")
    for row in rows:
        row_line = " | ".join(str(r).ljust(w) for r, w in zip(row, widths))
        print(f"| {row_line} |")


def make_plot(results: dict, output_path: Path):
    """Generate comparison bar chart."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not available, skipping plot")
        return

    # Collect all benchmark names and library times
    benchmarks = []
    lib_names = ["dtwcpp", "dtaidistance", "tslearn", "aeon"]
    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    for section in ["distance", "distmat", "cluster"]:
        if section not in results:
            continue
        for config_key, lib_results in results[section].items():
            benchmarks.append((section, config_key, lib_results))

    if not benchmarks:
        return

    n_benchmarks = len(benchmarks)
    fig, axes = plt.subplots(1, n_benchmarks, figsize=(4 * n_benchmarks, 5))
    if n_benchmarks == 1:
        axes = [axes]

    for ax, (section, config_key, lib_results) in zip(axes, benchmarks):
        times = []
        labels = []
        bar_colors = []
        for lib, color in zip(lib_names, colors):
            if lib in lib_results:
                r = lib_results[lib]
                t = r.get("median")
                if t is not None:
                    times.append(t)
                    labels.append(lib)
                    bar_colors.append(color)

        if times:
            bars = ax.bar(range(len(times)), times, color=bar_colors, width=0.6)
            ax.set_xticks(range(len(times)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Time (s)")
            ax.set_title(f"{section}\n{config_key}", fontsize=9)

            # Add time labels on bars
            for bar, t in zip(bars, times):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        format_time(t), ha="center", va="bottom", fontsize=7)

    fig.suptitle("Cross-Library DTW Benchmark", fontsize=12, fontweight="bold")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nPlot saved to {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Cross-library DTW benchmark")
    parser.add_argument("--quick", action="store_true", help="Run quick subset")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Per-benchmark timeout in seconds (default: 300)")
    parser.add_argument("--repeats", type=int, default=3,
                        help="Number of repetitions (default: 3)")
    args = parser.parse_args()

    timeout = args.timeout
    n_repeats = args.repeats
    distance_configs, distmat_configs, cluster_configs = get_configs(args.quick)

    # Check library availability
    available = {}
    for lib in LIBRARIES:
        available[lib] = check_available(lib)
        status = "available" if available[lib] else "NOT FOUND"
        print(f"  {lib:20s} {status}")
    print()

    results: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "timeout_sec": timeout,
        "n_repeats": n_repeats,
        "libraries": {k: v for k, v in available.items()},
    }

    # --- Pairwise DTW distance ---
    print("=" * 60)
    print("Benchmark 1: Pairwise DTW Distance")
    print("=" * 60)

    dist_results: dict[str, dict] = {}
    for length, seed_x, seed_y in distance_configs:
        config_key = f"L={length}"
        x = random_series(length, seed_x)
        y = random_series(length, seed_y)

        lib_times: dict[str, dict] = {}
        for lib_name, funcs in LIBRARIES.items():
            if not available[lib_name]:
                lib_times[lib_name] = {"status": "not installed"}
                continue

            print(f"  {lib_name:20s} L={length:5d} ... ", end="", flush=True)
            r = time_fn(funcs["distance"], x, y,
                        n_repeats=n_repeats, timeout_sec=timeout)
            lib_times[lib_name] = r
            if r["status"] == "ok":
                print(f"{format_time(r['median']):>10s}")
            else:
                print(f"  {r['status']}")

        dist_results[config_key] = lib_times

    results["distance"] = dist_results

    # Print distance table
    headers = ["Config"] + [lib for lib in LIBRARIES if available.get(lib)]
    rows = []
    for config_key, lib_times in dist_results.items():
        row = [config_key]
        for lib in LIBRARIES:
            if not available.get(lib):
                continue
            r = lib_times.get(lib, {})
            if r.get("status") == "ok":
                row.append(format_time(r["median"]))
            elif r.get("status") == "timeout":
                row.append("TIMEOUT")
            else:
                row.append(r.get("status", "—"))
        rows.append(row)
    print_table("Pairwise DTW Distance", headers, rows)

    # --- Distance matrix ---
    print("\n" + "=" * 60)
    print("Benchmark 2: Distance Matrix Construction")
    print("=" * 60)

    dm_results: dict[str, dict] = {}
    for n_series, length, band in distmat_configs:
        config_key = f"N={n_series} L={length}" + (f" band={band}" if band > 0 else " full")
        data = random_dataset(n_series, length)

        lib_times: dict[str, dict] = {}
        for lib_name, funcs in LIBRARIES.items():
            if not available[lib_name]:
                lib_times[lib_name] = {"status": "not installed"}
                continue

            print(f"  {lib_name:20s} {config_key:25s} ... ", end="", flush=True)
            r = time_fn(funcs["distmat"], data, band,
                        n_repeats=n_repeats, timeout_sec=timeout)
            lib_times[lib_name] = r
            if r["status"] == "ok":
                print(f"{format_time(r['median']):>10s}")
            else:
                print(f"  {r['status']}")

        dm_results[config_key] = lib_times

    results["distmat"] = dm_results

    headers = ["Config"] + [lib for lib in LIBRARIES if available.get(lib)]
    rows = []
    for config_key, lib_times in dm_results.items():
        row = [config_key]
        for lib in LIBRARIES:
            if not available.get(lib):
                continue
            r = lib_times.get(lib, {})
            if r.get("status") == "ok":
                row.append(format_time(r["median"]))
            elif r.get("status") == "timeout":
                row.append("TIMEOUT")
            else:
                row.append(r.get("status", "—"))
        rows.append(row)
    print_table("Distance Matrix Construction", headers, rows)

    # --- Clustering ---
    print("\n" + "=" * 60)
    print("Benchmark 3: k-Medoids Clustering")
    print("=" * 60)

    cl_results: dict[str, dict] = {}
    for n_series, length, k in cluster_configs:
        config_key = f"N={n_series} L={length} k={k}"
        data = random_dataset(n_series, length)

        lib_times: dict[str, dict] = {}
        for lib_name, funcs in LIBRARIES.items():
            if not available[lib_name]:
                lib_times[lib_name] = {"status": "not installed"}
                continue

            print(f"  {lib_name:20s} {config_key:25s} ... ", end="", flush=True)
            r = time_fn(funcs["cluster"], data, k,
                        n_repeats=n_repeats, timeout_sec=timeout)
            lib_times[lib_name] = r
            if r["status"] == "ok":
                print(f"{format_time(r['median']):>10s}")
            else:
                print(f"  {r['status']}")

        cl_results[config_key] = lib_times

    results["cluster"] = cl_results

    headers = ["Config"] + [lib for lib in LIBRARIES if available.get(lib)]
    rows = []
    for config_key, lib_times in cl_results.items():
        row = [config_key]
        for lib in LIBRARIES:
            if not available.get(lib):
                continue
            r = lib_times.get(lib, {})
            if r.get("status") == "ok":
                row.append(format_time(r["median"]))
            elif r.get("status") == "timeout":
                row.append("TIMEOUT")
            else:
                row.append(r.get("status", "—"))
        rows.append(row)
    print_table("k-Medoids / k-Means Clustering", headers, rows)

    # --- Save results ---
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = results_dir / f"cross_library_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {results_path}")

    # --- Plot ---
    plots_dir = Path(__file__).parent / "plots"
    make_plot(results, plots_dir / "cross_library.png")


if __name__ == "__main__":
    main()
