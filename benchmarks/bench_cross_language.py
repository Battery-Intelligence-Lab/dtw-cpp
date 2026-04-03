"""Cross-language benchmark: Python bindings vs C++ baseline.

Run with: uv run benchmarks/bench_cross_language.py
"""
import time
import numpy as np


def bench_dtw_distance(N_pairs, L):
    """Benchmark pairwise DTW distance computation."""
    import dtwcpp

    rng = np.random.default_rng(42)
    series_a = [rng.standard_normal(L) for _ in range(N_pairs)]
    series_b = [rng.standard_normal(L) for _ in range(N_pairs)]

    # Warm up
    dtwcpp.dtw_distance(series_a[0], series_b[0])

    start = time.perf_counter()
    for i in range(N_pairs):
        dtwcpp.dtw_distance(series_a[i], series_b[i])
    elapsed = time.perf_counter() - start

    return elapsed


def bench_distance_matrix(N, L, band=-1):
    """Benchmark full NxN distance matrix computation."""
    import dtwcpp

    rng = np.random.default_rng(42)
    series = [rng.standard_normal(L).tolist() for _ in range(N)]

    # Warm up
    if N > 5:
        dtwcpp.compute_distance_matrix(series[:5], band=band)

    start = time.perf_counter()
    D = dtwcpp.compute_distance_matrix(series, band=band)
    elapsed = time.perf_counter() - start

    return elapsed, D.shape[0]


def bench_clustering(N, L, k, band=-1):
    """Benchmark FastPAM clustering end-to-end."""
    import dtwcpp

    rng = np.random.default_rng(42)
    data = rng.standard_normal((N, L))

    prob = dtwcpp.Problem("bench")
    series = [data[i].tolist() for i in range(N)]
    names = [f"s{i}" for i in range(N)]
    prob.set_data(series, names)
    prob.band = band

    start = time.perf_counter()
    result = dtwcpp.fast_pam(prob, k)
    elapsed = time.perf_counter() - start

    return elapsed, result.total_cost


def main():
    print("=" * 65)
    print("DTWC++ Python Benchmark")
    print("=" * 65)

    # DTW distance: 100 pairs
    for L in [100, 500, 1000]:
        t = bench_dtw_distance(100, L)
        print(f"  DTW distance  100 pairs x L={L:4d}: {t*1000:8.2f} ms  ({t/100*1000:.3f} ms/pair)")

    print()

    # Distance matrix
    for N, L in [(20, 100), (50, 100), (50, 500), (100, 500)]:
        t, n = bench_distance_matrix(N, L)
        pairs = n * (n - 1) // 2
        print(f"  Dist matrix   {N:3d}x{L:4d} ({pairs:6d} pairs): {t*1000:8.2f} ms")

    print()

    # Clustering (FastPAM)
    for N, L, k in [(20, 100, 3), (50, 100, 5), (50, 500, 5)]:
        t, cost = bench_clustering(N, L, k)
        print(f"  FastPAM       {N:3d}x{L:4d} k={k}: {t*1000:8.2f} ms  (cost={cost:.2f})")

    print()
    print("=" * 65)


if __name__ == "__main__":
    main()
