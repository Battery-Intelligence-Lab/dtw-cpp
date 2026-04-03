"""
@file 06_distance_matrix.py
@brief DTWC++ Distance Matrix — Fast pairwise DTW computation in C++.
@details
Demonstrates: compute_distance_matrix() with OpenMP parallelism,
metric options, and the Problem.distance_matrix_numpy() method.

A single C++ call computes the full NxN symmetric distance matrix
using OpenMP — much faster than a Python loop over dtw_distance().
@author Volkan Kumtepeli
"""

import numpy as np
import dtwcpp
import time

# --- 1. Basic distance matrix ---
rng = np.random.default_rng(42)
n_series = 20
length = 100
data = [list(rng.uniform(-1, 1, length)) for _ in range(n_series)]

dm = dtwcpp.compute_distance_matrix(data)
print(f"Distance matrix shape: {dm.shape}")
print(f"Min non-zero: {dm[dm > 0].min():.4f}")
print(f"Max:          {dm.max():.4f}")
print(f"Symmetric:    {np.allclose(dm, dm.T)}")
print(f"Diagonal zero: {np.allclose(np.diag(dm), 0)}")
print()

# --- 2. Banded DTW — faster for long series ---
dm_banded = dtwcpp.compute_distance_matrix(data, band=10)
print(f"Banded (band=10) max: {dm_banded.max():.4f}")
print(f"Speedup note: banded DTW is O(n*band) vs O(n^2) for full DTW")
print()

# --- 3. Squared Euclidean metric ---
dm_sq = dtwcpp.compute_distance_matrix(data, metric="squared_euclidean")
print(f"Squared Euclidean max: {dm_sq.max():.4f}")
print()

# --- 4. Timing: C++ vs Python loop ---
n_bench = 50
bench_data = [list(rng.uniform(-1, 1, length)) for _ in range(n_bench)]

t0 = time.perf_counter()
dm_cpp = dtwcpp.compute_distance_matrix(bench_data)
t_cpp = time.perf_counter() - t0

t0 = time.perf_counter()
dm_py = np.zeros((n_bench, n_bench))
for i in range(n_bench):
    for j in range(i + 1, n_bench):
        d = dtwcpp.dtw_distance(bench_data[i], bench_data[j])
        dm_py[i, j] = d
        dm_py[j, i] = d
t_py = time.perf_counter() - t0

print(f"Timing ({n_bench} series, length {length}):")
print(f"  C++ compute_distance_matrix: {t_cpp:.3f}s")
print(f"  Python loop:                 {t_py:.3f}s")
print(f"  Speedup:                     {t_py / t_cpp:.1f}x")
print(f"  Results match:               {np.allclose(dm_cpp, dm_py)}")

# --- 5. Via Problem class ---
prob = dtwcpp.Problem("dm_example")
prob.set_data(data, [f"s{i}" for i in range(n_series)])
prob.band = -1
dm_prob = prob.distance_matrix_numpy()
print(f"\nProblem.distance_matrix_numpy() shape: {dm_prob.shape}")
