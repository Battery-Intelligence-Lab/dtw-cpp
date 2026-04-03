---
title: Clustering Algorithms
weight: 4
---

# Clustering Algorithms

DTW-C++ implements several clustering algorithms for partitioning time series into groups based on DTW distance similarity.

## FastPAM (k-Medoids)

The Partitioning Around Medoids (PAM) algorithm is the default clustering method in DTW-C++. Unlike k-means, which uses computed centroids, k-medoids selects actual data points (medoids) as cluster representatives. This is particularly well-suited for DTW-based clustering because computing a meaningful "average" time series under DTW is non-trivial.

### Algorithm Overview

PAM consists of two phases:

1. **BUILD phase:** Select initial medoids (either randomly or via k++ initialization).
2. **SWAP phase:** Iteratively improve the clustering by considering swapping each medoid with each non-medoid and accepting swaps that reduce total cost.

### FastPAM1 Optimization

DTW-C++ uses the FastPAM1 optimization, which evaluates all swap candidates simultaneously using nearest/second-nearest medoid tracking. This reduces complexity from $$O(N^2 k^2)$$ to $$O(N^2 k)$$ per iteration — an $$O(k)$$ speedup.

**CLI:** `dtwc_cl -k 5 --method pam`

> **Reference:** Schubert, E. and Rousseeuw, P. J. (2021). "Fast and Eager k-Medoids Clustering: O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms." *Journal of Machine Learning Research (JMLR)*, 22(1), 4653-4688.

## FastCLARA (Scalable k-Medoids)

CLARA (Clustering Large Applications) scales k-medoids to large datasets by running FastPAM on random subsamples of size $$s \ll N$$, then assigning all $$N$$ points to the best medoids found. This avoids computing the full $$O(N^2)$$ distance matrix.

### How It Works

1. Draw a random subsample of size $$s$$ from the dataset.
2. Run FastPAM on the subsample to find $$k$$ medoids.
3. Assign all $$N$$ points to the nearest medoid (requires only $$N \times k$$ DTW computations).
4. Repeat for `n_samples` independent subsamples.
5. Return the result with the lowest total cost.

### Default Sample Size

When `sample_size = -1` (auto), the size is computed as:

$$s = \max(40 + 2k, \min(N, 10k + 100))$$

This follows the recommendation from Schubert & Rousseeuw (2021).

### Parameters

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `sample_size` | `--sample-size` | -1 (auto) | Subsample size per iteration |
| `n_samples` | `--n-samples` | 5 | Number of independent subsamples |
| `max_iter` | `--max-iter` | 100 | Max FastPAM iterations per subsample |
| `random_seed` | `--seed` | 42 | RNG seed for reproducibility |

**CLI:** `dtwc_cl -k 5 --method clara --sample-size 200 --n-samples 10`

> **Reference:** Kaufman, L. & Rousseeuw, P.J. (1990). "Finding Groups in Data." Wiley Series in Probability and Statistics.

## Hierarchical Agglomerative Clustering

Hierarchical clustering builds a dendrogram by iteratively merging the closest pair of clusters until a single cluster remains. You can then cut the dendrogram at any level to obtain $$k$$ flat clusters.

### Linkage Criteria

| Linkage | Formula | Description |
|---------|---------|-------------|
| Single | $$d(A \cup B, C) = \min(d(A,C), d(B,C))$$ | Minimum distance between clusters. Can produce elongated chains. |
| Complete | $$d(A \cup B, C) = \max(d(A,C), d(B,C))$$ | Maximum distance. Produces compact clusters. |
| Average (UPGMA) | $$d(A \cup B, C) = \frac{|A| \cdot d(A,C) + |B| \cdot d(B,C)}{|A| + |B|}$$ | Weighted average. Good general-purpose choice. |

```warning
Ward's linkage is intentionally excluded. Ward's formula requires squared Euclidean distances, which DTW does not satisfy. Using Ward's with DTW produces mathematically invalid results.
```

### Usage

```cpp
#include <dtwc/algorithms/hierarchical.hpp>

dtwc::algorithms::HierarchicalOptions opts;
opts.linkage = dtwc::algorithms::Linkage::Average;

auto dendrogram = dtwc::algorithms::build_dendrogram(prob, opts);
auto result = dtwc::algorithms::cut_dendrogram(dendrogram, k);
```

**CLI:** `dtwc_cl -k 5 --method hierarchical --linkage average`

```note
Hierarchical clustering requires the full distance matrix and has $$O(N^2)$$ memory complexity. A hard guard of `max_points = 2000` prevents accidental out-of-memory errors on large datasets. For larger datasets, use FastCLARA instead.
```

## CLARANS (Experimental)

CLARANS (Clustering Large Applications based on RANdomized Search) is a randomized variant of k-medoids that explores random neighbors instead of evaluating all possible swaps. It is more scalable than PAM for very large datasets but may not find the global optimum.

### Budget Controls

CLARANS uses budget controls to limit computation:

| Parameter | Description |
|-----------|-------------|
| `max_dtw_evals` | Maximum total DTW distance evaluations |
| `max_neighbor` | Maximum neighbors to explore per iteration |

```warning
CLARANS is currently experimental and not exposed in the CLI. It requires benchmark evidence before promotion to a production algorithm. Use FastCLARA for scalable clustering.
```

## Lloyd's Algorithm (k-Means Style)

Lloyd's algorithm is the iterative assignment-update approach commonly associated with k-means clustering. When adapted for k-medoids:

1. **Assignment:** Assign each time series to its nearest medoid. Complexity: $$O(Nk)$$.
2. **Update:** For each cluster, find the point that minimizes total within-cluster distance. Complexity: $$O(N^2)$$ in the worst case.

This is simpler than PAM but converges to a local minimum. FastPAM is generally preferred.

## Mixed-Integer Programming (MIP)

DTW-C++ supports solving the k-medoids problem exactly via mixed-integer programming using Gurobi or HiGHS solvers. This finds the globally optimal solution but is computationally expensive for large datasets.

### MIP Warm Start

By default, FastPAM is run first and its solution is fed to the MIP solver as a warm start. This dramatically reduces branch-and-bound solve time. Disable with `--no-warm-start`.

### Benders Decomposition

For large datasets ($$N > 200$$), Benders decomposition splits the problem into a master problem (medoid selection, $$N$$ binary variables) and an assignment subproblem. This is enabled automatically or can be controlled with `--benders auto|on|off`.

### MIP Solver Settings

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Solver | `--solver` | highs | `highs` (open-source) or `gurobi` (commercial) |
| MIP gap | `--mip-gap` | 1e-5 | Optimality gap tolerance |
| Time limit | `--time-limit` | -1 (unlimited) | Seconds |
| Warm start | `--no-warm-start` | enabled | Disable FastPAM warm start |
| Benders | `--benders` | auto | Benders decomposition: auto, on, off |

**CLI:** `dtwc_cl -k 5 --method mip --solver gurobi --mip-gap 1e-4`

## Algorithm Comparison

| Algorithm | Optimality | Memory | Scalability | Best For |
|-----------|-----------|--------|-------------|----------|
| FastPAM | Local optimum | $$O(N^2)$$ | Medium (N < 10k) | Default choice |
| FastCLARA | Approximate | $$O(s^2)$$ | Large (any N) | Large datasets |
| Hierarchical | N/A (dendrogram) | $$O(N^2)$$ | Small (N < 2000) | Exploratory analysis |
| MIP | Global optimum | $$O(N^2)$$ | Small (N < 500) | When optimality matters |
| Lloyd's | Local optimum | $$O(N^2)$$ | Medium | Legacy, simple |
