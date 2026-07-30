---
title: Clustering Algorithms
weight: 4
---

# Clustering Algorithms

DTWC++ implements eight CLI-selectable clustering methods (besides `auto`) for
partitioning time series from elastic distances.

## FastPAM (k-Medoids)

The Partitioning Around Medoids (PAM) algorithm is the default clustering method in DTW-C++. Unlike k-means, which uses computed centroids, k-medoids selects actual data points (medoids) as cluster representatives. This is particularly well-suited for DTW-based clustering because computing a meaningful "average" time series under DTW is non-trivial.

### Algorithm Overview

PAM consists of two phases:

1. **BUILD phase:** Select initial medoids (either randomly or via k++ initialization).
2. **SWAP phase:** Iteratively improve the clustering by considering swapping each medoid with each non-medoid and accepting swaps that reduce total cost.

### FastPAM1 Optimization

DTWC++ uses the FastPAM1 decomposition, which evaluates swap candidates with
nearest/second-nearest medoid tracking. Production SWAP work is
$$O(N^2)$$ per iteration rather than the direct-sum $$O(N^2k)$$ reference path.

**CLI:** `dtwc_cl -k 5 --method pam`

> **Reference:** Schubert, E. and Rousseeuw, P. J. (2021). "Fast and eager
> k-medoids clustering: O(k) runtime improvement of the PAM, CLARA, and CLARANS
> algorithms." *Information Systems* 101, 101804.
> <https://doi.org/10.1016/j.is.2021.101804>

## OneBatchPAM

OneBatchPAM forms a fixed objective batch and keeps all $$N$$ points eligible
as medoids, reducing the distance table to $$O(Nm)$$ for batch size $$m$$. The
CLI exposes explicit or logarithmic-auto batch sizes and uniform, debiased, and
nearest-neighbour importance weighting.

**CLI:** `dtwc_cl -k 5 --method onebatch --batch-size 200 --batch-weighting nniw`

## FastCLARA (Scalable k-Medoids)

CLARA (Clustering Large Applications) scales k-medoids to large datasets by running FastPAM on random subsamples of size $$s \ll N$$, then assigning all $$N$$ points to the best medoids found. This avoids computing the full $$O(N^2)$$ distance matrix. The current result ABI uses signed 32-bit medoid indices, so FastCLARA rejects datasets with more than `INT_MAX` series before allocation.

Assignment uses the `Problem`'s configured DTW function directly. A pre-existing
dense or memory-mapped parent distance cache is neither consulted nor modified;
FastCLARA results therefore cannot depend on injected cache contents.

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

## LR-core

LR-core is the in-tree exact p-median route. It combines a Lagrangian root
bound, reduced-cost fixing, and medoid-variable branch-and-bound. It does not
require a commercial MIP solver, but the current clustering entry point
materialises a dense distance matrix and therefore has an $$O(N^2)$$ memory
budget. It either returns a certified solution or fails loudly at its
configured limits.

**CLI:** `dtwc_cl -k 5 --method lrcore`

## TADPole

TADPole implements density-peaks clustering with lower/upper-bound pruning.
For finite, nonempty, equal-length Standard-L1 pairs whose series length is
representable by the integer band API, TADPole can classify some cutoff
comparisons without an exact DTW. In its density stage, `LB >= dc` proves that
a pair is not a neighbour, while `UB < dc` proves that it is; a bound interval
that straddles `dc` triggers exact DTW. In its separation stage, `LB >= best`
proves that a candidate cannot improve the current nearest-higher-density
distance.

The LB_Keogh radius covers the configured fixed DTW radius. Under full DTW,
TADPole constructs a global-minimum/global-maximum envelope; it does not pass
the negative band to the low-level helper. Unequal-length pairs and
unsupported variants take the exact route. Within this finite, nonempty,
integer-representable contract, exact arithmetic makes the bound decisions
admissible and preserves the deterministic brute-force density-peaks result.
The permanent exactly representable regression confirms that regime; it does
not establish bit-level identity when floating reductions straddle `dc` or
`best`. That threshold analysis remains D17. The proof and call-site oracle
are in the
[D2 derivation](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/docs/derivations/02-envelopes-lb-keogh.md).

Empty series are a known exception (F48): exact DTW returns the no-path
sentinel, while the current empty envelope and diagonal upper bound both
return zero. Do not rely on pruned/brute TADPole identity for data containing
empty series until that finding closes.

**CLI:** `dtwc_cl -k 5 --method tadpole --dc 2.0`

## Algorithm Comparison

| Algorithm | Optimality | Memory | Scalability | Best For |
|-----------|-----------|--------|-------------|----------|
| FastPAM | Local optimum | $$O(N^2)$$ distance storage | Dense | General k-medoids |
| OneBatchPAM | Approximate | $$O(Nm)$$ table | Matrix-free | Bounded distance budget |
| FastCLARA | Approximate | $$O(s^2)$$ sample storage plus assignment scratch | Matrix-free for non-full samples | Large datasets |
| Hierarchical | N/A (dendrogram) | $$O(N^2)$$ | Guarded at 2,000 points | Exploratory hierarchy |
| Lloyd's (`kmedoids`) | Local optimum | Uses configured distance storage | Dense | Simple assignment/update |
| MIP | Certified when solved to optimality | $$O(N^2)$$ distance storage plus solver model | Dense | Solver-backed exact result |
| LR-core | Certified or loud failure | $$O(N^2)$$ distance storage | Dense | In-tree exact result |
| TADPole | Exact-arithmetic identity for finite, nonempty supported inputs with integer-representable lengths; exactly representable regression confirmed; floating thresholds remain D17 and empty series F48 | Threshold routing may use mmap | Density/cutoff search | Admissible pair pruning |
