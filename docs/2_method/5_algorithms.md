---
layout: default
title: Clustering Algorithms
nav_order: 5
---

# Clustering Algorithms

DTW-C++ implements several clustering algorithms for partitioning time series into groups based on DTW distance similarity.

## k-Medoids (PAM)

The Partitioning Around Medoids (PAM) algorithm is the default clustering method in DTW-C++. Unlike k-means, which uses computed centroids, k-medoids selects actual data points (medoids) as cluster representatives. This is particularly well-suited for DTW-based clustering because computing a meaningful "average" time series under DTW is non-trivial.

### Algorithm Overview

PAM consists of two phases:

1. **BUILD phase:** Select initial medoids (either randomly or via k++ initialization).
2. **SWAP phase:** Iteratively improve the clustering by considering swapping each medoid with each non-medoid and accepting swaps that reduce total cost.

### Complexity

- **Assignment step:** $$O(Nk)$$ per iteration, where $$N$$ is the number of time series and $$k$$ is the number of clusters. Each point is compared to all $$k$$ medoids.
- **Update (swap) step:** $$O(N^2)$$ per iteration for evaluating all candidate swaps.

### FastPAM

DTW-C++ aims to incorporate the FastPAM optimization, which reduces the constant factor of the swap evaluation.

FastPAM1 achieves $$O(N^2 k)$$ per iteration (an $$O(k)$$ speedup over naive PAM's $$O(N^2 k^2)$$) by evaluating all swap candidates simultaneously using nearest/second-nearest medoid tracking. The current implementation matches this $$O(N^2 k)$$ complexity. A further optimization using a shared accumulator could reduce the inner loop, but this is not yet implemented.

> **Reference:** Schubert, E. and Rousseeuw, P. J. (2021). "Fast and Eager k-Medoids Clustering: O(k) Runtime Improvement of the PAM, CLARA, and CLARANS Algorithms." *Journal of Machine Learning Research (JMLR)*, 22(1), 4653-4688.

## Lloyd's Algorithm (k-Means style)

Lloyd's algorithm is the iterative assignment-update approach commonly associated with k-means clustering. When adapted for k-medoids:

1. **Assignment:** Assign each time series to its nearest medoid. Complexity: $$O(Nk)$$.
2. **Update:** For each cluster, find the point that minimizes total within-cluster distance. Complexity: $$O(N^2)$$ in the worst case (evaluating all points as potential medoids within each cluster).

## Mixed-Integer Programming (MIP)

DTW-C++ also supports solving the k-medoids problem exactly via mixed-integer programming formulations using Gurobi or HiGHS solvers. This approach finds the globally optimal solution but is computationally expensive for large datasets.

## Future Algorithms

The following algorithms are planned for future implementation:

- **CLARA:** Clustering Large Applications -- applies PAM to random subsamples for scalability.
- **CLARANS:** Randomized neighborhood search variant of PAM.
- **Hierarchical clustering:** Agglomerative clustering with various linkage criteria (single, complete, average, Ward).
