---
layout: default
title: Clustering Algorithms
nav_order: 5
---

# Clustering Algorithms

DTW-C++ provides several clustering algorithms that operate on a pairwise distance matrix $$D^{p \times p}$$, where $$d_{i,j}$$ is the DTW distance between time series $$i$$ and $$j$$. All algorithms produce $$k$$ clusters, each represented by a **medoid** -- an actual time series from the dataset that minimises the total within-cluster dissimilarity.

## Lloyd k-medoids iteration

The Lloyd-style k-medoids algorithm (sometimes called "alternating" k-medoids) is the simplest iterative approach. It alternates between two steps:

1. **Assignment step.** Assign each time series to the cluster whose medoid is nearest:

   $$
   c(i) = \arg\min_{m \in \mathcal{M}} \; d_{i,m}
   $$

   where $$\mathcal{M}$$ is the current set of $$k$$ medoids.

2. **Update step.** Within each cluster, select the time series that minimises the total distance to all other members:

   $$
   m^* = \arg\min_{j \in C_\ell} \sum_{i \in C_\ell} d_{i,j}
   $$

   where $$C_\ell$$ is the set of time series assigned to cluster $$\ell$$.

These two steps repeat until the medoids no longer change or a maximum number of iterations is reached.

**Complexity.** Each iteration costs $$O(N^2 k)$$ for the assignment step and $$O(N^2)$$ for the update step, where $$N = p$$ is the number of time series.

**Limitations.** Lloyd iteration is simple and fast per iteration, but it is more susceptible to getting stuck in local optima compared to PAM. The quality of the final clustering depends heavily on the initial medoid selection (random or k++ initialisation).

In DTW-C++, this algorithm is implemented as `cluster_by_kMedoidsLloyd`.

## PAM SWAP (FastPAM)

The Partitioning Around Medoids (PAM) algorithm provides higher-quality clusterings than Lloyd iteration by considering all possible swaps between current medoids and non-medoid points. DTW-C++ implements the **FastPAM** variant from Schubert & Rousseeuw (2021), which dramatically reduces the computational cost of the original PAM algorithm.

### Algorithm overview

For each candidate swap of medoid $$m \in \mathcal{M}$$ with non-medoid $$x \notin \mathcal{M}$$, FastPAM computes the change in total cost $$\Delta T_{mx}$$ using nearest and second-nearest distance tracking:

* For each point $$o$$ in the dataset, maintain:
  - $$d_{\text{nearest}}(o)$$: distance to the closest medoid
  - $$d_{\text{second}}(o)$$: distance to the second-closest medoid

* The cost change of swapping $$m$$ for $$x$$ is computed by examining how each point $$o$$ is affected:
  - If $$o$$ is currently assigned to $$m$$ (the medoid being removed), it will either move to $$x$$ or to its second-nearest medoid.
  - If $$o$$ is not assigned to $$m$$, it may switch to $$x$$ if $$x$$ is closer.

The swap with the most negative $$\Delta T_{mx}$$ is performed. The algorithm terminates when no swap produces a negative cost change.

**Complexity.** Each swap evaluation costs $$O(N k)$$ rather than $$O(N^2)$$ as in the original PAM, thanks to the nearest/second-nearest caching. The total number of candidate swaps per iteration is $$O(k(N-k))$$.

**Advantages over Lloyd iteration:**
* Converges to better local optima because it evaluates the global effect of each swap
* More robust to initialisation
* Deterministic convergence (no oscillation)

In DTW-C++, this algorithm is available as the `fast_pam()` free function.

**Reference:** Schubert, E. & Rousseeuw, P.J. (2021). Fast and eager k-medoids clustering: O(k) runtime improvement of the PAM, CLARA, and CLARANS algorithms. *Information Systems*, 101, 101804.

## CLARA (planned)

CLARA (Clustering LARge Applications) is a sampling-based extension of PAM designed for large datasets where computing the full $$N \times N$$ distance matrix is impractical.

The algorithm works as follows:

1. Draw a random sample $$S$$ of size $$s \ll N$$ from the dataset.
2. Run PAM on the sample to find $$k$$ medoids.
3. Assign all remaining points to their nearest sample medoid.
4. Compute the total cost over the full dataset.
5. Repeat steps 1--4 multiple times and keep the result with the lowest total cost.

**Complexity.** Each iteration costs $$O(s^2 k)$$ for PAM on the sample plus $$O(Nk)$$ for the full assignment, making CLARA suitable for datasets with $$N$$ in the tens of thousands or more.

CLARA is planned for a future release of DTW-C++.
