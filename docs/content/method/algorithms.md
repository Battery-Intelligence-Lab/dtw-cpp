---
title: Clustering Algorithms
weight: 4
aliases:
  - /method/mip/
  - /method/k-medoids/
---

# Clustering Algorithms

DTWC++ implements eight CLI-selectable clustering methods (besides `auto`) for
partitioning time series from elastic distances, plus sequence-valued DTW
barycentres and barycentre k-means.

## FastPAM (k-Medoids)

The Partitioning Around Medoids (PAM) algorithm is what the default `--method auto` runs for up to 5000 series on the CPU, and for any number on a GPU; above 5000 series on the CPU, or where PAM's distance matrix can be neither held in RAM nor mapped to a file, `auto` runs FastCLARA. Unlike k-means, which uses computed centroids, k-medoids selects actual data points (medoids) as cluster representatives.

### Algorithm Overview

PAM consists of two phases:

1. **BUILD phase:** Select initial medoids by k-medoids++: the first at random, each next drawn with probability proportional to its distance to the nearest medoid already chosen, from the `--seed` generator.
2. **SWAP phase:** Iteratively improve the clustering by considering swapping each medoid with each non-medoid and accepting swaps that reduce total cost.

### FasterPAM

DTWC++ runs the eager FasterPAM SWAP: nearest/second-nearest medoid tracking
evaluates one candidate against every medoid in $O(N)$, so a sweep costs
$O(N^2)$, and each improving swap is taken as soon as it is found, so few
sweeps are needed even at large $k$.

**CLI:** `dtwc_cl -i data.csv -k 5 --method pam`

> **Reference:** Schubert, E. and Rousseeuw, P. J. (2021). "Fast and eager
> k-medoids clustering: O(k) runtime improvement of the PAM, CLARA, and CLARANS
> algorithms." *Information Systems* 101, 101804.
> <https://doi.org/10.1016/j.is.2021.101804>

## OneBatchPAM

OneBatchPAM draws one fixed evaluation batch $B$ of size $m$ without replacement and keeps all $N$ series eligible as candidate medoids, reducing the distance table and memory to $O(Nm)$. When `batch_size = -1` (auto), it picks:

$$m = \min\!\big(N,\; \max\big(64,\; 20\lceil\log_2(N + 1)\rceil\big)\big),$$

requiring $m \ge k$ when an explicit positive batch size is supplied. Table construction runs in parallel across rows using packed $W$-lane SIMD panels (pre-packed up to 64 MiB when series reside in RAM without `--ram-limit`).

Following the `obpam` hybrid estimator of de Mathelin et al. (AAAI 2025) and Loog (2012) nearest-neighbour importance weighting (NNIW), finite raw distances $d_{i,j}$ are divided by the finite table maximum $S$ (or $1$ if zero), a sampled point's self-entry is replaced by normalised $1$, and each batch column $j \in B$ is weighted by $w_j = |V_j| / (N/m)$, where $|V_j|$ is the count of series nearest to $j$. Final assignment to out-of-batch medoids uses packed SIMD lanes where supported.

**CLI:** `dtwc_cl -i data.csv -k 5 --method onebatch --batch-size 200`

## FastCLARA (Scalable k-Medoids)

CLARA (Clustering Large Applications) scales k-medoids to large datasets by running FastPAM on random subsamples of size $s \ll N$, then assigning all $N$ points to the best medoids found. This avoids computing the full $O(N^2)$ distance matrix.

Assignment uses the `Problem`'s configured DTW function directly (packed CPU SIMD lanes or `cuda::compute_medoid_argmin_cuda` on CUDA). A pre-existing dense or memory-mapped parent distance cache is neither consulted nor modified; FastCLARA results therefore cannot depend on injected cache contents. During the assignment pass, per-sample labels use compact `uint8_t` ($k \le 256$) or `uint16_t` ($k \le 65536$) storage before copying the winning subsample's labels.

In `dtwc_cl`, FastCLARA is also the one method that streams a Parquet or Arrow IPC file's series in chunks when they do not fit the memory budget (`--ram-limit`, the run's soft budget: 0, the default, plans for 90 % of the memory available); every other method that cannot fit under an explicit limit is refused with an error. The limit, the memory plan and the streaming rules are on the [CLI page](../../getting-started/cli/).

### How It Works

1. Draw `n_samples` random subsamples of size $s$ from the dataset, one after another from one seeded generator.
2. Run FastPAM on each subsample to find its $k$ medoids.
3. Assign all $N$ points to each subsample's nearest medoid in blocks. On CPU, unique medoids are evaluated in one pass against packed point panels; on CUDA (`gpu` or `cpu+gpu` share), `compute_medoid_argmin_cuda` deduplicates unique medoids across all subsamples and reduces `(best_label, best_dist)` directly on the GPU.
4. Return the subsample with the lowest total cost (the first, on a tie).

### Default Sample Size

When `sample_size = -1` (auto), the size is computed as:

$$s = \min(N, 10k + 100)$$

The size is then clamped to $[k, N]$, following Schubert & Rousseeuw (2021).

### Parameters

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| `sample_size` | `--sample-size` | -1 (auto) | Subsample size per iteration |
| `n_samples` | `--n-samples` | 5 | Number of independent subsamples |
| `max_iter` | `--max-iter` | 100 | Max FastPAM iterations per subsample |
| `random_seed` | `--seed` | 42 | RNG seed for reproducibility |

**CLI:** `dtwc_cl -i data.csv -k 5 --method clara --sample-size 200 --n-samples 10`

> **Reference:** Kaufman, L. & Rousseeuw, P.J. (1990). "Finding Groups in Data." Wiley Series in Probability and Statistics.

## Hierarchical Agglomerative Clustering

Hierarchical clustering builds a dendrogram by iteratively merging the closest pair of clusters until a single cluster remains. You can then cut the dendrogram at any level to obtain $k$ flat clusters.

DTWC++ copies the packed lower-triangular distance matrix into a packed `work` buffer (half the memory of a dense $N \times N$ copy) and accelerates merges with a Day & Edelsbrunner (1984) per-row nearest-neighbour cache (`nn_dist`, `nn_idx`), giving $O(N^2)$ typical complexity ($O(N^3)$ worst-case when complete or average linkage increases distances and triggers row rescans), breaking ties deterministically on the lexicographically smallest $(a, b)$ with $a < b$.

### Linkage Criteria

| Linkage | Formula | Description |
|---------|---------|-------------|
| Single | $d(A \cup B, C) = \min(d(A,C), d(B,C))$ | Minimum distance between clusters. Can produce elongated chains. |
| Complete | $d(A \cup B, C) = \max(d(A,C), d(B,C))$ | Maximum distance. Produces compact clusters. |
| Average (UPGMA) | $d(A \cup B, C) = \frac{|A| \cdot d(A,C) + |B| \cdot d(B,C)}{|A| + |B|}$ | Weighted average. Good general-purpose choice. |

> **Warning:** Ward's linkage is intentionally excluded. Ward's formula requires squared Euclidean distances, which DTW does not satisfy. Using Ward's with DTW produces mathematically invalid results.

### Usage

```cpp
#include <algorithms/hierarchical.hpp>

dtwc::algorithms::HierarchicalOptions opts;
opts.linkage = dtwc::algorithms::Linkage::Average;

auto dendrogram = dtwc::algorithms::build_dendrogram(prob, opts);
auto result = dtwc::algorithms::cut_dendrogram(dendrogram, prob, k);
```

**CLI:** `dtwc_cl -i data.csv -k 5 --method hierarchical --linkage average`

> **Note:** Hierarchical clustering requires the full distance matrix plus a packed working copy ($O(N^2)$ memory). A hard guard of `max_points = 2000` prevents accidental out-of-memory errors on large datasets. For larger datasets, use FastCLARA instead.

## Lloyd's Algorithm (k-Means Style)

Lloyd's algorithm alternates assignment and medoid updates (`--method kmedoids`):

1. **Initialisation:** Pick $k$ distinct series at random from the `--seed` generator. In C++, setting `Problem`'s `init_fun` to `init::Kmeanspp` uses k-medoids++ instead, reducing poor local optima.
2. **Assignment:** Assign each time series to its nearest medoid ($O(Nk)$).
3. **Update (`calculate_medoids`):** Group series by assigned cluster and evaluate pairwise sums within each cluster in deterministic sequential index order ($O(\sum_c |C_c|^2)$ per sweep instead of scanning all $N^2$ pairs).
4. **Repeat** until medoids stop changing or `--max-iter` iterations complete. With `--n-init` restarts, the lowest-cost clustering is kept.

This is simpler than PAM but converges to a local minimum. FastPAM is generally preferred.

## Mixed-Integer Programming (MIP)

DTW-C++ solves the discrete $p$-median problem to global optimality using HiGHS (1.16.0) or Gurobi. In Python, the HiGHS solve runs via `highspy`, a dependency of `dtwcpp`; wheels are built without Gurobi.

### MIP Formulation

The model is Balinski's disaggregated $p$-median program over the symmetric distance matrix $D$ of $N$ series, built directly in CSC column order with $N$ binary diagonal medoid indicators $y_i = x_{ii} \in \{0,1\}$ and $N(N-1)$ continuous assignment variables $x_{ij} \in [0,1]$ ($N^2$ columns total). It minimises $\sum_i \sum_j D_{ij}\, x_{ij}$ subject to:

1. exactly $k$ medoids: $\sum_{i=1}^{N} x_{ii} = k$;
2. single assignment per series: $\sum_{i=1}^{N} x_{ij} = 1$ for all $j$;
3. linking constraints: $x_{ij} \le x_{ii}$ for all $i \ne j$.

Because all $D_{ij} \ge 0$ and the variables are built directly as $N$ binary + $N(N-1)$ continuous columns, LP relaxation vertices assign each series wholly to an open medoid; `decode_assignment_nearest` verifies primal feasibility (`1e-4` tolerance), extracts the $k$ open medoids from the diagonal, and assigns each series to its nearest open medoid. In the example below, the clusters are series 1, **2**, 5 and 3, **4** (medoids in bold):

<p align="center"><img src="../cluster_matrix_formation4.svg" alt="A 5 by 5 binary matrix: row 2 has ones in columns 1, 2 and 5, row 4 has ones in columns 3 and 4, and every other entry is 0" width="50%"/></p>

### Warm Start, Scaling, Shortcuts, and HiGHS Configuration

- **Trivial shortcuts:** $k = 1$ dispatches directly to `fast_pam_solve` with zero swaps; $k = N$ opens every point at cost 0; a warm start with zero objective decodes immediately without invoking the solver.
- **MIP warm start & scaling:** By default, FastPAM runs first and supplies an initial solution (`--no-warm-start` disables it). Objective costs are divided by `scaling_factor = min(max_distance / 2, pam_cost)` when a positive-cost warm start is used (else `max_distance / 2`, or `1.0` if `max_distance == 0`).
- **HiGHS 1.16 serial settings:** DTWC++ sets `mip_rel_gap = mip_gap`, `mip_abs_gap = 0.0`, `parallel = "off"`, and `threads = 1` (resetting the global HiGHS thread scheduler around `run()`) so equal-cost medoid ties are deterministic across machines and root solves avoid multi-thread contention.
- **Model size limit:** Because CSC indices use 32-bit `int` (roughly $3N^2$ nonzeros), instances exceeding `INT_MAX` nonzeros ($N \approx 26{,}750$) raise `SolverError` and direct callers to `LRCore`.

### Large N & Distance Scale

Because the compact model creates $N^2$ columns on the selected solver at every $N$, large instances should use [LR-core](#lr-core) (`--method lrcore`), which stays in $y$-space without building the $N^2$-column LP. A `Problem` with no data, or $k \notin [1, N]$, raises `InvalidInput`.

FastPAM, MIP, and LR-core measure gains and gaps relative to costs. Because the MIP divides costs so the largest scaled coefficient is at most 2 and then relies on solver tolerances, datasets whose clustering-decisive distances sit many orders of magnitude below $\max D_{ij}$ (such as unequal-length WDTW) can cause the MIP solver to misclassify suboptimal solutions as optimal; LR-core handles those reliably until floating-point cancellation reaches its rounding floor $64\,\varepsilon N \sum_j |\mu_j|$ (roughly $10^{-14}$ of the largest entry on 14–22 series).

### MIP Solver Settings

| Parameter | CLI Flag | Default | Description |
|-----------|----------|---------|-------------|
| Solver | `--solver` | highs | `highs` (open-source) or `gurobi` (commercial) |
| MIP gap | `--mip-gap` | 1e-5 | Relative optimality gap tolerance (`mip_rel_gap`) |
| Time limit | `--time-limit` | -1 (unlimited) | Seconds |
| Warm start | `--no-warm-start` | enabled | Disable FastPAM warm start |

**CLI:** `dtwc_cl -i data.csv -k 5 --method mip --solver gurobi --mip-gap 1e-4`

## LR-core

LR-core is the in-tree exact $p$-median route ([mathematical derivation](../../math/lr-core/)). It combines an Erlenkotter second-nearest Lagrangian root bound, Volume-style exponential primal averages, Beasley reduced-cost fixing, and zero-allocation branch-and-bound with precomputed suffix minima and Erlenkotter client-floor lifts. It requires no external MIP solver and operates zero-copy on packed or memory-mapped `.dtwm` distance matrices. Runs either return a certified optimum or throw `SolverError` when `--lr-max-nodes` (default `2000000`) is reached.

**CLI:** `dtwc_cl -i data.csv -k 5 --method lrcore`

## TADPole

TADPole implements density-peaks clustering (Rodriguez & Laio 2014; Begum et al. 2015) with admissible upper/lower-bound pruning.

For each point $i$ against cutoff radius $d_c > 0$, under the deterministic strict density order $a \succ_{\rho} b \iff (\rho_a > \rho_b) \lor (\rho_a = \rho_b \land a < b)$:

$$\rho_i = |\{j \ne i : d(i, j) < d_c\}|, \qquad \delta_i = \min_{j :\, j \succ_{\rho} i} d(i, j) \quad (\delta_{\text{densest}} = \max_{j \ne \text{densest}} \delta_j), \qquad \gamma_i = \rho_i \cdot \delta_i.$$

The top-$k$ points by $\gamma_i$ become cluster centres; every remaining point inherits its nearest higher-density neighbour's label in a single descending-$\rho$ pass. When `dc` is omitted in `dtwc_cl` or `tadpole_auto_dc`, DTWC++ takes a stratified subsample of $\min(N, 64)$ evenly spaced indices across $[0, N)$ and selects the 2nd percentile of their pairwise distances (or the smallest positive distance on ties).

When no distance matrix is pre-filled, TADPole does **not** allocate a full dense matrix: Stage 1 stores only unevaluated-by-bounds exact pairs in sparse sorted per-row vectors (`upper_cache[i]` and mirrored `lower_cache[j]`) plus up to 16 tight diagonal upper-bound seeds (`diagonal_ub_l1 < dc`) per row (`upper_ub` / `lower_ub`) to seed Stage 2's `best` bound. Note that `dtwc_cl`'s memory planner conservatively reserves worst-case pair flags/cache memory in RAM and never memory-maps TADPole caches.

Pruning activates for finite, nonempty, equal-length univariate Standard-L1 pairs with `float64` series/matrix, no missing-data strategy, and integer-representable lengths (other pairs fall back cleanly to exact evaluation):
- Stage 1 checks the $O(1)$ two-endpoint bound `lb_kim_l1 >= dc`, then cascading two-sided `LB_Keogh` (`lb_keogh_at_least >= dc`), then diagonal `UB < dc`; only straddling pairs run exact DTW.
- Stage 2 seeds `best` from Stage 1's exact cache and `UB` seeds, then prunes candidates with `lb_kim_l1` and cascading `lb_keogh_at_least` / `lb_keogh_greater`.

Under full DTW (`band == -1`), TADPole constructs global min/max envelopes (`env_band = series.size()`). The proof and call-site oracle are in the [D2 derivation](https://github.com/Battery-Intelligence-Lab/dtw-cpp/blob/main/docs/derivations/02-envelopes-lb-keogh.md).

**CLI:** `dtwc_cl -i data.csv -k 5 --method tadpole --dc 2.0`

## DTW Barycentres & Barycentre k-Means

Unlike k-medoids, which restricts centres to input series, `dtw_barycenter` and `barycenter_kmeans` compute sequence-valued average shapes (barycentres) using Standard, unbanded DTW with **squared** local cost $(x_i - y_j)^2$. Calling them on a `Problem` configured with any other DTW variant or a finite band throws `InvalidInput`.

### Averaging Methods (`BarycenterMethod`)

1. **`SSG` (default; Schultz & Jain 2018):** Stochastic subgradient epochs over shuffled cluster members. For squared DTW along an optimal alignment $(V, W)$ with coordinate valence $V_{ii}$ (number of sample points matched to $z_i$), the component gradient is $2(Vz - Wx)$; DTWC++ steps along the raw direction with decayed rate $\eta_t = \eta_0 / (1 + \lambda_{\mathrm{decay}}\, t)$ capped at $1 / (2 \max_i V_{ii})$ (inverse gradient-Lipschitz constant of the fixed-path quadratic) to maintain stability on unequal-length series.
2. **`DBA` (Petitjean et al. 2011):** Majorise-minimise coordinate averaging along optimal squared-DTW alignments, updating each coordinate $z_i \leftarrow \frac{1}{|A_i|} \sum_{(i,j)\in A} y_j$.
3. **`SoftDTW` (Cuturi & Blondel 2017):** Exact forward/adjoint gradient descent on squared-cost Soft-DTW (`gamma`, `learning_rate`) with up to 16 backtracking halvings requiring objective decrease before accepting a step.

### `barycenter_kmeans`

Lloyd-style clustering over sequence-valued barycentres:
- Seeds $k$ initial centres via squared-DTW $D^2$ (k-means++) sampling, linearly resampling each chosen series to `target_length` (`-1` keeps the sampled series' length).
- Assigns every series to its nearest centre using OpenMP and SIMD `PackedLanesPanel` blocks (`SquaredL2`, unbanded).
- Repairs any empty cluster deterministically in cluster order using the currently worst-represented point (preserving $k$ centres), then updates each non-empty cluster centre via `compute_barycenter` (parallelised across centres when total DP scratch fits the ~64 MiB budget).

## Algorithm Comparison

| Algorithm | Optimality | Memory | Scalability | Best For |
|-----------|-----------|--------|-------------|----------|
| FastPAM | Local optimum | $O(N^2)$ distance storage | Dense | General k-medoids |
| OneBatchPAM | Approximate | $O(Nm)$ table (plus optional $\le 64$ MiB panels) | Matrix-free | Bounded distance budget |
| FastCLARA | Approximate | $O(s^2)$ sample storage plus compact pass labels | Matrix-free / streaming | Large datasets |
| Hierarchical | N/A (dendrogram) | $O(N^2)$ matrix + packed `work` copy | Guarded at 2,000 points | Exploratory hierarchy |
| Lloyd's (`kmedoids`) | Local optimum | Uses configured distance storage | Dense; $O(\sum_c \|C_c\|^2)$ updates | Simple assignment/update |
| MIP | Certified when solved to optimality | $O(N^2)$ distance storage + CSC ($N$ binary, $N^2-N$ continuous) | Dense ($N \lesssim 26{,}750$) | Solver-backed exact result |
| LR-core | Certified or loud failure | $O(N^2)$ packed/mmapped distance storage | Dense or `.dtwm` | Large exact instances |
| TADPole | Exact-arithmetic identity (supported contract) | Sparse per-row exact caches + up to 16 UB seeds + envelopes (CLI conservatively plans worst-case RAM) | Density/cutoff search | Admissible pair pruning |
| Barycentre k-means | Local optimum | $O(N + kL)$ (plus bounded $\le 64$ MiB DP scratch) | Matrix-free | Sequence-centroid averages |
