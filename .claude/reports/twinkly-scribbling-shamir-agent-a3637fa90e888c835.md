# DTW Libraries Comparison Report

Research completed 2026-03-27. Information gathered from source code, READMEs, CRAN metadata, and documentation pages.

---

## 1. R: dtwclust (v6.0.0)

**Author:** Alexis Sarda-Espinosa | **License:** GPL-3 | **Last updated:** 2024-07-23

### DTW variants
- Standard DTW (via the `dtw` R package dependency)
- DTW Barycenter Averaging (DBA)
- Soft-DTW distance and centroid routines
- Global Alignment Kernel (GAK)
- `dtw_basic()` -- fast C++ implementation of DTW for clustering

### Metrics / distances
- L1 (Manhattan) and L2 (Euclidean) norms for LB and DTW
- Shape-Based Distance (SBD) for k-Shape

### Clustering algorithms
- **Partitional:** k-medoids (PAM), random + custom centroid functions
- **Hierarchical:** full linkage tree with custom distance
- **Fuzzy:** fuzzy c-medoids / fuzzy c-means for time series
- **k-Shape:** cross-correlation-based clustering
- **TADPole:** density-based DTW clustering with pruning

### Performance optimizations
- **Lower bounds:** LB_Keogh, LB_Improved (Lemire 2009) -- used to prune full DTW computations in nearest-neighbor search
- **C++ backend:** Links to Rcpp, RcppArmadillo, RcppParallel, RcppThread
- **Parallelization:** `foreach` + `doParallel` for R-level; RcppParallel/RcppThread for C++-level multi-threading
- Optimized cross-distance matrix loops
- No SIMD or GPU

### Missing data
- Not explicitly supported in DTW core

### Cluster validity indices
- Silhouette, Dunn, CVI, DB, modified DB, and others

### Unique features
- Interactive Shiny app for cluster exploration
- Extensible: custom distance measures and centroid definitions via R functions
- Proxy-compatible distance objects
- Most comprehensive R package for DTW + clustering

---

## 2. Python: dtaidistance (v2.x)

**Author:** Wannes Meert, KU Leuven DTAI | **License:** Apache 2.0

### DTW variants
- Standard DTW
- DTW with Sakoe-Chiba band (`window` parameter)
- DTW with max step size constraint
- DTW with psi relaxation (for cyclic sequences -- ignore begin/end)
- Multi-dimensional DTW (`dtw_ndim` package)
- Subsequence DTW / subsequence alignment
- DTW Barycenter Averaging (DBA) for clustering

### Metrics
- Squared Euclidean (default pointwise distance)
- Custom not directly exposed in C path

### Clustering
- Hierarchical clustering (custom + SciPy linkage wrapper)
- `HierarchicalTree` with dendrogram visualization
- DTW Barycenter Averaging for centroid computation
- No k-medoids / PAM built in (relies on external)

### Performance optimizations
- **C backend:** Pure C implementation via Cython, 30-300x faster than Python
- **PrunedDTW:** `use_pruning=True` sets `max_dist` to Euclidean upper bound, equivalent to Silva & Batista's PrunedDTW algorithm -- prunes partial distance computations
- **Early abandoning:** `max_dist` parameter stops computation when distance exceeds threshold
- **OpenMP parallelization:** Distance matrix computation parallelized in C via OpenMP (directly, not via Python threading)
- **Block computation:** Can compute sub-blocks of distance matrix for distributed computing
- **No SIMD, no GPU**

### Missing data
- Not explicitly supported

### Unique features
- Subsequence search (KNN, local concurrences)
- LoCoMotif motif discovery (separate package)
- Very lightweight: only Cython required (Numpy optional)
- Variable-length series natively supported
- Visualization utilities for warping paths and matrices

---

## 3. Python: tslearn (v0.6.x)

**Author:** Romain Tavenard et al. | **License:** BSD 2-Clause

### DTW variants
- Standard DTW with Sakoe-Chiba band and Itakura parallelogram constraints
- **Soft-DTW** (Cuturi & Blondel) -- differentiable DTW for gradient-based learning
- Soft-DTW normalized
- DTW with limited warping length
- Subsequence DTW
- **Canonical Time Warping (CTW)** -- DTW with linear projection alignment
- **Frechet distance**
- LCSS (Longest Common Subsequence)
- **LB_Keogh** lower bound implementation

### Metrics
- Squared Euclidean pointwise (default)
- Global Alignment Kernel (GAK) -- positive-definite kernel based on DTW
- SAX distance
- Normalized cross-correlation (for SBD)
- Custom metric via `dtw_path_from_metric` (any callable)

### Clustering
- **TimeSeriesKMeans:** k-means with DTW or soft-DTW barycenter averaging
- **KShape:** shape-based clustering
- **KernelKMeans:** kernel k-means with GAK
- **TimeSeriesDBSCAN:** density-based clustering

### Performance optimizations
- **Numba JIT:** Core DTW loops compiled with `@njit`
- **PyTorch backend:** Soft-DTW can run on GPU via PyTorch backend (`SoftDTWLossPyTorch`)
- No C backend, no OpenMP, no SIMD
- Supports parallel parameter in some functions

### Missing data
- **Variable-length series supported** (NaN padding with masking)

### Unique features
- **scikit-learn compatible API** (fit/predict/transform, pipelines, GridSearchCV)
- PyTorch backend for soft-DTW (GPU-capable)
- Time series classification (KNN, SVM, Learning Shapelets)
- Time series regression
- Preprocessing (scaling, resampling, piecewise approximation)
- Matrix Profile computation
- UCR dataset loader built in
- Multi-dimensional time series

---

## 4. Python: aeon (v1.4.0)

**Author:** aeon developers (forked from sktime v0.16.0 in 2022) | **License:** BSD 3-Clause

### DTW variants (most comprehensive of all libraries)
- **DTW** -- standard dynamic time warping
- **DDTW** -- derivative DTW (uses first derivative of series)
- **WDTW** -- weighted DTW (logistic weight penalty based on warping distance)
- **WDDTW** -- weighted derivative DTW
- **ADTW** -- amerced DTW (explicit constant warping penalty)
- **DTW-GI** -- DTW with global invariances
- **Shape-DTW** -- DTW using shape descriptors
- **Soft-DTW** -- differentiable relaxation

### Other elastic distances
- LCSS, ERP, EDR, TWE, MSM (Move-Split-Merge)
- Shape-Based Distance (SBD)
- Matrix Profile distance
- Shift-scale invariant distance
- SAX/SFA distances (MINDIST)

### Metrics / pointwise
- Squared Euclidean, Manhattan, Minkowski

### Constraints
- Sakoe-Chiba band (via `window` parameter, percentage-based)
- Itakura parallelogram (via `itakura_max_slope`)
- Custom bounding matrix

### Clustering algorithms (most comprehensive)
- **TimeSeriesKMedoids** (PAM k-medoids)
- **TimeSeriesCLARA** (scalable k-medoids)
- **TimeSeriesCLARANS** (randomized k-medoids)
- **TimeSeriesKMeans** (with DBA/soft-DTW barycenters)
- **KShape / TimeSeriesKShape**
- **KernelKMeans**
- **KASBA** (fast approximate k-medoids)
- **ElasticSOM** (self-organizing map with elastic distances)
- **KSpectralCentroid**

### Performance optimizations
- **Numba JIT:** All distance functions decorated with `@njit(cache=True, fastmath=True)`
- **Numba parallel:** Pairwise distance matrices use `prange` for parallel computation
- **Threading:** `numba_thread_handler` for thread management
- No C backend, no SIMD, no GPU
- Caching of JIT-compiled functions

### Missing data
- Not explicitly addressed in distance functions

### Unique features
- scikit-learn compatible API
- Largest collection of elastic distance measures
- Classification (KNN, distance-based, shapelet, deep learning)
- Anomaly detection, segmentation, similarity search modules
- NumFOCUS affiliated project
- Active development, large community

---

## 5. UCR Suite

**Authors:** Rakthanmanon, Campana, Mueen, Batista, Keogh (2012) | **License:** Free for research (custom restrictive license)

### Core purpose
Subsequence search under DTW -- finding the nearest neighbor of a query in a very long time series. The "gold standard" for fast exact DTW search.

### Optimizations (cascading, from cheapest to most expensive)

1. **Z-normalization on the fly:** Normalize subsequences incrementally using running mean/std, avoiding pre-computation

2. **Early abandoning of Euclidean distance:** Before any DTW, check if Euclidean distance already exceeds best-so-far

3. **LB_Kim (O(1)):** Constant-time lower bound using first, last, min, max points of the series. Cheap but effective pruning

4. **LB_Keogh on query envelope (O(n)):** Linear-time lower bound. Computes upper/lower envelope of the query using Lemire's streaming min/max algorithm (deque-based). Checks if candidate series falls outside envelope

5. **LB_Keogh on data envelope (O(n)):** Second pass -- compute envelope of the data subsequence, check query against it. Often tighter than the first LB_Keogh

6. **Cascading lower bounds:** Use the better of LB_Keogh and LB_Keogh2 to decide whether to compute full DTW

7. **Early abandoning of DTW:** During DTW computation, accumulate cumulative bound from LB_Keogh residuals. Abandon DTW row computation if cumulative cost + remaining lower bound exceeds best-so-far

8. **Query reordering:** Sort query indices by absolute z-normalized value (high to low). Compute DTW in this order so that the most discriminative dimensions are checked first, enabling earlier abandoning

### Key implementation details
- Single-pass streaming over data (can process arbitrarily long time series)
- Deque-based O(n) envelope computation (Lemire 2009)
- Sakoe-Chiba band constraint
- Pure C implementation, no dependencies
- Achieves 100x-1000x speedups over naive DTW search
- Claims to search trillions of subsequences

### No clustering, no multi-dimensional, no GPU, no SIMD
- Designed purely for 1-NN subsequence search

---

## 6. GPU DTW Implementations

### pytorch-softdtw-cuda (Maghoumi)
- **CUDA implementation of Soft-DTW** for PyTorch
- Both forward and backward passes on GPU
- **Diagonal-based Bellman recursion:** Anti-diagonal parallelism -- cells on the same anti-diagonal of the cost matrix are independent and can be computed in parallel
- Up to **100x faster** than CPU soft-DTW
- Depends on PyTorch + Numba
- Differentiable -- usable as a loss function for deep learning
- Batch processing of multiple series pairs
- MIT License

### General GPU DTW approaches (from literature)
- **Anti-diagonal parallelism:** The key insight -- cells along the same anti-diagonal of the DTW matrix have no data dependencies and can be computed simultaneously. For an NxM matrix, this gives up to min(N,M) parallel threads per anti-diagonal
- **Batch parallelism:** Compute many pairwise DTW distances simultaneously across GPU cores
- **Distance matrix computation:** Parallelize across all N*(N-1)/2 pairs
- **Challenges:** DTW's sequential dependency along rows/columns limits intra-pair parallelism. GPU overhead only pays off for large batches or long series
- **cuDTW:** Various research implementations exist but no widely adopted library

### No production-quality CUDA DTW library exists for general (non-soft) DTW
- The sequential nature of DTW's dynamic programming makes it inherently harder to parallelize than, e.g., matrix multiplication
- Anti-diagonal parallelism is the standard approach but provides limited speedup for short series

---

## 7. Apache Arrow C++ as a dependency

### License
- **Apache License 2.0** -- permissive, BSD-2 compatible (both are permissive open-source licenses; Apache 2.0 can be combined with BSD code)

### Size concerns
- Apache Arrow C++ is a **very large dependency:**
  - Full build: ~1-2 GB build artifacts
  - Installed shared library: ~50-100 MB
  - Build time: 10-30+ minutes
  - Pulls in many transitive dependencies (Boost, Thrift, Protobuf, zlib, lz4, zstd, snappy, etc.)
  - CMake build system is complex

### What Arrow provides
- Columnar in-memory data format
- Zero-copy reads, IPC, Flight RPC
- Compute kernels (aggregation, filtering, sorting)
- Parquet/CSV/JSON readers
- Integration with pandas, numpy

### Verdict for DTWC++
- **Massively overkill** for a DTW library
- Arrow is designed for columnar analytics, not time-series distance computation
- The dependency footprint alone (~100 MB installed, 30+ transitive deps) would dwarf the DTW library itself
- **Recommendation: Do not use.** For data loading, simple CSV parsing or a lightweight library (e.g., fast-cpp-csv-parser, single-header) is far more appropriate. For Python interop, pybind11/nanobind with numpy is sufficient.

---

## 8. DTW SIMD Optimization

### State of the art
- **Very limited adoption** in existing DTW libraries -- none of the major libraries (dtaidistance, tslearn, aeon, dtwclust) use SIMD
- The sequential dependency in DTW's DP recurrence makes SIMD challenging

### Possible SIMD strategies

1. **Anti-diagonal vectorization:** Process multiple cells on the same anti-diagonal using SIMD. Each cell on an anti-diagonal is independent. With AVX-256, process 4 doubles or 8 floats simultaneously. Limited by anti-diagonal length and gather/scatter overhead.

2. **Pointwise distance vectorization:** The inner product `(x_i - y_j)^2` for computing the cost matrix entries can be vectorized with SIMD. For multivariate series with D dimensions, compute the pointwise distance across channels using SIMD reductions.

3. **Multiple-pair parallelism:** Compute DTW for multiple pairs simultaneously -- interleave the DP matrices of K independent pairs in SIMD registers (K=4 for AVX2 doubles). Each SIMD lane handles a different pair. This is the most promising approach.

4. **LB_Keogh envelope computation:** The sliding min/max for envelope computation can benefit from SIMD, though Lemire's deque algorithm is already O(n).

5. **Distance matrix rows:** When computing a row of the pairwise distance matrix, multiple DTW computations are independent -- natural SIMD parallelism.

### Key references
- Rakthanmanon et al. (2012) "Searching and Mining Trillions of Time Series Subsequences under Dynamic Time Warping" -- UCR Suite paper
- Silva & Batista (2016) "PrunedDTW" -- pruning partial distance computations
- Lemire (2009) "Faster Retrieval with a Two-Pass Dynamic-Time-Warping Lower Bound"
- Mueen & Keogh (2016) "Extracting Optimal Performance from Dynamic Time Warping" -- KDD Tutorial

---

## 9. LB_Keogh Implementation Details

### Algorithm
1. For a query Q with Sakoe-Chiba band width r, compute upper envelope U and lower envelope L:
   - `U[i] = max(Q[i-r], ..., Q[i+r])`
   - `L[i] = min(Q[i-r], ..., Q[i+r])`

2. For a candidate series C, the lower bound is:
   ```
   LB_Keogh(Q, C) = sum over i:
     (C[i] - U[i])^2  if C[i] > U[i]
     (C[i] - L[i])^2  if C[i] < L[i]
     0                 otherwise
   ```

3. LB_Keogh <= DTW(Q, C) always holds (valid lower bound)

### Efficient envelope computation
- **Lemire's streaming algorithm:** Uses a double-ended queue (deque) to compute sliding window min/max in O(n) time
- UCR Suite implementation uses circular deque for memory efficiency
- Envelope computed once for the query, reused across all candidates

### LB_Improved (Lemire 2009)
- After computing LB_Keogh, project the candidate onto the envelope
- Compute LB_Keogh in reverse (envelope of candidate, query against it)
- Tighter bound at cost of ~2x computation

---

## 10. Scaling DTW to Billions of Comparisons

### Strategies (from UCR Suite and literature)

1. **Cascading lower bounds** (UCR Suite approach): LB_Kim (O(1)) -> LB_Keogh (O(n)) -> LB_Keogh2 (O(n)) -> full DTW. Each level prunes candidates cheaply. 90-99% of candidates pruned before full DTW.

2. **Early abandoning:** Stop DTW computation mid-way when cumulative cost exceeds best-so-far. Combined with LB residuals for tighter bounds.

3. **PrunedDTW (Silva & Batista):** Skip computation of cells in the DTW matrix that provably cannot improve the result. Prunes columns/rows rather than just early-stopping.

4. **Indexing:** For nearest-neighbor queries, use LB_Keogh as an index to avoid computing DTW for most candidates.

5. **Approximate methods:** CLARA/CLARANS for clustering avoid computing full distance matrices. Sample-based approaches.

6. **Parallelism:**
   - OpenMP for distance matrix computation (independent pairs)
   - GPU batch processing
   - Distributed computation (block-based distance matrix)

7. **Banded DTW:** Sakoe-Chiba band reduces DTW from O(nm) to O(n*w) where w << m

8. **Data reduction:** PAA (Piecewise Aggregate Approximation), SAX for dimensionality reduction before DTW

---

## Summary Comparison Table

| Feature | dtwclust (R) | dtaidistance | tslearn | aeon | UCR Suite |
|---------|-------------|-------------|---------|------|-----------|
| **Language** | R + C++ | Python + C | Python | Python | C |
| **License** | GPL-3 | Apache 2.0 | BSD 2-Clause | BSD 3-Clause | Research-only |
| **DTW variants** | Standard, Soft | Standard, Subseq | Standard, Soft, CTW, Frechet | DTW, DDTW, WDTW, WDDTW, ADTW, Shape-DTW, Soft-DTW, DTW-GI | Standard (banded) |
| **Other elastic** | SBD, GAK | -- | LCSS, GAK | LCSS, ERP, EDR, TWE, MSM, SBD | -- |
| **Lower bounds** | LB_Keogh, LB_Improved | Euclidean UB pruning | LB_Keogh | Bounding matrix | LB_Kim, LB_Keogh, LB_Keogh2 |
| **Clustering** | PAM, hierarchical, fuzzy, k-Shape, TADPole | Hierarchical only | KMeans, KShape, KernelKMeans, DBSCAN | KMedoids, CLARA, CLARANS, KMeans, KShape, KASBA, SOM | None |
| **C/native backend** | Yes (Rcpp) | Yes (Cython/C) | No (Numba JIT) | No (Numba JIT) | Yes (pure C) |
| **Parallelization** | foreach + RcppParallel | OpenMP in C | Limited | Numba prange | None |
| **GPU** | No | No | Soft-DTW via PyTorch | No | No |
| **SIMD** | No | No | No | No | No |
| **Missing data** | No | No | Variable-length (NaN pad) | No | No |
| **Multi-dim** | Yes | Yes (dtw_ndim) | Yes | Yes | No |
| **sklearn API** | No | No | Yes | Yes | No |
| **Differentiable** | Soft-DTW | No | Soft-DTW (PyTorch) | Soft-DTW | No |

---

## Key Takeaways for DTWC++

1. **aeon has the richest set of DTW variants** (DDTW, WDTW, ADTW, Shape-DTW, DTW-GI). Implementing these would be a differentiator.

2. **No library combines C++ performance with comprehensive DTW variants + clustering.** This is the gap DTWC++ can fill.

3. **UCR Suite optimizations are the gold standard** for performance: cascading lower bounds (LB_Kim -> LB_Keogh -> LB_Keogh2), early abandoning with cumulative LB residuals, query reordering. These should be implemented.

4. **SIMD is an untapped opportunity** -- no existing library uses it. Anti-diagonal vectorization and multi-pair SIMD lanes are viable strategies.

5. **GPU (CUDA)** -- anti-diagonal parallelism for soft-DTW is proven (100x speedup). General DTW is harder but batch parallelism over pairs is straightforward.

6. **Apache Arrow is not recommended** as a dependency -- too large and not relevant to the problem domain.

7. **Soft-DTW** is increasingly important for deep learning integration (differentiable loss). tslearn's PyTorch backend is the reference implementation.

8. **Missing data in DTW** remains underserved across all libraries -- an opportunity for DTWC++.

9. **License landscape:** GPL-3 (dtwclust) is restrictive; Apache 2.0, BSD 2-Clause, and BSD 3-Clause are all compatible with each other and with DTWC++'s likely BSD/MIT licensing.

10. **Clustering:** aeon has the most comprehensive set (PAM, CLARA, CLARANS, KASBA). dtwclust adds fuzzy and TADPole. DTWC++ should target at minimum: PAM, CLARA, CLARANS, hierarchical.
