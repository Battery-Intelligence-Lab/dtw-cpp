# DTWC++ Roadmap Design Spec

**Date:** 2026-04-02 (updated 2026-04-02)
**Author:** Volkan Kumtepeli + Claude
**Branch:** Claude
**Status:** Active — Waves 1A, 1B, 2A, 2B completed

## Completion Status (as of 2026-04-02)

| Wave | Scope | Status |
|------|-------|--------|
| **Wave 1A** | Metrics + Missing Data (missing_utils, MissingStrategy, DTW-AROW, 5 scoring metrics) | **DONE** |
| **Wave 1B** | Multivariate Foundation (ndim, MVL1/MVSquaredL2, MV DTW, MV DDTW) | **DONE** |
| **Wave 2A** | Clustering Algorithms (deferred alloc, medoid utils, hierarchical, CLARANS, FastCLARA fixes) | **DONE** |
| **Wave 2B** | MV Variants + Lower Bounds (MV WDTW/ADTW/DDTW, per-channel LB_Keogh, MV missing DTW) | **DONE** |
| **CUDA Next** | Device-side pruning, arch-aware dispatch, benchmark expansion | **PLANNED** (see plans/2026-04-02-cuda-next-phase.md) |
| **SIMD** | Google Highway (LB_Keogh, z_normalize, multi-pair DTW) | **PLANNED** (not started) |
| **A4** | GPU-CLARA (wire K-vs-N into clustering loop) | **NOT STARTED** |
| **A5** | Two-phase clustering (pre-categorized data) | **NOT STARTED** |
| **A6** | Condensed distance matrix | **NOT STARTED** |
| **A7** | Lazy loading | **NOT STARTED** |
| **A8** | Binary distance matrix storage | **NOT STARTED** |
| **Bindings** | MATLAB MEX, Python updates | **NOT STARTED** (skills drafted) |

## Context

Following two Codex sessions (2026-04-02):
- **Initial session:** 3-25x GPU speedups and 1.1-1.5x CPU improvements (wavefront/warp/regtile kernels, streams+pinned memory, CPU pruned distance matrix).
- **Adversarial follow-up:** Made GPU pruning genuine (device-side active-pair compaction, LB values stay on GPU, no host round-trip). Added reusable `OneVsAllLaunchWorkspace<T>` for 1-vs-N/K-vs-N query paths. **Key gap remaining:** K-vs-N kernel exists but is NOT wired into the clustering loop — `Problem.cpp` GPU strategy falls through to CPU brute-force.

This spec addresses five areas for DTWC++ evolution:

1. Clustering algorithms for large-scale data (100K-1M series)
2. Multivariate (ND array) time series support
3. Improved missing data handling (DTW-AROW)
4. Additional output/quality metrics
5. Binding parity (MATLAB catch-up, Python updates, future R)

**Real-world motivator:** Scooter usage data from ~70 cities, grouped by season/time — potentially millions of series with natural pre-categorization.

**Architecture:** Layered scale-aware with user override (A+B hybrid). Auto-selects algorithm by dataset cost `N^2 * min(L, band) * ndim`, with explicit `Algorithm` enum override.

**Concurrency principle:** Lock-free by design. Parallel decomposition guarantees non-overlapping memory regions. No atomics, no locks. Threads compute disjoint distance pairs.

---

## Workstream A: Clustering & Scale

### A1. Algorithm Suite

| Algorithm | Scale | Memory | Description |
|-----------|-------|--------|-------------|
| **PAM (FastPAM1)** | Small | O(N^2) full matrix | Exact k-medoids. Already implemented in `algorithms/fast_pam.hpp`. |
| **Hierarchical** | N < 50K | O(N*(N-1)/2) condensed | Single, complete, average linkage. **No Ward's** (mathematically invalid for DTW — requires Euclidean centroids, violates Lance-Williams recurrence with medoids). |
| **CLARANS** | 5K-100K | O(N * cache_size) | Ng & Han (2002) randomized local search. Distance cache for amortization. |
| **FastCLARA (improved)** | 50K-1M | O(sample^2) | Already in `algorithms/fast_clara.hpp`. Improve: GPU K-vs-N kernel, larger sample default, parallel sample evaluation. |
| **Mini-batch Hierarchical** | 50K-500K | O(sample^2) | Sample → dendrogram → flat assignment. Documented as approximation (no full dendrogram for non-sampled points). |

**Adaptive auto-selection** based on `cost = N^2 * min(L, band) * ndim`:

| Cost Tier | Threshold | Default Algorithm |
|-----------|-----------|-------------------|
| Small | cost < 10^10 | PAM + full matrix |
| Medium | N < 50K AND matrix fits in RAM | CLARANS or hierarchical |
| Large | Otherwise | GPU-CLARA or mini-batch HC |

User override: `Problem::set_algorithm(ClusteringAlgorithm::CLARANS)`.

### A2. Hierarchical Clustering

**Linkage types:** Single, complete, average (UPGMA). Ward's excluded — DTW is not a metric, no Euclidean centroid exists, Lance-Williams recurrence cannot be preserved when the "merge representative" requires re-scanning all points.

**Output:** `Dendrogram` struct:
```cpp
struct Dendrogram {
    std::vector<std::array<int, 2>> merges;  // merge pairs (row i merges[i][0] + merges[i][1])
    std::vector<double> distances;            // merge distance at each step
    std::vector<int> sizes;                   // cluster size after each merge
    
    std::vector<int> cut(int k) const;        // cut to get k clusters, returns labels
};
```

**Implementation:** Standard O(N^2 log N) with condensed distance matrix. Uses `CondensedDistanceMatrix` (new, see A6).

### A3. CLARANS

**Algorithm:** Randomized neighborhood search in the space of k-medoid sets.

```
for restart = 1..num_local:
    S = random k medoids
    for trial = 1..max_neighbor:
        pick random (medoid_to_remove, non_medoid_to_add)
        compute swap_cost using cached distances
        if swap improves cost: accept, reset trial counter
    keep best S across restarts
```

**Parameters:**
- `n_clusters`: k
- `num_local`: restarts (default 2)
- `max_neighbor`: max non-improving swaps before stopping (default `max(250, 0.0125 * k * (N - k))` per Ng & Han 2002)
- `random_seed`: deterministic seeding

**Distance cache:** `std::unordered_map<uint64_t, double>` keyed by `pack(min(i,j), max(i,j))`. Amortizes repeated DTW computations across swap evaluations. Eviction when cache exceeds memory budget.

**Complexity note:** Without spatial indexing (inapplicable to DTW), each swap evaluation is O(N). CLARANS wins over PAM through early termination (first improving swap), not per-evaluation cost. For DTW workloads, the distance cache is critical — without it, complexity approaches O(N^2 * k) per restart, same as PAM.

### A4. Improved GPU-CLARA

**Already exists (from Codex sessions):**
- `compute_dtw_k_vs_all()` API with wavefront/warp/regtile kernel variants (`cuda_dtw.cuh` lines 113+)
- `OneVsAllLaunchWorkspace<T>` with thread_local reuse, lazy capacity growth (anonymous namespace in `cuda_dtw.cu`)
- Device-side active-pair compaction (`compact_active_pairs_kernel`) for pruned pairwise distance matrix
- LB_Keogh fully on-device (`d_lb`, `d_upper`, `d_lower` in `DTWLaunchWorkspace`)

**Still needed:**
- **Wire K-vs-N into clustering:** `Problem::fillDistanceMatrix()` GPU strategy currently falls through to CPU (`[[fallthrough]]` at line 255). Must call `compute_distance_matrix_cuda()` or `compute_dtw_k_vs_all()` when `DistanceMatrixStrategy::GPU` is selected.
- **Add LB pruning to K-vs-N path:** `OneVsAllLaunchWorkspace` has no LB/envelope buffers. Extend with `d_lb`, `d_upper`, `d_lower`, `d_active_pairs` to enable pruning in the query path (currently pruning only works in pairwise `DTWLaunchWorkspace`).
- **Promote workspace types:** Both `DTWLaunchWorkspace` and `OneVsAllLaunchWorkspace` are in anonymous namespaces. If clustering needs to hold workspace state across iterations (e.g., reuse across CLARA samples), promote to an internal header or expose via opaque handle.
- **Larger sample default:** `max(40 + 2*k, min(N, 10*k + 100))` (Schubert & Rousseeuw 2021). For k=70: sample=800 instead of 180.
- Parallel sample evaluation: evaluate multiple CLARA samples concurrently
- Streaming: for N > 100K, stream series to GPU in chunks

### A5. Two-Phase Clustering (Pre-categorized Data)

For datasets with natural grouping (city/season/time):

```
Phase 1: Cluster within each group → get per-group medoids
Phase 2: Cluster across all group medoids → get super-clusters
```

**DataLoader gains hierarchical folder-tree awareness:**
```
data/
  london/
    summer/ → series files
    winter/ → series files
  paris/
    summer/ → series files
```

**API:**
```cpp
DataLoader loader;
auto groups = loader.path("data/").recursive(true).load_grouped();
// groups: std::map<std::string, Data> with keys "london/summer", etc.

// Within-group clustering
for (auto& [name, data] : groups) {
    Problem prob; prob.set_data(std::move(data));
    prob.set_algorithm(ClusteringAlgorithm::PAM);
    prob.cluster();
}

// Cross-group: collect medoid series, cluster them
```

**Python:**
```python
groups = dtwcpp.load_grouped("data/", recursive=True)
results = {name: dtwcpp.DTWClustering(n_clusters=5).fit(data)
           for name, data in groups.items()}
```

### A6. Condensed Distance Matrix

New `CondensedDistanceMatrix` storing only upper triangle: N*(N-1)/2 entries.

```cpp
class CondensedDistanceMatrix {
    std::vector<double> data_;  // N*(N-1)/2 entries
    size_t n_;
    
    size_t index(size_t i, size_t j) const {
        // i < j required
        return i * (2*n_ - i - 3) / 2 + j - 1;
    }
public:
    double get(size_t i, size_t j) const;
    void set(size_t i, size_t j, double val);
};
```

Halves memory vs full NxN. Used by hierarchical clustering and PAM for medium-scale datasets.

### A7. Lazy Loading

```cpp
/// Abstract lazy data source — loads series on demand
class LazyDataSource {
public:
    virtual ~LazyDataSource() = default;
    virtual size_t size() const = 0;
    virtual std::vector<data_t> load(size_t index) const = 0;
    virtual size_t ndim() const { return 1; }
};

/// File-backed: series from HDF5 rows, CSV files, or folder of files
class FileBackedDataSource : public LazyDataSource { ... };

/// Memory-backed: wraps existing Data (zero overhead for small datasets)
class MemoryDataSource : public LazyDataSource { ... };

/// LRU cache wrapper — caches frequently accessed series (medoids, candidates)
class CachedDataSource : public LazyDataSource {
    LazyDataSource& source_;
    mutable LRUCache<size_t, std::vector<data_t>> cache_;  // default 1000 entries
};
```

**Usage by algorithms:**
- **CLARA/CLARANS:** Load sample indices → `load(i)` for each → compute sample distances → discard
- **Mini-batch HC:** Same pattern
- **PAM:** Loads all N (only for small N where it fits)

**Problem integration:**
```cpp
Problem(std::shared_ptr<LazyDataSource> source);  // lazy mode
Problem(Data data);                                 // eager mode (existing)
```

### A8. Binary Distance Matrix Storage

**Dual format:**

| Format | Use Case | Dependency |
|--------|----------|------------|
| **HDF5** | Portable archival, cross-language exchange | Optional (`DTWC_HAS_HDF5`) |
| **mmap'd flat binary** | Random access (CLARANS), on-demand lookup | None |

**HDF5 layout:**
```
/dtwc/distance_matrix  — dataset: condensed upper triangle, float64
/dtwc/metadata          — attrs: N, variant, band, metric, version, checksum, algorithm, seed
/dtwc/series_names      — string dataset (optional)
```

**Flat binary layout:**
```
[Header: 32 bytes]
  magic: "DTWC" (4 bytes)
  version: uint32
  N: uint64
  flags: uint32 (condensed/full, float32/float64)
  checksum: uint64
[Data: N*(N-1)/2 * sizeof(double)]
```

Random access via `offset = header_size + index(i,j) * sizeof(double)` with `pread`/`ReadFile`.

**Note:** HDF5 chunked I/O is suboptimal for CLARANS random-row access (decompresses full chunks). Use mmap'd flat binary for CLARANS; HDF5 for archival/exchange.

### A9. Thread Safety (Lock-Free by Design)

Per project principle: "structure parallel decomposition so threads write to non-overlapping memory regions. No atomics or locks needed if the decomposition guarantees no data races."

**`distByInd` race (existing latent bug):** Current check-then-act pattern on `is_computed`/`set` in `Problem::distByInd()` is not thread-safe. Resolution is NOT atomics — instead, restructure algorithms to avoid shared mutable distance state:

- **CLARA:** Each sample computes its own local distance matrix (sample_size^2). The sample is small enough for a fresh `CondensedDistanceMatrix`. No shared state between samples. Samples can run in parallel on separate threads, each with its own matrix.
- **CLARANS:** Distance cache is per-restart (each restart gets its own `std::unordered_map`). Within a restart, swap evaluation is inherently sequential. No parallel writes.
- **Distance matrix fill (existing):** Already uses `#pragma omp parallel for` with row-based decomposition. Each thread writes to disjoint rows of the output matrix. No conflict.
- **GPU path:** All parallelism is within CUDA kernels. Host code is single-threaded per GPU call. The existing `DTWLaunchWorkspace` and `OneVsAllLaunchWorkspace` are thread_local.

**Global `randGenerator`:** Replace with per-algorithm `std::mt19937` seeded from `Problem::random_seed`. Already done in `fast_clara.cpp`; extend to all new algorithms (CLARANS, hierarchical init).

### A10. Algorithm Dispatch

New enum (separate from legacy `Method`):
```cpp
enum class ClusteringAlgorithm {
    Auto,          // auto-select based on cost tier
    PAM,           // FastPAM1
    CLARA,         // FastCLARA
    CLARANS,       // CLARANS
    Hierarchical,  // exact hierarchical (single/complete/average)
    MiniBatchHC,   // sample-based hierarchical
    MIP            // exact MIP (existing)
};
```

`Problem::set_algorithm()` sets the algorithm. `Problem::cluster()` dispatches accordingly.

---

## Workstream B: Multivariate (ND Array) Support

### B1. Data Layout

**CPU: Interleaved** — all D features for timestep i are contiguous:
```
[t0_f0, t0_f1, t0_f2, t1_f0, t1_f1, t1_f2, ...]
```
Cache-friendly for per-timestep access in DTW inner loop.

**GPU: Planar** — each channel's data is contiguous:
```
[series_idx * D * max_L + d * max_L + t]
```
Required because interleaved layout blows shared memory budget on GPU (D=10, L=500 → 92KB for series preloads alone, exceeding most GPU limits). Planar enables coalesced access and manageable shared memory (anti-diagonal buffers remain L-sized scalars).

**One-time CPU→GPU transposition:** O(N*L*D), amortized over O(N^2) DTW pairs.

**Constraint:** All series in a dataset must have the same `ndim`. Variable D across series is mathematically undefined (cannot compute pointwise distance between 3-channel and 5-channel vectors). Validated at data ingestion with assertion.

### B2. Data Structures

**`Data` struct** gains `ndim`:
```cpp
struct Data {
    std::vector<std::vector<data_t>> p_vec;  // flat interleaved per series
    std::vector<std::string> p_names;
    size_t ndim = 1;
    
    size_t series_length(size_t i) const { return p_vec[i].size() / ndim; }
    // Invariant: p_vec[i].size() % ndim == 0 for all i
};
```

**`TimeSeriesView`** gains `ndim`:
```cpp
template <typename T = double>
struct TimeSeriesView {
    const T *data;
    size_t length;  // timesteps
    size_t ndim;    // features per timestep (1 for univariate)
    const T *at(size_t i) const { return data + i * ndim; }
};
```

### B3. Compile-Time ndim Dispatch

**Critical:** Runtime `ndim` in the inner loop prevents compiler inlining/unrolling. Dispatch must happen at the public API level (same level as current metric dispatch), NOT per-cell.

```cpp
// Compile-time specializations
template<size_t D> struct MVL1Dist {
    template<typename T>
    T operator()(const T* a, const T* b) const noexcept {
        T sum = T(0);
        for (size_t d = 0; d < D; ++d) sum += std::abs(a[d] - b[d]);
        return sum;
    }
};

template<> struct MVL1Dist<1> {
    // Identical to current scalar L1Dist — zero overhead
    template<typename T>
    T operator()(const T* a, const T* b) const noexcept { return std::abs(*a - *b); }
};

// Runtime fallback for D > 3
struct MVL1DistRuntime {
    size_t ndim;
    template<typename T>
    T operator()(const T* a, const T* b) const noexcept { ... }
};
```

Public API dispatch: `switch (ndim) { case 1: ...<1>; case 2: ...<2>; case 3: ...<3>; default: ...Runtime; }`.

D=1 produces identical machine code to current — verify via assembly inspection.

### B4. DTW Function Changes

All DTW functions gain trailing `ndim=1`. Inner loop changes from `distance(x[i], y[j])` to `distance(x + i*ndim, y + j*ndim)`:

```cpp
template <typename data_t>
data_t dtwBanded(const data_t* x, size_t nx_steps, const data_t* y, size_t ny_steps,
                 int band, data_t early_abandon, core::MetricType metric,
                 size_t ndim = 1);
```

`nx_steps` and `ny_steps` are timestep counts (not flat array size). The flat array has `nx_steps * ndim` elements.

### B5. Variant Adaptations

| Variant | Change |
|---------|--------|
| Standard DTW | D-dimensional distance functor |
| DDTW | **Stride-aware derivative transform** — current `derivative_transform` treats flat vector as univariate, would mix channels. Must differentiate each channel independently with stride `ndim`. |
| WDTW | Weights depend on `\|i-j\|` only → `w * mv_distance(x_i, y_j)`. No structural change. |
| ADTW | Penalty on step type → multivariate distance replaces scalar. No structural change. |
| Soft-DTW | Replace scalar distance; gradient needs per-channel extension. |
| Missing | Per-channel NaN: zero-contribution for NaN channels (fast, consistent with existing zero-cost philosophy). Optionally: skip-channel + normalize (division per cell, more accurate, opt-in). |

### B6. Lower Bounds

**Per-channel LB_Keogh is a valid lower bound** (proven by adversarial reviewer via induction on the DP fill order — each per-channel envelope bound is path-independent, sum preserves validity).

- `Envelope` struct: `std::vector<double>` of size `length * ndim` (interleaved), plus `size_t ndim`
- `compute_envelopes`: per-channel sliding window min/max with stride `ndim`
- `lb_keogh`: sum per-channel contributions across all channels
- `SeriesSummary`: D-dimensional first/last/min/max for LB_Kim

**Metric consistency:** Current `lb_keogh` only handles L1. Need SquaredL2 variant (square the distance to envelope boundary). The `lb_keogh_valid` compile-time check already restricts valid metrics.

### B7. User Input APIs

**Python:**
```python
# Univariate (unchanged)
d = dtwcpp.dtw_distance(x_1d, y_1d)

# Multivariate: 2D array (L x D) per series
d = dtwcpp.dtw_distance(x_2d, y_2d)  # x_2d.shape = (100, 3)

# Dataset: 3D array (N x L x D) or list of 2D arrays (variable length)
dm = dtwcpp.compute_distance_matrix(array_3d)  # (200, 100, 3)
```

nanobind auto-detects dimensionality from array ndim (1D → univariate, 2D → multivariate).

**MATLAB:**
```matlab
d = dtwc.dtw_distance(X, Y);        % X is L x D matrix
D = dtwc.compute_distance_matrix(data_cell);  % cell array of L_i x D
```

### B8. GPU Kernels

- Planar layout: `all_series[series_idx * D * max_L + d * max_L + t]`
- Wavefront kernel: compute per-channel distance from global memory, accumulate scalar cost per cell. Anti-diagonal buffers remain L-sized.
- For D > 16: disable series preloading into shared memory (budget exceeded)
- Warp/regtile kernels: D=1 only (existing behavior). Multivariate uses wavefront kernel.
- Transposition buffer: `flatten_series_buffer` transposes from interleaved CPU to planar GPU

---

## Workstream C: Missing Data

### C0. URGENT FIX: `std::isnan()` Broken Under `-ffast-math`

**Existing bug.** `warping_missing.hpp` uses `std::isnan()`, but `StandardProjectSettings.cmake` enables `-ffast-math` for Release builds on GCC/Clang. `std::isnan()` may always return false — missing data feature is silently broken.

**Fix:** Replace with bitwise NaN check:
```cpp
template <typename T>
inline bool is_missing(T val) noexcept {
    static_assert(std::is_same_v<T, double> || std::is_same_v<T, float>);
    if constexpr (std::is_same_v<T, double>) {
        uint64_t bits;
        std::memcpy(&bits, &val, sizeof(bits));
        return (bits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL
            && (bits & 0x000FFFFFFFFFFFFFULL) != 0;
    } else {
        uint32_t bits;
        std::memcpy(&bits, &val, sizeof(bits));
        return (bits & 0x7F800000U) == 0x7F800000U && (bits & 0x007FFFFFU) != 0;
    }
}
```

Place in new `dtwc/missing_utils.hpp`. Update `MissingL1Dist` and `MissingSquaredL2Dist` to use it.

### C1. MissingStrategy Enum

```cpp
enum class MissingStrategy {
    Error,        // Throw if NaN encountered (default, backward-compatible)
    ZeroCost,     // Current behavior: zero local cost for NaN pairs
    AROW,         // DTW-AROW: one-to-one alignment for missing positions
    Interpolate   // Linear interpolation preprocessing, then standard DTW
};
```

Added to `DTWOptions`. Problem gains `missing_strategy` member.

### C2. DTW-AROW Algorithm

**AROW requires a dedicated `_impl` function** — the diagonal-only constraint for missing cells is a structural DP change that cannot be expressed via a distance functor swap.

**Modified recurrence:**
```
if is_missing(x[i]) or is_missing(y[j]):
    C(i, j) = C(i-1, j-1)    // diagonal only, zero local cost
else:
    C(i, j) = min(C(i-1,j-1), C(i-1,j), C(i,j-1)) + d(x[i], y[j])
```

**Boundary conditions (MUST verify against Yurtman et al. reference implementation before implementing):**

The naive approach (missing at boundary → +inf) cascades incorrectly: if `x[0]` is NaN, ALL cells in row 0 become +inf, poisoning the entire matrix. Leading NaN is common (sensor startup).

**Proposed boundary treatment (subject to verification):**
- `C(0,0)`: If either x[0] or y[0] is missing → `C(0,0) = 0` (zero cost, same as zero-cost DTW at boundary)
- First row/column: Missing values propagate from the single available predecessor (no many-to-one cheating possible in first row/column anyway)
- Interior cells: Full AROW diagonal-only constraint

**Action item:** Fetch and study `github.com/aras-y/DTW_with_missing_values` before implementing. The boundary definition is the single most critical detail.

**Variants:** `dtwAROW_L` (linear-space), `dtwAROW` (full-matrix), `dtwAROW_banded` (Sakoe-Chiba). All in new `warping_missing_arow.hpp`.

**Key invariant:** `dtwAROW(x,y) >= dtwZeroCost(x,y)` for all pairs with missing data. Proven by induction: at every missing cell, `C(i-1,j-1) >= min(C(i-1,j-1), C(i-1,j), C(i,j-1))`. Valid only if boundary conditions are consistent between the two methods.

### C3. Interpolation Strategy

`MissingStrategy::Interpolate`: linear interpolation preprocessing, then standard DTW.

**Edge handling:** LOCF/NOCB (Last Observation Carried Forward / Next Observation Carried Backward):
- Leading NaN: repeat first observed value backward
- Trailing NaN: repeat last observed value forward
- Interior NaN: linear interpolation between neighbors
- All NaN: error (no interpolation possible)

Document that boundary interpolation is extrapolation and less reliable.

### C4. Path-Length Normalization

Heuristic for linear-space DTW:
```cpp
double coverage = 1.0 - (double)(missing_x + missing_y) / (nx + ny);
return (coverage > min_coverage) ? raw_distance / coverage : infinity;
```

**Known limitation:** Path-independent approximation. Two series with 10% missing each could have 0-100% overlap in missing positions. Document as heuristic.

For full-matrix DTW: exact count matrix (track non-missing comparisons along the optimal path).

### C5. LB Pruning Guard

Auto-disable LB pruning when any series has NaN:
```cpp
bool pruning_applicable = /* existing checks */ && !dataset_has_missing;
```

In `fill_distance_matrix_pruned()`, scan for NaN presence before attempting pruning. Fall back to brute-force.

### C6. Problem Integration

`Problem::rebind_dtw_fn()` dispatches on `missing_strategy`:
- `Error` → standard DTW (pre-scan all series in `fillDistanceMatrix()` for NaN; throw before computing if found)
- `ZeroCost` → existing `dtwMissing_banded()`
- `AROW` → new `dtwAROW_banded()`
- `Interpolate` → `interpolate_linear()` preprocessing → standard DTW

`fillDistanceMatrix()` scans for NaN presence and adjusts strategy/pruning accordingly.

---

## Workstream D: Output Metrics

### D1. New Metrics

| Metric | Type | Input | Complexity |
|--------|------|-------|------------|
| **Dunn Index** | Internal | Distance matrix + labels | O(N^2) scan |
| **Inertia** | Internal | Distance matrix + labels | O(N) (wraps `findTotalCost()`) |
| **Calinski-Harabasz** | Internal | Distance matrix + labels | O(N^2) (find overall medoid) |
| **Adjusted Rand Index** | External | Two label vectors | O(N + k^2) |
| **Normalized Mutual Information** | External | Two label vectors | O(N + k^2) |

### D2. Calinski-Harabasz (Medoid-Adapted)

Standard CH uses Euclidean centroids. Adaptation for DTW + medoids:
- **W (within):** `sum_c sum_{x in c} d(x, medoid_c)^2`
- **B (between):** Overall medoid = `argmin_i sum_j d(i,j)` (row-sum minimizer). Then `B = sum_c |c| * d(medoid_c, overall_medoid)^2`
- **CH = (B/(k-1)) / (W/(N-k))**

Document clearly that this is a medoid substitution, not the standard centroid formula.

### D3. API

```cpp
namespace dtwc::scores {
    // Existing
    std::vector<double> silhouette(Problem &prob);
    double daviesBouldinIndex(Problem &prob);
    
    // New internal
    double dunnIndex(Problem &prob);
    double inertia(Problem &prob);
    double calinskiHarabaszIndex(Problem &prob);
    
    // New external (no Problem needed)
    double adjustedRandIndex(const std::vector<int> &true_labels,
                             const std::vector<int> &pred_labels);
    double normalizedMutualInformation(const std::vector<int> &true_labels,
                                       const std::vector<int> &pred_labels);
}
```

---

## Workstream E: Bindings

### E1. MATLAB MEX Expansion

Add to `bindings/matlab/dtwc_mex.cpp` (~15 new commands):

| Command | C++ Function |
|---------|-------------|
| `ddtw_distance` | `ddtwBanded()` |
| `wdtw_distance` | `wdtwBanded()` (param: g) |
| `adtw_distance` | `adtwBanded()` (param: penalty) |
| `soft_dtw_distance` | `soft_dtw()` (param: gamma) |
| `soft_dtw_gradient` | `soft_dtw_gradient()` |
| `dtw_distance_missing` | `dtwMissing_banded()` |
| `derivative_transform` | `derivative_transform()` |
| `z_normalize` | `core::z_normalize()` |
| `silhouette` | `scores::silhouette()` |
| `davies_bouldin_index` | `scores::daviesBouldinIndex()` |
| `dunn_index` | `scores::dunnIndex()` |
| `calinski_harabasz_index` | `scores::calinskiHarabaszIndex()` |
| `inertia` | `scores::inertia()` |
| `adjusted_rand_index` | `scores::adjustedRandIndex()` |
| `nmi` | `scores::normalizedMutualInformation()` |
| `fast_pam` | `fast_pam()` |
| `fast_clara` | `algorithms::fast_clara()` |
| `clarans` | `algorithms::clarans()` |
| `hierarchical` | `algorithms::hierarchical()` |
| `save_checkpoint` / `load_checkpoint` | Checkpoint API |
| CUDA stubs | `#ifdef DTWC_HAS_CUDA` guarded |

**Scoring MEX approach:** Pass precomputed distance matrix + labels + medoid indices directly. No persistent Problem handles.

**MATLAB `+dtwc` package updates:**
- New `.m` wrappers for all commands
- `DTWClustering.m`: add `Algorithm` property (`'pam'`/`'clara'`/`'clarans'`), scoring methods
- `compute_distance_matrix.m`: add `'Metric'`, `'Variant'`, `'UsePruning'` params

### E2. Python Updates

Expose all new algorithms and features:
- CLARANS, hierarchical clustering, mini-batch HC
- Lazy loading, HDF5 I/O, grouped data loading
- Multivariate: 2D/3D array acceptance
- New metrics (Dunn, CH, inertia, ARI, NMI)
- MissingStrategy parameter

### E3. Future: R Bindings

**Priority: Medium** (after MATLAB parity + multivariate stable).

- Rcpp + RcppArmadillo for C++ integration
- Expose core DTW, distance matrix, clustering, scoring
- R package infrastructure (`DESCRIPTION`, `NAMESPACE`, `.Rd` docs)
- testthat unit tests
- Julia: only on demonstrated demand. Rust/JS/WASM: skip.

---

## Execution Order

### Wave 1 (Parallel)

**1A: Metrics + Missing Data Foundation**
- Fix `std::isnan()` bug (C0) — immediate
- New scoring metrics (D1-D3) in `scores.hpp/cpp`
- `missing_utils.hpp` + `MissingStrategy` enum
- Study Yurtman et al. reference implementation
- DTW-AROW core (`warping_missing_arow.hpp`)
- Tests for all

**1B: Multivariate Foundation**
- `Data.ndim` + validation
- Compile-time MV metric functors
- Standard DTW with `ndim` parameter
- Stride-aware `derivative_transform`
- Benchmark: D=1 path matches current perf (within 2%)

### Wave 2 (Depends on Wave 1)

**2A: Clustering Algorithms**
- `CondensedDistanceMatrix`
- Hierarchical clustering (single, complete, average)
- CLARANS with distance cache
- Improved FastCLARA (larger samples, GPU wiring)
- `ClusteringAlgorithm` enum + dispatch
- Two-phase clustering + `load_grouped()`
- Mini-batch hierarchical

**2B: Multivariate DTW Variants + Lower Bounds**
- WDTW, ADTW, DDTW, Soft-DTW, Missing with ndim
- Per-channel LB_Keogh + LB_Kim
- SquaredL2 LB_Keogh variant

### Wave 3 (Depends on Waves 1+2)

**3A: Scale Infrastructure**
- Lazy loading (`LazyDataSource` + LRU cache)
- HDF5 binary storage (optional dep)
- mmap'd flat binary format
- Adaptive auto-selection

**3B: Bindings**
- MATLAB MEX expansion
- Python updates (all new features)
- GPU multivariate kernels (planar layout)

### Wave 4 (Future)

- R bindings (Rcpp)
- Itakura parallelogram constraint
- Cosine distance metric (with LB_Keogh disabled)

---

## Verification

1. **Unit tests:** Every new metric, AROW property, MV DTW vs hand-computed results
2. **Regression benchmarks:** D=1 multivariate path matches current perf (within 2%)
3. **Integration tests:** Clustering on synthetic multivariate datasets with known structure
4. **Cross-validation:** AROW results against Yurtman et al. reference Python code
5. **Scale tests:** CLARA/CLARANS on 10K+ synthetic datasets, verify memory stays bounded
6. **Binding tests:** Python pytest + MATLAB unit tests for every exposed function
7. **Build verification:** All optional deps remain optional, builds clean without HDF5/CUDA/OpenMP

---

## Critical Files

| Area | Files |
|------|-------|
| Clustering | `algorithms/fast_clara.hpp/cpp`, `algorithms/fast_pam.hpp/cpp`, new `algorithms/clarans.hpp/cpp`, new `algorithms/hierarchical.hpp/cpp` |
| Distance matrix | `core/distance_matrix.hpp`, new `core/condensed_distance_matrix.hpp` |
| Multivariate | `Data.hpp`, `core/time_series.hpp`, `core/distance_metric.hpp`, `warping.hpp`, all `warping_*.hpp` |
| Missing data | `warping_missing.hpp` (fix isnan), new `warping_missing_arow.hpp`, new `missing_utils.hpp`, `core/dtw_options.hpp` |
| Metrics | `scores.hpp/cpp` |
| Bindings | `bindings/matlab/dtwc_mex.cpp`, `python/src/_dtwcpp_core.cpp`, `bindings/matlab/+dtwc/*.m` |
| GPU | `cuda/cuda_dtw.cu`, `cuda/cuda_dtw.cuh` |
| I/O | `DataLoader.hpp`, `fileOperations.hpp`, new `core/hdf5_io.hpp` |
| Build | `StandardProjectSettings.cmake` (ffast-math fix scope) |

---

## Adversarial Review Summary

Three independent adversarial reviews were conducted. Key findings incorporated:

| Finding | Severity | Resolution |
|---------|----------|------------|
| Ward's linkage invalid for DTW | Critical | Dropped. Only single/complete/average. |
| GPU shared mem blowup with interleaved MV | Critical | Dual layout: interleaved CPU, planar GPU. |
| `std::isnan()` broken under `-ffast-math` | Critical | Bitwise NaN check in `missing_utils.hpp`. |
| AROW boundary cascades +inf | Critical | Study reference impl; zero-cost at boundaries. |
| CLARANS O(N^2*k) same as PAM for DTW | Major | Distance cache + early termination. Document. |
| HDF5 chunked I/O wrong for random access | Major | mmap'd flat binary for CLARANS; HDF5 for archival. |
| Runtime ndim prevents inlining | Major | Compile-time dispatch (D=1,2,3 + runtime fallback). |
| DDTW mixes channels in interleaved layout | Major | Stride-aware per-channel derivative transform. |
| NaN + MV aggregation unspecified | Major | Zero-contribution default; skip-channel opt-in. |
| AROW needs dedicated _impl | Major | New structural DP, not functor swap. |
| Interpolation boundary unspecified | Major | LOCF/NOCB edge handling. |
| CLARA sample too small for 70 clusters | Major | `max(40+2k, 10k+100)` default. |
| Scale boundaries ignore series length | Major | Adaptive cost = N^2 * min(L, band) * ndim. |
| No condensed matrix | Major | New `CondensedDistanceMatrix`. |
| Mini-batch HC loses dendrogram | Major | Document as flat clustering approximation. |
