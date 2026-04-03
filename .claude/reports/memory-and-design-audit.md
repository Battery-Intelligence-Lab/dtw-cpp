# DTWC++ Memory & Design Audit Report

**Date:** 2026-04-03
**Auditor:** Claude (glm-5.1)
**Scope:** C++ hot path allocations, Python/MATLAB binding copy analysis, design simplifications

---

## Table of Contents

1. [C++ Hot Path: Unnecessary Memory Allocations](#1-c-hot-path-unnecessary-memory-allocations)
2. [Python Binding Memory Analysis](#2-python-binding-memory-analysis)
3. [MATLAB Binding Memory Analysis](#3-matlab-binding-memory-analysis)
4. [Design Simplifications & Corrections](#4-design-simplifications--corrections)
5. [Priority Summary](#5-priority-summary)

---

## 1. C++ Hot Path: Unnecessary Memory Allocations

### 1.1 WDTW Weight Vector Recomputed on Every DTW Call [HIGH]

**File:** `dtwc/warping_wdtw.hpp:48-56, 262-268`

The convenience overloads `wdtwBanded(x, y, band, g)` and `wdtwFull(x, y, g)` call `wdtw_weights()` which allocates a new `std::vector<data_t>` every single time:

```cpp
template <typename data_t = double>
data_t wdtwBanded(const std::vector<data_t> &x, const std::vector<data_t> &y,
                  int band, data_t g)
{
  const int max_dev = static_cast<int>(std::max(x.size(), y.size()));
  auto w = wdtw_weights<data_t>(max_dev, g);  // <-- ALLOCATION per call
  return wdtwBanded(x, y, w, band);
}
```

During distance matrix computation (N*(N-1)/2 pairs), this means N^2 allocations. Each allocation involves `exp()` calls for `max_dev` elements plus heap allocation.

**Fix:** Cache the weight vector in `Problem` (keyed by `max_series_length` and `g`), or pass precomputed weights through the `dtw_fn_` lambda. The weight vector only depends on `max_dev` and `g` — it should be computed once per `Problem`, not once per DTW call.

**Impact:** For N=1000, that's ~500K heap allocations eliminated. Each `wdtw_weights` also computes `exp()` `max_dev` times. With caching, this becomes a single allocation of size `max_series_length`.

### 1.2 DDTW Derivative Transform Allocates Per Call [MEDIUM]

**File:** `dtwc/warping_ddtw.hpp:45-66, 127-131`

```cpp
template <typename data_t>
std::vector<data_t> derivative_transform(const std::vector<data_t> &x)
{
  std::vector<data_t> dx(n);  // <-- ALLOCATION
  // ... fill dx ...
  return dx;
}
```

The `ddtwBanded()` function uses `thread_local` buffers:
```cpp
thread_local std::vector<data_t> dx, dy;
dx = derivative_transform(x);  // <-- move from new vector into thread_local
dy = derivative_transform(y);  // <-- same
```

The `derivative_transform()` always creates a new vector, which is then move-assigned into the thread_local. This means 2 heap allocations per DTW call (one for each series' derivative). The move-assignment is efficient but the allocation/deallocation in `derivative_transform` is unnecessary.

**Fix:** Add an in-place derivative transform that writes into a pre-existing buffer:
```cpp
template <typename data_t>
void derivative_transform_inplace(const std::vector<data_t> &x, std::vector<data_t> &dx)
{
  dx.resize(x.size());  // no-op if already the right size
  // ... fill dx ...
}
```

**Impact:** 2 heap allocations/deallocations per DDTW call eliminated. For N=1000, that's ~1M fewer heap operations during distance matrix computation.

### 1.3 MV WDTW/ADTW ndim==1 Fallback Copies from Raw Pointers [MEDIUM]

**File:** `dtwc/warping_wdtw.hpp:303-306`, `dtwc/warping_adtw.hpp:253-255`

```cpp
// wdtwBanded_mv when ndim==1:
std::vector<data_t> vx(x, x + nx_steps), vy(y, y + ny_steps);  // <-- COPIES
return wdtwBanded(vx, vy, band, g);
```

When `ndim==1`, the multivariate overloads materialize full copies of the input data into `std::vector`s, then delegate to the scalar overload. But the scalar overloads have pointer-based versions that accept raw pointers directly.

**Fix:** For `ndim==1`, call the pointer+length overloads directly:
```cpp
if (ndim == 1) return wdtwBanded(x, nx_steps, y, ny_steps, band, -1.0, core::MetricType::L1);
```
The same pattern applies to `wdtwFull_mv`, `adtwBanded_mv`, and `adtwFull_L_mv`.

**Impact:** Eliminates 2 full series copies per DTW call when going through MV paths with ndim=1. Particularly important in the `rebind_dtw_fn()` DDTW MV path where `derivative_transform_mv` returns a vector that then gets copied again.

### 1.4 WDTW MV Banded Falls Back to Unbanded [MEDIUM]

**File:** `dtwc/warping_wdtw.hpp:378-379`, `dtwc/warping_adtw.hpp:257-258`

```cpp
// For banded MV WDTW, delegate to full MV (optimize later if needed)
return wdtwFull_mv(x, nx_steps, y, ny_steps, ndim, g);
```

When `band >= 0` and `ndim > 1`, both `wdtwBanded_mv` and `adtwBanded_mv` silently fall back to the full (unbanded) implementation. The user specifies a band constraint but gets O(n*m) computation instead of O(n*band). This is both a **performance** and **correctness** concern — the user expects banded computation.

**Fix:** Implement `wdtwBanded_mv_impl` and `adtwBanded_mv_impl` using the same rolling-buffer pattern as `dtwBanded_mv_impl` in `warping.hpp`. The banded MV implementations for standard DTW exist; the WDTW/ADTW variants need the same treatment.

**Impact:** For band=10% and n=1000, this is a ~10x speedup for MV WDTW/ADTW distance matrices.

### 1.5 `thread_local` Buffer Proliferation [LOW]

**Files:** `warping.hpp:95,172,256,321`, `warping_wdtw.hpp:84,164,330`, `soft_dtw.hpp:80,133,167`

Each DTW variant maintains its own set of `thread_local` scratch buffers:
- `dtwFull_L_impl`: `thread_local static std::vector<data_t> short_side`
- `dtwBanded_impl`: `thread_local std::vector<data_t> col` + `thread_local std::vector<int> low_bounds` + `thread_local std::vector<int> high_bounds`
- `dtwBanded_mv_impl`: `thread_local std::vector<data_t> col_mv` + `thread_local std::vector<int> low_bounds_mv` + ...
- `wdtwBanded`: `thread_local std::vector<data_t> col`
- `soft_dtw`: `thread_local core::ScratchMatrix<T> C` + `E`
- `ddtwBanded`: `thread_local std::vector<data_t> dx, dy`

Each `thread_local` is per-thread, so with 16 OpenMP threads, that's 16× the buffer count. When switching between DTW variants (e.g., DDTW calling dtwBanded internally), both sets of buffers are retained.

**Fix:** Consider a unified scratch allocator per thread that manages a single memory pool, or at minimum share the rolling buffers where possible. This is a low priority optimization since the buffers "resize, never shrink" and typically fit in L1/L2 cache.

**Impact:** Memory footprint reduction, especially with many threads and large series.

---

## 2. Python Binding Memory Analysis

### 2.1 Zero-Copy Functions (GOOD — No Changes Needed)

These functions use `nb::ndarray<const double, nb::ndim<1>, nb::c_contig>` which gives direct pointer access to numpy memory:

- `dtw_distance` (line 266-277)
- `dtw_distance_missing` (line 319-332)
- `dtw_arow_distance` (line 334-352)
- `set_distance_matrix_from_numpy` (line 463-478)
- `DenseDistanceMatrix::to_numpy` (line 250-255) — `reference_internal` policy, true zero-copy

### 2.2 Functions with Unnecessary Copies [HIGH]

These functions take `const std::vector<double> &x` parameters, forcing nanobind to allocate a new vector and copy from the numpy array:

| Function | Lines | Copies per call |
|----------|-------|-----------------|
| `ddtw_distance` | 279-285 | 2 (x, y) |
| `wdtw_distance` | 287-293 | 2 (x, y) |
| `adtw_distance` | 295-301 | 2 (x, y) |
| `soft_dtw_distance` | 303-309 | 2 (x, y) |
| `soft_dtw_gradient` | 311-317 | 2 (x, y) |
| `derivative_transform` | 358-360 | 1 (x) |
| `z_normalize` | 362-365 | 1 (x, by value!) |

**Fix:** Replace `const std::vector<double> &x` with `nb::ndarray<const double, nb::ndim<1>, nb::c_contig> x` and pass `x.data()` / `x.size()` to the pointer-based C++ overloads. Example:

```cpp
m.def("ddtw_distance", [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> x,
                           nb::ndarray<const double, nb::ndim<1>, nb::c_contig> y,
                           int band) {
  nb::gil_scoped_release release;
  // Call pointer-based overload directly:
  return dtwc::ddtwBanded<double>(x.data(), x.size(), y.data(), y.size(), band);
}, "x"_a, "y"_a, "band"_a = -1);
```

Note: `ddtwBanded` currently only has a vector overload, not a pointer overload. It would need one added. Alternatively, call `derivative_transform` on raw pointers and then `dtwBanded` on the resulting vectors.

**Impact:** Eliminates 2 heap allocations + memcpy per DTW call. For large series (e.g., n=10K), that's 80KB of copying eliminated per call.

### 2.3 `distance_matrix_numpy()` — Double Copy + O(N^2) Function Calls [HIGH]

**File:** `python/src/_dtwcpp_core.cpp:432-442`

```cpp
.def("distance_matrix_numpy", [](dtwc::Problem &prob) {
    prob.fillDistanceMatrix();
    const size_t n = prob.size();
    double* ptr = new double[n * n];
    for (size_t i = 0; i < n; ++i)
        for (size_t j = 0; j < n; ++j)
            ptr[i * n + j] = prob.distByInd(static_cast<int>(i), static_cast<int>(j));
    ...
```

This:
1. Fills the distance matrix in C++
2. Allocates a new N*N buffer
3. Copies element-by-element using `distByInd()` which does bounds checking + computed-flag checking per element

**Fix:** Use the existing `DenseDistanceMatrix::to_numpy()` zero-copy method, or `std::memcpy` from `prob.distance_matrix().raw()`:

```cpp
.def("distance_matrix_numpy", [](dtwc::Problem &prob) {
    nb::gil_scoped_release release;
    prob.fillDistanceMatrix();
    nb::gil_scoped_acquire acquire;
    const auto &dm = prob.distance_matrix();
    size_t n = dm.size();
    size_t shape[2] = {n, n};
    // Note: need to transpose from column-major to row-major, OR document layout
    return nb::ndarray<nb::numpy, double>(dm.raw(), 2, shape, nb::handle());
}, nb::rv_policy::reference_internal);
```

**Impact:** Eliminates N^2 function calls + N^2 double copies. For N=1000, that's 1M `distByInd()` calls eliminated. Note: `DenseDistanceMatrix` is column-major; numpy expects row-major by default. Either document the layout or transpose.

### 2.4 `compute_distance_matrix()` — Input Data Copy [MEDIUM]

**File:** `python/src/_dtwcpp_core.cpp:488-533`

```cpp
m.def("compute_distance_matrix", [](const std::vector<std::vector<double>> &series, ...)
```

nanobind converts the Python list-of-lists (or 2D numpy array) to `std::vector<std::vector<double>>`, which requires a full copy of the entire dataset.

**Fix:** Accept `nb::ndarray<const double, nb::ndim<2>, nb::c_contig>` (2D numpy array), then provide pointer-based access to each row. This requires the user to pass a 2D numpy array (same-length series) instead of a list of variable-length series. Could be offered as a fast-path alternative alongside the existing general-purpose function.

**Impact:** For N=1000, L=500, this eliminates 500K doubles (4MB) of copying.

### 2.5 `z_normalize` — Takes By Value + Returns New Vector [LOW]

**File:** `python/src/_dtwcpp_core.cpp:362-365`

```cpp
m.def("z_normalize", [](std::vector<double> x) {  // <-- COPY (by value)
    dtwc::core::z_normalize(x.data(), x.size());
    return x;  // <-- converted to Python list (another copy)
}, ...);
```

The parameter is taken by value (copy), then normalized in place, then returned as a Python list (another copy). Total: 2 copies.

**Fix:** Accept `nb::ndarray`, operate in-place, or return a numpy array directly.

### 2.6 GIL Release Pattern — Correctly Applied [GOOD]

All computationally expensive functions properly release the GIL (`nb::gil_scoped_release release`) before calling C++ code and re-acquire it before manipulating Python objects. This is correct.

---

## 3. MATLAB Binding Memory Analysis

### 3.1 Matrix-to-Series Conversion — Unavoidable Copy [KNOWN LIMITATION]

**File:** `bindings/matlab/dtwc_mex.cpp:98-109`

```cpp
static std::vector<std::vector<double>> matrix_to_series(const mxArray *mx) {
  size_t N = mxGetM(mx);  // rows = number of series
  size_t L = mxGetN(mx);  // cols = series length
  const double *data = mxGetDoubles(mx);
  std::vector<std::vector<double>> series(N, std::vector<double>(L));
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < L; ++j)
      series[i][j] = data[i + j * N];  // column-major to row-major
  return series;
}
```

This is necessary because MATLAB uses column-major layout while C++ uses row-major. The transposition copy is unavoidable with the legacy C MEX API.

**Potential fix (if worth it):** Store data in column-major order internally and adjust indexing, OR use the modern C++ MEX API (since R2018a) which provides `matlab::data::TypedArray<double>` with potential zero-copy access.

### 3.2 Stateless DTW Functions Copy Per Call [MEDIUM]

**File:** `bindings/matlab/dtwc_mex.cpp:496-575`

Every stateless DTW function (`cmd_dtw_distance`, `cmd_ddtw_distance`, etc.) copies the MATLAB arrays to `std::vector`:

```cpp
static void cmd_dtw_distance(...) {
  auto x = to_std_vector(prhs[1]);  // <-- COPY
  auto y = to_std_vector(prhs[2]);  // <-- COPY
  plhs[0] = mxCreateDoubleScalar(dtwc::dtwBanded<double>(x, y, band));
}
```

**Fix:** Use `mxGetDoubles()` directly with pointer-based C++ overloads:

```cpp
static void cmd_dtw_distance(...) {
  const double *x = mxGetDoubles(prhs[1]);
  size_t nx = mxGetNumberOfElements(prhs[1]);
  const double *y = mxGetDoubles(prhs[2]);
  size_t ny = mxGetNumberOfElements(prhs[2]);
  int band = dtwc::settings::DEFAULT_BAND_LENGTH;
  if (nrhs > 3) band = static_cast<int>(get_scalar(prhs[3]));
  plhs[0] = mxCreateDoubleScalar(dtwc::dtwBanded<double>(x, nx, y, ny, band));
}
```

This eliminates 2 copies per DTW call.

**Impact:** 2 heap allocations + memcpy per DTW call eliminated. The MATLAB overhead of MEX dispatch dominates for single calls, but for tight loops calling DTW many times, this matters.

### 3.3 Distance Matrix Retrieval — Copy [LOW]

**File:** `bindings/matlab/dtwc_mex.cpp:456-470`

`cmd_Problem_get_distance_matrix()` copies element-by-element from `DenseDistanceMatrix` to a MATLAB matrix. Since the C++ storage is column-major (matching MATLAB), this could be a `memcpy`.

**Fix:** Replace the double loop with:
```cpp
std::memcpy(out, dm.raw(), N * N * sizeof(double));
```

### 3.4 Longjmp-Safe Error Handling — Correct [GOOD]

The MEX gateway correctly catches C++ exceptions before calling `mexErrMsgIdAndTxt`, ensuring RAII destructors run before the longjmp. This follows the pattern documented in LESSONS.md.

---

## 4. Design Simplifications & Corrections

### 4.1 Triple Metric Type System [SIMPLIFY]

The codebase has THREE overlapping metric abstractions:

1. **`core::MetricType`** (enum: L1, L2, SquaredL2) — used in `dtw_options.hpp`, runtime dispatch in `warping.hpp`
2. **`core::DistanceMetric`** (enum: L1, L2, SquaredL2) — used in `lower_bound_impl.hpp`, separate from #1
3. **Functor types** — `L1Metric`, `L2Metric`, `SquaredL2Metric` (in `distance_metric.hpp`) AND `L1Dist`, `SquaredL2Dist`, `MVL1Dist`, `MVSquaredL2Dist` (in `warping.hpp` detail namespace)

The functors in `distance_metric.hpp` (`L1Metric`, `L2Metric`, `SquaredL2Metric`) are **never used in any hot path** — `warping.hpp` uses its own `L1Dist`/`SquaredL2Dist`. The `core::DistanceMetric` enum in `lower_bound_impl.hpp` duplicates `core::MetricType`.

**Recommendation:** Consolidate to:
- One runtime metric enum (`core::MetricType`) for bindings
- One set of compile-time functors in `warping.hpp` detail namespace (already the de facto standard)
- Remove `distance_metric.hpp` functors and `DistanceMetric` enum, or merge them

### 4.2 `L2Metric` == `L1Metric` Confusion [DOCUMENT]

**File:** `dtwc/core/distance_metric.hpp:27-32`

```cpp
struct L2Metric
{
  template <typename T>
  T operator()(T a, T b) const noexcept { return std::abs(a - b); }
};
```

`L2Metric::operator()` returns `|a-b|` (same as L1). The comment "sqrt((a-b)^2) == |a-b| for scalars" is mathematically correct, but exposing this as a separate metric type is misleading. Users might expect L2 to mean `sqrt(sum_of_squared_diffs)` across multivariate features.

**Recommendation:** Either remove `L2` as a separate metric (since it's identical to L1 for scalar DTW), or document clearly that L2 in this context means "Euclidean pointwise distance for scalars (which equals L1)", and users who want squared-L2 DTW should use `SquaredL2` and take `sqrt` of the result.

### 4.3 DenseDistanceMatrix CSV I/O in Core Header [SEPARATE]

**File:** `dtwc/core/distance_matrix.hpp:115-174`

`write_csv()` and `read_csv()` methods include `<filesystem>`, `<fstream>`, `<iomanip>`, `<sstream>`, `<string>` — all I/O headers — in a core data structure header. This forces these heavy dependencies into every compilation unit that includes `distance_matrix.hpp`.

**Recommendation:** Move CSV I/O to a separate `distance_matrix_io.hpp` or free functions. Keep the core header minimal.

### 4.4 `DenseDistanceMatrix::computed_` Tracking Uses Full Bytes [LOW]

**File:** `dtwc/core/distance_matrix.hpp:33`

```cpp
std::vector<char> computed_;  // 1 byte per entry
```

For N=10K, this uses 100MB of tracking data. Using a `std::vector<bool>` (bit-packed) or `std::bitset` would reduce this to ~12.5MB.

**Recommendation:** Replace `std::vector<char>` with bit-packed storage. The `is_computed()` check is in the hot path of `distByInd()`, so benchmark first — bit manipulation may add latency that negates the cache benefit.

### 4.5 `compute_distance_matrix_pruned` Standalone — Missing nn_dist[j] Update [BUG]

**File:** `dtwc/core/pruned_distance_matrix.cpp:406`

```cpp
if (dist < nn_dist[i]) nn_dist[i] = dist;
// nn_dist[j] is NEVER updated!
```

The Problem-based version (`fill_distance_matrix_pruned`) correctly updates both `nn_dist[i]` and `nn_dist[j]` via `atomic_min_double`. The standalone version (used by Python bindings) only updates `nn_dist[i]`. This means `nn_dist[j]` never decreases, reducing pruning effectiveness for later pairs involving series j.

**Fix:** Add `atomic_min_double(&nn_dist[j], dist)` or, since each thread "owns" row i only, use a relaxed atomic store:
```cpp
// Note: safe because nn_dist is only used for pruning (approximate is OK)
double old_j = nn_dist[j];  // may be stale
if (dist < old_j) nn_dist[j] = dist;  // benign race, improves pruning
```

**Impact:** Pruning ratio may improve significantly (10-30% more pairs pruned) because nn_dist values actually tighten.

### 4.6 `wdtwFull_mv` Creates Unnecessary Weight Vector For ndim==1 [LOW]

**File:** `dtwc/warping_wdtw.hpp:303-307`

When `ndim==1`, `wdtwFull_mv` creates a weight vector and calls the scalar `wdtwFull`. But it also first copies the raw pointer data into vectors. Two inefficiencies for one call.

### 4.7 `std::function` in Problem — Correct but Worth Noting

**File:** `dtwc/Problem.hpp:80`

```cpp
using dtw_fn_t = std::function<data_t(const std::vector<data_t>&, const std::vector<data_t>&)>;
```

`std::function` has ~2ns overhead per call (virtual dispatch + potential allocation for large captures). This is documented in `design.md` as "negligible vs 1-100ms DTW" — correct. The current lambda captures only `this` (8 bytes), well within the small-buffer optimization. **No change needed.**

However, for the WDTW case, the lambda should capture a pre-computed weight vector to avoid the per-call allocation described in [1.1].

### 4.8 `compute_envelopes` O(n*band) → O(n) Potential [LOW]

**File:** `dtwc/core/lower_bound_impl.hpp:56-67`

```cpp
for (std::size_t p = 0; p < n; ++p) {
    const std::size_t lo = (p >= w) ? p - w : 0;
    const std::size_t hi = std::min(p + w + 1, n);
    T max_val = series[lo];
    T min_val = series[lo];
    for (std::size_t j = lo + 1; j < hi; ++j) {  // O(band) per point
```

This is O(n * band). A monotonic deque would give O(n). For typical band widths (10% of n), the constant factor is small and the current approach is cache-friendly. Not a priority.

### 4.9 Banded MV Missing DTW Falls Back to Unbanded [TODO]

**Files:** `dtwc/warping_missing.hpp:307-309`

```cpp
// TODO: implement a full banded MV missing variant
return dtwMissing_L_mv(x, nx_steps, y, ny_steps, ndim, early_abandon, metric);
```

Same pattern as WDTW/ADTW — banded MV is deferred. When `band >= 0` and `ndim > 1`, the user gets O(n*m) instead of O(n*band).

### 4.10 MATLAB MEX String Dispatch — Linear Scan [COSMETIC]

**File:** `bindings/matlab/dtwc_mex.cpp:864-919`

The command dispatch is a ~45-case `if/else if` chain. Worst case: 45 string comparisons. This is dominated by MEX call overhead (~microseconds) so not a real performance concern. For cleanliness, a `static std::unordered_map<std::string, function_ptr>` would be O(1).

---

## 5. Priority Summary

### HIGH Priority (Performance Impact)

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 1.1 | WDTW weight recomputed per call | `warping_wdtw.hpp:262` | N^2 allocations during dist matrix |
| 2.2 | Python DTW variants copy inputs | `_dtwcpp_core.cpp:279-317` | 2 heap allocs + memcpy per call |
| 2.3 | `distance_matrix_numpy()` double copy | `_dtwcpp_core.cpp:432-442` | N^2 function calls + N^2 doubles copied |
| 4.5 | Standalone pruned DM missing nn_dist[j] update | `pruned_distance_matrix.cpp:406` | Reduced pruning ratio |

### MEDIUM Priority (Correctness + Performance)

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 1.2 | DDTW derivative allocates per call | `warping_ddtw.hpp:45-66` | 2 allocs per DDTW call |
| 1.3 | MV ndim==1 fallback copies from pointers | `warping_wdtw.hpp:303`, `warping_adtw.hpp:253` | Full series copy per call |
| 1.4 | MV WDTW/ADTW banded falls back to unbanded | `warping_wdtw.hpp:378`, `warping_adtw.hpp:257` | ~10x slower than expected |
| 3.2 | MATLAB DTW functions copy inputs | `dtwc_mex.cpp:496-575` | 2 copies per call |
| 2.4 | Python `compute_distance_matrix` copies dataset | `_dtwcpp_core.cpp:488` | Full dataset copy |

### LOW Priority (Cleanup + Polish)

| # | Issue | Location | Impact |
|---|-------|----------|--------|
| 1.5 | thread_local buffer proliferation | Multiple files | Memory waste per thread |
| 2.5 | `z_normalize` double copy | `_dtwcpp_core.cpp:362` | 2 copies per call |
| 3.3 | MATLAB DM retrieval could be memcpy | `dtwc_mex.cpp:456-470` | Minor speedup |
| 4.1 | Triple metric type system | Multiple files | Maintenance burden |
| 4.2 | L2Metric == L1Metric confusion | `distance_metric.hpp:27-32` | User confusion |
| 4.3 | CSV I/O in core header | `distance_matrix.hpp:115` | Compile time / deps |
| 4.4 | computed_ uses full bytes | `distance_matrix.hpp:33` | 8x memory waste |
| 4.9 | Banded MV missing DTW unbanded | `warping_missing.hpp:307` | Performance |

---

## Appendix: What's Already Done Well

These patterns are **correct and should be preserved**:

---

## 6. Continuation Log (Codex Session - 2026-04-03)

This section records the work completed after the original audit so the benchmark and refactor trail survives context limits.

### 6.1 Eigen Decision

Eigen is still **not** recommended as the first refactor for performance. The current custom matrix wrappers are not the primary hot-path problem:

- `dtwc/core/scratch_matrix.hpp` is already a very small flat contiguous wrapper.
- `dtwc/core/distance_matrix.hpp` carries custom semantics beyond storage: symmetric writes, `computed_` tracking, deferred allocation, and CSV helpers.
- The main maintenance burden is duplicated recurrence code across `warping.hpp`, `warping_wdtw.hpp`, `warping_adtw.hpp`, and `warping_ddtw.hpp`, not the matrix wrapper itself.

Eigen may still make sense later for selected full-matrix paths where maintainability matters more than raw hot-loop control, for example `soft_dtw.hpp` or the full-matrix ADTW scratch path. It should not be the first optimization wave.

### 6.2 Benchmark Coverage Added

The benchmark harness in `benchmarks/bench_dtw_baseline.cpp` was extended to cover the allocation-heavy and variant-specific paths that were missing from the original suite:

- `BM_wdtwBanded_g`
- `BM_wdtwBanded_precomputed`
- `BM_ddtwBanded`
- `BM_adtwBanded`
- `BM_wdtwBanded_mv_ndim1`
- `BM_adtwBanded_mv_ndim1`
- `BM_fillDistanceMatrix_variant`

Result artifacts captured during this session:

- `benchmarks/results/baseline_head_20260403.json`
- `benchmarks/results/variant_alloc_baseline_20260403_before.json`
- `benchmarks/results/variant_alloc_baseline_20260403_after.json`

### 6.3 Wave 1 Refactor Implemented

The first pass focused on eliminating per-call allocations and copy-heavy fallbacks without changing public behavior.

#### DDTW buffer reuse

File: `dtwc/warping_ddtw.hpp`

- Added `derivative_transform_inplace(...)`
- Added `derivative_transform_mv_inplace(...)`
- Reworked `ddtwBanded(...)` and `ddtwFull_L(...)` to reuse `thread_local` derivative buffers instead of allocating fresh vectors on every call

#### WDTW scalar pointer paths and copy removal

File: `dtwc/warping_wdtw.hpp`

- Added scalar pointer overloads for `wdtwFull(...)` and `wdtwBanded(...)`
- Added MV overloads that accept precomputed weights
- Reworked `ndim == 1` MV fallback to call the scalar pointer path directly instead of copying into temporary `std::vector`s
- Preserved historical MV semantics where `max_dev = max_steps - 1` for the `g` convenience path

#### ADTW scalar pointer paths and copy removal

File: `dtwc/warping_adtw.hpp`

- Added scalar pointer overloads for `adtwFull_L(...)` and `adtwBanded(...)`
- Reworked `ndim == 1` MV fallback to use the scalar pointer path directly

#### WDTW cache moved into `Problem`

Files: `dtwc/Problem.hpp`, `dtwc/Problem.cpp`

- Added `wdtw_weights_cache_` keyed by `max_dev`
- Added `refresh_variant_caches()`
- Rebound WDTW `Problem` lambdas to reuse cached weights instead of rebuilding them on each distance evaluation

Important constraint noted during this pass: `Problem::fillDistanceMatrix_BruteForce()` runs under OpenMP, so mutable scratch buffers must remain per-thread or `thread_local`. A single shared `Problem` scratch area would not be safe for those paths.

### 6.4 Behavioral Fixes Required During Refactor

Two regressions appeared while introducing the pointer-based overloads and were fixed immediately:

1. Empty-input handling

- Initial pointer overloads checked `x == y && nx == ny` before checking for empty inputs.
- With empty vectors this could incorrectly return `0` because both data pointers compare equal.
- Fixed by checking `nx == 0 || ny == 0` first and preserving prior behavior.

2. MV WDTW weighting semantics

- Initial refactor made `wdtwFull_mv(ndim == 1, g)` use the scalar `g` convenience weighting.
- Historical MV behavior instead used `max_dev = max_steps - 1`.
- Fixed so the ndim-1 path matches previous MV semantics and existing tests.

### 6.5 Measurements

The clearest wins from Wave 1 came from the allocation-heavy microbenchmarks.

Before -> after:

- `DDTW banded 1000/50`: `609.38 us -> 530.13 us` (`-13.0%`)
- `DDTW banded 4000/50`: `2508.36 us -> 2299.33 us` (`-8.3%`)
- `WDTW MV ndim==1 1000/50`: `558.04 us -> 531.25 us` (`-4.8%`)
- `WDTW MV ndim==1 4000/50`: `2403.85 us -> 2083.33 us` (`-13.3%`)
- `ADTW MV ndim==1 1000/50`: `1032.37 us -> 857.60 us` (`-16.9%`)
- `ADTW MV ndim==1 4000/50`: `11997.77 us -> 11160.71 us` (`-7.0%`)

The current end-to-end `BM_fillDistanceMatrix_variant(50, 500, 50, ...)` case was noisy and did **not** show a clear WDTW/DDTW gain. That likely means this specific case is dominated by DP work, OpenMP scheduling, and dataset shape rather than the allocations that were removed.

### 6.6 Verification

The following unit tests were built and passed after the refactor:

- `unit_test_wdtw`
- `unit_test_ddtw`
- `unit_test_adtw`
- `unit_test_mv_variants`

### 6.7 Current Recommended Next Wave

The next performance wave should target the remaining larger structural misses, in this order:

1. Implement true banded MV WDTW

- Current MV WDTW still falls back to full MV for `ndim > 1`.
- This is likely a much larger end-to-end gain than further scalar allocation cleanup.

2. Implement true banded MV ADTW

- Same issue as WDTW: the multivariate banded API still degrades to the full path.

3. Add more sensitive end-to-end benchmark shapes

- Include smaller series, variable-length series, and WDTW-heavy cases where per-call setup costs are a larger share of total runtime.

4. Fix the standalone pruning bug

- `dtwc/core/pruned_distance_matrix.cpp` still appears to miss the `nn_dist[j]` update in the standalone pruning path noted in Section 4.5.

5. Revisit selective Eigen use only after the above

- If code-size reduction is still a goal, evaluate Eigen only for full-matrix non-hot paths.
- Do not replace rolling-buffer or custom symmetric matrix storage blindly.

1. **Pointer+length overloads in core DTW** (`warping.hpp`) — enables zero-copy from bindings
2. **`thread_local` scratch buffers** — resize-never-shrink, avoids per-call allocation
3. **Rolling buffer O(min(n,m)) space** for linear DTW — memory-optimal
4. **Deferred distance matrix allocation** — `refreshDistanceMatrix()` doesn't force N^2 memory
5. **Lock-free parallel design** — row-based partitioning avoids synchronization
6. **DenseDistanceMatrix::to_numpy()** — true zero-copy with `reference_internal`
7. **GIL release** — properly applied for all C++ compute calls
8. **nanobind over pybind11** — smaller binaries, native ndarray support
9. **mexLock + longjmp-safe error handling** — prevents DLL unload crashes
10. **NaN handling via `is_missing()`** — bitwise check, safe under `-ffast-math`
