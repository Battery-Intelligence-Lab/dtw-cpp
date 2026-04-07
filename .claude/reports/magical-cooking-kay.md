# Cleanup Plan: Remove Over-Engineering from dtw-cpp

## Context

Previous Claude sessions introduced unnecessary complexity: a `std::vector<bool>` to track computed distance entries (when -1 sentinel suffices), duplicate metric enums/dispatch code, public Eigen inheritance, dead code, and MSVC `/fp:fast` breaking `std::isnan`. This plan removes that complexity for a more maintainable, correct, and performant codebase.

---

## Step 1: DenseDistanceMatrix — replace `computed_` bitvector with -1 sentinel

**Files:** [distance_matrix.hpp](dtwc/core/distance_matrix.hpp)

- Remove `std::vector<bool> computed_` (line 26)
- Add `std::atomic<size_t> n_uncomputed_{0}` for O(1) `all_computed()`/`count_computed()`
- Initialize `data_` with `-1.0` instead of `0.0`
- `set()`: if `data_[k] < 0` then `n_uncomputed_.fetch_sub(1, relaxed)`, then write value
- `is_computed()`: return `data_[k] >= 0.0`
- `max()`: skip entries `< 0`
- `count_computed()`: return `packed_count() - n_uncomputed_.load(relaxed)`
- `all_computed()`: return `n_uncomputed_.load(relaxed) == 0`
- Update file header comment (remove reference to "bit-packed boolean vector" and "-ffast-math")

**Why atomic:** `set()` is called from OpenMP parallel loops with disjoint (i,j) pairs; each cell set once, but the counter is shared.

## Step 2: Remove `is_distMat_filled` from Problem

**Files:** [Problem.hpp](dtwc/Problem.hpp), [Problem.cpp](dtwc/Problem.cpp), [checkpoint.cpp](dtwc/checkpoint.cpp), [pruned_distance_matrix.cpp](dtwc/core/pruned_distance_matrix.cpp), [dtwc_cl.cpp](dtwc/dtwc_cl.cpp), [_dtwcpp_core.cpp](python/src/_dtwcpp_core.cpp), [dtwc_mex.cpp](bindings/matlab/dtwc_mex.cpp)

- Delete `bool is_distMat_filled{false}` member
- `isDistanceMatrixFilled()` becomes `return distMat.size() > 0 && distMat.all_computed();` (now O(1))
- Delete `set_distance_matrix_filled()` method and all call sites
- Remove from Python/MATLAB bindings

## Step 3: Remove dead metric enum and function

**Files:** [lower_bound_impl.hpp](dtwc/core/lower_bound_impl.hpp)

- Delete `enum class DistanceMetric` (line 326) and `lb_pruning_compatible()` (lines 329-332)
- Delete any test references

## Step 4: Metric dispatch deduplication

**Files:** [warping.hpp](dtwc/warping.hpp), [warping_missing.hpp](dtwc/warping_missing.hpp), [warping_missing_arow.hpp](dtwc/warping_missing_arow.hpp)

Add `dispatch_scalar_metric` and `dispatch_mv_metric` helpers in `detail` namespace (single switch each). Replace 18+ identical switch blocks with one-liner lambdas:
```cpp
template <typename Fn>
auto dispatch_scalar_metric(core::MetricType m, Fn&& fn) -> decltype(fn(L1Dist{})) {
  switch (m) {
  case core::MetricType::SquaredL2: return fn(SquaredL2Dist{});
  default: return fn(L1Dist{});
  }
}
```

## Step 5: ScratchMatrix — private inheritance

**File:** [scratch_matrix.hpp](dtwc/core/scratch_matrix.hpp)

- Change `public Eigen::Matrix` to `private Eigen::Matrix`
- Add `using Base::operator(); using Base::data; using Base::size; using Base::rows; using Base::cols;`

## Step 6: MSVC `/fp:fast` fix

**File:** [StandardProjectSettings.cmake](cmake/StandardProjectSettings.cmake)

- Replace `/fp:fast` with `/fp:precise /fp:contract` (preserves NaN semantics, keeps FMA)
- Add target-level `-fno-finite-math-only` on the library target for subproject safety

## Step 7: Minor cleanups

| What | File | Change |
|------|------|--------|
| `thread_local` 10000 prealloc | [warping_adtw.hpp:54](dtwc/warping_adtw.hpp#L54) | Remove initial size: `thread_local static std::vector<data_t> short_side;` |
| Useless `thread_local` | [scores.cpp:62](dtwc/scores.cpp#L62) | Make plain local, remove redundant `.assign()` |
| Unused `#include <numeric>` | [warping_wdtw.hpp:30](dtwc/warping_wdtw.hpp#L30) | Delete the include |
| Band-boundary duplication | 5 files | Extract `band_bounds(slope, window, row)` helper, replace 5 copy-paste sites |
| `atomic_min_double` aliasing | [pruned_distance_matrix.cpp:60](dtwc/core/pruned_distance_matrix.cpp#L60) | Add diagnostic pragma to suppress strict-aliasing warning; document the reliance |

## Step 8: Delete stale reports/summaries

Delete all 6 obsolete files:
- `.claude/reports/dapper-gliding-candle.md`
- `.claude/summaries/2026-04-04-session.md`
- `.claude/summaries/2026-04-04-simd-session.md`
- `.claude/summaries/2026-04-04-hpc-build-session.md`
- `.claude/summaries/2026-04-04-simd-improvements.md`
- `.claude/summaries/2026-04-04-highway-removal.md`

Update LESSONS.md: remove references to deleted Highway SIMD files.

---

## Execution Order

Steps 1-2 must be sequential (2 depends on 1). Steps 3-8 are independent and can be parallelized. Step 4 (dispatch dedup) is the largest diff.

## Verification

- Full CMake build (GCC/Clang + MSVC if available)
- Full test suite (`ctest`)
- Grep for `computed_`, `is_distMat_filled`, `DistanceMetric`, `lb_pruning_compatible` to confirm zero residual references
- Benchmark DTW distance computation to confirm no regression
