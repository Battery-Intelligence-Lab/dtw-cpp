# DTWC++ Development TODO

This file tracks ongoing development tasks and future work items.

**Last Updated:** 2026-01-29 (Session 3 - Path refactoring complete, all 9 tests passing)

---

## Current Sprint: Milestone 0 + 1 (Hygiene & Bug Fixes) - COMPLETE

### Completed
- [x] Create cpp-style.md
- [x] Create python-style.md
- [x] Initialize TODO.md
- [x] Fix Bug #1: Missing `throw` in fileOperations.hpp:76
- [x] Fix Bug #6: `throw 2;` in fileOperations.hpp:178
- [x] Fix Bug #2: int vs size_t in parallelisation.hpp:41,44
- [x] Fix Bug #3: numMaxParallelWorkers ignored
- [x] Add tests for exception handling
- [x] Update CHANGELOG.md
- [x] **PR 1.4**: Remove DTWC_ROOT_FOLDER runtime dependency
- [x] **PR 1.5**: Add rapidcsv (v8.84) for multi-column CSV support
- [x] **PR 1.6**: Refactor to `settings::paths` namespace with runtime-settable paths

### Deferred
- [ ] Make OpenMP properly optional via CMake option (already works via `#ifdef _OPENMP`)

---

## Verified Bugs (Priority Order)

| ID | File | Line | Severity | Description | Status |
|----|------|------|----------|-------------|--------|
| #1 | fileOperations.hpp | 76 | **HIGH** | `std::runtime_error("")` constructed but not thrown | **FIXED** |
| #6 | fileOperations.hpp | 178 | Medium | `throw 2;` instead of proper exception | **FIXED** |
| #2 | parallelisation.hpp | 41, 44 | Medium | `int` loop var vs `size_t` bound | **FIXED** |
| #3 | parallelisation.hpp | 63 | Medium | `numMaxParallelWorkers` parameter ignored | **FIXED** |
| #4 | settings.hpp | 53-54 | Medium | Build-time `DTWC_ROOT_FOLDER` dependency | **FIXED** |
| #5 | fileOperations.hpp | 103 | Low | Single-column CSV only | **FIXED** (rapidcsv) |

---

## Future Milestones

### Milestone 2: CMake & Packaging
- [ ] Modern CMake target export (`dtwc::dtwc`)
- [ ] Install rules and config generation
- [ ] Single version source of truth

### Milestone 3: Core API Refactor
- [ ] Introduce `TimeSeriesView<T>`
- [ ] Introduce `DTWOptions` struct
- [ ] Distance metric abstraction
- [ ] Move internal helpers to src/

### Milestone 4: Performance & Algorithms
- [ ] dtw_full with new API
- [ ] dtw_banded (Sakoe-Chiba)
- [ ] dtw_early_abandon
- [ ] LB_Keogh lower bound
- [ ] LB_Kim lower bound
- [ ] Optimize distance matrix builder
- [ ] Microbenchmarks

### Milestone 5: LP Relaxation & Tiered MIP Solving
*(See `.claude/UNIMODULAR.md` for full research report)*

The k-medoids constraint matrix is NOT totally unimodular, but LP relaxation
is often naturally integer (especially for well-separated clusters). Strategy:

- [ ] Add LP-relaxation-first mode to HiGHS solver (skip integrality constraints)
- [ ] Add LP-relaxation-first mode to Gurobi solver
- [ ] Implement integrality check on LP solution (tolerance-based)
- [ ] Add branching priority on facility (diagonal) variables in Gurobi
- [ ] Warm-start MIP from PAM solution (provide upper bound)
- [ ] Report LP lower bound; skip MIP if PAM matches LP bound
- [ ] Implement tiered strategy: LP -> branch on A[i,i] -> full MIP fallback
- [ ] Add solver strategy enum (LP_Only, LP_then_MIP, MIP_Direct)
- [ ] Tests: verify LP = MIP for well-separated clusters
- [ ] Tests: verify tiered strategy finds optimal for hard instances

### Milestone 6: DTW with Missing Data
*(See `.claude/MISSING.md` for full literature review and implementation plan)*

**Phase 1: Foundation**
- [ ] Add `is_missing()` utility and `MISSING` constant (NaN-based)
- [ ] Add `MissingStrategy` enum (Error, ZeroCost, ZeroCostNorm, Interpolate, Skip)
- [ ] Add `DTWOptions` struct (band, missing strategy, min_coverage)
- [ ] Unit tests for missing value detection

**Phase 2: Core DTW Modifications**
- [ ] Modify `dtwFull` to handle zero-cost missing data
- [ ] Modify `dtwFull_L` to handle zero-cost missing data
- [ ] Modify `dtwBanded` to handle zero-cost missing data
- [ ] Add unified `dtw()` dispatcher based on DTWOptions
- [ ] Comprehensive unit tests with known expected values

**Phase 3: Normalization**
- [ ] Implement `ZeroCostNorm` with path-length normalization
- [ ] Implement `min_coverage` threshold (maxValue for low overlap)
- [ ] Tests comparing normalized vs unnormalized results

**Phase 4: Imputation Utilities**
- [ ] `interpolate_linear()` -- in-place, handles interior gaps
- [ ] `interpolate_spline()` -- cubic spline, in-place
- [ ] `MissingStrategy::Interpolate` dispatches interpolation before DTW
- [ ] Imputation correctness tests

**Phase 5: Problem Class Integration**
- [ ] Add `DTWOptions` member to `Problem` class
- [ ] Modify `distByInd()` to pass options through to DTW
- [ ] Coverage-based distance masking in distance matrix
- [ ] Modify CSV loader to insert NaN for non-numeric values
- [ ] Integration tests: clustering on data with missing values

### Milestone 7: Clustering Algorithms & Evaluation
- [ ] Define clustering interface
- [ ] Refactor K-medoids (PAM)
- [ ] Implement CLARA
- [ ] Hierarchical clustering
- [ ] Davies-Bouldin index
- [ ] Calinski-Harabasz index

### Milestone 8: Python Bindings
- [ ] Modernize build (scikit-build-core)
- [ ] Expose core DTW functions
- [ ] Add OOP classes (DTW, KMedoids, CLARA)
- [ ] Zero-copy NumPy support
- [ ] pytest suite
- [ ] CI wheel builds

### Milestone 9: MATLAB Interface
- [ ] Create MEX wrapper
- [ ] MATLAB OOP wrappers (mirrors Python)
- [ ] MATLAB documentation
- [ ] MATLAB tests

### Milestone 10: GPU/CUDA (Deferred)
- [ ] Backend abstraction layer
- [ ] CUDA distance matrix computation
- [ ] Thrust-based reductions
- [ ] CPU/GPU parity tests

---

## Design Notes

### Interface Consistency (Python ↔ MATLAB)
Both bindings should provide the same OOP interface:
- `DTW(band=10)` / `dtwc.DTW('band', 10)`
- `KMedoids(n_clusters=3)` / `dtwc.KMedoids('n_clusters', 3)`
- Common methods: `.distance()`, `.distance_matrix()`, `.fit()`, `.labels`, `.cluster_centers`

### GPU Compatibility Considerations
- Use `DistanceProvider` abstraction
- Keep hot-path data in contiguous memory
- Avoid std:: containers in inner loops where raw pointers needed

---

## Technical Debt

- [ ] Replace `.at()` with `operator[]` in hot paths
- [ ] Add scratch buffer mechanism for allocations
- [ ] Improve error messages throughout
- [ ] Add input validation at public API boundaries

---

## Documentation TODOs

- [ ] "How to add a new metric" guide
- [ ] "How to add a new clustering algorithm" guide
- [ ] Update installation docs for all platforms
- [ ] API reference improvements

---

## Testing TODOs

- [ ] Add edge case tests (empty arrays, single element, etc.)
- [ ] Add performance regression tests
- [ ] Increase code coverage
- [ ] Cross-platform CI validation
