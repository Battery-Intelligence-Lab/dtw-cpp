# DTWC++ Development TODO

This file tracks ongoing development tasks and future work items.

**Last Updated:** 2026-01-29 (Build verified, all 9 tests passing)

---

## Current Sprint: Milestone 0 + 1 (Hygiene & Bug Fixes)

### In Progress
- [ ] Verify build and tests pass

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

### Pending
- [ ] Make OpenMP properly optional via CMake (DTWC_ENABLE_OPENMP option)

---

## Verified Bugs (Priority Order)

| ID | File | Line | Severity | Description | Status |
|----|------|------|----------|-------------|--------|
| #1 | fileOperations.hpp | 76 | **HIGH** | `std::runtime_error("")` constructed but not thrown | **FIXED** |
| #6 | fileOperations.hpp | 178 | Medium | `throw 2;` instead of proper exception | **FIXED** |
| #2 | parallelisation.hpp | 41, 44 | Medium | `int` loop var vs `size_t` bound | **FIXED** |
| #3 | parallelisation.hpp | 63 | Medium | `numMaxParallelWorkers` parameter ignored | **FIXED** |
| #4 | settings.hpp | 53-54 | Medium | Build-time `DTWC_ROOT_FOLDER` dependency | Deferred |
| #5 | fileOperations.hpp | 103 | Low | Single-column CSV only | Deferred |

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

### Milestone 5: Clustering & Evaluation
- [ ] Define clustering interface
- [ ] Refactor K-medoids (PAM)
- [ ] Implement CLARA
- [ ] Hierarchical clustering
- [ ] Davies-Bouldin index
- [ ] Calinski-Harabasz index

### Milestone 6: Python Bindings
- [ ] Modernize build (scikit-build-core)
- [ ] Expose core DTW functions
- [ ] Add OOP classes (DTW, KMedoids, CLARA)
- [ ] Zero-copy NumPy support
- [ ] pytest suite
- [ ] CI wheel builds

### Milestone 7: MATLAB Interface
- [ ] Create MEX wrapper
- [ ] MATLAB OOP wrappers (mirrors Python)
- [ ] MATLAB documentation
- [ ] MATLAB tests

### Milestone 8: GPU/CUDA (Deferred)
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
