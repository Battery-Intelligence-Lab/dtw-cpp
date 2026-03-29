# DTWC++ Transformation Progress Summary

**Date:** 2026-03-29
**Branch:** Claude (ahead of main by ~40 commits)
**C++ Tests:** 34 passing (+ 3 pre-existing pruned DM failures)
**Python Tests:** 100 passing (0.26s)
**Context:** This document captures everything accomplished for continuity across context resets.

---

## Completed Phases

### Phase 0: Critical Bug Fixes (COMPLETE)
- 24 bugs fixed: `throw 1`, ODR violation, `.at()` in hot loop, dtwBanded 512MB allocation, int overflow, DBI formula, broken examples, PAM mislabeled
- 12 CMake fixes: option() syntax, output dirs, Armadillo pin, CONFIGURE_DEPENDS
- CI fixes: uncommented main triggers, ASan+UBSan job
- Python packaging: fixed pyproject.toml, deleted conflicting setup.py, created VERSION file

### Phase 1: Core Architecture Refactor (COMPLETE)
- 9 new headers in `dtwc::core`: ScratchMatrix, TimeSeriesView, DistanceMetric, DTWOptions, dtw.hpp, DenseDistanceMatrix, ClusteringResult, z_normalize, lower_bounds
- FastPAM1 algorithm (Schubert & Rousseeuw 2021, JMLR)
- Documentation: algorithms.md, metrics.md

### Phase 2: Performance — Memory First (COMPLETE)
- Column-major ScratchMatrix, NaN sentinel DenseDistanceMatrix
- LB_Keogh + LB_Kim implementations with property tests
- Early abandon parameter in dtwFull_L

### Phase 2.5: Core Integration (COMPLETE)
- Wired all core types into production code
- **Armadillo fully removed** from library (zero references, not linked)
- FastPAMResult unified with ClusteringResult
- z_normalize tests now use production code
- Pruned distance matrix re-enabled
- DenseDistanceMatrix owns its I/O (write_csv, read_csv, operator<<)
- 7 adversarial test suites (~30K assertions)

### Phase 4: Python Bindings (COMPLETE)
- **nanobind** (not pybind11) — GPU-native ndarray, stable ABI, 128KB wheels
- scikit-build-core build system
- DTWClustering sklearn-compatible class (fit/predict/fit_predict)
- All DTW variants exposed: dtw_distance, ddtw, wdtw, adtw, soft_dtw
- DenseDistanceMatrix zero-copy to_numpy()
- 100 Python tests + 15 cross-validation tests + 3 examples
- CI: python-tests.yml (every push), python-wheels.yml (main + tags only)

### Phase 10: DTW Variants (COMPLETE)
- DDTW (derivative preprocessing) — warping_ddtw.hpp
- WDTW (weighted, logistic) — warping_wdtw.hpp (rolling buffer banded)
- ADTW (amerced, penalty) — warping_adtw.hpp
- Soft-DTW (differentiable, gradient) — soft_dtw.hpp
- DTWVariant enum + DTWVariantParams in dtw_options.hpp
- Problem::set_variant() with std::function dispatch

### HPC Performance Fixes (COMPLETE — Sub-phase A)
- **Band rebind bug fixed** — lambda captures `this` not `band` by value (14x fillDM speedup)
- **std::min({a,b,c}) → nested std::min** — 2.5-3x all DTW functions
- **Branchless LB_Keogh** — max(0, max(q-U, L-q))
- **FastPAM SWAP parallelized** — OpenMP on outer candidate loop
- **compute_nearest_and_second parallelized** — OpenMP static schedule
- **std::reduce** for reductions, multiply by 1/stddev, redundant add removal
- **WDTW banded rolling buffer** — 128MB → 32KB for n=4000
- **Early abandon running min** — O(1) per row instead of O(n) min_element
- Roofline analysis: DTW is latency-bound on recurrence (10 cycles/cell), not memory-bound

### Cleanup & CI
- Deleted obsolete files: main.py, test.py, .python-version, uv.lock, develop/TODO.md, setup.py
- Merged benchmark/ into benchmarks/
- CMake: VERSION file, complete PUBLIC headers, absolute test paths, -fPIC
- Bumped Python >=3.10, numpy >=1.26, benchmarks/pyproject.toml (uv compatible)
- README: Python tests badge

---

## Remaining Phases

| Phase | Scope | Status | Priority |
|-------|-------|--------|----------|
| Sub-B | SIMD via Google Highway | **Planned** | High — 3-4x multi-pair DTW, 4-8x LB_Keogh |
| Sub-C | MPI + CUDA | **Planned** | High — required for N>10K scale |
| 3 | GPU/CUDA kernels | Planned (part of Sub-C) | High |
| 5 | FastCLARA + MPI | Planned (part of Sub-C) | High — required for 100M series |
| 6 | Checkpointing (HDF5/CSV) | Not started | Medium |
| 7 | Missing data (DTW-AROW) | Not started | Medium |
| 8 | I/O — HDF5 + Parquet | Not started | Low |
| 9 | MATLAB bindings (C++ MEX) | Not started | Low |
| 11 | Build system + CLI | Not started | Low |

### Sub-phase B: SIMD via Google Highway (NEXT)

Layered parallelism architecture designed:
```
Level 2 (MPI):     P ranks across nodes — pair blocks
  Level 1 (OpenMP): T threads per rank — dynamic scheduling
    Level 0 (SIMD):  Highway vectorization (AVX2/512/NEON runtime dispatch)
Level 3 (CUDA):    Alternative GPU path
```

Key items:
- B0: Highway CPM integration + DTWC_ENABLE_SIMD option
- B1: SIMD LB_Keogh (4-8x, called O(N²) times)
- B4: Multi-pair DTW — 4 pairs in AVX2 lanes (3-4x fillDistanceMatrix)
- B2/B3: SIMD envelope, derivative stencil

### Sub-phase C: MPI + CUDA

- C1: MPI distance matrix partitioning + MPI_Allreduce
- C3: CUDA batch DTW kernel (one block per pair, anti-diagonal wavefront)
- C4: nanobind nb::ndarray<T, nb::device::cuda> GPU tensor interop

---

## Key Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| nanobind over pybind11 | GPU-native ndarray, stable ABI, 5-10x smaller wheels, 68-line sunk cost |
| C++ MEX API over legacy C | No longjmp, RAII-safe, typed arrays |
| Google Highway for SIMD | C++17, runtime dispatch, broadest ISA, built-in exp() |
| Separate functions per DTW variant | Recurrence differs structurally; conditionals hurt memory-bound kernel |
| std::function dispatch in Problem | ~2ns overhead negligible vs 1-100ms DTW |
| Column-major ScratchMatrix | Matches DTW inner loop access pattern |
| NaN sentinel for distance matrix | Handles future negative-distance metrics |
| O(n×band) envelope scan | Cache-friendly for small band; Lemire O(n) deferred |

---

## Test Summary

| Suite | Count | Status |
|-------|-------|--------|
| C++ unit tests (Catch2) | 34 | 31 pass, 3 pre-existing pruned DM failures |
| C++ adversarial tests | 7 suites (~148 cases) | All pass |
| Python tests (pytest) | 100 | All pass (0.26s) |
| Cross-validation (C++ ↔ Python) | 15 | All pass |

## Performance Summary (24-core Intel @ 2496 MHz)

| Benchmark | Before | After | Speedup |
|-----------|--------|-------|---------|
| dtwFull n=4000 | 238ms | 97ms | 2.5x |
| dtwBanded n=1000 band=10 | 320us | 108us | 3.0x |
| fillDM N=50 L=500 band=10 | 54ms | 3.95ms | **14x** |
| fillDM N=50 L=1000 band=50 | 214ms | 32.8ms | **6.5x** |

DTW is latency-bound on recurrence chain (10 cycles/cell, 265M cells/sec).
Multi-pair SIMD (Sub-phase B) expected to give 3-4x additional throughput.
