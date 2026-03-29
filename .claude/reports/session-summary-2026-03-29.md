# DTWC++ Session Summary — 2026-03-29

**Branch:** Claude (ahead of main by ~60 commits)
**Machine:** Windows 11, 24-core Intel @ 2496 MHz, MSVC 19.50, no CUDA

---

## What Was Built This Session

### New Features (All Phases Complete)

| Phase | Feature | Key Files | Tests |
|-------|---------|-----------|-------|
| Sub-B | SIMD Highway infrastructure (disabled by default) | `dtwc/simd/` | 26 tests |
| 5 | **FastCLARA** scalable k-medoids | `dtwc/algorithms/fast_clara.hpp/cpp` | 11 tests |
| 6 | **Checkpointing** save/resume | `dtwc/checkpoint.hpp/cpp` | 8 tests |
| 7 | **DTW missing data** (NaN-aware) | `dtwc/warping_missing.hpp` | 31 tests |
| 8 | **HDF5/Parquet/CSV I/O** (Python) | `python/dtwcpp/io.py` | 14 tests |
| 9 | **MATLAB MEX bindings** | `bindings/matlab/` | 12 MATLAB tests |
| 11 | **CLI with TOML config** | `dtwc/dtwc_cl.cpp` | — |
| Sub-C | **MPI distributed** distance matrix | `dtwc/mpi/` | 6 MPI tests |
| Sub-C | **CUDA batch DTW** kernel | `dtwc/cuda/` | — (needs GPU) |
| — | **MetricType** (L1, SquaredL2) | `dtwc/warping.hpp` refactored | 9 tests |
| — | **Pointer-based DTW** (zero-copy) | `dtwc/warping.hpp`, `core/dtw.hpp` | 4 tests |
| — | **Pruned distance matrix** (LB early-abandon) | `dtwc/core/pruned_distance_matrix.cpp` | 17 tests |
| — | **compute_distance_matrix** Python (OpenMP) | `python/src/_dtwcpp_core.cpp` | Integration |
| — | **Cross-library benchmarks** | `benchmarks/bench_cross_library.py` | — |
| — | `/fp:fast` + `#pragma omp simd` | `cmake/StandardProjectSettings.cmake` | — |

### Performance Results (identical data, all libraries)

**Pairwise DTW (SquaredL2 metric):**
- dtwcpp ~1.5x slower than aeon per-pair (MSVC vs Numba LLVM codegen)

**Distance Matrix (N=50, L=1000, pruned):**
- **dtwcpp 276ms** vs aeon 3400ms (**12x faster**) vs dtaidistance 486ms (**1.75x faster**)

**Clustering (N=50, L=500, k=3):**
- **dtwcpp 81ms** vs aeon 3390ms (**42x faster**)

### Test Counts

- **C++ tests:** 39 executables, ~41,000+ assertions, all pass
- **Python unit tests:** 114 pass
- **Python integration tests:** 36 pass
- **Python I/O tests:** 14 pass
- **MATLAB tests:** 12 (require MATLAB SDK)
- **MPI tests:** 6 (require mpiexec)

---

## Critical Bugs Found & Fixed

1. **DenseDistanceMatrix NaN sentinel broken under `-ffast-math`** — `std::isnan()` optimized away. Fixed with boolean vector.
2. **SIMD fillDistanceMatrix ignored band + DTW variants** — correctness bug, removed SIMD dispatch.
3. **DTWClustering.predict() crashed** — dtw_distance rejected lists after pointer refactor. Fixed with numpy wrapper.
4. **Data race on nn_dist[]** in pruned distance matrix — fixed with lock-free design (each thread writes only to its own row).
5. **silhouette/DBI broken after fast_pam()** — results not stored back to Problem. Fixed.

## Key Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| Compiler auto-vectorizes > Highway SIMD | MSVC `/O2` already vectorizes simple loops; Highway adds dispatch overhead |
| Lock-free parallel design | No atomics — partition so threads write to non-overlapping regions |
| LB_Keogh early-abandon (not pair-skipping) | Full matrix needed for k-medoids; early-abandon terminates DTW rows early |
| Template lambda metric dispatch | Zero inner-loop overhead; metric selected once outside all loops |
| Python wrappers for list/numpy compat | C++ binding uses nb::ndarray (zero-copy); Python wrapper calls np.asarray |
| `uv` for Python package management | Never pip; pyproject.toml + uv sync/add |

## What's NOT Done / Known Issues

1. **MPI untested** — MS-MPI not installed on this machine. Code compiles with MPI=OFF.
2. **CUDA untested** — No NVIDIA GPU. Code compiles with CUDA=OFF.
3. **Per-pair DTW 1.5x slower than aeon** — MSVC codegen issue. Try Clang/GCC on Linux.
4. **WDTW/ADTW hardcode L1** — no MetricType parameter on these variants.
5. **MetricType::L2 == L1 for scalars** — documented, not a bug but confusing.
6. **CI DNS failures** — transient GitHub Actions issue, not code-related. Re-run.
7. **~30 stale remote worktree branches** — deleted locally, may still exist on origin.
8. **Old build/ folder** (~4GB) — kept one build dir for incremental rebuilds.

## Continuation Instructions

### To continue on another machine:

```bash
git clone <repo> && cd dtw-cpp && git checkout Claude

# C++ build
cmake -S . -B build -DDTWC_BUILD_TESTING=ON
cmake --build build --config Release -j
cd build && ctest -C Release

# Python
cd benchmarks && uv sync  # installs all deps including dtwcpp
uv run python bench_cross_library.py  # cross-library benchmark

# MPI (after installing MS-MPI or OpenMPI)
cmake -S . -B build -DDTWC_ENABLE_MPI=ON -DDTWC_BUILD_TESTING=ON
cmake --build build --config Release -j
mpiexec -n 4 ./build/bin/unit_test_mpi

# CUDA (after installing CUDA toolkit)
cmake -S . -B build -DDTWC_ENABLE_CUDA=ON
cmake --build build --config Release -j
```

### Key files to read first:
- `.claude/CLAUDE.md` — project rules and architecture
- `.claude/LESSONS.md` — lessons learned (performance, SIMD, metrics)
- `.claude/CITATIONS.md` — academic references
- `CHANGELOG.md` — complete feature list
- `benchmarks/bench_cross_library.py` — performance comparison

### CMake options:
```
DTWC_BUILD_TESTING=ON      # Unit tests (Catch2)
DTWC_BUILD_BENCHMARK=ON    # Google Benchmark
DTWC_BUILD_PYTHON=ON       # nanobind Python bindings
DTWC_BUILD_MATLAB=ON       # C++ MEX bindings (requires MATLAB)
DTWC_ENABLE_SIMD=OFF       # Highway SIMD (disabled — compiler auto-vectorizes better)
DTWC_ENABLE_MPI=OFF        # MPI distributed (requires MPI library)
DTWC_ENABLE_CUDA=OFF       # CUDA GPU (requires CUDA toolkit + NVIDIA GPU)
DTWC_ENABLE_GUROBI=ON      # Gurobi MIP solver (optional)
DTWC_ENABLE_HIGHS=ON       # HiGHS MIP solver (optional)
```

### Priority improvements for next session:
1. Test MPI on a machine with MPI installed
2. Test CUDA on a machine with NVIDIA GPU
3. Try GCC/Clang build to close the 1.5x per-pair gap
4. Add SquaredL2 metric to WDTW/ADTW variants
5. Consider release tagging (v2.0.0)
