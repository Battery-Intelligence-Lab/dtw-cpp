# Session Handoff: Mmap Architecture + C++20 Migration (2026-04-08)

## Session Summary

Massive infrastructure session: 17 commits implementing mmap distance matrices, mmap data store, benchmarks, C++20 upgrade, and std::span DTW interface migration. The library now has a complete mmap storage backend and modern C++20 interfaces.

## What Was Done

### 1. Count-Before-Load + Auto Method Selection (commit 75c760d)
- `DataLoader::count()` — count series without loading data
- `--method auto` (new CLI default) — pam for N≤5000, clara for N>5000
- CLARA sample_size auto-scaling for N>50K

### 2. Mmap Distance Matrix (commits 6457644 → 535bdf4)
- llfio integrated via CPM
- `MmapDistanceMatrix` class — 32-byte binary header, NaN sentinel, warmstart
- `std::variant<DenseDistanceMatrix, MmapDistanceMatrix>` in Problem
- `visit_distmat()` helper for variant dispatch
- 18 unit tests for MmapDistanceMatrix

### 3. Binary Checkpoint + CLI (commits 1134695, 60e0a3e, 55844ef)
- `save_binary_checkpoint` / `load_binary_checkpoint` for clustering state
- `--restart` and `--mmap-threshold` CLI flags
- Warmstart: 3.99s → 0.11s (36x speedup) on dummy dataset

### 4. Benchmark Suite (commit adfa375)
- 7 benchmark categories at N=5000 (~95MB) using Google Benchmark
- **Key results:**
  - Mmap random access: only 5% slower than heap
  - Mmap startup: 78x faster than fread (1.4ms vs 109ms)
  - CLARA views: 48x faster than vector copies
  - `std::visit` overhead: negligible (0.000025%)

### 5. llfio Non-Optional (commit 1f91d71)
- Removed all `#ifdef DTWC_HAS_MMAP` guards (12 files)
- llfio is now a core dependency (like Eigen)

### 6. MmapDataStore (commits 32061d5, 52d1e20)
- `dtwc/core/mmap_data_store.hpp` — contiguous mmap cache for series data
- 64-byte header + offset table + contiguous doubles
- Variable-length + multivariate support
- Extracted shared `crc32.hpp` utility
- 6 unit tests

### 7. C++20 Upgrade (commit 6f0a3ed)
- `CMAKE_CXX_STANDARD 20` everywhere (CXX + CUDA)
- `cxx_std_20` in all cmake targets
- Dropped GCC 10, Clang 12, Clang 13 from CI

### 8. std::span Migration (commits e7bd00d, dec7ff6)
- `dtw_fn_t` changed to `std::function<data_t(std::span<const data_t>, std::span<const data_t>)>`
- ~22 DTW function signatures changed from `const vector<T>&` to `std::span<const T>`
- Vector overloads kept alongside for backward compatibility
- All 67 tests pass, no performance regression

## Current State

- **Branch:** Claude (17 commits ahead of origin/Claude)
- **Tests:** 67/67 pass, 2 CUDA skipped
- **Build:** Clang 21, C++20, Ninja, Windows 11
- **Key dependencies:** llfio (core), Eigen 5.0.1 (core), Catch2, CLI11, Google Benchmark

## What To Do Next

### IMMEDIATE: Phase 4 — Storage Policy + DataAccessor

1. **DataAccessor abstraction** — Virtual interface over vector-of-vectors vs MmapDataStore:
   ```cpp
   struct DataAccessor {
     virtual const double* series_data(size_t i) const = 0;
     virtual size_t series_flat_size(size_t i) const = 0;
     virtual size_t size() const = 0;
     virtual size_t ndim() const = 0;
   };
   ```
   - `VectorDataAccessor`: wraps existing `Data.p_vec`
   - `MmapDataAccessor`: wraps `MmapDataStore`

2. **Change `Problem::p_vec(i)` return type** from `vector<data_t>&` to accessor-based access. The DTW lambdas now accept span, so the bridge is ready.

3. **CLARA zero-copy views** — Replace `sub_vecs.push_back(prob.p_vec(idx))` (copies entire series) with span views into parent data. Benchmarked: 48x faster.

4. **StoragePolicy enum** — `Auto`/`Heap`/`Mmap` with threshold-based auto-selection.

### INVESTIGATE: llfio C++20 features
- llfio may have C++20-specific APIs (coroutines, concepts, better error handling)
- Check if `mapped_file_handle` has C++20 improvements worth using
- **Note:** User specifically asked for this investigation

### DEFERRED (lower priority)

5. **DDTW kernel fusion** — Compute derivative on-the-fly inside DTW recurrence instead of pre-allocating full derivative vectors. User identified this as wasteful for large data. Use thread_local buffer for one series, compute on-the-fly for the other. Separate PR.

6. **Streaming CLARA** — Load only subsamples from disk, never full dataset.

7. **Stale cache detection** — Hash input filenames + sizes + mtimes in mmap header.

8. **Checkpoint warm-start in algorithms** — `--restart` loads checkpoint but doesn't yet skip re-clustering.

## Open Bugs

1. **hierarchical + SoftDTW crashes** — build_dendrogram or distance fill
2. **set_if_unset in YAML** — unconditionally overrides CLI values
3. **MV banded DTW silently ignores band** for Missing/WDTW/ADTW variants
4. **Pruned distance matrix + MmapDistanceMatrix** — `fill_distance_matrix_pruned()` calls `prob.dense_distance_matrix()` which throws `bad_variant_access` when mmap is active. Workaround: BruteForce strategy. Fix: make pruned strategy work with any distance matrix type.

## Clangd False Positives (ignore)

- `std::span` errors in Problem.hpp — clangd configured for C++17, actual build is C++20
- llfio `LLFIO_V2_NAMESPACE` macro confuses clangd
- `benchmark/benchmark.h` not found — build-time CPM dependency
- `nanobind.h` / `mex.h` not found — optional binding deps

## Real Diagnostic to Address

- `mmap_data_store.hpp:254` — clang-tidy `bugprone-sizeof-expression`: `data_base_ + offsets_[i] / sizeof(double)`. Code is correct (byte offset → element count), but consider storing element offsets instead of byte offsets for clarity.

## Key Architectural Decisions Made This Session

1. **llfio is a core dependency** (not optional) — user preference for libraries over custom code
2. **Mmap is safe as default** — benchmarks show negligible overhead (5% random access)
3. **std::span is the vocabulary type** for function interfaces; Eigen::Map for internal compute
4. **Binary cache formats are internal** — users keep CSV/HDF5/Parquet, we cache internally
5. **Eigen::Map doesn't help** for packed triangular distance matrix (Eigen has no packed symmetric storage)
6. **C++20 is the new minimum** — all CI compilers support it
7. **DDTW derivative allocation is wasteful** but deferred — fuse into kernel in separate PR

## Files Created/Modified This Session

### New files:
- `dtwc/core/mmap_distance_matrix.hpp` — MmapDistanceMatrix class
- `dtwc/core/mmap_data_store.hpp` — MmapDataStore class  
- `dtwc/core/crc32.hpp` — shared CRC32 utility
- `benchmarks/bench_mmap_access.cpp` — mmap access benchmarks
- `tests/unit/core/unit_test_mmap_distance_matrix.cpp` — 18 tests
- `tests/unit/core/unit_test_mmap_data_store.cpp` — 6 tests
- `tests/unit/unit_test_variant_distmat.cpp` — 3 tests
- `tests/unit/unit_test_checkpoint_binary.cpp` — 3 tests

### Heavily modified:
- `dtwc/Problem.hpp` — variant distMat, visit_distmat, span dtw_fn_t
- `dtwc/Problem.cpp` — variant dispatch, use_mmap_distance_matrix
- `dtwc/warping*.hpp` (6 files) — span overloads
- `dtwc/soft_dtw.hpp` — span overloads
- `dtwc/missing_utils.hpp` — span overloads
- `dtwc/checkpoint.hpp/.cpp` — binary checkpoint
- `dtwc/dtwc_cl.cpp` — --restart, --mmap-threshold, auto method
- `cmake/Dependencies.cmake` — llfio always linked
- `cmake/StandardProjectSettings.cmake` — C++20

## Build Instructions

```bash
cmake --preset clang-win    # or: cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build --parallel 8
ctest --test-dir build -C Release -j8   # 67/67 pass, 2 CUDA skip

# Benchmarks (optional):
cmake -B build -G Ninja -DDTWC_BUILD_BENCHMARK=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build --target bench_mmap_access --parallel 8
./build/bin/bench_mmap_access
```