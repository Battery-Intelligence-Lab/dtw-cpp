# Plan: Benchmark Google Highway SIMD, Then Remove If Not Useful

## Context

The `dtwc/simd/` folder contains 3 Google Highway SIMD kernels (LB_Keogh, z_normalize, multi-pair DTW) that are **not wired into any production code path**. They are OFF by default (`DTWC_ENABLE_SIMD=OFF`) and only called from `tests/unit/unit_test_simd.cpp`. LESSONS.md already documents that:
- DTW is latency-bound (10-cycle recurrence) — multi-pair SIMD gives max ~1.29x
- MSVC auto-vectorizes LB_Keogh/z_normalize to SSE2; explicit Highway adds dispatch overhead

The goal is to **benchmark scalar vs Highway** for all 3 operations, then remove the entire SIMD infrastructure if the speedup is < 2x (not worth the dependency + complexity).

---

## Step 1: Add SIMD Benchmarks to `benchmarks/bench_dtw_baseline.cpp`

Append after line 445, guarded by `#ifdef DTWC_HAS_HIGHWAY`:

**Includes** (after line 20):
```cpp
#ifdef DTWC_HAS_HIGHWAY
#include <simd/multi_pair_dtw.hpp>
namespace dtwc::simd {
  double lb_keogh_highway(const double*, const double*, const double*, std::size_t);
  void z_normalize_highway(double*, std::size_t);
}
#endif
```

**Benchmarks** (3 pairs — scalar already exists, add Highway counterparts + multi-pair comparison):

| Benchmark | What it measures | Lengths |
|-----------|-----------------|---------|
| `BM_lb_keogh_highway` | `dtwc::simd::lb_keogh_highway()` vs existing `BM_lb_keogh` | 100, 500, 1000, 4000, 8000 |
| `BM_z_normalize_highway` | `dtwc::simd::z_normalize_highway()` vs existing `BM_z_normalize` | 100, 500, 1000, 4000, 8000 |
| `BM_dtw_4pairs_sequential` | 4× `dtwFull_L<double>()` (baseline) | 100, 500, 1000 |
| `BM_dtw_4pairs_simd` | 1× `dtwc::simd::dtw_multi_pair()` | 100, 500, 1000 |

---

## Step 2: Build & Run Benchmarks

```bash
cmake -B build -DDTWC_ENABLE_SIMD=ON -DDTWC_BUILD_BENCHMARK=ON
cmake --build build --config Release --target bench_dtw_baseline
./build/benchmarks/Release/bench_dtw_baseline --benchmark_format=json > benchmarks/results/simd_comparison.json
```

**Decision threshold**: If no kernel shows >= 2x speedup at any tested length, proceed to removal.

---

## Step 3: Remove SIMD Infrastructure

### 3a. Delete files
- `dtwc/simd/lb_keogh_simd.cpp`
- `dtwc/simd/z_normalize_simd.cpp`
- `dtwc/simd/multi_pair_dtw.cpp`
- `dtwc/simd/multi_pair_dtw.hpp`
- `dtwc/simd/highway_targets.hpp`
- `dtwc/simd/.gitkeep`
- `dtwc/simd/` directory

### 3b. Edit `tests/unit/unit_test_simd.cpp`
- Remove lines 30-32 (`#ifdef DTWC_HAS_HIGHWAY` / include / `#endif`)
- Remove lines 83-106 (`dtw_scalar` helper, only used by multi-pair tests)
- Remove lines 397-645 (all multi-pair DTW tests inside `#ifdef DTWC_HAS_HIGHWAY`)
- Remove `#include <warping.hpp>` (line 17) and `#include <dtwc.hpp>` (line 18) — only used by multi-pair tests
- Keep all scalar LB_Keogh + z_normalize tests (lines 150-395) — these test production code

### 3c. Edit CMake files

**`CMakeLists.txt`** (root, line 28): Remove `option(DTWC_ENABLE_SIMD ...)`

**`cmake/Dependencies.cmake`** (lines 97-116): Remove entire Highway CPM block

**`dtwc/CMakeLists.txt`** (lines 90-100): Remove SIMD source/link/define block

**`benchmarks/CMakeLists.txt`** (lines 12-15): Remove Highway link block

### 3d. Edit docs

**`README.md`** (line 134): Remove `DTWC_ENABLE_SIMD` row from options table

**`CHANGELOG.md`**: Add to Unreleased: "Removed experimental Google Highway SIMD (`DTWC_ENABLE_SIMD`). Compiler auto-vectorization (`#pragma omp simd`) provides equivalent performance without the dependency."

### 3e. Clean benchmark file
Remove the `#ifdef DTWC_HAS_HIGHWAY` benchmark code added in Step 1 from `bench_dtw_baseline.cpp`.

---

## Step 4: Rebuild & Verify

```bash
cmake -B build
cmake --build build --config Release
ctest --test-dir build -C Release
```

Grep for dangling references: `DTWC_HAS_HIGHWAY`, `DTWC_ENABLE_SIMD`, `hwy::hwy`, `highway_targets`, `simd::lb_keogh`, `simd::z_normalize`, `simd::dtw_multi_pair`.

---

## Files Modified (total)

| Action | File |
|--------|------|
| DELETE | `dtwc/simd/*` (6 files + directory) |
| EDIT | `tests/unit/unit_test_simd.cpp` (strip SIMD sections, keep scalar tests) |
| EDIT | `CMakeLists.txt` (remove option) |
| EDIT | `cmake/Dependencies.cmake` (remove Highway CPM) |
| EDIT | `dtwc/CMakeLists.txt` (remove SIMD block) |
| EDIT | `benchmarks/CMakeLists.txt` (remove hwy link) |
| EDIT | `benchmarks/bench_dtw_baseline.cpp` (add then remove SIMD benchmarks) |
| EDIT | `README.md` (remove SIMD option row) |
| EDIT | `CHANGELOG.md` (add removal entry) |

`#pragma omp simd` hints in `lower_bound_impl.hpp` and `z_normalize.hpp` are **kept** — they are OpenMP auto-vectorization, not Highway.
