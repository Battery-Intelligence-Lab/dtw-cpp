# Allocation Refactor & Benchmark Plan

## Context

The memory-and-design-audit.md identified ~15 performance issues across C++ core, Python bindings, and MATLAB bindings. Codex (commit `5c24edb`) already implemented partial fixes:
- WDTW weight caching via `refresh_variant_caches()` in Problem.cpp
- DDTW in-place derivative transforms (`derivative_transform_inplace`, `derivative_transform_mv_inplace`)
- MV ndim==1 pointer dispatch for WDTW/ADTW
- New variant benchmarks in `bench_dtw_baseline.cpp`

**However, the Codex before/after benchmark data is unreliable** — some "after" numbers are worse (e.g., wdtwBanded_g 4000/50: 1715us before → 2346us after), likely due to machine load variance. We need to:
1. Verify and complete what Codex started
2. Fix remaining issues not yet addressed
3. Run proper isolated benchmarks with statistical confidence

### Codex Benchmark Summary (before → after, cpu_time in us)

| Benchmark | Before | After | Delta | Notes |
|-----------|--------|-------|-------|-------|
| wdtwBanded_g/1000/10 | 96.3 | 96.3 | 0% | Same — suspicious |
| wdtwBanded_g/1000/50 | 414.4 | 399.0 | -3.7% | Marginal |
| wdtwBanded_g/4000/50 | 1604 | 2380 | +48% | WORSE — machine noise |
| wdtwBanded_precomputed/1000/10 | 128.7 | 128.3 | ~0% | Baseline |
| ddtwBanded/1000/10 | 146.5 | 122.8 | -16% | **Good improvement** |
| ddtwBanded/1000/50 | 609.4 | 530.1 | -13% | **Good improvement** |
| adtwBanded/1000/10 | 515.6 | 376.7 | -27% | **Significant** |
| adtwBanded/1000/50 | 976.6 | 878.9 | -10% | Good |
| fillDM Standard | 17.0ms | 17.9ms | +5% | Noise |
| fillDM WDTW | 16.9ms | 17.3ms | +2% | Noise |
| fillDM DDTW | 17.4ms | 17.7ms | +2% | Noise |
| fillDM ADTW | 30.0ms | 33.5ms | +12% | WORSE — noise |

**Conclusion:** DDTW and ADTW single-call improvements are real (~13-27%). End-to-end fillDM numbers are dominated by noise. Need proper benchmarking methodology.

---

## Remaining Work (What Codex Did NOT Do)

### HIGH priority — not yet addressed:
1. **[2.2] Python bindings still copy inputs** — `ddtw_distance`, `wdtw_distance`, `adtw_distance`, `soft_dtw_distance`, `soft_dtw_gradient` still use `const std::vector<double>&`
2. **[2.3] `distance_matrix_numpy()` double copy** — still does element-by-element via `distByInd()`
3. **[4.5] Standalone pruned DM missing nn_dist[j] update** — bug not fixed

### MEDIUM priority — not yet addressed:
4. **[1.4] WDTW/ADTW MV banded falls back to unbanded** — still O(n*m) instead of O(n*band) for ndim>1
5. **[2.4] Python `compute_distance_matrix` copies entire dataset**
6. **[3.2] MATLAB DTW functions copy inputs** — `to_std_vector()` instead of pointer access

### NEW improvement identified:
7. **DenseDistanceMatrix stores full N*N but is symmetric** — wastes ~50% memory. For N=10K that's 800MB vs 400MB. Should use packed triangular storage: `k = max(i,j)*(max(i,j)+1)/2 + min(i,j)`, storing only `N*(N+1)/2` elements. Also `computed_` vector should use bit-packing (`vector<bool>` or bitset) instead of `vector<char>`.

### Needs verification (Codex implemented but untested):
8. **[1.1] WDTW weight caching** — `refresh_variant_caches()` logic needs review for correctness
9. **[1.2] DDTW in-place transforms** — need to verify no regressions
10. **[1.3] MV ndim==1 pointer dispatch** — need to verify correctness

---

## Implementation Plan

### Phase 0: Verify Codex Changes (Read-only)
**Goal:** Audit what Codex did, ensure correctness before building on top.

Files to verify:
- `dtwc/Problem.cpp` — `refresh_variant_caches()` logic
- `dtwc/Problem.hpp` — new member `wdtw_weights_cache_`
- `dtwc/warping_wdtw.hpp` — new pointer overloads, ndim==1 dispatch
- `dtwc/warping_ddtw.hpp` — `derivative_transform_inplace`, `derivative_transform_mv_inplace`
- `dtwc/warping_adtw.hpp` — ndim==1 pointer dispatch

Key checks:
- Does `refresh_variant_caches()` compute weights for ALL unique series lengths?
- Are the new in-place transforms numerically identical to the originals?
- Do the pointer-based WDTW/ADTW overloads handle edge cases (empty series, single element)?
- Is the `wdtw_weights_cache_` properly invalidated on `set_variant()` / `set_data()`?

### Phase 1: Build & Run Baseline Benchmarks
**Goal:** Get reliable before numbers on a quiet machine.

```bash
# Build Release with benchmarks
cmake -S . -B build_bench -DCMAKE_BUILD_TYPE=Release -DDTWC_BUILD_BENCHMARK=ON -DDTWC_BUILD_TESTING=ON
cmake --build build_bench --config Release -j

# Run tests first to verify correctness
ctest --test-dir build_bench -j2 -C Release --output-on-failure

# Run benchmarks with 5 repetitions for statistical confidence
./build_bench/bin/bench_dtw_baseline --benchmark_format=json --benchmark_repetitions=5 --benchmark_out=benchmarks/results/phase1_baseline.json
```

### Phase 2: Fix Remaining C++ Issues
**Files to modify:**

#### 2a. Standalone pruned DM bug fix [BUG]
- `dtwc/core/pruned_distance_matrix.cpp` ~line 406
- Add `atomic_min_double(&nn_dist[j], dist)` after existing `nn_dist[i]` update

#### 2b. WDTW convenience overloads — use thread_local cache
- `dtwc/warping_wdtw.hpp` — convenience overloads `wdtwBanded(x, y, band, g)` and `wdtwFull(x, y, g)`
- Add `thread_local` cache keyed on `(max_dev, g)` to avoid recomputing weights for standalone Python calls

#### 2c. DDTW pointer-based overload (needed for Python zero-copy)
- `dtwc/warping_ddtw.hpp` — add `ddtwBanded(const T* x, size_t nx, const T* y, size_t ny, int band)` that uses thread_local buffers internally

### Phase 3: DenseDistanceMatrix → Packed Triangular Storage

**Goal:** Cut memory by ~50% and eliminate redundant double-writes.

**File:** `dtwc/core/distance_matrix.hpp`

Current: full N*N `vector<double>` + N*N `vector<char>` (row-major, symmetric writes)
New: `N*(N+1)/2` packed lower-triangular `vector<double>` + bit-packed `vector<bool>` computed flags

**Index mapping:** For `i >= j`: `k = i*(i+1)/2 + j`. For `i < j`: swap to `k = j*(j+1)/2 + i`.

**API changes (internal only, external API stays the same):**

- `get(i, j)` → single lookup via triangular index
- `set(i, j, v)` → single write (no more dual write)
- `is_computed(i, j)` → bit-packed lookup
- `raw()` → returns packed triangular data (NOT a square matrix anymore)
- `size()` → still returns N
- `max()`, `count_computed()`, `all_computed()` → iterate over packed storage (fewer elements = faster)

**Breaking changes to handle:**

- `to_numpy()` in Python binding (line 250-255): currently returns `raw()` as NxN. Must expand to full square on demand, or return scipy-compatible condensed form.
- `distance_matrix_numpy()` (line 432-442): already broken (element-by-element copy). Fix to expand from triangular.
- MATLAB `cmd_Problem_get_distance_matrix` (line 456-470): element-by-element into column-major MATLAB matrix. Change to use `get(i,j)` — already works since API is unchanged.
- `write_csv()` / `read_csv()`: iterate using `get(i,j)` — already works since API is unchanged.
- `pruned_distance_matrix.cpp`: uses `dm.set(i, j, dist)` — works unchanged.

**Memory savings:**

| N | Current (data + computed) | Triangular (data + bits) | Savings |
|---|---------------------------|--------------------------|---------|
| 100 | 80 KB + 10 KB = 90 KB | 40 KB + 0.6 KB = 41 KB | 54% |
| 1,000 | 8 MB + 1 MB = 9 MB | 4 MB + 62 KB = 4.1 MB | 54% |
| 10,000 | 800 MB + 100 MB = 900 MB | 400 MB + 6.2 MB = 406 MB | 55% |

### Phase 4: Fix Python Bindings

**File:** `python/src/_dtwcpp_core.cpp`

#### 4a. Convert variant DTW bindings to ndarray (lines 279-317)
Replace `const std::vector<double>&` with `nb::ndarray<const double, nb::ndim<1>, nb::c_contig>` for:
- `ddtw_distance` → call new pointer-based `ddtwBanded`
- `wdtw_distance` → call pointer-based `wdtwBanded`
- `adtw_distance` → call pointer-based `adtwBanded`
- `soft_dtw_distance` → keep vector for now (soft_dtw has no pointer overload, lower priority)
- `soft_dtw_gradient` → keep vector for now

#### 4b. Fix `distance_matrix_numpy()` (lines 432-442)
After Phase 3 (triangular storage), `raw()` returns packed data. Expand to full NxN for numpy, or offer both condensed and square forms.

#### 4c. Fix `z_normalize` (lines 362-365)
Accept `nb::ndarray`, normalize in-place, return numpy array.

### Phase 5: Fix MATLAB Bindings

**File:** `bindings/matlab/dtwc_mex.cpp`

Replace `to_std_vector()` calls with `mxGetDoubles()` + pointer-based C++ overloads for:
- `cmd_dtw_distance`, `cmd_ddtw_distance`, `cmd_wdtw_distance`, `cmd_adtw_distance`

### Phase 6: Run After Benchmarks & Compare
```bash
./build_bench/bin/bench_dtw_baseline --benchmark_format=json --benchmark_repetitions=5 --benchmark_out=benchmarks/results/phase5_after.json
```

Compare using Google Benchmark tools or manual JSON parsing.

### Phase 7: Tests & Cleanup
- Run full test suite: `ctest --test-dir build_bench -j2`
- Run Python tests: `uv run pytest tests/python/ -v`
- Update CHANGELOG.md
- Update `.claude/TODO.md`

---

## Verification Plan

### Correctness checks:
1. All existing C++ unit tests pass (`ctest`)
2. All Python tests pass (`pytest tests/python/`)
3. WDTW/DDTW/ADTW distances match reference values (existing test coverage)
4. Distance matrix symmetry preserved
5. Pruned distance matrix produces correct results

### Performance checks:
1. Single-call DTW: compare wdtw/ddtw/adtw at L=1000 and L=4000
2. Distance matrix fill: compare fillDM variants at N=50,L=500
3. Python binding overhead: time `dtwcpp.wdtw_distance()` from Python
4. Use `--benchmark_repetitions=5` and compare medians, not means

### Key benchmarks to watch:
| Benchmark | Expected improvement | Why |
|-----------|---------------------|-----|
| BM_wdtwBanded_g vs BM_wdtwBanded_precomputed | Gap should close | Weight caching |
| BM_ddtwBanded/1000/10 | 10-20% faster | In-place derivative |
| BM_fillDistanceMatrix_variant WDTW | 5-15% faster | Cached weights in Problem |
| BM_fillDistanceMatrix_variant DDTW | 5-10% faster | In-place derivatives |
| Python wdtw_distance | 30-50% faster for small series | Zero-copy binding |

---

## Critical Files

| File | Changes |
|------|---------|
| `dtwc/warping_wdtw.hpp` | Verify Codex changes, add thread_local weight cache for convenience overloads |
| `dtwc/warping_ddtw.hpp` | Verify Codex changes, add pointer-based `ddtwBanded` overload |
| `dtwc/warping_adtw.hpp` | Verify Codex changes |
| `dtwc/Problem.cpp` | Verify `refresh_variant_caches()` |
| `dtwc/Problem.hpp` | Verify `wdtw_weights_cache_` member |
| `dtwc/core/distance_matrix.hpp` | Packed triangular storage, bit-packed computed_ |
| `dtwc/core/pruned_distance_matrix.cpp` | Fix nn_dist[j] bug |
| `python/src/_dtwcpp_core.cpp` | Zero-copy bindings, fix distance_matrix_numpy |
| `bindings/matlab/dtwc_mex.cpp` | Pointer-based DTW calls |
| `benchmarks/bench_dtw_baseline.cpp` | Already updated by Codex |
| `CHANGELOG.md` | Document all changes |

## Decision: Eigen as Required Dependency

### Why Eigen, not custom

Eigen is header-only, BSD-3/MPL2 licensed, cross-platform, and has zero runtime requirements. Unlike CUDA (needs GPU) or HiGHS (needs solver install), there's no reason to make it optional. It replaces ~260 lines of custom matrix code with a battle-tested library used by TensorFlow, ROS, and most scientific C++ projects.

### What Eigen gives us for free

- **Aligned memory allocation** (16/32-byte SIMD-ready) — automatic, no code changes
- **`Eigen::Map`** — zero-copy wrapping of numpy/MATLAB raw pointers, replaces our entire zero-copy binding effort
- **Vectorized fill/copy/zero** via SSE/AVX/NEON — automatic
- **Block operations** (`.col()`, `.row()`, `.block()`) — zero-copy views
- **`conservativeResize()`** — grow without losing data (useful for incremental computation)
- **Expression templates** — eliminates temporaries in chained operations

### Replacement plan

**`ScratchMatrix<T>`** → `Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>` directly.
- Column-major by default (matches current layout)
- `resize()`, `operator()(i,j)`, `.setZero()` all available
- Delete `scratch_matrix.hpp` entirely

**`DenseDistanceMatrix`** → Thin wrapper around packed triangular storage:

```cpp
class DenseDistanceMatrix {
  // Packed lower-triangular: N*(N+1)/2 elements
  Eigen::VectorXd data_;
  Eigen::Matrix<bool, Eigen::Dynamic, 1> computed_;  // or std::vector<bool>
  size_t n_{0};

  size_t tri_index(size_t i, size_t j) const {
    // Ensure i >= j for lower-triangular
    if (i < j) std::swap(i, j);
    return i * (i + 1) / 2 + j;
  }

public:
  void set(size_t i, size_t j, double v);
  double get(size_t i, size_t j) const;
  bool is_computed(size_t i, size_t j) const;

  // Export to full NxN for numpy/MATLAB
  Eigen::MatrixXd to_full_matrix() const;
  // CSV I/O, max(), count_computed()
};
```

### CMake integration

```cmake
# Add to root CMakeLists.txt:
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
# Or via FetchContent:
FetchContent_Declare(eigen URL https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.bz2)
FetchContent_MakeAvailable(eigen)

target_link_libraries(dtwc++ PUBLIC Eigen3::Eigen)
```

### Python bindings with Eigen::Map

```cpp
// Before (copies numpy → vector → DTW):
m.def("wdtw_distance", [](const std::vector<double> &x, ...) { ... });

// After (zero-copy numpy → Eigen::Map → DTW):
m.def("wdtw_distance", [](nb::ndarray<const double, nb::ndim<1>, nb::c_contig> x,
                           nb::ndarray<const double, nb::ndim<1>, nb::c_contig> y, ...) {
  Eigen::Map<const Eigen::VectorXd> mx(x.data(), x.size());
  Eigen::Map<const Eigen::VectorXd> my(y.data(), y.size());
  // ... call DTW with Eigen vectors
});
```

### Compile-time mitigation

Include only `<Eigen/Core>` (not `<Eigen/Dense>`). Use precompiled headers:

```cmake
target_precompile_headers(dtwc++ PUBLIC <Eigen/Core>)
```
