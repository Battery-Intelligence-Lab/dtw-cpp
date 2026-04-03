# DTWC++ Comprehensive Development Plan

## Context

DTWC++ is a high-performance C++ DTW library for time-series clustering with CUDA, Python/MATLAB bindings. The driving use case is **5 TB of variable-length time series** that won't fit in RAM. This fundamentally reshapes priorities: out-of-core streaming becomes the #1 infrastructure need, above micro-optimizations.

**Priorities:** (1) Performance, (2) Maintainability.  
**Philosophy:** Prefer license-compatible high-performance libraries over rewriting.  
**Cross-cutting rule:** Always be cache-friendly. Every loop must respect the memory layout (column-major for Eigen matrices, row-major for C arrays). Every new data structure must document its layout. Verify with benchmarks.

---

## Adversarial TODO Review

Before planning new work, here's what should happen to the existing TODO items:

| # | Current TODO Item | Verdict | Rationale |
|---|---|---|---|
| 1 | NaN/-ffast-math robustification | **SIMPLIFY** | `is_missing()` bitwise check works. But we can get both correctness AND performance by replacing `-ffast-math` with explicit sub-flags minus `-ffinite-math-only`. Then `std::isnan` works and we can simplify `missing_utils.hpp`. Needs benchmark to confirm no regression. |
| 2 | Eigen 5.0.1 exploration | **CUT** | Too vague. Eigen is used as aligned allocator + packed storage, not for linear algebra. No concrete Eigen 5 feature closes a gap. Delete this TODO. |
| 3 | Unnecessary memory allocations | **CUT** | Already solved. 30+ `thread_local` declarations, no naked `new`/`delete`, move semantics in place. This describes current state, not desired future. |
| 4 | Buffer > thread_local >> heap | **CUT** | Same as above -- this IS the current pattern. Convert to a code comment/guideline. |
| 5 | Contiguous arrays, not vector-of-vectors | **MODIFY** | `Data::p_vec` as `vector<vector>` is fine for variable-length series. The real problem is 5TB data that won't fit in RAM. Replace this TODO with out-of-core data source infrastructure. |
| 6 | One-liner std algorithms | **LOW PRIORITY** | No performance impact, but good for maintainability. Grab low-hanging fruits in Phase 3. |
| 7 | MIP odd-cycle cuts | **DEFER** | No measured MIP gap. Benders + FastPAM warm start works. Instrument gap first. |
| 8 | CUDA device-side pruning | **PARTIALLY DONE** | Pair-level LB pruning IS implemented (`compact_active_pairs`). Intra-kernel early-abandon is risky (warp divergence). Relabel. |
| 9 | Condensed distance matrix | **DONE on CPU** | `DenseDistanceMatrix` already uses packed triangular N*(N+1)/2. CUDA outputs full N*N temporarily but it's ephemeral. Remove from TODO. |
| 10 | HIPify for AMD | **REPLACE** | Instead of AMD GPUs, investigate ARM Mac Studios (Apple Silicon M-series) -- huge unified memory + excellent CPU parallelism. More practical target than AMD discrete GPUs. Could leverage Metal or Accelerate, or simply rely on strong ARM NEON auto-vectorization with `-march=native`. |

**Items MISSING from TODO (now added):**

| New Item | Impact | Rationale |
|---|---|---|
| `-march=native` in build system | **HIGH** | Not present anywhere. Free 10-20% from AVX2/AVX-512 auto-vectorization. One CMake option. |
| O(n) monotone-deque envelope computation | **MEDIUM** | Current `compute_envelopes` is O(n*band). Lemire (2009) deque algorithm is O(n). |
| `adtwBanded` rolling buffer | **MEDIUM** | Uses full `ScratchMatrix` O(n*m) instead of O(n) rolling column like `dtwBanded_impl`. |
| Early abandon for ADTW | **MEDIUM** | ADTW penalty only increases cost, so LB_Keogh is a valid lower bound. Enables pruned strategy. |
| **Out-of-core streaming data** | **CRITICAL** | 5TB dataset. CLARA subsample fits in RAM; assignment pass streams from disk sequentially. |

---

## Phased Execution Plan

### Phase 0: Build System Quick Wins (< 1 day)

**Goal:** Free performance from compiler flags with zero code risk.

#### 0A. Add `-march=native` CMake option

**File:** `cmake/StandardProjectSettings.cmake` (after line 53)

Add `DTWC_ENABLE_NATIVE_ARCH` option (default ON for top-level builds, OFF when consumed via FetchContent or building Python wheels). For GCC/Clang: `-march=native`. For MSVC: `/arch:AVX2`.

Guard: `PROJECT_IS_TOP_LEVEL AND NOT DTWC_BUILD_PYTHON` to keep wheels portable.

**Expected:** 10-20% speedup on distance matrix computation from auto-vectorized inner loops.

#### 0B. Replace `-ffast-math` with explicit sub-flags

**File:** `cmake/StandardProjectSettings.cmake` (lines 47-53)

Replace:
```cmake
-ffast-math
```
With:
```cmake
-fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros
```

This gives 95% of `-ffast-math` performance (FMA fusion, reassociation) while keeping `std::isnan()` working. Then simplify `missing_utils.hpp` -- replace bitwise `is_missing()` with `std::isnan()` wrapper.

**Risk:** Possible ~1-2% regression if `-ffinite-math-only` was enabling specific optimizations.  
**Mitigation:** Benchmark before/after with `bench_dtw_baseline`. If regression >3%, revert and keep bitwise approach.

#### 0C. O(n) envelope computation

**File:** `dtwc/core/lower_bound_impl.hpp` (lines 46-68)

Replace O(n*band) direct-scan `compute_envelopes` with Lemire (2009) sliding-window min/max. **Do NOT use `std::deque`** -- use a contiguous ring buffer (`std::vector<size_t>` with head/tail indices) for cache-friendly operation. Same change for `compute_envelopes_mv` (lines 304-331).

**Reference:** Lemire, "Streaming Maximum-Minimum Filter Using No More Than Three Comparisons per Element," 2006. Already in CITATIONS.md.

**Expected:** Significant improvement for large bands (band >= 50). Envelopes computed N times, each O(n) instead of O(n*band). Verify with `BM_compute_envelopes` benchmark.

---

### Phase 1: Index-Based Out-of-Core Architecture (1-2 weeks)

**Goal:** Enable clustering on 5 TB datasets that don't fit in RAM, with checkpointing, cross-binding support, and CPU/GPU runtime choice.

**Core insight:** Algorithms operate on **indices** [0, N). Indices always fit in memory (100M indices = 800 MB, plus labels/bookkeeping ~2-3 GB total). Actual series data is fetched only when DTW computation is needed. This decouples algorithm logic from data residency.

**Existing precedent:** `Problem::distByInd(i, j)` (Problem.cpp:324) already implements index-based lazy compute-and-cache: looks up indices in the distance matrix, computes DTW on cache miss, stores result. The streaming CLARA extends this pattern -- same index-based interface, but for disk-backed data where caching the full matrix is impossible, so we compute on-the-fly without caching (only N*k distances needed, not N^2).

#### 1A. `DataSource` interface -- index-centric design

**New file:** `dtwc/core/data_source.hpp`

The interface is index-based. Algorithms see indices; data fetching is the DataSource's concern:

```cpp
class DataSource {
public:
  virtual ~DataSource() = default;
  virtual size_t size() const = 0;                              // N (total series count)
  virtual size_t ndim() const = 0;                              // Features per timestep
  virtual size_t series_length(size_t index) const = 0;         // Length of series i
  virtual TimeSeriesView<data_t> get(size_t index) const = 0;   // Fetch series data by index
  virtual std::string name(size_t index) const = 0;             // Series name/ID
  
  // Block access: load a contiguous range of series into buffer (sequential I/O)
  // Returns count actually loaded. Buffer is reused across calls (resize, never shrink).
  virtual size_t load_block(size_t start, size_t count,
                            std::vector<std::vector<data_t>>& buffer) const = 0;
  
  // Hint for algorithm selection
  virtual bool is_in_memory() const = 0;
};
```

**`InMemoryDataSource`**: Wraps existing `Data` struct. Zero-overhead -- `get()` returns view into `p_vec[i]`. `is_in_memory()` = true. **Default for small datasets, no behavior change.**

**`MappedDataSource`**: Disk-backed via memory-mapped binary file. `is_in_memory()` = false. `get()` returns `TimeSeriesView` pointing into mmap region (zero-copy). `load_block()` for sequential streaming.

**Key design principles:**
- `Problem` stores a `std::shared_ptr<DataSource>` alongside `Data data` (not replacing it). Backward compatible: when null, falls back to `Data` directly. Existing code unchanged.
- **All algorithms work with indices.** The index arrays (labels, medoid_indices, sample_indices) always fit in RAM. Only the actual series data may be on disk.
- Algorithm selection uses `is_in_memory()`: full-matrix strategies for in-memory; streaming CLARA for disk-backed.
- **Cross-binding support:** Both Python (nanobind) and MATLAB (MEX) can construct `InMemoryDataSource` from numpy arrays / MATLAB matrices, or `MappedDataSource` from a file path. The interface is simple enough for both bindings.

#### 1B. Binary format with in-memory index

**New file:** `dtwc/core/mapped_data_source.hpp`

Binary format for variable-length series:

- **Index file** (`.dtwi`): `[uint64 N] [uint64 ndim] [uint64 offset_0] [uint64 length_0] ... [uint64 offset_{N-1}] [uint64 length_{N-1}]`
  - The entire index is loaded into RAM: 100M series = ~1.6 GB (2 × uint64 per series + header). Always fits.
  - Provides O(1) lookup of any series' offset and length.
- **Data file** (`.dtwd`): raw `data_t` values, concatenated series

**Memory-mapped I/O** for the data file:
- Cross-platform: [`mio`](https://github.com/vimpunk/mio) (MIT, header-only) or minimal custom wrapper (~100 lines using `mmap`/`CreateFileMapping`)
- OS handles page-in/page-out automatically
- `get(i)` returns a `TimeSeriesView` pointing directly into the mmap region -- zero-copy
- Sequential access patterns (block streaming) get excellent OS prefetching
- Random access (subsample loading) works too -- OS pages in on demand

**Conversion tool:** Utility to convert CSV/HDF5 to binary format (offline preprocessing step).

#### 1C. Block-streaming CLARA with index-based iteration

**File:** `dtwc/algorithms/fast_clara.cpp` -- modify `assign_all_points()`

Current `assign_all_points` requires all data in RAM. New streaming version:

```
assign_all_points_streaming(DataSource& source, medoid_series[k], labels_out[N]):
  // labels_out is an index array -- always fits in RAM (N × 4 bytes)
  total_cost = 0
  block_size = RAM_budget / avg_series_bytes  // e.g., 1 GB block
  for start in range(0, N, block_size):
    count = source.load_block(start, block_size, buffer)  // Sequential disk read
    // OpenMP parallel: each thread handles independent series in the block
    for p in [0, count):
      for m in [0, k):
        d = dtw(buffer[p], medoid_series[m])
        track nearest
      labels_out[start + p] = nearest_medoid
      total_cost += nearest_dist
    // Block done -- buffer memory reused for next block
  return total_cost
```

**"Load once, do many operations, then reload."** Each block loads once, computes all k medoid distances per series, then discards. I/O cost per full scan:
- NVMe: ~28 min for 5 TB
- SATA SSD: ~2.8 hours
- HDD: ~9.3 hours

For CLARA with 5 subsamples: 5 full scans. Subsample phase (FastPAM on ~1000 series) is negligible.

#### 1D. Subsample loading and sample size scaling

`CLARAOptions.sample_size` is already user-adjustable (default -1 = auto). But the current auto formula `max(40 + 2*k, min(N, 10*k + 100))` caps at ~200 for k=10 -- far too small for 100M series. For large N, a better default might be `max(40 + 2*k, min(N, max(10*k + 100, sqrt(N))))`. For N=100M, sqrt(N) ~= 10K, which is reasonable (10K^2 = 100M entries = ~800 MB for the subsample distance matrix). This should be benchmarked to find the quality/cost sweet spot.

Loading a random subsample from disk:

1. Select `sample_size` random indices (indices always in RAM)
2. Sort selected indices by file offset (minimize disk seeks)
3. Read series in sorted order through `DataSource::get(index)` (mmap handles page-in)
4. Copy into `InMemoryDataSource` for FastPAM

For sample_size=10K with ~8KB series: ~80 MB of reads, fast on any storage.

#### 1E. Distance matrix and assignment checkpointing

**File:** `dtwc/core/checkpoint.hpp` (new or extend existing checkpointing)

For long-running out-of-core CLARA, checkpoint progress to disk periodically:

1. **Subsample distance matrix:** Already small (~4 MB for 1000 series). Save after computation for each subsample so FastPAM can resume.
2. **Assignment results:** Save `labels[N]` + `costs[N]` + `best_total_cost` + `current_subsample_index`. At 100M series: ~1.2 GB per checkpoint. Allows resuming the assignment scan at the last completed block.
3. **Best-so-far state:** Which subsample gave the best cost, its medoid indices, its labels.

Checkpoint format: simple binary header + arrays. Load on startup, skip completed work.

The existing `save_checkpoint` / `load_checkpoint` in Python bindings provides precedent -- extend to C++ core for the streaming CLARA path.

#### 1F. CPU vs GPU runtime choice

The `DataSource::is_in_memory()` flag plus series length and N determine the optimal compute path:

| Data location | N | Series length | Best path |
|---|---|---|---|
| In-memory | Small (<10K) | Any | CPU full matrix (BruteForce/Pruned) |
| In-memory | Medium | Any | GPU full matrix if available |
| Disk-backed | Any | Short (<1K pts) | GPU streaming K-vs-N (compute-bound, GPU wins) |
| Disk-backed | Any | Long (>1K pts) | CPU streaming (more data per series fills GPU RAM faster) |

This is a **runtime heuristic**, not a hard rule. Expose as `DistanceMatrixStrategy::Auto` logic. User can always override.

#### 1G. OpenMP-parallel block assignment

```cpp
#pragma omp parallel for reduction(+:total_cost) schedule(dynamic, 64)
for (size_t p = 0; p < block_count; ++p) {
  // Compute k distances for series p, find nearest medoid
}
```

Each thread works on independent series within the block. No synchronization needed.

---

### Phase 2: CPU Performance Improvements (1 week)

#### 2A. `adtwBanded` rolling buffer

**File:** `dtwc/warping_adtw.hpp` (lines 117-181)

Note: `ScratchMatrix` IS an Eigen matrix (thin wrapper adding `fill()`). The issue is not ScratchMatrix itself but that `adtwBanded` allocates a **full 2D Eigen matrix** O(n*m) when only 1-2 columns are needed for the banded recurrence. Replace with a `thread_local std::vector<data_t>` rolling column, identical to how `dtwBanded_impl` in `warping.hpp` works. The ADTW recurrence `min(diag, left+penalty, below+penalty)` is compatible with this technique.

**Cache note:** The rolling column is contiguous and fits in L1 cache for typical band sizes. Ensure the inner loop iterates along the contiguous dimension.

**Expected:** Reduced memory from O(n*m) to O(n), modest cache improvement. Verify with benchmarks.

#### 2B. Early abandon for ADTW

**File:** `dtwc/warping_adtw.hpp`

Add `early_abandon` parameter to `adtwFull_L`. Track `row_min` in the inner loop (same pattern as `dtwFull_L_impl`). ADTW penalties only increase cost, so LB_Keogh is a valid lower bound.

Then update `pruned_strategy_applicable()` in `Problem.cpp` (line 338) to allow `Pruned` strategy for ADTW variant.

**Expected:** 30-60% pair pruning for correlated ADTW datasets.

#### 2C. OpenMP scheduling for triangular workload

**File:** `dtwc/Problem.cpp` (line 364)

The brute-force distance matrix uses `schedule(dynamic, 1)` over rows. Row 0 processes N-1 pairs, row N-1 processes 0. This creates load imbalance.

Options:
1. Switch to linearized pair index with `schedule(dynamic, 16)` (like `pruned_distance_matrix.cpp` line 188)
2. Increase chunk size to `schedule(dynamic, 4)` and reverse row order
3. Use `schedule(guided)` which auto-adapts chunk sizes

Benchmark all three, pick the winner.

#### 2D. NaN cleanup (after Phase 0B)

**File:** `dtwc/missing_utils.hpp`

Once `-ffinite-math-only` is disabled, simplify `is_missing()` to:
```cpp
template <typename T>
inline bool is_missing(T val) noexcept { return std::isnan(val); }
```

Remove the bitwise IEEE 754 manipulation. Keep the function name for semantic clarity.

---

### Phase 3: CUDA & Bindings (2-4 weeks, parallelizable)

#### 3-GPU. GPU RAM-aware streaming for large datasets

GPU RAM is even more constrained than system RAM (typically 8-48 GB vs hundreds of GB). For disk-backed data:

- **Subsample phase (FastPAM):** The subsample (~1000 series) easily fits in GPU memory. Upload once, compute subsample distance matrix on GPU, run FastPAM on CPU with the result.
- **Assignment phase (K-vs-N):** Stream blocks from disk -> pin in host memory -> async copy to GPU -> compute k distances per series -> copy results back -> discard block. Use double-buffering: while GPU computes block N, CPU loads block N+1 from disk.
- **Key API:** Wire the existing `compute_dtw_k_vs_all` kernel (already in `cuda_dtw.cuh` lines 108-117) into the streaming CLARA assignment pass. This kernel computes distances from k query series against N target series -- exactly what the assignment pass needs.

The `DataSource::is_in_memory()` flag also gates CUDA strategy: full GPU distance matrix only for in-memory data; streaming K-vs-N for disk-backed.

#### 3A. CUDA: Relabel device-side pruning as partially done

Pair-level LB pruning (`compact_active_pairs` in `cuda_dtw.cu` line 1555) is already implemented. Mark done.

Remaining: architecture-aware dispatch (`DispatchProfile` by compute capability) and wiring K-vs-N kernel into CLARA loop for GPU-accelerated assignment pass.

#### 3B. Python PyPI first release

The CI is ready (`python-wheels.yml` with cibuildwheel, OIDC trusted publishing). The blocker is GitHub trusted publisher setup. This is a configuration task, not code.

#### 3C. MATLAB Phase 2 parity

Add MIPSettings, CUDA dispatch, checkpointing, and I/O utilities to the MEX gateway (`bindings/matlab/dtwc_mex.cpp`). Create `compile.m` standalone build script.

#### 3D. CUDA wavefront cleanup

Clean up the unreachable preload branch (preload=true requires max_L<=512, but wavefront only launches for max_L>256, so preload is only reachable for 257-512 range). Either:
- Remove the preload path and let regtile handle up to 512 (requires extending TILE_W)
- Or keep it but document the valid range clearly

#### 3E. Unified kernel dispatch

Consolidate the dispatch logic in `launch_dtw_kernel`, `launch_dtw_one_vs_all_kernel`, and `launch_dtw_k_vs_all_kernel` into a single dispatch table.

#### 3F. ARM Mac Studio investigation (replaces HIPify)

Instead of AMD GPU HIPify, investigate Apple Silicon Mac Studios as a target platform:
- Unified memory (up to 192 GB on M2 Ultra) eliminates CPU<->GPU copy overhead
- Excellent CPU parallelism (up to 24 cores)
- ARM NEON auto-vectorization via `-march=native` (covered by Phase 0A)
- Potential Metal compute shaders for GPU-like parallelism
- Lower effort than HIPify: CPU path already works on ARM, just needs testing + tuning

#### 3G. Low-hanging std algorithm cleanups

Replace verbose for-loops with one-liner std algorithms where it improves readability without hurting performance. Focus on non-hot-path code (I/O, setup, result aggregation). Leave hot inner loops untouched.

---

### Phase 4: Deferred / Cut Items

| Item | Status |
|---|---|
| Eigen 5.0.1 exploration | **CUT** - no concrete gap identified |
| Memory allocation audit | **CUT** - already optimized with thread_local pattern |
| Buffer > thread_local >> heap | **CUT** - already the convention |
| Condensed distance matrix | **DONE** on CPU, CUDA N*N is ephemeral |
| MIP odd-cycle cuts | **DEFERRED** - instrument Benders gap first, implement only if >50 iterations observed |
| HIPify for AMD | **REPLACED** with ARM Mac Studio investigation (Phase 3F) |
| Two-phase clustering | **DEFERRED** - until streaming infra (Phase 1) is proven |
| One-liner std algorithms | **DEFERRED** - style only, Phase 3+ maintenance pass |

---

## External Libraries to Consider

| Library | License | Purpose | Decision |
|---|---|---|---|
| **mio** | MIT | Cross-platform mmap (header-only, ~500 stars) | **EVALUATE** -- simple and clean, but verify it's maintained and handles large files (>4GB) correctly on Windows. Alternatives: **Boost.Iostreams** (heavy), **cpp-mmf** (MIT), or a minimal ~100-line custom wrapper using `mmap`/`CreateFileMapping`. Pick the one with least dependency weight that handles 5TB files on all platforms. |
| **Google Highway** | Apache-2.0 | SIMD acceleration | Already integrated. Per LESSONS.md, DTW inner loop has limited SIMD benefit (1.29x max due to 10-cycle recurrence). Focus SIMD effort on embarrassingly parallel ops: LB_Keogh (already done), z_normalize, envelope computation, block distance reductions. Only pursue new SIMD work where profiling shows it would make a huge difference given the overhead. |
| **Eigen 5** | MPL2 | Aligned allocation, packed storage | Already a dependency, no new usage needed |
| **Intel oneTBB** | Apache-2.0 | Task parallelism | **DEFER** - OpenMP adequate for now |
| **HDF5** | BSD-like | Chunked data storage | Consider as alternative DataSource backend (users may already have HDF5 data) |

---

## Verification Strategy

**HARD RULE: Every performance change MUST be verified by benchmarks before/after.** No performance claim without measured evidence. Use `benchmarks/bench_dtw_baseline.cpp` (Google Benchmark) as the primary tool. Record baseline numbers BEFORE making changes, then compare after.

### Phase 0 verification

- **Benchmark (mandatory):** Run `bench_dtw_baseline` BEFORE any changes. Save JSON output as baseline.
- After `-march=native` + flag changes: re-run `bench_dtw_baseline`, compare `BM_dtwFull_L`, `BM_dtwBanded`, `BM_fillDistanceMatrix`. Expect 10-20% improvement. If regression >3%, investigate.
- After O(n) envelopes: compare `BM_compute_envelopes` with band=10, 50, 100. Expect proportional improvement with band size.
- Run full test suite (`ctest`) to confirm no regressions.
- Specifically run `tests/unit/adversarial/adversarial_missing_utils.cpp` to verify NaN handling still works.

### Phase 1 verification

- Unit tests for `DataSource` interface, `InMemoryDataSource`, `MappedDataSource`
- Integration test: create binary dataset from existing CSV test data, run CLARA through `MappedDataSource`, compare clustering results with in-memory CLARA (must produce identical results)
- **Benchmark (mandatory):** Measure streaming throughput (GB/s) on sequential block reads vs in-memory baseline
- Test checkpoint save/load roundtrip: interrupt CLARA mid-run, resume, verify same final result
- Cross-binding test: verify Python and MATLAB can construct and use both DataSource types

### Phase 2 verification

- **Benchmark (mandatory):** Run `bench_dtw_baseline` for ADTW benchmarks before/after rolling buffer + early abandon. Record exact numbers.
- Run `tests/unit/unit_test_adtw.cpp` for correctness
- Test pruned strategy with ADTW variant enabled -- verify same clustering results as BruteForce
- **Benchmark (mandatory):** Compare OpenMP scheduling variants with `BM_fillDistanceMatrix` at N=100, N=500, N=1000. Pick winner based on numbers, not theory.

### Phase 3 verification

- CUDA: `tests/unit/test_cuda_correctness.cpp`
- Python: `tests/python/` full suite + manual PyPI test upload
- MATLAB: `tests/matlab/test_dtwc.m`

---

## Critical Files Summary

| File | Changes |
|---|---|
| `cmake/StandardProjectSettings.cmake` | `-march=native`, replace `-ffast-math` with explicit sub-flags |
| `dtwc/core/lower_bound_impl.hpp` | O(n) monotone-deque envelopes |
| `dtwc/missing_utils.hpp` | Simplify to `std::isnan()` after flag change |
| `dtwc/core/data_source.hpp` | **NEW** - DataSource interface |
| `dtwc/core/mapped_data_source.hpp` | **NEW** - Memory-mapped disk-backed data |
| `dtwc/Data.hpp` | Add `InMemoryDataSource` adapter |
| `dtwc/Problem.hpp` | Store `shared_ptr<DataSource>` instead of `Data data` |
| `dtwc/Problem.cpp` | Adapt `fillDistanceMatrix`, `distByInd` to use DataSource |
| `dtwc/algorithms/fast_clara.cpp` | Streaming `assign_all_points` for out-of-core CLARA |
| `dtwc/warping_adtw.hpp` | Rolling buffer + early abandon |
| `.claude/TODO.md` | Rewrite with corrected priorities |
