# Plan: Mmap Benchmarks + Architecture for First-Class Mmap Storage

## Context

We implemented mmap distance matrices (Steps 1-5) and benchmarked warmstart (36x speedup at N=25). Now we need:
1. Rigorous benchmarks at N=5000 comparing mmap vs in-memory access patterns
2. A path toward mmap as the **default** storage backend (not optional)
3. Mmap for time series data (not just distance matrices)
4. Use Eigen::Map to wrap mmap'd series data — eliminate hand-written view code

**Key insight**: Eigen::Map doesn't help for the packed triangular distance matrix (irregular `tri_index` stride), but it's perfect for time series — each series is contiguous doubles, and `Eigen::Map<const VectorXd>` gives SIMD-optimized operations for free.

## Phase 1: Benchmark Suite

**New file:** `benchmarks/bench_mmap_access.cpp`
**Modify:** `benchmarks/CMakeLists.txt` (add target)

Generate N=5000 random series (L=25, seed=42) in-place. Run 7 benchmarks using Google Benchmark:

| Benchmark | What it measures |
|-----------|-----------------|
| `BM_fill_dense` / `BM_fill_mmap` | Sequential fill of 12.5M doubles (~95MB) — DTW compute + write |
| `BM_random_get_dense` / `BM_random_get_mmap` | 500K random `tri_index` lookups — FastPAM-like pattern |
| `BM_sequential_scan_dense` / `BM_sequential_scan_mmap` | Sum all packed doubles — mmap best case |
| `BM_open_mmap` / `BM_read_binary_to_vector` | Startup: mmap open (lazy) vs fread (eager ~95MB) |
| `BM_medoid_access_pattern` | k=10 medoids, look up distances for all N points — realistic FastPAM |
| `BM_series_contiguous` / `BM_series_vec_of_vec` | Iterate N=5000 series of L=25: flat array vs vector\<vector\> |
| `BM_clara_copy` / `BM_clara_view` | CLARA subsample: copy 100 vectors vs create 100 Eigen::Map views |

**Decision gate after benchmarks:**
- If mmap random access is >3x slower than dense at N=5000 → implement eager-read fallback
- If contiguous series access is meaningfully faster → proceed to MmapDataStore

**Files:**
- Create: `benchmarks/bench_mmap_access.cpp`
- Modify: `benchmarks/CMakeLists.txt`

## Phase 2: Make llfio Non-Optional

Remove all `DTWC_ENABLE_MMAP` / `DTWC_HAS_MMAP` conditional compilation. llfio becomes a core dependency like Eigen.

**Files to modify (remove `#ifdef DTWC_HAS_MMAP` guards):**

| File | Change |
|------|--------|
| `cmake/Dependencies.cmake` | Remove `option(DTWC_ENABLE_MMAP ...)`, always fetch llfio |
| `dtwc/CMakeLists.txt` | Always link `llfio_hl`, always define `DTWC_HAS_MMAP` |
| `dtwc/Problem.hpp` | `distMat_t` always `std::variant`, remove `#ifdef` from visit_distmat |
| `dtwc/core/mmap_distance_matrix.hpp` | Remove outer `#ifdef` |
| `dtwc/dtwc.hpp` | Remove `#ifdef` around mmap include |
| `dtwc/core/matrix_io.hpp` | Remove `#ifdef` around MmapDistanceMatrix `operator<<` |
| `dtwc/dtwc_cl.cpp` | Remove `#ifdef` around mmap threshold logic |
| `dtwc/Problem.cpp` | Remove `#ifdef` around `use_mmap_distance_matrix()` |
| `tests/unit/core/unit_test_mmap_distance_matrix.cpp` | Remove outer `#ifdef` |
| `tests/unit/unit_test_variant_distmat.cpp` | Remove `#ifdef` around mmap tests |
| `.claude/CLAUDE.md` | Update: llfio is now a core dependency |

## Phase 3: Mmap Time Series with Eigen::Map

### 3a. Internal mmap cache for series data

**The series data format is user-supplied** (CSV, HDF5, Parquet, folder of files). We do NOT impose a binary format on users. Instead, the mmap layer is an **internal cache**:

1. User provides data in any format → `DataLoader::load()` reads it (unchanged)
2. After loading, we write a contiguous mmap cache file (internal, in output_dir)
3. On `--restart`, if cache exists and is fresh (hash of input path + file sizes + mtimes), use cache instead of re-loading
4. Cache file is auto-generated, auto-cleaned, never user-facing

**Cache binary format** (same pattern as distance matrix cache):

```text
Header (64 bytes):
  magic "DTWS", version, endian, elem_size, N, ndim, CRC32, reserved

Offset table (N+1 uint64 entries):
  offsets[i] = byte offset from data start to series i
  offsets[N] = total data size (sentinel)

Data section:
  series 0 doubles, series 1 doubles, ..., contiguous, no padding
```

### 3b. MmapDataStore class

**New file:** `dtwc/core/mmap_data_store.hpp`

Wraps the internal cache. Created from loaded `Data`, reopened on restart.

```cpp
class MmapDataStore {
  llfio::mapped_file_handle mfh_;
  const double* data_base_;
  const uint64_t* offsets_;
  size_t n_, ndim_;

public:
  // Create cache from already-loaded Data (writes mmap file)
  static MmapDataStore create(const fs::path& cache_path, const Data& data);
  // Reopen existing cache (for --restart)
  static MmapDataStore open(const fs::path& cache_path);

  size_t size() const;
  size_t ndim() const;

  // Return Eigen::Map view of series i — zero copy, SIMD-ready
  Eigen::Map<const Eigen::VectorXd> series(size_t i) const {
    return {data_base_ + offsets_[i] / sizeof(double),
            static_cast<Eigen::Index>(series_flat_size(i))};
  }

  const double* series_data(size_t i) const;
  size_t series_length(size_t i) const;  // timesteps
  size_t series_flat_size(size_t i) const;  // timesteps * ndim
};
```

### 3c. Integration with Data/Problem

**Strategy:** Add a `DataAccessor` that abstracts over vector-of-vectors vs mmap. Both DTW paths already accept `const double*, size_t` — the accessor just provides different sources for that pointer.

```cpp
// In Data.hpp or a new accessor header
struct DataAccessor {
  virtual size_t size() const = 0;
  virtual const double* series_data(size_t i) const = 0;
  virtual size_t series_flat_size(size_t i) const = 0;
  virtual size_t ndim() const = 0;
  virtual ~DataAccessor() = default;
};

// VectorDataAccessor: wraps existing vector<vector<double>>
// MmapDataAccessor: wraps MmapDataStore
```

`Problem::p_vec(i)` currently returns `const vector<double>&`. Migration:
1. DTW functions already accept `const double*, size_t` internally
2. Change `rebind_dtw_fn()` lambda to use accessor pointer interface
3. Existing `vector<vector<double>>` code keeps working through VectorDataAccessor

### 3d. CLARA fix

Current code copies full vectors into sub-Problems. With Eigen::Map views:

- Create Eigen::Map views pointing into parent's data (zero copy, works with both heap and mmap backends)
- Sub-Problem holds index mapping + reference to parent accessor
- DTW on subsample reads directly from mmap — no allocation

### 3e. Eigen::Map benefits

Where Eigen::Map replaces hand-written code:
- `TimeSeriesView` (currently unused) → replaced by `Eigen::Map<const VectorXd>`
- Element-wise ops in DTW metric functions → Eigen SIMD for free
- z-normalization of series → `(series - mean) / stddev` as Eigen expression
- CLARA subsample "copy" → just a vector of Eigen::Map views (zero copy)
- No new dependency — Eigen is already linked

## Phase 4: Storage Policy

Add auto-selection to Problem:

```cpp
enum class StoragePolicy { Auto, Heap, Mmap };
```

- `Auto` (default): N < threshold → Dense in heap, N >= threshold → Mmap
- `Heap`: force `DenseDistanceMatrix` + vector-of-vectors
- `Mmap`: force `MmapDistanceMatrix` + `MmapDataStore`

Default threshold: determined by Phase 1 benchmarks (likely 5000-10000).

For small files on mmap: the OS page cache makes this fast anyway. If benchmarks show no penalty, we can lower the threshold to 0 (always mmap) and simplify the code.

## Implementation Order

```
Phase 1 (benchmarks)  ← START HERE, informs all decisions
    │
    ├── Phase 2 (remove #ifdef guards) — independent, low risk
    │
    └── Phase 3 (mmap series + Eigen::Map)
            │
            └── Phase 4 (storage policy + auto-selection)
```

## Verification

- Phase 1: `cmake -B build -DDTWC_BUILD_BENCHMARK=ON && cmake --build build --target bench_mmap_access && ./build/bin/bench_mmap_access`
- Phase 2: `ctest --test-dir build -C Release -j8` — all existing tests pass
- Phase 3: New tests for MmapDataStore + Eigen::Map series access
- Phase 4: CLI with `--storage auto|heap|mmap` flag
- Full suite: 64+ tests pass after each phase

## What NOT to do

- Don't use Eigen::Map for packed triangular distance matrix — Eigen has no packed symmetric storage. `SelfAdjointView`/`TriangularView` both require full NxN underneath. Our `tri_index` + `double*` is already optimal (1-2 cycles, benchmarked)
- Don't add a virtual DataAccessor until benchmarks prove the need (YAGNI unless Phase 1 shows contiguous is significantly faster)
- Don't change the DTW core recurrence — it's already pointer-based
- Don't remove CSV checkpoint — it coexists with binary checkpoint for human readability
