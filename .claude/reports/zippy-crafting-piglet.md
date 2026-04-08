# Plan: Auto-Algorithm Selection + Memory-Mapped Distance Matrix

## Context

Target: 100K+ time series. At N=100K the packed triangular distance matrix is N*(N+1)/2 doubles = ~40GB — doesn't fit in RAM. CLARA already avoids computing the full matrix by subsampling, but currently all series data must be loaded into RAM and there's no way to know N before loading. The user must also manually pick `--method`.

**Goal**: Count series before loading, auto-select algorithm, and support mmap-backed distance matrix for when it IS needed (e.g., FastPAM on moderate N, or checkpoint/resume).

## 5 Incremental Steps (each independently testable)

### Step 1: Count-Before-Load

Add `DataLoader::count()` — returns number of series without loading data.

**File**: `dtwc/DataLoader.hpp`

- Directory mode: count entries via `fs::directory_iterator` (no file reads)
- Batch file mode: count lines (skip `start_row` header lines), respects `Ndata` limit
- ~20 lines of code

**Test**: Unit test with temp folder/file, assert count matches.

### Step 2: Auto Method Selection

Add `"auto"` to `--method` (make it the default). After loading data, pick algorithm based on `prob.size()`.

**File**: `dtwc/dtwc_cl.cpp`

1. Change default: `std::string method = "auto";`
2. Add `{"auto", "auto"}` to the CheckedTransformer map (line 118)
3. After `Problem prob{prob_name, dl};` (line 346), insert:

```cpp
if (method == "auto") {
  const size_t N = prob.size();
  method = (N <= 5000) ? "pam" : "clara";
  if (verbose)
    std::cout << "Auto-selected method: " << method << " (N=" << N << ")\n";
}
```

4. For large N CLARA, scale sample size with sqrt(N):

```cpp
if (clara_opts.sample_size < 0 && prob.size() > 50000)
  clara_opts.sample_size = static_cast<int>(std::sqrt(prob.size()) * n_clusters);
```

**Test**: Run CLI with `--method auto` on Coffee dataset (28 series → should pick pam).

### Step 3: Memory-Mapped Distance Matrix

Cross-platform mmap wrapper + `MmapDistanceMatrix` class with same API as `DenseDistanceMatrix`.

**New file**: `dtwc/core/mmap_distance_matrix.hpp`

**Dependency**: [llfio](https://github.com/mandreyel/llfio) (MIT, header-only, C++11, no deps).
Add via CPM in `cmake/Dependencies.cmake`. Fits the project pattern (header-only deps like rapidcsv, Eigen).

**Binary file layout** (internal, not user-facing):

```
bytes 0-7:    uint64_t magic ("DTWMAT01")
bytes 8-15:   uint64_t n
bytes 16...:  double[n*(n+1)/2] (NaN = uncomputed)
```

**Implementation** using llfio (handles mmap, file locking, pre-allocation cross-platform):

```cpp
class MmapDistanceMatrix {
  llfio::mapped_file_handle mfh_;  // RAII: mmap + file handle
  double* data_ = nullptr;         // points past header
  size_t n_ = 0;

  // Same API as DenseDistanceMatrix
  double get(size_t i, size_t j) const;
  void set(size_t i, size_t j, double v);
  bool is_computed(size_t i, size_t j) const;
  size_t size() const { return n_; }

  // Periodic flush (mmap pages auto-flush, but explicit sync for safety)
  void sync();
};
```

**Binary header** (24 bytes, from adversarial review):

| Offset | Size | Field | Purpose |
|--------|------|-------|---------|
| 0 | 4 | magic `"DTWM"` | Format identification |
| 4 | 2 | version (1) | Future format evolution |
| 6 | 4 | endian marker `0x01020304` | Detect byte-order mismatch |
| 10 | 1 | elem_size (8) | float vs double detection |
| 11 | 1 | reserved | Padding |
| 12 | 8 | N (uint64) | Number of series |
| 20 | 4 | data CRC32 | Corruption detection |

**Key features**:
- **Warmstart**: Open existing file → mmap → NaN entries are uncomputed, resume from there
- **Periodic save**: mmap pages auto-flush to disk by OS. Add explicit `sync()` every N pairs for crash safety
- **File locking**: llfio provides byte-range locking — prevent two processes corrupting same cache
- **Pre-allocation**: llfio uses `posix_fallocate`/`SetEndOfFile` — no sparse file SIGBUS risk
- **Header written last**: On create, write data (NaN-filled) first, header last. Crash during init → no valid header → detected on reopen

**Test**: Create, write, close, reopen (warmstart), verify values persist. Test NaN sentinel. Test concurrent open (lock check). Test N=0, 1, 1000.

### Step 4: Integrate into Problem via std::variant

**Adversarial review finding**: Do NOT extend `DenseDistanceMatrix` with a raw pointer — vector reallocation invalidates it (P0 bug). Use a **separate class** behind `std::variant`.

**File**: `dtwc/Problem.hpp`

```cpp
using DistMat = std::variant<core::DenseDistanceMatrix, core::MmapDistanceMatrix>;
DistMat distMat_;

// Helper to dispatch get/set through the variant
double distMat_get(size_t i, size_t j) const {
  return std::visit([&](auto& m) { return m.get(i, j); }, distMat_);
}
void distMat_set(size_t i, size_t j, double v) {
  std::visit([&](auto& m) { m.set(i, j, v); }, distMat_);
}
```

**File**: `dtwc/Problem.cpp`

In `fillDistanceMatrix()` and lazy-alloc in `distByInd()`:

```cpp
constexpr size_t MMAP_THRESHOLD = 50000;
if (N > MMAP_THRESHOLD) {
  auto cache = output_folder / (name + "_distmat.cache");
  distMat_ = core::MmapDistanceMatrix(cache, N);  // create or warmstart
} else {
  distMat_ = core::DenseDistanceMatrix(N);
}
```

Add periodic sync during fill (every 10000 pairs for mmap path).

**Test**: Existing 63 tests pass (N < 50K → in-memory path). Add test forcing mmap with small N.

### Step 5: Clustering Checkpoint + Restart (resolves GitHub issue #22)

Save clustering state alongside the distance matrix so both distance computation AND clustering can resume after crash/exit.

**What gets saved** (alongside the mmap distance matrix cache file):

- `name_checkpoint.bin` — small binary file:
  - Current medoid indices (`vector<int>`, k values)
  - Current labels (`vector<int>`, N values)
  - Best cost so far
  - Current iteration number
  - Algorithm identifier (pam/clara/lloyd)

**Warmstart flow** (CLI `--restart` flag):

1. Open existing distance matrix cache → mmap → NaN entries are uncomputed
2. Load medoid checkpoint if present
3. Resume: fill remaining NaN distance entries, then continue clustering from saved medoids
4. For CLARA: save best-so-far result after each subsample iteration

**Integration**:

- `Problem::save_checkpoint()` — writes medoids + labels + cost
- `Problem::load_checkpoint()` — reads and restores state
- FastPAM: checkpoint after each SWAP iteration
- CLARA: checkpoint after each subsample evaluation
- Lloyd: checkpoint after each iteration

**Files**: `dtwc/checkpoint.hpp`, `dtwc/checkpoint.cpp`, `dtwc/dtwc_cl.cpp` (add `--restart` flag)

**This resolves [Battery-Intelligence-Lab/dtw-cpp#22](https://github.com/Battery-Intelligence-Lab/dtw-cpp/issues/22)** — the request for a restart option after crash/exit on HPC systems.

### Step 6: Streaming CLARA (deferred — after Steps 1-5 proven)

For N > 50K, avoid loading all series into RAM. This is the most complex step.

**File**: `dtwc/DataLoader.hpp` — Add `load_subset(vector<int> indices)`
**File**: `dtwc/fileOperations.hpp` — Add `load_folder_subset`, `load_batch_file_subset`
**File**: `dtwc/algorithms/fast_clara.cpp` — Add `fast_clara_streaming(DataLoader&, N, opts)`
**File**: `dtwc/dtwc_cl.cpp` — Branch: if clara and N > 50K, use streaming path

The streaming path:

1. `dl.count()` → get N
2. For each subsample: `dl.load_subset(sample_indices)` → load only s series
3. Build sub-Problem, run FastPAM → get medoid indices
4. Assignment: iterate all N series one-by-one, compute distance to k medoids, assign label, discard series
5. Memory: O(s² + k * L) instead of O(N * L + N²)

**Defer this step** until Steps 1-4 are proven in practice.

## Implementation Order

```
Step 1 (count)   ─── independent ──┐
Step 3 (mmap)    ─── independent ──┤
                                   ├── Step 2 (auto-select, needs Step 1)
                                   ├── Step 4 (integrate mmap, needs Step 3)
                                   ├── Step 5 (checkpoint + restart, needs Step 3+4, resolves #22)
                                   └── Step 6 (streaming CLARA, needs 1+2+3+4+5)
```

**Steps 1 and 3 can be implemented in parallel** (different files, no overlap).

## Files to Modify

| File | Steps | Change |
|------|-------|--------|
| `dtwc/DataLoader.hpp` | 1, 6 | Add `count()`, later `load_subset()` |
| `dtwc/dtwc_cl.cpp` | 2, 5, 6 | Add "auto" method, `--restart` flag, streaming |
| `dtwc/core/mmap_distance_matrix.hpp` (NEW) | 3 | MmapDistanceMatrix using llfio |
| `cmake/Dependencies.cmake` | 3 | Add llfio via CPM |
| `dtwc/Problem.hpp` | 4 | `std::variant` distMat, checkpoint methods |
| `dtwc/Problem.cpp` | 4, 5 | Auto-select mmap, save/load checkpoint |
| `dtwc/checkpoint.{hpp,cpp}` | 5 | Extend with medoid/label/cost checkpoint |
| `dtwc/algorithms/fast_pam.cpp` | 5 | Checkpoint after each SWAP iteration |
| `dtwc/algorithms/fast_clara.{hpp,cpp}` | 5, 6 | Checkpoint per subsample, streaming variant |
| `dtwc/fileOperations.hpp` | 6 | Subset loading functions |

## Verification

- All 63 existing unit tests must pass after each step
- Stress test (`tests/integration/stress_test_cli.sh`) must pass
- Step 2: `dtwc_cl --method auto -k 3 -i data/dummy` should print "Auto-selected method: pam (N=18)"
- Step 3: Unit test creates 1000x1000 mmap matrix, writes/reads, reopens from file
- Step 4: `dtwc_cl -k 3 -i large_dataset/` with N > 50K should create `.cache` file

## Open Decisions (from adversarial reviews)

### llfio vs mio vs custom wrapper
- **llfio**: has file locking + pre-allocation built in, active. But heavy CMake, slow configure (~30s), complex internals.
- **mio**: clean 3-line API, header-only, trivial CPM. But dormant, no file locking, no pre-allocation.
- **custom ~80-line wrapper**: zero deps, full control. But maintenance burden, platform-specific bugs.
- **Decision needed at implementation time**: try llfio first. If CMake integration is painful, fall back to mio + manual `flock`/`LockFileEx` (~20 extra lines).

### std::visit overhead on hot path
- `distByInd` is called millions of times. `std::visit` on variant adds indirect call overhead per access.
- **Fix**: Resolve variant once at algorithm entry (e.g., in `fillDistanceMatrix`, `fast_pam`, `cluster_by_kMedoidsLloyd`). Pass concrete `DenseDistanceMatrix&` or `MmapDistanceMatrix&` via template to inner loops. The variant lives in Problem but hot paths don't go through it.

### CLARA + mmap interaction
- Auto-selecting CLARA for N>50K but also allocating full mmap matrix is contradictory — CLARA doesn't need the full N²/2 matrix.
- **Fix**: mmap distance matrix ONLY when method explicitly requires full matrix (pam, kmedoids, hierarchical, mip). CLARA skips it entirely (as it already does). The auto-select logic should NOT trigger mmap when it picks CLARA.

### Checkpoint format unification
- Existing CSV checkpoint (`checkpoint.hpp`) vs new binary checkpoint coexist.
- **Fix**: Keep CSV checkpoint for small N / human-readable use. Binary checkpoint is a superset (mmap cache + medoid state). `--restart` uses binary. Old `--checkpoint` flag uses CSV. Document both.

### Binary header alignment
- Pad header to **32 bytes** (not 24). Doubles start at byte 32 = 8-byte aligned. Safe on ARM.

### Stale cache detection
- Store hash of input filenames + sizes + modification times in the binary header.
- On `--restart`, compare. Warn if stale, don't silently use wrong cache.

### CRC32 scope
- CRC covers header only (20 bytes), not data. Data integrity is OS responsibility (page cache + fsync). Computing CRC over multi-GB mmap is expensive and pointless for a cache file.

### CLARA sample size formula
- `clara_sample_size = max(40 + 2*k, min(N, sqrt(N) * k))` for N > 50K
- Number of CLARA iterations: default 5 (existing `CLARAOptions::n_samples = 5`)

## What NOT to Do

- Don't add HDF5 C++ dependency yet — Python has it, C++ can wait
- Don't abstract DataSource interface yet — YAGNI until streaming CLARA is proven
- Don't change the Method enum — FastPAM/FastCLARA work fine as standalone functions
- Don't CRC the entire data region — OS handles data integrity for mmap'd files
- Don't allocate full mmap matrix when auto-selecting CLARA — CLARA doesn't need it
