# Session: Full Mmap Implementation + Benchmarks (2026-04-08)

## What was done (15 commits on branch Claude)

### Steps 1-2: Count-Before-Load + Auto Method Selection
- `DataLoader::count()`, `--method auto` (default), CLARA auto-scaling

### Steps 3-5: Mmap Distance Matrix
- llfio integrated via CPM → made non-optional (core dependency)
- `MmapDistanceMatrix` with 32-byte binary header, llfio backend
- `std::variant<DenseDistanceMatrix, MmapDistanceMatrix>` in Problem
- Binary checkpoint (`save_binary_checkpoint`/`load_binary_checkpoint`)
- CLI: `--restart`, `--mmap-threshold`

### Benchmarks (N=5000, ~95MB matrix)
- Mmap random access: only 5% slower than heap
- Mmap startup: 78x faster than fread (1.4ms vs 109ms)
- CLARA views: 48x faster than vector copies
- `std::visit` overhead: 0.000025% — negligible

### Phase 2: llfio non-optional
- Removed all `#ifdef DTWC_HAS_MMAP` guards (12 files)
- llfio always linked, always available

### MmapDataStore for time series
- `dtwc/core/mmap_data_store.hpp` — contiguous mmap cache for series data
- 64-byte header + offset table + contiguous doubles
- Supports variable-length, multivariate series
- Extracted shared `crc32.hpp` utility
- 6 test cases, all passing

## Test results
- 67/67 pass, 2 CUDA skipped

## Commits
```
77d0745 docs: add mmap benchmark results to LESSONS.md, update plans and handoff
52d1e20 feat: implement MmapDataStore with llfio backend
32061d5 test(RED): add MmapDataStore unit tests
1f91d71 refactor: make llfio non-optional, remove DTWC_ENABLE_MMAP guards
adfa375 feat(bench): add mmap access pattern benchmark suite
55844ef docs: update CHANGELOG, fix --mmap-threshold 0 logic
60e0a3e feat: add binary checkpoint and --restart/--mmap-threshold CLI flags
1134695 test(RED): add binary checkpoint tests
535bdf4 feat: integrate MmapDistanceMatrix into Problem via std::variant
d8dea97 test(RED): add variant distance matrix integration tests
9a6cfe0 feat: implement MmapDistanceMatrix with llfio backend
6457644 Add llfio as optional dependency for memory-mapped distance matrices
75c760d Add DataLoader::count() and --method auto (default) for CLI
c4fb30c Add lessons learned, mmap/auto-select implementation plan, session handoff
```

## What to do next

### IMMEDIATE: C++20 upgrade + std::span integration (Phase 3c)
User approved C++20 upgrade. This unlocks:
1. **Change CMake minimum to C++20** (`CMAKE_CXX_STANDARD 20`)
2. **Change `dtw_fn_` type** from `std::function<data_t(const vector<data_t>&, const vector<data_t>&)>` to `std::function<data_t(std::span<const data_t>, std::span<const data_t>)>`
3. **std::span is constructible from both `vector<double>` and `{double*, size_t}`** — works seamlessly with heap AND mmap backends
4. **Replace `TimeSeriesView`** with `std::span<const double>` (standard version)
5. **Fix CLARA** to pass spans into mmap data instead of copying vectors
6. **Problem::p_vec(i)** can return `std::span<const data_t>` instead of `const vector<data_t>&`

### Then: Phase 4 (Storage Policy)
- `StoragePolicy` enum: Auto/Heap/Mmap
- Auto-select based on N threshold (benchmarks say 5000 is fine)
- Default to mmap for all sizes (benchmarks show negligible overhead)

### Deferred
- Streaming CLARA (Step 6 from original plan)
- Stale cache detection (hash of input filenames in mmap header)
- Checkpoint warm-start in algorithms (--restart loads but doesn't skip re-clustering yet)

## Open bugs
- hierarchical + SoftDTW crashes
- set_if_unset in YAML overrides CLI values
- MV banded DTW silently ignores band
- Pruned distance matrix incompatible with MmapDistanceMatrix (calls `dense_distance_matrix()`)

## Key decisions
- llfio is now a core dependency (not optional)
- Mmap has negligible overhead — safe as default backend
- C++20 approved for std::span (user confirmed)
- Eigen::Map useful for series (SIMD ops), not for packed triangular distance matrix
- Binary formats are internal caches, not user-facing — users keep CSV/HDF5/Parquet
