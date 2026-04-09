# Plan: RAM-Limit Chunked CLARA + Float32 Views + Docs + Cleanup

## Context

Phase 4 (data access, I/O, float32) is complete. The `--ram-limit` CLI flag is parsed but discarded with `(void)ram_limit`. Float32 and view-mode don't combine (CLARA subsampling creates `span<const data_t>` views, ignoring float32 data). CHANGELOG/README are stale. Older handoff files should be cleaned up.

Target: 100M series × 8K samples on SLURM with 2TB RAM — needs chunked streaming.

---

## Step 1: Float32 View-Mode in Data

**Files:** [Data.hpp](dtwc/Data.hpp), [fast_clara.cpp](dtwc/algorithms/fast_clara.cpp)

### Data.hpp changes

Add alongside existing `p_spans_` (line 135):
```cpp
std::vector<std::span<const float>> p_spans_f32_;
```

Add float32 view-mode constructor:
```cpp
Data(std::vector<std::span<const float>> &&spans,
     std::vector<std::string_view> &&name_views, size_t ndim_)
  : ndim{ndim_}, precision{core::Precision::Float32},
    p_spans_f32_(std::move(spans)), p_name_views_(std::move(name_views)), is_view_(true)
{ /* validate */ }
```

Update accessors:
- `size()`: `if (is_view_) return is_f32() ? p_spans_f32_.size() : p_spans_.size();`
- `series_f32(i)`: add `if (is_view_) return p_spans_f32_[i];` at top
- `series_flat_size(i)`: `if (is_view_) return is_f32() ? p_spans_f32_[i].size() : p_spans_[i].size();`

### fast_clara.cpp changes (lines 142-159)

Branch on `prob.data.is_f32()` when building subsample spans:
```cpp
if (prob.data.is_f32()) {
    std::vector<std::span<const float>> sub_spans;
    sub_spans.reserve(sample_size);
    for (int idx : sample_indices)
        sub_spans.push_back(prob.data.series_f32(idx));
    sub_prob.set_view_data(Data(std::move(sub_spans), std::move(sub_names), prob.data.ndim));
} else {
    // existing float64 path (unchanged)
}
```

### Tests
- Unit test: create float32 Data, build view from it, verify `size()`, `series_f32()`, `is_view()`, `is_f32()`
- Integration test: run `fast_clara` on float32 data, verify valid result

---

## Step 2: ParquetChunkReader

**New file:** [dtwc/io/parquet_chunk_reader.hpp](dtwc/io/parquet_chunk_reader.hpp)

Guarded by `#ifdef DTWC_HAS_PARQUET`. Reuses existing Arrow/Parquet deps.

```cpp
class ParquetChunkReader {
public:
    ParquetChunkReader(const fs::path &path, const std::string &col_name = "");

    int num_row_groups() const;
    int64_t total_rows() const;
    size_t estimated_bytes_per_series() const; // from metadata, no data read

    Data read_row_group(int rg) const;         // single row group → Data
    Data read_row_groups(int rg_start, int count) const; // batch
    Data read_rows(const std::vector<int64_t> &indices) const; // sparse access for subsampling
};
```

Implementation uses `reader->RowGroup(rg)->ReadTable({col_idx}, &table)` — Arrow's native chunking. `read_rows()` scans row-group metadata to find which groups contain needed indices, reads only those groups, filters.

### Tests
- Create small Parquet fixture (100 series, 5 row groups)
- Verify row-group-by-row-group reading matches `load_parquet_file()` result
- Verify `read_rows()` returns correct subset

---

## Step 3: RAM-Aware Chunked CLARA Assignment

**Files:** [fast_clara.hpp](dtwc/algorithms/fast_clara.hpp), [fast_clara.cpp](dtwc/algorithms/fast_clara.cpp), [Problem.hpp](dtwc/Problem.hpp)

### CLARAOptions additions
```cpp
size_t ram_limit_bytes = 0;          ///< 0 = no limit
std::filesystem::path parquet_path;  ///< empty = data already in RAM
std::string parquet_column;
```

### Problem.hpp — expose DTW functions
```cpp
const dtw_fn_t& dtw_function() const { return dtw_fn_; }
const dtw_fn_f32_t& dtw_function_f32() const { return dtw_fn_f32_; }
```

Needed because chunked assignment computes DTW directly on spans without a distance matrix.

### Chunked assignment function (new, in fast_clara.cpp anonymous namespace)

```cpp
double assign_all_points_chunked(
    const dtw_fn_t &dtw_fn,              // or f32 variant
    const std::vector<Data> &medoid_data, // k medoid series
    std::vector<int> &labels,
    const ParquetChunkReader &reader,
    size_t ram_budget,
    int band)
```

Algorithm:
1. Compute `rg_per_batch = max(1, chunk_budget / (bytes_per_series * rows_per_rg))`
2. For each batch of row groups:
   - `Data chunk = reader.read_row_groups(rg_start, batch_size)`
   - For each point in chunk, compute DTW to each medoid, record best label
   - Accumulate total_cost; chunk goes out of scope → memory freed
3. Return total_cost, labels filled for all N points

### Revised CLARA main loop (chunked mode)

When `opts.ram_limit_bytes > 0 && !opts.parquet_path.empty()`:
1. Open `ParquetChunkReader` — get `total_rows()` from metadata (no data load)
2. For each CLARA sample:
   - Draw sample indices from `[0, total_rows)`
   - Load sample via `reader.read_rows(sample_indices)` — small, fits in RAM
   - Run FastPAM on subsample → get medoid global indices
   - Load medoid series via `reader.read_rows(medoid_indices)` — tiny
   - Run `assign_all_points_chunked()` streaming row groups within budget
3. Track best result across samples

**Key invariant**: The main `prob` object holds settings only (band, variant, etc.) — NOT the full dataset. Data is streamed from Parquet.

### Edge cases
- Non-Parquet input with `--ram-limit`: print warning "RAM limit requires Parquet input for streaming", fall through to in-RAM path
- `ram_limit > total_data_bytes`: skip chunked mode, use normal path
- Subsample always fits in RAM (by design: max ~10k+100 series)

### Tests
- Create 100-series Parquet file, run CLARA with `ram_limit=1M` to force chunking
- Verify result matches in-RAM CLARA (same medoids, cost within tolerance)

---

## Step 4: CLI Wiring

**File:** [dtwc_cl.cpp](dtwc/dtwc_cl.cpp)

Replace `(void)ram_limit;` (line 509) — wire into `clara_opts` (around line 688):

```cpp
clara_opts.ram_limit_bytes = ram_limit;
if (ram_limit > 0 && is_parquet_input) {
    clara_opts.parquet_path = input_file;
    clara_opts.parquet_column = parquet_column;
}
```

Add warning for non-Parquet + ram_limit:
```cpp
if (ram_limit > 0 && !is_parquet_input)
    std::cerr << "Warning: --ram-limit only effective with Parquet input\n";
```

---

## Step 5: CHANGELOG.md + README.md

**File:** [CHANGELOG.md](CHANGELOG.md)

Add to `## Unreleased` section:
- **Added**: Float32 view-mode in Data class — CLARA subsampling now works with float32 data
- **Added**: `ParquetChunkReader` for row-group streaming (chunked I/O)
- **Added**: RAM-aware chunked CLARA assignment via `--ram-limit` with Parquet streaming
- **Added**: Arrow IPC zero-copy reader, Parquet reader, `dtwc-convert` Python tool
- **Added**: Runtime float32 precision (`--precision float32`)
- **Added**: CLARA zero-copy span views (48x subsample speedup)
- **Changed**: `Data::series(i)` returns `span<const data_t>` (uniform accessor for heap/mmap/view)

**File:** [README.md](README.md)

Update feature list:
- Add "Arrow IPC (zero-copy mmap) and Parquet" to I/O formats
- Add "Runtime float32 precision (2x memory saving)" 
- Add "RAM-aware streaming CLARA for datasets exceeding memory"
- Update CMake options table with `DTWC_ENABLE_ARROW`

---

## Step 6: Arrow CPM Build on Linux/SLURM

This is a verification step, not code change. The CPM build logic in [cmake/Dependencies.cmake](cmake/Dependencies.cmake) (lines 157-239) already handles Linux correctly:
- Static Arrow+Parquet build from source via CPM
- `ARROW_BUILD_STATIC ON`, `ARROW_DEPENDENCY_SOURCE BUNDLED`
- No Windows+Clang flag quoting issue on Linux

**Verification**: SSH to SLURM, build with `-DDTWC_ENABLE_ARROW=ON`, run tests. Report result. If `module load arrow` available, test find_package path too.

---

## Step 7: Delete Older Handoffs

**Delete these 6 files** (keep only the 2 newest):
- `.claude/summaries/2026-04-07-cleanup.md`
- `.claude/summaries/2026-04-07-stress-test-session.md`
- `.claude/summaries/2026-04-08-build-env-and-mmap-plan.md`
- `.claude/summaries/2026-04-08-mmap-implementation.md`
- `.claude/summaries/2026-04-08-mmap-full-session.md`
- `.claude/summaries/2026-04-08-final-handoff.md`

**Keep:**
- `.claude/summaries/2026-04-08-phase4-data-access.md`
- `.claude/summaries/handoff-2026-04-08-phase4.md`

---

## Execution Order

1. **Step 7** (delete old handoffs) — trivial, no deps
2. **Step 1** (float32 view-mode) — self-contained, no external deps
3. **Step 2** (ParquetChunkReader) — needs Arrow but independent of Step 1
4. **Step 3** (chunked CLARA) — depends on Steps 1+2
5. **Step 4** (CLI wiring) — depends on Step 3
6. **Step 5** (CHANGELOG/README) — after all code changes
7. **Step 6** (Linux/SLURM test) — after build is ready

Steps 1 and 2 can be parallelized. Steps 1+7 can be parallelized.

## Verification

- All existing 67 tests pass
- New tests for float32 view-mode, ParquetChunkReader, chunked CLARA
- CLI test: `dtwc --ram-limit 1M --precision float32 input.parquet -k 3 --method clara`
- Build: `cmake -DDTWC_ENABLE_ARROW=ON` on Linux
