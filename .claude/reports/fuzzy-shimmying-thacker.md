# Phase 5: Parquet-Native Access + float32 Support + Benchmark

## Context

**The real question:** We have 100GB Parquet (battery data, Zstd, 21x compression). Converting to Arrow IPC inflates to ~2TB on disk. With 128GB RAM and NVMe SSD, is inflation worth it — or can we work directly from Parquet?

**Key numbers (researched):**

| Operation | Speed | For 20MB series |
| --- | --- | --- |
| NVMe sequential read | ~7 GB/s | 2.9 ms |
| Zstd decompress (1 core) | ~1.5 GB/s output | 13.3 ms |
| Zstd decompress (8 cores) | ~12 GB/s output | 1.7 ms |
| Read 1MB compressed + decompress | — | ~13.5 ms (1 core) |

**For full dataset scans:** Compressed wins at high compression ratios with multi-core. Reading 100GB compressed + 8-core decompress = ~17s vs reading 2TB uncompressed = ~286s.

**For random single-series access:** Uncompressed wins (3ms vs 13ms). But CLARA only does this for ~1000 series per subsample — total ~13 seconds compressed vs ~3 seconds uncompressed. Negligible vs DTW computation time (minutes-hours).

**Conclusion:** Arrow IPC inflation (100GB → 2TB) is **not justified** for this workload. Reading Parquet directly with row-group-level access is fast enough, saves 1.9TB of disk, and 100GB compressed fits comfortably in 128GB RAM page cache after first scan.

**Additional insight from user:** Real time series are typically ≤10K samples. At float32, a 10K-sample series = 40KB. 100M series × 40KB = 4TB raw, 200GB Parquet. CLARA only needs subsamples in RAM at a time.

## What to build

### 1. Benchmark: Parquet vs Arrow IPC vs heap on real battery data

Using the 30 Parquet files in `data/test_parquet/` (328MB, ~25M rows), measure:

- **Load time:** Parquet read-to-vectors vs Arrow IPC mmap vs .dtws mmap
- **Per-series DTW cost:** Read series from each source, compute DTW, measure total
- **CLARA simulation:** Random subsample + full assignment scan
- **float32 vs float64:** DTW accuracy and speed comparison

This gives us real numbers to make the architecture decision.

### 2. Optional C++ Parquet reading via Arrow

Add `DTWC_ENABLE_PARQUET` CMake option. When enabled, the CLI can read `.parquet` files directly — no conversion step needed. Arrow's `parquet::arrow::FileReader` with `MemoryMappedFile` handles row-group-level access.

Since we already have `DTWC_ENABLE_ARROW`, just add `ARROW_PARQUET=ON` when Parquet is requested. The combined Arrow IPC + Parquet static library is ~9-14MB.

### 3. float32 support via CMake option

DTW functions are already templated on `data_t`. Add `DTWC_PRECISION` CMake option (`float` or `double`). This halves memory for time series data, letting us fit 2x more series in RAM. For battery voltage (6-digit precision), float32 is more than adequate.

## Implementation Plan

### Step 1: Python benchmark script

**File:** NEW `benchmarks/bench_parquet_access.py`

Benchmark with the real 328MB battery data (`data/test_parquet/`):

```
Test A: Load all 30 cells' Voltage into memory
  - From Parquet (pq.read_table per file)
  - From Arrow IPC (single converted file)
  - From .dtws (single converted file)
  
Test B: Random access 1000 series (simulating CLARA subsample)
  - Parquet: read specific row groups
  - Arrow IPC: mmap + offset lookup
  - Heap vectors: direct index

Test C: Sequential scan all 25M rows (simulating CLARA assignment)
  - Parquet: iter_batches with parallel decompression
  - Arrow IPC: mmap sequential read
  - Heap vectors: direct iteration

Test D: DTW computation on pairs (the REAL bottleneck)
  - Time DTW on 1000 random pairs from each source
  - Show that I/O strategy is irrelevant vs DTW cost

Test E: float32 vs float64
  - DTW accuracy: max relative error on battery voltage data
  - DTW speed: float32 vs float64 throughput
  - Memory: 2x saving confirmation
```

### Step 2: C++ Parquet reader (optional dep)

**File:** [cmake/Dependencies.cmake](cmake/Dependencies.cmake)

Add Parquet detection when `DTWC_ENABLE_PARQUET=ON`:
```cmake
if(DTWC_ENABLE_PARQUET)
  set(DTWC_ENABLE_ARROW ON)  # Parquet requires Arrow
  # find_package(Parquet) — comes with Arrow installation
endif()
```

**File:** NEW [dtwc/io/parquet_reader.hpp](dtwc/io/parquet_reader.hpp)

```cpp
#ifdef DTWC_HAS_PARQUET
class ParquetDataSource {
  // Read Parquet file, optionally selecting columns
  // Supports row-group-level access for CLARA subsampling
  static ParquetDataSource open(path, column_name);
  
  span<const data_t> series(size_t i) const;  // decompresses on demand
  size_t size() const;
  // ...
};
#endif
```

Two access modes:
- **Bulk load:** Read entire column into `vector<vector<data_t>>` (for datasets that fit in RAM)
- **Row-group streaming:** Read row groups on demand (for datasets larger than RAM)

**File:** [dtwc/dtwc_cl.cpp](dtwc/dtwc_cl.cpp)

Add `.parquet` extension detection:
```cpp
else if (input_ext == ".parquet" || input_ext == ".pq") {
  #ifdef DTWC_HAS_PARQUET
    // Read Parquet directly
  #else
    std::cerr << "Parquet support requires -DDTWC_ENABLE_PARQUET=ON\n";
  #endif
}
```

CLI flags:

- `--column <name>` — which column to use as time series data (default: auto-detect numeric)
- `--precision float32|float64` — runtime precision selection (default: float32)
- `--ram-limit <size>` — max RAM for series data, e.g. `2G`, `500M`, `128G`. Controls chunked processing strategy. Default: 80% of system RAM.

### Step 3: Runtime float32/float64 precision selection

**Not a compile-time option.** DTW functions are already templated. Both float32 and float64 codepaths are always compiled and available. The choice is made at runtime per-Problem.

**File:** [dtwc/settings.hpp](dtwc/settings.hpp)

Keep `data_t = double` as the distance/internal type. Add a separate `series_t` concept:

```cpp
enum class Precision { Float32, Float64 };
```

**File:** [dtwc/Problem.hpp](dtwc/Problem.hpp)

```cpp
Precision precision{Precision::Float32};  // default float32 (user can request float64)
```

When loading data, the Problem stores series as either `vector<float>` or `vector<double>` based on `precision`. DTW dispatch calls the appropriate template instantiation. Distance matrix always stays `double` (distances need full precision).

**Default: float32** unless user specifies `--precision float64`. For battery voltage (6.615V with 3 decimal places), float32 gives ~7 significant digits — more than enough.

### Step 4: RAM-aware chunked processing

When dataset exceeds `--ram-limit`, process in chunks that minimize re-reads:

**CLARA with RAM budget:**

1. **Subsample phase:** Only the subsample (~1000 series) needs to be in RAM. Load just those series from Parquet row groups. Tiny RAM cost — always fits.

2. **Assignment phase (all N × k distances):** Stream through data in chunks sized to RAM limit. For each chunk:
   - Load chunk of series from Parquet (decompress once)
   - Compute distances from chunk's series to all k medoids (medoids always in RAM)
   - Write assignments, discard chunk, load next

3. **Smart ordering heuristic:** Sort series indices by Parquet row group so consecutive series come from the same row group. This means each row group is decompressed once per pass, not randomly re-accessed. Turns random access into sequential scan.

4. **Medoid caching:** The k medoid series are accessed N times each. Always keep them pinned in RAM regardless of budget.

This means CLARA can cluster datasets of **any size** with bounded RAM. The only cost is streaming through the Parquet file once per CLARA iteration (5-10 passes typically).

### Step 5: Keep both Arrow IPC and Parquet paths

Both formats serve different purposes:

- **Parquet** = default input. No conversion needed. Compact storage. Decompress on demand.
- **Arrow IPC** = optional pre-converted cache for repeated runs. Zero-copy mmap. Best for interactive exploration where you re-cluster the same data many times with different k/band/variant.

The user chooses:

```
dtwc_cl --input data.parquet --column Voltage -k 5          # Direct Parquet (default)
dtwc_cl --input data.arrow -k 5                             # Pre-converted Arrow IPC
dtwc-convert data.parquet -o data.arrow --column Voltage     # One-time conversion
```

## File Changes Summary

| File | Change |
| --- | --- |
| `benchmarks/bench_parquet_access.py` | NEW — comprehensive benchmark |
| `dtwc/io/parquet_reader.hpp` | NEW — C++ Parquet reader (guarded) |
| `cmake/Dependencies.cmake` | Add Parquet detection |
| `dtwc/CMakeLists.txt` | Link Parquet lib, add compile defs |
| `CMakeLists.txt` | Add `DTWC_ENABLE_PARQUET`, `DTWC_USE_FLOAT32` options |
| `dtwc/settings.hpp` | Conditional `data_t = float` |
| `dtwc/dtwc_cl.cpp` | `.parquet` input, `--column`, `--precision` flags |

## Verification

1. Run `bench_parquet_access.py` with `data/test_parquet/` — get real numbers
2. Build with `-DDTWC_ENABLE_PARQUET=ON` (requires Arrow+Parquet installed)
3. Build with `-DDTWC_USE_FLOAT32=ON` — all 67 tests pass
4. CLI: `dtwc_cl --input data.parquet --column Voltage -k 5 --method clara`
5. Compare clustering results: float32 vs float64 on same data

## Key Decision Points (benchmark-driven)

The benchmark will answer with real numbers:

- **Q1:** What fraction of total CLARA time is I/O vs DTW? → Quantifies whether I/O strategy matters at all
- **Q2:** float32 vs float64 DTW: max relative error on battery voltage? → Validates float32 as default
- **Q3:** Parquet row-group streaming throughput vs bulk load? → Sizes the RAM-limit tradeoff
- **Q4:** How much does smart row-group ordering help vs random access? → Validates the sorting heuristic
