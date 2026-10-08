# 2026-10-09 — dtype at load, and block-wise memory access (approved by Volkan 2026-10-09; DECISIONS §3)

Rulings it answers (DECISIONS §3, 2026-10-08, verbatim there): double stays the default, float32 "easily changable
like DataLoader(d_type = f32)", the distance matrix double "most of the time"; algorithms access memory "block by
block like a well-written matrix multiplication algorithm ... read enough time series that fit in the ram, then do
the parallel thing, then load the other block". Evidence: two read-only surveys (2026-10-09). [verified] marks a line
the orchestrator opened; the rest is the surveys' reading of the code; [inferred] marks an estimate, not a measurement.

## Today

**Precision.** Double everywhere by default; float32 exists only in pieces.
- C++ `Data` holds float32 (`Data.hpp:33-37`), but no reader makes it: `DataLoader::load()` calls the double readers
  (`DataLoader.hpp:128-137`) [verified]; Parquet, Arrow IPC and the C-Data ingest widen to double
  (`read_arrow.cpp:109-113`, `arrow_c_data.cpp:39-108`); only FastCLARA's Parquet stream keeps float.
- The CLI's `--dtype float32` loads double, then converts (`cli/run.cpp:300-303`) [verified]: the peak holds both
  copies, and a double *view* stays double without a word (G2): a silent fallback, which the runbook forbids.
- Python: only `Data.from_float32` and `Problem.set_data`; `load`, `Dataset` and `Config` have no dtype. MATLAB: no
  float32 path (`load.m:38-44`, `Problem.m:77-84`; the MEX requires double). Neither language has a `DataLoader`.
- The matrix is double end to end: heap or mapped `.dtwm` v4 (no width field); 23 `dist_by_ind` reads in 9 files.

**Memory access.** No sweep is blocked at the RAM level.
- The fill hands each thread dynamic chunks of N/(8·threads) consecutive rows (`base/parallelisation.hpp:59-63`)
  [verified]. Row i writes `tri_index(j, i) = j(j+1)/2 + i` for every later j: one entry per packed row, down a column
  of the matrix, so one cache line per entry and one 4 KB page per entry once j > 511. Measured at N 8000, L 40–60,
  24 threads (`baselines/2026-10-08-rl-ragged-lanes.md`, ETW): about 1.2 last-level-cache misses per entry and about
  7 L3 reads per pair (each column's series fetched again).
- Checkpoints: a heap save rewrites the whole matrix every 100 rows (`checkpoint.hpp:40-43`) [verified], O(N³/100)
  bytes in all (about 2.6 TB at N 40k [inferred]); a mapped save flushes the whole map. A new matrix is NaN-filled by
  one thread (`distance_matrix.cpp:165`) [verified]: serial first touch, so on a two-socket node every page sits on
  one socket.
- No series are ever mapped (`Problem.hpp:278` and `:306` name an "mmap series store" that does not exist); Arrow IPC
  is mapped, then copied. FastCLARA's Parquet stream decodes serially, one full pass per sample (5 samples).
- FastPAM's SWAP is serial on purpose (`.claude/MAP.md` §7) [verified]: N² reads a sweep, half of them one per packed
  row.

**How mmap works.** Mapping a file reserves addresses only. The first touch of a 4 KB page faults it in, from the
page cache (about a microsecond) or from disk (tens of microseconds on NVMe), with readahead after sequential faults.
Pages stay while RAM allows; under pressure the OS evicts cold pages, writing dirty ones back first. So touching a
block is loading it, and the loop's order is the I/O schedule: a row sweep over a matrix larger than RAM keeps every
page half-written and thrashes, while a blocked sweep finishes each page before it moves on. Hints exist (`madvise`,
Windows `PrefetchVirtualMemory`, llfio's `map_handle::prefetch`) to read the next block while this one computes; none
is used today.

## Units (proposed order)

1. **DT — `dtype` where data is loaded, in every language.** The word is the CLI's and NumPy's (`dtype`; MATLAB
   `'Dtype'`); its values are the C++ name table (`float32`/`f32`/`fp32`/`float`, `float64`/`f64`/`fp64`/`double`);
   the default is float64.
   - C++: `DataLoader::dtype(core::Precision)` (an addition to a frozen class); the text, Parquet and Arrow readers
     produce float32 directly; Tier-1 `dtwc::load(…, dtype)` and `Dataset::dtype()`, honoured by `cluster()`; the CLI
     reads straight into float32; G2 fixed (a double view under float32 is converted, never kept); the verbose memory
     line reports the stored width.
   - Python: `dtwcpp.load(…, dtype=…)`, `Dataset.dtype`, `cluster(path, k, dtype=…)`; pyarrow casts to float32; one
     precision-aware series accessor replaces the `.p_vec` reads.
   - MATLAB: `dtwc.load(src, 'Dtype', 'float32')` and `Dataset.Dtype`; the MEX returns `single`.
   - Each value is parsed as double and narrowed, as the CLI does today, so a float32 run equals today's
     `--dtype float32`. Explicit only: float32 input data does not choose float32 storage.
   - Gates: float64 output digit-identical (CLI 156 + 65 files, pytest, matlab_suite); float32 output equal to today's
     `--dtype float32`. Band to register: the peak memory of a float32 Parquet load at most 0.6× today's.
2. **BF — the blocked CPU fill.** Slabs of packed rows R = [R0, R1), each one contiguous byte range of the matrix;
   tiles (R, C) of the pairs (r, c) with c < r. Inside a tile x = series c (the smaller index, as today) and the lanes
   take R's series longest first, so every distance stays bit-identical; each c writes one slot in each of 16 packed
   rows and the next c writes beside it. In memory: one dynamic loop over all tiles, no barrier. Mapped: slab by
   slab, prefetching the next slab and flushing only the finished range. Checkpoints per slab group write the
   finished range in place (no whole-matrix rewrite). The NaN prefill runs in parallel (first touch where the fill
   runs). Tile width: a constant from L2 (about 512 series at f64, L 100 [inferred]); slab bytes: from the memory
   budget (below).
   - Expected [inferred]: in memory at N 8000, L 40–60, misses per entry 1.2 → about 0.13 but wall time at most
     about 5 % faster, because the fill is compute-bound (RL's 21 % fewer misses bought 1.5 % fewer cycles). The gain
     is for a mapped matrix larger than RAM (one slab of about 0.8 GB open at a time instead of every page, each page
     written once) and for checkpoints.
   - Draft bands, registered with the unit's kit before any run: misses per entry ≤ 0.4 at N 8000, L 40–60; every
     in-memory shape ≥ 0.98× (L 100 and L 8000, ragged and equal, 1 and 24 threads; 24-thread controls over 20
     alternating process pairs); a mapped matrix 2–4× the memory the run may use ≥ 2× faster than today's sweep; a
     checkpointed fill writes at most 3× the matrix's bytes.
2a. **AL — CLARA's assignment through the lanes** (added 2026-10-09 by Volkan's ruling, after DT): each point
   against W medoids a call (`core::dtw_kernel_lanes`, the fill's block function), the per-pair kernel only where the
   fill would use it (a band with unequal lengths; cl on unequal lengths). Distances as the fill's lanes give them, so
   labels, medoids and costs stay as they are. Measured today: the CPU assignment at 17.9 Gcell/s on 24 threads
   (`baselines/2026-10-01-gc-clara-gpu-assignment.md`) against the lanes fill's 105–131 Gcell/s; about 6× expected
   [inferred].
3. **CS — CLARA reads its data once.** All samples are drawn first (the RNG feeds only the sampling, so the result is
   identical), their union is read once, one assignment pass runs against every sample's medoids, and decoding
   overlaps computing. Draft band: decode passes 6 → 2, and a large Parquet run ≥ 1.5× faster.
4. **SW — a blocked SWAP** (needs Volkan: MAP §7). Candidates are evaluated in parallel blocks over contiguous runs,
   and a block is redone after an accepted swap, so the swaps accepted are exactly the sequential sweep's. With it,
   the free fixes: Dunn's loop order, Calinski–Harabasz's sums, silhouette's column reads.
5. **FM — an opt-in float32 matrix** (`matrix_dtype`, double by default). One `DistanceMatrix` whose value width is
   chosen at run time (reads widen, writes narrow, NaN still marks "not computed"); `.dtwm` v5 records the width. It
   buys capacity (N 100k: 40 → 20 GB), not swap speed (a strided read costs a cache line at either width
   [inferred]); ties within one float step can change medoids, so it is documented as changing results.
6. **MS — mapped series** (needs Volkan): an uncompressed Arrow IPC file mapped into `Data`'s view spans, with no
   custom format. Measure first whether CLARA's time at 100M series goes to decoding or to the disk.

## Memory budget

One key, `ram_limit` (today `--ram-limit`, Parquet only), automatic by default: what the process may use (available
memory, or the cgroup limit a SLURM job sets; `GlobalMemoryStatusEx` on Windows). It sizes the slabs and CLARA's
batches and decides when the matrix is mapped, by its bytes, replacing the N-based `--mmap-threshold`. All four memory
and precision flags (`--dtype`, `--ram-limit`, `--mmap-threshold`, `--checkpoint-interval`) are 2.0-born: v1.0.0's
CLI had none of them.

## Questions for Volkan

Asked with the question tool on 2026-10-09; every answer the recommended one (DECISIONS §3): the order DT, BF, CS,
SW, FM; SW yes, with the accepted swaps identical to the sequential sweep's; MS measured first, after CS; one memory
key, automatic.
