# DTWC++ Lessons Learned

Critical knowledge to avoid repeating mistakes.

---

## Mathematical

- **DTW is NOT a metric.** Violates triangle inequality. MIP gap bounds don't formally apply.
- **DTW-AROW ≠ zero-cost DTW.** AROW constrains missing values to diagonal alignment.
- **LB_Keogh valid only for L1 and squared L2.** Not cosine or Huber.
- **WDTW/ADTW/DDTW/Soft-DTW are separate functions, not metric swaps.** Each modifies the recurrence differently.

## C++ Performance

- **DTW is latency-bound** (10-cycle recurrence). Only 3% of L1 bandwidth used. SIMD gives max ~1.29x.
- **Branchless scalar matches explicit SIMD for DTW.** Highway was tried and removed. scatter/gather in recurrence kills perf.
- **Don't replace integer arithmetic with lookup tables.** `i*(i+1)/2` = 1-2 cycles (imul). Table lookup = 4 cycles + cache pollution. Benchmarked 5% slower.
- **`std::min({a,b,c})` is catastrophically slow.** Creates temporaries. Use `std::min(a, std::min(b,c))` for 2.5-3x speedup.
- **Lambda capture-by-value creates stale parameter bugs.** Capture `[this]` and read at invocation time, not `[b=band]`.
- **NaN is the ONLY safe sentinel** for distance matrix uncomputed entries. Soft-DTW returns negatives — any fixed sentinel collides.
- **Mmap is safe as default.** Only 5% slower random access, 78x faster open, 48x faster CLARA views vs copy. OS page cache handles everything.
- **llfio > mio.** Active, has file locking, pre-allocation, production-proven. mio dormant since 2020.

## Data Formats & I/O (benchmarked 2026-04-08)

- **DTW dominates I/O by 10-100x.** Don't over-optimize I/O when compute kernel dominates. Just read Parquet directly.
- **Battery voltage compresses 21x** with Parquet Zstd (199.6 MB → 9.7 MB).
- **Arrow IPC = same speed as .dtws** after open (~0.4 us/series, pointer+offset). Open overhead ~4ms (flatbuffers).
- **Arrow IPC inflation not justified** for high-compression data. 100GB Parquet → 2TB Arrow IPC wastes disk. Keep both paths.
- **Parquet is NOT zero-copy** (requires decode+decompress). Arrow IPC IS zero-copy (mmap + pointer cast).
- **HDF5 mmap is a fragile hack** via `H5Dget_offset()`. Don't use.
- **Always use LargeListArray** (int64 offsets) in Arrow IPC. ListArray int32 overflows silently at >2B elements.
- **Zstd decompression vs NVMe:** Single access: uncompressed wins. Full scan with 8 cores and >5x compression: compressed wins.

## Float32 (benchmarked 2026-04-08)

- **Float32 DTW speed = identical to float64.** DTW is latency-bound — narrower data width doesn't help.
- **Float32 benefit is purely memory:** 2x more series in RAM and cache. Fewer page faults for large N.
- **Float32 accuracy is excellent for clustering.** Max relative DTW error: 2.74e-05 (0.003%). Negligible for medoid selection.
- **Default to float32.** Battery voltage (6.615V, 3 decimal places) needs only ~4 significant digits. float32 gives 7.

## C++ Implementation

- **NaN for missing data.** Use `quiet_NaN()`, check via `std::isnan()`. Safe because `-ffinite-math-only` is NOT set.
- **HiGHS: row-major. Gurobi: column-major.** Both: diagonal at `i*(Nb+1)`.
- **View-mode Data: guard `p_vec(i)` and `get_name(i)`.** These access empty vectors in view mode. Assert `!is_view()`.

## Cross-Language Bindings

- **MEX longjmp skips destructors.** Catch exception → exit scope → then call `mexErrMsgIdAndTxt`.
- **mexLock() prevents shutdown crashes.** `mexAtExit` must drain HandleManager before DLL teardown.
- **MATLAB + MSVC OpenMP: exit segfault in `-batch` mode.** Functionality works; segfault after output. Check output, not exit code.
- **nanobind over pybind11.** Stable ABI, 5-10x smaller binaries, native CUDA ndarray. GIL release for >10ms calls.

## HiGHS MIP Solver (IMPORTANT — workaround in place)

- **HiGHS <=1.14.0 `assert(ub_consistent)` fires on warm-start MIP.** The assertion is in `updatePrimalDualIntegral()` — a performance metric tracker, NOT solution correctness. `prev_lb/prev_ub/prev_gap` are documented "Only for checking/debugging" (line 2802). The P-D integral is never used to accept/reject incumbents. Presolve restart rebases bounds with offset arithmetic that introduces roundoff exceeding the 1e-12 tolerance. **Current workaround:** `target_compile_definitions(highs PRIVATE NDEBUG)` in `cmake/Dependencies.cmake` — too blunt (suppresses ALL HiGHS assertions). **Proper fix needed:** patch HiGHS to skip `check_prev_data` after restart, or relax the tolerance in this specific block. File upstream issue at github.com/ERGO-Code/HiGHS.
- **Verified by Codex (GPT-5.4, xhigh reasoning):** Not a solution-correctness bug. The workaround is legitimate short-term.

## Build System

- **CUDA multi-version on Windows:** Generate `Directory.Build.props` with `<CudaToolkitCustomDir>`.
- **MSVC flags leak into nvcc:** Use `$<$<COMPILE_LANGUAGE:C,CXX>:...>` generator expressions.
- **`find_package` vars don't propagate from CMake functions.** Check `TARGET X` instead of `X_FOUND`.
- **PyArrow bundles Arrow C++ libs** but DLL loading on Windows is fragile. Use vcpkg/conda for proper install.

## ARC SLURM Hardware

- **htc GPU compute capabilities (corrected from docs).** The ARC docs list CUDA toolkit version, not compute capability. Actual values: P100=6.0, V100=7.0, RTX8000/TitanRTX=7.5, A100=8.0, RTXA6000=8.6, L40S=8.9, H100/GH200=9.0.
- **Rome (htc-g019) and Broadwell (htc-g045-049) lack AVX-512.** Use `DTWC_ARCH_LEVEL=v3` for portable htc builds. All arc nodes support v4.
- **Grace Hopper (htc-g057) is AArch64.** Needs separate ARM build. CUDA kernel not yet ported.

## Arrow/Parquet

- **Never `static_pointer_cast<DoubleArray>` without checking value type.** Parquet list columns can store Float (32-bit) values. Casting to DoubleArray reinterprets float bits as double — silent data corruption. Always check `values->type_id()` first.
- **Parquet row-group metadata is free.** `num_rows`, `total_uncompressed_size` per row group available without reading data. Use for RAM budgeting.

## Research Process

- **Always verify citations.** Author names, venues, volume numbers can be hallucinated.
