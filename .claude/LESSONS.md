# DTWC++ Lessons Learned

Critical knowledge to avoid repeating mistakes.

---

## Mathematical / Theoretical

### DTW is NOT a metric
- Violates triangle inequality. MIP integrality gap bounds (assume metric D) don't formally apply.
- In practice the gap is small. References: Marteau (2009), Jain (2018).

### DTW-AROW ≠ simple zero-cost DTW
- DTW-AROW constrains each missing value to one-to-one diagonal alignment.
- Simple zero-cost is less restrictive and underestimates distances more.

## C++ Performance

### DTW is latency-bound, not memory-bound
- 10-cycle recurrence chain, uses only 3% of L1 bandwidth.
- Multi-pair SIMD gives max ~1.29x due to scatter/gather overhead.
- MSVC auto-vectorizes LB_Keogh/z_normalize to SSE2; explicit Highway adds dispatch overhead.

### SIMD: branchless scalar matches explicit SIMD for DTW
- Highway SIMD was tried and removed (2026-04-04). Branchless scalar lb_keogh matched Highway perf.
- DTW is latency-bound (10-cycle recurrence), not throughput-bound — explicit SIMD gives max ~1.29x.
- **Lesson**: Any batch SIMD kernel loading from multiple pointers must pre-transpose to SoA layout;
  scatter/gather in a recurrence inner loop kills perf.

### tri_index precomputed row offsets: no benefit (benchmarked 2026-04-08)
- Replaced `i*(i+1)/2 + j` with `row_offsets_[i] + j` lookup table in DenseDistanceMatrix.
- Also tried `schedule(guided)` instead of `schedule(dynamic,1)` in fillDistanceMatrix.
- **Result**: random_get 500K lookups: 2.61ms baseline → 2.75ms optimized (+5% slower).
  Fill benchmarks: within noise (±10%), no consistent improvement.
- **Why**: Modern CPUs compute `i*(i+1)/2` in 1-2 cycles (imul+shr). Table lookup adds
  an L1 cache load (~4 cycles) plus the table pollutes L1. The conditional swap `if(i<j)`
  already compiles to branchless cmov. The multiplication is faster than the indirection.
- **Lesson**: Don't replace simple integer arithmetic with lookup tables on modern CPUs.
  imul is cheap; L1 loads are not free. Profile before "optimizing".

### std::min with initializer_list is catastrophically slow
- `std::min({a,b,c})` creates a temporary on every call. 2.5-3x speedup from `std::min(a, std::min(b,c))`.

### Lambda capture-by-value creates stale parameter bugs
- `rebind_dtw_fn()` captured `[b=band]` — froze band at creation time.
- Fix: capture `[this]` and read `this->band` at invocation time.

### LB_Keogh is only valid for L1 and squared L2
- NOT valid for cosine or Huber metrics. Wrong pruning → wrong clustering results.

### WDTW/ADTW/DDTW/Soft-DTW: separate functions, not metric swaps
- WDTW weights depend on (i,j) position. ADTW penalizes non-diagonal steps.
- DDTW is a preprocessing step. Soft-DTW replaces min with softmin.

### NaN is the ONLY safe sentinel for distance matrix uncomputed entries
- Soft-DTW returns negative distances (softmin with gamma). Any fixed negative sentinel (-1, -10000) can collide.
- Previous session used -1 sentinel → broke Soft-DTW caching. Changed to NaN (2026-04-07 bugfix).
- In binary files: NaN is just 8 bytes (IEEE 754 bit pattern), trivial to write via pointer assignment.
- In CSV: write empty field (`,,`), load as NaN. Already implemented in `matrix_io.hpp`.
- Don't use NaN payload bits for status encoding — compilers may canonicalize, SIMD strips them. Use separate status array if needed.

### Don't replace simple integer arithmetic with lookup tables (benchmarked 2026-04-08)
- `tri_index: i*(i+1)/2 + j` is 1-2 cycles (imul). Precomputed row-offset table adds L1 load (~4 cycles) + cache pollution.
- Benchmarked: table version was 5% SLOWER on 500K random lookups. Modern CPUs do integer multiply faster than memory indirection.

### llfio > mio for memory-mapped I/O (evaluated 2026-04-08)
- mio (MIT, header-only): clean 3-line API but dormant since 2020, no file locking, no pre-allocation.
- llfio (Apache 2.0, header-only option): active (Jan 2025), has file locking, `posix_fallocate`, sparse files, production-proven (SEC MIDAS).
- llfio solves all P0/P1 issues from adversarial review (pre-allocation, file locking) without custom platform code.
- Both have zero external dependencies. llfio requires C++17 (which we already require).

## C++ Implementation

### NaN for missing data
- Use `quiet_NaN()`. Check via `std::isnan()` — safe because `-ffinite-math-only` is NOT in the build flags.
- `is_missing()` in `missing_utils.hpp` is a thin wrapper over `std::isnan()`. Use it in missing-data paths.
- `is_missing()` MUST happen BEFORE calling the distance function.

### HiGHS vs Gurobi indexing
- HiGHS: row-major `A[i,j]` at `i*Nb + j`. Gurobi: column-major at `i + j*Nb`.
- Both: diagonal `A[i,i]` at `i*(Nb+1)`.

## Cross-Language Bindings

### MEX longjmp-safe pattern (CRITICAL)
- `mexErrMsgIdAndTxt` calls `longjmp` — skips C++ destructors.
- Fix: catch exception → exit scope → then call `mexErrMsgIdAndTxt`.

### mexLock prevents shutdown crashes
- Call `mexLock()` on first MEX entry. Prevents MATLAB from unloading DLL while handles exist.
- `mexAtExit` callback must drain HandleManager before DLL teardown.

### MATLAB + MSVC OpenMP: exit segfault in `-batch` mode
- OpenMP runtime teardown conflicts with MATLAB DLL unload order.
- All functionality works; segfault is after output completes. Check output, not exit code.

### nanobind over pybind11
- Native `nb::ndarray<T, nb::device::cuda>`, stable ABI, 5-10x smaller binaries.
- GIL release essential for any method >10ms.

### MATLAB handles as double lose precision above 2^53
- Use uint64 in MATLAB class, but `mxGetScalar` returns double. Counter never reaches 2^53 in practice.

## Build System

### CUDA multi-version on Windows
- MSBuild picks latest `.targets`. Generate `Directory.Build.props` with `<CudaToolkitCustomDir>`.
- Toolkit version must not exceed driver's supported version.

### MSVC flags leak into nvcc
- Use `$<$<COMPILE_LANGUAGE:C,CXX>:...>` generator expressions on all MSVC-specific options.

### OpenMP find_package fails with CUDA enabled
- Fix: `find_package(OpenMP COMPONENTS CXX)`.

### find_package variables don't propagate from CMake functions
- Check `TARGET MPI::MPI_CXX` instead of `MPI_CXX_FOUND`.

## Research Process

### Always verify citations
- Author names, venues, volume numbers can be hallucinated. Use a separate review agent.
