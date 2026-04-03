# DTWC++ Lessons Learned

Critical knowledge to avoid repeating mistakes.

---

## Mathematical / Theoretical

### DTW is NOT a metric
- Violates triangle inequality. MIP integrality gap bounds (assume metric D) don't formally apply.
- In practice the gap is small. References: Marteau (2009), Jain (2018).

### k-Medoids constraint matrix is NOT totally unimodular
- TU boundary is p=3. For p≤2, the matrix IS TU.
- Odd cycles among facilities break TU (det = (-1)^n - 1 for n-cycle).
- With fixed medoid set, the assignment IS a transportation problem (TU) → enables Benders.

### DTW-AROW ≠ simple zero-cost DTW
- DTW-AROW constrains each missing value to one-to-one diagonal alignment.
- Simple zero-cost is less restrictive and underestimates distances more.

## C++ Performance

### DTW is latency-bound, not memory-bound
- 10-cycle recurrence chain, uses only 3% of L1 bandwidth.
- Multi-pair SIMD gives max ~1.29x due to scatter/gather overhead.
- MSVC auto-vectorizes LB_Keogh/z_normalize to SSE2; explicit Highway adds dispatch overhead.

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

## C++ Implementation

### NaN for missing data
- Use `quiet_NaN()`. Bitwise check via `std::memcpy` (not union type-punning — UB in C++).
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
