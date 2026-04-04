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

### Highway SIMD benchmark results (measured 2026-04-04, MSVC, AVX2, i7-20 core)
Results saved in `benchmarks/results/simd_comparison.json`.

**LB_Keogh**: Highway 2.7–3.3× faster than scalar. MSVC cannot auto-vectorize `std::max` chains;
Highway's explicit vectorization wins decisively.

**z_normalize**:
- Old (3-pass): Highway 0.67–1.2× scalar — slower on large series. Each pass read the full array
  independently, giving no cache reuse benefit over scalar.
- Fix: fuse passes 1+2 into one pass computing `sum(x)` and `sum(x²)` simultaneously →
  `var = E[x²] - mean²`. Reduces to 2 passes.
- New (2-pass): Highway 0.92–1.0× scalar — now essentially at parity. 40% faster at n=8000.

**multi_pair_dtw**:
- Old (scatter/gather): 0.45–0.64× scalar — 1.6–2.2× *slower*. `gather_short(i)` called inside
  O(m²) inner loop; each call did 4 scalar reads from separate memory + stack write + SIMD Load.
  Additionally the OOB mask was re-computed per-cell (4 branches + Load + Gt per iteration) even
  for equal-length pairs where it is always all-false.
- Fix: pre-pack all 4 series into interleaved SoA buffers before the kernel. O(n) packing up front
  → inner loop uses contiguous `Load()` — no scatter. Hoist j_oob mask outside inner loop.
- New (SoA): 0.65–0.98× scalar — at n=1000 essentially at parity. 37% faster than old at n=1000.
- **Lesson**: Any "batch SIMD" kernel that loads data from multiple independent pointers must
  pre-transpose into a contiguous SoA layout; scatter/gather in a recurrence inner loop kills perf.

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
