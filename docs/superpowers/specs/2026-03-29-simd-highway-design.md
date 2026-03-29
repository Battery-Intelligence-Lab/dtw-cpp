# Sub-phase B: SIMD via Google Highway

**Date:** 2026-03-29
**Status:** Approved
**Branch:** Claude

---

## Problem

DTW distance matrix construction is the dominant cost in k-medoids clustering. The recurrence chain is latency-bound (10 cycles/cell), and LB_Keogh pruning is called O(N^2) times. Both leave significant performance on the table without SIMD.

## Solution

Integrate Google Highway (v1.2.0) for runtime-dispatched SIMD across AVX2, AVX-512, SSE4, and NEON. Four work items in priority order:

### B0: CMake Integration

- Add `DTWC_ENABLE_SIMD` option (default ON)
- Fetch Highway v1.2.0 via CPM (same pattern as Google Benchmark)
- Define `DTWC_HAS_HIGHWAY` when available
- Scalar fallback when disabled — no functional change

### B1: SIMD LB_Keogh (Expected 4-8x)

LB_Keogh is a pure reduction: three contiguous array reads, element-wise max, horizontal sum. Zero data dependencies between iterations. Textbook SIMD target.

```
for i in 0..n:
    excess = max(0, max(query[i] - upper[i], lower[i] - query[i]))
    sum += excess
```

Vectorize with Highway `Load`, `Max`, `Sub`, `ReduceSum`. Process 4 doubles (AVX2) or 8 doubles (AVX-512) per iteration. Tail handling via Highway's `FirstN` mask.

### B2: SIMD z_normalize (Expected 4-8x)

Three independent loops (sum, squared-deviation, normalize). All are embarrassingly parallel reductions or element-wise transforms. Same Highway pattern as LB_Keogh.

### B3: SIMD compute_envelopes (Expected 2-4x)

Sliding window min/max over band width. Use Highway `Min`/`Max` instructions. Less impact than B1/B2 because the band is typically small.

### B4: Multi-pair DTW (Expected 3-4x on fillDistanceMatrix)

The key architectural insight: DTW's recurrence chain (`min(diag, min(left, below)) + cost`) is sequential within one pair, but **independent across pairs**. Process 4 pairs simultaneously:

```
Lane 0: dtw(x0, y0)    Lane 1: dtw(x1, y1)
Lane 2: dtw(x2, y2)    Lane 3: dtw(x3, y3)
```

All lanes execute the same recurrence in lockstep. No cross-lane communication. The rolling buffer becomes 4-wide: each element is a `Vec<double, 4>`.

Implementation: `dtwFull_L_multi<4>` processes 4 pairs, with a scalar tail for remaining pairs. Called from `fillDistanceMatrix` which batches pairs into groups of 4.

## File Structure

```
dtwc/
  simd/
    highway_targets.hpp    -- HWY_DYNAMIC_DISPATCH boilerplate
    lb_keogh_simd.cpp      -- SIMD LB_Keogh (B1)
    z_normalize_simd.cpp   -- SIMD z_normalize (B2)
    envelopes_simd.cpp     -- SIMD envelope computation (B3)
    multi_pair_dtw.cpp     -- 4-wide DTW for distance matrix (B4)
```

## Build System Changes

```cmake
option(DTWC_ENABLE_SIMD "Enable SIMD via Google Highway" ON)
if(DTWC_ENABLE_SIMD)
  CPMAddPackage("gh:google/highway@1.2.0")
  if(TARGET hwy::hwy)
    target_link_libraries(dtwc++ PRIVATE hwy::hwy)
    target_compile_definitions(dtwc++ PUBLIC DTWC_HAS_HIGHWAY)
  else()
    message(WARNING "Highway not found, SIMD disabled")
    set(DTWC_ENABLE_SIMD OFF)
  endif()
endif()
```

## Integration Pattern

Each SIMD function has a scalar fallback in the existing `core/` headers. The dispatch is compile-time via `#ifdef DTWC_HAS_HIGHWAY`:

```cpp
#ifdef DTWC_HAS_HIGHWAY
  T lb_keogh_dispatch(const T* q, const T* u, const T* l, size_t n);
#endif

inline T lb_keogh(const T* q, const T* u, const T* l, size_t n) {
#ifdef DTWC_HAS_HIGHWAY
    return lb_keogh_dispatch(q, u, l, n);
#else
    return lb_keogh_scalar(q, u, l, n);
#endif
}
```

Inside `lb_keogh_simd.cpp`, Highway's `HWY_DYNAMIC_DISPATCH` handles runtime ISA selection (AVX-512 on server, AVX2 on desktop, NEON on ARM).

## What Does NOT Get SIMD

- **Soft-DTW**: exp/log in softmin, full cost matrix requirement dominates
- **ADTW/WDTW**: Same recurrence dependency; multi-pair applies but lower priority
- **FastPAM SWAP**: Already OpenMP-parallelized, bottleneck is distance matrix

## Benchmarks

Extend `bench_dtw_baseline.cpp` with:
- `BM_lb_keogh` (scalar vs SIMD, lengths 100-8000)
- `BM_z_normalize` (scalar vs SIMD, lengths 100-8000)
- `BM_fillDistanceMatrix` (before/after multi-pair DTW)

Save results as `bench_simd.json` for comparison with `bench_subphase_a.json`.

## Success Criteria

- All existing tests pass unchanged (SIMD produces identical results)
- LB_Keogh: >= 3x speedup at n=1000
- fillDistanceMatrix: >= 2x speedup at N=50, L=500
- Graceful fallback: builds and runs identically with `-DDTWC_ENABLE_SIMD=OFF`
- Cross-platform: compiles on Windows (MSVC), Linux (GCC), macOS (Clang/ARM)

## References

- Google Highway: https://github.com/google/highway
- Schubert & Rousseeuw (2021) "Fast and eager k-medoids clustering" JMLR
- Sakoe & Chiba (1978) band constraint
- Keogh & Ratanamahatana (2005) LB_Keogh lower bound
