# Plan: SIMD Improvements — Branchless Scalar, Algorithm Transfer & Highway Optimization

## Context

The `dtwc/simd/` folder has 3 Google Highway SIMD kernels. Benchmarks (`benchmarks/results/simd_comparison.json`) show:

| Kernel | Speedup vs scalar | Status |
|--------|-------------------|--------|
| lb_keogh_highway | **2.7–3.6x faster** | Wired into production |
| z_normalize_highway | 0.64–0.99x (SLOWER) | Not wired (correct) |
| multi_pair_dtw | 0.49–0.69x (SLOWER) | Not wired (correct) |

The SIMD z_normalize uses a smarter 2-pass algorithm (König-Huygens fused sum+sum-of-squares) vs the scalar's 3 separate passes. This algorithmic improvement transfers to scalar.

The scalar lb_keogh is 2.7–3.6x slower than Highway primarily because `std::max()` calls inhibit auto-vectorization (especially on MSVC, which has no `#pragma omp simd reduction`). A branchless rewrite can close this gap significantly.

---

## Step 1: Branchless scalar lb_keogh (highest impact)

**File:** [lower_bound_impl.hpp](dtwc/core/lower_bound_impl.hpp)

### 1a. L1 variant (`lb_keogh`, line 179-183)

Current:
```cpp
sum += std::max(T(0), std::max(excess_upper, excess_lower));
```

`std::max` takes `const T&` — may not inline/vectorize, especially on MSVC. Nested call creates a data dependency chain.

**Rewrite using decomposed branchless max-with-zero:**
```cpp
T eu = query[i] - upper[i];   // positive when query above upper
T el = lower[i] - query[i];   // positive when query below lower
// For a valid envelope (L <= U), at most one of eu, el can be positive.
// Decomposing max(0, max(eu,el)) into max(0,eu) + max(0,el) is equivalent
// and lets both clamps execute independently — two vmaxpd, no dependency chain.
T cu = eu > T(0) ? eu : T(0);
T cl = el > T(0) ? el : T(0);
sum += cu + cl;
```

Why this works: if `L[i] <= U[i]` (always true for valid envelopes), then `eu + el = (q-U) + (L-q) = L-U <= 0`, so at most one of `eu, el` is positive. Therefore `max(0,eu) + max(0,el) == max(0, max(eu,el))`.

Each ternary `x > 0 ? x : 0` maps to a single `vmaxpd` instruction on x86, giving the compiler two independent SIMD operations per element.

### 1b. SquaredL2 variant (`lb_keogh_squared`, lines 491-496)

Current uses explicit `if/else if` branches — completely blocks auto-vectorization:
```cpp
if (query[i] > upper[i]) excess = query[i] - upper[i];
else if (query[i] < lower[i]) excess = lower[i] - query[i];
```

**Rewrite branchless (same decomposition):**
```cpp
T eu = query[i] - upper[i];
T el = lower[i] - query[i];
T cu = eu > T(0) ? eu : T(0);
T cl = el > T(0) ? el : T(0);
T excess = cu + cl;
sum += excess * excess;
```

### 1c. Multivariate variants (`lb_keogh_mv`, `lb_keogh_mv_squared`)

Apply the same branchless rewrite to the inner loops at lines ~456 and ~522-529.

### 1d. Add `#pragma omp simd reduction` to all variants

Currently only `lb_keogh` (L1) has the pragma. Add to `lb_keogh_squared`, `lb_keogh_mv`, `lb_keogh_mv_squared` with the same MSVC guard.

### Expected impact
The branchless rewrite should bring scalar lb_keogh much closer to Highway performance on GCC/Clang (where `#pragma omp simd` is active). On MSVC, MSVC's auto-vectorizer should handle ternary operators in simple loops. If the scalar becomes competitive (within ~1.5x of Highway), Highway dispatch for lb_keogh is still worthwhile as a guaranteed floor.

---

## Step 2: Transfer König-Huygens fused pass to scalar z_normalize

**File:** [z_normalize.hpp](dtwc/core/z_normalize.hpp)

Current scalar: **3 passes** (mean → squared-deviation → normalize)
SIMD version: **2 passes** (fused sum+sum² → normalize)

**Rewrite:**
```cpp
// Pass 1 (fused): sum(x) and sum(x²) in one sweep.
// König-Huygens: var = E[x²] - mean². Halves pass-1 memory traffic.
T sum = 0, sq_sum = 0;
#pragma omp simd reduction(+:sum, sq_sum)  // double reduction
for (size_t i = 0; i < n; ++i) {
    sum    += series[i];
    sq_sum += series[i] * series[i];
}
T mean = sum / T(n);
// Guard: König-Huygens can produce tiny negatives from fp rounding.
T variance = std::max(T(0), sq_sum / T(n) - mean * mean);
T stddev = std::sqrt(variance);

// Pass 2: normalize with FMA-friendly formulation.
// (x - mean) * inv_stddev == x * inv_stddev + (-mean * inv_stddev)
if (stddev > T(1e-10)) {
    T inv_sd = T(1) / stddev;
    T bias = -mean * inv_sd;  // precompute once
    #pragma omp simd
    for (size_t i = 0; i < n; ++i)
        series[i] = series[i] * inv_sd + bias;  // single FMA
} else { /* zero-fill as before */ }
```

**Numerical note:** König-Huygens loses precision when mean >> stddev (catastrophic cancellation in `E[x²] - mean²`). For double precision (~15 digits), this is acceptable for typical time series values. The `std::max(T(0), ...)` guard handles the edge case. Comment this in the code.

**Benchmark verification required:** Confirm the fused 2-pass doesn't regress for small n where L1 cache hides the extra-pass cost.

---

## Step 3: multi_pair_dtw — uniform-length mask-free fast path

**File:** [multi_pair_dtw.cpp](dtwc/simd/multi_pair_dtw.cpp)

### Problem
The inner loop (lines 181–201) recomputes per-row OOB masks (`imask`: 4 scalar comparisons + stack write + SIMD load + Gt) at every `(i,j)` cell. These masks only depend on `i`, not `j`. For **equal-length pairs** (common case in DTW clustering on uniform-length datasets), ALL masks are all-false and every `IfThenElse` is a wasted no-op (~30% of inner loop work).

### Solution: template on `bool kMasked`

Extract the DTW recurrence into a templated helper:

```cpp
template <bool kMasked>
void DtwRecurrence(D4 d, double* buf, const double* short_soa, const double* long_soa,
                   std::size_t max_short, std::size_t max_long,
                   const std::size_t m_shorts[], const std::size_t m_longs[],
                   auto abs_diff);
```

- **`kMasked = false`:** Pure recurrence. No `imask`, no `j_oob`, no `IfThenElse`. Just: Load → Min → Min → Add → Store.
- **`kMasked = true`:** Current masked code (optionally with pre-hoisted `i_oob` masks for the j-invariant row masks).

Dispatch at the top of `DtwMultiPairImpl`:
```cpp
const bool uniform = std::all_of(...)  // all short lens equal && all long lens equal
if (uniform)
    DtwRecurrence<false>(...);
else
    DtwRecurrence<true>(...);
```

### Pre-hoist row masks (for the masked path)

Even in the masked path, hoist `i_oob` computation out of the j-loop. Precompute `max_short` mask values once, store in a `HWY_ALIGN double[max_short * 4]` buffer, load in the inner loop. Saves 4 comparisons + stack write per cell.

### Expected impact
Uniform-length path: ~30% fewer operations per cell. For n=1000, that's ~300M fewer ops for 4 pairs. Combined with mask pre-hoisting in the variable-length path, overall improvement estimated at 15–30%.

---

## Step 4: SIMD z_normalize — FMA optimization + comment fix

**File:** [z_normalize_simd.cpp](dtwc/simd/z_normalize_simd.cpp)

### 4a. Fix wrong header comment (line 5)
Current: "Three embarrassingly parallel loops vectorized"
Should be: "Two-pass fused approach: sum(x) + sum(x²) in one sweep (König-Huygens), then normalize"

### 4b. FMA in normalize pass (lines 62–65)
Current: `Mul(Sub(val, mean_vec), inv_sd_vec)` — 2 ops (Sub + Mul)
Proposed: `MulAdd(val, inv_sd_vec, neg_mean_over_sd)` — 1 FMA op

```cpp
const auto neg_mean_over_sd = hn::Set(d, -mean / stddev);
// in loop:
const auto normed = hn::MulAdd(val, inv_sd_vec, neg_mean_over_sd);
```

Saves 1 SIMD op per element. Won't make z_normalize_highway faster than scalar overall (the kernel has fundamental overhead from Highway dispatch), but it's a free improvement.

---

## Step 5: Documentation — algorithmic comments

### 5a. multi_pair_dtw.cpp — add inline comments

The file header (lines 1–15) is decent. Add targeted inline comments:

- **Before the SoA transposition loop (line 114):** Explain why interleaving is critical: "Without SoA, each cell would require 4 scalar loads from non-contiguous addresses (scatter-gather). SoA converts these to a single contiguous 32-byte Load per element."

- **Before the inner loop (line 181):** "Standard DTW rolling-buffer recurrence, running in 4 SIMD lanes simultaneously. Each lane computes one independent DTW pair. The `diag` register carries the diagonal predecessor across iterations to avoid a separate buffer."

- **At the `abs_diff` lambda (line 132):** "L1 (absolute difference) cost, consistent with the L1 metric in the production scalar DTW path. No sqrt needed."

- **Near `FixedTag<double, 4>` (line 33):** "Fixed 4-lane width matches AVX2 natively. On narrower ISAs (SSE4), Highway emulates with 2×128-bit ops. On wider ISAs (AVX-512), this limits to 4 lanes — see open question about ScalableTag for 8-lane operation."

### 5b. lb_keogh_simd.cpp — add note about ScalableTag

After line 30 (`ScalableTag<double>`):
```
// ScalableTag adapts to the widest available ISA at runtime:
// 2 doubles on SSE4, 4 on AVX2, 8 on AVX-512.
// Unlike FixedTag, this automatically benefits from wider hardware.
```

### 5c. highway_targets.hpp — add "why Highway" note

Add after line 28:
```
// Why Highway over manual intrinsics:
// - Single binary, runtime ISA dispatch — no need to build separate SSE/AVX/AVX-512 variants
// - Portable across x86, ARM NEON, RISC-V V — future-proofs for non-x86 HPC (Graviton, etc.)
// - Tested, maintained SIMD abstraction — avoids the footgun surface of raw intrinsics
```

---

## Files Modified

| File | Change |
|------|--------|
| `dtwc/core/lower_bound_impl.hpp` | Branchless lb_keogh + lb_keogh_squared + MV variants; add `#pragma omp simd` to squared variants |
| `dtwc/core/z_normalize.hpp` | Fused 2-pass König-Huygens; FMA-friendly normalize |
| `dtwc/simd/multi_pair_dtw.cpp` | Template `<bool kMasked>` fast path; pre-hoist row masks |
| `dtwc/simd/z_normalize_simd.cpp` | Fix header comment; FMA in normalize pass |
| `dtwc/simd/lb_keogh_simd.cpp` | Add ScalableTag comment |
| `dtwc/simd/highway_targets.hpp` | Add "why Highway" comment |
| `dtwc/simd/multi_pair_dtw.hpp` | (No code changes, comments only if needed) |

---

## Verification

1. **Build:** `cmake -B build -DDTWC_ENABLE_SIMD=ON -DDTWC_BUILD_TESTING=ON -DDTWC_BUILD_BENCHMARK=ON && cmake --build build --config Release`
2. **Tests:** `ctest --test-dir build -C Release` — all SIMD and core tests must pass
3. **Benchmarks:** Run `bench_dtw_baseline` and compare against `simd_comparison.json`:
   - lb_keogh scalar should improve significantly (target: within 1.5x of Highway)
   - z_normalize scalar should be similar or slightly faster (2 passes vs 3)
   - multi_pair_dtw with equal lengths should improve ~15-30%
4. **Precision check:** Verify z_normalize large-value test still passes with König-Huygens

## NOT doing (with rationale)

- **Wire z_normalize_highway into production:** Benchmarks show it's 0.64x scalar — would be a regression.
- **ScalableTag for multi_pair_dtw:** Requires dynamic batch size, API change, caller restructure. Note as future work for AVX-512 nodes.
- **Early abandon in multi_pair_dtw:** Requires per-lane thresholds and API changes. Future work for pruned distance matrix integration.
