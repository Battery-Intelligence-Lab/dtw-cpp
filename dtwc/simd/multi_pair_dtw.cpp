/**
 * @file multi_pair_dtw.cpp
 * @brief Multi-pair DTW: compute 4 independent DTW distances in SIMD lanes.
 *
 * @details
 * **Why inter-pair parallelism?**
 * DTW's inner recurrence is latency-bound (~10 cycles/cell on modern CPUs) because
 * each cell C(i,j) = min(C(i-1,j-1), C(i-1,j), C(i,j-1)) + dist(x[i], y[j])
 * depends on three neighbors. Trying to SIMD-ify a single pair gives at most a
 * marginal win due to this serial dependency chain.
 *
 * Instead, when building a distance matrix we have N*(N-1)/2 independent pairs.
 * By processing 4 pairs simultaneously — one per AVX2 SIMD lane — we execute 4
 * independent recurrences in lockstep. The CPU can hide the 10-cycle latency of
 * one lane's min-chain behind the other three lanes' independent work (ILP).
 *
 * **Data layout: Structure-of-Arrays (SoA) transposition**
 * Without SoA, each DTW cell would need 4 scalar loads from 4 non-contiguous
 * addresses (scatter-gather). SoA interleaving converts these to a single
 * contiguous 32-byte AVX2 Load per element — eliminating O(m*n) gather operations
 * at the cost of an O(m+n) upfront copy.
 *
 * **Rolling buffer**
 * Space: O(max_short * 4) doubles — 4-wide because each position holds one element
 * per pair. The `diag` register carries the diagonal predecessor value to avoid a
 * separate buffer for it.
 *
 * **Uniform-length fast path**
 * When all 4 pairs have the same short and long lengths, OOB masks are always
 * all-false. The fast path removes all mask computation and IfThenElse calls
 * (~30% of per-cell work), leaving a pure: Load → Min → Min → Add → Store loop.
 *
 * **Variable-length masked path**
 * Pre-hoists per-row OOB masks out of the j-loop: `i_oob[i]` is computed once
 * for all i before the main recurrence, saving 4 comparisons + stack write per
 * inner-loop cell vs recomputing inline.
 *
 * **Distance metric**
 * L1 (absolute difference) — consistent with the production scalar DTW path.
 * No sqrt needed; the running sum is already in the same units as the DTW result.
 *
 * **Highway: why not manual intrinsics?**
 * Highway compiles this kernel for SSE4, AVX2, and AVX-512 in a single binary
 * and dispatches at runtime — no separate build variants needed. On HPC clusters
 * with mixed node generations, one binary works optimally on every node.
 *
 * @note On AVX-512, `FixedTag<double, 4>` leaves 4 SIMD lanes unused (AVX-512
 *       supports 8 doubles). Switching to `ScalableTag<double>` with a dynamic
 *       batch size would double throughput on AVX-512 nodes but requires an API
 *       change (dynamic kDtwBatchSize). Tracked as a future improvement.
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "simd/multi_pair_dtw.cpp"
#include "simd/highway_targets.hpp"

#include "simd/multi_pair_dtw.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

HWY_BEFORE_NAMESPACE();
namespace dtwc::simd::HWY_NAMESPACE {

namespace hn = hwy::HWY_NAMESPACE;

// Fixed 4-lane width: native on AVX2 (4 × 64-bit = 256-bit register).
// On narrower ISAs (SSE4), Highway emulates with two 128-bit operations.
using D4 = hn::FixedTag<double, 4>;

MultiPairResult DtwMultiPairImpl(
    const double* const x_ptrs[],
    const double* const y_ptrs[],
    const std::size_t x_lens[],
    const std::size_t y_lens[],
    std::size_t n_pairs)
{
  constexpr double inf     = std::numeric_limits<double>::infinity();
  constexpr double max_val = std::numeric_limits<double>::max();
  MultiPairResult result;

  if (n_pairs == 0) {
    for (std::size_t p = 0; p < kDtwBatchSize; ++p)
      result.distances[p] = max_val;
    return result;
  }

  const D4 d;

  // Orient each pair so short_side <= long_side.
  // Also determine the max short/long lengths across all pairs.
  const double* short_ptrs[kDtwBatchSize];
  const double* long_ptrs[kDtwBatchSize];
  std::size_t m_shorts[kDtwBatchSize];
  std::size_t m_longs[kDtwBatchSize];
  std::size_t max_short = 0;
  std::size_t max_long  = 0;

  for (std::size_t p = 0; p < kDtwBatchSize; ++p) {
    // Duplicate pair 0 into unused lanes to avoid special-casing the inner loop.
    std::size_t src = (p < n_pairs) ? p : 0;
    std::size_t xl = x_lens[src];
    std::size_t yl = y_lens[src];

    if (xl == 0 || yl == 0) {
      short_ptrs[p] = nullptr;
      long_ptrs[p]  = nullptr;
      m_shorts[p]   = 0;
      m_longs[p]    = 0;
      continue;
    }

    if (xl <= yl) {
      short_ptrs[p] = x_ptrs[src];  long_ptrs[p] = y_ptrs[src];
      m_shorts[p]   = xl;            m_longs[p]   = yl;
    } else {
      short_ptrs[p] = y_ptrs[src];  long_ptrs[p] = x_ptrs[src];
      m_shorts[p]   = yl;            m_longs[p]   = xl;
    }
    max_short = std::max(max_short, m_shorts[p]);
    max_long  = std::max(max_long,  m_longs[p]);
  }

  if (max_short == 0 || max_long == 0) {
    for (std::size_t p = 0; p < kDtwBatchSize; ++p)
      result.distances[p] = max_val;
    return result;
  }

  // Pre-pack all 4 pairs into interleaved SoA buffers so the inner DTW loop
  // uses a single contiguous Load() instead of 4 scatter-gather loads.
  //   short_soa[i * 4 + lane] = short_ptrs[lane][i]   (padded with 0.0)
  //   long_soa [j * 4 + lane] = long_ptrs [lane][j]   (padded with 0.0)
  // Cost: O(max_short + max_long) scalar ops once; saves O(m * n) gather ops.
  thread_local std::vector<double> buf, short_soa, long_soa, i_oob_buf;
  buf.resize(max_short * kDtwBatchSize);
  short_soa.assign(max_short * kDtwBatchSize, 0.0);
  long_soa.assign(max_long  * kDtwBatchSize, 0.0);

  for (std::size_t p = 0; p < kDtwBatchSize; ++p) {
    for (std::size_t i = 0; i < m_shorts[p]; ++i)
      short_soa[i * kDtwBatchSize + p] = short_ptrs[p][i];
    for (std::size_t j = 0; j < m_longs[p]; ++j)
      long_soa[j * kDtwBatchSize + p] = long_ptrs[p][j];
  }

  // L1 (absolute difference) cost per cell — consistent with production scalar DTW.
  auto abs_diff = [&](decltype(hn::Zero(d)) a, decltype(hn::Zero(d)) b) HWY_ATTR {
    return hn::Abs(hn::Sub(a, b));
  };

  // Detect uniform-length batch: all pairs have the same short and long lengths.
  // For equal lengths, all OOB masks are all-false and IfThenElse is a no-op.
  // The uniform path removes ~30% of inner-loop work by skipping mask operations.
  const std::size_t min_short = *std::min_element(m_shorts, m_shorts + kDtwBatchSize);
  const std::size_t min_long  = *std::min_element(m_longs,  m_longs  + kDtwBatchSize);
  const bool uniform = (max_short > 0) && (min_short == max_short) && (min_long == max_long);

  if (uniform) {
    // ── Uniform-length fast path: no OOB masking ──────────────────────────────
    // All pairs have the same dimensions; every lane is always active.

    // Init rolling buffer (first column, j=0): cumulative sum of short × long[0].
    {
      const auto l0 = hn::Load(d, long_soa.data());
      auto prev = abs_diff(hn::Load(d, short_soa.data()), l0);
      hn::Store(prev, d, buf.data());
      for (std::size_t i = 1; i < max_short; ++i) {
        const auto si = hn::Load(d, short_soa.data() + i * kDtwBatchSize);
        prev = hn::Add(prev, abs_diff(si, l0));
        hn::Store(prev, d, buf.data() + i * kDtwBatchSize);
      }
    }

    // Main recurrence: standard DTW rolling-buffer recurrence, running in all
    // 4 SIMD lanes simultaneously. Each lane computes one independent DTW pair.
    // The `diag` register carries C(i-1, j-1) across iterations.
    for (std::size_t j = 1; j < max_long; ++j) {
      const auto lj = hn::Load(d, long_soa.data() + j * kDtwBatchSize);

      auto diag = hn::Load(d, buf.data());

      // Row i=0: only left predecessor (C(0, j-1) is buf[0] before update)
      hn::Store(hn::Add(diag, abs_diff(hn::Load(d, short_soa.data()), lj)),
                d, buf.data());

      for (std::size_t i = 1; i < max_short; ++i) {
        const auto cur  = hn::Load(d, buf.data() + i * kDtwBatchSize);
        const auto left = hn::Load(d, buf.data() + (i - 1) * kDtwBatchSize);
        const auto si   = hn::Load(d, short_soa.data() + i * kDtwBatchSize);
        // min(diag, left, cur) + |short[i] - long[j]|
        const auto next = hn::Add(hn::Min(diag, hn::Min(left, cur)), abs_diff(si, lj));
        diag = cur;  // save before overwrite
        hn::Store(next, d, buf.data() + i * kDtwBatchSize);
      }
    }

  } else {
    // ── Variable-length masked path ───────────────────────────────────────────
    // Different pairs may have different lengths. Finished lanes are kept inert
    // via SIMD masks so they don't corrupt the active lanes' recurrence.

    // Pre-hoist per-row OOB masks: i_oob[i] only depends on i, not j.
    // Computing this once before the j-loop saves 4 comparisons + stack write
    // per inner-loop cell (would otherwise be recomputed each j iteration).
    i_oob_buf.resize(max_short * kDtwBatchSize);
    for (std::size_t i = 0; i < max_short; ++i) {
      for (std::size_t p = 0; p < kDtwBatchSize; ++p)
        i_oob_buf[i * kDtwBatchSize + p] = (i < m_shorts[p]) ? 0.0 : 1.0;
    }

    const auto v_inf  = hn::Set(d, inf);
    const auto v_zero = hn::Zero(d);

    // Init rolling buffer (first column, j=0) with row-level masking.
    {
      const auto l0 = hn::Load(d, long_soa.data());
      auto prev = abs_diff(hn::Load(d, short_soa.data()), l0);
      hn::Store(prev, d, buf.data());
      for (std::size_t i = 1; i < max_short; ++i) {
        const auto si = hn::Load(d, short_soa.data() + i * kDtwBatchSize);
        auto val = hn::Add(prev, abs_diff(si, l0));
        // Lanes where this row doesn't exist in the pair get infinity.
        const auto i_oob = hn::Gt(hn::Load(d, i_oob_buf.data() + i * kDtwBatchSize), v_zero);
        val = hn::IfThenElse(i_oob, v_inf, val);
        hn::Store(val, d, buf.data() + i * kDtwBatchSize);
        prev = val;
      }
    }

    // Main recurrence with column-level and row-level masking.
    for (std::size_t j = 1; j < max_long; ++j) {
      const auto lj = hn::Load(d, long_soa.data() + j * kDtwBatchSize);

      // Column-level mask: lanes where j >= m_longs[p] are done for this column.
      HWY_ALIGN double jmask[4];
      for (std::size_t p = 0; p < kDtwBatchSize; ++p)
        jmask[p] = (j < m_longs[p] && m_shorts[p] > 0) ? 0.0 : 1.0;
      const auto j_oob = hn::Gt(hn::Load(d, jmask), v_zero);

      auto diag = hn::Load(d, buf.data());

      // Row i=0
      {
        auto new_val = hn::Add(diag, abs_diff(hn::Load(d, short_soa.data()), lj));
        new_val = hn::IfThenElse(j_oob, diag, new_val);
        hn::Store(new_val, d, buf.data());
      }

      // Inner loop: load pre-hoisted row masks — no per-cell scalar comparisons.
      for (std::size_t i = 1; i < max_short; ++i) {
        const auto cur  = hn::Load(d, buf.data() + i * kDtwBatchSize);
        const auto left = hn::Load(d, buf.data() + (i - 1) * kDtwBatchSize);
        const auto si   = hn::Load(d, short_soa.data() + i * kDtwBatchSize);
        auto next = hn::Add(hn::Min(diag, hn::Min(left, cur)), abs_diff(si, lj));
        diag = cur;
        // Load pre-hoisted row mask (computed once before the j-loop)
        const auto i_oob = hn::Gt(hn::Load(d, i_oob_buf.data() + i * kDtwBatchSize), v_zero);
        next = hn::IfThenElse(hn::Or(i_oob, j_oob), cur, next);
        hn::Store(next, d, buf.data() + i * kDtwBatchSize);
      }
    }
  }

  // Extract results: for pair p, the answer is at buf[(m_shorts[p]-1) * 4 + p].
  for (std::size_t p = 0; p < kDtwBatchSize; ++p) {
    if (p < n_pairs && m_shorts[p] > 0 && m_longs[p] > 0)
      result.distances[p] = buf[(m_shorts[p] - 1) * kDtwBatchSize + p];
    else
      result.distances[p] = max_val;
  }

  return result;
}

}  // namespace dtwc::simd::HWY_NAMESPACE
HWY_AFTER_NAMESPACE();

// --- Dispatch table (compiled once) ---
#if HWY_ONCE
namespace dtwc::simd {

HWY_EXPORT(DtwMultiPairImpl);

MultiPairResult dtw_multi_pair(
    const double* const x_ptrs[],
    const double* const y_ptrs[],
    const std::size_t x_lens[],
    const std::size_t y_lens[],
    std::size_t n_pairs)
{
  return HWY_DYNAMIC_DISPATCH(DtwMultiPairImpl)(x_ptrs, y_ptrs, x_lens, y_lens, n_pairs);
}

}  // namespace dtwc::simd
#endif  // HWY_ONCE
