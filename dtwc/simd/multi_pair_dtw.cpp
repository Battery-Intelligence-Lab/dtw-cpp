/**
 * @file multi_pair_dtw.cpp
 * @brief Multi-pair DTW: compute 4 independent DTW distances in SIMD lanes.
 *
 * @details Implements the rolling-buffer DTW recurrence with a 4-wide interleaved
 *          buffer. Each SIMD lane runs one pair's DTW computation in lockstep.
 *          Uses FixedTag<double, 4> for guaranteed 4-lane operation (native on
 *          AVX2, emulated on narrower ISAs).
 *
 *          Variable-length pairs are handled by masking: finished lanes receive
 *          infinity so they don't affect the ongoing recurrence for other pairs.
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

using D4 = hn::FixedTag<double, 4>;

MultiPairResult DtwMultiPairImpl(
    const double* const x_ptrs[],
    const double* const y_ptrs[],
    const std::size_t x_lens[],
    const std::size_t y_lens[],
    std::size_t n_pairs)
{
  constexpr double inf = std::numeric_limits<double>::infinity();
  MultiPairResult result;

  constexpr double max_val = std::numeric_limits<double>::max();

  // Handle zero pairs
  if (n_pairs == 0) {
    for (std::size_t p = 0; p < kDtwBatchSize; ++p)
      result.distances[p] = max_val;
    return result;
  }

  const D4 d;

  // Orient each pair: short_side is the shorter series, long_side the longer.
  // Also determine max short/long lengths across all pairs.
  const double* short_ptrs[kDtwBatchSize];
  const double* long_ptrs[kDtwBatchSize];
  std::size_t m_shorts[kDtwBatchSize];
  std::size_t m_longs[kDtwBatchSize];
  std::size_t max_short = 0;
  std::size_t max_long = 0;

  for (std::size_t p = 0; p < kDtwBatchSize; ++p) {
    // For unused lanes (p >= n_pairs), duplicate pair 0's data
    std::size_t src = (p < n_pairs) ? p : 0;
    std::size_t xl = x_lens[src];
    std::size_t yl = y_lens[src];

    if (xl == 0 || yl == 0) {
      // Empty series: mark as degenerate; will produce inf
      short_ptrs[p] = nullptr;
      long_ptrs[p] = nullptr;
      m_shorts[p] = 0;
      m_longs[p] = 0;
      continue;
    }

    if (xl <= yl) {
      short_ptrs[p] = x_ptrs[src];
      long_ptrs[p] = y_ptrs[src];
      m_shorts[p] = xl;
      m_longs[p] = yl;
    } else {
      short_ptrs[p] = y_ptrs[src];
      long_ptrs[p] = x_ptrs[src];
      m_shorts[p] = yl;
      m_longs[p] = xl;
    }

    max_short = std::max(max_short, m_shorts[p]);
    max_long = std::max(max_long, m_longs[p]);
  }

  // Handle all-empty case
  if (max_short == 0 || max_long == 0) {
    for (std::size_t p = 0; p < kDtwBatchSize; ++p)
      result.distances[p] = max_val;
    return result;
  }

  // Pre-pack all 4 pairs' series into interleaved SoA buffers so the inner
  // DTW loop uses contiguous Load() instead of scatter-gather.
  // short_soa[i*4 + lane] = short_ptrs[lane][i]  (0.0 if out of bounds)
  // long_soa [j*4 + lane] = long_ptrs [lane][j]  (0.0 if out of bounds)
  // Cost: O(max_short + max_long) scalar ops up front; saves O(m²) gather
  // ops in the inner loop.
  thread_local std::vector<double> buf, short_soa, long_soa;
  buf.resize(max_short * kDtwBatchSize);
  short_soa.resize(max_short * kDtwBatchSize, 0.0);
  long_soa.resize(max_long  * kDtwBatchSize, 0.0);

  for (std::size_t p = 0; p < kDtwBatchSize; ++p) {
    for (std::size_t i = 0; i < m_shorts[p]; ++i)
      short_soa[i * kDtwBatchSize + p] = short_ptrs[p][i];
    for (std::size_t j = 0; j < m_longs[p]; ++j)
      long_soa[j * kDtwBatchSize + p] = long_ptrs[p][j];
  }

  const auto v_inf = hn::Set(d, inf);

  // Precompute per-column (j) and per-row (i) OOB masks so the inner loop
  // never recomputes them from scratch.
  // j_active[j] = mask of lanes that are still computing at column j.
  // short_active[i] = mask of lanes whose short series has row i.
  // Both are computed once and stored as SIMD masks.
  // For equal-length pairs (the common case), all masks are all-true and the
  // IfThenElse calls become no-ops that the compiler can optimise away.

  // abs_diff: |a - b|
  auto abs_diff = [&](decltype(hn::Zero(d)) a, decltype(hn::Zero(d)) b) HWY_ATTR {
    return hn::Abs(hn::Sub(a, b));
  };

  // --- Initialize rolling buffer (first column, j=0) ---
  {
    const auto l0 = hn::Load(d, long_soa.data());  // contiguous!

    auto prev = abs_diff(hn::Load(d, short_soa.data()), l0);
    hn::Store(prev, d, buf.data());

    for (std::size_t i = 1; i < max_short; ++i) {
      const auto si = hn::Load(d, short_soa.data() + i * kDtwBatchSize);  // contiguous!
      auto val = hn::Add(prev, abs_diff(si, l0));

      // Mask out lanes where this row is beyond the pair's short length.
      // Precompute per-row mask (hoisted: only depends on i, not j).
      HWY_ALIGN double inf_mask[4];
      for (std::size_t p = 0; p < kDtwBatchSize; ++p)
        inf_mask[p] = (i < m_shorts[p]) ? 0.0 : 1.0;
      val = hn::IfThenElse(hn::Gt(hn::Load(d, inf_mask), hn::Zero(d)), v_inf, val);

      hn::Store(val, d, buf.data() + i * kDtwBatchSize);
      prev = val;
    }
  }

  // --- Main recurrence: for j = 1..max_long-1 ---
  for (std::size_t j = 1; j < max_long; ++j) {
    const auto lj = hn::Load(d, long_soa.data() + j * kDtwBatchSize);  // contiguous!

    // Precompute column-level OOB mask: lanes where j >= m_longs[p] are done.
    HWY_ALIGN double jmask[4];
    for (std::size_t p = 0; p < kDtwBatchSize; ++p)
      jmask[p] = (j < m_longs[p] && m_shorts[p] > 0) ? 0.0 : 1.0;
    const auto j_oob = hn::Gt(hn::Load(d, jmask), hn::Zero(d));

    auto diag = hn::Load(d, buf.data());

    // Row i=0
    {
      auto new_val = hn::Add(diag, abs_diff(hn::Load(d, short_soa.data()), lj));
      new_val = hn::IfThenElse(j_oob, diag, new_val);
      hn::Store(new_val, d, buf.data());
    }

    // Inner loop: i = 1..max_short-1
    // Per-row OOB check (i < m_shorts[p]) is constant across j iterations,
    // but we only enter this branch for rows that exist in at least one pair.
    for (std::size_t i = 1; i < max_short; ++i) {
      const auto cur  = hn::Load(d, buf.data() + i * kDtwBatchSize);
      const auto left = hn::Load(d, buf.data() + (i - 1) * kDtwBatchSize);

      auto best = hn::Min(diag, hn::Min(left, cur));
      const auto si   = hn::Load(d, short_soa.data() + i * kDtwBatchSize);  // contiguous!
      auto next = hn::Add(best, abs_diff(si, lj));

      diag = cur;

      // Apply both OOB guards: column-level (j_oob) is already computed above.
      // Row-level (i >= m_shorts[p]) is hoistable per i but kept inline here
      // since it's a single Load+Gt vs the old per-cell 4-branch scalar loop.
      HWY_ALIGN double imask[4];
      for (std::size_t p = 0; p < kDtwBatchSize; ++p)
        imask[p] = (i < m_shorts[p]) ? 0.0 : 1.0;
      const auto i_oob = hn::Gt(hn::Load(d, imask), hn::Zero(d));
      next = hn::IfThenElse(hn::Or(i_oob, j_oob), cur, next);

      hn::Store(next, d, buf.data() + i * kDtwBatchSize);
    }
  }

  // Extract results: for pair p, answer is at buf[(m_shorts[p]-1) * 4 + p].
  // Use max() (not inf) for empty pairs: matches the production DTW convention.
  for (std::size_t p = 0; p < kDtwBatchSize; ++p) {
    if (p < n_pairs && m_shorts[p] > 0 && m_longs[p] > 0) {
      result.distances[p] = buf[(m_shorts[p] - 1) * kDtwBatchSize + p];
    } else {
      result.distances[p] = max_val;
    }
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
