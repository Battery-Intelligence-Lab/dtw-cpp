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
 * @date 29 Mar 2026
 */

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "dtwc/simd/multi_pair_dtw.cpp"
#include "dtwc/simd/highway_targets.hpp"

#include "dtwc/simd/multi_pair_dtw.hpp"

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

  // Handle zero pairs
  if (n_pairs == 0) {
    for (std::size_t p = 0; p < kDtwBatchSize; ++p)
      result.distances[p] = inf;
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
      result.distances[p] = inf;
    return result;
  }

  // Thread-local rolling buffer: interleaved layout.
  // buf[i * 4 + lane] = short_side[i] for pair `lane`.
  // Size: max_short * 4 doubles.
  thread_local std::vector<double> buf;
  buf.resize(max_short * kDtwBatchSize);

  const auto v_inf = hn::Set(d, inf);

  // Helper: gather one element from each pair's series into a SIMD vector.
  // If pair p's index is out of bounds, use 0.0 (won't matter due to masking).
  auto gather_short = [&](std::size_t i) HWY_ATTR -> decltype(hn::Zero(d)) {
    HWY_ALIGN double vals[4];
    for (std::size_t p = 0; p < kDtwBatchSize; ++p) {
      vals[p] = (i < m_shorts[p]) ? short_ptrs[p][i] : 0.0;
    }
    return hn::Load(d, vals);
  };

  auto gather_long = [&](std::size_t j) HWY_ATTR -> decltype(hn::Zero(d)) {
    HWY_ALIGN double vals[4];
    for (std::size_t p = 0; p < kDtwBatchSize; ++p) {
      vals[p] = (j < m_longs[p]) ? long_ptrs[p][j] : 0.0;
    }
    return hn::Load(d, vals);
  };

  // abs_diff: |a - b|
  auto abs_diff = [&](decltype(hn::Zero(d)) a, decltype(hn::Zero(d)) b) HWY_ATTR {
    return hn::Abs(hn::Sub(a, b));
  };

  // --- Initialize rolling buffer: short_side[i] = cumsum of |s[i] - l[0]| ---
  {
    const auto l0 = gather_long(0);  // long_vec[0] for each pair

    // short_side[0] = |s[0] - l[0]|
    const auto s0 = gather_short(0);
    auto prev = abs_diff(s0, l0);

    // For pairs with m_shorts[p] == 0, the gather returns 0 so prev=0.
    // Those lanes are handled at result extraction (set to inf).
    hn::Store(prev, d, buf.data());

    for (std::size_t i = 1; i < max_short; ++i) {
      const auto si = gather_short(i);
      auto cost = abs_diff(si, l0);
      auto val = hn::Add(prev, cost);

      // For pairs where i >= m_shorts[p], set to inf so they don't
      // pollute the min() recurrence for other pairs sharing the buffer.
      HWY_ALIGN double inf_mask[4];
      for (std::size_t p = 0; p < kDtwBatchSize; ++p) {
        inf_mask[p] = (i < m_shorts[p]) ? 0.0 : 1.0;
      }
      auto do_inf = hn::Load(d, inf_mask);
      // Where do_inf > 0, set val to inf
      auto is_oob = hn::Gt(do_inf, hn::Zero(d));
      val = hn::IfThenElse(is_oob, v_inf, val);

      hn::Store(val, d, buf.data() + i * kDtwBatchSize);
      prev = val;
    }
  }

  // --- Main recurrence: for j = 1..max_long-1 ---
  for (std::size_t j = 1; j < max_long; ++j) {
    const auto lj = gather_long(j);

    // diag = short_side[0] (old value before update)
    auto diag = hn::Load(d, buf.data());

    // short_side[0] += |s[0] - l[j]|
    {
      const auto s0 = gather_short(0);
      auto cost = abs_diff(s0, lj);
      auto new_val = hn::Add(diag, cost);

      // For pairs where j >= m_longs[p], this row is past their computation.
      // Their buffer values should remain at the correct final state, which is
      // the value from j = m_longs[p]-1. So we don't update them.
      HWY_ALIGN double jmask[4];
      for (std::size_t p = 0; p < kDtwBatchSize; ++p) {
        jmask[p] = (j < m_longs[p] && m_shorts[p] > 0) ? 0.0 : 1.0;
      }
      auto j_oob = hn::Gt(hn::Load(d, jmask), hn::Zero(d));
      new_val = hn::IfThenElse(j_oob, hn::Load(d, buf.data()), new_val);

      hn::Store(new_val, d, buf.data());
    }

    // Inner loop: i = 1..max_short-1
    for (std::size_t i = 1; i < max_short; ++i) {
      const auto cur = hn::Load(d, buf.data() + i * kDtwBatchSize);      // short_side[i] = C(i, j-1)
      const auto left = hn::Load(d, buf.data() + (i - 1) * kDtwBatchSize); // short_side[i-1] = C(i-1, j) (already updated)

      // min(left, cur) = min(C(i-1,j), C(i,j-1))
      auto min1 = hn::Min(left, cur);
      // min(diag, min1) = min(C(i-1,j-1), C(i-1,j), C(i,j-1))
      auto best = hn::Min(diag, min1);

      const auto si = gather_short(i);
      auto cost = abs_diff(si, lj);
      auto next = hn::Add(best, cost);

      // Save diag before overwrite
      diag = cur;

      // For out-of-bounds lanes, preserve old value
      HWY_ALIGN double mask_vals[4];
      for (std::size_t p = 0; p < kDtwBatchSize; ++p) {
        mask_vals[p] = (i < m_shorts[p] && j < m_longs[p]) ? 0.0 : 1.0;
      }
      auto oob = hn::Gt(hn::Load(d, mask_vals), hn::Zero(d));
      next = hn::IfThenElse(oob, cur, next);

      hn::Store(next, d, buf.data() + i * kDtwBatchSize);
    }
  }

  // Extract results: for pair p, answer is at buf[(m_shorts[p]-1) * 4 + p]
  for (std::size_t p = 0; p < kDtwBatchSize; ++p) {
    if (p < n_pairs && m_shorts[p] > 0 && m_longs[p] > 0) {
      result.distances[p] = buf[(m_shorts[p] - 1) * kDtwBatchSize + p];
    } else {
      result.distances[p] = inf;
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
