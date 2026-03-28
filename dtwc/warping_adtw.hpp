/**
 * @file warping_adtw.hpp
 * @brief Amerced Dynamic Time Warping (ADTW) functions.
 *
 * @details ADTW adds a penalty for non-diagonal (horizontal/vertical) steps in
 * the warping path, discouraging time stretching/compression:
 *
 *   C(i,j) = d(x[i], y[j]) + min(C(i-1,j-1),
 *                                  C(i-1,j) + penalty,
 *                                  C(i,j-1) + penalty)
 *
 * Reference: Herrmann, M. & Shifaz, A. (2023). "Amercing: An intuitive and
 *            effective constraint for dynamic time warping."
 *            Pattern Recognition, 137, 109301.
 *
 * @date 28 Mar 2026
 */

#pragma once

#include "settings.hpp"
#include "warping.hpp"
#include "core/scratch_matrix.hpp"

#include <cstdlib>   // for abs, size_t
#include <algorithm> // for min, max
#include <cmath>     // for ceil, floor, round
#include <limits>    // for numeric_limits
#include <vector>    // for vector
#include <utility>   // for pair, tie

namespace dtwc {

/**
 * @brief Computes the full ADTW distance using a rolling-buffer (O(min(m,n)) memory).
 *
 * @tparam data_t Data type of sequence elements.
 * @param x First sequence.
 * @param y Second sequence.
 * @param penalty Penalty added to non-diagonal (horizontal/vertical) warping steps.
 * @return The ADTW distance.
 */
template <typename data_t>
data_t adtwFull_L(const std::vector<data_t> &x, const std::vector<data_t> &y, data_t penalty)
{
  if (&x == &y) return 0; // Same object => distance 0.
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();
  thread_local static std::vector<data_t> short_side(10000);

  const auto &[short_vec, long_vec] = (x.size() < y.size()) ? std::tie(x, y) : std::tie(y, x);
  const auto m_short{ short_vec.size() }, m_long{ long_vec.size() };

  short_side.resize(m_short);

  auto distance = [](data_t a, data_t b) { return std::abs(a - b); };

  if ((m_short == 0) || (m_long == 0)) return maxValue;

  // Base: C(0,0) = d(short[0], long[0])
  short_side[0] = distance(short_vec[0], long_vec[0]);

  // First column: only vertical steps (non-diagonal) => each incurs penalty
  for (size_t i = 1; i < m_short; i++)
    short_side[i] = short_side[i - 1] + penalty + distance(short_vec[i], long_vec[0]);

  // Fill remaining columns
  for (size_t j = 1; j < m_long; j++) {
    auto diag = short_side[0];
    // First row of this column: horizontal step (non-diagonal) => incurs penalty
    short_side[0] += penalty + distance(short_vec[0], long_vec[j]);

    for (size_t i = 1; i < m_short; i++) {
      // short_side[i-1] = C(i-1, j)  — already updated = left
      // short_side[i]   = C(i, j-1)  — not yet updated = below
      // diag            = C(i-1, j-1)
      const data_t min1 = std::min(short_side[i - 1], short_side[i]) + penalty;
      const data_t dist = distance(short_vec[i], long_vec[j]);
      const data_t next = std::min(diag, min1) + dist;

      diag = short_side[i];
      short_side[i] = next;
    }
  }

  return short_side.back();
}


/**
 * @brief Computes the banded ADTW distance using a Sakoe-Chiba band.
 *
 * @tparam data_t Data type of sequence elements.
 * @param x First sequence.
 * @param y Second sequence.
 * @param band Sakoe-Chiba bandwidth. Negative means unbanded (falls back to adtwFull_L).
 * @param penalty Penalty added to non-diagonal warping steps.
 * @return The ADTW distance.
 */
template <typename data_t = double>
data_t adtwBanded(const std::vector<data_t> &x, const std::vector<data_t> &y,
                  int band, data_t penalty)
{
  if (band < 0) return adtwFull_L<data_t>(x, y, penalty);

  thread_local core::ScratchMatrix<data_t> C;
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  const auto &[short_vec, long_vec] = (x.size() < y.size()) ? std::tie(x, y) : std::tie(y, x);
  const int m_short(short_vec.size()), m_long(long_vec.size());

  C.resize(m_long, m_short);
  C.fill(maxValue);

  auto distance = [](data_t a, data_t b) { return std::abs(a - b); };

  if ((m_short == 0) || (m_long == 0)) return maxValue;
  if ((m_short == 1) || (m_long == 1)) return adtwFull_L<data_t>(x, y, penalty);
  if (m_long <= (band + 1)) return adtwFull_L<data_t>(x, y, penalty);

  const double slope = static_cast<double>(m_long - 1) / (m_short - 1);
  const auto window = std::max((double)band, slope / 2);

  auto get_bounds = [slope, window](int idx) {
    const auto yval = slope * idx;
    const int low = std::ceil(std::round(100 * (yval - window)) / 100.0);
    const int high = std::floor(std::round(100 * (yval + window)) / 100.0) + 1;
    return std::pair(low, high);
  };

  C(0, 0) = distance(long_vec[0], short_vec[0]);

  {
    const auto [lo, hi] = get_bounds(0);
    for (int i = 1; i < hi; ++i)
      C(i, 0) = C(i - 1, 0) + penalty + distance(long_vec[i], short_vec[0]);
  }

  for (int j = 1; j < m_short; j++) {
    const auto [lo, hi] = get_bounds(j);
    if (lo <= 0)
      C(0, j) = C(0, j - 1) + penalty + distance(long_vec[0], short_vec[j]);

    const auto high = std::min(hi, m_long);
    for (int i = std::max(lo, 1); i < high; ++i) {
      const auto diag = C(i - 1, j - 1);
      const auto left = C(i, j - 1);
      const auto below = C(i - 1, j);
      // Diagonal is free; horizontal and vertical incur penalty
      const auto minimum = std::min(diag, std::min(left + penalty, below + penalty));
      C(i, j) = minimum + distance(long_vec[i], short_vec[j]);
    }
  }

  return C(m_long - 1, m_short - 1);
}

} // namespace dtwc
