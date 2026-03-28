/**
 * @file warping_wdtw.hpp
 * @brief Weighted Dynamic Time Warping (WDTW) functions.
 *
 * @details Implements WDTW as described in:
 *   Jeong, Y.-S., Jeong, M. K., & Omitaomu, O. A. (2011).
 *   Weighted dynamic time warping for time series classification.
 *   Pattern Recognition, 44(9), 2231-2240.
 *
 * The local distance at cell (i,j) is multiplied by a position-dependent weight
 * w(|i-j|) that penalizes diagonal deviation:
 *   w(k) = 1 / (1 + exp(-g * (k - max_len/2)))
 * where g controls steepness (higher g = stricter diagonal preference).
 *
 * @date 28 Mar 2026
 */

#pragma once

#include "settings.hpp"
#include "core/scratch_matrix.hpp"

#include <cstdlib>   // for abs, size_t
#include <algorithm> // for min, max
#include <cmath>     // for exp
#include <limits>    // for numeric_limits
#include <vector>    // for vector
#include <utility>   // for pair

namespace dtwc {

namespace detail {

/**
 * @brief Precompute the WDTW weight vector for a given maximum length.
 *
 * @param max_len Maximum of the two series lengths.
 * @param g       Steepness parameter for the logistic weight function.
 * @return        Weight vector of size max_len, where weights[k] = w(|i-j| = k).
 */
template <typename data_t>
inline void computeWDTWWeights(std::vector<data_t> &weights, int max_len, data_t g)
{
  weights.resize(static_cast<size_t>(max_len));
  const data_t half = static_cast<data_t>(max_len) / 2.0;
  for (int k = 0; k < max_len; ++k) {
    weights[k] = static_cast<data_t>(1.0) / (static_cast<data_t>(1.0) + std::exp(-g * (static_cast<data_t>(k) - half)));
  }
}

} // namespace detail


/**
 * @brief Computes the full Weighted DTW distance between two sequences.
 *
 * Uses a rolling-buffer (single vector) approach identical to dtwFull_L,
 * but multiplies each local distance by the positional weight w(|i-j|).
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x First sequence.
 * @param y Second sequence.
 * @param g Steepness parameter for the logistic weight function.
 * @return The weighted dynamic time warping distance.
 */
template <typename data_t>
data_t wdtwFull(const std::vector<data_t> &x, const std::vector<data_t> &y, data_t g)
{
  if (&x == &y) return 0; // Same object => distance is 0.
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  const auto &[short_vec, long_vec] = (x.size() < y.size()) ? std::tie(x, y) : std::tie(y, x);
  const auto m_short = short_vec.size();
  const auto m_long = long_vec.size();

  if ((m_short == 0) || (m_long == 0)) return maxValue;

  // Precompute weights: w[k] for k = 0, 1, ..., max_len-1
  const int max_len = static_cast<int>(std::max(m_short, m_long));
  thread_local std::vector<data_t> weights;
  detail::computeWDTWWeights(weights, max_len, g);

  // Rolling buffer along the short side
  thread_local std::vector<data_t> short_side;
  short_side.resize(m_short);

  auto distance = [](data_t a, data_t b) { return std::abs(a - b); };

  // Note: after the swap, short_side[i] corresponds to short_vec[i],
  // long_vec[j] corresponds to the other series. The weight index is |i - j|.
  // Since w depends only on |i-j| (symmetric), the swap doesn't affect correctness.

  short_side[0] = weights[0] * distance(short_vec[0], long_vec[0]); // |0-0| = 0

  for (size_t i = 1; i < m_short; i++) {
    const int dev = static_cast<int>(i); // |i - 0|
    short_side[i] = short_side[i - 1] + weights[dev] * distance(short_vec[i], long_vec[0]);
  }

  for (size_t j = 1; j < m_long; j++) {
    auto diag = short_side[0];
    {
      const int dev = static_cast<int>(j); // |0 - j|
      short_side[0] += weights[dev] * distance(short_vec[0], long_vec[j]);
    }

    for (size_t i = 1; i < m_short; i++) {
      const int dev = std::abs(static_cast<int>(i) - static_cast<int>(j));
      const data_t min1 = std::min(short_side[i - 1], short_side[i]);
      const data_t dist = weights[dev] * distance(short_vec[i], long_vec[j]);
      const data_t next = std::min(diag, min1) + dist;

      diag = short_side[i];
      short_side[i] = next;
    }
  }

  return short_side.back();
}


/**
 * @brief Computes the banded Weighted DTW distance between two sequences.
 *
 * Uses the same Sakoe-Chiba banding as dtwBanded, but with WDTW positional weights.
 * If band < 0, falls back to full WDTW.
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x     First sequence.
 * @param y     Second sequence.
 * @param band  The bandwidth parameter (-1 for full).
 * @param g     Steepness parameter for the logistic weight function.
 * @return The weighted banded dynamic time warping distance.
 */
template <typename data_t = double>
data_t wdtwBanded(const std::vector<data_t> &x, const std::vector<data_t> &y,
                  int band, data_t g)
{
  if (band < 0) return wdtwFull<data_t>(x, y, g);

  if (&x == &y) return 0;
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  const auto &[short_vec, long_vec] = (x.size() < y.size()) ? std::tie(x, y) : std::tie(y, x);
  const int m_short(short_vec.size()), m_long(long_vec.size());

  if ((m_short == 0) || (m_long == 0)) return maxValue;
  if ((m_short == 1) || (m_long == 1)) return wdtwFull<data_t>(x, y, g);
  if (m_long <= (band + 1)) return wdtwFull<data_t>(x, y, g);

  // Precompute weights
  const int max_len = std::max(m_short, m_long);
  thread_local std::vector<data_t> weights;
  detail::computeWDTWWeights(weights, max_len, g);

  thread_local core::ScratchMatrix<data_t> C;
  C.resize(m_long, m_short);
  C.fill(maxValue);

  auto distance = [](data_t a, data_t b) { return std::abs(a - b); };

  const double slope = static_cast<double>(m_long - 1) / (m_short - 1);
  const auto window = std::max(static_cast<double>(band), slope / 2);

  auto get_bounds = [slope, window](int x_val) {
    const auto y_val = slope * x_val;
    const int low = static_cast<int>(std::ceil(std::round(100 * (y_val - window)) / 100.0));
    const int high = static_cast<int>(std::floor(std::round(100 * (y_val + window)) / 100.0)) + 1;
    return std::pair(low, high);
  };

  // In the banded version, i indexes long_vec and j indexes short_vec.
  // The weight deviation is |i - j| in cost matrix coordinates.

  C(0, 0) = weights[0] * distance(long_vec[0], short_vec[0]);

  {
    const auto [lo, hi] = get_bounds(0);
    for (int i = 1; i < hi; ++i) {
      const int dev = i; // |i - 0|
      C(i, 0) = C(i - 1, 0) + weights[dev] * distance(long_vec[i], short_vec[0]);
    }
  }

  for (int j = 1; j < m_short; ++j) {
    const auto [lo, hi] = get_bounds(j);
    if (lo <= 0) {
      const int dev = j; // |0 - j|
      C(0, j) = C(0, j - 1) + weights[dev] * distance(long_vec[0], short_vec[j]);
    }

    const auto high = std::min(hi, m_long);
    for (int i = std::max(lo, 1); i < high; ++i) {
      const int dev = std::abs(i - j);
      const auto minimum = std::min({ C(i - 1, j), C(i, j - 1), C(i - 1, j - 1) });
      C(i, j) = minimum + weights[dev] * distance(long_vec[i], short_vec[j]);
    }
  }

  return C(m_long - 1, m_short - 1);
}

} // namespace dtwc
