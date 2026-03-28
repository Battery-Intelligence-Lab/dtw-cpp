/**
 * @file warping.hpp
 * @brief Time warping functions.
 *
 * @details This file contains functions for dynamic time warping, which is a method to
 * compare two temporal sequences that may vary in time or speed. It includes
 * different versions of the algorithm for full, light (L), and banded computations.
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 08 Dec 2022
 */

#pragma once

#include <cstdlib>   // for abs, size_t
#include <algorithm> // for min, max
#include <cmath>     // for floor, round
#include <limits>    // for numeric_limits
#include <vector>    // for vector
#include <utility>   // for pair

#include "core/scratch_matrix.hpp"

namespace dtwc {

/**
 * @brief Computes the full dynamic time warping distance between two sequences.
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x First sequence.
 * @param y Second sequence.
 * @return The dynamic time warping distance.
 */
template <typename data_t>
data_t dtwFull(const std::vector<data_t> &x, const std::vector<data_t> &y)
{
  thread_local core::ScratchMatrix<data_t> C;
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  const int mx = x.size();
  const int my = y.size();

  if ((mx == 0) || (my == 0)) return maxValue;
  if (&x == &y) return 0; // If they are the same data then distance is 0.

  C.resize(mx, my);

  auto distance = [](data_t x, data_t y) { return std::abs(x - y); };

  C(0, 0) = distance(x[0], y[0]);

  for (int i = 1; i < mx; i++)
    C(i, 0) = C(i - 1, 0) + distance(x[i], y[0]); // j = 0

  for (int j = 1; j < my; j++)
    C(0, j) = C(0, j - 1) + distance(x[0], y[j]); // i = 0


  for (int j = 1; j < my; j++) {
    for (int i = 1; i < mx; i++) {
      const auto minimum = std::min({ C(i - 1, j), C(i, j - 1), C(i - 1, j - 1) });
      C(i, j) = minimum + distance(x[i], y[j]);
    }
  }

  return C(mx - 1, my - 1);
}

/**
 * @brief Computes the dynamic time warping distance using the light method.
 *
 * This function uses the light method for computation but cannot backtrack.
 * It only uses one vector to traverse instead of matrices.
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x First sequence.
 * @param y Second sequence.
 * @return The dynamic time warping distance.
 */
template <typename data_t>
data_t dtwFull_L(const std::vector<data_t> &x, const std::vector<data_t> &y,
                 data_t early_abandon = -1)
{
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  if (x.empty() || y.empty()) return maxValue;
  if (&x == &y) return 0; // If they are the same data then distance is 0.

  thread_local static std::vector<data_t> short_side;

  const auto &[short_vec, long_vec] = (x.size() < y.size()) ? std::tie(x, y) : std::tie(y, x);
  const auto m_short{ short_vec.size() }, m_long{ long_vec.size() };

  short_side.resize(m_short);

  auto distance = [](data_t x, data_t y) { return std::abs(x - y); };

  short_side[0] = distance(short_vec[0], long_vec[0]);

  for (size_t i = 1; i < m_short; i++)
    short_side[i] = short_side[i - 1] + distance(short_vec[i], long_vec[0]);

  for (size_t j = 1; j < m_long; j++) {
    auto diag = short_side[0];
    short_side[0] += distance(short_vec[0], long_vec[j]);

    for (size_t i = 1; i < m_short; i++) {
      const data_t min1 = std::min(short_side[i - 1], short_side[i]);
      const auto shr = short_vec[i];
      const auto lng = long_vec[j];
      const data_t dist = std::abs(shr - lng);
      const data_t next = std::min(diag, min1) + dist;

      diag = short_side[i];
      short_side[i] = next;
    }

    // Early abandon: if all values in the current row exceed the threshold,
    // the final DTW distance cannot be lower than the threshold.
    if (early_abandon >= 0) {
      data_t row_min = *std::min_element(short_side.begin(),
                                          short_side.begin() + static_cast<std::ptrdiff_t>(m_short));
      if (row_min > early_abandon) return maxValue;
    }
  }

  return short_side.back();
}


/**
 * @brief Computes the banded dynamic time warping distance between two sequences.
 *
 * @details This version of the algorithm introduces banding to limit the computation to
 * a certain vicinity around the diagonal, reducing computational complexity.
 *
 * Actual banding with skewness. Uses Sakoe-Chiba band.
 * Reference: H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization
 *            for spoken word recognition". IEEE Transactions on Acoustics,
 *            Speech, and Signal Processing, 26(1), 43-49 (1978).
 *
 * Code is inspired from pyts.
 * See https://pyts.readthedocs.io/en/stable/auto_examples/metrics/plot_sakoe_chiba.html
 * for a detailed explanation.
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x First sequence.
 * @param y Second sequence.
 * @param band The bandwidth parameter that controls the vicinity around the diagonal.
 * @return The dynamic time warping distance.
 */
template <typename data_t = double>
data_t dtwBanded(const std::vector<data_t> &x, const std::vector<data_t> &y, int band = settings::DEFAULT_BAND_LENGTH)
{
  if (band < 0) return dtwFull_L<data_t>(x, y); //<! Band is negative, so returning full dtw.

  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  const auto &[short_vec, long_vec] = (x.size() < y.size()) ? std::tie(x, y) : std::tie(y, x);
  const int m_short(short_vec.size()), m_long(long_vec.size());

  auto distance = [](data_t xi, data_t yi) { return std::abs(xi - yi); };

  if ((m_short == 0) || (m_long == 0)) return maxValue;
  if ((m_short == 1) || (m_long == 1)) return dtwFull_L<data_t>(x, y); //<! Band is meaningless when one length is one, so return full DTW
  if (m_long <= (band + 1)) return dtwFull_L<data_t>(x, y);            //<! Band is bigger than long side so full DTW can be done.


  const double slope = static_cast<double>(m_long - 1) / (m_short - 1);
  const auto window = std::max(static_cast<double>(band), slope / 2);

  auto get_bounds = [slope, window](int x) {
    const auto y = slope * x;
    const int low = std::ceil(std::round(100 * (y - window)) / 100.0);
    const int high = std::floor(std::round(100 * (y + window)) / 100.0) + 1;
    return std::pair(low, high);
  };

  // Rolling buffer: col[i] stores C(i, current_j).
  // Uses O(m_long) memory instead of O(m_long * m_short).
  thread_local std::vector<data_t> col;
  col.assign(m_long, maxValue);

  // Initialize column j=0.
  col[0] = distance(long_vec[0], short_vec[0]);
  {
    const auto [lo, hi] = get_bounds(0);
    for (int i = 1; i < std::min(hi, m_long); ++i)
      col[i] = col[i - 1] + distance(long_vec[i], short_vec[0]);
  }

  // Process columns j=1..m_short-1.
  for (int j = 1; j < m_short; j++) //<! Scan the short part!
  {
    const auto [lo, hi] = get_bounds(j);
    const auto [prev_lo, prev_hi] = get_bounds(j - 1);
    const int high = std::min(hi, m_long);
    const int low = std::max(lo, 0);

    // diag holds C(i-1, j-1) -- the old value of col[i-1] before it was
    // overwritten in the current column sweep.
    data_t diag = maxValue;

    // Capture the diagonal for the first row in the band.
    // C(first_row-1, j-1) is needed as diag when processing row i=first_row.
    const int first_row = std::max(low, 1);
    if (first_row - 1 >= std::max(prev_lo, 0) && first_row - 1 < std::min(prev_hi, m_long)) {
      diag = col[first_row - 1]; // Save before invalidation may overwrite it.
    }

    // Handle row i=0 specially when it falls inside the band.
    if (low == 0) {
      diag = col[0];                                             // C(0, j-1)
      col[0] = col[0] + distance(long_vec[0], short_vec[j]);    // C(0,j) = C(0,j-1) + dist
    }

    // Invalidate rows that were in the previous column's band but are now
    // below the current band's lower bound. They hold stale values that must
    // not leak into the min() calculation via the "left" (C(i,j-1)) term.
    for (int i = std::max(prev_lo, 0); i < std::min(low, std::min(prev_hi, m_long)); ++i)
      col[i] = maxValue;

    for (int i = first_row; i < high; ++i) {
      const data_t old_col_i = col[i]; // C(i, j-1)

      // C(i-1, j) is col[i-1] (already updated for current column j).
      // C(i, j-1) is old_col_i (not yet updated).
      // C(i-1, j-1) is diag.
      const auto minimum = std::min({ col[i - 1], old_col_i, diag });
      diag = old_col_i;
      col[i] = minimum + distance(long_vec[i], short_vec[j]);
    }

    // Invalidate rows above the band that held values from the previous column.
    for (int i = std::max(high, std::max(prev_lo, 0)); i < std::min(prev_hi, m_long); ++i)
      col[i] = maxValue;
  }

  return col[m_long - 1];
}
} // namespace dtwc
