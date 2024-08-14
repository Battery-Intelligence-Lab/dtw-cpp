/**
 * @file warping.hpp
 * @brief Time warping functions.
 *
 * This file contains functions for dynamic time warping, which is a method to
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

#include <armadillo>

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
  thread_local arma::Mat<data_t> C;
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  if (&x == &y) return 0; // If they are the same data then distance is 0.

  const int mx = x.size();
  const int my = y.size();

  C.resize(mx, my);

  auto distance = [](data_t x, data_t y) { return std::abs(x - y); };

  if ((mx == 0) || (my == 0)) return maxValue;

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
data_t dtwFull_L(const std::vector<data_t> &x, const std::vector<data_t> &y)
{
  if (&x == &y) return 0; // If they are the same data then distance is 0.
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();
  thread_local static std::vector<data_t> short_side(10000);

  const auto &[short_vec, long_vec] = (x.size() < y.size()) ? std::tie(x, y) : std::tie(y, x);
  const auto m_short{ short_vec.size() }, m_long{ long_vec.size() };

  short_side.resize(m_short);

  auto distance = [](data_t x, data_t y) { return std::abs(x - y); };

  if ((m_short == 0) || (m_long == 0)) return maxValue;

  short_side[0] = distance(short_vec[0], long_vec[0]);

  for (size_t i = 1; i < m_short; i++)
    short_side[i] = short_side[i - 1] + distance(short_vec[i], long_vec[0]);

  for (size_t j = 1; j < m_long; j++) {
    auto diag = short_side[0];
    short_side[0] += distance(short_vec[0], long_vec[j]);

    for (size_t i = 1; i < m_short; i++) {
      const data_t min1 = std::min(short_side[i - 1], short_side[i]);
      const auto shr = short_vec.at(i);
      const auto lng = long_vec.at(j);
      const data_t dist = std::abs(shr - lng);
      const data_t next = std::min(diag, min1) + dist;

      diag = short_side[i];
      short_side[i] = next;
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
template <typename data_t = float>
data_t dtwBanded(const std::vector<data_t> &x, const std::vector<data_t> &y, int band = settings::DEFAULT_BAND_LENGTH)
{
  if (band < 0) return dtwFull_L<data_t>(x, y); //<! Band is negative, so returning full dtw.

  thread_local arma::Mat<data_t> C;
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  const auto &[short_vec, long_vec] = (x.size() < y.size()) ? std::tie(x, y) : std::tie(y, x);
  const int m_short(short_vec.size()), m_long(long_vec.size());

  C.resize(m_long, m_short);
  C.fill(maxValue);

  auto distance = [](data_t xi, data_t yi) { return std::abs(xi - yi); };

  if ((m_short == 0) || (m_long == 0)) return maxValue;
  if ((m_short == 1) || (m_long == 1)) return dtwFull_L<data_t>(x, y); //<! Band is meaningless when one length is one, so return full DTW
  if (m_long <= (band + 1)) return dtwFull_L<data_t>(x, y);            //<! Band is bigger than long side so full DTW can be done.


  const double slope = static_cast<double>(m_long - 1) / (m_short - 1);
  const auto window = std::max((double)band, slope / 2);

  auto get_bounds = [slope, window](int x) {
    const auto y = slope * x;
    const int low = std::ceil(std::round(100 * (y - window)) / 100.0);
    const int high = std::floor(std::round(100 * (y + window)) / 100.0) + 1;
    return std::pair(low, high);
  };

  C(0, 0) = distance(long_vec[0], short_vec[0]);

  {
    const auto [lo, hi] = get_bounds(0);
    for (int i = 1; i < hi; ++i)
      C(i, 0) = C(i - 1, 0) + distance(long_vec[i], short_vec[0]);
  }

  for (int j = 1; j < m_short; j++) //<! Scan the short part!
  {
    const auto [lo, hi] = get_bounds(j);
    if (lo <= 0)
      C(0, j) = C(0, j - 1) + distance(long_vec[0], short_vec[j]);

    const auto high = std::min(hi, m_long);
    for (int i = std::max(lo, 1); i < high; ++i) {
      const auto minimum = std::min({ C(i - 1, j), C(i, j - 1), C(i - 1, j - 1) });
      C(i, j) = minimum + distance(long_vec[i], short_vec[j]);
    }
  }

  return C(m_long - 1, m_short - 1);
}
} // namespace dtwc