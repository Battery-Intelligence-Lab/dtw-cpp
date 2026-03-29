/**
 * @file warping.hpp
 * @brief Time warping functions.
 *
 * @details This file contains functions for dynamic time warping, which is a method to
 * compare two temporal sequences that may vary in time or speed. It includes
 * different versions of the algorithm for full, light (L), and banded computations.
 *
 * Each public function accepts an optional core::MetricType parameter (default L1).
 * Metric dispatch happens ONCE outside the inner loop via a template lambda
 * passed to an _impl helper, so inlining is preserved and there is zero overhead
 * in the hot path.
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 08 Dec 2022
 */

#pragma once

#include "settings.hpp"           // for DEFAULT_BAND_LENGTH
#include "core/scratch_matrix.hpp"
#include "core/dtw_options.hpp"    // for core::MetricType

#include <cstdlib>   // for abs, size_t
#include <algorithm> // for min, max
#include <cmath>     // for floor, round
#include <limits>    // for numeric_limits
#include <vector>    // for vector
#include <utility>   // for pair

namespace dtwc {

// =========================================================================
//  Implementation helpers (take a distance callable as template parameter)
// =========================================================================

namespace detail {

/// Full-matrix DTW (O(n*m) space). Supports backtracking.
template <typename data_t, typename DistFn>
data_t dtwFull_impl(const std::vector<data_t> &x, const std::vector<data_t> &y,
                    DistFn distance)
{
  thread_local core::ScratchMatrix<data_t> C;
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  const int mx = x.size();
  const int my = y.size();

  if ((mx == 0) || (my == 0)) return maxValue;
  if (&x == &y) return 0;

  C.resize(mx, my);

  C(0, 0) = distance(x[0], y[0]);

  for (int i = 1; i < mx; i++)
    C(i, 0) = C(i - 1, 0) + distance(x[i], y[0]);

  for (int j = 1; j < my; j++)
    C(0, j) = C(0, j - 1) + distance(x[0], y[j]);

  for (int j = 1; j < my; j++) {
    for (int i = 1; i < mx; i++) {
      const auto minimum = std::min(C(i - 1, j - 1), std::min(C(i - 1, j), C(i, j - 1)));
      C(i, j) = minimum + distance(x[i], y[j]);
    }
  }

  return C(mx - 1, my - 1);
}

/// Linear-space DTW (O(min(n,m)) space, no backtracking).
/// Supports early abandon when early_abandon >= 0.
template <typename data_t, typename DistFn>
data_t dtwFull_L_impl(const std::vector<data_t> &x, const std::vector<data_t> &y,
                      data_t early_abandon, DistFn distance)
{
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  if (x.empty() || y.empty()) return maxValue;
  if (&x == &y) return 0;

  thread_local static std::vector<data_t> short_side;

  const auto &[short_vec, long_vec] = (x.size() < y.size()) ? std::tie(x, y) : std::tie(y, x);
  const auto m_short{ short_vec.size() }, m_long{ long_vec.size() };

  short_side.resize(m_short);

  short_side[0] = distance(short_vec[0], long_vec[0]);

  for (size_t i = 1; i < m_short; i++)
    short_side[i] = short_side[i - 1] + distance(short_vec[i], long_vec[0]);

  const bool do_early_abandon = (early_abandon >= 0);

  for (size_t j = 1; j < m_long; j++) {
    auto diag = short_side[0];
    short_side[0] += distance(short_vec[0], long_vec[j]);

    data_t row_min = do_early_abandon ? short_side[0] : data_t{0};

    for (size_t i = 1; i < m_short; i++) {
      const data_t min1 = std::min(short_side[i - 1], short_side[i]);
      const data_t dist = distance(short_vec[i], long_vec[j]);
      const data_t next = std::min(diag, min1) + dist;

      diag = short_side[i];
      short_side[i] = next;
      if (do_early_abandon) row_min = std::min(row_min, next);
    }

    if (do_early_abandon && row_min > early_abandon) return maxValue;
  }

  return short_side.back();
}

/// Sakoe-Chiba banded DTW.
template <typename data_t, typename DistFn>
data_t dtwBanded_impl(const std::vector<data_t> &x, const std::vector<data_t> &y,
                      int band, data_t early_abandon, DistFn distance)
{
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  const auto &[short_vec, long_vec] = (x.size() < y.size()) ? std::tie(x, y) : std::tie(y, x);
  const int m_short(short_vec.size()), m_long(long_vec.size());

  if ((m_short == 0) || (m_long == 0)) return maxValue;

  const double slope = static_cast<double>(m_long - 1) / (m_short - 1);
  const auto window = std::max(static_cast<double>(band), slope / 2);

  auto get_bounds = [slope, window](int x) {
    const auto y = slope * x;
    const int low = std::ceil(std::round(100 * (y - window)) / 100.0);
    const int high = std::floor(std::round(100 * (y + window)) / 100.0) + 1;
    return std::pair(low, high);
  };

  thread_local std::vector<data_t> col;
  col.assign(m_long, maxValue);

  col[0] = distance(long_vec[0], short_vec[0]);
  {
    const auto [lo, hi] = get_bounds(0);
    for (int i = 1; i < std::min(hi, m_long); ++i)
      col[i] = col[i - 1] + distance(long_vec[i], short_vec[0]);
  }

  for (int j = 1; j < m_short; j++) {
    const auto [lo, hi] = get_bounds(j);
    const auto [prev_lo, prev_hi] = get_bounds(j - 1);
    const int high = std::min(hi, m_long);
    const int low = std::max(lo, 0);

    data_t diag = maxValue;

    const int first_row = std::max(low, 1);
    if (first_row - 1 >= std::max(prev_lo, 0) && first_row - 1 < std::min(prev_hi, m_long)) {
      diag = col[first_row - 1];
    }

    if (low == 0) {
      diag = col[0];
      col[0] = col[0] + distance(long_vec[0], short_vec[j]);
    }

    for (int i = std::max(prev_lo, 0); i < std::min(low, std::min(prev_hi, m_long)); ++i)
      col[i] = maxValue;

    for (int i = first_row; i < high; ++i) {
      const data_t old_col_i = col[i];
      const auto minimum = std::min(diag, std::min(col[i - 1], old_col_i));
      diag = old_col_i;
      col[i] = minimum + distance(long_vec[i], short_vec[j]);
    }

    for (int i = std::max(high, std::max(prev_lo, 0)); i < std::min(prev_hi, m_long); ++i)
      col[i] = maxValue;
  }

  return col[m_long - 1];
}

} // namespace detail

// =========================================================================
//  Metric dispatch helpers
// =========================================================================

namespace detail {

/// L1 (absolute difference) pointwise distance.
struct L1Dist {
  template <typename T>
  T operator()(T a, T b) const { return std::abs(a - b); }
};

/// Squared-L2 pointwise distance: (a - b)^2.
struct SquaredL2Dist {
  template <typename T>
  T operator()(T a, T b) const { auto d = a - b; return d * d; }
};

} // namespace detail

// =========================================================================
//  Public API
// =========================================================================

/**
 * @brief Computes the full dynamic time warping distance between two sequences.
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x First sequence.
 * @param y Second sequence.
 * @param metric Pointwise distance metric (default: L1).
 * @return The dynamic time warping distance.
 */
template <typename data_t>
data_t dtwFull(const std::vector<data_t> &x, const std::vector<data_t> &y,
               core::MetricType metric = core::MetricType::L1)
{
  switch (metric) {
  case core::MetricType::SquaredL2:
    return detail::dtwFull_impl(x, y, detail::SquaredL2Dist{});
  case core::MetricType::L2:  // L2 on scalars is same as L1
  case core::MetricType::L1:
  default:
    return detail::dtwFull_impl(x, y, detail::L1Dist{});
  }
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
 * @param early_abandon Threshold for early abandon; negative disables.
 * @param metric Pointwise distance metric (default: L1).
 * @return The dynamic time warping distance.
 */
template <typename data_t>
data_t dtwFull_L(const std::vector<data_t> &x, const std::vector<data_t> &y,
                 data_t early_abandon = -1,
                 core::MetricType metric = core::MetricType::L1)
{
  switch (metric) {
  case core::MetricType::SquaredL2:
    return detail::dtwFull_L_impl(x, y, early_abandon, detail::SquaredL2Dist{});
  case core::MetricType::L2:
  case core::MetricType::L1:
  default:
    return detail::dtwFull_L_impl(x, y, early_abandon, detail::L1Dist{});
  }
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
 * @param early_abandon Threshold for early abandon; negative disables (only used in full fallback).
 * @param metric Pointwise distance metric (default: L1).
 * @return The dynamic time warping distance.
 */
template <typename data_t = double>
data_t dtwBanded(const std::vector<data_t> &x, const std::vector<data_t> &y,
                 int band = settings::DEFAULT_BAND_LENGTH,
                 data_t early_abandon = -1,
                 core::MetricType metric = core::MetricType::L1)
{
  if (band < 0) return dtwFull_L<data_t>(x, y, early_abandon, metric);

  const auto &[short_vec, long_vec] = (x.size() < y.size()) ? std::tie(x, y) : std::tie(y, x);
  const int m_short(short_vec.size()), m_long(long_vec.size());

  if ((m_short == 0) || (m_long == 0)) return std::numeric_limits<data_t>::max();
  if ((m_short == 1) || (m_long == 1)) return dtwFull_L<data_t>(x, y, early_abandon, metric);
  if (m_long <= (band + 1)) return dtwFull_L<data_t>(x, y, early_abandon, metric);

  switch (metric) {
  case core::MetricType::SquaredL2:
    return detail::dtwBanded_impl(x, y, band, early_abandon, detail::SquaredL2Dist{});
  case core::MetricType::L2:
  case core::MetricType::L1:
  default:
    return detail::dtwBanded_impl(x, y, band, early_abandon, detail::L1Dist{});
  }
}

} // namespace dtwc
