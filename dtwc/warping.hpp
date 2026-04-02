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
/// Accepts raw pointers + lengths for zero-copy binding support.
template <typename data_t, typename DistFn>
data_t dtwFull_impl(const data_t* x, size_t nx, const data_t* y, size_t ny,
                    DistFn distance)
{
  thread_local core::ScratchMatrix<data_t> C;
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  const int mx = static_cast<int>(nx);
  const int my = static_cast<int>(ny);

  if ((mx == 0) || (my == 0)) return maxValue;
  if (x == y && nx == ny) return 0;

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

/// Full-matrix DTW — vector overload (forwards to pointer version).
template <typename data_t, typename DistFn>
data_t dtwFull_impl(const std::vector<data_t> &x, const std::vector<data_t> &y,
                    DistFn distance)
{
  return dtwFull_impl(x.data(), x.size(), y.data(), y.size(), distance);
}

/// Linear-space DTW (O(min(n,m)) space, no backtracking).
/// Supports early abandon when early_abandon >= 0.
/// Accepts raw pointers + lengths for zero-copy binding support.
template <typename data_t, typename DistFn>
data_t dtwFull_L_impl(const data_t* x, size_t nx, const data_t* y, size_t ny,
                      data_t early_abandon, DistFn distance)
{
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  if (nx == 0 || ny == 0) return maxValue;
  if (x == y && nx == ny) return 0;

  thread_local static std::vector<data_t> short_side;

  const data_t* short_ptr;
  const data_t* long_ptr;
  size_t m_short, m_long;
  if (nx < ny) {
    short_ptr = x; m_short = nx;
    long_ptr  = y; m_long  = ny;
  } else {
    short_ptr = y; m_short = ny;
    long_ptr  = x; m_long  = nx;
  }

  short_side.resize(m_short);

  short_side[0] = distance(short_ptr[0], long_ptr[0]);

  for (size_t i = 1; i < m_short; i++)
    short_side[i] = short_side[i - 1] + distance(short_ptr[i], long_ptr[0]);

  const bool do_early_abandon = (early_abandon >= 0);

  for (size_t j = 1; j < m_long; j++) {
    auto diag = short_side[0];
    short_side[0] += distance(short_ptr[0], long_ptr[j]);

    data_t row_min = do_early_abandon ? short_side[0] : data_t{0};

    for (size_t i = 1; i < m_short; i++) {
      const data_t min1 = std::min(short_side[i - 1], short_side[i]);
      const data_t dist = distance(short_ptr[i], long_ptr[j]);
      const data_t next = std::min(diag, min1) + dist;

      diag = short_side[i];
      short_side[i] = next;
      if (do_early_abandon) row_min = std::min(row_min, next);
    }

    if (do_early_abandon && row_min > early_abandon) return maxValue;
  }

  return short_side.back();
}

/// Linear-space DTW — vector overload (forwards to pointer version).
template <typename data_t, typename DistFn>
data_t dtwFull_L_impl(const std::vector<data_t> &x, const std::vector<data_t> &y,
                      data_t early_abandon, DistFn distance)
{
  return dtwFull_L_impl(x.data(), x.size(), y.data(), y.size(), early_abandon, distance);
}

/// Sakoe-Chiba banded DTW.
/// Accepts raw pointers + lengths for zero-copy binding support.
template <typename data_t, typename DistFn>
data_t dtwBanded_impl(const data_t* x, size_t nx, const data_t* y, size_t ny,
                      int band, data_t early_abandon, DistFn distance)
{
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  const data_t* short_ptr;
  const data_t* long_ptr;
  int m_short, m_long;
  if (nx < ny) {
    short_ptr = x; m_short = static_cast<int>(nx);
    long_ptr  = y; m_long  = static_cast<int>(ny);
  } else {
    short_ptr = y; m_short = static_cast<int>(ny);
    long_ptr  = x; m_long  = static_cast<int>(nx);
  }

  if ((m_short == 0) || (m_long == 0)) return maxValue;

  const double slope = static_cast<double>(m_long - 1) / (m_short - 1);
  const auto window = std::max(static_cast<double>(band), slope / 2);

  thread_local std::vector<data_t> col;
  col.assign(m_long, maxValue);
  thread_local std::vector<int> low_bounds;
  thread_local std::vector<int> high_bounds;
  low_bounds.resize(m_short);
  high_bounds.resize(m_short);
  const bool do_early_abandon = (early_abandon >= 0);

  for (int row = 0; row < m_short; ++row) {
    const double center = slope * row;
    low_bounds[row] = static_cast<int>(
      std::ceil(std::round(100.0 * (center - window)) / 100.0));
    high_bounds[row] = static_cast<int>(
      std::floor(std::round(100.0 * (center + window)) / 100.0)) + 1;
  }

  col[0] = distance(long_ptr[0], short_ptr[0]);
  {
    const int hi = high_bounds[0];
    for (int i = 1; i < std::min(hi, m_long); ++i)
      col[i] = col[i - 1] + distance(long_ptr[i], short_ptr[0]);
  }
  if (do_early_abandon && col[0] > early_abandon) return maxValue;

  for (int j = 1; j < m_short; j++) {
    const int lo = low_bounds[j];
    const int hi = high_bounds[j];
    const int prev_lo = low_bounds[j - 1];
    const int prev_hi = high_bounds[j - 1];
    const int high = std::min(hi, m_long);
    const int low = std::max(lo, 0);

    data_t diag = maxValue;
    data_t row_min = do_early_abandon ? maxValue : data_t{0};

    const int first_row = std::max(low, 1);
    if (first_row - 1 >= std::max(prev_lo, 0) && first_row - 1 < std::min(prev_hi, m_long)) {
      diag = col[first_row - 1];
    }

    if (low == 0) {
      diag = col[0];
      col[0] = col[0] + distance(long_ptr[0], short_ptr[j]);
      if (do_early_abandon) row_min = col[0];
    }

    for (int i = std::max(prev_lo, 0); i < std::min(low, std::min(prev_hi, m_long)); ++i)
      col[i] = maxValue;

    for (int i = first_row; i < high; ++i) {
      const data_t old_col_i = col[i];
      const auto minimum = std::min(diag, std::min(col[i - 1], old_col_i));
      diag = old_col_i;
      col[i] = minimum + distance(long_ptr[i], short_ptr[j]);
      if (do_early_abandon) row_min = std::min(row_min, col[i]);
    }

    for (int i = std::max(high, std::max(prev_lo, 0)); i < std::min(prev_hi, m_long); ++i)
      col[i] = maxValue;

    if (do_early_abandon && row_min > early_abandon) return maxValue;
  }

  return col[m_long - 1];
}

/// Sakoe-Chiba banded DTW — vector overload (forwards to pointer version).
template <typename data_t, typename DistFn>
data_t dtwBanded_impl(const std::vector<data_t> &x, const std::vector<data_t> &y,
                      int band, data_t early_abandon, DistFn distance)
{
  return dtwBanded_impl(x.data(), x.size(), y.data(), y.size(), band, early_abandon, distance);
}

/// Linear-space multivariate DTW.
/// Each timestep has ndim features in interleaved layout: [t0_f0, t0_f1, ..., t1_f0, ...].
/// distance(const T*, const T*, size_t ndim) -> T
template <typename data_t, typename DistFn>
data_t dtwFull_L_mv_impl(const data_t* x, size_t nx_steps, const data_t* y, size_t ny_steps,
                          size_t ndim, data_t early_abandon, DistFn distance)
{
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();
  if (nx_steps == 0 || ny_steps == 0) return maxValue;
  if (x == y && nx_steps == ny_steps) return 0;

  thread_local static std::vector<data_t> short_side;

  const data_t* short_ptr;
  const data_t* long_ptr;
  size_t m_short, m_long;
  if (nx_steps < ny_steps) {
    short_ptr = x; m_short = nx_steps;
    long_ptr  = y; m_long  = ny_steps;
  } else {
    short_ptr = y; m_short = ny_steps;
    long_ptr  = x; m_long  = nx_steps;
  }

  short_side.resize(m_short);
  short_side[0] = distance(short_ptr, long_ptr, ndim);

  for (size_t i = 1; i < m_short; i++)
    short_side[i] = short_side[i - 1] + distance(short_ptr + i * ndim, long_ptr, ndim);

  const bool do_ea = (early_abandon >= 0);

  for (size_t j = 1; j < m_long; j++) {
    const data_t* long_j = long_ptr + j * ndim;
    auto diag = short_side[0];
    short_side[0] += distance(short_ptr, long_j, ndim);
    data_t row_min = do_ea ? short_side[0] : data_t{0};

    for (size_t i = 1; i < m_short; i++) {
      const data_t min1 = std::min(short_side[i - 1], short_side[i]);
      const data_t dist = distance(short_ptr + i * ndim, long_j, ndim);
      const data_t next = std::min(diag, min1) + dist;
      diag = short_side[i];
      short_side[i] = next;
      if (do_ea) row_min = std::min(row_min, next);
    }
    if (do_ea && row_min > early_abandon) return maxValue;
  }
  return short_side.back();
}

/// Sakoe-Chiba banded multivariate DTW.
/// Each timestep has ndim features in interleaved layout.
/// distance(const T*, const T*, size_t ndim) -> T
template <typename data_t, typename DistFn>
data_t dtwBanded_mv_impl(const data_t* x, size_t nx, const data_t* y, size_t ny,
                          size_t ndim, int band, data_t early_abandon, DistFn distance)
{
  constexpr data_t maxValue = std::numeric_limits<data_t>::max();

  const data_t* short_ptr;
  const data_t* long_ptr;
  int m_short, m_long;
  if (nx < ny) {
    short_ptr = x; m_short = static_cast<int>(nx);
    long_ptr  = y; m_long  = static_cast<int>(ny);
  } else {
    short_ptr = y; m_short = static_cast<int>(ny);
    long_ptr  = x; m_long  = static_cast<int>(nx);
  }

  if ((m_short == 0) || (m_long == 0)) return maxValue;

  const double slope = static_cast<double>(m_long - 1) / (m_short - 1);
  const auto window = std::max(static_cast<double>(band), slope / 2);

  thread_local std::vector<data_t> col_mv;
  col_mv.assign(m_long, maxValue);
  thread_local std::vector<int> low_bounds_mv;
  thread_local std::vector<int> high_bounds_mv;
  low_bounds_mv.resize(m_short);
  high_bounds_mv.resize(m_short);
  const bool do_early_abandon = (early_abandon >= 0);

  for (int row = 0; row < m_short; ++row) {
    const double center = slope * row;
    low_bounds_mv[row] = static_cast<int>(
      std::ceil(std::round(100.0 * (center - window)) / 100.0));
    high_bounds_mv[row] = static_cast<int>(
      std::floor(std::round(100.0 * (center + window)) / 100.0)) + 1;
  }

  col_mv[0] = distance(long_ptr, short_ptr, ndim);
  {
    const int hi = high_bounds_mv[0];
    for (int i = 1; i < std::min(hi, m_long); ++i)
      col_mv[i] = col_mv[i - 1] + distance(long_ptr + i * ndim, short_ptr, ndim);
  }
  if (do_early_abandon && col_mv[0] > early_abandon) return maxValue;

  for (int j = 1; j < m_short; j++) {
    const int lo = low_bounds_mv[j];
    const int hi = high_bounds_mv[j];
    const int prev_lo = low_bounds_mv[j - 1];
    const int prev_hi = high_bounds_mv[j - 1];
    const int high = std::min(hi, m_long);
    const int low = std::max(lo, 0);

    data_t diag = maxValue;
    data_t row_min = do_early_abandon ? maxValue : data_t{0};

    const int first_row = std::max(low, 1);
    if (first_row - 1 >= std::max(prev_lo, 0) && first_row - 1 < std::min(prev_hi, m_long)) {
      diag = col_mv[first_row - 1];
    }

    if (low == 0) {
      diag = col_mv[0];
      col_mv[0] = col_mv[0] + distance(long_ptr, short_ptr + j * ndim, ndim);
      if (do_early_abandon) row_min = col_mv[0];
    }

    for (int i = std::max(prev_lo, 0); i < std::min(low, std::min(prev_hi, m_long)); ++i)
      col_mv[i] = maxValue;

    for (int i = first_row; i < high; ++i) {
      const data_t old_col_i = col_mv[i];
      const auto minimum = std::min(diag, std::min(col_mv[i - 1], old_col_i));
      diag = old_col_i;
      col_mv[i] = minimum + distance(long_ptr + i * ndim, short_ptr + j * ndim, ndim);
      if (do_early_abandon) row_min = std::min(row_min, col_mv[i]);
    }

    for (int i = std::max(high, std::max(prev_lo, 0)); i < std::min(prev_hi, m_long); ++i)
      col_mv[i] = maxValue;

    if (do_early_abandon && row_min > early_abandon) return maxValue;
  }

  return col_mv[m_long - 1];
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

/// Multivariate L1: sum of |a[d] - b[d]| across ndim dimensions.
struct MVL1Dist {
  template <typename T>
  T operator()(const T* a, const T* b, size_t ndim) const noexcept {
    T sum = T(0);
    for (size_t d = 0; d < ndim; ++d)
      sum += std::abs(a[d] - b[d]);
    return sum;
  }
};

/// Multivariate Squared L2: sum of (a[d] - b[d])^2 across ndim dimensions.
struct MVSquaredL2Dist {
  template <typename T>
  T operator()(const T* a, const T* b, size_t ndim) const noexcept {
    T sum = T(0);
    for (size_t d = 0; d < ndim; ++d) {
      T diff = a[d] - b[d];
      sum += diff * diff;
    }
    return sum;
  }
};

} // namespace detail

// =========================================================================
//  Public API — pointer + length overloads (zero-copy for bindings)
// =========================================================================

/**
 * @brief Computes the full dynamic time warping distance (pointer + length).
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x Pointer to first sequence.
 * @param nx Length of first sequence.
 * @param y Pointer to second sequence.
 * @param ny Length of second sequence.
 * @param metric Pointwise distance metric (default: L1).
 * @return The dynamic time warping distance.
 */
template <typename data_t>
data_t dtwFull(const data_t* x, size_t nx, const data_t* y, size_t ny,
               core::MetricType metric = core::MetricType::L1)
{
  switch (metric) {
  case core::MetricType::SquaredL2:
    return detail::dtwFull_impl(x, nx, y, ny, detail::SquaredL2Dist{});
  case core::MetricType::L2:
    // L2 for 1D scalars: |a-b| (same as L1). For Euclidean DTW (sqrt of sum
    // of squared diffs, as in dtaidistance/tslearn), use SquaredL2 and take
    // sqrt of the result externally.
    return detail::dtwFull_impl(x, nx, y, ny, detail::L1Dist{});
  case core::MetricType::L1:
  default:
    return detail::dtwFull_impl(x, nx, y, ny, detail::L1Dist{});
  }
}

/**
 * @brief Computes the linear-space DTW distance (pointer + length).
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x Pointer to first sequence.
 * @param nx Length of first sequence.
 * @param y Pointer to second sequence.
 * @param ny Length of second sequence.
 * @param early_abandon Threshold for early abandon; negative disables.
 * @param metric Pointwise distance metric (default: L1).
 * @return The dynamic time warping distance.
 */
template <typename data_t>
data_t dtwFull_L(const data_t* x, size_t nx, const data_t* y, size_t ny,
                 data_t early_abandon = -1,
                 core::MetricType metric = core::MetricType::L1)
{
  switch (metric) {
  case core::MetricType::SquaredL2:
    return detail::dtwFull_L_impl(x, nx, y, ny, early_abandon, detail::SquaredL2Dist{});
  case core::MetricType::L2:  // L2 for 1D scalars == L1; for Euclidean DTW use SquaredL2 + sqrt
  case core::MetricType::L1:
  default:
    return detail::dtwFull_L_impl(x, nx, y, ny, early_abandon, detail::L1Dist{});
  }
}

/**
 * @brief Computes the banded DTW distance (pointer + length).
 *
 * @details Uses Sakoe-Chiba band. Falls back to dtwFull_L when band < 0.
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x Pointer to first sequence.
 * @param nx Length of first sequence.
 * @param y Pointer to second sequence.
 * @param ny Length of second sequence.
 * @param band The bandwidth parameter; negative means unconstrained.
 * @param early_abandon Threshold for early abandon; negative disables.
 * @param metric Pointwise distance metric (default: L1).
 * @return The dynamic time warping distance.
 */
template <typename data_t = double>
data_t dtwBanded(const data_t* x, size_t nx, const data_t* y, size_t ny,
                 int band = settings::DEFAULT_BAND_LENGTH,
                 data_t early_abandon = -1,
                 core::MetricType metric = core::MetricType::L1)
{
  if (band < 0) return dtwFull_L<data_t>(x, nx, y, ny, early_abandon, metric);

  const size_t min_sz = std::min(nx, ny);
  const size_t max_sz = std::max(nx, ny);
  const int m_short = static_cast<int>(min_sz);
  const int m_long  = static_cast<int>(max_sz);

  if ((m_short == 0) || (m_long == 0)) return std::numeric_limits<data_t>::max();
  if ((m_short == 1) || (m_long == 1)) return dtwFull_L<data_t>(x, nx, y, ny, early_abandon, metric);
  if (m_long <= (band + 1)) return dtwFull_L<data_t>(x, nx, y, ny, early_abandon, metric);

  switch (metric) {
  case core::MetricType::SquaredL2:
    return detail::dtwBanded_impl(x, nx, y, ny, band, early_abandon, detail::SquaredL2Dist{});
  case core::MetricType::L2:  // L2 for 1D scalars == L1; for Euclidean DTW use SquaredL2 + sqrt
  case core::MetricType::L1:
  default:
    return detail::dtwBanded_impl(x, nx, y, ny, band, early_abandon, detail::L1Dist{});
  }
}

// =========================================================================
//  Public API — vector overloads (forward to pointer versions)
// =========================================================================

/// Full-matrix DTW (vector overload).
template <typename data_t>
data_t dtwFull(const std::vector<data_t> &x, const std::vector<data_t> &y,
               core::MetricType metric = core::MetricType::L1)
{
  return dtwFull<data_t>(x.data(), x.size(), y.data(), y.size(), metric);
}

/// Linear-space DTW (vector overload).
template <typename data_t>
data_t dtwFull_L(const std::vector<data_t> &x, const std::vector<data_t> &y,
                 data_t early_abandon = -1,
                 core::MetricType metric = core::MetricType::L1)
{
  return dtwFull_L<data_t>(x.data(), x.size(), y.data(), y.size(), early_abandon, metric);
}

/// Banded DTW (vector overload).
template <typename data_t = double>
data_t dtwBanded(const std::vector<data_t> &x, const std::vector<data_t> &y,
                 int band = settings::DEFAULT_BAND_LENGTH,
                 data_t early_abandon = -1,
                 core::MetricType metric = core::MetricType::L1)
{
  return dtwBanded<data_t>(x.data(), x.size(), y.data(), y.size(), band, early_abandon, metric);
}

// =========================================================================
//  Public API — multivariate (interleaved layout: [t0_f0, t0_f1, ..., t1_f0, ...])
// =========================================================================

/**
 * @brief Linear-space multivariate DTW (pointer + timestep counts + ndim).
 *
 * @details For ndim==1 dispatches to the existing scalar dtwFull_L (zero overhead).
 *          For ndim>1 uses dtwFull_L_mv_impl with pointer-stride indexing.
 *          Input layout is interleaved: x[t * ndim + d] is feature d at timestep t.
 *
 * @tparam data_t Data type of the elements.
 * @param x       Pointer to first series (interleaved, nx_steps * ndim elements).
 * @param nx_steps Number of timesteps in x.
 * @param y       Pointer to second series (interleaved, ny_steps * ndim elements).
 * @param ny_steps Number of timesteps in y.
 * @param ndim    Number of features per timestep.
 * @param early_abandon Threshold for early abandon; negative disables.
 * @param metric  Pointwise distance metric (default: L1).
 * @return The dynamic time warping distance.
 */
template <typename data_t = double>
data_t dtwFull_L_mv(const data_t* x, size_t nx_steps, const data_t* y, size_t ny_steps,
                    size_t ndim, data_t early_abandon = -1,
                    core::MetricType metric = core::MetricType::L1)
{
  if (ndim == 1) return dtwFull_L(x, nx_steps, y, ny_steps, early_abandon, metric);
  switch (metric) {
  case core::MetricType::SquaredL2:
    return detail::dtwFull_L_mv_impl(x, nx_steps, y, ny_steps, ndim, early_abandon,
                                      detail::MVSquaredL2Dist{});
  case core::MetricType::L2:
  case core::MetricType::L1:
  default:
    return detail::dtwFull_L_mv_impl(x, nx_steps, y, ny_steps, ndim, early_abandon,
                                      detail::MVL1Dist{});
  }
}

/**
 * @brief Banded multivariate DTW (pointer + timestep counts + ndim).
 *
 * @details For ndim==1 dispatches to existing scalar dtwBanded (zero overhead).
 *          For ndim>1 uses dtwBanded_mv_impl with pointer-stride indexing.
 *          Falls back to dtwFull_L_mv when band<0 or band covers the full sequence.
 *          Input layout is interleaved: x[t * ndim + d] is feature d at timestep t.
 *
 * @tparam data_t Data type of the elements.
 * @param x       Pointer to first series (interleaved).
 * @param nx_steps Number of timesteps in x.
 * @param y       Pointer to second series (interleaved).
 * @param ny_steps Number of timesteps in y.
 * @param ndim    Number of features per timestep.
 * @param band    Sakoe-Chiba band width; negative means unconstrained.
 * @param early_abandon Threshold for early abandon; negative disables.
 * @param metric  Pointwise distance metric (default: L1).
 * @return The dynamic time warping distance.
 */
template <typename data_t = double>
data_t dtwBanded_mv(const data_t* x, size_t nx_steps, const data_t* y, size_t ny_steps,
                    size_t ndim, int band = settings::DEFAULT_BAND_LENGTH,
                    data_t early_abandon = -1,
                    core::MetricType metric = core::MetricType::L1)
{
  if (band < 0) return dtwFull_L_mv(x, nx_steps, y, ny_steps, ndim, early_abandon, metric);
  if (ndim == 1) return dtwBanded(x, nx_steps, y, ny_steps, band, early_abandon, metric);

  const size_t min_sz = std::min(nx_steps, ny_steps);
  const size_t max_sz = std::max(nx_steps, ny_steps);
  if (min_sz == 0 || max_sz == 0) return std::numeric_limits<data_t>::max();
  if (min_sz == 1 || max_sz == 1 || static_cast<int>(max_sz) <= band + 1)
    return dtwFull_L_mv(x, nx_steps, y, ny_steps, ndim, early_abandon, metric);

  switch (metric) {
  case core::MetricType::SquaredL2:
    return detail::dtwBanded_mv_impl(x, nx_steps, y, ny_steps, ndim, band, early_abandon,
                                      detail::MVSquaredL2Dist{});
  case core::MetricType::L2:
  case core::MetricType::L1:
  default:
    return detail::dtwBanded_mv_impl(x, nx_steps, y, ny_steps, ndim, band, early_abandon,
                                      detail::MVL1Dist{});
  }
}

} // namespace dtwc
