/**
 * @file warping_missing.hpp
 * @brief DTW with missing data support (NaN-aware, DTW-AROW).
 *
 * @details Handles missing values (NaN) in time series by treating pairs where
 * one or both values are NaN as zero-cost, so missing data does not artificially
 * inflate or deflate the distance.  The recurrence is identical to standard DTW:
 *
 *   C(i,j) = cost(x[i], y[j]) + min(C(i-1,j-1), C(i-1,j), C(i,j-1))
 *
 * where cost(a, b) = 0 if either a or b is NaN, otherwise the normal pointwise
 * distance (L1 or squared-L2).
 *
 * Reference: Yurtman, A., Soenen, J., Meert, W. & Blockeel, H. (2023).
 *            "Estimating DTW Distance Between Time Series with Missing Data."
 *            ECML-PKDD 2023, LNCS 14173.
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 */

#pragma once

#include "settings.hpp"
#include "warping.hpp"           // for dtwFull_L (fallback)
#include "core/scratch_matrix.hpp"
#include "core/dtw_options.hpp"  // for core::MetricType
#include "missing_utils.hpp"     // for is_missing() — bitwise NaN, safe under -ffast-math

#include <cstdlib>   // for abs, size_t
#include <algorithm> // for min, max
#include <cmath>     // for isnan, abs, ceil, floor, round
#include <limits>    // for numeric_limits
#include <vector>    // for vector
#include <span>      // for span
#include <utility>   // for pair, tie

namespace dtwc {

// =========================================================================
//  NaN-aware distance functors
// =========================================================================

namespace detail {

/// L1 distance that returns 0 when either operand is NaN.
struct MissingL1Dist {
  template <typename T>
  T operator()(T a, T b) const {
    if (is_missing(a) || is_missing(b)) return T(0);
    return std::abs(a - b);
  }
};

/// Squared-L2 distance that returns 0 when either operand is NaN.
struct MissingSquaredL2Dist {
  template <typename T>
  T operator()(T a, T b) const {
    if (is_missing(a) || is_missing(b)) return T(0);
    auto d = a - b;
    return d * d;
  }
};

/// Multivariate missing-aware L1: sum of |a[d]-b[d]| skipping NaN channels.
struct MissingMVL1Dist {
  template <typename T>
  T operator()(const T* a, const T* b, size_t ndim) const noexcept {
    T sum = T(0);
    for (size_t d = 0; d < ndim; ++d) {
      if (is_missing(a[d]) || is_missing(b[d])) continue;
      sum += std::abs(a[d] - b[d]);
    }
    return sum;
  }
};

/// Multivariate missing-aware SquaredL2: sum of (a[d]-b[d])^2 skipping NaN channels.
struct MissingMVSquaredL2Dist {
  template <typename T>
  T operator()(const T* a, const T* b, size_t ndim) const noexcept {
    T sum = T(0);
    for (size_t d = 0; d < ndim; ++d) {
      if (is_missing(a[d]) || is_missing(b[d])) continue;
      T diff = a[d] - b[d];
      sum += diff * diff;
    }
    return sum;
  }
};

/// Dispatch MetricType to missing-aware scalar distance functor.
template <typename Fn>
auto dispatch_missing_metric(core::MetricType m, Fn&& fn) -> decltype(fn(MissingL1Dist{}))
{
  switch (m) {
  case core::MetricType::SquaredL2: return fn(MissingSquaredL2Dist{});
  case core::MetricType::L2:
  case core::MetricType::L1:
  default: return fn(MissingL1Dist{});
  }
}

/// Dispatch MetricType to missing-aware multivariate distance functor.
template <typename Fn>
auto dispatch_missing_mv_metric(core::MetricType m, Fn&& fn) -> decltype(fn(MissingMVL1Dist{}))
{
  switch (m) {
  case core::MetricType::SquaredL2: return fn(MissingMVSquaredL2Dist{});
  case core::MetricType::L2:
  case core::MetricType::L1:
  default: return fn(MissingMVL1Dist{});
  }
}

} // namespace detail

// =========================================================================
//  Public API — DTW with missing data
// =========================================================================

/**
 * @brief Computes DTW distance with missing data support (linear space).
 *
 * @details Missing values are represented as NaN.  Pairs where either value
 * is NaN contribute zero cost, allowing the warping path to pass through
 * missing regions without penalty.  Uses the same rolling-buffer O(min(m,n))
 * algorithm as dtwFull_L.
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x First sequence (may contain NaN for missing values).
 * @param y Second sequence (may contain NaN for missing values).
 * @param early_abandon Threshold for early abandon; negative disables.
 * @param metric Pointwise distance metric (default: L1).
 * @return The DTW distance with missing data handling.
 */
template <typename data_t>
data_t dtwMissing_L(std::span<const data_t> x, std::span<const data_t> y,
                    data_t early_abandon = -1,
                    core::MetricType metric = core::MetricType::L1)
{
  return detail::dispatch_missing_metric(metric, [&](auto dist) {
    return detail::dtwFull_L_impl(x, y, early_abandon, dist);
  });
}

/**
 * @brief Computes full-matrix DTW distance with missing data support.
 *
 * @details Like dtwFull but NaN-aware.  Uses O(n*m) memory.
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x First sequence (may contain NaN for missing values).
 * @param y Second sequence (may contain NaN for missing values).
 * @param metric Pointwise distance metric (default: L1).
 * @return The DTW distance with missing data handling.
 */
template <typename data_t>
data_t dtwMissing(std::span<const data_t> x, std::span<const data_t> y,
                  core::MetricType metric = core::MetricType::L1)
{
  return detail::dispatch_missing_metric(metric, [&](auto dist) {
    return detail::dtwFull_impl(x, y, dist);
  });
}

/**
 * @brief Computes banded DTW distance with missing data support (Sakoe-Chiba).
 *
 * @tparam data_t Data type of the elements in the sequences.
 * @param x First sequence (may contain NaN for missing values).
 * @param y Second sequence (may contain NaN for missing values).
 * @param band Sakoe-Chiba bandwidth.  Negative means unbanded (falls back to dtwMissing_L).
 * @param early_abandon Threshold for early abandon; negative disables.
 * @param metric Pointwise distance metric (default: L1).
 * @return The banded DTW distance with missing data handling.
 */
template <typename data_t = double>
data_t dtwMissing_banded(std::span<const data_t> x, std::span<const data_t> y,
                         int band = settings::DEFAULT_BAND,
                         data_t early_abandon = -1,
                         core::MetricType metric = core::MetricType::L1)
{
  if (band < 0) return dtwMissing_L<data_t>(x, y, early_abandon, metric);

  const auto &[short_vec, long_vec] = (x.size() < y.size()) ? std::tie(x, y) : std::tie(y, x);
  const int m_short = static_cast<int>(short_vec.size());
  const int m_long = static_cast<int>(long_vec.size());

  if ((m_short == 0) || (m_long == 0)) return std::numeric_limits<data_t>::max();
  if ((m_short == 1) || (m_long == 1)) return dtwMissing_L<data_t>(x, y, early_abandon, metric);
  if (m_long <= (band + 1)) return dtwMissing_L<data_t>(x, y, early_abandon, metric);

  return detail::dispatch_missing_metric(metric, [&](auto dist) {
    return detail::dtwBanded_impl(x, y, band, early_abandon, dist);
  });
}

// Vector convenience overloads (vector -> span implicit conversion is non-deduced).
template <typename data_t>
data_t dtwMissing_L(const std::vector<data_t> &x, const std::vector<data_t> &y,
                    data_t early_abandon = -1,
                    core::MetricType metric = core::MetricType::L1)
{
  return dtwMissing_L<data_t>(std::span<const data_t>{x}, std::span<const data_t>{y}, early_abandon, metric);
}

template <typename data_t>
data_t dtwMissing(const std::vector<data_t> &x, const std::vector<data_t> &y,
                  core::MetricType metric = core::MetricType::L1)
{
  return dtwMissing<data_t>(std::span<const data_t>{x}, std::span<const data_t>{y}, metric);
}

template <typename data_t = double>
data_t dtwMissing_banded(const std::vector<data_t> &x, const std::vector<data_t> &y,
                         int band = settings::DEFAULT_BAND,
                         data_t early_abandon = -1,
                         core::MetricType metric = core::MetricType::L1)
{
  return dtwMissing_banded<data_t>(std::span<const data_t>{x}, std::span<const data_t>{y}, band, early_abandon, metric);
}

// =========================================================================
//  Pointer + length overloads (zero-copy for bindings)
// =========================================================================

/// DTW with missing data, linear space (pointer + length).
template <typename data_t>
data_t dtwMissing_L(const data_t* x, size_t nx, const data_t* y, size_t ny,
                    data_t early_abandon = -1,
                    core::MetricType metric = core::MetricType::L1)
{
  return detail::dispatch_missing_metric(metric, [&](auto dist) {
    return detail::dtwFull_L_impl(x, nx, y, ny, early_abandon, dist);
  });
}

/// Banded DTW with missing data (pointer + length).
template <typename data_t = double>
data_t dtwMissing_banded(const data_t* x, size_t nx, const data_t* y, size_t ny,
                         int band = settings::DEFAULT_BAND,
                         data_t early_abandon = -1,
                         core::MetricType metric = core::MetricType::L1)
{
  if (band < 0) return dtwMissing_L<data_t>(x, nx, y, ny, early_abandon, metric);

  const size_t min_sz = std::min(nx, ny);
  const size_t max_sz = std::max(nx, ny);
  const int m_short = static_cast<int>(min_sz);
  const int m_long  = static_cast<int>(max_sz);

  if ((m_short == 0) || (m_long == 0)) return std::numeric_limits<data_t>::max();
  if ((m_short == 1) || (m_long == 1)) return dtwMissing_L<data_t>(x, nx, y, ny, early_abandon, metric);
  if (m_long <= (band + 1)) return dtwMissing_L<data_t>(x, nx, y, ny, early_abandon, metric);

  return detail::dispatch_missing_metric(metric, [&](auto dist) {
    return detail::dtwBanded_impl(x, nx, y, ny, band, early_abandon, dist);
  });
}

// =========================================================================
//  Public API — multivariate DTW with missing data
// =========================================================================

/**
 * @brief Multivariate missing-data DTW (linear space, zero-cost for NaN channels).
 *
 * @details For interleaved multivariate series where individual channels may be NaN.
 * NaN channels contribute zero cost at the affected timestep pair; other channels
 * still contribute normally. This is consistent with the scalar zero-cost philosophy.
 *
 * Input layout: x[t * ndim + d] is feature d at timestep t (interleaved).
 *
 * @tparam data_t Data type of the elements.
 * @param x Pointer to first series (interleaved, nx_steps * ndim elements).
 * @param nx_steps Number of timesteps in x.
 * @param y Pointer to second series (interleaved, ny_steps * ndim elements).
 * @param ny_steps Number of timesteps in y.
 * @param ndim Number of features per timestep.
 * @param early_abandon Threshold for early abandon; negative disables.
 * @param metric Pointwise distance metric (default: L1).
 * @return The DTW distance with per-channel NaN handling.
 */
template <typename data_t = double>
data_t dtwMissing_L_mv(const data_t* x, size_t nx_steps, const data_t* y, size_t ny_steps,
                       size_t ndim, data_t early_abandon = -1,
                       core::MetricType metric = core::MetricType::L1)
{
  if (ndim == 1) return dtwMissing_L(x, nx_steps, y, ny_steps, early_abandon, metric);
  return detail::dispatch_missing_mv_metric(metric, [&](auto dist) {
    return detail::dtwFull_L_mv_impl(x, nx_steps, y, ny_steps, ndim, early_abandon, dist);
  });
}

/**
 * @brief Multivariate missing-data banded DTW (Sakoe-Chiba).
 *
 * @details For ndim==1 delegates to scalar dtwMissing_banded. For ndim>1 with
 * band < 0 delegates to dtwMissing_L_mv (unbanded). Full banded MV missing
 * implementation defers to dtwMissing_L_mv for simplicity; a dedicated banded
 * MV missing impl can be added when needed.
 *
 * @tparam data_t Data type of the elements.
 * @param x Pointer to first series (interleaved).
 * @param nx_steps Number of timesteps in x.
 * @param y Pointer to second series (interleaved).
 * @param ny_steps Number of timesteps in y.
 * @param ndim Number of features per timestep.
 * @param band Sakoe-Chiba band width; negative means unconstrained.
 * @param early_abandon Threshold for early abandon; negative disables.
 * @param metric Pointwise distance metric (default: L1).
 * @return The DTW distance with per-channel NaN handling.
 */
template <typename data_t = double>
data_t dtwMissing_banded_mv(const data_t* x, size_t nx_steps, const data_t* y, size_t ny_steps,
                            size_t ndim, int band = settings::DEFAULT_BAND,
                            data_t early_abandon = -1,
                            core::MetricType metric = core::MetricType::L1)
{
  if (band < 0) return dtwMissing_L_mv(x, nx_steps, y, ny_steps, ndim, early_abandon, metric);
  if (ndim == 1) return dtwMissing_banded(x, nx_steps, y, ny_steps, band, early_abandon, metric);

  // TODO: implement a full banded MV missing variant using dtwBanded_mv_impl +
  // MissingMVL1Dist/MissingMVSquaredL2Dist. For now, delegate to unbanded.
  return dtwMissing_L_mv(x, nx_steps, y, ny_steps, ndim, early_abandon, metric);
}

} // namespace dtwc
