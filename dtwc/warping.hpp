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

#include "settings.hpp"           // for DEFAULT_BAND
#include "core/scratch_matrix.hpp"
#include "core/dtw_options.hpp"    // for core::MetricType
#include "core/dtw_kernel.hpp"     // unified DTW kernels
#include "core/dtw_cost.hpp"       // cost functors + dispatch_metric

#include <cstdlib>   // for abs, size_t
#include <algorithm> // for min, max
#include <cmath>     // for floor, round
#include <limits>    // for numeric_limits
#include <vector>    // for vector
#include <span>      // for span
#include <utility>   // for pair

namespace dtwc {

// =========================================================================
//  Implementation helpers â€” THIN shims over the unified core::dtw_kernel_*.
//  Kept for warping_missing.hpp / warping_missing_arow.hpp which still use
//  `distance(a, b)`-style functors. New code should prefer the kernel
//  directly with a Cost functor from core/dtw_cost.hpp.
// =========================================================================

namespace detail {

/// Compute inclusive [lo, hi) column range for a banded DTW at the given row.
/// Kept as a thin alias over the core::dtw_band_bounds helper â€” several older
/// variant files depend on this name.
inline std::pair<int, int> band_bounds(double slope, double window, int row)
{
  return core::dtw_band_bounds(slope, window, row);
}

/// Full-matrix DTW shim â€” forwards to core::dtw_kernel_full with StandardCell.
template <typename data_t, typename DistFn>
data_t dtwFull_impl(const data_t* x, size_t nx, const data_t* y, size_t ny,
                    DistFn distance)
{
  if (nx == 0 || ny == 0) return std::numeric_limits<data_t>::max();
  if (x == y && nx == ny) return 0;

  // Kernel requires n_short <= n_long.
  const bool swap = nx > ny;
  const data_t* xs = swap ? y : x;
  const data_t* ys = swap ? x : y;
  const size_t ns = swap ? ny : nx;
  const size_t nl = swap ? nx : ny;

  auto cost = [xs, ys, distance](size_t row, size_t col) noexcept {
    return distance(xs[row], ys[col]);
  };
  return core::dtw_kernel_full<data_t>(ns, nl, cost, core::StandardCell{});
}

template <typename data_t, typename DistFn>
data_t dtwFull_impl(std::span<const data_t> x, std::span<const data_t> y,
                    DistFn distance)
{
  return dtwFull_impl(x.data(), x.size(), y.data(), y.size(), distance);
}

/// Linear-space DTW shim.
template <typename data_t, typename DistFn>
data_t dtwFull_L_impl(const data_t* x, size_t nx, const data_t* y, size_t ny,
                      data_t early_abandon, DistFn distance)
{
  if (nx == 0 || ny == 0) return std::numeric_limits<data_t>::max();
  if (x == y && nx == ny) return 0;

  const bool swap = nx > ny;
  const data_t* xs = swap ? y : x;
  const data_t* ys = swap ? x : y;
  const size_t ns = swap ? ny : nx;
  const size_t nl = swap ? nx : ny;

  auto cost = [xs, ys, distance](size_t row, size_t col) noexcept {
    return distance(xs[row], ys[col]);
  };
  return core::dtw_kernel_linear<data_t>(ns, nl, cost, core::StandardCell{}, early_abandon);
}

template <typename data_t, typename DistFn>
data_t dtwFull_L_impl(std::span<const data_t> x, std::span<const data_t> y,
                      data_t early_abandon, DistFn distance)
{
  return dtwFull_L_impl(x.data(), x.size(), y.data(), y.size(), early_abandon, distance);
}

/// Sakoe-Chiba banded DTW shim.
template <typename data_t, typename DistFn>
data_t dtwBanded_impl(const data_t* x, size_t nx, const data_t* y, size_t ny,
                      int band, data_t early_abandon, DistFn distance)
{
  if (nx == 0 || ny == 0) return std::numeric_limits<data_t>::max();
  if (x == y && nx == ny) return 0;

  const bool swap = nx > ny;
  const data_t* xs = swap ? y : x;
  const data_t* ys = swap ? x : y;
  const size_t ns = swap ? ny : nx;
  const size_t nl = swap ? nx : ny;

  auto cost = [xs, ys, distance](size_t row, size_t col) noexcept {
    return distance(xs[row], ys[col]);
  };
  return core::dtw_kernel_banded<data_t>(ns, nl, band, cost, core::StandardCell{}, early_abandon);
}

template <typename data_t, typename DistFn>
data_t dtwBanded_impl(std::span<const data_t> x, std::span<const data_t> y,
                      int band, data_t early_abandon, DistFn distance)
{
  return dtwBanded_impl(x.data(), x.size(), y.data(), y.size(), band, early_abandon, distance);
}

/// Linear-space multivariate DTW shim â€” ndim-aware distance functor.
/// Each timestep has ndim features in interleaved layout: [t0_f0, t0_f1, ..., t1_f0, ...].
/// distance(const T*, const T*, size_t ndim) -> T
template <typename data_t, typename DistFn>
data_t dtwFull_L_mv_impl(const data_t* x, size_t nx_steps, const data_t* y, size_t ny_steps,
                          size_t ndim, data_t early_abandon, DistFn distance)
{
  if (nx_steps == 0 || ny_steps == 0) return std::numeric_limits<data_t>::max();
  if (x == y && nx_steps == ny_steps) return 0;

  const bool swap = nx_steps > ny_steps;
  const data_t* xs = swap ? y : x;
  const data_t* ys = swap ? x : y;
  const size_t ns = swap ? ny_steps : nx_steps;
  const size_t nl = swap ? nx_steps : ny_steps;

  auto cost = [xs, ys, ndim, distance](size_t row, size_t col) noexcept {
    return distance(xs + row * ndim, ys + col * ndim, ndim);
  };
  return core::dtw_kernel_linear<data_t>(ns, nl, cost, core::StandardCell{}, early_abandon);
}

/// Sakoe-Chiba banded multivariate DTW shim â€” ndim-aware distance functor.
template <typename data_t, typename DistFn>
data_t dtwBanded_mv_impl(const data_t* x, size_t nx, const data_t* y, size_t ny,
                          size_t ndim, int band, data_t early_abandon, DistFn distance)
{
  if (nx == 0 || ny == 0) return std::numeric_limits<data_t>::max();
  if (x == y && nx == ny) return 0;

  const bool swap = nx > ny;
  const data_t* xs = swap ? y : x;
  const data_t* ys = swap ? x : y;
  const size_t ns = swap ? ny : nx;
  const size_t nl = swap ? nx : ny;

  auto cost = [xs, ys, ndim, distance](size_t row, size_t col) noexcept {
    return distance(xs + row * ndim, ys + col * ndim, ndim);
  };
  return core::dtw_kernel_banded<data_t>(ns, nl, band, cost, core::StandardCell{}, early_abandon);
}

} // namespace detail

// =========================================================================
//  Distance functors
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

/// Dispatch MetricType to scalar distance functor, invoke fn(functor).
template <typename Fn>
auto dispatch_metric(core::MetricType m, Fn&& fn) -> decltype(fn(L1Dist{}))
{
  switch (m) {
  case core::MetricType::SquaredL2: return fn(SquaredL2Dist{});
  case core::MetricType::L2:
  case core::MetricType::L1:
  default: return fn(L1Dist{});
  }
}

/// Dispatch MetricType to multivariate distance functor, invoke fn(functor).
template <typename Fn>
auto dispatch_mv_metric(core::MetricType m, Fn&& fn) -> decltype(fn(MVL1Dist{}))
{
  switch (m) {
  case core::MetricType::SquaredL2: return fn(MVSquaredL2Dist{});
  case core::MetricType::L2:
  case core::MetricType::L1:
  default: return fn(MVL1Dist{});
  }
}

} // namespace detail

// =========================================================================
//  Public API â€” pointer + length overloads (zero-copy for bindings)
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
  return detail::dispatch_metric(metric, [&](auto dist) {
    return detail::dtwFull_impl(x, nx, y, ny, dist);
  });
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
  return detail::dispatch_metric(metric, [&](auto dist) {
    return detail::dtwFull_L_impl(x, nx, y, ny, early_abandon, dist);
  });
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
template <typename data_t = dtwc::settings::default_data_t>
data_t dtwBanded(const data_t* x, size_t nx, const data_t* y, size_t ny,
                 int band = settings::DEFAULT_BAND,
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

  return detail::dispatch_metric(metric, [&](auto dist) {
    return detail::dtwBanded_impl(x, nx, y, ny, band, early_abandon, dist);
  });
}

// =========================================================================
//  Public API â€” vector overloads (forward to pointer versions)
// =========================================================================

/// Full-matrix DTW (span overload).
template <typename data_t>
data_t dtwFull(std::span<const data_t> x, std::span<const data_t> y,
               core::MetricType metric = core::MetricType::L1)
{
  return dtwFull<data_t>(x.data(), x.size(), y.data(), y.size(), metric);
}

/// Linear-space DTW (span overload).
template <typename data_t>
data_t dtwFull_L(std::span<const data_t> x, std::span<const data_t> y,
                 data_t early_abandon = -1,
                 core::MetricType metric = core::MetricType::L1)
{
  return dtwFull_L<data_t>(x.data(), x.size(), y.data(), y.size(), early_abandon, metric);
}

/// Banded DTW (span overload).
template <typename data_t = dtwc::settings::default_data_t>
data_t dtwBanded(std::span<const data_t> x, std::span<const data_t> y,
                 int band = settings::DEFAULT_BAND,
                 data_t early_abandon = -1,
                 core::MetricType metric = core::MetricType::L1)
{
  return dtwBanded<data_t>(x.data(), x.size(), y.data(), y.size(), band, early_abandon, metric);
}

// Vector convenience overloads (vector -> span implicit conversion is non-deduced).
template <typename data_t>
data_t dtwFull(const std::vector<data_t> &x, const std::vector<data_t> &y,
               core::MetricType metric = core::MetricType::L1)
{
  return dtwFull<data_t>(std::span<const data_t>{x}, std::span<const data_t>{y}, metric);
}

template <typename data_t>
data_t dtwFull_L(const std::vector<data_t> &x, const std::vector<data_t> &y,
                 data_t early_abandon = -1,
                 core::MetricType metric = core::MetricType::L1)
{
  return dtwFull_L<data_t>(std::span<const data_t>{x}, std::span<const data_t>{y}, early_abandon, metric);
}

template <typename data_t = dtwc::settings::default_data_t>
data_t dtwBanded(const std::vector<data_t> &x, const std::vector<data_t> &y,
                 int band = settings::DEFAULT_BAND,
                 data_t early_abandon = -1,
                 core::MetricType metric = core::MetricType::L1)
{
  return dtwBanded<data_t>(std::span<const data_t>{x}, std::span<const data_t>{y}, band, early_abandon, metric);
}

// =========================================================================
//  Public API â€” multivariate (interleaved layout: [t0_f0, t0_f1, ..., t1_f0, ...])
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
template <typename data_t = dtwc::settings::default_data_t>
data_t dtwFull_L_mv(const data_t* x, size_t nx_steps, const data_t* y, size_t ny_steps,
                    size_t ndim, data_t early_abandon = -1,
                    core::MetricType metric = core::MetricType::L1)
{
  if (ndim == 1) return dtwFull_L(x, nx_steps, y, ny_steps, early_abandon, metric);
  return detail::dispatch_mv_metric(metric, [&](auto dist) {
    return detail::dtwFull_L_mv_impl(x, nx_steps, y, ny_steps, ndim, early_abandon, dist);
  });
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
template <typename data_t = dtwc::settings::default_data_t>
data_t dtwBanded_mv(const data_t* x, size_t nx_steps, const data_t* y, size_t ny_steps,
                    size_t ndim, int band = settings::DEFAULT_BAND,
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

  return detail::dispatch_mv_metric(metric, [&](auto dist) {
    return detail::dtwBanded_mv_impl(x, nx_steps, y, ny_steps, ndim, band, early_abandon, dist);
  });
}

} // namespace dtwc

