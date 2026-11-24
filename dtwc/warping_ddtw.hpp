/**
 * @file warping_ddtw.hpp
 * @brief Derivative Dynamic Time Warping (DDTW) functions.
 *
 * @details DDTW preprocesses both series with a derivative transform, then
 *          runs standard DTW on the derivative series. This captures shape
 *          similarity rather than amplitude similarity.
 *
 *          Derivative formula (Keogh & Pazzani 2001):
 *            x'[i] = ((x[i] - x[i-1]) + (x[i+1] - x[i-1]) / 2) / 2
 *          for interior points (1 <= i <= n-2).
 *          Boundary cases:
 *            x'[0]   = x[1] - x[0]
 *            x'[n-1] = x[n-1] - x[n-2]
 *
 * Reference: E. J. Keogh and M. J. Pazzani, "Derivative Dynamic Time Warping,"
 *            SIAM International Conference on Data Mining, 2001.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#pragma once

#include "warping.hpp"
#include "base/settings.hpp"

#include <vector>
#include <span>
#include <cstddef>

namespace dtwc {

/// Compute the local-slope derivative transform of `x` into `dx` (same length as `x`).
template <typename data_t>
void derivative_transform_inplace(std::span<const data_t> x, std::vector<data_t> &dx)
{
  const auto n = x.size();

  if (n == 0) {
    dx.clear();
    return;
  }

  dx.resize(n);
  if (n == 1) {
    dx[0] = data_t(0);
    return;
  }

  // Boundary: first point
  dx[0] = x[1] - x[0];

  // Interior points
  for (std::size_t i = 1; i + 1 < n; ++i) {
    dx[i] = ((x[i] - x[i - 1]) + (x[i + 1] - x[i - 1]) / data_t(2)) / data_t(2);
  }

  // Boundary: last point
  dx[n - 1] = x[n - 1] - x[n - 2];
}

/// Convenience overload: vector -> span (implicit conversion is non-deduced).
template <typename data_t>
void derivative_transform_inplace(const std::vector<data_t> &x, std::vector<data_t> &dx)
{
  derivative_transform_inplace(std::span<const data_t>{x}, dx);
}

template <typename data_t>
std::vector<data_t> derivative_transform(std::span<const data_t> x)
{
  std::vector<data_t> dx;
  derivative_transform_inplace(x, dx);
  return dx;
}

/// Convenience overload: vector -> span (implicit conversion is non-deduced).
template <typename data_t>
std::vector<data_t> derivative_transform(const std::vector<data_t> &x)
{
  return derivative_transform(std::span<const data_t>{x});
}

/// Multivariate Keogh-Pazzani derivative per channel on interleaved `x[t * ndim + d]`, written into `dx`
/// (same layout and size; empty for empty input, `ndim` zeros for a single time step).
template <typename data_t>
void derivative_transform_mv_inplace(std::span<const data_t> x, size_t ndim,
                                     std::vector<data_t> &dx)
{
  if (ndim == 1) {
    derivative_transform_inplace(x, dx);
    return;
  }

  const size_t n = x.size() / ndim; // number of timesteps
  if (n == 0) {
    dx.clear();
    return;
  }
  if (n == 1) {
    dx.assign(ndim, data_t(0));
    return;
  }

  dx.resize(x.size());

  for (size_t d = 0; d < ndim; ++d) {
    // Boundary: first point
    dx[0 * ndim + d] = x[1 * ndim + d] - x[0 * ndim + d];

    // Interior points
    for (size_t i = 1; i + 1 < n; ++i) {
      dx[i * ndim + d] = ((x[i * ndim + d] - x[(i - 1) * ndim + d])
                        + (x[(i + 1) * ndim + d] - x[(i - 1) * ndim + d]) / data_t(2)) / data_t(2);
    }

    // Boundary: last point
    dx[(n - 1) * ndim + d] = x[(n - 1) * ndim + d] - x[(n - 2) * ndim + d];
  }
}

/// Convenience overload: vector -> span.
template <typename data_t>
void derivative_transform_mv_inplace(const std::vector<data_t> &x, size_t ndim,
                                     std::vector<data_t> &dx)
{
  derivative_transform_mv_inplace(std::span<const data_t>{x}, ndim, dx);
}

template <typename data_t>
std::vector<data_t> derivative_transform_mv(std::span<const data_t> x, size_t ndim)
{
  std::vector<data_t> dx;
  derivative_transform_mv_inplace(x, ndim, dx);
  return dx;
}

/// Convenience overload: vector -> span.
template <typename data_t>
std::vector<data_t> derivative_transform_mv(const std::vector<data_t> &x, size_t ndim)
{
  return derivative_transform_mv(std::span<const data_t>{x}, ndim);
}

namespace detail {

/// DDTW's two derivative series, one pair a T (ddtwBanded and the multivariate DDTW the dispatcher binds
/// share them): reached here so that core::release_thread_scratch() can empty them (detail::dispatch_scratch, core/dtw_dispatch.cpp); two
/// thread_locals returned as references, as core::detail::lanes_bufs says why.
template <typename T>
std::pair<std::vector<T> &, std::vector<T> &> derivative_bufs()
{
  thread_local std::vector<T> dx, dy;
  return { dx, dy };
}

} // namespace detail

/**
 * @brief DDTW using banded DTW on derivative series.
 *
 * @details Computes the derivative transform of both input series, then
 *          calls dtwBanded on the derivative series.
 *
 * @tparam X, Y The series: contiguous ranges of one floating-point type (core::SeriesPair).
 * @param x First time series.
 * @param y Second time series.
 * @param band Sakoe-Chiba band width. Use -1, the default, for full (unconstrained) DTW.
 * @return The DDTW distance.
 */
template <typename X, typename Y>
  requires core::SeriesPair<X, Y>
core::series_value_t<X> ddtwBanded(const X &x, const Y &y, int band = settings::DEFAULT_BAND,
                                   core::MetricType metric = core::MetricType::L1)
{
  using data_t = core::series_value_t<X>;
  auto [dx, dy] = detail::derivative_bufs<data_t>();
  derivative_transform_inplace(std::span<const data_t>{ x }, dx);
  derivative_transform_inplace(std::span<const data_t>{ y }, dy);
  return dtwBanded(dx, dy, band, static_cast<data_t>(-1), metric);
}

} // namespace dtwc

