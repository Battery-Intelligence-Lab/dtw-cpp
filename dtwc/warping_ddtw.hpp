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
 * @author Claude (AI assistant)
 * @date 28 Mar 2026
 */

#pragma once

#include "warping.hpp"
#include "settings.hpp"

#include <vector>
#include <cstddef>

namespace dtwc {

/**
 * @brief Compute the derivative transform of a time series.
 *
 * @details The derivative transform produces a series of the same length
 *          that captures the local slope at each point.
 *
 * @tparam data_t Data type of the elements.
 * @param x Input time series.
 * @return Derivative-transformed series (same length as input).
 */
template <typename data_t>
std::vector<data_t> derivative_transform(const std::vector<data_t> &x)
{
  const auto n = x.size();

  if (n == 0) return {};
  if (n == 1) return { data_t(0) };

  std::vector<data_t> dx(n);

  // Boundary: first point
  dx[0] = x[1] - x[0];

  // Interior points
  for (std::size_t i = 1; i + 1 < n; ++i) {
    dx[i] = ((x[i] - x[i - 1]) + (x[i + 1] - x[i - 1]) / data_t(2)) / data_t(2);
  }

  // Boundary: last point
  dx[n - 1] = x[n - 1] - x[n - 2];

  return dx;
}

/**
 * @brief DDTW using banded DTW on derivative series.
 *
 * @details Computes the derivative transform of both input series, then
 *          calls dtwBanded on the derivative series.
 *
 * @tparam data_t Data type of the elements.
 * @param x First time series.
 * @param y Second time series.
 * @param band Sakoe-Chiba band width. Use -1 for full (unconstrained) DTW.
 * @return The DDTW distance.
 */
template <typename data_t = double>
data_t ddtwBanded(const std::vector<data_t> &x, const std::vector<data_t> &y,
                  int band = settings::DEFAULT_BAND_LENGTH,
                  core::MetricType metric = core::MetricType::L1)
{
  thread_local std::vector<data_t> dx, dy;
  dx = derivative_transform(x);
  dy = derivative_transform(y);
  return dtwBanded<data_t>(dx, dy, band, static_cast<data_t>(-1), metric);
}

/**
 * @brief DDTW using full linear-space DTW on derivative series.
 *
 * @details Computes the derivative transform of both input series, then
 *          calls dtwFull_L (the memory-efficient O(N) space variant) on
 *          the derivative series.
 *
 * @tparam data_t Data type of the elements.
 * @param x First time series.
 * @param y Second time series.
 * @return The DDTW distance.
 */
template <typename data_t = double>
data_t ddtwFull_L(const std::vector<data_t> &x, const std::vector<data_t> &y,
                  core::MetricType metric = core::MetricType::L1)
{
  thread_local std::vector<data_t> dx, dy;
  dx = derivative_transform(x);
  dy = derivative_transform(y);
  return dtwFull_L<data_t>(dx, dy, static_cast<data_t>(-1), metric);
}

} // namespace dtwc
