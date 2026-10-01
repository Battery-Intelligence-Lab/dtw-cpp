/**
 * @file warping_wdtw.hpp
 * @brief Weighted Dynamic Time Warping (WDTW) â€” thin wrappers over the
 *        unified DTW kernel.
 *
 * @details WDTW multiplies each pointwise cost by a weight that depends on the
 *          absolute index difference |i - j|. Larger deviations get heavier
 *          penalties. The weight vector is typically logistic:
 *
 *            w(d) = w_max / (1 + exp(-g * (d - m/2)))
 *
 *          where m is the maximum possible deviation (max(L1, L2) - 1).
 *
 *          The weighted cost is position-dependent, so WDTW's "Cost"
 *          functor is `core::SpanWeightedL1Cost` / `core::SpanMVWeightedL1Cost`
 *          rather than the position-agnostic L1. The cell-recurrence is
 *          Standard DTW (no penalty modifier, unlike ADTW).
 *
 *          Reference: Y.-S. Jeong, M.-K. Jeong, O.-A. Omitaomu, "Weighted
 *          dynamic time warping for time series classification", Pattern
 *          Recognition, 44(9), 2231-2240 (2011).
 *
 * @author Volkan Kumtepeli
 * @author Claude 4.6
 * @date 28 Mar 2026
 */

#pragma once

#include "base/settings.hpp"
#include "warping.hpp"
#include "core/dtw_kernel.hpp"
#include "core/dtw_cost.hpp"
#include "base/error.hpp"

#include <algorithm>      // std::max
#include <cmath>          // std::exp
#include <cstddef>        // size_t
#include <limits>
#include <span>
#include <string>         // std::to_string (weight-length diagnostic)
#include <type_traits>    // std::type_identity_t
#include <vector>

namespace dtwc {

// ---------------------------------------------------------------------------
// Weight vector construction
// ---------------------------------------------------------------------------

/// Logistic WDTW weight vector: w(d) = w_max / (1 + exp(-g * (d - m/2))); g >= 0, unchecked.
template <typename data_t>
std::vector<data_t> wdtw_weights(int max_dev, data_t g = 0.05, data_t w_max = 1.0)
{
  std::vector<data_t> weights(max_dev + 1);
  const data_t half_dev = static_cast<data_t>(max_dev) / data_t(2);
  // The sum and the quotient are double for float32 too: a weight rounds once.
  for (int d = 0; d <= max_dev; ++d) {
    weights[d] = static_cast<data_t>(w_max / (1.0 + std::exp(-g * (d - half_dev))));
  }
  return weights;
}

namespace detail {
/// WDTW cost functors index `weights[|row - col|]`, whose maximum is
/// max(nx, ny) - 1, so the table must hold at least max(nx, ny) entries.
/// Validated once per call at the public boundary — never per cell.
inline void require_wdtw_weight_span(std::size_t weights_size,
                                     std::size_t nx, std::size_t ny)
{
  const std::size_t needed = std::max(nx, ny);
  if (weights_size < needed)
    throw InvalidInput(
      "WDTW: weights array holds " + std::to_string(weights_size)
      + " entries but the series pair needs " + std::to_string(needed)
      + " (max(|x|, |y|)). Build it with wdtw_weights(max(|x|, |y|) - 1, g).");
}
} // namespace detail

// ---------------------------------------------------------------------------
// Scalar WDTW â€” full matrix / banded, with precomputed weights
// ---------------------------------------------------------------------------

template <typename data_t>
data_t wdtwBanded(const data_t *x, size_t nx, const data_t *y, size_t ny,
                  std::type_identity_t<std::span<const data_t>> weights,
                  int band = settings::DEFAULT_BAND)
{
  detail::require_wdtw_weight_span(weights.size(), nx, ny);
  return core::run_dtw<core::SpanWeightedL1Cost>(x, nx, y, ny, band, core::StandardCell{}, data_t(-1),
                                                 weights.data());
}

template <typename data_t>
data_t wdtwFull(const data_t *x, size_t nx, const data_t *y, size_t ny,
                std::type_identity_t<std::span<const data_t>> weights)
{
  return wdtwBanded<data_t>(x, nx, y, ny, weights, -1);
}

template <typename data_t>
data_t wdtwFull(std::span<const data_t> x, std::span<const data_t> y,
                std::type_identity_t<std::span<const data_t>> weights)
{
  return wdtwFull(x.data(), x.size(), y.data(), y.size(), weights);
}

template <typename data_t>
data_t wdtwBanded(std::span<const data_t> x, std::span<const data_t> y,
                  std::type_identity_t<std::span<const data_t>> weights,
                  int band = settings::DEFAULT_BAND)
{
  return wdtwBanded<data_t>(x.data(), x.size(), y.data(), y.size(), weights, band);
}

// ---------------------------------------------------------------------------
// Convenience overloads: accept `g` instead of precomputed weights.
// Thread-local weight cache avoids recomputing exp() when max_dev / g are stable.
// ---------------------------------------------------------------------------

namespace detail {
template <typename data_t>
const std::vector<data_t>& cached_wdtw_weights(int max_dev, data_t g)
{
  thread_local std::vector<data_t> cached_w;
  thread_local int cached_max_dev = -1;
  thread_local data_t cached_g = data_t(-1);
  if (max_dev != cached_max_dev || g != cached_g) {
    cached_w = wdtw_weights<data_t>(max_dev, g);
    cached_max_dev = max_dev;
    cached_g = g;
  }
  return cached_w;
}
} // namespace detail

template <typename data_t = dtwc::settings::default_data_t>
data_t wdtwBanded(const data_t *x, size_t nx, const data_t *y, size_t ny,
                  int band, data_t g)
{
  const auto max_len = std::max(nx, ny);
  if (max_len == 0) return std::numeric_limits<data_t>::max();
  const int max_dev = static_cast<int>(max_len) - 1;
  const auto& w = detail::cached_wdtw_weights<data_t>(max_dev, g);
  return wdtwBanded(x, nx, y, ny, std::span<const data_t>{w}, band);
}

template <typename data_t = dtwc::settings::default_data_t>
data_t wdtwBanded(std::span<const data_t> x, std::span<const data_t> y,
                  int band, data_t g)
{
  return wdtwBanded(x.data(), x.size(), y.data(), y.size(), band, g);
}

template <typename data_t = dtwc::settings::default_data_t>
data_t wdtwFull(const data_t *x, size_t nx, const data_t *y, size_t ny,
                data_t g)
{
  return wdtwBanded<data_t>(x, nx, y, ny, -1, g);
}

template <typename data_t = dtwc::settings::default_data_t>
data_t wdtwFull(std::span<const data_t> x, std::span<const data_t> y, data_t g)
{
  return wdtwFull(x.data(), x.size(), y.data(), y.size(), g);
}

// Vector convenience overloads
template <typename data_t>
data_t wdtwFull(const std::vector<data_t> &x, const std::vector<data_t> &y,
                std::type_identity_t<std::span<const data_t>> weights)
{
  return wdtwFull<data_t>(std::span<const data_t>{x}, std::span<const data_t>{y}, weights);
}

template <typename data_t>
data_t wdtwBanded(const std::vector<data_t> &x, const std::vector<data_t> &y,
                  std::type_identity_t<std::span<const data_t>> weights,
                  int band = settings::DEFAULT_BAND)
{
  return wdtwBanded<data_t>(std::span<const data_t>{x}, std::span<const data_t>{y}, weights, band);
}

template <typename data_t = dtwc::settings::default_data_t>
data_t wdtwBanded(const std::vector<data_t> &x, const std::vector<data_t> &y,
                  int band, data_t g)
{
  return wdtwBanded<data_t>(std::span<const data_t>{x}, std::span<const data_t>{y}, band, g);
}

template <typename data_t = dtwc::settings::default_data_t>
data_t wdtwFull(const std::vector<data_t> &x, const std::vector<data_t> &y, data_t g)
{
  return wdtwFull<data_t>(std::span<const data_t>{x}, std::span<const data_t>{y}, g);
}

// ---------------------------------------------------------------------------
// Multivariate WDTW (interleaved layout x[t * ndim + d])
// ---------------------------------------------------------------------------

/// Multivariate WDTW banded: the weighted L1 cost summed over the channels.
template <typename data_t = dtwc::settings::default_data_t>
data_t wdtwBanded_mv(const data_t *x, size_t nx_steps, const data_t *y, size_t ny_steps,
                     size_t ndim, std::type_identity_t<std::span<const data_t>> weights,
                     int band = settings::DEFAULT_BAND)
{
  if (ndim == 1) return wdtwBanded<data_t>(x, nx_steps, y, ny_steps, weights, band);
  detail::require_wdtw_weight_span(weights.size(), nx_steps, ny_steps);
  return core::run_dtw<core::SpanMVWeightedL1Cost>(x, nx_steps, y, ny_steps, band, core::StandardCell{},
                                                   data_t(-1), weights.data(), ndim);
}

template <typename data_t = dtwc::settings::default_data_t>
data_t wdtwFull_mv(const data_t *x, size_t nx_steps, const data_t *y, size_t ny_steps,
                   size_t ndim, std::type_identity_t<std::span<const data_t>> weights)
{
  return wdtwBanded_mv<data_t>(x, nx_steps, y, ny_steps, ndim, weights, -1);
}

template <typename data_t = dtwc::settings::default_data_t>
data_t wdtwBanded_mv(const data_t *x, size_t nx_steps, const data_t *y, size_t ny_steps,
                     size_t ndim, int band = settings::DEFAULT_BAND, data_t g = 0.05)
{
  if (ndim == 1) return wdtwBanded<data_t>(x, nx_steps, y, ny_steps, band, g);
  const size_t max_steps = std::max(nx_steps, ny_steps);
  const int max_dev = (max_steps > 0) ? static_cast<int>(max_steps - 1) : 0;
  const auto& w = detail::cached_wdtw_weights<data_t>(max_dev, g);
  return wdtwBanded_mv(x, nx_steps, y, ny_steps, ndim, std::span<const data_t>{w}, band);
}

template <typename data_t = dtwc::settings::default_data_t>
data_t wdtwFull_mv(const data_t *x, size_t nx_steps, const data_t *y, size_t ny_steps,
                   size_t ndim, data_t g = 0.05)
{
  return wdtwBanded_mv<data_t>(x, nx_steps, y, ny_steps, ndim, -1, g);
}

} // namespace dtwc

