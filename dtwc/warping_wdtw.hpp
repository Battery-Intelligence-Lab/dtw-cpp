/**
 * @file warping_wdtw.hpp
 * @brief Weighted Dynamic Time Warping (WDTW) — thin wrappers over the
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
 * @date 28 Mar 2026
 */

#pragma once

#include "base/settings.hpp"
#include "warping.hpp"
#include "core/dtw_kernel.hpp"
#include "core/dtw_cost.hpp"
#include "core/series.hpp"
#include "base/error.hpp"

#include <algorithm> // std::max
#include <cmath>     // std::exp
#include <cstddef>   // size_t
#include <limits>
#include <span>
#include <string>      // std::to_string (weight-length diagnostic)
#include <type_traits> // std::type_identity_t
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
  // The sum and the quotient are double for float32 too, as the table has always been built.
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
// Scalar WDTW — banded, with precomputed weights
// ---------------------------------------------------------------------------

/// WDTW of two series (core::SeriesPair) under `weights`, which must hold max(|x|, |y|) entries (InvalidInput
/// otherwise; wdtw_weights builds them). A negative band, the default, is unconstrained.
template <typename X, typename Y>
  requires core::SeriesPair<X, Y>
core::series_value_t<X> wdtwBanded(const X &x, const Y &y,
                                   std::type_identity_t<std::span<const core::series_value_t<X>>> weights,
                                   int band = settings::DEFAULT_BAND)
{
  using data_t = core::series_value_t<X>;
  const std::span<const data_t> xs{ x }, ys{ y };
  detail::require_wdtw_weight_span(weights.size(), xs.size(), ys.size());
  return core::run_dtw<core::SpanWeightedL1Cost>(xs.data(), xs.size(), ys.data(), ys.size(), band,
                                                 core::StandardCell{}, data_t(-1), weights.data());
}

// ---------------------------------------------------------------------------
// Convenience overloads: accept `g` instead of precomputed weights.
// Thread-local weight cache avoids recomputing exp() when max_dev / g are stable.
// ---------------------------------------------------------------------------

namespace detail {
  /// cached_wdtw_weights' table and the max_dev and g it was built for (-1: none). Its key lives with it, so that
  /// core::release_thread_scratch(), which empties it (detail::dispatch_scratch, core/dtw_dispatch.cpp), resets the key too.
  template <typename T>
  struct WdtwCache
  {
    std::vector<T> w;
    int max_dev = -1;
    T g = T(-1);
  };

  /// This thread's WdtwCache, one a T.
  template <typename T>
  WdtwCache<T> &wdtw_cache()
  {
    thread_local WdtwCache<T> cache;
    return cache;
  }

  template <typename data_t>
  const std::vector<data_t> &cached_wdtw_weights(int max_dev, data_t g)
  {
    auto &[cached_w, cached_max_dev, cached_g] = wdtw_cache<data_t>();
    if (max_dev != cached_max_dev || g != cached_g) {
      cached_w = wdtw_weights<data_t>(max_dev, g);
      cached_max_dev = max_dev;
      cached_g = g;
    }
    return cached_w;
  }
} // namespace detail

/// WDTW of two series (core::SeriesPair) under the logistic weights of steepness `g` (wdtw_weights), cached per thread.
template <typename X, typename Y>
  requires core::SeriesPair<X, Y>
core::series_value_t<X> wdtwBanded(const X &x, const Y &y, int band, core::series_value_t<X> g)
{
  using data_t = core::series_value_t<X>;
  const std::span<const data_t> xs{ x }, ys{ y };
  const auto max_len = std::max(xs.size(), ys.size());
  if (max_len == 0) return std::numeric_limits<data_t>::max();
  const int max_dev = static_cast<int>(max_len) - 1;
  const auto &w = detail::cached_wdtw_weights<data_t>(max_dev, g);
  return wdtwBanded(xs, ys, std::span<const data_t>{ w }, band);
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
  if (ndim == 1)
    return wdtwBanded(std::span<const data_t>(x, nx_steps), std::span<const data_t>(y, ny_steps), weights, band);
  detail::require_wdtw_weight_span(weights.size(), nx_steps, ny_steps);
  return core::run_dtw<core::SpanMVWeightedL1Cost>(x, nx_steps, y, ny_steps, band, core::StandardCell{}, data_t(-1), weights.data(), ndim);
}

template <typename data_t = dtwc::settings::default_data_t>
data_t wdtwBanded_mv(const data_t *x, size_t nx_steps, const data_t *y, size_t ny_steps,
                     size_t ndim, int band = settings::DEFAULT_BAND, data_t g = 0.05)
{
  if (ndim == 1)
    return wdtwBanded(std::span<const data_t>(x, nx_steps), std::span<const data_t>(y, ny_steps), band, g);
  const size_t max_steps = std::max(nx_steps, ny_steps);
  const int max_dev = (max_steps > 0) ? static_cast<int>(max_steps - 1) : 0;
  const auto &w = detail::cached_wdtw_weights<data_t>(max_dev, g);
  return wdtwBanded_mv(x, nx_steps, y, ny_steps, ndim, std::span<const data_t>{ w }, band);
}

} // namespace dtwc
