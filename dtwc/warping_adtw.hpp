/**
 * @file warping_adtw.hpp
 * @brief Amerced Dynamic Time Warping (ADTW) — thin wrappers over the unified DTW kernel.
 *
 * @details ADTW adds a penalty for non-diagonal (horizontal/vertical) warping
 *          steps, discouraging time stretching/compression:
 *
 *            C(i,j) = d(x[i], y[j]) + min(C(i-1,j-1),
 *                                         C(i-1,j) + penalty,
 *                                         C(i,j-1) + penalty)
 *
 *          Implementation-wise, ADTW differs from Standard DTW only in the
 *          cell-recurrence rule, so every ADTW entry point is a call of
 *          `core::run_dtw` with `core::ADTWCell<T>` and an L1 / MV-L1 cost
 *          functor.
 *
 *          Reference: Herrmann, M. & Webb, G. I. (2023), "Amercing: An intuitive
 *          and effective constraint for dynamic time warping." Pattern
 *          Recognition, 137, 109333.
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

#include <cstddef>   // size_t
#include <span>
#include <vector>

namespace dtwc {

// ---------------------------------------------------------------------------
// Scalar ADTW (univariate)
// ---------------------------------------------------------------------------

/// ADTW of two series (core::SeriesPair); a negative band is unconstrained.
template <typename X, typename Y>
  requires core::SeriesPair<X, Y>
core::series_value_t<X> adtwBanded(const X &x, const Y &y, int band, core::series_value_t<X> penalty,
                                   core::series_value_t<X> early_abandon = -1)
{
  using value_t = core::series_value_t<X>;
  const std::span<const value_t> xs{ x }, ys{ y };
  return core::run_dtw<core::SpanL1Cost>(xs.data(), xs.size(), ys.data(), ys.size(), band,
                                         core::ADTWCell<value_t>{ penalty }, early_abandon);
}

// ---------------------------------------------------------------------------
// Multivariate ADTW (interleaved layout: x[t*ndim + d])
// ---------------------------------------------------------------------------

/// Multivariate ADTW banded: the L1 cost summed over the channels.
template <typename data_t = dtwc::settings::default_data_t>
data_t adtwBanded_mv(const data_t *x, size_t nx_steps, const data_t *y, size_t ny_steps,
                     size_t ndim, int band = settings::DEFAULT_BAND, data_t penalty = 1.0)
{
  if (ndim == 1)
    return adtwBanded(std::span<const data_t>(x, nx_steps), std::span<const data_t>(y, ny_steps), band, penalty);
  return core::run_dtw<core::SpanMVL1Cost>(x, nx_steps, y, ny_steps, band, core::ADTWCell<data_t>{penalty},
                                           data_t{-1}, ndim);
}

} // namespace dtwc

