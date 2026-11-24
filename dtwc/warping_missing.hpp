/**
 * @file warping_missing.hpp
 * @brief DTW with missing data (ZeroCost strategy) — thin wrappers over the
 *        unified core::dtw_kernel_*.
 *
 * @details Missing values (NaN) contribute 0 cost so the warping path can pass
 *          through missing regions without penalty. Recurrence is identical
 *          to Standard DTW:
 *
 *            C(i,j) = cost(x[i], y[j]) + min(C(i-1,j-1), C(i-1,j), C(i,j-1))
 *
 *          where `cost(a, b) = 0` if either is NaN, else the regular L1 /
 *          squared-L2 distance. All entry points route through core::run_dtw
 *          with `core::SpanNanAwareL1Cost` (or its MV / SquaredL2 variants) +
 *          `core::StandardCell`.
 *
 *          Reference: Yurtman, A., Soenen, J., Meert, W. & Blockeel, H. (2023),
 *          "Estimating DTW Distance Between Time Series with Missing Data."
 *          ECML-PKDD 2023, LNCS 14173.
 *
 *          Unchecked per-pair layer (see warping.hpp): ±inf is not rejected
 *          here; dtwc::distance::missing is the checked entry point.
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 */

#pragma once

#include "base/settings.hpp"
#include "warping.hpp" // transitive dtwBanded visibility for callers
#include "core/dtw_kernel.hpp"
#include "core/dtw_cost.hpp"
#include "core/dtw_options.hpp"
#include "core/series.hpp"

#include <cstddef>
#include <span>
#include <vector>

namespace dtwc {

// =========================================================================
//  Scalar missing-data DTW (ZeroCost)
// =========================================================================

/// ZeroCost missing-data DTW of two series (core::SeriesPair; NaN marks a missing value); a negative band is
/// unconstrained.
template <typename X, typename Y>
  requires core::SeriesPair<X, Y>
core::series_value_t<X> dtwMissing_banded(const X &x, const Y &y, int band = settings::DEFAULT_BAND,
                                          core::series_value_t<X> early_abandon = -1,
                                          core::MetricType metric = core::MetricType::L1)
{
  using data_t = core::series_value_t<X>;
  const std::span<const data_t> xs{ x }, ys{ y };
  // A univariate L2 cost is L1: sqrt((a-b)^2) == |a-b|.
  if (metric == core::MetricType::SquaredL2)
    return core::run_dtw<core::SpanNanAwareSquaredL2Cost>(xs.data(), xs.size(), ys.data(), ys.size(), band,
                                                          core::StandardCell{}, early_abandon);
  return core::run_dtw<core::SpanNanAwareL1Cost>(xs.data(), xs.size(), ys.data(), ys.size(), band,
                                                 core::StandardCell{}, early_abandon);
}

// =========================================================================
//  Multivariate missing-data DTW (interleaved layout)
// =========================================================================

/// Multivariate missing-data banded DTW: each cost skips the channels where
/// either value is NaN; L2 is the Euclidean norm over the rest.
template <typename data_t = dtwc::settings::default_data_t>
data_t dtwMissing_banded_mv(const data_t *x, size_t nx_steps, const data_t *y, size_t ny_steps,
                            size_t ndim, int band = settings::DEFAULT_BAND,
                            data_t early_abandon = -1,
                            core::MetricType metric = core::MetricType::L1)
{
  if (ndim == 1)
    return dtwMissing_banded(std::span<const data_t>(x, nx_steps), std::span<const data_t>(y, ny_steps), band,
                             early_abandon, metric);
  if (metric == core::MetricType::SquaredL2)
    return core::run_dtw<core::SpanMVNanAwareSquaredL2Cost>(x, nx_steps, y, ny_steps, band, core::StandardCell{}, early_abandon, ndim);
  if (metric == core::MetricType::L2)
    return core::run_dtw<core::SpanMVNanAwareL2Cost>(x, nx_steps, y, ny_steps, band, core::StandardCell{}, early_abandon, ndim);
  return core::run_dtw<core::SpanMVNanAwareL1Cost>(x, nx_steps, y, ny_steps, band, core::StandardCell{}, early_abandon, ndim);
}

} // namespace dtwc
