/**
 * @file dtw.hpp
 * @brief Unified DTW entry point wrapping existing warping.hpp functions.
 *
 * @details Templates on constraint type only (not metric).  Metric dispatch
 *          overhead is negligible (~0.003 % of DTW cost) so metrics are
 *          resolved at runtime via a callable parameter.
 *
 *          This header does NOT rewrite any DTW algorithm; it delegates to
 *          dtwFull_L / dtwBanded in warping.hpp.
 *
 * @date 28 Mar 2026
 */

#pragma once

#include "../warping.hpp"        // existing DTW implementations
#include "dtw_options.hpp"
#include "distance_metric.hpp"

#include <vector>
#include <cstddef>

namespace dtwc::core {

// -----------------------------------------------------------------------
//  Template entry point (compile-time metric, runtime band selection)
// -----------------------------------------------------------------------

/// Compute the DTW distance between two time series.
///
/// @tparam T        Element type (default: double).
/// @tparam Metric   Pointwise cost callable (default: L1Metric).
/// @param  x        First time series.
/// @param  y        Second time series.
/// @param  band     Sakoe-Chiba band width; negative means unconstrained.
/// @param  metric   Metric instance (unused until warping.hpp is refactored;
///                  provided for forward API compatibility).
/// @return DTW distance.
/// @note Currently only L1Metric is supported. Passing a different metric
///       will produce a compile error. Full metric abstraction will be
///       wired when warping.hpp is refactored to accept a metric callable.
template <typename T = double, typename Metric = L1Metric>
T dtw_distance(const std::vector<T>& x, const std::vector<T>& y,
               int band = -1, const Metric& /*metric*/ = {})
{
  static_assert(std::is_same_v<Metric, L1Metric>,
    "Only L1Metric is currently supported. Other metrics will be added "
    "when warping.hpp is refactored to accept a metric callable.");
  if (band < 0)
    return dtwFull_L<T>(x, y);
  else
    return dtwBanded<T>(x, y, band);
}

// -----------------------------------------------------------------------
//  Pointer-based overload (for TimeSeriesView / binding layer)
// -----------------------------------------------------------------------

/// Pointer + length overload; wraps data into vectors and delegates.
/// This temporary copy will be eliminated when warping.hpp accepts pointers.
template <typename T = double>
T dtw_distance(const T* x, std::size_t nx, const T* y, std::size_t ny,
               int band = -1)
{
  const std::vector<T> vx(x, x + nx);
  const std::vector<T> vy(y, y + ny);
  return dtw_distance(vx, vy, band);
}

// -----------------------------------------------------------------------
//  Runtime-dispatched version (for Python / MATLAB bindings)
// -----------------------------------------------------------------------

/// DTW with fully runtime-configured options. Defined in dtw.cpp.
double dtw_runtime(const double* x, std::size_t nx,
                   const double* y, std::size_t ny,
                   const DTWOptions& opts);

} // namespace dtwc::core
