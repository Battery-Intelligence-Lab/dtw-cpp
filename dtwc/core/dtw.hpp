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
 * @author Volkan Kumtepeli
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
/// @param  x        First time series.
/// @param  y        Second time series.
/// @param  band     Sakoe-Chiba band width; negative means unconstrained.
/// @param  metric   Runtime metric selector (default: L1).
/// @return DTW distance.
template <typename T = double>
T dtw_distance(const std::vector<T>& x, const std::vector<T>& y,
               int band = -1, MetricType metric = MetricType::L1)
{
  if (band < 0)
    return dtwFull_L<T>(x, y, static_cast<T>(-1), metric);
  else
    return dtwBanded<T>(x, y, band, static_cast<T>(-1), metric);
}

// -----------------------------------------------------------------------
//  Pointer-based overload (for TimeSeriesView / binding layer)
// -----------------------------------------------------------------------

/// Pointer + length overload; calls warping.hpp pointer overloads directly (zero-copy).
template <typename T = double>
T dtw_distance(const T* x, std::size_t nx, const T* y, std::size_t ny,
               int band = -1, MetricType metric = MetricType::L1)
{
  if (band < 0)
    return dtwFull_L<T>(x, nx, y, ny, static_cast<T>(-1), metric);
  else
    return dtwBanded<T>(x, nx, y, ny, band, static_cast<T>(-1), metric);
}

// -----------------------------------------------------------------------
//  Runtime-dispatched version (for Python / MATLAB bindings)
// -----------------------------------------------------------------------

/// DTW with fully runtime-configured options. Defined in dtw.cpp.
double dtw_runtime(const double* x, std::size_t nx,
                   const double* y, std::size_t ny,
                   const DTWOptions& opts);

} // namespace dtwc::core
