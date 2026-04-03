/**
 * @file distance_metric.hpp
 * @brief Pointwise distance metric abstractions for DTW.
 *
 * @details Provides lightweight callable metric types used as the pointwise
 *          cost function inside DTW.  Each metric satisfies:
 *              T operator()(T a, T b) const
 *          Built-in metrics: L1, L2, SquaredL2.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#pragma once

#include <cmath>

namespace dtwc::core {

/// Absolute difference  |a - b|
struct L1Metric
{
  template <typename T>
  T operator()(T a, T b) const noexcept { return std::abs(a - b); }
};

/// Euclidean pointwise distance  sqrt((a-b)^2) == |a-b| for scalars
struct L2Metric
{
  template <typename T>
  T operator()(T a, T b) const noexcept { return std::abs(a - b); }
};

/// Squared Euclidean  (a - b)^2
struct SquaredL2Metric
{
  template <typename T>
  T operator()(T a, T b) const noexcept
  {
    const T d = a - b;
    return d * d;
  }
};

} // namespace dtwc::core
