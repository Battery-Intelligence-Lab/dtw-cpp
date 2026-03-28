/**
 * @file distance_metric.hpp
 * @brief Pointwise distance metric callables for DTW computation.
 *
 * @details Each metric satisfies the concept: T operator()(T a, T b) const.
 * For L2Metric, the caller must take the square root of the final DTW result
 * since only squared differences are accumulated during computation.
 *
 * @date 28 Mar 2026
 */

#pragma once

#include <cmath>
#include <cstdlib>

namespace dtwc::core {

struct L1Metric {
  template <typename T>
  T operator()(T a, T b) const noexcept
  {
    return std::abs(a - b);
  }
};

struct SquaredL2Metric {
  template <typename T>
  T operator()(T a, T b) const noexcept
  {
    auto d = a - b;
    return d * d;
  }
};

struct L2Metric {
  // Accumulates squared differences; caller must sqrt the final DTW result.
  template <typename T>
  T operator()(T a, T b) const noexcept
  {
    auto d = a - b;
    return d * d;
  }
};

} // namespace dtwc::core
