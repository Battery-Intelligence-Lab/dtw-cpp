/**
 * @file public_distance.hpp
 * @brief Normalize compute-type distance sentinels at double-valued APIs.
 */

#pragma once

#include <limits>

namespace dtwc::core {

/**
 * Convert a Float32-compute distance to the public binary64 representation.
 *
 * Float32 kernels use `float::max()` as their finite no-path sentinel. Public
 * distance containers and callables use doubles, whose no-path sentinel is
 * `double::max()`. Every other value is widened without reinterpretation.
 */
inline constexpr double normalize_public_distance(float value) noexcept
{
  return value == std::numeric_limits<float>::max()
      ? std::numeric_limits<double>::max()
      : static_cast<double>(value);
}

/// Float64 compute already has the public representation.
inline constexpr double normalize_public_distance(double value) noexcept
{
  return value;
}

} // namespace dtwc::core
