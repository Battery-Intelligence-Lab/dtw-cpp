/**
 * @file public_distance.hpp
 * @brief Normalize compute-type distance sentinels at double-valued APIs.
 */

#pragma once

#include <cfloat>

// The CUDA kernels widen their result on the device, so the packed matrix they
// write is already the public binary64 one.
#if defined(__CUDACC__)
#define DTWC_PUBLIC_DISTANCE_HD __host__ __device__
#else
#define DTWC_PUBLIC_DISTANCE_HD
#endif

namespace dtwc::core {

/**
 * Convert a Float32-compute distance to the public binary64 representation.
 *
 * Float32 kernels use `float::max()` as their finite no-path sentinel. Public
 * distance containers and callables use doubles, whose no-path sentinel is
 * `double::max()`. Every other value is widened without reinterpretation.
 * FLT_MAX and DBL_MAX, not numeric_limits, so the device can call it.
 */
DTWC_PUBLIC_DISTANCE_HD inline constexpr double normalize_public_distance(float value) noexcept
{
  return value == FLT_MAX ? DBL_MAX : static_cast<double>(value);
}

/// Float64 compute already has the public representation.
DTWC_PUBLIC_DISTANCE_HD inline constexpr double normalize_public_distance(double value) noexcept
{
  return value;
}

} // namespace dtwc::core
