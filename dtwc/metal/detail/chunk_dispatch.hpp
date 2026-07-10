/**
 * @file chunk_dispatch.hpp
 * @brief Host-side arithmetic shared by Metal chunk dispatch and its tests.
 */

#pragma once

#include <cstddef>
#include <cstdint>

namespace dtwc::metal::detail {

/**
 * Convert a chunk's first pair index to the signed 64-bit value bound to the
 * Metal kernel ABI. Keeping this conversion in host-compilable C++ lets every
 * platform regression-test the production path without requiring Metal.
 *
 * The caller's pair count is derived from an int-sized series count, so the
 * largest possible triangular index is below INT64_MAX.
 */
[[nodiscard]] constexpr std::int64_t pair_chunk_offset(std::size_t offset) noexcept
{
  return static_cast<std::int64_t>(offset);
}

} // namespace dtwc::metal::detail
