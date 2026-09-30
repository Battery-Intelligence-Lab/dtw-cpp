/**
 * @file launch_prep.hpp
 * @brief Host-only preconditions and geometry for every CUDA entry point.
 *
 * @details Deliberately free of CUDA headers, for two reasons:
 *   1. The guards must be assertable on a build with `DTWC_ENABLE_CUDA=OFF`
 *      (repository convention: a guard that must fire without an optional
 *      dependency lives OUTSIDE that dependency's `#ifdef`).
 *   2. Neither guard needs a GPU to be tested. The pair-count limit fires at
 *      N >= 65536, whose series data cannot be allocated in a unit test, so the
 *      size-only helper is the only testable form of the check.
 *
 * Both are hard errors, never fallbacks: a truncated pair count silently drops
 * or over-runs work, and a missing device previously produced an all-zero N*N
 * matrix — a valid-looking wrong answer.
 *
 * @date 02 Sep 2026
 */

#pragma once

#include "../base/error.hpp"

#include <cstddef>
#include <limits>
#include <string>
#include <vector>

namespace dtwc::cuda::detail {

/// CUDA caps grid.x at 2^31-1 blocks and every pair-indexed kernel carries its
/// pair count as `int`; both limits are the same number.
inline constexpr std::size_t kMaxPairsPerLaunch =
    static_cast<std::size_t>(std::numeric_limits<int>::max());

/// Upper-triangle pair count for @p n series, evaluated in 64 bits throughout.
/// At the project's 100M-series target this is ~5e15 — representable only as
/// `size_t`, which is exactly why the count must never be narrowed before the
/// guard below has run.
inline constexpr std::size_t upper_triangle_pairs(std::size_t n) noexcept
{
  return (n < 2) ? std::size_t{ 0 } : n * (n - 1) / 2;
}

/// @brief Reject a workload whose pair count cannot be indexed by a CUDA launch.
/// @throws dtwc::InvalidInput when @p num_pairs exceeds kMaxPairsPerLaunch.
inline void require_pair_count_fits(std::size_t num_pairs, const char *entry)
{
  if (num_pairs > kMaxPairsPerLaunch)
    throw dtwc::InvalidInput(
      std::string(entry) + ": too many DTW pairs (" + std::to_string(num_pairs)
      + ") for a single CUDA kernel launch. Maximum: "
      + std::to_string(kMaxPairsPerLaunch)
      + ". Reduce N, or cluster on device cpu with method onebatch or clara,"
        " which never build the N*N matrix.");
}

/// @brief Reject a CUDA call on a host with no usable device.
/// @throws dtwc::DeviceError when @p available is false.
inline void require_cuda_device(bool available, const char *entry)
{
  if (!available)
    throw dtwc::DeviceError(
      std::string(entry) + ": DTWC++ was built with CUDA support, but no "
      "usable CUDA GPU was detected. No CPU fallback was attempted.");
}

/// @brief Fill @p lengths with the series' lengths in upload order, the last
/// series first (cuda_dtw.cu's packed_slot says why), and return the maximum.
inline std::size_t scan_series_lengths(
  const std::vector<std::vector<double>> &series,
  std::vector<int> &lengths)
{
  const std::size_t n = series.size();
  lengths.resize(n);
  std::size_t max_L = 0;
  for (std::size_t d = 0; d < n; ++d) {
    const std::size_t length = series[n - 1 - d].size();
    lengths[d] = static_cast<int>(length);
    if (length > max_L) max_L = length;
  }
  return max_L;
}

/// @brief Shared-memory buffer count for the anti-diagonal wavefront kernels.
///
/// L<=512   : preload mode (2 series + 3 anti-diagonal buffers).
/// 512<L<=1024 : 3 anti-diagonal buffers.
/// 1024<L<=2048: double-buffer mode (2 buffers, better occupancy).
/// L>2048   : back to 3 buffers — the double-buffer register cache holds
///            blockDim.x(256) * MAX_SI(8) = 2048 cells and would drop
///            anti-diagonal cells beyond that (Task 0.1).
inline constexpr std::size_t wavefront_buffer_count(std::size_t max_L) noexcept
{
  if (max_L <= 512) return 5;
  if (max_L > 1024 && max_L <= 2048) return 2;
  return 3;
}

} // namespace dtwc::cuda::detail
