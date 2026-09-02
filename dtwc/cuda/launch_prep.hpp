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

#include "../error.hpp"

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
      + ". Reduce N or use the MPI backend for distributed computation.");
}

/// @brief Reject a CUDA call on a host with no usable device.
/// @throws dtwc::DeviceError when @p available is false.
inline void require_cuda_device(bool available, const char *entry)
{
  if (!available)
    throw dtwc::DeviceError(
      std::string(entry) + ": no CUDA device is available. DTWC++ was built "
      "with CUDA support but no usable device was found; no CPU fallback was "
      "attempted.");
}

/// @brief Fill @p lengths with each series' length and return the maximum.
/// @param initial_max Seed for the maximum (an external query's length, or 0).
inline std::size_t scan_series_lengths(
  const std::vector<std::vector<double>> &series,
  std::vector<int> &lengths,
  std::size_t initial_max = 0)
{
  const std::size_t n = series.size();
  lengths.resize(n);
  std::size_t max_L = initial_max;
  for (std::size_t i = 0; i < n; ++i) {
    lengths[i] = static_cast<int>(series[i].size());
    if (series[i].size() > max_L) max_L = series[i].size();
  }
  return max_L;
}

/// @brief Series length that drives kernel selection.
///
/// `max_length_hint` lets a caller steer the choice without round-tripping the
/// real lengths (e.g. later rows will be longer). It only ever raises the
/// heuristic input: buffer sizing must keep using the scanned maximum. Metal
/// has always honoured it (metal_dtw.mm); CUDA silently ignored it until now.
inline constexpr std::size_t kernel_selection_length(
  std::size_t scanned_max_L, int max_length_hint) noexcept
{
  const std::size_t hint = (max_length_hint > 0)
    ? static_cast<std::size_t>(max_length_hint) : std::size_t{ 0 };
  return (hint > scanned_max_L) ? hint : scanned_max_L;
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
