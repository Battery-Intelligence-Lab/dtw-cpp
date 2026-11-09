/**
 * @file launch_prep.hpp
 * @brief Host-only preconditions and geometry for every CUDA entry point.
 *
 * @details Deliberately free of CUDA headers, so its rules can be asserted on a
 * build with `DTWC_ENABLE_CUDA=OFF` (repository convention: a guard that must
 * fire without an optional dependency lives OUTSIDE that dependency's
 * `#ifdef`), and none needs a GPU to be tested.
 *
 * A missing device is a hard error, never a fallback: it previously produced an
 * all-zero N*N matrix — a valid-looking wrong answer.
 *
 * @author Volkan Kumtepeli
 * @date 02 Sep 2026
 */

#pragma once

#include "../base/error.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

namespace dtwc::cuda::detail {

/// A launch covers at most this many consecutive pairs. Its span of packed
/// slots, at most this many plus one per row it crosses (1 GiB of doubles and
/// a row), is the one output buffer the device holds whatever N is; and the
/// count stays far below INT_MAX, in which a launch counts its pairs (grid.x,
/// and the persistent wavefront's work counter, which overshoots the count by
/// up to one per block).
inline constexpr std::int64_t kMaxPairsPerLaunch = std::int64_t{ 1 } << 27;

/// The blocks of the medoid entry's pipeline (launch_medoid_kernel): `block` series a launch; `series_slots` host and
/// device buffers of a block's series padded to the longest (two, so that a block is padded and uploaded while the one
/// before computes; one where a block holds every series) and `distance_slots` pinned host buffers of a block's
/// distances (three, so that the host assigns a block while the next two compute; fewer where there are fewer blocks).
struct MedoidBlocks {
  std::size_t block = 0, series_slots = 0, distance_slots = 0;

  /// The host bytes the pipeline stages for k medoids and series of up to max_L samples of `value` bytes each: the
  /// medoids and the series slots padded to max_L, with their lengths (an int each), and the distance slots.
  std::size_t host_bytes(std::size_t k, std::size_t max_L, std::size_t value) const
  {
    return (k + series_slots * block) * (max_L * value + sizeof(int)) + distance_slots * block * k * sizeof(double);
  }
};

/// The pipeline's blocks for n >= 1 series against k >= 1 medoids, the longest of max_L samples of `value` bytes: a
/// block holds at most kMaxPairsPerLaunch / max(k, max_L) series (a launch's samples and its distances) and at least
/// one. Where `host_cap` > 0 and those blocks stage more than host_cap bytes, a block shrinks to the most series that
/// fit it (priced with two series slots and three distance slots), down to one series, which may stage more.
inline MedoidBlocks medoid_blocks(std::size_t n, std::size_t k, std::size_t max_L, std::size_t value,
                                  std::size_t host_cap = 0)
{
  const auto of = [n](std::size_t block) {
    const std::size_t blocks = (n + block - 1) / block;
    return MedoidBlocks{ block, std::min<std::size_t>(2, blocks), std::min<std::size_t>(3, blocks) };
  };
  const std::size_t launch =
    std::clamp<std::size_t>(static_cast<std::size_t>(kMaxPairsPerLaunch) / std::max(k, max_L), 1, n);
  if (host_cap == 0 || of(launch).host_bytes(k, max_L, value) <= host_cap) return of(launch);
  const std::size_t row = max_L * value + sizeof(int), medoids = k * row;
  const std::size_t per_series = 2 * row + 3 * k * sizeof(double);
  return of(std::clamp<std::size_t>(host_cap > medoids ? (host_cap - medoids) / per_series : 0, 1, launch));
}

/// Upper-triangle pair count for @p n series, evaluated in 64 bits throughout.
/// At the project's 100M-series target this is ~5e15 — representable only as
/// `size_t`; the fill splits it into launches, never narrows it.
inline constexpr std::size_t upper_triangle_pairs(std::size_t n) noexcept
{
  return (n < 2) ? std::size_t{ 0 } : n * (n - 1) / 2;
}

/// @brief Reject a GPU older than compute capability 8.0 (Ampere, 2021), the
/// oldest the kernels are built and tuned for.
/// @throws dtwc::DeviceError naming device @p device_id's compute capability.
inline void require_compute_capability(int major, int minor, int device_id)
{
  if (major < 8)
    throw dtwc::DeviceError(
      "CUDA device " + std::to_string(device_id) + " has compute capability "
      + std::to_string(major) + "." + std::to_string(minor)
      + "; DTWC++ needs 8.0 or newer (Ampere, 2021: A30, A100, RTX 30 and later). "
        "No CPU fallback was attempted.");
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

/// The wavefront stages both series in shared memory up to this length.
inline constexpr std::size_t kPreloadMaxLength = 512;

/// @brief Shared-memory buffer count for the anti-diagonal wavefront kernels.
///
/// L<=512   : preload mode (2 series + 3 anti-diagonal buffers).
/// 512<L<=1024 : 3 anti-diagonal buffers.
/// 1024<L<=2048: double-buffer mode (2 buffers, better occupancy).
/// L>2048   : back to 3 buffers — the double-buffer register cache holds
///            blockDim.x(256) * MAX_SI(8) = 2048 cells and would drop
///            anti-diagonal cells beyond that.
inline constexpr std::size_t wavefront_buffer_count(std::size_t max_L) noexcept
{
  if (max_L <= kPreloadMaxLength) return 5;
  if (max_L > 1024 && max_L <= 2048) return 2;
  return 3;
}

} // namespace dtwc::cuda::detail
