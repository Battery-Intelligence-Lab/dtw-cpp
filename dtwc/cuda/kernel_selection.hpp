/**
 * @file kernel_selection.hpp
 * @brief Host-only choice of the CUDA DTW kernel family.
 *
 * The launcher runs the returned path and reports it to the caller as
 * `kernel_used`.
 */

#pragma once

#include "launch_prep.hpp"

#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace dtwc::cuda::detail {

enum class KernelPath {
  Warp,
  RegTileW4,
  RegTileW8,
  Wavefront,
  WavefrontGlobal ///< the wavefront with its anti-diagonals in global memory
};

/// The kernel for a batch whose longest series has @p max_length samples of
/// @p sample_bytes each, on a device that gives a wavefront block at most
/// @p shared_bytes of dynamic shared memory. Each path up to the wavefront is
/// the fastest one that accepts its range, by more than 5 % on the RTX 4000 Ada
/// (.claude/baselines/2026-09-29-w4a-cuda-kernel-ab.md); the wavefront keeps its
/// anti-diagonals in global memory where they do not fit shared memory.
inline KernelPath select_kernel(std::size_t max_length, std::size_t sample_bytes,
                                std::size_t shared_bytes) noexcept
{
  if (max_length <= 32) return KernelPath::Warp;
  if (max_length <= 128) return KernelPath::RegTileW4;
  if (max_length <= 256) return KernelPath::RegTileW8;
  if (wavefront_buffer_count(max_length) * max_length * sample_bytes <= shared_bytes)
    return KernelPath::Wavefront;
  return KernelPath::WavefrontGlobal;
}

inline std::string_view kernel_path_name(KernelPath path)
{
  switch (path) {
  case KernelPath::Warp: return "warp";
  case KernelPath::RegTileW4: return "regtile_w4";
  case KernelPath::RegTileW8: return "regtile_w8";
  case KernelPath::Wavefront: return "wavefront";
  case KernelPath::WavefrontGlobal: return "wavefront_global";
  }
  throw std::logic_error("kernel_path_name: unknown KernelPath");
}

} // namespace dtwc::cuda::detail
