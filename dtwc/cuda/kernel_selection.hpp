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
/// @p sample_bytes each, on a device whose SM has @p sm_shared_bytes of shared
/// memory, of which every wavefront block also takes @p block_overhead_bytes
/// (the kernel's static and the runtime's reserved bytes). Each path up to the
/// wavefront is the fastest one that accepts its range, by more than 5 % on the
/// RTX 4000 Ada (.claude/baselines/2026-09-29-w4a-cuda-kernel-ab.md). Up to
/// L = 2048 a wavefront block's anti-diagonals (32 KB at most) fit any supported
/// GPU; above it they stay in shared memory only where three blocks fit an SM,
/// and go to global memory otherwise, which is faster there: 0.64-0.68 of the
/// time at FP32 L 6000-8446 and 0.76-0.84 at FP64 L 2049-4223 on an RTX 4000
/// Ada (.claude/baselines/2026-09-30-c2-cuda-route.md).
inline KernelPath select_kernel(std::size_t max_length, std::size_t sample_bytes,
                                std::size_t sm_shared_bytes,
                                std::size_t block_overhead_bytes) noexcept
{
  if (max_length <= 32) return KernelPath::Warp;
  if (max_length <= 128) return KernelPath::RegTileW4;
  if (max_length <= 256) return KernelPath::RegTileW8;
  const std::size_t block_bytes =
      wavefront_buffer_count(max_length) * max_length * sample_bytes + block_overhead_bytes;
  if (max_length <= 2048 || 3 * block_bytes <= sm_shared_bytes) return KernelPath::Wavefront;
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
