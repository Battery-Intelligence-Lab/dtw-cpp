/**
 * @file kernel_selection.hpp
 * @brief Host-only choice of the CUDA DTW kernel family.
 *
 * The launcher runs the returned path and reports it to the caller as
 * `kernel_used`.
 */

#pragma once

#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace dtwc::cuda::detail {

enum class KernelPath {
  Warp,
  RegTileW4,
  RegTileW8,
  Wavefront
};

/// The kernel for a batch whose longest series has @p max_length samples. Each
/// path is the fastest one that accepts its range, by more than 5 % on the RTX
/// 4000 Ada (.claude/baselines/2026-09-29-w4a-cuda-kernel-ab.md).
inline KernelPath select_kernel(std::size_t max_length) noexcept
{
  if (max_length <= 32) return KernelPath::Warp;
  if (max_length <= 128) return KernelPath::RegTileW4;
  if (max_length <= 256) return KernelPath::RegTileW8;
  return KernelPath::Wavefront;
}

inline std::string_view kernel_path_name(KernelPath path)
{
  switch (path) {
  case KernelPath::Warp: return "warp";
  case KernelPath::RegTileW4: return "regtile_w4";
  case KernelPath::RegTileW8: return "regtile_w8";
  case KernelPath::Wavefront: return "wavefront";
  }
  throw std::logic_error("kernel_path_name: unknown KernelPath");
}

} // namespace dtwc::cuda::detail
