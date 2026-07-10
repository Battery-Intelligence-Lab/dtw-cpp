/**
 * @file kernel_selection.hpp
 * @brief Host-only selection of the CUDA DTW kernel family.
 *
 * Keeping this policy free of CUDA headers makes every boundary and fallback
 * independently testable in ordinary CPU builds.  Device launchers consume
 * the returned path verbatim and report the same path to callers.
 */

#pragma once

#include "../enums/KernelOverride.hpp"

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

struct KernelSelection {
  KernelPath path;
  bool fell_back_to_auto{false};
};

inline KernelPath auto_kernel(std::size_t max_length) noexcept
{
  if (max_length <= 32) return KernelPath::Warp;
  if (max_length <= 128) return KernelPath::RegTileW4;
  if (max_length <= 256) return KernelPath::RegTileW8;
  return KernelPath::Wavefront;
}

inline KernelSelection select_kernel(
    std::size_t max_length, dtwc::KernelOverride requested)
{
  dtwc::validate_kernel_override(requested);
  switch (requested) {
  case dtwc::KernelOverride::Auto:
    return {auto_kernel(max_length), false};
  case dtwc::KernelOverride::Wavefront:
    return {KernelPath::Wavefront, false};
  case dtwc::KernelOverride::RegTile:
    if (max_length <= 128) return {KernelPath::RegTileW4, false};
    if (max_length <= 256) return {KernelPath::RegTileW8, false};
    return {auto_kernel(max_length), true};
  case dtwc::KernelOverride::WavefrontGlobal:
  case dtwc::KernelOverride::BandedRow:
    // CUDA has no distinct global-wavefront or row-major banded kernel.
    return {auto_kernel(max_length), true};
  }
  throw std::logic_error("select_kernel: unreachable KernelOverride");
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
