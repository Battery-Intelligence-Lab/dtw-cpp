/**
 * @file kernel_selection.hpp
 * @brief Host-only choice of the CUDA DTW kernel family.
 *
 * The launcher runs the returned path and reports it to the caller as
 * `kernel_used`.
 * @author Volkan Kumtepeli
 */

#pragma once

#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <string_view>

namespace dtwc::cuda::detail {

enum class KernelPath {
  WarpTileW1,      ///< register-tiled warp kernel, TILE_W=1 (1-pass L<=32)
  WarpTileW4,      ///< register-tiled warp kernel, TILE_W=4 (1-pass 32<L<=128, multi-pass L>128)
  WarpTileW8,      ///< register-tiled warp kernel, TILE_W=8 (FP32: 1-pass 128<L<=256, multi-pass L>256)
  Wavefront,       ///< anti-diagonal wavefront fallback in shared memory
  WavefrontGlobal  ///< anti-diagonal wavefront fallback in global memory
};

/// The kernel for a batch of @p num_pairs pairs whose longest series has
/// @p max_length samples of @p sample_bytes each, on a device with @p sm_count
/// SMs and @p max_shared_per_block opt-in shared memory per block (minus
/// @p block_overhead_bytes).
///
/// The unified multi-pass warp register-tiled kernel (`warptile_w1` for
/// L <= 32, `warptile_w4` for 32 < L <= 128 or FP64 L > 32, `warptile_w8`
/// for FP32 L > 128) handles all batches except when pair parallelism is too
/// low for warps on long series (`num_pairs < 8 * sm_count && max_length > 1024`)
/// or when the persistent warp border buffer does not fit available VRAM
/// (`!warptile_border_fits`), in which case it falls back to the 3-buffer
/// anti-diagonal wavefront (`wavefront` in shared memory if
/// `3 * max_length * sample_bytes` fits a block, otherwise `wavefront_global`).
inline KernelPath select_kernel(std::size_t max_length, std::size_t sample_bytes,
                                std::size_t num_pairs, int sm_count,
                                bool warptile_border_fits,
                                std::size_t max_shared_per_block,
                                std::size_t block_overhead_bytes) noexcept
{
  const bool fp32 = (sample_bytes == 4);
  const std::size_t min_pairs = static_cast<std::size_t>(8) * static_cast<std::size_t>(std::max(sm_count, 1));
  const bool use_wavefront_fallback =
      (num_pairs < min_pairs && max_length > 1024) || !warptile_border_fits;
  if (!use_wavefront_fallback) {
    if (max_length <= 32) return KernelPath::WarpTileW1;
    return (fp32 && max_length > 128) ? KernelPath::WarpTileW8 : KernelPath::WarpTileW4;
  }
  const std::size_t needed_shared = 3 * max_length * sample_bytes + block_overhead_bytes;
  if (needed_shared <= max_shared_per_block) return KernelPath::Wavefront;
  return KernelPath::WavefrontGlobal;
}

inline std::string_view kernel_path_name(KernelPath path)
{
  switch (path) {
  case KernelPath::WarpTileW1: return "warptile_w1";
  case KernelPath::WarpTileW4: return "warptile_w4";
  case KernelPath::WarpTileW8: return "warptile_w8";
  case KernelPath::Wavefront: return "wavefront";
  case KernelPath::WavefrontGlobal: return "wavefront_global";
  }
  throw std::logic_error("kernel_path_name: unknown KernelPath");
}

} // namespace dtwc::cuda::detail
