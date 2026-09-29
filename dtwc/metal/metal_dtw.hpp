/**
 * @file metal_dtw.hpp
 * @brief Metal GPU kernels for batch DTW computation on Apple Silicon.
 *
 * @details Mirrors the shape of cuda_dtw.cuh: the pairwise distance matrix
 *          behind a common options struct.
 *
 *          Algorithmic references:
 *            - Register-tile + warp-shuffle cost propagation: Schmidt &
 *              Hundt (2020), "cuDTW++: Ultra-Fast Dynamic Time Warping on
 *              CUDA-Enabled GPUs", Euro-Par 2020, LNCS 12247, 597-612.
 *              https://doi.org/10.1007/978-3-030-57675-2_37
 *            - Sakoe-Chiba band: Sakoe & Chiba (1978), IEEE TASSP 26(1).
 *
 *          See also `.claude/CITATIONS.md` for the full bibliography.
 *
 * @date 2026-04-12
 */

#pragma once

#ifdef DTWC_HAS_METAL

#include "../base/error.hpp"
#include "../core/gpu_dtw_common.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace dtwc::metal {

/// Precision selection for Metal DTW kernels.
/// Apple GPUs have fast FP32 but emulated (slow) FP64 — default to FP32.
enum class MetalPrecision {
  Auto, ///< Always FP32 on Apple GPUs (FP64 is emulated).
  FP32, ///< Single precision.
  FP64  ///< Not implemented: every Metal entry point throws DeviceError.
};

inline void validate_metal_precision(MetalPrecision value)
{
  if (value == MetalPrecision::FP64)
    throw DeviceError(
      "Metal: precision FP64 is not implemented (the Metal kernels compute in "
      "FP32); no backend call or CPU fallback was attempted. Use precision Auto "
      "or FP32, or device cpu for Float64 distances.");
}

struct MetalDistMatOptions : public dtwc::gpu::DistMatOptionsBase {
  MetalPrecision precision = MetalPrecision::Auto;

  // Inherited from DistMatOptionsBase: band, use_squared_l2, verbose
};

struct MetalDistMatResult : public dtwc::gpu::DistMatResultBase {
  // All fields (matrix, n, gpu_time_sec, pairs_computed, kernel_used) come
  // from DistMatResultBase. `kernel_used`
  // strings for Metal: "wavefront" / "wavefront_global" / "banded_row" /
  // "regtile_w4" / "regtile_w8".
};

/// Check if Metal is available (MTLCreateSystemDefaultDevice succeeds).
bool metal_available();

/// Get Metal device info string (GPU name, core count, unified memory size).
std::string metal_device_info();

/// Compute NxN DTW distance matrix on the default Metal device.
/// Series are uploaded (zero-copy under unified memory where possible), all
/// pairs computed in parallel, result matrix returned on host.
MetalDistMatResult compute_distance_matrix_metal(
    const std::vector<std::vector<double>> &series,
    const MetalDistMatOptions &opts = {});

} // namespace dtwc::metal

#endif // DTWC_HAS_METAL
