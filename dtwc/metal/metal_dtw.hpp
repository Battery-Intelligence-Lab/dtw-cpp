/**
 * @file metal_dtw.hpp
 * @brief Metal GPU kernels for batch DTW computation on Apple Silicon.
 *
 * @details Mirrors the shape of cuda_dtw.cuh. The pairwise distance matrix
 *          and its LB_Keogh-pruned variant are exposed through a common
 *          options struct.
 *
 *          Algorithmic references:
 *            - Register-tile + warp-shuffle cost propagation: Schmidt &
 *              Hundt (2020), "cuDTW++: Ultra-Fast Dynamic Time Warping on
 *              CUDA-Enabled GPUs", Euro-Par 2020, LNCS 12247, 597-612.
 *              https://doi.org/10.1007/978-3-030-57675-2_37
 *            - LB_Keogh: Keogh & Ratanamahatana (2005), "Exact Indexing of
 *              Dynamic Time Warping", KAIS 7(3), 358-386.
 *            - Sakoe-Chiba band: Sakoe & Chiba (1978), IEEE TASSP 26(1).
 *
 *          See also `.claude/CITATIONS.md` for the full bibliography.
 *
 * @date 2026-04-12
 */

#pragma once

#ifdef DTWC_HAS_METAL

#include "../enums/KernelOverride.hpp"
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
  switch (value) {
  case MetalPrecision::Auto:
  case MetalPrecision::FP32:
    return;
  case MetalPrecision::FP64:
    throw DeviceError(
      "Metal: precision FP64 is not implemented (the Metal kernels compute in "
      "FP32); no backend call or CPU fallback was attempted. Use precision Auto "
      "or FP32, or device cpu for Float64 distances.");
  }
  throw InvalidInput("Invalid MetalPrecision value.");
}

struct MetalDistMatOptions : public dtwc::gpu::DistMatOptionsBase {
  MetalPrecision precision = MetalPrecision::Auto;

  /// Pruning threshold applied to max(LB(i→j), LB(j→i)). Pairs with lower
  /// bound > threshold are pruned and read NaN (not computed). 0 means "prune
  /// everything that isn't an exact envelope match"; +∞ means "compute all
  /// pairs anyway". The bound squares each excess under use_squared_l2.
  ///
  /// Note: Metal uses 0.0 as the default (always-applied threshold), while
  /// CUDA uses -1.0 (threshold-off sentinel). Kept per-backend for backward
  /// compatibility.
  double lb_threshold = 0.0;

  /// Envelope radius for LB_Keogh. Negative (default) means the DTW window:
  /// `band`, or the whole series for full DTW. A radius narrower than that
  /// window would make the bound inadmissible and throws InvalidInput; a wider
  /// one is accepted (a looser bound).
  int lb_envelope_band = -1;

  // Inherited from DistMatOptionsBase:
  //   band, use_squared_l2, verbose, use_lb_keogh, max_length_hint,
  //   kernel_override
};

struct MetalDistMatResult : public dtwc::gpu::DistMatResultBase {
  // All fields (matrix, n, gpu_time_sec, lb_time_sec, pairs_computed,
  // pairs_pruned, kernel_used) come from DistMatResultBase. `kernel_used`
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

/// Result type for standalone LB_Keogh computation on Metal.
struct MetalLBResult {
  std::vector<double> lb_values; ///< N*(N-1)/2 symmetric LB values (upper triangle, row-major).
  size_t n = 0;                  ///< Number of series.
  double gpu_time_sec = 0;       ///< Envelope + pairwise LB kernel time.
};

/// Compute LB_Keogh lower bounds for all N*(N-1)/2 pairs on the default
/// Metal device. Returns symmetric LB: max(LB(i→j), LB(j→i)).
/// Requires `band >= 0` (Sakoe-Chiba envelope window); returns empty on
/// band < 0, N <= 1, or when Metal is unavailable.
MetalLBResult compute_lb_keogh_metal(
    const std::vector<std::vector<double>> &series, int band);

} // namespace dtwc::metal

#endif // DTWC_HAS_METAL
