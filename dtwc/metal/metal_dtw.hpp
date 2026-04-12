/**
 * @file metal_dtw.hpp
 * @brief Metal GPU kernels for batch DTW computation on Apple Silicon.
 *
 * @details Mirrors the shape of cuda_dtw.cuh. Each Metal threadgroup computes
 *          one DTW pair via anti-diagonal wavefront: cells on the same
 *          anti-diagonal are independent and computed in parallel across
 *          the threadgroup.
 *
 *          For a distance matrix, launch N*(N-1)/2 threadgroups.
 *
 * @date 2026-04-12
 */

#pragma once

#ifdef DTWC_HAS_METAL

#include <cstddef>
#include <string>
#include <vector>

namespace dtwc::metal {

/// Precision selection for Metal DTW kernels.
/// Apple GPUs have fast FP32 but emulated (slow) FP64 — default to FP32.
enum class MetalPrecision {
  Auto, ///< Always FP32 on Apple GPUs (FP64 is emulated).
  FP32, ///< Single precision.
  FP64  ///< Not implemented yet — falls back to FP32 with a warning.
};

struct MetalDistMatOptions {
  int band = -1;               ///< Sakoe-Chiba band width (-1 = full DTW).
  bool use_squared_l2 = false; ///< Squared L2 metric instead of L1.
  bool verbose = false;        ///< Print timing info.
  MetalPrecision precision = MetalPrecision::Auto;
};

struct MetalDistMatResult {
  std::vector<double> matrix;     ///< N*N flat row-major distance matrix.
  size_t n = 0;                   ///< Number of series.
  double gpu_time_sec = 0;        ///< GPU kernel execution time.
  size_t pairs_computed = 0;      ///< Number of DTW pairs computed.
  std::string kernel_used;        ///< "wavefront" / "wavefront_global" / "banded_row".
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
