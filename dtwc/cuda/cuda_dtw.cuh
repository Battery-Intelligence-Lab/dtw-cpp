/**
 * @file cuda_dtw.cuh
 * @brief CUDA GPU kernels for batch DTW computation.
 *
 * @details Each CUDA block computes one DTW pair. Within a block,
 *          threads cooperate on the anti-diagonal wavefront:
 *          cells on the same anti-diagonal are independent.
 *
 *          For the distance matrix, launch N*(N-1)/2 blocks.
 *          Each block has min(band, L) threads.
 *
 * @date 29 Mar 2026
 */

#pragma once

#ifdef DTWC_HAS_CUDA

#include "../enums/KernelOverride.hpp"
#include "../base/error.hpp"
#include "../core/gpu_dtw_common.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace dtwc::cuda {

/// Precision selection for CUDA DTW kernels
enum class CUDAPrecision {
  Auto,  ///< FP32 on consumer GPUs (slow FP64), FP64 on HPC GPUs
  FP32,  ///< Always use single precision (fastest, ~1e-7 relative error)
  FP64   ///< Always use double precision (bit-identical to CPU path)
};

inline void validate_cuda_precision(CUDAPrecision value)
{
  switch (value) {
  case CUDAPrecision::Auto:
  case CUDAPrecision::FP32:
  case CUDAPrecision::FP64:
    return;
  }
  throw InvalidInput("Invalid CUDAPrecision value.");
}

struct CUDADistMatOptions : public dtwc::gpu::DistMatOptionsBase {
  int device_id = 0;                             ///< CUDA device to use
  CUDAPrecision precision = CUDAPrecision::Auto; ///< Compute precision

  /// When positive (and band >= 0), pairs with LB > threshold are not
  /// computed and read NaN. A pair without a warping path reads the finite public
  /// double-max no-result sentinel, not IEEE infinity. The bound squares each
  /// excess under use_squared_l2.
  /// CUDA default is -1.0 (threshold-off sentinel); Metal uses 0.0 with
  /// different semantics. Kept per-backend for backward compatibility.
  double lb_threshold = -1.0;

  // Inherited from DistMatOptionsBase:
  //   band, use_squared_l2, verbose, use_lb_keogh, max_length_hint,
  //   kernel_override
};

struct CUDADistMatResult : public dtwc::gpu::DistMatResultBase {
  /// True only when a valid but unsupported CUDA override used Auto instead.
  bool kernel_override_fell_back = false;
};

/// Check if CUDA is available (device count > 0).
bool cuda_available();

/// Get CUDA device info string.
std::string cuda_device_info(int device_id = 0);

/// Compute NxN DTW distance matrix on GPU.
/// Series data is transferred to GPU, all pairs computed in parallel,
/// results transferred back.
CUDADistMatResult compute_distance_matrix_cuda(
    const std::vector<std::vector<double>> &series,
    const CUDADistMatOptions &opts = {});

/// Result type for standalone LB_Keogh computation.
struct CUDALBResult {
  std::vector<double> lb_values; ///< N*(N-1)/2 lower bounds (upper triangle, row-major)
  size_t n = 0;                  ///< Number of series
  double gpu_time_sec = 0;       ///< GPU kernel execution time
};

/// Compute LB_Keogh lower bounds for all N*(N-1)/2 pairs on GPU.
/// Returns symmetric LB_Keogh: max(LB(i->j), LB(j->i)) for each pair.
/// Requires band >= 0 (Sakoe-Chiba constraint); returns empty result if band < 0.
CUDALBResult compute_lb_keogh_cuda(
    const std::vector<std::vector<double>> &series,
    int band, int device_id = 0);

}  // namespace dtwc::cuda

#endif  // DTWC_HAS_CUDA
