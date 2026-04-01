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

struct CUDADistMatOptions {
  int band = -1;           ///< Sakoe-Chiba band width (-1 = full DTW)
  bool use_squared_l2 = false; ///< Use squared L2 metric instead of L1
  int device_id = 0;       ///< CUDA device to use
  bool verbose = false;     ///< Print timing info
  CUDAPrecision precision = CUDAPrecision::Auto; ///< Compute precision
};

struct CUDADistMatResult {
  std::vector<double> matrix; ///< N*N flat row-major distance matrix
  size_t n = 0;               ///< Number of series
  double gpu_time_sec = 0;    ///< GPU kernel execution time
  size_t pairs_computed = 0;  ///< Number of DTW pairs computed
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

}  // namespace dtwc::cuda

#endif  // DTWC_HAS_CUDA
