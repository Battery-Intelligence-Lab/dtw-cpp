/**
 * @file cuda_dtw.cuh
 * @brief CUDA GPU kernels for batch DTW computation.
 *
 * @details The kernels write the packed lower triangle of the distance matrix,
 *          DistanceMatrix's own layout, in launches of consecutive pairs whose
 *          slots stream into the caller's matrix (cuda_dtw.cu).
 *
 * @date 29 Mar 2026
 */

#pragma once

#ifdef DTWC_HAS_CUDA

#include "../base/env.hpp"
#include "../base/error.hpp"
#include "../core/distance_matrix.hpp"
#include "../core/gpu_dtw_common.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace dtwc::cuda {

struct CUDADistMatOptions : public dtwc::gpu::DistMatOptionsBase {
  int device_id = 0;                           ///< CUDA device to use
  GpuPrecision precision = GpuPrecision::Auto; ///< Auto: FP32 where FP64 is slow (consumer GPUs)

  // Inherited from DistMatOptionsBase: band, use_squared_l2, verbose
};

struct CUDADistMatResult : public dtwc::gpu::DistMatResultBase {};

/// Check if CUDA is available (device count > 0).
bool cuda_available();

/// Get CUDA device info string.
std::string cuda_device_info(int device_id = 0);

/// Fill @p out with the DTW distance of every pair of @p series on the GPU.
/// Every refusal (an invalid precision, no device, all series empty, a
/// wavefront that does not fit the device) comes before @p out is touched;
/// then @p out is resized to series.size() unless it already has that size
/// (a mapped matrix keeps its file), and every entry is written.
CUDADistMatResult compute_distance_matrix_cuda(
    const std::vector<std::vector<double>> &series,
    const CUDADistMatOptions &opts, core::DistanceMatrix &out);

}  // namespace dtwc::cuda

#endif  // DTWC_HAS_CUDA
