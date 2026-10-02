/**
 * @file cuda_dtw.cuh
 * @brief CUDA GPU kernels for batch DTW computation.
 *
 * @details The kernels write the packed lower triangle of the distance matrix,
 *          DistanceMatrix's own layout, in launches of consecutive pairs whose
 *          slots stream into the caller's matrix (cuda_dtw.cu); or, on the same
 *          kernels, each series' distances to a few medoids.
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
#include <functional>
#include <span>
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

/// The DTW distance of each of @p series to each of the k @p medoids on the
/// GPU, on the kernels, route rule and precision of compute_distance_matrix_cuda
/// (the longest of all the series picks the kernel). @p consume takes them a
/// block of consecutive series at a time, in order, each series once:
/// consume(first, count, distances), where distances[i * k + m] is the distance
/// of series first + i to medoid m. A block holds at most kMaxPairsPerLaunch
/// (launch_prep.hpp) samples and as many distances, which bounds the device's
/// and this call's memory whatever the number of series. The same refusals as
/// compute_distance_matrix_cuda come before the first block.
CUDADistMatResult compute_medoid_distances_cuda(
    const std::vector<std::vector<double>> &series,
    const std::vector<std::vector<double>> &medoids,
    const CUDADistMatOptions &opts,
    const std::function<void(std::size_t first, std::size_t count,
                             std::span<const double> distances)> &consume);

}  // namespace dtwc::cuda

#endif  // DTWC_HAS_CUDA
