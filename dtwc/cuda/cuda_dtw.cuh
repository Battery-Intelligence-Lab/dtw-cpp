/**
 * @file cuda_dtw.cuh
 * @brief CUDA GPU kernels for batch DTW computation.
 *
 * @details The kernels write the packed lower triangle of the distance matrix,
 *          DistanceMatrix's own layout, in launches of consecutive pairs whose
 *          slots stream into the caller's matrix (cuda_dtw.cu); or, on the same
 *          kernels, each series' distances to a few medoids.
 *
 * @author Volkan Kumtepeli
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
  /// compute_medoid_distances_cuda: its pipeline's host buffers within this many bytes, a block down to one series
  /// (detail::medoid_blocks); 0: blocks of a launch's bound.
  std::size_t host_staging_bytes = 0;

  // Inherited from DistMatOptionsBase: band, use_squared_l2, verbose
};

/// Series i of an entry's series, its Float64 values where they lie (a view's or owned); it does not throw.
using SeriesAt = std::function<std::span<const double>(std::size_t)>;

/// A loop on the caller's threads: for_each(count, task) calls task(i) once for every i in [0, count); task(i) writes
/// only row i's slots and does not throw (CUDA code is compiled without OpenMP, so the caller runs it: run_openmp).
using ParallelFor = std::function<void(std::size_t count, const std::function<void(std::size_t)> &task)>;

struct CUDADistMatResult : public dtwc::gpu::DistMatResultBase {};

/// Check if CUDA is available (device count > 0).
bool cuda_available();

/// The bytes the calling thread's launch workspaces hold, pinned or pageable staging and every device buffer, of both
/// precisions: what core::release_thread_scratch() gives back (a test hook).
std::size_t thread_workspace_bytes();

/// Frees the calling thread's launch workspaces of both precisions: their staging buffer and device buffers, stream and
/// events (each made again when it is next used). core::release_thread_scratch() calls it on every thread.
void release_thread_workspace() noexcept;

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

/// The same fill of the @p n_series series series(i), read where they lie (a
/// view's or owned: no copy of them is made): @p parallel pads them into the
/// launch workspace's staging, from which they are uploaded once, as the
/// overload above uploads its vectors (which it pads on the calling thread).
/// A CUDA call that fails throws DeviceError naming it, after the stream has
/// drained.
CUDADistMatResult compute_distance_matrix_cuda(
    std::size_t n_series, const SeriesAt &series,
    const CUDADistMatOptions &opts, const ParallelFor &parallel,
    core::DistanceMatrix &out);

/// The DTW distance of each of the @p n_series series (series(i), read where
/// they lie: no copy of them is made) to each of the k @p medoids on the GPU,
/// on the kernels, route rule and precision of compute_distance_matrix_cuda
/// (the longest of all the series picks the kernel). @p consume takes them a
/// block of consecutive series at a time, in order, each series once, on the
/// calling thread: consume(first, count, distances), where distances[i * k + m]
/// is the distance of series first + i to medoid m. A block holds at most
/// kMaxPairsPerLaunch (launch_prep.hpp) samples and as many distances, fewer
/// within opts.host_staging_bytes (detail::medoid_blocks), which bounds the
/// device's and this call's memory whatever the number of series; while the
/// GPU computes, @p parallel pads the next block from the spans and consume
/// takes the one before. The same refusals as compute_distance_matrix_cuda come
/// before the first block; a CUDA call that fails later throws DeviceError
/// naming it, after both streams have drained.
/// @pre @p medoids is not empty (k >= 1); with none, no block is handed over.
CUDADistMatResult compute_medoid_distances_cuda(
    std::size_t n_series, const SeriesAt &series,
    const std::vector<std::span<const double>> &medoids,
    const CUDADistMatOptions &opts, const ParallelFor &parallel,
    const std::function<void(std::size_t first, std::size_t count,
                             std::span<const double> distances)> &consume);

}  // namespace dtwc::cuda

#endif  // DTWC_HAS_CUDA
