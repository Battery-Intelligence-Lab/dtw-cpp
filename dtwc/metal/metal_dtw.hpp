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
 *          See also `docs/CITATIONS.md` for the full bibliography.
 *
 * @author Volkan Kumtepeli
 * @date 2026-04-12
 */

#pragma once

#ifdef DTWC_HAS_METAL

#include "../base/env.hpp"
#include "../base/error.hpp"
#include "../core/distance_matrix.hpp"
#include "../core/gpu_dtw_common.hpp"

#include <cstddef>
#include <functional>
#include <span>
#include <string>
#include <vector>

namespace dtwc::metal {

/// The Metal kernels compute in FP32 (Apple GPUs emulate FP64), so Auto is FP32
/// and FP64 is refused at every Metal entry point.
inline void validate_metal_precision(GpuPrecision value)
{
  if (value == GpuPrecision::FP64)
    throw GpuRefusal(
      "Metal: precision FP64 is not implemented (the Metal kernels compute in "
      "FP32); no backend call or CPU fallback was attempted. Use precision Auto "
      "or FP32, or device cpu for Float64 distances.",
      "Metal computes in FP32, so precision FP64 cannot run");
}

struct MetalDistMatOptions : public dtwc::gpu::DistMatOptionsBase {
  GpuPrecision precision = GpuPrecision::Auto;

  // Inherited from DistMatOptionsBase: band, use_squared_l2, verbose
};

struct MetalDistMatResult : public dtwc::gpu::DistMatResultBase {
  // All fields (n, gpu_time_sec, pairs_computed, kernel_used) come
  // from DistMatResultBase. `kernel_used`
  // strings for Metal: "wavefront" / "wavefront_global" / "banded_row" /
  // "regtile_w4" / "regtile_w8".
};

/// Check if Metal is available (MTLCreateSystemDefaultDevice succeeds).
bool metal_available();

/// Get Metal device info string (GPU name, core count, unified memory size).
std::string metal_device_info();

/// Series i of an entry's series, its Float64 values where they lie (a view's or owned); it does not throw.
using SeriesAt = std::function<std::span<const double>(std::size_t)>;

/// A loop on the caller's threads: for_each(count, task) calls task(i) once for every i in [0, count); task(i) writes
/// only its own slots and does not throw.
using ParallelFor = std::function<void(std::size_t count, const std::function<void(std::size_t)> &task)>;

/// Fill @p out with the DTW distance of every pair of @p series on the default
/// Metal device. Every refusal (FP64, no device, all series empty, a scratch
/// buffer that cannot be allocated) comes before @p out is touched; then @p out
/// is resized to series.size() unless it already has that size (a mapped
/// matrix keeps its file), and every entry is written.
MetalDistMatResult compute_distance_matrix_metal(
    const std::vector<std::vector<double>> &series,
    const MetalDistMatOptions &opts, core::DistanceMatrix &out);

/// The same fill of the @p n_series series series(i), read where they lie (a
/// view's or owned: no copy of them is made): @p parallel stages FP64->FP32
/// inputs and widens FP32->FP64 outputs in parallel.
MetalDistMatResult compute_distance_matrix_metal(
    std::size_t n_series, const SeriesAt &series,
    const MetalDistMatOptions &opts, const ParallelFor &parallel,
    core::DistanceMatrix &out);

} // namespace dtwc::metal

#endif // DTWC_HAS_METAL
