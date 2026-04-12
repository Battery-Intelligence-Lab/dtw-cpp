/**
 * @file metal_dtw.hpp
 * @brief Metal GPU kernels for batch DTW computation on Apple Silicon.
 *
 * @details Mirrors the shape of cuda_dtw.cuh. Pairwise distance matrix,
 *          K-vs-N, 1-vs-N, and LB_Keogh-pruned variants are all exposed
 *          through a common options struct.
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
  FP64  ///< Not implemented yet — falls back to FP32 with a warning.
};

struct MetalDistMatOptions : public dtwc::gpu::DistMatOptionsBase {
  MetalPrecision precision = MetalPrecision::Auto;

  /// Pruning threshold applied to max(LB(i→j), LB(j→i)). Pairs with lower
  /// bound > threshold are pruned. 0 means "prune everything that isn't an
  /// exact envelope match"; +∞ means "compute all pairs anyway".
  ///
  /// Note: Metal uses 0.0 as the default (always-applied threshold), while
  /// CUDA uses -1.0 (threshold-off sentinel). Kept per-backend for backward
  /// compatibility.
  double lb_threshold = 0.0;

  /// Envelope window for LB_Keogh. Negative means "use L/10 (min 1)".
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

// ===========================================================================
// 1-vs-N and K-vs-N DTW computation (k-medoids assignment loop).
// ===========================================================================

/// Result of a 1-vs-N DTW computation.
struct MetalOneVsNResult {
  std::vector<double> distances; ///< N distances from query to each target.
  size_t n = 0;                  ///< Number of target series.
  double gpu_time_sec = 0;       ///< GPU kernel execution time.
  std::string kernel_used;       ///< Which kernel variant ran.
};

/// Compute DTW distances from one series (by index) to all N series.
/// Result: `distances[query_index] == 0`, others are DTW(series[query_index], series[j]).
MetalOneVsNResult compute_dtw_one_vs_all_metal(
    const std::vector<std::vector<double>> &series,
    size_t query_index,
    const MetalDistMatOptions &opts = {});

/// Compute DTW distances from an external query series to all N targets.
MetalOneVsNResult compute_dtw_one_vs_all_metal(
    const std::vector<double> &query,
    const std::vector<std::vector<double>> &targets,
    const MetalDistMatOptions &opts = {});

/// Result of a K-vs-N DTW computation. `distances[k*n + j]` = DTW(queries[k], targets[j]).
struct MetalKVsNResult {
  std::vector<double> distances;
  size_t k = 0;
  size_t n = 0;
  double gpu_time_sec = 0;
  std::string kernel_used;
};

/// Compute K rows of the NxN matrix in one dispatch.
/// `query_indices[K]` picks which series become the queries; all N series remain targets.
MetalKVsNResult compute_dtw_k_vs_all_metal(
    const std::vector<std::vector<double>> &series,
    const std::vector<size_t> &query_indices,
    const MetalDistMatOptions &opts = {});

/// Compute K rows where the queries are a separate collection (not drawn from targets).
MetalKVsNResult compute_dtw_k_vs_all_metal(
    const std::vector<std::vector<double>> &queries,
    const std::vector<std::vector<double>> &targets,
    const MetalDistMatOptions &opts = {});

} // namespace dtwc::metal

#endif // DTWC_HAS_METAL
