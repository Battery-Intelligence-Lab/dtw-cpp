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

  /// Opt-in LB_Keogh pruning path. When true, each pair's LB_Keogh lower
  /// bound is computed on-GPU; pairs whose LB exceeds `lb_threshold` are
  /// marked `+∞` in the result matrix and skipped by the DTW kernel.
  /// Supported only on wavefront / wavefront_global paths (i.e. band == -1
  /// with max_L > 256). On other paths the flag is silently ignored.
  bool enable_lb_keogh = false;

  /// Pruning threshold applied to max(LB(i→j), LB(j→i)). Pairs with lower
  /// bound > threshold are pruned. 0 means "prune everything that isn't an
  /// exact envelope match"; +∞ means "compute all pairs anyway".
  double lb_threshold = 0.0;

  /// Envelope window for LB_Keogh. Negative means "use L/10 (min 1)".
  int lb_envelope_band = -1;
};

struct MetalDistMatResult {
  std::vector<double> matrix;     ///< N*N flat row-major distance matrix.
  size_t n = 0;                   ///< Number of series.
  double gpu_time_sec = 0;        ///< GPU kernel execution time.
  size_t pairs_computed = 0;      ///< Number of DTW pairs computed.
  size_t pairs_pruned   = 0;      ///< LB_Keogh-pruned pairs (0 when disabled).
  std::string kernel_used;        ///< "wavefront" / "wavefront_global" / "banded_row" / "regtile_w4" / "regtile_w8".
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
