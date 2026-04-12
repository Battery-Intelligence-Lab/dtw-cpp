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
  bool use_lb_keogh = false; ///< Compute LB_Keogh on GPU and skip pairs exceeding threshold
  double lb_threshold = -1.0; ///< When positive, pairs with LB > threshold get INF (no DTW)
};

struct CUDADistMatResult {
  std::vector<double> matrix; ///< N*N flat row-major distance matrix
  size_t n = 0;               ///< Number of series
  double gpu_time_sec = 0;    ///< GPU kernel execution time
  size_t pairs_computed = 0;  ///< Number of DTW pairs computed
  size_t pairs_pruned = 0;    ///< Number of pairs skipped by LB_Keogh pruning
  double lb_time_sec = 0;     ///< Time for LB_Keogh computation (envelope + LB kernels)
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

// =========================================================================
// 1-vs-N and K-vs-N DTW computation
// =========================================================================

/// Result of a 1-vs-N DTW computation.
struct CUDAOneVsNResult {
  std::vector<double> distances; ///< N distances from query to each series
  double gpu_time_sec = 0;       ///< GPU kernel execution time
  size_t n = 0;                  ///< Number of target series
};

/// Compute DTW distances from one query series (by index) to all N series.
/// Result: distances[query_index] == 0, others are DTW distances.
CUDAOneVsNResult compute_dtw_one_vs_all(
    const std::vector<std::vector<double>> &series,
    size_t query_index,
    const CUDADistMatOptions &opts = {});

/// Compute DTW distances from an external query to all N series.
CUDAOneVsNResult compute_dtw_one_vs_all(
    const std::vector<double> &query,
    const std::vector<std::vector<double>> &series,
    const CUDADistMatOptions &opts = {});

/// Result of a K-vs-N DTW computation.
struct CUDAKVsNResult {
  std::vector<double> distances; ///< K*N distances (row-major: result[k*N + j])
  double gpu_time_sec = 0;       ///< GPU kernel execution time
  size_t k = 0;                  ///< Number of query series
  size_t n = 0;                  ///< Number of target series
};

/// Compute multiple rows of the distance matrix at once.
/// query_indices[K] specifies which series are queries.
/// Returns K*N distances (row-major: result[k*N + j] = DTW(query_k, series_j)).
CUDAKVsNResult compute_dtw_k_vs_all(
    const std::vector<std::vector<double>> &series,
    const std::vector<size_t> &query_indices,
    const CUDADistMatOptions &opts = {});

}  // namespace dtwc::cuda

#endif  // DTWC_HAS_CUDA
