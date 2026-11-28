/**
 * @file gpu_dtw_common.hpp
 * @brief Shared base structs for CUDA/Metal distance matrix options + results.
 *
 * @details `CUDADistMatOptions` / `MetalDistMatOptions` and their result
 *          types inherit from `DistMatOptionsBase` / `DistMatResultBase` and
 *          append backend-specific fields.
 *
 * @author Volkan Kumtepeli
 * @date 12 Apr 2026
 */

#pragma once

#include "public_distance.hpp" // normalize_public_distance (used by GPU backends)

#include <cstddef>
#include <string>

namespace dtwc::gpu {

/// Common options fields shared by CUDA and Metal distance-matrix entry points.
struct DistMatOptionsBase {
  int band = -1;                  ///< Sakoe-Chiba band width (-1 = full DTW).
  bool use_squared_l2 = false;    ///< Squared L2 metric instead of L1.
  bool verbose = false;           ///< Print timing info.
};

/// Common result fields shared by CUDA and Metal distance-matrix entry points;
/// the distances are in the caller's DistanceMatrix.
struct DistMatResultBase {
  size_t n = 0;                   ///< Number of series.
  double gpu_time_sec = 0;        ///< Full GPU execution time.
  size_t pairs_computed = 0;      ///< Number of DTW pairs computed.
  std::string kernel_used;        ///< Kernel path taken (backend-specific string).
};

} // namespace dtwc::gpu
