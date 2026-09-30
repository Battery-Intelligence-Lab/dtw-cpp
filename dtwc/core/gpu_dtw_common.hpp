/**
 * @file gpu_dtw_common.hpp
 * @brief Shared base structs for CUDA/Metal distance matrix options + results.
 *
 * @details Both backends historically duplicated the common fields (band,
 *          use_squared_l2, verbose).
 *          This header factors those into `DistMatOptionsBase` and
 *          `DistMatResultBase`; `CUDADistMatOptions` / `MetalDistMatOptions`
 *          inherit and append backend-specific fields. Designated aggregate
 *          initialisation still works in C++20 (`opts.band = 5; ...`).
 *
 * @date 2026-04-12
 */

#pragma once

#include "public_distance.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace dtwc::gpu {

/// Common options fields shared by CUDA and Metal distance-matrix entry points.
struct DistMatOptionsBase {
  int band = -1;                  ///< Sakoe-Chiba band width (-1 = full DTW).
  bool use_squared_l2 = false;    ///< Squared L2 metric instead of L1.
  bool verbose = false;           ///< Print timing info.
};

/// Common result fields shared by CUDA and Metal distance-matrix entry points.
struct DistMatResultBase {
  std::vector<double> matrix;     ///< N*N flat row-major distance matrix.
  size_t n = 0;                   ///< Number of series.
  double gpu_time_sec = 0;        ///< Full GPU execution time.
  size_t pairs_computed = 0;      ///< Number of DTW pairs computed.
  std::string kernel_used;        ///< Kernel path taken (backend-specific string).
};

} // namespace dtwc::gpu
