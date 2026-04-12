/**
 * @file gpu_dtw_common.hpp
 * @brief Shared base structs for CUDA/Metal distance matrix options + results.
 *
 * @details Both backends historically duplicated the common fields (band,
 *          verbose, use_lb_keogh, max_length_hint, kernel_override, etc.).
 *          This header factors those into `DistMatOptionsBase` and
 *          `DistMatResultBase`; `CUDADistMatOptions` / `MetalDistMatOptions`
 *          inherit and append backend-specific fields. Designated aggregate
 *          initialisation still works in C++20 (`opts.band = 5; ...`).
 *
 *          `lb_threshold` intentionally stays in the derived structs —
 *          CUDA defaults to `-1.0` (threshold off when non-positive) while
 *          Metal defaults to `0.0` (always applied); normalising would be a
 *          user-visible behaviour change and is deferred.
 *
 * @date 2026-04-12
 */

#pragma once

#include "../enums/KernelOverride.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace dtwc::gpu {

/// Common options fields shared by CUDA and Metal distance-matrix entry points.
struct DistMatOptionsBase {
  int band = -1;                  ///< Sakoe-Chiba band width (-1 = full DTW).
  bool use_squared_l2 = false;    ///< Squared L2 metric instead of L1.
  bool verbose = false;           ///< Print timing info.
  bool use_lb_keogh = false;      ///< Enable LB_Keogh pruning path.
  int max_length_hint = 0;        ///< Hint for expected max series length (0 = scan).
  dtwc::KernelOverride kernel_override = dtwc::KernelOverride::Auto;
};

/// Common result fields shared by CUDA and Metal distance-matrix entry points.
struct DistMatResultBase {
  std::vector<double> matrix;     ///< N*N flat row-major distance matrix.
  size_t n = 0;                   ///< Number of series.
  double gpu_time_sec = 0;        ///< Full GPU execution time (includes LB pre-pass).
  double lb_time_sec = 0;         ///< Time for LB_Keogh pre-pass; 0 when disabled.
  size_t pairs_computed = 0;      ///< Number of DTW pairs computed.
  size_t pairs_pruned = 0;        ///< Pairs skipped by LB_Keogh pruning.
  std::string kernel_used;        ///< Kernel path taken (backend-specific string).
};

} // namespace dtwc::gpu
