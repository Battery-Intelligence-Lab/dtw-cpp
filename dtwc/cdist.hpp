/**
 * @file cdist.hpp
 * @brief Cross-collection DTW distance matrix computation across CPU SIMD lanes and GPU.
 * @author Volkan Kumtepeli
 */

#pragma once

#include "Data.hpp"
#include "base/env.hpp"
#include "core/dtw_options.hpp"

#include <string>
#include <vector>

namespace dtwc {

struct Config;

/**
 * @brief Compute the dense row-major (X.size() x Y.size()) cross-collection DTW distance matrix.
 *
 * @details Element `out[i * Y.size() + j]` is the DTW distance between `X[i]` and `Y[j]`.
 * - On `Device::CPU`: pre-packs `Y` into SIMD lane panels (`PackedLanesPanel<T>`) once and
 *   parallelizes over `X` (or `(X, panel)` tiles when `X` is small) with OpenMP; falls back
 *   to parallel per-pair kernels for variants/multivariate modes without packed lanes.
 * - On `Device::GPU` / `Device::CPU_GPU`: streams cross-collection distances through the
 *   CUDA rectangular kernels (`compute_medoid_distances_cuda`), or splits `X`'s rows across
 *   CUDA and CPU SIMD panels when `Device::CPU_GPU` is selected.
 *
 * Validates `ndim` agreement, `core::validate(distance)`, finite/missing series values, and
 * Sakoe-Chiba band feasibility across `(X, Y)` before computing any pair. When one side is
 * `Float32` and the other `Float64`, widens the `Float32` side to `Float64`.
 */
std::vector<double> cdist(const Data &X, const Data &Y,
                          const core::DistanceConfig &distance = {},
                          Device device = Device::CPU,
                          int device_index = 0,
                          GpuPrecision gpu_precision = GpuPrecision::Auto,
                          double gpu_share = -1.0);

/**
 * @brief Cross-collection DTW distance matrix reading distance and device settings from `config`.
 */
std::vector<double> cdist(const Data &X, const Data &Y, const Config &config);

} // namespace dtwc
