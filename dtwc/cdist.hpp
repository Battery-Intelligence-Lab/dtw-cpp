/**
 * @file cdist.hpp
 * @brief Cross-collection DTW distance matrix computation across CPU SIMD lanes and GPU.
 *
 * @author Volkan Kumtepeli
 * @date 13 Nov 2026
 */

#pragma once

#include "Data.hpp"
#include "base/env.hpp"
#include "core/dtw_options.hpp"

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
 *
 * A GPU request the build, the machine or the kernels cannot honour (see Config::fallback) is a DeviceError
 * before any pair is computed, or, with `fallback`, a computation on the CPU with one line on stderr saying so.
 * A GPU index below 0 and a band below -1 are InvalidInput before the device is asked, whatever `fallback` says.
 */
std::vector<double> cdist(const Data &X, const Data &Y,
                          const core::DistanceConfig &distance = {},
                          Device device = Device::CPU,
                          int device_index = 0,
                          GpuPrecision gpu_precision = GpuPrecision::Auto,
                          double gpu_share = -1.0,
                          bool fallback = false);

/**
 * @brief Cross-collection DTW distance matrix reading distance and device settings from `config`.
 * @throws InvalidInput if `config.matrix_dtype == core::Precision::Float32`; call `cdist_f32` for single-precision output.
 */
std::vector<double> cdist(const Data &X, const Data &Y, const Config &config);

/**
 * @brief Compute the dense row-major (X.size() x Y.size()) cross-collection DTW distance matrix in float32.
 * @throws InvalidInput if any distance overflows float32 range.
 */
std::vector<float> cdist_f32(const Data &X, const Data &Y,
                             const core::DistanceConfig &distance = {},
                             Device device = Device::CPU,
                             int device_index = 0,
                             GpuPrecision gpu_precision = GpuPrecision::Auto,
                             double gpu_share = -1.0,
                             bool fallback = false);

/**
 * @brief Float32 cross-collection DTW distance matrix reading distance and device settings from `config`.
 */
std::vector<float> cdist_f32(const Data &X, const Data &Y, const Config &config);

} // namespace dtwc
