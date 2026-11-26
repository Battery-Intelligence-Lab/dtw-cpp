/**
 * @file cuda_dtw.cu
 * @brief CUDA implementation of batch DTW distance computation.
 *
 * @details Two kernel families cover all series lengths:
 *   1. dtw_warptile_kernel: unified multi-pass warp register-tiled kernel
 *      (Euro-Par 2020 cuDTW++ register-tiling extended to arbitrary length).
 *      Zero shared memory (0 B): column tiles load via __ldg into registers,
 *      row values load coalesced 32 elements at a time and shift across lanes
 *      via __shfl_up_sync. Single-pass (L <= 32 * TILE_W) touches no border
 *      buffer; multi-pass runs a persistent warp grid where lane 31 writes
 *      column borders into per-warp L2-resident global slices and lane 0 reads
 *      them on the next pass. TILE_W=4 for L <= 128 or FP64; TILE_W=8 for
 *      FP32 L > 128.
 *   2. dtw_wavefront_fallback_kernel: clean 3-buffer anti-diagonal wavefront
 *      used only when pair parallelism is too small for warp-per-pair on long
 *      series (num_pairs < 8 * sm_count && max_L > 1024) or when the warp
 *      border buffer would exceed available VRAM. Uses native LDS/STS with
 *      three explicit integer offsets.
 *
 * @author Volkan Kumtepeli
 */

#include "cuda_dtw.cuh"
#include "cuda_memory.cuh"
#include "kernel_selection.hpp"
#include "launch_prep.hpp"
#include "../core/medoid_assignment_policy.hpp"
#include "../detail/decode_pair.hpp"

#ifdef DTWC_HAS_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <array>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace dtwc::cuda {

using dtwc::core::normalize_public_distance;
using dtwc::detail::decode_pair;
using dtwc::detail::decode_rectangle_pair;

/// The slot of DistanceMatrix's packed lower triangle (row r holds columns
/// 0..r) that takes the distance of the pair decode_pair gives, si < sj.
__host__ __device__ inline std::int64_t packed_slot(std::int64_t si, std::int64_t sj,
                                                    std::int64_t N)
{
  const std::int64_t row = N - 1 - si;
  return row * (row + 1) / 2 + (N - 1 - sj);
}

enum class Pairs { Triangle, Rectangle };

/// Typed decode_pair: FP32 uses an FP32 sqrt seed + 64-bit integer correction
/// so float Triangle kernels execute zero FP64 instructions while matching
/// 64-bit integer math identically.
template <typename T>
__device__ __forceinline__ void decode_pair_typed(std::int64_t pair, std::int64_t n,
                                                  std::int64_t &si, std::int64_t &sj)
{
  if constexpr (std::is_same_v<T, float>) {
    const std::uint64_t un = static_cast<std::uint64_t>(n);
    const std::uint64_t uk = static_cast<std::uint64_t>(pair);
    const std::uint64_t a = 2 * un - 1;
    const std::uint64_t disc = a * a - 8 * uk;
    std::uint64_t s = static_cast<std::uint64_t>(__fsqrt_rn(static_cast<float>(disc)));
    while ((s + 1) * (s + 1) <= disc) ++s;
    while (s * s > disc) --s;

    std::int64_t row = static_cast<std::int64_t>((a - s) / 2);
    if (row > n - 2) row = n - 2;
    if (row < 0) row = 0;
    std::int64_t row_start = row * (2 * n - row - 1) / 2;
    while (row > 0 && row_start > pair) {
      --row;
      row_start = row * (2 * n - row - 1) / 2;
    }
    while (row + 1 < n && row_start + (n - row - 1) <= pair) {
      row_start += (n - row - 1);
      ++row;
    }
    si = row;
    sj = row + 1 + (pair - row_start);
  } else {
    decode_pair(pair, n, si, sj);
  }
}

template <Pairs P, typename T>
__device__ __forceinline__ void pair_series(std::int64_t pair, std::int64_t n,
                                            std::int64_t &si, std::int64_t &sj)
{
  if constexpr (P == Pairs::Triangle) {
    decode_pair_typed<T>(pair, n, si, sj);
  } else {
    decode_rectangle_pair(pair, n, si, sj);
    si += n;
  }
}

template <Pairs P>
__device__ __forceinline__ std::int64_t pair_slot(std::int64_t si, std::int64_t sj, std::int64_t n)
{
  if constexpr (P == Pairs::Triangle)
    return packed_slot(si, sj, n);
  else
    return (si - n) * n + sj;
}

template <typename T, typename OutT>
__device__ __forceinline__ void store_distance(OutT *__restrict__ out, std::int64_t idx,
                                               T dist, int *__restrict__ d_overflow)
{
  if constexpr (std::is_same_v<OutT, float>) {
    float out_val;
    if constexpr (std::is_same_v<T, float>) {
      // Zero FP64 ops in FP32 -> Float32 direct store: FLT_MAX is the no-path sentinel
      // (which no float32 matrix can hold) and becomes +inf; any non-finite distance flags overflow.
      out_val = (dist == FLT_MAX) ? __int_as_float(0x7f800000) : dist;
    } else {
      out_val = static_cast<float>(dist);
    }
    if (d_overflow && !isfinite(out_val)) {
      atomicExch(d_overflow, 1);
    }
    out[idx] = out_val;
  } else {
    out[idx] = normalize_public_distance(dist);
  }
}

// Forward declaration for DeviceLimits setup.
template <typename T, Pairs P, typename OutT, bool SquaredL2, bool Banded, bool UseGlobal>
__global__ void dtw_wavefront_fallback_kernel(
    const T *__restrict__ all_series, const int *__restrict__ lengths,
    OutT *__restrict__ out, int *__restrict__ d_overflow,
    int n, int max_L, int num_pairs, int band,
    int *__restrict__ work_counter, std::int64_t first_pair, std::int64_t first_slot,
    T *__restrict__ scratch);

namespace {

/// Active row range [i_lo, i_hi] on anti-diagonal k = i + j within the M x N_len grid
/// and Sakoe-Chiba band |i - j| <= band. Empty (i_lo > i_hi) when k < 0 or no cell lies in band.
template <bool Banded>
__device__ __forceinline__ void wavefront_range(int k, int M, int N_len, int band, int &i_lo, int &i_hi)
{
  if (k < 0) {
    i_lo = 0;
    i_hi = -1;
    return;
  }
  i_lo = max(0, k - N_len + 1);
  i_hi = min(k, M - 1);
  if constexpr (Banded) {
    if (band < N_len - 1) {
      if (k > band) i_lo = max(i_lo, (k - band + 1) / 2);
      i_hi = min(i_hi, k / 2 + band / 2 + ((k & 1) & (band & 1)));
    }
  }
}

struct DeviceLimits {
  int compute_major = 0, compute_minor = 0;
  bool slow_fp64 = false;
  int sm_count = 0;
  size_t max_shared_per_block = 0;
  size_t l2_bytes = 0;
  size_t shared_per_sm = 0;
  size_t reserved_shared_per_block = 0;
  size_t total_global_mem = 0;
  size_t wavefront_static_bytes[2][2] = {};
  cudaError_t setup_error = cudaSuccess;
  std::once_flag set_up;
};

template <typename T, Pairs P, typename OutT, bool SquaredL2, bool Banded>
cudaError_t open_wavefront_one(DeviceLimits &device)
{
  const auto kernel = dtw_wavefront_fallback_kernel<T, P, OutT, SquaredL2, Banded, false>;
  cudaFuncAttributes attributes{};
  const cudaError_t error = cudaFuncGetAttributes(&attributes, kernel);
  if (error != cudaSuccess) return error;
  auto &slot = device.wavefront_static_bytes[static_cast<int>(P)][std::is_same_v<T, double>];
  slot = std::max(slot, attributes.sharedSizeBytes);
  return cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(device.max_shared_per_block - attributes.sharedSizeBytes));
}

template <typename T, Pairs P, typename OutT>
cudaError_t open_wavefront_variants(DeviceLimits &device)
{
  cudaError_t err = open_wavefront_one<T, P, OutT, false, false>(device);
  if (err == cudaSuccess) err = open_wavefront_one<T, P, OutT, false, true>(device);
  if (err == cudaSuccess) err = open_wavefront_one<T, P, OutT, true, false>(device);
  if (err == cudaSuccess) err = open_wavefront_one<T, P, OutT, true, true>(device);
  return err;
}

const DeviceLimits &device_limits(int device_id)
{
  static std::unique_ptr<DeviceLimits[]> limits;
  static cudaError_t read_error = cudaSuccess;
  static std::once_flag read_once;
  std::call_once(read_once, [] {
    const auto read = [](cudaDeviceAttr attribute, int device) {
      int value = 0;
      if (read_error == cudaSuccess)
        read_error = cudaDeviceGetAttribute(&value, attribute, device);
      return value;
    };
    int count = 0;
    read_error = cudaGetDeviceCount(&count);
    if (read_error != cudaSuccess) return;
    limits = std::make_unique<DeviceLimits[]>(static_cast<size_t>(count));
    for (int d = 0; d < count; ++d) {
      DeviceLimits &device = limits[static_cast<size_t>(d)];
      device.compute_major = read(cudaDevAttrComputeCapabilityMajor, d);
      device.compute_minor = read(cudaDevAttrComputeCapabilityMinor, d);
      device.slow_fp64 = read(cudaDevAttrSingleToDoublePrecisionPerfRatio, d) > 2;
      device.sm_count = read(cudaDevAttrMultiProcessorCount, d);
      device.max_shared_per_block =
          static_cast<size_t>(read(cudaDevAttrMaxSharedMemoryPerBlockOptin, d));
      device.l2_bytes = static_cast<size_t>(read(cudaDevAttrL2CacheSize, d));
      device.shared_per_sm =
          static_cast<size_t>(read(cudaDevAttrMaxSharedMemoryPerMultiprocessor, d));
      device.reserved_shared_per_block =
          static_cast<size_t>(read(cudaDevAttrReservedSharedMemoryPerBlock, d));
      cudaDeviceProp prop{};
      if (read_error == cudaSuccess) {
        read_error = cudaGetDeviceProperties(&prop, d);
        device.total_global_mem = prop.totalGlobalMem;
      }
    }
  });
  CUDA_CHECK(read_error);
  DeviceLimits &device = limits[static_cast<size_t>(device_id)];
  detail::require_compute_capability(device.compute_major, device.compute_minor, device_id);
  std::call_once(device.set_up, [&device] {
    using Open = cudaError_t (*)(DeviceLimits &);
    const Open opens[] = {
      open_wavefront_variants<float, Pairs::Triangle, double>,
      open_wavefront_variants<float, Pairs::Triangle, float>,
      open_wavefront_variants<double, Pairs::Triangle, double>,
      open_wavefront_variants<double, Pairs::Triangle, float>,
      open_wavefront_variants<float, Pairs::Rectangle, double>,
      open_wavefront_variants<double, Pairs::Rectangle, double>,
    };
    for (const Open open : opens)
      if (device.setup_error == cudaSuccess) device.setup_error = open(device);
  });
  CUDA_CHECK(device.setup_error);
  return device;
}

bool resolve_fp32(GpuPrecision precision, int device_id)
{
  switch (precision) {
  case GpuPrecision::FP32: return true;
  case GpuPrecision::FP64: return false;
  case GpuPrecision::Auto: return device_limits(device_id).slow_fp64;
  }
  throw std::logic_error("resolve_fp32: unreachable GpuPrecision");
}

} // namespace

// =========================================================================
// Primary Kernel: Unified Multi-Pass Warp Register-Tiled DTW (0 B smem)
// =========================================================================

constexpr int PAIRS_PER_BLOCK = 8; // 8 warps x 32 threads = 256 threads

template <typename T, int TILE_W, Pairs P, typename OutT, bool SquaredL2, bool Banded>
__global__ void
__launch_bounds__(256, (TILE_W == 1 && sizeof(T) == 4) ? 6 : (sizeof(T) == 4 ? 4 : 3))
dtw_warptile_kernel(
    const T *__restrict__ all_series,
    const int *__restrict__ lengths,
    OutT *__restrict__ out,
    int *__restrict__ d_overflow,
    int n, int max_L, int num_pairs, int band,
    int *__restrict__ work_counter,
    T *__restrict__ border_buf,
    std::int64_t first_pair, std::int64_t first_slot)
{
  constexpr int WARP_SIZE = 32;
  constexpr int PASS_COLS = WARP_SIZE * TILE_W;
  constexpr unsigned FULL_MASK = 0xFFFFFFFFu;

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane = threadIdx.x % WARP_SIZE;
  const int warp_global_id = static_cast<int>(blockIdx.x) * PAIRS_PER_BLOCK + warp_id;

  const T INF_VAL = std::is_same_v<T, float> ? static_cast<T>(FLT_MAX)
                                             : static_cast<T>(DBL_MAX);

  if constexpr (TILE_W == 1) {
    // Fast single-pass specialization for L <= 32: all M <= 32 row values and
    // N_len <= 32 col values reside in 2 registers across the warp, with 0 B smem.
    if (warp_global_id >= num_pairs) return;

    long long si_ll = 0, sj_ll = 0;
    if (lane == 0) {
      std::int64_t si0 = 0, sj0 = 0;
      pair_series<P, T>(first_pair + warp_global_id, n, si0, sj0);
      si_ll = static_cast<long long>(si0);
      sj_ll = static_cast<long long>(sj0);
    }
    const std::int64_t si = static_cast<std::int64_t>(__shfl_sync(FULL_MASK, si_ll, 0));
    const std::int64_t sj = static_cast<std::int64_t>(__shfl_sync(FULL_MASK, sj_ll, 0));
    const int ni = lengths[si];
    const int nj = lengths[sj];

    const T *const x = all_series + si * max_L;
    const T *const y = all_series + sj * max_L;
    const T *const row_s = (ni <= nj) ? x : y;
    const T *const col_s = (ni <= nj) ? y : x;
    const int M = min(ni, nj);
    const int N_len = max(ni, nj);

    if (M == 0 || N_len == 0 || (Banded && (N_len - M > band))) {
      if (lane == 0) {
        store_distance<T, OutT>(out, pair_slot<P>(si, sj, n) - first_slot, INF_VAL, d_overflow);
      }
      return;
    }

    const T my_row = (lane < M) ? __ldg(&row_s[lane]) : T(0);
    const T my_col = (lane < N_len) ? __ldg(&col_s[lane]) : T(0);
    T dp = INF_VAL;
    T prev_last = INF_VAL;
    const int total_steps = M + N_len - 1;

    for (int step = 0; step < total_steps; ++step) {
      const int i = step - lane;
      const T row_val = __shfl_sync(FULL_MASK, my_row, i & 31);
      T penalty_from_left = __shfl_up_sync(FULL_MASK, dp, 1);
      T diag_from_left = __shfl_up_sync(FULL_MASK, prev_last, 1);

      const bool row_valid = (i >= 0) && (i < M) && (lane < N_len);
      if (lane == 0 && step == 0) {
        penalty_from_left = INF_VAL;
        diag_from_left = T(0);
      } else if (lane == 0) {
        penalty_from_left = INF_VAL;
        diag_from_left = INF_VAL;
      }

      if (row_valid) {
        const T above = dp;
        T new_cost = INF_VAL;
        if constexpr (Banded) {
          if (abs(i - lane) <= band) {
            const T diff = row_val - my_col;
            const T d = SquaredL2 ? (diff * diff) : fabs(diff);
            new_cost = fmin(diag_from_left, fmin(above, penalty_from_left)) + d;
          }
        } else {
          const T diff = row_val - my_col;
          const T d = SquaredL2 ? (diff * diff) : fabs(diff);
          new_cost = fmin(diag_from_left, fmin(above, penalty_from_left)) + d;
        }
        prev_last = above;
        dp = new_cost;
      }
    }

    const T final_result = __shfl_sync(FULL_MASK, dp, N_len - 1);
    if (lane == 0) {
      store_distance<T, OutT>(out, pair_slot<P>(si, sj, n) - first_slot, final_result, d_overflow);
    }
    return;
  }

  T *const my_border = border_buf
      ? (border_buf + static_cast<std::int64_t>(warp_global_id) * max_L)
      : nullptr;

  while (true) {
    int pid = 0;
    if (work_counter) {
      if (lane == 0) pid = atomicAdd(work_counter, 1);
      pid = __shfl_sync(FULL_MASK, pid, 0);
      if (pid >= num_pairs) return;
    } else {
      pid = warp_global_id;
      if (pid >= num_pairs) return;
    }

    // Decode pair once on lane 0 and broadcast across the warp.
    long long si_ll = 0, sj_ll = 0;
    if (lane == 0) {
      std::int64_t si0 = 0, sj0 = 0;
      pair_series<P, T>(first_pair + pid, n, si0, sj0);
      si_ll = static_cast<long long>(si0);
      sj_ll = static_cast<long long>(sj0);
    }
    const std::int64_t si = static_cast<std::int64_t>(__shfl_sync(FULL_MASK, si_ll, 0));
    const std::int64_t sj = static_cast<std::int64_t>(__shfl_sync(FULL_MASK, sj_ll, 0));
    const int ni = lengths[si];
    const int nj = lengths[sj];

    const T *const x = all_series + si * max_L;
    const T *const y = all_series + sj * max_L;

    // Orient: rows = short side (M), columns = long side (N_len).
    const T *const row_s = (ni <= nj) ? x : y;
    const T *const col_s = (ni <= nj) ? y : x;
    const int M = min(ni, nj);
    const int N_len = max(ni, nj);

    if (M == 0 || N_len == 0 || (Banded && (N_len - M > band))) {
      if (lane == 0) {
        store_distance<T, OutT>(out, pair_slot<P>(si, sj, n) - first_slot, INF_VAL, d_overflow);
      }
      if (!work_counter) return;
      continue;
    }

    const int num_passes = (N_len + PASS_COLS - 1) / PASS_COLS;
    T dp[TILE_W];
    T col_val[TILE_W];

    for (int pass = 0; pass < num_passes; ++pass) {
      const int c0 = pass * PASS_COLS;
      const int c_end = min(N_len, c0 + PASS_COLS);
      const int col_start = c0 + lane * TILE_W;

      #pragma unroll
      for (int tw = 0; tw < TILE_W; ++tw) {
        const int j = col_start + tw;
        col_val[tw] = (j < c_end) ? __ldg(&col_s[j]) : T(0);
        dp[tw] = INF_VAL;
      }
      T prev_last = INF_VAL;

      int row_begin = 0;
      int row_end = M - 1;
      if constexpr (Banded) {
        row_begin = max(0, c0 - band);
        row_end = min(M - 1, c_end - 1 + band);
      }

      T prev_border = INF_VAL;
      if (pass > 0 && lane == 0 && row_begin > 0) {
        prev_border = my_border[row_begin - 1];
      }

      const int num_col_threads = (c_end - c0 + TILE_W - 1) / TILE_W;
      const int num_rows = row_end - row_begin + 1;
      const int total_steps = num_rows + num_col_threads - 1;

      T row_val = T(0);
      T row_buf = T(0);

      for (int step = 0; step < total_steps; ++step) {
        // Coalesced 32-element row load once every 32 steps into 1 register per lane.
        const int phase = step & 31;
        if (phase == 0) {
          const int load_i = row_begin + step + lane;
          row_buf = (load_i <= row_end) ? __ldg(&row_s[load_i]) : T(0);
        }
        const T new_row = __shfl_sync(FULL_MASK, row_buf, phase);
        const T shifted_row = __shfl_up_sync(FULL_MASK, row_val, 1);
        row_val = (lane == 0) ? new_row : shifted_row;

        // Inter-lane DP boundary shuffles (all 32 lanes participate unconditionally).
        T penalty_from_left = __shfl_up_sync(FULL_MASK, dp[TILE_W - 1], 1);
        T diag_from_left = __shfl_up_sync(FULL_MASK, prev_last, 1);

        const int i = row_begin + step - lane;
        const bool row_valid = (i >= row_begin) && (i <= row_end) && (lane < num_col_threads);

        if (lane == 0 && row_valid) {
          if (pass == 0) {
            penalty_from_left = INF_VAL;
            diag_from_left = (step == 0) ? T(0) : INF_VAL;
          } else {
            T b_val = INF_VAL;
            if constexpr (Banded) {
              if (abs(i - (c0 - 1)) <= band) {
                b_val = my_border[i];
              }
            } else {
              b_val = my_border[i];
            }
            penalty_from_left = b_val;
            diag_from_left = prev_border;
            prev_border = b_val;
          }
        }

        if (row_valid) {
          const T saved_prev_last = dp[TILE_W - 1];

          if constexpr (Banded) {
            if (i - (col_start + TILE_W - 1) > band || col_start - i > band) {
              #pragma unroll
              for (int tw = 0; tw < TILE_W; ++tw) dp[tw] = INF_VAL;
            } else {
              T left = penalty_from_left;
              T diag = diag_from_left;
              #pragma unroll
              for (int tw = 0; tw < TILE_W; ++tw) {
                const int j = col_start + tw;
                const T above = dp[tw];
                T new_cost = INF_VAL;
                if (abs(i - j) <= band) {
                  const T diff = row_val - col_val[tw];
                  const T d = SquaredL2 ? (diff * diff) : fabs(diff);
                  new_cost = fmin(diag, fmin(above, left)) + d;
                }
                diag = above;
                left = new_cost;
                dp[tw] = new_cost;
              }
            }
          } else {
            T left = penalty_from_left;
            T diag = diag_from_left;
            #pragma unroll
            for (int tw = 0; tw < TILE_W; ++tw) {
              const T above = dp[tw];
              const T diff = row_val - col_val[tw];
              const T d = SquaredL2 ? (diff * diff) : fabs(diff);
              const T new_cost = fmin(diag, fmin(above, left)) + d;
              diag = above;
              left = new_cost;
              dp[tw] = new_cost;
            }
          }

          prev_last = saved_prev_last;

          if (lane == 31 && (pass + 1 < num_passes)) {
            my_border[i] = dp[TILE_W - 1];
          }
        }
      }

      if (pass + 1 < num_passes) {
        __threadfence_block();
        __syncwarp();
      }
    }

    const int last_c0 = (num_passes - 1) * PASS_COLS;
    const int result_thread = (N_len - 1 - last_c0) / TILE_W;
    const int result_tw = (N_len - 1) % TILE_W;

    T my_result = INF_VAL;
    if (lane == result_thread) {
      #pragma unroll
      for (int tw = 0; tw < TILE_W; ++tw) {
        if (tw == result_tw) my_result = dp[tw];
      }
    }
    const T final_result = __shfl_sync(FULL_MASK, my_result, result_thread);

    if (lane == 0) {
      store_distance<T, OutT>(out, pair_slot<P>(si, sj, n) - first_slot, final_result, d_overflow);
    }

    if (!work_counter) return;
  }
}

// =========================================================================
// Fallback Kernel: 3-buffer anti-diagonal wavefront (native LDS/STS)
// =========================================================================

template <typename T, Pairs P, typename OutT, bool SquaredL2, bool Banded, bool UseGlobal>
__global__ void
__launch_bounds__(256)
dtw_wavefront_fallback_kernel(
    const T *__restrict__ all_series,
    const int *__restrict__ lengths,
    OutT *__restrict__ out,
    int *__restrict__ d_overflow,
    int n, int max_L, int num_pairs, int band,
    int *__restrict__ work_counter,
    std::int64_t first_pair, std::int64_t first_slot,
    T *__restrict__ scratch)
{
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const T INF = std::is_same_v<T, float> ? static_cast<T>(FLT_MAX)
                                         : static_cast<T>(DBL_MAX);

  extern __shared__ char smem_raw[];
  T *const buf = UseGlobal ? (scratch + static_cast<std::int64_t>(blockIdx.x) * 3 * max_L)
                           : reinterpret_cast<T *>(smem_raw);

  __shared__ int s_pid;
  __shared__ std::int64_t s_si, s_sj;

  while (true) {
    if (tid == 0) {
      const int p = work_counter ? atomicAdd(work_counter, 1) : static_cast<int>(blockIdx.x);
      s_pid = p;
      if (p < num_pairs) {
        std::int64_t si0 = 0, sj0 = 0;
        pair_series<P, T>(first_pair + p, n, si0, sj0);
        s_si = si0;
        s_sj = sj0;
      }
    }
    __syncthreads();

    const int pid = s_pid;
    if (pid >= num_pairs) return;

    const std::int64_t si = s_si;
    const std::int64_t sj = s_sj;
    const int ni = lengths[si];
    const int nj = lengths[sj];

    const T *const x = all_series + si * max_L;
    const T *const y = all_series + sj * max_L;

    const T *const row_s = (ni <= nj) ? x : y;
    const T *const col_s = (ni <= nj) ? y : x;
    const int M = min(ni, nj);
    const int N_len = max(ni, nj);

    if (M == 0 || N_len == 0 || (Banded && (N_len - M > band))) {
      if (tid == 0) {
        store_distance<T, OutT>(out, pair_slot<P>(si, sj, n) - first_slot, INF, d_overflow);
      }
      if (!work_counter) return;
      __syncthreads();
      continue;
    }

    int off_cur = 0;
    int off_prev = max_L;
    int off_prev2 = 2 * max_L;
    const int total_diags = M + N_len - 1;

    for (int k = 0; k < total_diags; ++k) {
      int i_min, i_max, lo1, hi1, lo2, hi2;
      wavefront_range<Banded>(k, M, N_len, band, i_min, i_max);
      const int len_k = i_max - i_min + 1;
      if (len_k > 0) {
        wavefront_range<Banded>(k - 1, M, N_len, band, lo1, hi1);
        wavefront_range<Banded>(k - 2, M, N_len, band, lo2, hi2);

        for (int p = tid; p < len_k; p += nthreads) {
          const int i = i_min + p;
          const int j = k - i;
          const T diff = __ldg(&row_s[i]) - __ldg(&col_s[j]);
          const T d = SquaredL2 ? (diff * diff) : fabs(diff);
          if (i == 0 && j == 0) {
            buf[off_cur + p] = d;
          } else {
            const int ra = i - 1;
            const T ca = (ra >= lo1 && ra <= hi1) ? buf[off_prev + (ra - lo1)] : INF;
            const T cl = (i >= lo1 && i <= hi1) ? buf[off_prev + (i - lo1)] : INF;
            const T cd = (ra >= lo2 && ra <= hi2) ? buf[off_prev2 + (ra - lo2)] : INF;
            buf[off_cur + p] = fmin(cd, fmin(ca, cl)) + d;
          }
        }
      }
      __syncthreads();

      const int old_prev2 = off_prev2;
      off_prev2 = off_prev;
      off_prev = off_cur;
      off_cur = old_prev2;
    }

    if (tid == 0) {
      const T dist = buf[off_prev];
      store_distance<T, OutT>(out, pair_slot<P>(si, sj, n) - first_slot, dist, d_overflow);
    }

    if (!work_counter) return;
    __syncthreads();
  }
}

// =========================================================================
// Medoid Argmin Reduction Kernel (GPU CLARA assignment)
// =========================================================================

__global__ void medoid_argmin_kernel(
    const double *__restrict__ d_dist,
    const int *__restrict__ d_slot_to_unique,
    const index_t *__restrict__ d_medoid_indices,
    index_t *__restrict__ out_labels,
    double *__restrict__ out_best,
    int *__restrict__ d_nonfinite,
    int count, int n_unique, int n_samples, int k,
    index_t first_point_index)
{
  const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int total = count * n_samples;
  if (idx >= total) return;

  const int s = idx / count;
  const int i = idx - s * count;
  const index_t global_p = first_point_index + static_cast<index_t>(i);
  const double *const row = d_dist + static_cast<std::int64_t>(i) * n_unique;
  const int base_slot = s * k;

  double best_dist = DBL_MAX;
  index_t best_label = 0;
  for (int m = 0; m < k; ++m) {
    const int slot = base_slot + m;
    const int u = d_slot_to_unique[slot];
    const bool is_self = (d_medoid_indices != nullptr) && (global_p == d_medoid_indices[slot]);
    const double d = is_self ? 0.0 : row[u];
    if (!isfinite(d) && d_nonfinite) {
      atomicExch(d_nonfinite, 1);
    }
    if (m == 0 || d < best_dist || (d == best_dist && is_self)) {
      best_dist = d;
      best_label = static_cast<index_t>(m);
    }
  }
  out_labels[static_cast<std::int64_t>(s) * count + i] = best_label;
  out_best[static_cast<std::int64_t>(s) * count + i] = best_dist;
}

// =========================================================================
// Host functions: availability, device info, refusal
// =========================================================================

bool cuda_available()
{
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  if (err != cudaSuccess) (void)cudaGetLastError();
  return (err == cudaSuccess && count > 0);
}

bool cuda_uses_fp32(GpuPrecision precision, int device_id)
{
  if (precision == GpuPrecision::FP32) return true;
  if (precision == GpuPrecision::FP64) return false;
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || device_id < 0 || device_id >= count) {
    (void)cudaGetLastError();
    return false;
  }
  int ratio = 0;
  if (cudaDeviceGetAttribute(&ratio, cudaDevAttrSingleToDoublePrecisionPerfRatio, device_id) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return ratio > 2;
}

std::string cuda_device_info(int device_id)
{
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    return "No CUDA device";
  }

  return std::string(prop.name) + " (compute " +
         std::to_string(prop.major) + "." + std::to_string(prop.minor) +
         ", " +
         std::to_string(prop.totalGlobalMem / (1024 * 1024)) + " MB)";
}

std::string cuda_device_refusal(int device_id)
{
  if (!cuda_available()) return detail::no_cuda_device_reason();
  const std::string device = "CUDA device " + std::to_string(device_id);
  int major = 0, minor = 0;
  cudaError_t err = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device_id);
  if (err == cudaSuccess) err = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device_id);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    return device + " cannot be queried: " + cudaGetErrorString(err);
  }
  if (const std::string too_old = detail::compute_capability_refusal(major, minor, device_id); !too_old.empty())
    return too_old;
  int previous = 0;
  const bool had_previous = cudaGetDevice(&previous) == cudaSuccess;
  if (!had_previous) (void)cudaGetLastError();
  const auto give_back = [&] {
    int current = previous;
    if (had_previous && cudaGetDevice(&current) == cudaSuccess && current != previous) (void)cudaSetDevice(previous);
    (void)cudaGetLastError();
  };
  err = cudaSetDevice(device_id);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    give_back();
    return device + " cannot be selected: " + cudaGetErrorString(err);
  }
  err = cudaFree(nullptr);
  if (err != cudaSuccess) {
    (void)cudaGetLastError();
    give_back();
    return device + " cannot start a context: " + cudaGetErrorString(err);
  }
  give_back();
  return {};
}

// =========================================================================
// Workspace and Launch Pipeline
// =========================================================================

namespace {

constexpr size_t PINNED_THRESHOLD = 256 * 1024;
constexpr size_t D2H_SLAB_BYTES = 16ULL << 20; // 16 MiB per pinned D2H slab (double-buffered)

template <typename T>
struct CachedHostBuffer {
  size_t capacity = 0;
  PinnedPtr<T> pinned;
  std::vector<T> fallback;
  bool warned = false;

  T *ensure(size_t count, bool prefer_pinned)
  {
    if (pinned && capacity >= count)
      return pinned.get();

    if (prefer_pinned) {
      auto new_pinned = pinned_alloc_nothrow<T>(count);
      if (new_pinned) {
        pinned = std::move(new_pinned);
        fallback.clear();
        capacity = count;
        warned = false;
        return pinned.get();
      }
      if (!warned) {
        std::cerr << "[dtwc] warning: CUDA could not pin " << count * sizeof(T)
                  << " bytes of host memory; the run continues on pageable memory, with slower copies and the same "
                     "results.\n";
        warned = true;
      }
    }

    pinned.reset();
    if (fallback.size() < count)
      fallback.resize(count);
    capacity = count;
    return fallback.data();
  }
};

template <typename T>
struct DTWLaunchWorkspace {
  int device_id = -1;
  size_t series_capacity = 0;
  size_t length_capacity = 0;
  size_t out_byte_capacity = 0;
  size_t counter_capacity = 0;
  size_t scratch_capacity = 0;
  size_t overflow_capacity = 0;
  size_t slot_map_capacity = 0;
  size_t medoid_idx_capacity = 0;
  size_t argmin_capacity = 0;

  CachedHostBuffer<T> host_series;
  CachedHostBuffer<int> host_lengths;
  CachedHostBuffer<double> host_out;
  CachedHostBuffer<int> host_overflow;
  std::array<CachedHostBuffer<std::byte>, 2> host_d2h;
  CachedHostBuffer<index_t> host_argmin_labels;
  CachedHostBuffer<double> host_argmin_dists;

  CudaPtr<T> d_series;
  CudaPtr<int> d_lengths;
  CudaPtr<std::byte> d_out;
  CudaPtr<int> d_counter;
  CudaPtr<T> d_scratch;
  CudaPtr<int> d_overflow;
  CudaPtr<int> d_slot_to_unique;
  CudaPtr<index_t> d_medoid_indices;
  CudaPtr<index_t> d_argmin_labels;
  CudaPtr<double> d_argmin_dists;

  CudaStream stream;
  CudaStream copy_stream;
  CudaStream d2h_stream;
  CudaEvent evt_start;
  CudaEvent evt_end;
  CudaEvent d2h_computed;
  std::array<CudaEvent, 2> uploaded, computed, d_out_free, d2h_done;
  std::array<CudaEvent, 3> downloaded;

  void release() noexcept
  {
    host_series = CachedHostBuffer<T>{};
    host_lengths = CachedHostBuffer<int>{};
    host_out = CachedHostBuffer<double>{};
    host_overflow = CachedHostBuffer<int>{};
    host_d2h[0] = CachedHostBuffer<std::byte>{};
    host_d2h[1] = CachedHostBuffer<std::byte>{};
    host_argmin_labels = CachedHostBuffer<index_t>{};
    host_argmin_dists = CachedHostBuffer<double>{};
    reset_events();
    copy_stream.reset();
    d2h_stream.reset();
    d_series.reset();
    d_lengths.reset();
    d_out.reset();
    d_counter.reset();
    d_scratch.reset();
    d_overflow.reset();
    d_slot_to_unique.reset();
    d_medoid_indices.reset();
    d_argmin_labels.reset();
    d_argmin_dists.reset();
    stream.reset();
    evt_start.reset();
    evt_end.reset();
    series_capacity = length_capacity = out_byte_capacity = counter_capacity = scratch_capacity = 0;
    overflow_capacity = slot_map_capacity = medoid_idx_capacity = argmin_capacity = 0;
    device_id = -1;
  }

  void ensure_runtime(int new_device_id)
  {
    if (device_id != new_device_id) {
      d_series.reset();
      d_lengths.reset();
      d_out.reset();
      d_counter.reset();
      d_scratch.reset();
      d_overflow.reset();
      d_slot_to_unique.reset();
      d_medoid_indices.reset();
      d_argmin_labels.reset();
      d_argmin_dists.reset();
      stream.reset();
      copy_stream.reset();
      d2h_stream.reset();
      evt_start.reset();
      evt_end.reset();
      reset_events();
      series_capacity = length_capacity = out_byte_capacity = counter_capacity = scratch_capacity = 0;
      overflow_capacity = slot_map_capacity = medoid_idx_capacity = argmin_capacity = 0;
      device_id = new_device_id;
    }

    if (!stream) stream = make_cuda_stream();
    if (!evt_start) evt_start = make_cuda_event(false); // timed event
    if (!evt_end) evt_end = make_cuda_event(false);     // timed event
  }

  void ensure_d2h()
  {
    if (!d2h_stream) d2h_stream = make_cuda_stream();
    if (!d2h_computed) d2h_computed = make_cuda_event(true);
    for (auto &event : d2h_done)
      if (!event) event = make_cuda_event(true);
  }

  void ensure_pipeline()
  {
    if (!copy_stream) copy_stream = make_cuda_stream();
    ensure_d2h();
    const auto make = [](auto &events) {
      for (auto &event : events)
        if (!event) event = make_cuda_event(true);
    };
    make(uploaded);
    make(computed);
    make(d_out_free);
    make(downloaded);
  }

  void reset_events() noexcept
  {
    d2h_computed.reset();
    for (auto &event : uploaded) event.reset();
    for (auto &event : computed) event.reset();
    for (auto &event : d_out_free) event.reset();
    for (auto &event : d2h_done) event.reset();
    for (auto &event : downloaded) event.reset();
  }
};

template <typename T>
DTWLaunchWorkspace<T> &workspace_of()
{
  thread_local DTWLaunchWorkspace<T> workspace;
  return workspace;
}

template <typename T>
std::size_t workspace_bytes(const DTWLaunchWorkspace<T> &w)
{
  const auto host = [](const auto &buffer) {
    using Elem = std::remove_pointer_t<decltype(buffer.fallback.data())>;
    return (buffer.pinned ? buffer.capacity : buffer.fallback.capacity()) * sizeof(Elem);
  };
  return host(w.host_series) + host(w.host_lengths) + host(w.host_out)
         + host(w.host_overflow) + host(w.host_d2h[0]) + host(w.host_d2h[1])
         + host(w.host_argmin_labels) + host(w.host_argmin_dists)
         + w.series_capacity * sizeof(T) + w.length_capacity * sizeof(int)
         + w.out_byte_capacity + w.counter_capacity * sizeof(int)
         + w.scratch_capacity * sizeof(T) + w.overflow_capacity * sizeof(int)
         + w.slot_map_capacity * sizeof(int) + w.medoid_idx_capacity * sizeof(index_t)
         + w.argmin_capacity * (sizeof(index_t) + sizeof(double));
}

template <typename T>
DTWLaunchWorkspace<T> &get_dtw_launch_workspace(int device_id)
{
  DTWLaunchWorkspace<T> &workspace = workspace_of<T>();
  workspace.ensure_runtime(device_id);
  return workspace;
}

template <typename T>
void ensure_dtw_device_capacity(
    DTWLaunchWorkspace<T> &workspace,
    size_t series_elems,
    size_t length_elems,
    size_t out_bytes,
    size_t scratch_elems,
    size_t counter_elems,
    size_t overflow_elems = 1)
{
  if (workspace.series_capacity < series_elems) {
    workspace.d_series = cuda_alloc<T>(series_elems);
    workspace.series_capacity = series_elems;
  }
  if (workspace.length_capacity < length_elems) {
    workspace.d_lengths = cuda_alloc<int>(length_elems);
    workspace.length_capacity = length_elems;
  }
  if (workspace.out_byte_capacity < out_bytes) {
    workspace.d_out = cuda_alloc<std::byte>(out_bytes);
    workspace.out_byte_capacity = out_bytes;
  }
  if (workspace.scratch_capacity < scratch_elems) {
    workspace.d_scratch = cuda_alloc<T>(scratch_elems);
    workspace.scratch_capacity = scratch_elems;
  }
  if (workspace.counter_capacity < counter_elems) {
    workspace.d_counter = cuda_alloc<int>(counter_elems);
    workspace.counter_capacity = counter_elems;
  }
  if (workspace.overflow_capacity < overflow_elems) {
    workspace.d_overflow = cuda_alloc<int>(overflow_elems);
    workspace.overflow_capacity = overflow_elems;
  }
}

template <typename T>
int pad_series(T *dst, std::span<const double> src, size_t max_L)
{
  const size_t len = src.size();
  if constexpr (std::is_same_v<T, double>) {
    if (len > 0)
      std::memcpy(dst, src.data(), len * sizeof(double));
  } else {
    for (size_t k = 0; k < len; ++k)
      dst[k] = static_cast<T>(src[k]);
  }
  std::fill(dst + len, dst + max_L, T(0));
  return static_cast<int>(len);
}

/// Exact scalar double DTW for rare Float32 overflow error reporting.
double exact_dtw_double(std::span<const double> a, std::span<const double> b,
                        int band, bool squared_l2)
{
  if (a.empty() || b.empty()) return DBL_MAX;
  if (a.size() > b.size()) std::swap(a, b);
  const size_t M = a.size(), N_len = b.size();
  if (band >= 0 && N_len - M > static_cast<size_t>(band)) return DBL_MAX;
  std::vector<double> prev(N_len, DBL_MAX), cur(N_len, DBL_MAX);
  for (size_t i = 0; i < M; ++i) {
    const size_t j0 = (band >= 0 && i > static_cast<size_t>(band)) ? (i - static_cast<size_t>(band)) : 0;
    const size_t j1 = (band >= 0) ? std::min(N_len - 1, i + static_cast<size_t>(band)) : (N_len - 1);
    std::fill(cur.begin(), cur.end(), DBL_MAX);
    for (size_t j = j0; j <= j1; ++j) {
      const double diff = a[i] - b[j];
      const double d = squared_l2 ? (diff * diff) : std::abs(diff);
      if (i == 0 && j == 0) {
        cur[j] = d;
      } else {
        const double above = (i > 0) ? prev[j] : DBL_MAX;
        const double left = (j > 0) ? cur[j - 1] : DBL_MAX;
        const double diag = (i > 0 && j > 0) ? prev[j - 1] : DBL_MAX;
        cur[j] = std::min(diag, std::min(above, left)) + d;
      }
    }
    prev.swap(cur);
  }
  return prev[N_len - 1];
}

/// Dispatch helper resolving compile-time SquaredL2 & Banded specializations.
template <typename F>
void dispatch_bools(bool squared_l2, bool banded, F &&fn)
{
  if (squared_l2) {
    if (banded) fn(std::true_type{}, std::true_type{});
    else fn(std::true_type{}, std::false_type{});
  } else {
    if (banded) fn(std::false_type{}, std::true_type{});
    else fn(std::false_type{}, std::false_type{});
  }
}

template <typename T>
int warptile_blocks_per_sm(int tile_w, Pairs pairs, bool out_f32, bool squared_l2, bool banded)
{
  int blocks_per_sm = 0;
  const auto query = [&](auto out_tag, auto sq_tag, auto band_tag) {
    using OutT = decltype(out_tag);
    constexpr bool sq = decltype(sq_tag)::value;
    constexpr bool bd = decltype(band_tag)::value;
    if (pairs == Pairs::Triangle) {
      if (tile_w == 1)
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, dtw_warptile_kernel<T, 1, Pairs::Triangle, OutT, sq, bd>, 256, 0));
      else if (tile_w == 4 || !std::is_same_v<T, float>)
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, dtw_warptile_kernel<T, 4, Pairs::Triangle, OutT, sq, bd>, 256, 0));
      else if constexpr (std::is_same_v<T, float>)
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, dtw_warptile_kernel<T, 8, Pairs::Triangle, OutT, sq, bd>, 256, 0));
    } else {
      if constexpr (std::is_same_v<OutT, double>) {
        if (tile_w == 1)
          CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &blocks_per_sm, dtw_warptile_kernel<T, 1, Pairs::Rectangle, double, sq, bd>, 256, 0));
        else if (tile_w == 4 || !std::is_same_v<T, float>)
          CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &blocks_per_sm, dtw_warptile_kernel<T, 4, Pairs::Rectangle, double, sq, bd>, 256, 0));
        else if constexpr (std::is_same_v<T, float>)
          CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
              &blocks_per_sm, dtw_warptile_kernel<T, 8, Pairs::Rectangle, double, sq, bd>, 256, 0));
      }
    }
  };
  dispatch_bools(squared_l2, banded, [&](auto sq_tag, auto band_tag) {
    if (out_f32) query(float{}, sq_tag, band_tag);
    else query(double{}, sq_tag, band_tag);
  });
  return std::max(blocks_per_sm, 1);
}

template <typename T, Pairs P>
struct Launches {
  static constexpr int block_size = 256;
  detail::KernelPath path;
  int max_L;
  bool use_squared_l2;
  int band;
  bool out_f32;
  int tile_w = 4;
  int warptile_persistent_blocks = 0;
  int warptile_active_warps = 0;
  bool warptile_multipass = false;
  size_t wavefront_shared_mem = 0;
  int wavefront_persistent_grid = 0;
  int wavefront_global_grid = 0;

  Launches(detail::KernelPath kernel_path, size_t max_length, bool squared_l2, int band_width,
           bool output_f32, std::int64_t chunk, const DeviceLimits &device)
      : path(kernel_path), max_L(static_cast<int>(max_length)), use_squared_l2(squared_l2),
        band(band_width), out_f32(output_f32)
  {
    const bool banded = (band >= 0 && band < max_L - 1);
    if (path == detail::KernelPath::WarpTileW1 || path == detail::KernelPath::WarpTileW4
        || path == detail::KernelPath::WarpTileW8) {
      tile_w = (path == detail::KernelPath::WarpTileW1) ? 1
             : (path == detail::KernelPath::WarpTileW8 ? 8 : 4);
      warptile_multipass = (max_L > 32 * tile_w);
      if (warptile_multipass) {
        const int bpsm = warptile_blocks_per_sm<T>(tile_w, P, out_f32, use_squared_l2, banded);
        const std::int64_t max_blocks = static_cast<std::int64_t>(bpsm) * device.sm_count;
        const std::int64_t needed_blocks = (chunk + PAIRS_PER_BLOCK - 1) / PAIRS_PER_BLOCK;
        warptile_persistent_blocks = static_cast<int>(std::min(max_blocks, std::max<std::int64_t>(1, needed_blocks)));
        warptile_active_warps = warptile_persistent_blocks * PAIRS_PER_BLOCK;
      }
      return;
    }

    const bool global = (path == detail::KernelPath::WavefrontGlobal);
    wavefront_shared_mem = global ? 0 : static_cast<size_t>(3) * max_length * sizeof(T);
    int blocks_per_sm = 0;
    const auto query_wf = [&](auto out_tag, auto sq_tag, auto band_tag, auto global_tag) {
      using OutT = decltype(out_tag);
      constexpr bool sq = decltype(sq_tag)::value;
      constexpr bool bd = decltype(band_tag)::value;
      constexpr bool gl = decltype(global_tag)::value;
      if constexpr (P == Pairs::Triangle || std::is_same_v<OutT, double>) {
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm,
            dtw_wavefront_fallback_kernel<T, P, OutT, sq, bd, gl>,
            block_size, gl ? 0 : wavefront_shared_mem));
      }
    };
    dispatch_bools(use_squared_l2, banded, [&](auto sq_tag, auto band_tag) {
      if (out_f32 && P == Pairs::Triangle) {
        if (global) query_wf(float{}, sq_tag, band_tag, std::true_type{});
        else query_wf(float{}, sq_tag, band_tag, std::false_type{});
      } else {
        if (global) query_wf(double{}, sq_tag, band_tag, std::true_type{});
        else query_wf(double{}, sq_tag, band_tag, std::false_type{});
      }
    });
    wavefront_persistent_grid = device.sm_count * std::max(blocks_per_sm, 1);
    if (global) {
      const auto l2_slices =
          static_cast<std::int64_t>(device.l2_bytes / (3 * max_length * sizeof(T)));
      wavefront_global_grid = static_cast<int>(std::min(
          std::min<std::int64_t>(chunk, wavefront_persistent_grid),
          std::max<std::int64_t>(device.sm_count, l2_slices)));
    }
  }

  size_t scratch_elems() const
  {
    if (path == detail::KernelPath::WarpTileW1 || path == detail::KernelPath::WarpTileW4
        || path == detail::KernelPath::WarpTileW8) {
      return warptile_multipass ? static_cast<size_t>(warptile_active_warps) * max_L : 0;
    }
    if (path == detail::KernelPath::WavefrontGlobal) {
      return static_cast<size_t>(wavefront_global_grid) * 3 * max_L;
    }
    return 0;
  }

  size_t counter_elems() const
  {
    if (path == detail::KernelPath::WarpTileW1 || path == detail::KernelPath::WarpTileW4
        || path == detail::KernelPath::WarpTileW8) {
      return warptile_multipass ? 1 : 0;
    }
    return wavefront_persistent_grid > 0 ? 1 : 0;
  }

  template <typename OutT>
  void launch_typed(DTWLaunchWorkspace<T> &workspace, const T *series, const int *lengths, int n,
                    std::int64_t first, int count, std::int64_t first_slot, cudaStream_t stream,
                    OutT *out_ptr, int *d_overflow) const
  {
    // Normalize bands >= max_L - 1 to unbanded (-1) for zero-branch unbanded inner loop.
    const bool banded = (band >= 0 && band < max_L - 1);
    const int eff_band = banded ? band : -1;

    dispatch_bools(use_squared_l2, banded, [&](auto sq_tag, auto band_tag) {
      constexpr bool sq = decltype(sq_tag)::value;
      constexpr bool bd = decltype(band_tag)::value;

      if (path == detail::KernelPath::WarpTileW1 || path == detail::KernelPath::WarpTileW4
          || path == detail::KernelPath::WarpTileW8) {
        const int grid_size = warptile_multipass
            ? warptile_persistent_blocks
            : static_cast<int>((std::int64_t{ count } + PAIRS_PER_BLOCK - 1) / PAIRS_PER_BLOCK);
        int *counter_ptr = nullptr;
        T *border_ptr = nullptr;
        if (warptile_multipass) {
          CUDA_CHECK(cudaMemsetAsync(workspace.d_counter.get(), 0, sizeof(int), stream));
          counter_ptr = workspace.d_counter.get();
          border_ptr = workspace.d_scratch.get();
        }
        if (path == detail::KernelPath::WarpTileW1) {
          dtw_warptile_kernel<T, 1, P, OutT, sq, bd><<<grid_size, block_size, 0, stream>>>(
              series, lengths, out_ptr, d_overflow, n, max_L, count, eff_band,
              counter_ptr, border_ptr, first, first_slot);
        } else if (path == detail::KernelPath::WarpTileW4 || !std::is_same_v<T, float>) {
          dtw_warptile_kernel<T, 4, P, OutT, sq, bd><<<grid_size, block_size, 0, stream>>>(
              series, lengths, out_ptr, d_overflow, n, max_L, count, eff_band,
              counter_ptr, border_ptr, first, first_slot);
        } else if constexpr (std::is_same_v<T, float>) {
          dtw_warptile_kernel<T, 8, P, OutT, sq, bd><<<grid_size, block_size, 0, stream>>>(
              series, lengths, out_ptr, d_overflow, n, max_L, count, eff_band,
              counter_ptr, border_ptr, first, first_slot);
        }
      } else if (path == detail::KernelPath::Wavefront) {
        const bool persistent = count > wavefront_persistent_grid * 4;
        if (persistent)
          CUDA_CHECK(cudaMemsetAsync(workspace.d_counter.get(), 0, sizeof(int), stream));
        dtw_wavefront_fallback_kernel<T, P, OutT, sq, bd, false>
            <<<persistent ? wavefront_persistent_grid : count, block_size, wavefront_shared_mem, stream>>>(
                series, lengths, out_ptr, d_overflow, n, max_L, count, eff_band,
                persistent ? workspace.d_counter.get() : nullptr, first, first_slot, nullptr);
      } else if (path == detail::KernelPath::WavefrontGlobal) {
        CUDA_CHECK(cudaMemsetAsync(workspace.d_counter.get(), 0, sizeof(int), stream));
        dtw_wavefront_fallback_kernel<T, P, OutT, sq, bd, true>
            <<<wavefront_global_grid, block_size, 0, stream>>>(
                series, lengths, out_ptr, d_overflow, n, max_L, count, eff_band,
                workspace.d_counter.get(), first, first_slot, workspace.d_scratch.get());
      } else {
        throw std::logic_error("Launches: unknown KernelPath");
      }
    });
    CUDA_CHECK(cudaGetLastError());
  }

  void operator()(DTWLaunchWorkspace<T> &workspace, const T *series, const int *lengths, int n,
                  std::int64_t first, int count, std::int64_t first_slot, cudaStream_t stream,
                  void *d_out_dest = nullptr, int *d_overflow = nullptr) const
  {
    void *const raw_out = d_out_dest ? d_out_dest : static_cast<void *>(workspace.d_out.get());
    if constexpr (P == Pairs::Triangle) {
      if (out_f32) {
        launch_typed<float>(workspace, series, lengths, n, first, count, first_slot, stream,
                            static_cast<float *>(raw_out), d_overflow);
      } else {
        launch_typed<double>(workspace, series, lengths, n, first, count, first_slot, stream,
                             static_cast<double *>(raw_out), d_overflow);
      }
    } else {
      launch_typed<double>(workspace, series, lengths, n, first, count, first_slot, stream,
                           static_cast<double *>(raw_out), d_overflow);
    }
  }
};

/// Select precision and kernel path, checking border-buffer VRAM budget for multi-pass warptile.
std::pair<bool, detail::KernelPath> select_route(size_t max_L, size_t num_pairs,
                                                 const CUDADistMatOptions &opts, Pairs pairs,
                                                 bool out_f32, bool &warptile_border_fits)
{
  const bool use_fp32 = resolve_fp32(opts.precision, opts.device_id);
  const auto &device = device_limits(opts.device_id);
  const size_t sample_bytes = use_fp32 ? sizeof(float) : sizeof(double);
  const int tile_w = (max_L <= 32) ? 1 : ((use_fp32 && max_L > 128) ? 8 : 4);

  warptile_border_fits = true;
  if (max_L > static_cast<size_t>(32 * tile_w)) {
    const bool banded = (opts.band >= 0 && static_cast<size_t>(opts.band) < max_L - 1);
    const int bpsm = use_fp32
        ? warptile_blocks_per_sm<float>(tile_w, pairs, out_f32, opts.use_squared_l2, banded)
        : warptile_blocks_per_sm<double>(tile_w, pairs, out_f32, opts.use_squared_l2, banded);
    const size_t active_warps = static_cast<size_t>(bpsm) * device.sm_count * PAIRS_PER_BLOCK;
    const size_t border_bytes = active_warps * max_L * sample_bytes;
    size_t free_bytes = 0, total_bytes = 0;
    if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
      (void)cudaGetLastError();
      free_bytes = device.total_global_mem;
    }
    const size_t budget = std::max<size_t>(free_bytes / 2, device.total_global_mem / 4);
    warptile_border_fits = (border_bytes <= budget);
  }

  const size_t overhead =
      device.wavefront_static_bytes[static_cast<int>(pairs)][use_fp32 ? 0 : 1]
      + device.reserved_shared_per_block;

  return { use_fp32,
           detail::select_kernel(max_L, sample_bytes, num_pairs, device.sm_count,
                                 warptile_border_fits, device.max_shared_per_block, overhead) };
}

template <typename T>
struct TriangleFill {
  std::int64_t n;
  size_t max_L;
  std::int64_t chunk;
  SeriesAt series_;
  ParallelFor parallel_;
  bool use_squared_l2;
  int band;
  bool out_f32;
  Launches<T, Pairs::Triangle> launches;
  DTWLaunchWorkspace<T> &workspace;
  int device_id;
  bool timed_from_upload = true;

  TriangleFill(size_t N, const SeriesAt &series, const ParallelFor &parallel, size_t max_length,
               bool squared_l2, int band_width, int dev_id, detail::KernelPath kernel_path,
               core::DistanceMatrix &out)
      : n(static_cast<std::int64_t>(N)), max_L(max_length),
        chunk(std::min(n * (n - 1) / 2, detail::kMaxPairsPerLaunch)),
        series_(series), parallel_(parallel), use_squared_l2(squared_l2), band(band_width),
        out_f32(out.width() == core::Precision::Float32),
        launches(kernel_path, max_L, squared_l2, band_width, out_f32, chunk, device_limits(dev_id)),
        workspace(get_dtw_launch_workspace<T>(dev_id)), device_id(dev_id)
  {
    const size_t span_elems = static_cast<size_t>(chunk) + N;
    const size_t out_elem_bytes = out_f32 ? sizeof(float) : sizeof(double);
    ensure_dtw_device_capacity(workspace, N * max_L, N, span_elems * out_elem_bytes,
                               launches.scratch_elems(), launches.counter_elems(), 1);
    workspace.ensure_d2h();

    const size_t series_bytes = N * max_L * sizeof(T);
    T *const h_flat_series = workspace.host_series.ensure(
        N * max_L, series_bytes >= PINNED_THRESHOLD);
    int *const h_lengths = workspace.host_lengths.ensure(N, N * sizeof(int) >= PINNED_THRESHOLD);
    parallel_(N, [&](size_t d) { h_lengths[d] = pad_series(h_flat_series + d * max_L, series_(N - 1 - d), max_L); });

    if (out.size() != N) out.resize(N);

    const cudaStream_t stream = workspace.stream.get();
    try {
      CUDA_CHECK(cudaEventRecord(workspace.evt_start.get(), stream));
      CUDA_CHECK(cudaMemcpyAsync(workspace.d_series.get(), h_flat_series,
                                 series_bytes, cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaMemcpyAsync(workspace.d_lengths.get(), h_lengths,
                                 N * sizeof(int), cudaMemcpyHostToDevice, stream));
    } catch (...) {
      (void)cudaStreamSynchronize(stream);
      (void)cudaGetLastError();
      throw;
    }
  }

  TriangleFill(const TriangleFill &) = delete;
  TriangleFill &operator=(const TriangleFill &) = delete;
  ~TriangleFill()
  {
    (void)cudaStreamSynchronize(workspace.stream.get());
    if (workspace.d2h_stream) (void)cudaStreamSynchronize(workspace.d2h_stream.get());
  }

  void rows(size_t r0, size_t r1, core::DistanceMatrix &out, size_t &pairs, double &gpu_time_sec)
  {
    const auto pairs_before = [this](std::int64_t s) { return s * (n - 1) - s * (s - 1) / 2; };
    const auto R0 = static_cast<std::int64_t>(r0), R1 = static_cast<std::int64_t>(r1);
    const std::int64_t begin = pairs_before(n - R1), last = pairs_before(n - R0);
    if (begin == last) {
      for (size_t r = r0; r < r1; ++r) out.set(r, r, 0.0);
      return;
    }
    CUDA_CHECK(cudaSetDevice(device_id));
    const cudaStream_t stream = workspace.stream.get();
    const cudaStream_t d2h = workspace.d2h_stream.get();

    bool overflowed = false;
    try {
      if (!timed_from_upload) CUDA_CHECK(cudaEventRecord(workspace.evt_start.get(), stream));
      timed_from_upload = false;

      if (out_f32) {
        CUDA_CHECK(cudaMemsetAsync(workspace.d_overflow.get(), 0, sizeof(int), stream));
      }

      std::int64_t end = R1 * (R1 + 1) / 2;
      std::int64_t diag_r = R1 - 1;

      for (std::int64_t first = begin; first < last; first += chunk) {
        const int count = static_cast<int>(std::min(chunk, last - first));
        std::int64_t first_slot = R0 * (R0 + 1) / 2;
        if (first + count < last) {
          std::int64_t si = 0, sj = 0;
          decode_pair(first + count - 1, n, si, sj);
          first_slot = packed_slot(si, sj, n);
        }
        const size_t span = static_cast<size_t>(end - first_slot);

        // Ensure previous launch's D2H from d_out has completed before this launch overwrites d_out.
        if (first > begin) {
          CUDA_CHECK(cudaStreamWaitEvent(stream, workspace.d2h_done[0].get(), 0));
          CUDA_CHECK(cudaStreamWaitEvent(stream, workspace.d2h_done[1].get(), 0));
        }

        launches(workspace, workspace.d_series.get(), workspace.d_lengths.get(), static_cast<int>(n),
                 first, count, first_slot, stream, nullptr,
                 out_f32 ? workspace.d_overflow.get() : nullptr);
        CUDA_CHECK(cudaEventRecord(workspace.d2h_computed.get(), stream));
        CUDA_CHECK(cudaStreamWaitEvent(d2h, workspace.d2h_computed.get(), 0));

        // Double-buffered pinned D2H staging + parallel memcpy into DistanceMatrix (heap or mmap),
        // zeroing only the <= N diagonal entries of [r0, r1) inside the staged slabs.
        out.visit([&](auto *values) {
          using V = std::remove_pointer_t<decltype(values)>;
          const size_t slab_elems = std::max<size_t>(1, D2H_SLAB_BYTES / sizeof(V));
          const size_t num_slabs = (span + slab_elems - 1) / slab_elems;
          const size_t alloc_elems = std::min(span, slab_elems);
          std::array<V *, 2> h_slabs{
            reinterpret_cast<V *>(workspace.host_d2h[0].ensure(alloc_elems * sizeof(V), true)),
            num_slabs > 1
                ? reinterpret_cast<V *>(workspace.host_d2h[1].ensure(alloc_elems * sizeof(V), true))
                : nullptr
          };
          const V *const d_src = reinterpret_cast<const V *>(workspace.d_out.get());

          // Copy slabs from highest address down (matching diag_r stepping down) or 0..num_slabs-1.
          // Going backwards (num_slabs-1 down to 0) walks diag_r monotonically without searching.
          const auto enqueue_slab = [&](size_t s_idx) {
            const size_t off = s_idx * slab_elems;
            const size_t len = std::min(slab_elems, span - off);
            const size_t buf_id = s_idx & 1;
            CUDA_CHECK(cudaMemcpyAsync(h_slabs[buf_id], d_src + off, len * sizeof(V),
                                       cudaMemcpyDeviceToHost, d2h));
            CUDA_CHECK(cudaEventRecord(workspace.d2h_done[buf_id].get(), d2h));
          };

          if (num_slabs > 0) enqueue_slab(num_slabs - 1);
          if (num_slabs > 1) enqueue_slab(num_slabs - 2);

          for (size_t step_b = 0; step_b < num_slabs; ++step_b) {
            const size_t s_idx = num_slabs - 1 - step_b;
            const size_t off = s_idx * slab_elems;
            const size_t len = std::min(slab_elems, span - off);
            const size_t buf_id = s_idx & 1;

            CUDA_CHECK(cudaEventSynchronize(workspace.d2h_done[buf_id].get()));

            V *const h_buf = h_slabs[buf_id];
            const std::int64_t slab_lo = first_slot + static_cast<std::int64_t>(off);
            const std::int64_t slab_hi = slab_lo + static_cast<std::int64_t>(len);
            while (diag_r >= R0) {
              const std::int64_t d_slot = diag_r * (diag_r + 1) / 2 + diag_r;
              if (d_slot < slab_lo) break;
              if (d_slot < slab_hi) {
                h_buf[d_slot - slab_lo] = V(0);
              }
              --diag_r;
            }

            V *const dst = values + slab_lo;
            const size_t copy_bytes = len * sizeof(V);
            if (copy_bytes >= (size_t{ 1 } << 20)) {
              // > 64 tasks so Problem's parallel lambda engages OpenMP threads.
              constexpr size_t kCopyChunks = 128;
              parallel_(kCopyChunks, [&](size_t c) {
                const size_t c0 = len * c / kCopyChunks;
                const size_t c1 = len * (c + 1) / kCopyChunks;
                if (c1 > c0) std::memcpy(dst + c0, h_buf + c0, (c1 - c0) * sizeof(V));
              });
            } else {
              std::memcpy(dst, h_buf, copy_bytes);
            }

            if (step_b + 2 < num_slabs) enqueue_slab(num_slabs - 1 - (step_b + 2));
          }
        });

        end = first_slot;
      }

      if (out_f32) {
        int *const h_ov = workspace.host_overflow.ensure(1, true);
        CUDA_CHECK(cudaMemcpyAsync(h_ov, workspace.d_overflow.get(), sizeof(int),
                                   cudaMemcpyDeviceToHost, d2h));
        CUDA_CHECK(cudaEventRecord(workspace.evt_end.get(), d2h));
        CUDA_CHECK(cudaEventSynchronize(workspace.evt_end.get()));
        overflowed = (*h_ov != 0);
      } else {
        CUDA_CHECK(cudaEventRecord(workspace.evt_end.get(), d2h));
        CUDA_CHECK(cudaEventSynchronize(workspace.evt_end.get()));
      }
    } catch (...) {
      (void)cudaStreamSynchronize(stream);
      (void)cudaStreamSynchronize(d2h);
      (void)cudaGetLastError();
      throw;
    }

    if (overflowed) {
      const auto [oi, oj] = out.first_infinite(r0, r1);
      size_t err_i = oi, err_j = oj;
      if (err_i == out.size()) {
        // Fallback scan in case a non-finite NaN rather than +/-inf triggered d_overflow.
        out.visit([&](const auto *values) {
          for (size_t r = r0; r < r1 && err_i == out.size(); ++r) {
            const size_t row_base = core::packed_size(r);
            for (size_t c = 0; c <= r; ++c) {
              if (!std::isfinite(static_cast<double>(values[row_base + c]))) {
                err_i = r;
                err_j = c;
                break;
              }
            }
          }
        });
      }
      if (err_i < out.size()) {
        const double exact_dist = exact_dtw_double(series_(err_j), series_(err_i), band, use_squared_l2);
        out.clear_rows(r0, r1);
        core::refuse_float32_range("compute_distance_matrix_cuda", err_i, err_j, exact_dist);
      }
    }

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms,
                                    workspace.evt_start.get(),
                                    workspace.evt_end.get()));
    gpu_time_sec += static_cast<double>(elapsed_ms) / 1000.0;
    pairs += static_cast<size_t>(last - begin);
  }
};

template <typename T>
void launch_medoid_kernel(
    size_t N, const SeriesAt &series, const std::vector<std::span<const double>> &medoids, size_t max_L,
    const CUDADistMatOptions &opts, double &gpu_time_sec, detail::KernelPath kernel_path,
    const ParallelFor &parallel, const std::function<void(size_t, size_t, std::span<const double>)> &consume)
{
  const size_t k = medoids.size();
  const auto plan = detail::medoid_blocks(N, k, max_L, sizeof(T), opts.host_staging_bytes);
  const size_t block = plan.block, blocks = (N + block - 1) / block;
  const Launches<T, Pairs::Rectangle> launches(kernel_path, max_L, opts.use_squared_l2, opts.band,
                                               false, static_cast<std::int64_t>(block * k),
                                               device_limits(opts.device_id));

  auto &workspace = get_dtw_launch_workspace<T>(opts.device_id);
  workspace.ensure_pipeline();
  const size_t device_slot = (k + block) * max_L;
  ensure_dtw_device_capacity(workspace, plan.series_slots * device_slot, plan.series_slots * (k + block),
                             plan.series_slots * block * k * sizeof(double),
                             launches.scratch_elems(), launches.counter_elems(), 1);
  T *const h_series = workspace.host_series.ensure((k + plan.series_slots * block) * max_L, true);
  int *const h_lengths = workspace.host_lengths.ensure(k + plan.series_slots * block, true);
  double *const h_out = workspace.host_out.ensure(plan.distance_slots * block * k, true);
  T *const d_series = workspace.d_series.get();
  int *const d_lengths = workspace.d_lengths.get();
  double *const d_out = reinterpret_cast<double *>(workspace.d_out.get());
  const cudaStream_t copy = workspace.copy_stream.get(), compute = workspace.stream.get(),
                     d2h = workspace.d2h_stream.get();

  const auto enqueue = [&](size_t b) {
    const size_t s = b % 2, first = b * block, count = std::min(block, N - first);
    T *const h_slot = h_series + (k + s * block) * max_L;
    int *const h_slot_lengths = h_lengths + k + s * block;
    if (b >= 2) CUDA_CHECK(cudaEventSynchronize(workspace.uploaded[s].get()));
    parallel(count, [&](size_t i) { h_slot_lengths[i] = pad_series(h_slot + i * max_L, series(first + i), max_L); });
    if (b >= 2) CUDA_CHECK(cudaStreamWaitEvent(copy, workspace.computed[s].get(), 0));
    T *const d_slot = d_series + s * device_slot;
    int *const d_slot_lengths = d_lengths + s * (k + block);
    double *const d_out_slot = d_out + s * block * k;
    CUDA_CHECK(cudaMemcpyAsync(d_slot + k * max_L, h_slot, count * max_L * sizeof(T), cudaMemcpyHostToDevice, copy));
    CUDA_CHECK(cudaMemcpyAsync(d_slot_lengths + k, h_slot_lengths, count * sizeof(int), cudaMemcpyHostToDevice, copy));
    CUDA_CHECK(cudaEventRecord(workspace.uploaded[s].get(), copy));
    CUDA_CHECK(cudaStreamWaitEvent(compute, workspace.uploaded[s].get(), 0));
    if (b >= 2) CUDA_CHECK(cudaStreamWaitEvent(compute, workspace.d_out_free[s].get(), 0));
    launches(workspace, d_slot, d_slot_lengths, static_cast<int>(k), 0, static_cast<int>(count * k), 0, compute,
             d_out_slot, nullptr);
    CUDA_CHECK(cudaEventRecord(workspace.computed[s].get(), compute));
    CUDA_CHECK(cudaStreamWaitEvent(d2h, workspace.computed[s].get(), 0));
    CUDA_CHECK(cudaMemcpyAsync(h_out + (b % 3) * block * k, d_out_slot, count * k * sizeof(double),
                               cudaMemcpyDeviceToHost, d2h));
    CUDA_CHECK(cudaEventRecord(workspace.d_out_free[s].get(), d2h));
    CUDA_CHECK(cudaEventRecord(workspace.downloaded[b % 3].get(), d2h));
  };

  try {
    for (size_t m = 0; m < k; ++m) h_lengths[m] = pad_series(h_series + m * max_L, medoids[m], max_L);
    CUDA_CHECK(cudaEventRecord(workspace.evt_start.get(), copy));
    for (size_t s = 0; s < plan.series_slots; ++s) {
      CUDA_CHECK(cudaMemcpyAsync(d_series + s * device_slot, h_series, k * max_L * sizeof(T), cudaMemcpyHostToDevice,
                                 copy));
      CUDA_CHECK(cudaMemcpyAsync(d_lengths + s * (k + block), h_lengths, k * sizeof(int), cudaMemcpyHostToDevice,
                                 copy));
    }
    enqueue(0);
    if (blocks > 1) enqueue(1);
    for (size_t b = 0; b < blocks; ++b) {
      if (b + 2 < blocks) enqueue(b + 2);
      CUDA_CHECK(cudaEventSynchronize(workspace.downloaded[b % 3].get()));
      const size_t first = b * block, count = std::min(block, N - first);
      consume(first, count, std::span<const double>(h_out + (b % 3) * block * k, count * k));
    }
    CUDA_CHECK(cudaEventRecord(workspace.evt_end.get(), d2h));
    CUDA_CHECK(cudaEventSynchronize(workspace.evt_end.get()));
  } catch (...) {
    (void)cudaStreamSynchronize(copy);
    (void)cudaStreamSynchronize(compute);
    (void)cudaStreamSynchronize(d2h);
    (void)cudaGetLastError();
    throw;
  }
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, workspace.evt_start.get(), workspace.evt_end.get()));
  gpu_time_sec = static_cast<double>(elapsed_ms) / 1000.0;
}

template <typename T>
void launch_medoid_argmin_kernel(
    size_t N, const SeriesAt &series,
    const std::vector<std::span<const double>> &unique_medoids,
    const std::vector<int> &slot_to_unique,
    std::span<const index_t> medoid_indices,
    index_t first_point_index,
    size_t n_samples, size_t k, size_t max_L,
    const CUDADistMatOptions &opts, double &gpu_time_sec, detail::KernelPath kernel_path,
    const ParallelFor &parallel,
    const std::function<void(size_t, size_t, std::span<const index_t>, std::span<const double>)> &consume)
{
  const size_t u = unique_medoids.size();
  const size_t slots = n_samples * k;
  const auto plan = detail::medoid_blocks(N, u, max_L, sizeof(T), opts.host_staging_bytes);
  const size_t block = plan.block, blocks = (N + block - 1) / block;
  const Launches<T, Pairs::Rectangle> launches(kernel_path, max_L, opts.use_squared_l2, opts.band,
                                               false, static_cast<std::int64_t>(block * u),
                                               device_limits(opts.device_id));

  auto &workspace = get_dtw_launch_workspace<T>(opts.device_id);
  workspace.ensure_pipeline();
  const size_t device_slot = (u + block) * max_L;
  ensure_dtw_device_capacity(workspace, plan.series_slots * device_slot, plan.series_slots * (u + block),
                             plan.series_slots * block * u * sizeof(double),
                             launches.scratch_elems(), launches.counter_elems(), 1);
  if (workspace.slot_map_capacity < slots) {
    workspace.d_slot_to_unique = cuda_alloc<int>(slots);
    workspace.slot_map_capacity = slots;
  }
  if (!medoid_indices.empty() && workspace.medoid_idx_capacity < slots) {
    workspace.d_medoid_indices = cuda_alloc<index_t>(slots);
    workspace.medoid_idx_capacity = slots;
  }
  const size_t argmin_device_elems = plan.series_slots * block * n_samples;
  if (workspace.argmin_capacity < argmin_device_elems) {
    workspace.d_argmin_labels = cuda_alloc<index_t>(argmin_device_elems);
    workspace.d_argmin_dists = cuda_alloc<double>(argmin_device_elems);
    workspace.argmin_capacity = argmin_device_elems;
  }

  T *const h_series = workspace.host_series.ensure((u + plan.series_slots * block) * max_L, true);
  int *const h_lengths = workspace.host_lengths.ensure(u + plan.series_slots * block, true);
  index_t *const h_labels = workspace.host_argmin_labels.ensure(plan.distance_slots * block * n_samples, true);
  double *const h_dists = workspace.host_argmin_dists.ensure(plan.distance_slots * block * n_samples, true);
  int *const h_nonfinite = workspace.host_overflow.ensure(plan.distance_slots, true);

  T *const d_series = workspace.d_series.get();
  int *const d_lengths = workspace.d_lengths.get();
  double *const d_dist = reinterpret_cast<double *>(workspace.d_out.get());
  index_t *const d_labels = workspace.d_argmin_labels.get();
  double *const d_dists = workspace.d_argmin_dists.get();
  const cudaStream_t copy = workspace.copy_stream.get(), compute = workspace.stream.get(),
                     d2h = workspace.d2h_stream.get();

  const auto enqueue = [&](size_t b) {
    const size_t s = b % 2, first = b * block, count = std::min(block, N - first);
    T *const h_slot = h_series + (u + s * block) * max_L;
    int *const h_slot_lengths = h_lengths + u + s * block;
    if (b >= 2) CUDA_CHECK(cudaEventSynchronize(workspace.uploaded[s].get()));
    parallel(count, [&](size_t i) { h_slot_lengths[i] = pad_series(h_slot + i * max_L, series(first + i), max_L); });
    if (b >= 2) CUDA_CHECK(cudaStreamWaitEvent(copy, workspace.computed[s].get(), 0));
    T *const d_slot = d_series + s * device_slot;
    int *const d_slot_lengths = d_lengths + s * (u + block);
    double *const d_dist_slot = d_dist + s * block * u;
    index_t *const d_label_slot = d_labels + s * block * n_samples;
    double *const d_best_slot = d_dists + s * block * n_samples;

    CUDA_CHECK(cudaMemcpyAsync(d_slot + u * max_L, h_slot, count * max_L * sizeof(T), cudaMemcpyHostToDevice, copy));
    CUDA_CHECK(cudaMemcpyAsync(d_slot_lengths + u, h_slot_lengths, count * sizeof(int), cudaMemcpyHostToDevice, copy));
    CUDA_CHECK(cudaEventRecord(workspace.uploaded[s].get(), copy));
    CUDA_CHECK(cudaStreamWaitEvent(compute, workspace.uploaded[s].get(), 0));
    if (b >= 2) CUDA_CHECK(cudaStreamWaitEvent(compute, workspace.d_out_free[s].get(), 0));

    launches(workspace, d_slot, d_slot_lengths, static_cast<int>(u), 0, static_cast<int>(count * u), 0, compute,
             d_dist_slot, nullptr);
    CUDA_CHECK(cudaMemsetAsync(workspace.d_overflow.get(), 0, sizeof(int), compute));
    const int total_threads = static_cast<int>(count * n_samples);
    const int arg_grid = (total_threads + 255) / 256;
    medoid_argmin_kernel<<<arg_grid, 256, 0, compute>>>(
        d_dist_slot, workspace.d_slot_to_unique.get(),
        medoid_indices.empty() ? nullptr : workspace.d_medoid_indices.get(),
        d_label_slot, d_best_slot, workspace.d_overflow.get(),
        static_cast<int>(count), static_cast<int>(u), static_cast<int>(n_samples), static_cast<int>(k),
        first_point_index + static_cast<index_t>(first));
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaEventRecord(workspace.computed[s].get(), compute));
    CUDA_CHECK(cudaStreamWaitEvent(d2h, workspace.computed[s].get(), 0));

    const size_t d_slot_idx = b % 3;
    CUDA_CHECK(cudaMemcpyAsync(h_labels + d_slot_idx * block * n_samples, d_label_slot,
                               count * n_samples * sizeof(index_t), cudaMemcpyDeviceToHost, d2h));
    CUDA_CHECK(cudaMemcpyAsync(h_dists + d_slot_idx * block * n_samples, d_best_slot,
                               count * n_samples * sizeof(double), cudaMemcpyDeviceToHost, d2h));
    CUDA_CHECK(cudaMemcpyAsync(h_nonfinite + d_slot_idx, workspace.d_overflow.get(),
                               sizeof(int), cudaMemcpyDeviceToHost, d2h));
    CUDA_CHECK(cudaEventRecord(workspace.d_out_free[s].get(), d2h));
    CUDA_CHECK(cudaEventRecord(workspace.downloaded[d_slot_idx].get(), d2h));
  };

  try {
    for (size_t m = 0; m < u; ++m) h_lengths[m] = pad_series(h_series + m * max_L, unique_medoids[m], max_L);
    CUDA_CHECK(cudaEventRecord(workspace.evt_start.get(), copy));
    CUDA_CHECK(cudaMemcpyAsync(workspace.d_slot_to_unique.get(), slot_to_unique.data(),
                               slots * sizeof(int), cudaMemcpyHostToDevice, copy));
    if (!medoid_indices.empty()) {
      CUDA_CHECK(cudaMemcpyAsync(workspace.d_medoid_indices.get(), medoid_indices.data(),
                                 slots * sizeof(index_t), cudaMemcpyHostToDevice, copy));
    }
    for (size_t s = 0; s < plan.series_slots; ++s) {
      CUDA_CHECK(cudaMemcpyAsync(d_series + s * device_slot, h_series, u * max_L * sizeof(T), cudaMemcpyHostToDevice,
                                 copy));
      CUDA_CHECK(cudaMemcpyAsync(d_lengths + s * (u + block), h_lengths, u * sizeof(int), cudaMemcpyHostToDevice,
                                 copy));
    }
    enqueue(0);
    if (blocks > 1) enqueue(1);
    for (size_t b = 0; b < blocks; ++b) {
      if (b + 2 < blocks) enqueue(b + 2);
      const size_t d_slot_idx = b % 3;
      CUDA_CHECK(cudaEventSynchronize(workspace.downloaded[d_slot_idx].get()));
      const size_t first = b * block, count = std::min(block, N - first);
      const std::span<const index_t> lbl(h_labels + d_slot_idx * block * n_samples, count * n_samples);
      const std::span<const double> bst(h_dists + d_slot_idx * block * n_samples, count * n_samples);
      if (h_nonfinite[d_slot_idx] != 0) {
        // Locate the first non-finite point/sample to throw the exact diagnostic.
        for (size_t i = 0; i < count; ++i) {
          for (size_t s = 0; s < n_samples; ++s) {
            if (!std::isfinite(bst[s * count + i])) {
              const index_t m = lbl[s * count + i];
              const index_t med_idx = medoid_indices.empty()
                  ? static_cast<index_t>(-1)
                  : medoid_indices[s * k + static_cast<size_t>(m)];
              core::detail::throw_nonfinite_medoid_distance(
                  "compute_medoid_argmin_cuda",
                  static_cast<size_t>(first_point_index) + first + i,
                  m, med_idx);
            }
          }
        }
        throw dtwc::InvalidInput("compute_medoid_argmin_cuda: non-finite medoid distance.");
      }
      consume(first, count, lbl, bst);
    }
    CUDA_CHECK(cudaEventRecord(workspace.evt_end.get(), d2h));
    CUDA_CHECK(cudaEventSynchronize(workspace.evt_end.get()));
  } catch (...) {
    (void)cudaStreamSynchronize(copy);
    (void)cudaStreamSynchronize(compute);
    (void)cudaStreamSynchronize(d2h);
    (void)cudaGetLastError();
    throw;
  }
  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, workspace.evt_start.get(), workspace.evt_end.get()));
  gpu_time_sec = static_cast<double>(elapsed_ms) / 1000.0;
}

} // anonymous namespace

std::size_t thread_workspace_bytes()
{
  return workspace_bytes(workspace_of<float>()) + workspace_bytes(workspace_of<double>());
}

void release_thread_workspace() noexcept
{
  workspace_of<float>().release();
  workspace_of<double>().release();
}

CUDADistMatResult compute_distance_matrix_cuda(
    const std::vector<std::vector<double>> &series,
    const CUDADistMatOptions &opts, core::DistanceMatrix &out)
{
  return compute_distance_matrix_cuda(
      series.size(), [&series](size_t i) { return std::span<const double>(series[i]); }, opts,
      [](size_t count, const std::function<void(size_t)> &task) {
        for (size_t i = 0; i < count; ++i) task(i);
      },
      out);
}

struct DistanceFill::Impl {
  CUDADistMatResult result;
  bool fp32 = false;
  std::optional<TriangleFill<float>> f32;
  std::optional<TriangleFill<double>> f64;
};

DistanceFill::DistanceFill(std::size_t N, const SeriesAt &series, const CUDADistMatOptions &opts,
                           const ParallelFor &parallel, core::DistanceMatrix &out)
    : impl_(std::make_unique<Impl>())
{
  detail::require_cuda_device(cuda_available(), "compute_distance_matrix_cuda");

  CUDADistMatResult &result = impl_->result;
  result.kernel_used = "none";
  result.n = N;
  if (N <= 1) {
    if (out.size() != N) out.resize(N);
    if (N == 1) out.set(0, 0, 0.0);
    return;
  }

  CUDA_CHECK(cudaSetDevice(opts.device_id));

  size_t max_L = 0;
  for (size_t i = 0; i < N; ++i) max_L = std::max(max_L, series(i).size());

  if (max_L == 0)
    throw dtwc::InvalidInput("compute_distance_matrix_cuda: every series is "
                             "empty, so there is no distance to compute.");

  const size_t total_pairs = detail::upper_triangle_pairs(N);
  const bool out_f32 = (out.width() == core::Precision::Float32);
  bool warptile_border_fits = true;
  const auto [use_fp32, kernel_path] =
      select_route(max_L, total_pairs, opts, Pairs::Triangle, out_f32, warptile_border_fits);
  impl_->fp32 = use_fp32;
  if (use_fp32)
    impl_->f32.emplace(N, series, parallel, max_L, opts.use_squared_l2, opts.band, opts.device_id, kernel_path, out);
  else
    impl_->f64.emplace(N, series, parallel, max_L, opts.use_squared_l2, opts.band, opts.device_id, kernel_path, out);
  result.kernel_used = std::string(detail::kernel_path_name(kernel_path));
}

DistanceFill::~DistanceFill() = default;

void DistanceFill::rows(std::size_t r0, std::size_t r1, core::DistanceMatrix &out)
{
  CUDADistMatResult &result = impl_->result;
  if (impl_->f32)
    impl_->f32->rows(r0, r1, out, result.pairs_computed, result.gpu_time_sec);
  else if (impl_->f64)
    impl_->f64->rows(r0, r1, out, result.pairs_computed, result.gpu_time_sec);
}

const CUDADistMatResult &DistanceFill::result() const noexcept { return impl_->result; }

bool DistanceFill::fp32() const noexcept { return impl_->fp32; }

CUDADistMatResult compute_distance_matrix_cuda(
    std::size_t N, const SeriesAt &series,
    const CUDADistMatOptions &opts, const ParallelFor &parallel,
    core::DistanceMatrix &out)
{
  DistanceFill fill(N, series, opts, parallel, out);
  fill.rows(0, N, out);
  const CUDADistMatResult &result = fill.result();

  if (opts.verbose && N > 1) {
    std::cout << "CUDA DTW: " << result.pairs_computed << " pairs"
              << (fill.fp32() ? " [FP32]" : " [FP64]")
              << " in " << result.gpu_time_sec * 1000 << "ms"
              << " on " << cuda_device_info(opts.device_id) << std::endl;
  }

  return result;
}

CUDADistMatResult compute_medoid_distances_cuda(
    std::size_t n_series, const SeriesAt &series,
    const std::vector<std::span<const double>> &medoids,
    const CUDADistMatOptions &opts, const ParallelFor &parallel,
    const std::function<void(size_t, size_t, std::span<const double>)> &consume)
{
  detail::require_cuda_device(cuda_available(), "compute_medoid_distances_cuda");

  CUDADistMatResult result;
  result.kernel_used = "none";
  result.n = n_series;
  if (n_series == 0 || medoids.empty()) return result;

  CUDA_CHECK(cudaSetDevice(opts.device_id));

  size_t max_L = 0;
  for (size_t i = 0; i < n_series; ++i) max_L = std::max(max_L, series(i).size());
  for (const auto &s : medoids) max_L = std::max(max_L, s.size());

  if (max_L == 0)
    throw dtwc::InvalidInput("compute_medoid_distances_cuda: every series is "
                             "empty, so there is no distance to compute.");

  bool warptile_border_fits = true;
  const auto [use_fp32, kernel_path] =
      select_route(max_L, n_series * medoids.size(), opts, Pairs::Rectangle, false, warptile_border_fits);
  result.kernel_used = std::string(detail::kernel_path_name(kernel_path));
  result.pairs_computed = n_series * medoids.size();

  if (use_fp32)
    launch_medoid_kernel<float>(n_series, series, medoids, max_L, opts, result.gpu_time_sec, kernel_path, parallel,
                                consume);
  else
    launch_medoid_kernel<double>(n_series, series, medoids, max_L, opts, result.gpu_time_sec, kernel_path, parallel,
                                 consume);

  if (opts.verbose) {
    const size_t block = detail::medoid_blocks(n_series, medoids.size(), max_L,
                                               use_fp32 ? sizeof(float) : sizeof(double),
                                               opts.host_staging_bytes).block;
    std::cout << "CUDA DTW: " << result.pairs_computed << " pairs (" << n_series
              << " series x " << medoids.size() << " medoids)"
              << (use_fp32 ? " [FP32]" : " [FP64]")
              << " in " << result.gpu_time_sec * 1000 << "ms"
              << " on " << cuda_device_info(opts.device_id)
              << ", in blocks of " << block << " series" << std::endl;
  }

  return result;
}

CUDADistMatResult compute_medoid_argmin_cuda(
    std::size_t n_series, const SeriesAt &series,
    const std::vector<std::span<const double>> &medoids,
    std::span<const index_t> medoid_indices,
    index_t first_point_index,
    std::size_t n_samples, std::size_t k,
    const CUDADistMatOptions &opts, const ParallelFor &parallel,
    const std::function<void(std::size_t first, std::size_t count,
                             std::span<const index_t> labels,
                             std::span<const double> best_dists)> &consume)
{
  detail::require_cuda_device(cuda_available(), "compute_medoid_argmin_cuda");

  if (n_samples > 0 && k > 0 && medoids.size() != n_samples * k)
    throw dtwc::InvalidInput("compute_medoid_argmin_cuda: medoids.size() must equal n_samples * k.");
  if (!medoid_indices.empty() && medoid_indices.size() != medoids.size())
    throw dtwc::InvalidInput("compute_medoid_argmin_cuda: medoid_indices.size() must match medoids.size().");

  CUDADistMatResult result;
  result.kernel_used = "none";
  result.n = n_series;
  if (n_series == 0 || medoids.empty() || n_samples == 0 || k == 0) return result;

  CUDA_CHECK(cudaSetDevice(opts.device_id));

  size_t max_L = 0;
  for (size_t i = 0; i < n_series; ++i) max_L = std::max(max_L, series(i).size());
  for (const auto &s : medoids) max_L = std::max(max_L, s.size());

  if (max_L == 0)
    throw dtwc::InvalidInput("compute_medoid_argmin_cuda: every series is "
                             "empty, so there is no distance to compute.");

  // Deduplicate medoid spans across samples so repeated medoids compute once on GPU.
  const size_t slots = medoids.size();
  std::vector<std::span<const double>> unique_medoids;
  std::vector<int> slot_to_unique(slots);
  unique_medoids.reserve(slots);
  for (size_t j = 0; j < slots; ++j) {
    int found = -1;
    for (size_t prev = 0; prev < j; ++prev) {
      const bool same = !medoid_indices.empty()
          ? (medoid_indices[prev] == medoid_indices[j])
          : (medoids[prev].data() == medoids[j].data() && medoids[prev].size() == medoids[j].size());
      if (same) {
        found = slot_to_unique[prev];
        break;
      }
    }
    if (found < 0) {
      found = static_cast<int>(unique_medoids.size());
      unique_medoids.push_back(medoids[j]);
    }
    slot_to_unique[j] = found;
  }

  bool warptile_border_fits = true;
  const auto [use_fp32, kernel_path] = select_route(
      max_L, n_series * unique_medoids.size(), opts, Pairs::Rectangle, false, warptile_border_fits);
  result.kernel_used = std::string(detail::kernel_path_name(kernel_path));
  result.pairs_computed = n_series * unique_medoids.size();

  if (use_fp32)
    launch_medoid_argmin_kernel<float>(n_series, series, unique_medoids, slot_to_unique,
                                       medoid_indices, first_point_index, n_samples, k, max_L,
                                       opts, result.gpu_time_sec, kernel_path, parallel, consume);
  else
    launch_medoid_argmin_kernel<double>(n_series, series, unique_medoids, slot_to_unique,
                                        medoid_indices, first_point_index, n_samples, k, max_L,
                                        opts, result.gpu_time_sec, kernel_path, parallel, consume);

  return result;
}

}  // namespace dtwc::cuda

#endif  // DTWC_HAS_CUDA
