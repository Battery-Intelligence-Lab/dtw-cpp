/**
 * @file cuda_dtw.cu
 * @brief CUDA implementation of batch DTW distance computation.
 *
 * @details Simple version: one block per DTW pair, single-threaded per block.
 *          Anti-diagonal wavefront parallelism within a block is future work.
 *          Uses shared memory for the rolling buffer (one column of the cost matrix).
 */

#include "cuda_dtw.cuh"
#include "cuda_memory.cuh"
#include "gpu_config.cuh"

#ifdef DTWC_HAS_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call)                                                     \
  do {                                                                       \
    cudaError_t err = (call);                                                \
    if (err != cudaSuccess) {                                                \
      throw std::runtime_error(std::string("CUDA error at ") + __FILE__ +   \
                               ":" + std::to_string(__LINE__) + ": " +      \
                               cudaGetErrorString(err));                     \
    }                                                                        \
  } while (0)

namespace dtwc::cuda {

// =========================================================================
// Device kernel: anti-diagonal wavefront — multiple threads per block
// =========================================================================
//
// Each block computes one DTW pair. Threads cooperate on the anti-diagonal
// wavefront: cells (i,j) where i+j=k are independent and computed in
// parallel. Three rotating shared-memory buffers store anti-diagonals k,
// k-1, and k-2.
//
// For an m_short × m_long matrix (m_short <= m_long), there are
// (m_short + m_long - 1) anti-diagonals. The widest has min(m_short, m_long)
// = m_short cells.
//
// Shared memory layout: 3 * max_L doubles (rotating anti-diagonal buffers).

template <typename T>
__global__ void dtw_wavefront_kernel(
    const T *__restrict__ all_series, // [N * max_L] padded
    const int *__restrict__ lengths,       // [N] actual lengths
    const int *__restrict__ pair_i,        // [num_pairs] first index
    const int *__restrict__ pair_j,        // [num_pairs] second index
    T *__restrict__ distances,        // [num_pairs] output
    int max_L, int num_pairs, bool use_squared_l2, int band)
{
  const int pid = blockIdx.x;
  if (pid >= num_pairs) return;

  const int si = pair_i[pid];
  const int sj = pair_j[pid];
  const int ni = lengths[si];
  const int nj = lengths[sj];

  const T *x = all_series + static_cast<long long>(si) * max_L;
  const T *y = all_series + static_cast<long long>(sj) * max_L;

  // Orient: rows = short side, columns = long side.
  // CPU banded DTW uses short_ptr for rows (i), long_ptr for columns (j).
  const T *row_s = (ni <= nj) ? x : y;  // indexed by i (rows, short)
  const T *col_s = (ni <= nj) ? y : x;  // indexed by j (columns, long)
  const int M = min(ni, nj);  // rows (short)
  const int N_len = max(ni, nj);  // columns (long)

  const int tid = threadIdx.x;
  // Use the largest representable value for the compute type
  const T INF = (sizeof(T) == 4)
      ? static_cast<T>(3.402823466e+38f)    // FLT_MAX
      : static_cast<T>(1.7976931348623157e+308); // DBL_MAX

  // Guard: zero-length series produce INF distance
  if (M == 0 || N_len == 0) {
    if (tid == 0) distances[pid] = INF;
    return;
  }

  // Precompute slope-adjusted Sakoe-Chiba band parameters (same formula as
  // CPU dtwBanded_impl in warping.hpp). These are uniform across all threads.
  // In CPU code: rows = short side (M), columns = long side (N_len).
  // slope maps row index i to expected column index j.
  const bool use_band = (band >= 0) && (M > 1) && (N_len > band + 1);
  const double slope  = (M > 1) ? (double)(N_len - 1) / (double)(M - 1) : 0.0;
  const double window = (band >= 0) ? fmax((double)band, slope / 2.0) : 0.0;

  // 3 rotating anti-diagonal buffers in shared memory
  extern __shared__ char smem_raw[];
  T *smem = reinterpret_cast<T *>(smem_raw);
  T *diag[3];
  diag[0] = smem;
  diag[1] = smem + max_L;
  diag[2] = smem + 2 * max_L;

  const int nthreads = blockDim.x;
  const int total_diags = M + N_len - 1;

  for (int k = 0; k < total_diags; ++k) {
    // Anti-diagonal k: cells (i, j) where i + j = k
    const int i_min = max(0, k - N_len + 1);
    const int i_max = min(k, M - 1);
    const int len_k = i_max - i_min + 1;

    // i_min for the two previous anti-diagonals
    const int i_min_k1 = max(0, (k - 1) - N_len + 1);
    const int i_min_k2 = max(0, (k - 2) - N_len + 1);

    // Which buffer slot for k, k-1, k-2
    T *cur  = diag[k % 3];
    T *prev = diag[(k - 1 + 3) % 3]; // k-1
    T *prev2 = diag[(k - 2 + 3) % 3]; // k-2

    for (int p = tid; p < len_k; p += nthreads) {
      const int i = i_min + p;
      const int j = k - i;

      // Banded check: if cell (i, j) is outside the Sakoe-Chiba window,
      // write INF and skip computation. Matches CPU get_bounds():
      //   j_low  = ceil(round(100*(slope*i - window)) / 100.0)
      //   j_high = floor(round(100*(slope*i + window)) / 100.0)
      if (use_band) {
        const double center = slope * i;
        const int j_low  = (int)ceil(round(100.0 * (center - window)) / 100.0);
        const int j_high = (int)floor(round(100.0 * (center + window)) / 100.0);
        if (j < j_low || j > j_high) {
          cur[p] = INF;
          continue;
        }
      }

      // Use __ldg() for read-only texture cache path on global memory reads
      T diff = __ldg(&row_s[i]) - __ldg(&col_s[j]);
      T d = use_squared_l2 ? (diff * diff) : fabs(diff);

      if (i == 0 && j == 0) {
        cur[p] = d;
      } else {
        // cost[i-1][j] is on anti-diag k-1 at position (i-1) - i_min(k-1)
        T cost_above = (i > 0) ? prev[(i - 1) - i_min_k1] : INF;
        // cost[i][j-1] is on anti-diag k-1 at position i - i_min(k-1)
        T cost_left  = (j > 0) ? prev[i - i_min_k1] : INF;
        // cost[i-1][j-1] is on anti-diag k-2 at position (i-1) - i_min(k-2)
        T cost_diag  = (i > 0 && j > 0) ? prev2[(i - 1) - i_min_k2] : INF;

        cur[p] = fmin(cost_diag, fmin(cost_above, cost_left)) + d;
      }
    }
    __syncthreads();
  }

  // Result is the last anti-diagonal (single cell: (M-1, N_len-1))
  if (tid == 0) {
    distances[pid] = diag[(total_diags - 1) % 3][0];
  }
}

// =========================================================================
// Host functions
// =========================================================================

bool cuda_available()
{
  int count = 0;
  cudaError_t err = cudaGetDeviceCount(&count);
  return (err == cudaSuccess && count > 0);
}

std::string cuda_device_info(int device_id)
{
  cudaDeviceProp prop;
  cudaError_t err = cudaGetDeviceProperties(&prop, device_id);
  if (err != cudaSuccess) return "No CUDA device";

  return std::string(prop.name) + " (compute " +
         std::to_string(prop.major) + "." + std::to_string(prop.minor) +
         ", " +
         std::to_string(prop.totalGlobalMem / (1024 * 1024)) + " MB)";
}

// =========================================================================
// Templated kernel launch helper (anonymous namespace — internal only)
// =========================================================================

namespace {

/// Launch the DTW wavefront kernel for a given compute type T (float or double).
/// Returns the per-pair distances as double (converting from T if needed).
template <typename T>
std::vector<double> launch_dtw_kernel(
    const std::vector<std::vector<double>> &series,
    const std::vector<int> &lengths,
    const std::vector<int> &h_pair_i,
    const std::vector<int> &h_pair_j,
    size_t N, size_t max_L, size_t num_pairs,
    bool use_squared_l2, int band, double &gpu_time_sec)
{
  using dtwc::cuda::cuda_alloc;

  // Flatten and convert to T
  std::vector<T> flat_series(N * max_L, T(0));
  for (size_t i = 0; i < N; ++i)
    for (size_t k = 0; k < series[i].size(); ++k)
      flat_series[i * max_L + k] = static_cast<T>(series[i][k]);

  // Allocate device memory (RAII — freed automatically)
  auto d_series    = cuda_alloc<T>(N * max_L);
  auto d_lengths   = cuda_alloc<int>(N);
  auto d_pair_i    = cuda_alloc<int>(num_pairs);
  auto d_pair_j    = cuda_alloc<int>(num_pairs);
  auto d_distances = cuda_alloc<T>(num_pairs);

  // Copy to device
  CUDA_CHECK(cudaMemcpy(d_series.get(), flat_series.data(),
                         N * max_L * sizeof(T), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_lengths.get(), lengths.data(), N * sizeof(int),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pair_i.get(), h_pair_i.data(),
                         num_pairs * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pair_j.get(), h_pair_j.data(),
                         num_pairs * sizeof(int), cudaMemcpyHostToDevice));

  // Shared memory: 3 rotating anti-diagonal buffers of type T
  const size_t shared_mem = 3 * max_L * sizeof(T);

  // Choose block size based on the widest anti-diagonal (= min series length,
  // bounded by max_L). Round up to warp boundaries.
  int block_size;
  if (max_L <= 32)
    block_size = 32;
  else if (max_L <= 128)
    block_size = 64;
  else if (max_L <= 512)
    block_size = 128;
  else
    block_size = 256;

  // Request extended shared memory if needed (>48 KB)
  if (shared_mem > 48 * 1024) {
    cudaFuncSetAttribute(dtw_wavefront_kernel<T>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shared_mem));
  }

  auto start = std::chrono::high_resolution_clock::now();

  dtw_wavefront_kernel<T><<<static_cast<int>(num_pairs), block_size, shared_mem>>>(
      d_series.get(), d_lengths.get(), d_pair_i.get(), d_pair_j.get(),
      d_distances.get(), static_cast<int>(max_L),
      static_cast<int>(num_pairs), use_squared_l2, band);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  auto end = std::chrono::high_resolution_clock::now();
  gpu_time_sec = std::chrono::duration<double>(end - start).count();

  // Copy results back
  std::vector<T> h_distances(num_pairs);
  CUDA_CHECK(cudaMemcpy(h_distances.data(), d_distances.get(),
                         num_pairs * sizeof(T), cudaMemcpyDeviceToHost));

  // Convert to double for output
  std::vector<double> result(num_pairs);
  for (size_t i = 0; i < num_pairs; ++i)
    result[i] = static_cast<double>(h_distances[i]);
  return result;
}

} // anonymous namespace

CUDADistMatResult compute_distance_matrix_cuda(
    const std::vector<std::vector<double>> &series,
    const CUDADistMatOptions &opts)
{
  CUDADistMatResult result;
  const size_t N = series.size();
  result.n = N;
  result.matrix.resize(N * N, 0.0);

  if (N <= 1 || !cuda_available()) return result;

  CUDA_CHECK(cudaSetDevice(opts.device_id));

  // Find max length for padding
  size_t max_L = 0;
  std::vector<int> lengths(N);
  for (size_t i = 0; i < N; ++i) {
    lengths[i] = static_cast<int>(series[i].size());
    max_L = std::max(max_L, series[i].size());
  }

  if (max_L == 0) return result;

  // Generate upper-triangle pair indices
  const size_t num_pairs = N * (N - 1) / 2;
  std::vector<int> h_pair_i(num_pairs), h_pair_j(num_pairs);
  {
    size_t idx = 0;
    for (size_t i = 0; i < N; ++i)
      for (size_t j = i + 1; j < N; ++j) {
        h_pair_i[idx] = static_cast<int>(i);
        h_pair_j[idx] = static_cast<int>(j);
        ++idx;
      }
  }

  result.pairs_computed = num_pairs;

  // Determine compute precision
  bool use_fp32 = false;
  switch (opts.precision) {
    case CUDAPrecision::FP32:
      use_fp32 = true;
      break;
    case CUDAPrecision::FP64:
      use_fp32 = false;
      break;
    case CUDAPrecision::Auto:
    default: {
      auto gpu_cfg = query_gpu_config(opts.device_id);
      use_fp32 = (gpu_cfg.fp64_rate == FP64Rate::Slow);
      break;
    }
  }

  // Launch kernel with the chosen precision
  std::vector<double> h_distances;
  if (use_fp32) {
    h_distances = launch_dtw_kernel<float>(
        series, lengths, h_pair_i, h_pair_j,
        N, max_L, num_pairs,
        opts.use_squared_l2, opts.band, result.gpu_time_sec);
  } else {
    h_distances = launch_dtw_kernel<double>(
        series, lengths, h_pair_i, h_pair_j,
        N, max_L, num_pairs,
        opts.use_squared_l2, opts.band, result.gpu_time_sec);
  }

  // Fill symmetric matrix
  {
    size_t idx = 0;
    for (size_t i = 0; i < N; ++i)
      for (size_t j = i + 1; j < N; ++j) {
        result.matrix[i * N + j] = h_distances[idx];
        result.matrix[j * N + i] = h_distances[idx];
        ++idx;
      }
  }

  if (opts.verbose) {
    std::cout << "CUDA DTW: " << num_pairs << " pairs"
              << (use_fp32 ? " [FP32]" : " [FP64]")
              << " in " << result.gpu_time_sec * 1000 << "ms on "
              << cuda_device_info(opts.device_id) << std::endl;
  }

  return result;
}

}  // namespace dtwc::cuda

#endif  // DTWC_HAS_CUDA
