/**
 * @file cuda_dtw.cu
 * @brief CUDA implementation of batch DTW distance computation.
 *
 * @details Simple version: one block per DTW pair, single-threaded per block.
 *          Anti-diagonal wavefront parallelism within a block is future work.
 *          Uses shared memory for the rolling buffer (one column of the cost matrix).
 */

#include "cuda_dtw.cuh"

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

__global__ void dtw_wavefront_kernel(
    const double *__restrict__ all_series, // [N * max_L] padded
    const int *__restrict__ lengths,       // [N] actual lengths
    const int *__restrict__ pair_i,        // [num_pairs] first index
    const int *__restrict__ pair_j,        // [num_pairs] second index
    double *__restrict__ distances,        // [num_pairs] output
    int max_L, int num_pairs, bool use_squared_l2)
{
  const int pid = blockIdx.x;
  if (pid >= num_pairs) return;

  const int si = pair_i[pid];
  const int sj = pair_j[pid];
  const int ni = lengths[si];
  const int nj = lengths[sj];

  const double *x = all_series + static_cast<long long>(si) * max_L;
  const double *y = all_series + static_cast<long long>(sj) * max_L;

  // Orient: rows = short side, columns = long side
  const double *row_s = (ni <= nj) ? x : y;  // indexed by i (rows)
  const double *col_s = (ni <= nj) ? y : x;  // indexed by j (columns)
  const int M = min(ni, nj);  // rows (short)
  const int N_len = max(ni, nj);  // columns (long)

  // 3 rotating anti-diagonal buffers in shared memory
  extern __shared__ double smem[];
  double *diag[3];
  diag[0] = smem;
  diag[1] = smem + max_L;
  diag[2] = smem + 2 * max_L;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int total_diags = M + N_len - 1;
  constexpr double INF = 1e300;

  for (int k = 0; k < total_diags; ++k) {
    // Anti-diagonal k: cells (i, j) where i + j = k
    const int i_min = max(0, k - N_len + 1);
    const int i_max = min(k, M - 1);
    const int len_k = i_max - i_min + 1;

    // i_min for the two previous anti-diagonals
    const int i_min_k1 = max(0, (k - 1) - N_len + 1);
    const int i_min_k2 = max(0, (k - 2) - N_len + 1);

    // Which buffer slot for k, k-1, k-2
    double *cur  = diag[k % 3];
    double *prev = diag[(k - 1 + 3) % 3]; // k-1
    double *prev2 = diag[(k - 2 + 3) % 3]; // k-2

    for (int p = tid; p < len_k; p += nthreads) {
      const int i = i_min + p;
      const int j = k - i;

      double diff = row_s[i] - col_s[j];
      double d = use_squared_l2 ? (diff * diff) : fabs(diff);

      if (i == 0 && j == 0) {
        cur[p] = d;
      } else {
        // cost[i-1][j] is on anti-diag k-1 at position (i-1) - i_min(k-1)
        double cost_above = (i > 0) ? prev[(i - 1) - i_min_k1] : INF;
        // cost[i][j-1] is on anti-diag k-1 at position i - i_min(k-1)
        double cost_left  = (j > 0) ? prev[i - i_min_k1] : INF;
        // cost[i-1][j-1] is on anti-diag k-2 at position (i-1) - i_min(k-2)
        double cost_diag  = (i > 0 && j > 0) ? prev2[(i - 1) - i_min_k2] : INF;

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

  // Pad and flatten series into contiguous array [N * max_L]
  std::vector<double> flat_series(N * max_L, 0.0);
  for (size_t i = 0; i < N; ++i) {
    std::copy(series[i].begin(), series[i].end(),
              flat_series.begin() + static_cast<ptrdiff_t>(i * max_L));
  }

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

  // Allocate device memory
  double *d_series = nullptr;
  double *d_distances = nullptr;
  int *d_lengths = nullptr;
  int *d_pair_i = nullptr;
  int *d_pair_j = nullptr;

  CUDA_CHECK(cudaMalloc(&d_series, N * max_L * sizeof(double)));
  CUDA_CHECK(cudaMalloc(&d_lengths, N * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_pair_i, num_pairs * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_pair_j, num_pairs * sizeof(int)));
  CUDA_CHECK(cudaMalloc(&d_distances, num_pairs * sizeof(double)));

  // Copy to device
  CUDA_CHECK(cudaMemcpy(d_series, flat_series.data(),
                         N * max_L * sizeof(double),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_lengths, lengths.data(), N * sizeof(int),
                         cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pair_i, h_pair_i.data(),
                         num_pairs * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_pair_j, h_pair_j.data(),
                         num_pairs * sizeof(int), cudaMemcpyHostToDevice));

  // Launch wavefront kernel: one block per pair, multiple threads per block.
  // Shared memory: 3 * max_L doubles for rotating anti-diagonal buffers.
  const size_t shared_mem = 3 * max_L * sizeof(double);

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
    cudaFuncSetAttribute(dtw_wavefront_kernel,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         static_cast<int>(shared_mem));
  }

  auto start = std::chrono::high_resolution_clock::now();

  dtw_wavefront_kernel<<<static_cast<int>(num_pairs), block_size, shared_mem>>>(
      d_series, d_lengths, d_pair_i, d_pair_j, d_distances,
      static_cast<int>(max_L), static_cast<int>(num_pairs),
      opts.use_squared_l2);

  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  auto end = std::chrono::high_resolution_clock::now();
  result.gpu_time_sec =
      std::chrono::duration<double>(end - start).count();

  // Copy results back
  std::vector<double> h_distances(num_pairs);
  CUDA_CHECK(cudaMemcpy(h_distances.data(), d_distances,
                         num_pairs * sizeof(double),
                         cudaMemcpyDeviceToHost));

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

  // Free device memory
  cudaFree(d_series);
  cudaFree(d_lengths);
  cudaFree(d_pair_i);
  cudaFree(d_pair_j);
  cudaFree(d_distances);

  if (opts.verbose) {
    std::cout << "CUDA DTW: " << num_pairs << " pairs in "
              << result.gpu_time_sec * 1000 << "ms on "
              << cuda_device_info(opts.device_id) << std::endl;
  }

  return result;
}

}  // namespace dtwc::cuda

#endif  // DTWC_HAS_CUDA
