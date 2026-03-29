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
// Device kernel: one block per DTW pair, single-threaded per block
// =========================================================================

__global__ void dtw_batch_kernel(
    const double *__restrict__ all_series, // [N * max_L] padded
    const int *__restrict__ lengths,       // [N] actual lengths
    const int *__restrict__ pair_i,        // [num_pairs] first index
    const int *__restrict__ pair_j,        // [num_pairs] second index
    double *__restrict__ distances,        // [num_pairs] output
    int max_L, int num_pairs, bool use_squared_l2)
{
  int pid = blockIdx.x; // one pair per block
  if (pid >= num_pairs) return;

  int si = pair_i[pid];
  int sj = pair_j[pid];
  int ni = lengths[si];
  int nj = lengths[sj];

  const double *x = all_series + static_cast<long long>(si) * max_L;
  const double *y = all_series + static_cast<long long>(sj) * max_L;

  // Use shared memory for the rolling buffer
  extern __shared__ double shared_buf[];

  // Orient: short side in rolling buffer for memory efficiency
  const double *short_s = (ni <= nj) ? x : y;
  const double *long_s = (ni <= nj) ? y : x;
  int m_short = min(ni, nj);
  int m_long = max(ni, nj);

  double *col = shared_buf; // [m_short] rolling buffer

  // Initialize first column
  {
    double d = short_s[0] - long_s[0];
    col[0] = use_squared_l2 ? (d * d) : fabs(d);
  }

  for (int i = 1; i < m_short; ++i) {
    double d = short_s[i] - long_s[0];
    col[i] = col[i - 1] + (use_squared_l2 ? (d * d) : fabs(d));
  }

  // Fill remaining columns using rolling buffer
  for (int j = 1; j < m_long; ++j) {
    double diag = col[0];
    {
      double d = short_s[0] - long_s[j];
      col[0] += use_squared_l2 ? (d * d) : fabs(d);
    }

    for (int i = 1; i < m_short; ++i) {
      double min1 = fmin(col[i - 1], col[i]);
      double best = fmin(diag, min1);
      double d = short_s[i] - long_s[j];
      double cost = use_squared_l2 ? (d * d) : fabs(d);

      diag = col[i];
      col[i] = best + cost;
    }
  }

  distances[pid] = col[m_short - 1];
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

  // Launch kernel: one block per pair, 1 thread per block
  // Shared memory: max_L doubles for the rolling buffer
  const size_t shared_mem = max_L * sizeof(double);

  auto start = std::chrono::high_resolution_clock::now();

  dtw_batch_kernel<<<static_cast<int>(num_pairs), 1, shared_mem>>>(
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
