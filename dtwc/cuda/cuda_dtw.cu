/**
 * @file cuda_dtw.cu
 * @brief CUDA implementation of batch DTW distance computation.
 *
 * @details Two kernel strategies:
 *   1. dtw_wavefront_kernel: anti-diagonal wavefront parallelism with
 *      shared-memory buffers. Used for series longer than 32. Supports two
 *      scheduling modes:
 *        - Non-persistent (default for small workloads): one block per pair.
 *        - Persistent (auto-enabled for large-N): blocks loop over pairs via
 *          a global atomic counter, eliminating block scheduling overhead.
 *   2. dtw_warp_kernel: multiple pairs per block (8 warps = 8 pairs), each warp
 *      computes one DTW pair using register shuffles (__shfl_sync). Used for
 *      short series (max_L <= 32) where the wavefront kernel wastes block capacity.
 */

#include "cuda_dtw.cuh"
#include "cuda_memory.cuh"
#include "gpu_config.cuh"

#ifdef DTWC_HAS_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <limits>
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
// Each block computes one DTW pair (non-persistent) or loops over many pairs
// (persistent mode). Threads cooperate on the anti-diagonal wavefront:
// cells (i,j) where i+j=k are independent and computed in parallel.
// Three rotating shared-memory buffers store anti-diagonals k, k-1, and k-2.
//
// Persistent mode: when work_counter is non-null, blocks atomically grab
// pair indices from a global counter and loop until all pairs are done.
// This eliminates block scheduling overhead for large-N workloads where
// num_pairs >> resident blocks (e.g. N=1000 -> 499,500 pairs but only
// ~80-160 blocks resident). When work_counter is null, behavior is identical
// to the original one-pair-per-block design.
//
// Shared memory layout: 3 * max_L T's (rotating anti-diagonal buffers),
// optionally preceded by 2 * max_L T's for preloaded series data.

template <typename T>
__global__ void dtw_wavefront_kernel(
    const T *__restrict__ all_series, // [N * max_L] padded
    const int *__restrict__ lengths,       // [N] actual lengths
    const int *__restrict__ pair_i,        // [num_pairs] first index
    const int *__restrict__ pair_j,        // [num_pairs] second index
    T *__restrict__ distances,        // [num_pairs] output
    int max_L, int num_pairs, bool use_squared_l2, int band,
    int *__restrict__ work_counter)   // persistent mode when non-null
{
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  // Use the largest representable value for the compute type
  const T INF = (sizeof(T) == 4)
      ? static_cast<T>(3.402823466e+38f)    // FLT_MAX
      : static_cast<T>(1.7976931348623157e+308); // DBL_MAX

  // Preload threshold: series shorter than this are loaded into shared memory
  constexpr int PRELOAD_THRESHOLD = 256;
  const bool preload = (max_L <= PRELOAD_THRESHOLD);

  // Shared memory layout:
  //   Preload mode:  [0..max_L) row_buf, [max_L..2*max_L) col_buf,
  //                  [2*max_L..5*max_L) 3 anti-diagonal buffers
  //   Non-preload:   [0..3*max_L) 3 anti-diagonal buffers
  extern __shared__ char smem_raw[];
  T *smem = reinterpret_cast<T *>(smem_raw);

  // Shared variable for persistent work distribution — declared once,
  // outside the loop, to avoid issues with __syncthreads convergence.
  __shared__ int s_pid;

  // Two modes: 3-buffer (classic) for L<=1024, 2-buffer (double-buffer) for L>1024.
  // The 2-buffer mode saves max_L*sizeof(T) shared memory, improving occupancy
  // for long series at the cost of an extra sync + register pressure per anti-diag.
  // For medium series the 3-buffer mode is faster (no extra sync overhead).
  constexpr int DOUBLE_BUF_THRESHOLD = 1024;
  const bool use_double_buf = (max_L > DOUBLE_BUF_THRESHOLD) && !preload;

  T *s_row_buf = nullptr;
  T *s_col_buf = nullptr;
  T *diag_buf[3]; // [0],[1] always used; [2] only in 3-buffer mode
  int n_diag_bufs;

  if (preload) {
    s_row_buf = smem;
    s_col_buf = smem + max_L;
    diag_buf[0] = smem + 2 * max_L;
    diag_buf[1] = smem + 3 * max_L;
    diag_buf[2] = smem + 4 * max_L; // 3-buffer always for preload
    n_diag_bufs = 3;
  } else if (use_double_buf) {
    diag_buf[0] = smem;
    diag_buf[1] = smem + max_L;
    diag_buf[2] = nullptr; // not used
    n_diag_bufs = 2;
  } else {
    diag_buf[0] = smem;
    diag_buf[1] = smem + max_L;
    diag_buf[2] = smem + 2 * max_L;
    n_diag_bufs = 3;
  }

  // ---------------------------------------------------------------------------
  // Persistent kernel loop: each block grabs work atomically and computes
  // one DTW pair per iteration. When work_counter is null, falls back to
  // the original one-pair-per-block behavior (pid = blockIdx.x, single pass).
  // ---------------------------------------------------------------------------
  while (true) {
    int pid;
    if (work_counter) {
      // Persistent mode: thread 0 grabs next pair, broadcasts to block
      if (tid == 0) {
        s_pid = atomicAdd(work_counter, 1);
      }
      __syncthreads();
      pid = s_pid;
      if (pid >= num_pairs) return;
    } else {
      // Non-persistent mode (backwards compatible): one pair per block
      pid = blockIdx.x;
      if (pid >= num_pairs) return;
    }

    const int si = pair_i[pid];
    const int sj = pair_j[pid];
    const int ni = lengths[si];
    const int nj = lengths[sj];

    const T *x = all_series + static_cast<long long>(si) * max_L;
    const T *y = all_series + static_cast<long long>(sj) * max_L;

    // Orient: rows = short side, columns = long side.
    const T *row_s = (ni <= nj) ? x : y;  // indexed by i (rows, short)
    const T *col_s = (ni <= nj) ? y : x;  // indexed by j (columns, long)
    const int M = min(ni, nj);  // rows (short)
    const int N_len = max(ni, nj);  // columns (long)

    // Guard: zero-length series produce INF distance
    if (M == 0 || N_len == 0) {
      if (tid == 0) distances[pid] = INF;
      // In non-persistent mode, exit; in persistent mode, loop for next pair
      if (!work_counter) return;
      // Sync before next iteration so all threads agree before atomicAdd
      __syncthreads();
      continue;
    }

    // Precompute slope-adjusted Sakoe-Chiba band parameters
    const bool use_band = (band >= 0) && (M > 1) && (N_len > band + 1);
    const double slope  = (M > 1) ? (double)(N_len - 1) / (double)(M - 1) : 0.0;
    const double window = (band >= 0) ? fmax((double)band, slope / 2.0) : 0.0;

    // Series data pointers (re-set each iteration for persistent mode)
    const T *s_row;
    const T *s_col;

    if (preload) {
      // Cooperatively preload both series into shared memory
      for (int t = tid; t < M; t += nthreads)
        s_row_buf[t] = row_s[t];
      for (int t = tid; t < N_len; t += nthreads)
        s_col_buf[t] = col_s[t];
      __syncthreads();

      s_row = s_row_buf;
      s_col = s_col_buf;
    } else {
      s_row = row_s;  // read from global via __ldg
      s_col = col_s;
    }

    const int total_diags = M + N_len - 1;

    if (use_double_buf) {
      // ── Double-buffer mode (L > 1024): 2 ping-pong buffers ───────────
      // Pre-fetch cost_diag from k-2 buffer into registers before overwriting.
      // Saves max_L*sizeof(T) shared memory → better occupancy for long series.
      for (int k = 0; k < total_diags; ++k) {
        const int i_min = max(0, k - N_len + 1);
        const int i_max = min(k, M - 1);
        const int len_k = i_max - i_min + 1;
        const int i_min_k1 = max(0, (k - 1) - N_len + 1);
        const int i_min_k2 = max(0, (k - 2) - N_len + 1);

        T *cur  = diag_buf[k & 1];        // output for k (also holds k-2)
        T *prev = diag_buf[(k & 1) ^ 1];  // k-1

        // Phase 1: cache cost_diag from k-2 (= cur) before overwriting
        constexpr int MAX_SI = 8;
        T cd[MAX_SI];
        for (int p = tid, s = 0; p < len_k && s < MAX_SI; p += nthreads, ++s) {
          int i = i_min + p;
          cd[s] = (k >= 2 && i > 0 && (k - i) > 0)
                      ? cur[(i - 1) - i_min_k2] : INF;
        }
        __syncthreads();

        // Phase 2: compute anti-diag k
        for (int p = tid, s = 0; p < len_k && s < MAX_SI; p += nthreads, ++s) {
          int i = i_min + p, j = k - i;
          if (use_band) {
            double center = slope * i;
            int jl = (int)ceil(round(100.0 * (center - window)) / 100.0);
            int jh = (int)floor(round(100.0 * (center + window)) / 100.0);
            if (j < jl || j > jh) { cur[p] = INF; continue; }
          }
          T diff = __ldg(&s_row[i]) - __ldg(&s_col[j]);
          T d = use_squared_l2 ? (diff * diff) : fabs(diff);
          if (i == 0 && j == 0) { cur[p] = d; }
          else {
            T ca = (i > 0) ? prev[(i-1) - i_min_k1] : INF;
            T cl = (j > 0) ? prev[i - i_min_k1] : INF;
            cur[p] = fmin(cd[s], fmin(ca, cl)) + d;
          }
        }
        __syncthreads();
      }
    } else {
      // ── 3-buffer mode (L <= 1024): classic rotating buffers ──────────
      // Faster for medium series (no extra sync or register pressure).
      for (int k = 0; k < total_diags; ++k) {
        const int i_min = max(0, k - N_len + 1);
        const int i_max = min(k, M - 1);
        const int len_k = i_max - i_min + 1;
        const int i_min_k1 = max(0, (k - 1) - N_len + 1);
        const int i_min_k2 = max(0, (k - 2) - N_len + 1);

        T *cur   = diag_buf[k % 3];
        T *prev  = diag_buf[(k - 1 + 3) % 3];
        T *prev2 = diag_buf[(k - 2 + 3) % 3];

        for (int p = tid; p < len_k; p += nthreads) {
          int i = i_min + p, j = k - i;
          if (use_band) {
            double center = slope * i;
            int jl = (int)ceil(round(100.0 * (center - window)) / 100.0);
            int jh = (int)floor(round(100.0 * (center + window)) / 100.0);
            if (j < jl || j > jh) { cur[p] = INF; continue; }
          }
          T diff = preload ? (s_row[i] - s_col[j])
                           : (__ldg(&s_row[i]) - __ldg(&s_col[j]));
          T d = use_squared_l2 ? (diff * diff) : fabs(diff);
          if (i == 0 && j == 0) { cur[p] = d; }
          else {
            T ca = (i > 0) ? prev[(i-1) - i_min_k1] : INF;
            T cl = (j > 0) ? prev[i - i_min_k1] : INF;
            T cd = (i > 0 && j > 0) ? prev2[(i-1) - i_min_k2] : INF;
            cur[p] = fmin(cd, fmin(ca, cl)) + d;
          }
        }
        __syncthreads();
      }
    }

    // Result is the last anti-diagonal (single cell: (M-1, N_len-1))
    if (tid == 0) {
      int last_buf = use_double_buf ? ((total_diags - 1) & 1)
                                    : ((total_diags - 1) % 3);
      distances[pid] = diag_buf[last_buf][0];
    }

    // Non-persistent mode: exit after one pair
    if (!work_counter) return;

    // Persistent mode: sync before grabbing next pair (ensures result is
    // written and shared memory is safe to reuse)
    __syncthreads();
  }
}

// =========================================================================
// Device kernel: warp-level DTW — multiple pairs per block (L <= 32)
// =========================================================================
//
// For short series (M <= 32), one warp of 32 threads suffices per DTW pair.
// This kernel packs PAIRS_PER_BLOCK warps (= pairs) into each block,
// dramatically improving occupancy for short series.
//
// Each warp processes one pair using register-based anti-diagonal propagation
// with __shfl_sync() for cross-lane communication — no shared-memory buffers
// needed for the cost matrix, only for preloading series data.
//
// Thread assignment: lane `t` (0..31) is row `t`. For a pair with short side
// M, lanes >= M are inactive but still participate in shuffles.

constexpr int PAIRS_PER_BLOCK = 8;   // 8 warps x 32 threads = 256 threads

template <typename T>
__global__ void dtw_warp_kernel(
    const T *__restrict__ all_series,
    const int *__restrict__ lengths,
    const int *__restrict__ pair_i,
    const int *__restrict__ pair_j,
    T *__restrict__ distances,
    int max_L, int num_pairs, bool use_squared_l2, int band)
{
  const int warp_id = threadIdx.x / 32;       // which warp within block [0..7]
  const int lane    = threadIdx.x % 32;       // lane within warp [0..31]
  const int pid     = blockIdx.x * PAIRS_PER_BLOCK + warp_id;  // global pair id

  const T INF = (sizeof(T) == 4)
      ? static_cast<T>(3.402823466e+38f)
      : static_cast<T>(1.7976931348623157e+308);

  if (pid >= num_pairs) return;

  const int si = pair_i[pid];
  const int sj = pair_j[pid];
  const int ni = lengths[si];
  const int nj = lengths[sj];

  const T *x = all_series + static_cast<long long>(si) * max_L;
  const T *y = all_series + static_cast<long long>(sj) * max_L;

  // Orient: rows = short side (M <= 32), columns = long side
  const T *row_g = (ni <= nj) ? x : y;
  const T *col_g = (ni <= nj) ? y : x;
  const int M     = min(ni, nj);
  const int N_len = max(ni, nj);

  // Guard: zero-length series
  if (M == 0 || N_len == 0) {
    if (lane == 0) distances[pid] = INF;
    return;
  }

  // Shared memory layout: each warp gets 2 * 32 elements for series data
  // Total: PAIRS_PER_BLOCK * 2 * 32 * sizeof(T)
  extern __shared__ char smem_raw[];
  T *smem = reinterpret_cast<T *>(smem_raw);
  T *my_row = smem + warp_id * 64;        // 32 elements for row series
  T *my_col = smem + warp_id * 64 + 32;   // 32 elements for col series

  // Cooperatively load series data (each lane loads one element)
  if (lane < M)
    my_row[lane] = row_g[lane];
  // For column data, which may be up to 32 elements (since max_L <= 32)
  if (lane < N_len)
    my_col[lane] = col_g[lane];
  __syncwarp();

  // Banded DTW parameters (same formula as wavefront kernel)
  const bool use_band = (band >= 0) && (M > 1) && (N_len > band + 1);
  const double slope  = (M > 1) ? (double)(N_len - 1) / (double)(M - 1) : 0.0;
  const double window = (band >= 0) ? fmax((double)band, slope / 2.0) : 0.0;

  const unsigned FULL_MASK = 0xFFFFFFFF;

  // Each thread (lane) represents row i = lane.
  // We sweep anti-diagonals k = 0 .. M + N_len - 2.
  // On anti-diagonal k, thread lane computes cell (lane, k - lane)
  // if both indices are valid.
  //
  // Register state per thread:
  //   prev_val  = this thread's cost from anti-diagonal k-1
  //   prev2_val = this thread's cost from anti-diagonal k-2
  //
  // Predecessor lookup via shuffle:
  //   cost(i-1, j)   = lane-1's value from anti-diag k-1  -> shfl(prev_val, lane-1)
  //   cost(i, j-1)   = this thread's value from anti-diag k-1 -> prev_val
  //   cost(i-1, j-1) = lane-1's value from anti-diag k-2  -> shfl(prev2_val, lane-1)

  T prev_val  = INF;   // my value from anti-diagonal k-1
  T prev2_val = INF;   // my value from anti-diagonal k-2

  const int total_diags = M + N_len - 1;

  for (int k = 0; k < total_diags; ++k) {
    const int i = lane;
    const int j = k - lane;

    // All 32 lanes must participate in __shfl_sync (FULL_MASK requires it).
    // Perform shuffles unconditionally before branching on cell validity.
    T cost_above = __shfl_sync(FULL_MASK, prev_val, lane - 1);
    T cost_diag  = __shfl_sync(FULL_MASK, prev2_val, lane - 1);
    T cost_left  = prev_val;  // same row, previous column

    // Check if this thread's cell is valid
    const bool valid = (i < M) && (j >= 0) && (j < N_len);

    T my_current = INF;

    if (valid) {
      // Banded check
      bool in_band = true;
      if (use_band) {
        const double center = slope * i;
        const int j_low  = (int)ceil(round(100.0 * (center - window)) / 100.0);
        const int j_high = (int)floor(round(100.0 * (center + window)) / 100.0);
        if (j < j_low || j > j_high)
          in_band = false;
      }

      if (in_band) {
        T diff = my_row[i] - my_col[j];
        T d = use_squared_l2 ? (diff * diff) : fabs(diff);

        if (i == 0 && j == 0) {
          my_current = d;
        } else {
          // Fix up boundary cases for the shuffled predecessors
          if (lane == 0) {
            cost_above = INF;  // no row i-1 when i=0
            cost_diag  = INF;  // no (i-1, j-1) when i=0
          }
          if (j == 0) cost_left = INF;  // no column j-1 when j=0

          my_current = fmin(cost_diag, fmin(cost_above, cost_left)) + d;
        }
      }
    }

    // Rotate register state
    prev2_val = prev_val;
    prev_val  = my_current;
  }

  // The result is at cell (M-1, N_len-1), which is on the last anti-diagonal.
  // Thread lane = M-1 holds this value in prev_val.
  if (lane == M - 1) {
    distances[pid] = prev_val;
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
    bool use_squared_l2, int band, int device_id, double &gpu_time_sec)
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

  // Validate grid dimension fits in int (CUDA limit: 2^31-1 blocks in x)
  if (num_pairs > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(
        "Too many DTW pairs (" + std::to_string(num_pairs) +
        ") for a single CUDA kernel launch. Maximum: " +
        std::to_string(std::numeric_limits<int>::max()) +
        ". Reduce N or use the MPI backend for distributed computation.");
  }

  auto start = std::chrono::high_resolution_clock::now();

  if (max_L <= 32) {
    // Warp-level kernel: 8 pairs per block, 256 threads (8 warps)
    // Each warp independently computes one DTW pair using register shuffles.
    constexpr int pairs_per_block = PAIRS_PER_BLOCK;  // 8
    const int grid_size = static_cast<int>(
        (num_pairs + pairs_per_block - 1) / pairs_per_block);
    constexpr int block_size = pairs_per_block * 32;  // 256
    const size_t shared_mem = pairs_per_block * 2 * 32 * sizeof(T);

    dtw_warp_kernel<T><<<grid_size, block_size, shared_mem>>>(
        d_series.get(), d_lengths.get(), d_pair_i.get(), d_pair_j.get(),
        d_distances.get(), static_cast<int>(max_L),
        static_cast<int>(num_pairs), use_squared_l2, band);
  } else {
    // Wavefront kernel: shared memory and block size configuration
    const bool preload = (max_L <= 256);
    // L<=256: preload mode (2 series + 3 anti-diag buffers = 5)
    // 256<L<=1024: 3-buffer mode (3 anti-diag buffers)
    // L>1024: double-buffer mode (2 anti-diag buffers, saves occupancy)
    const size_t n_bufs = preload ? 5 : (max_L > 1024 ? 2 : 3);
    const size_t shared_mem = n_bufs * max_L * sizeof(T);

    // Block size heuristic tuned for the anti-diagonal wavefront pattern.
    int block_size;
    if (max_L <= 128)
      block_size = 64;
    else if (max_L <= 512)
      block_size = 128;
    else
      block_size = 256;

    // Request extended shared memory if needed (>48 KB)
    if (shared_mem > 48 * 1024) {
      CUDA_CHECK(cudaFuncSetAttribute(dtw_wavefront_kernel<T>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(shared_mem)));
    }

    // Determine whether to use persistent mode: query how many blocks
    // the GPU can run simultaneously, and use persistent mode only when
    // num_pairs significantly exceeds that (4x threshold to amortize the
    // atomicAdd overhead per pair).
    auto gpu_cfg = query_gpu_config(device_id);
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, dtw_wavefront_kernel<T>, block_size, shared_mem);
    const int persistent_grid = gpu_cfg.sm_count * std::max(blocks_per_sm, 1);
    const bool use_persistent =
        (static_cast<int>(num_pairs) > persistent_grid * 4);

    if (use_persistent) {
      // Persistent kernel: launch exactly enough blocks to fill the GPU.
      // A global atomic counter distributes pairs to blocks on-the-fly,
      // eliminating block scheduling overhead for large-N workloads.
      auto d_counter = cuda_alloc<int>(1);
      CUDA_CHECK(cudaMemset(d_counter.get(), 0, sizeof(int)));

      dtw_wavefront_kernel<T><<<persistent_grid, block_size, shared_mem>>>(
          d_series.get(), d_lengths.get(), d_pair_i.get(), d_pair_j.get(),
          d_distances.get(), static_cast<int>(max_L),
          static_cast<int>(num_pairs), use_squared_l2, band,
          d_counter.get());
    } else {
      // Non-persistent: one block per pair (original behavior)
      dtw_wavefront_kernel<T><<<static_cast<int>(num_pairs), block_size, shared_mem>>>(
          d_series.get(), d_lengths.get(), d_pair_i.get(), d_pair_j.get(),
          d_distances.get(), static_cast<int>(max_L),
          static_cast<int>(num_pairs), use_squared_l2, band,
          nullptr);
    }
  }

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

    // Sort pairs by min-length descending when series have variable lengths.
    // This reduces warp divergence / tail effects from unequal work per block.
    // Skip sorting for uniform-length data (the common case) to avoid overhead.
    const int min_len = *std::min_element(lengths.begin(), lengths.end());
    const int max_len = *std::max_element(lengths.begin(), lengths.end());
    if (min_len != max_len) {
      std::vector<size_t> order(num_pairs);
      std::iota(order.begin(), order.end(), size_t(0));
      std::sort(order.begin(), order.end(),
                [&](size_t a, size_t b) {
                  int ma = std::min(lengths[h_pair_i[a]], lengths[h_pair_j[a]]);
                  int mb = std::min(lengths[h_pair_i[b]], lengths[h_pair_j[b]]);
                  return ma > mb;
                });
      // Apply permutation in-place
      std::vector<int> sorted_i(num_pairs), sorted_j(num_pairs);
      for (size_t k = 0; k < num_pairs; ++k) {
        sorted_i[k] = h_pair_i[order[k]];
        sorted_j[k] = h_pair_j[order[k]];
      }
      h_pair_i = std::move(sorted_i);
      h_pair_j = std::move(sorted_j);
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
        opts.use_squared_l2, opts.band, opts.device_id, result.gpu_time_sec);
  } else {
    h_distances = launch_dtw_kernel<double>(
        series, lengths, h_pair_i, h_pair_j,
        N, max_L, num_pairs,
        opts.use_squared_l2, opts.band, opts.device_id, result.gpu_time_sec);
  }

  // Fill symmetric matrix using the (possibly sorted) pair indices
  for (size_t idx = 0; idx < num_pairs; ++idx) {
    size_t i = static_cast<size_t>(h_pair_i[idx]);
    size_t j = static_cast<size_t>(h_pair_j[idx]);
    result.matrix[i * N + j] = h_distances[idx];
    result.matrix[j * N + i] = h_distances[idx];
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
