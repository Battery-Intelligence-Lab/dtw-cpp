/**
 * @file cuda_dtw.cu
 * @brief CUDA implementation of batch DTW distance computation.
 *
 * @details Three kernel strategies:
 *   1. dtw_wavefront_kernel: anti-diagonal wavefront parallelism with
 *      shared-memory buffers. Used for series longer than 256. Supports two
 *      scheduling modes:
 *        - Non-persistent (default for small workloads): one block per pair.
 *        - Persistent (auto-enabled for large-N): blocks loop over pairs via
 *          a global atomic counter, eliminating block scheduling overhead.
 *   2. dtw_warp_kernel: multiple pairs per block (8 warps = 8 pairs), each warp
 *      computes one DTW pair using register shuffles (__shfl_sync). Used for
 *      short series (max_L <= 32) where the wavefront kernel wastes block capacity.
 *   3. dtw_regtile_kernel: register-tiled warp kernel inspired by cuDTW++
 *      (Euro-Par 2020). Each thread handles a stripe of TILE_W columns in
 *      registers; inter-thread communication via __shfl_sync. Used for medium
 *      series (32 < max_L <= 256). TILE_W=4 covers up to 128 columns,
 *      TILE_W=8 covers up to 256 columns.
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
#include <cstring>
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
// Device helper: decode flat upper-triangle pair index to (i, j)
// =========================================================================
//
// The upper triangle of an NxN matrix has N*(N-1)/2 entries enumerated as:
//   k=0: (0,1), k=1: (0,2), ..., k=N-2: (0,N-1), k=N-1: (1,2), ...
// Row i starts at linear index: i * (2*N - i - 1) / 2.
// This matches the CPU enumeration in compute_distance_matrix_cuda() and
// the MPI decode_pair() in mpi_distance_matrix.cpp.

__device__ __forceinline__ void decode_pair(int k, int N, int &i, int &j)
{
  const double Nd = static_cast<double>(N);
  const double kd = static_cast<double>(k);
  i = static_cast<int>(
      floor(Nd - 0.5 - sqrt((Nd - 0.5) * (Nd - 0.5) - 2.0 * kd)));

  // Row i starts at linear index: i * (2*N - i - 1) / 2
  int row_start = i * (2 * N - i - 1) / 2;

  // Correct for floating-point imprecision
  if (row_start + (N - i - 1) <= k) {
    row_start += (N - i - 1);
    ++i;
  }

  j = i + 1 + (k - row_start);
}

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
    T *__restrict__ result_matrix,    // [N_series * N_series] output (symmetric)
    int N_series, int max_L, int num_pairs, bool use_squared_l2, int band,
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

    int si, sj;
    decode_pair(pid, N_series, si, sj);
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
      if (tid == 0) {
        result_matrix[si * N_series + sj] = INF;
        result_matrix[sj * N_series + si] = INF;
      }
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

    // ── Fix 3: Precompute integer band boundaries into shared memory ──
    // Eliminates 6 FP64 ops per cell in the inner loop. On consumer GPUs
    // with 1:64 FP64 rate, this avoids ~192 FP32-equivalent cycles per cell.
    // Band arrays are placed after the diag buffers in shared memory.
    int *s_j_low = nullptr;
    int *s_j_high = nullptr;
    if (use_band) {
      // Compute offset past all diag buffers (in bytes, then cast)
      T *band_base;
      if (preload) {
        band_base = smem + 5 * max_L;  // after 2 series + 3 diag buffers
      } else if (use_double_buf) {
        band_base = smem + 2 * max_L;  // after 2 diag buffers
      } else {
        band_base = smem + 3 * max_L;  // after 3 diag buffers
      }
      s_j_low  = reinterpret_cast<int *>(band_base);
      s_j_high = s_j_low + max_L;

      for (int t = tid; t < M; t += nthreads) {
        double center = slope * t;
        s_j_low[t]  = max(0, (int)ceil(round(100.0 * (center - window)) / 100.0));
        s_j_high[t] = min(N_len - 1, (int)floor(round(100.0 * (center + window)) / 100.0));
      }
      __syncthreads();
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
          if (use_band && (j < s_j_low[i] || j > s_j_high[i])) {
            cur[p] = INF; continue;
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
          if (use_band && (j < s_j_low[i] || j > s_j_high[i])) {
            cur[p] = INF; continue;
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
      T dist = diag_buf[last_buf][0];
      result_matrix[si * N_series + sj] = dist;
      result_matrix[sj * N_series + si] = dist;
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
    T *__restrict__ result_matrix,
    int N_series, int max_L, int num_pairs, bool use_squared_l2, int band)
{
  const int warp_id = threadIdx.x / 32;       // which warp within block [0..7]
  const int lane    = threadIdx.x % 32;       // lane within warp [0..31]
  const int pid     = blockIdx.x * PAIRS_PER_BLOCK + warp_id;  // global pair id

  const T INF = (sizeof(T) == 4)
      ? static_cast<T>(3.402823466e+38f)
      : static_cast<T>(1.7976931348623157e+308);

  if (pid >= num_pairs) return;

  int si, sj;
  decode_pair(pid, N_series, si, sj);
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
    if (lane == 0) {
      result_matrix[si * N_series + sj] = INF;
      result_matrix[sj * N_series + si] = INF;
    }
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

  // Fix 3: Precompute band boundaries into registers (M <= 32, one row per lane)
  int my_j_low = 0, my_j_high = N_len - 1;
  if (use_band && lane < M) {
    double center = slope * lane;
    my_j_low  = max(0, (int)ceil(round(100.0 * (center - window)) / 100.0));
    my_j_high = min(N_len - 1, (int)floor(round(100.0 * (center + window)) / 100.0));
  }

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
      // Banded check (boundaries precomputed in registers)
      bool in_band = true;
      if (use_band) {
        if (j < my_j_low || j > my_j_high)
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
    result_matrix[si * N_series + sj] = prev_val;
    result_matrix[sj * N_series + si] = prev_val;
  }
}

// =========================================================================
// Device kernel: register-tiled DTW — extends warp kernel to L <= 256
// =========================================================================
//
// Inspired by cuDTW++ (Euro-Par 2020). Each warp handles one pair.
// Thread lane `t` handles a stripe of TILE_W consecutive columns:
//   columns [t*TILE_W .. (t+1)*TILE_W - 1].
//
// The DTW matrix has M rows (short side) and N_len columns (long side,
// <= 32 * TILE_W). The inter-thread wavefront sweeps anti-diagonals at the
// stripe level: at step `s`, thread `t` processes row `s - t` of its stripe.
// Within a stripe, the TILE_W columns are processed left-to-right in registers.
//
// Cross-thread communication uses __shfl_sync to pass the rightmost column
// cost from thread t-1 to thread t (left boundary of the stripe).
//
// Template: T = float/double, TILE_W = columns per thread (4 or 8).
// Block structure: PAIRS_PER_BLOCK warps (=8), one pair per warp.
// Shared memory: series data preloading only.

template <typename T, int TILE_W>
__global__ void dtw_regtile_kernel(
    const T *__restrict__ all_series,
    const int *__restrict__ lengths,
    T *__restrict__ result_matrix,
    int N_series, int max_L, int num_pairs, bool use_squared_l2, int band)
{
  constexpr int WARP_SIZE = 32;
  constexpr unsigned FULL_MASK = 0xFFFFFFFF;

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane    = threadIdx.x % WARP_SIZE;
  const int pid     = blockIdx.x * PAIRS_PER_BLOCK + warp_id;

  const T INF_VAL = (sizeof(T) == 4)
      ? static_cast<T>(3.402823466e+38f)
      : static_cast<T>(1.7976931348623157e+308);

  if (pid >= num_pairs) return;

  // Decode pair from flat upper-triangle index
  int si, sj;
  decode_pair(pid, N_series, si, sj);
  const int ni = lengths[si];
  const int nj = lengths[sj];
  const T *x = all_series + static_cast<long long>(si) * max_L;
  const T *y = all_series + static_cast<long long>(sj) * max_L;

  // Orient: rows = short side (M), columns = long side (N_len <= 32*TILE_W)
  const T *row_g = (ni <= nj) ? x : y;
  const T *col_g = (ni <= nj) ? y : x;
  const int M     = min(ni, nj);
  const int N_len = max(ni, nj);

  if (M == 0 || N_len == 0) {
    if (lane == 0) {
      result_matrix[si * N_series + sj] = INF_VAL;
      result_matrix[sj * N_series + si] = INF_VAL;
    }
    return;
  }

  // Shared memory layout: each warp gets 2 * max_L elements for series data.
  // Row data occupies [0..M), column data occupies [max_L..max_L+N_len).
  extern __shared__ char smem_raw[];
  T *smem = reinterpret_cast<T *>(smem_raw);
  T *my_row = smem + warp_id * 2 * max_L;
  T *my_col = my_row + max_L;

  // Cooperatively load row series (up to max_L elements)
  for (int t = lane; t < M; t += WARP_SIZE)
    my_row[t] = row_g[t];

  // Cooperatively load column series (up to max_L elements)
  for (int t = lane; t < N_len; t += WARP_SIZE)
    my_col[t] = col_g[t];
  __syncwarp();

  // Banded DTW parameters
  const bool use_band = (band >= 0) && (M > 1) && (N_len > band + 1);
  const double slope  = (M > 1) ? (double)(N_len - 1) / (double)(M - 1) : 0.0;
  const double window = (band >= 0) ? fmax((double)band, slope / 2.0) : 0.0;

  // Fix 3: Precompute band boundaries into shared memory (after series data).
  // Each warp's band arrays are placed after its 2*max_L series data region.
  // Band arrays use 2 * max_L ints = 2 * max_L * 4 bytes per warp.
  int *my_j_low_arr = nullptr;
  int *my_j_high_arr = nullptr;
  if (use_band) {
    // Place band arrays in the shared memory region after all warps' series data.
    // All warps' series data: PAIRS_PER_BLOCK * 2 * max_L * sizeof(T)
    // Band arrays for warp w: start at offset (all series data) + w * 2 * max_L ints
    char *band_base = smem_raw + PAIRS_PER_BLOCK * 2 * max_L * sizeof(T)
                    + warp_id * 2 * max_L * sizeof(int);
    my_j_low_arr  = reinterpret_cast<int *>(band_base);
    my_j_high_arr = my_j_low_arr + max_L;

    for (int t = lane; t < M; t += WARP_SIZE) {
      double center = slope * t;
      my_j_low_arr[t]  = max(0, (int)ceil(round(100.0 * (center - window)) / 100.0));
      my_j_high_arr[t] = min(N_len - 1, (int)floor(round(100.0 * (center + window)) / 100.0));
    }
    __syncwarp();
  }

  // This thread's column stripe: [col_start .. col_start + TILE_W - 1]
  const int col_start = lane * TILE_W;

  // Preload column data into registers
  T col_val[TILE_W];
  for (int tw = 0; tw < TILE_W; ++tw) {
    int j = col_start + tw;
    col_val[tw] = (j < N_len) ? my_col[j] : T(0);
  }

  // Per-thread register state:
  //   penalty[tw]      = cost[last_row_processed][col_start+tw]
  //   prev_penalty[tw] = cost[last_row_processed - 1][col_start+tw]
  //   prev_last        = saved prev_penalty[TILE_W-1] before overwrite
  //
  // Wavefront timing: at step s, thread t processes row i = s - t.
  // Thread t-1 processed row i at step s-1, so at step s:
  //   - penalty[TILE_W-1] of thread t-1 = cost[i][col_start-1] (left predecessor)
  //   - prev_last of thread t-1 = cost[i-1][col_start-1] (diagonal predecessor)
  //
  // The key subtlety: after step s-1, thread t-1 overwrites prev_penalty with
  // penalty (both become cost[i][...]). So we cannot shuffle prev_penalty for
  // the diagonal — we must shuffle the separately saved `prev_last`.
  T penalty[TILE_W];
  T prev_penalty[TILE_W];
  for (int tw = 0; tw < TILE_W; ++tw) {
    penalty[tw]      = INF_VAL;
    prev_penalty[tw] = INF_VAL;
  }
  T prev_last = INF_VAL;  // saved prev_penalty[TILE_W-1] before overwrite

  // Number of threads covering columns (may be fewer than WARP_SIZE)
  const int num_col_threads = (N_len + TILE_W - 1) / TILE_W;

  // Wavefront sweep: total_steps = M + num_col_threads - 1
  // At step s, thread t processes row i = s - t (if valid).
  // Thread t is active when: 0 <= s - t < M  AND  t < num_col_threads
  const int total_steps = M + num_col_threads - 1;

  for (int step = 0; step < total_steps; ++step) {
    const int i = step - lane;
    const bool row_valid = (i >= 0) && (i < M) && (lane < num_col_threads);

    // Row value for this thread's row (loaded once per step)
    T row_val = (row_valid) ? my_row[i] : T(0);

    // --- Shuffle communication (ALL lanes must participate) ---
    // penalty[TILE_W-1] from thread t-1 = cost[i][col_start-1] (left boundary)
    T penalty_from_left = __shfl_sync(FULL_MASK, penalty[TILE_W - 1], lane - 1);
    // prev_last from thread t-1 = cost[i-1][col_start-1] (diagonal boundary)
    T diag_from_left = __shfl_sync(FULL_MASK, prev_last, lane - 1);

    // Thread 0 has no left neighbor
    if (lane == 0) {
      penalty_from_left = INF_VAL;
      diag_from_left    = INF_VAL;
    }

    if (row_valid) {
      // Save prev_penalty[TILE_W-1] before overwrite (for next step's diagonal)
      T saved_prev_last = prev_penalty[TILE_W - 1];

      // Process TILE_W columns left-to-right within this thread's stripe
      T left = penalty_from_left;  // cost[i][col_start - 1]
      T diag = diag_from_left;     // cost[i-1][col_start - 1]

      for (int tw = 0; tw < TILE_W; ++tw) {
        const int j = col_start + tw;
        if (j >= N_len) {
          penalty[tw] = INF_VAL;
          diag = prev_penalty[tw];
          left = INF_VAL;
          continue;
        }

        T above = prev_penalty[tw];  // cost[i-1][j]

        // Banded check (boundaries precomputed in shared memory)
        bool in_band = true;
        if (use_band) {
          if (j < my_j_low_arr[i] || j > my_j_high_arr[i])
            in_band = false;
        }

        if (!in_band) {
          diag = prev_penalty[tw];
          left = INF_VAL;
          penalty[tw] = INF_VAL;
          continue;
        }

        T diff = row_val - col_val[tw];
        T d = use_squared_l2 ? (diff * diff) : fabs(diff);

        T new_cost;
        if (i == 0 && j == 0) {
          new_cost = d;
        } else {
          // Boundary conditions:
          // i == 0: no above, no diagonal predecessor
          // j == 0: no left, no diagonal predecessor
          T eff_above = (i == 0) ? INF_VAL : above;
          T eff_diag  = (i == 0 || j == 0) ? INF_VAL : diag;
          T eff_left  = (j == 0) ? INF_VAL : left;
          new_cost = fmin(eff_diag, fmin(eff_above, eff_left)) + d;
        }

        // Advance for next column: current above becomes diagonal,
        // current cost becomes left
        diag = prev_penalty[tw];
        left = new_cost;
        penalty[tw] = new_cost;
      }

      // Update prev_last for next step's diagonal shuffle
      prev_last = saved_prev_last;

      // Update prev_penalty for next row: prev_penalty <- penalty (= cost[i][...])
      for (int tw = 0; tw < TILE_W; ++tw)
        prev_penalty[tw] = penalty[tw];
    }
    // If !row_valid, penalty[], prev_penalty[], and prev_last are unchanged,
    // which is correct: after a thread finishes its last valid row, the values
    // persist for result extraction and do not corrupt other threads' shuffles.
  }

  // Result extraction: cost[M-1][N_len-1].
  // Thread holding column N_len-1 is: result_thread = (N_len-1) / TILE_W
  // Index within stripe: result_tw = (N_len-1) % TILE_W
  // That thread processed row M-1 at step = (M-1) + result_thread, after
  // which row_valid became false for subsequent steps, so penalty[] is
  // preserved.
  const int result_thread = (N_len - 1) / TILE_W;
  const int result_tw     = (N_len - 1) % TILE_W;

  // Each thread puts its candidate result value; only the result_thread
  // has the real answer.
  T my_result = (lane == result_thread) ? penalty[result_tw] : INF_VAL;
  T final_result = __shfl_sync(FULL_MASK, my_result, result_thread);

  if (lane == 0) {
    result_matrix[si * N_series + sj] = final_result;
    result_matrix[sj * N_series + si] = final_result;
  }
}

// =========================================================================
// Device kernel: compute upper/lower envelopes for all N series
// =========================================================================
//
// Each block handles one series. Each thread computes one element's envelope
// using a brute-force O(band) scan — simple and efficient for small bands.
// Output: upper_envelopes[i * max_L + k] and lower_envelopes[i * max_L + k].

template <typename T>
__global__ void compute_envelopes_kernel(
    const T *__restrict__ all_series,    // [N * max_L] padded
    const int *__restrict__ lengths,     // [N]
    T *__restrict__ upper_envelopes,     // [N * max_L] output
    T *__restrict__ lower_envelopes,     // [N * max_L] output
    int max_L, int N, int band)
{
  const int series_idx = blockIdx.x;
  if (series_idx >= N) return;

  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;
  const int L = lengths[series_idx];
  const T *series = all_series + static_cast<long long>(series_idx) * max_L;
  T *upper = upper_envelopes + static_cast<long long>(series_idx) * max_L;
  T *lower = lower_envelopes + static_cast<long long>(series_idx) * max_L;

  const int w = (band >= 0) ? band : 0;

  for (int k = tid; k < L; k += nthreads) {
    const int lo = (k >= w) ? k - w : 0;
    const int hi = (k + w + 1 < L) ? k + w + 1 : L;

    T max_val = series[lo];
    T min_val = series[lo];
    for (int j = lo + 1; j < hi; ++j) {
      T val = series[j];
      if (val > max_val) max_val = val;
      if (val < min_val) min_val = val;
    }
    upper[k] = max_val;
    lower[k] = min_val;
  }

  // Zero-fill padding beyond series length
  for (int k = L + tid; k < max_L; k += nthreads) {
    upper[k] = T(0);
    lower[k] = T(0);
  }
}

// =========================================================================
// Device kernel: compute LB_Keogh for all N*(N-1)/2 pairs
// =========================================================================
//
// Each thread handles one pair. Computes symmetric LB_Keogh:
//   max(LB_Keogh(query=j, env=i), LB_Keogh(query=i, env=j))
// This is embarrassingly parallel — no dependencies between pairs.

template <typename T>
__global__ void compute_lb_keogh_kernel(
    const T *__restrict__ all_series,       // [N * max_L]
    const int *__restrict__ lengths,        // [N]
    const T *__restrict__ upper_envelopes,  // [N * max_L]
    const T *__restrict__ lower_envelopes,  // [N * max_L]
    T *__restrict__ lb_values,              // [num_pairs] output
    int max_L, int N, int num_pairs)
{
  const int pid = blockIdx.x * blockDim.x + threadIdx.x;
  if (pid >= num_pairs) return;

  // Decode flat pair index to (i, j) using the same upper-triangle encoding
  int si, sj;
  decode_pair(pid, N, si, sj);

  const int Li = lengths[si];
  const int Lj = lengths[sj];
  const int n = min(Li, Lj);  // compare up to shorter length

  const T *series_i = all_series + static_cast<long long>(si) * max_L;
  const T *series_j = all_series + static_cast<long long>(sj) * max_L;
  const T *upper_i  = upper_envelopes + static_cast<long long>(si) * max_L;
  const T *lower_i  = lower_envelopes + static_cast<long long>(si) * max_L;
  const T *upper_j  = upper_envelopes + static_cast<long long>(sj) * max_L;
  const T *lower_j  = lower_envelopes + static_cast<long long>(sj) * max_L;

  // LB_Keogh(query=j, envelope=i)
  T lb1 = T(0);
  for (int k = 0; k < n; ++k) {
    T val = series_j[k];
    T excess_upper = val - upper_i[k];
    T excess_lower = lower_i[k] - val;
    T contrib = fmax(T(0), fmax(excess_upper, excess_lower));
    lb1 += contrib;
  }

  // LB_Keogh(query=i, envelope=j)
  T lb2 = T(0);
  for (int k = 0; k < n; ++k) {
    T val = series_i[k];
    T excess_upper = val - upper_j[k];
    T excess_lower = lower_j[k] - val;
    T contrib = fmax(T(0), fmax(excess_upper, excess_lower));
    lb2 += contrib;
  }

  lb_values[pid] = fmax(lb1, lb2);
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

/// Launch the DTW kernel for a given compute type T (float or double).
/// Returns the NxN distance matrix as a flat vector<double> (row-major).
///
/// Fix 1: Pair indices are computed on-device via decode_pair(), eliminating
///        the host-side pair index arrays and their H2D transfers (4 MB for N=1000).
/// Fix 2: Kernels write directly to the NxN result matrix on GPU, eliminating
///        the per-pair distance array, its D2H transfer, and the host-side fill loop.
///
/// Uses pinned host memory and a CUDA stream for overlapping H2D transfers,
/// kernel execution, and D2H transfers. GPU timing is measured with CUDA
/// events for accurate results that include the full async pipeline.
template <typename T>
std::vector<double> launch_dtw_kernel(
    const std::vector<std::vector<double>> &series,
    const std::vector<int> &lengths,
    size_t N, size_t max_L, size_t num_pairs,
    bool use_squared_l2, int band, int device_id, double &gpu_time_sec)
{
  using dtwc::cuda::cuda_alloc;
  using dtwc::cuda::pinned_alloc_nothrow;
  using dtwc::cuda::make_cuda_stream;
  using dtwc::cuda::make_cuda_event;

  // Validate grid dimension fits in int (CUDA limit: 2^31-1 blocks in x)
  if (num_pairs > static_cast<size_t>(std::numeric_limits<int>::max())) {
    throw std::runtime_error(
        "Too many DTW pairs (" + std::to_string(num_pairs) +
        ") for a single CUDA kernel launch. Maximum: " +
        std::to_string(std::numeric_limits<int>::max()) +
        ". Reduce N or use the MPI backend for distributed computation.");
  }

  const size_t series_bytes = N * max_L * sizeof(T);
  const size_t matrix_elems = N * N;
  const size_t matrix_bytes = matrix_elems * sizeof(T);

  // ---------------------------------------------------------------------------
  // Allocate pinned host memory for the two large buffers (flat_series,
  // h_result_matrix). Pinned memory enables cudaMemcpyAsync to truly overlap
  // with kernel execution on the GPU. Fall back to regular std::vector
  // allocation if pinned fails (e.g. limited pinned memory budget).
  // ---------------------------------------------------------------------------
  constexpr size_t PINNED_THRESHOLD = 256 * 1024;
  const size_t series_transfer = N * max_L * sizeof(T);
  auto h_pinned_series = (series_transfer >= PINNED_THRESHOLD)
      ? pinned_alloc_nothrow<T>(N * max_L) : PinnedPtr<T>(nullptr);
  auto h_pinned_matrix = (matrix_bytes >= PINNED_THRESHOLD)
      ? pinned_alloc_nothrow<T>(matrix_elems) : PinnedPtr<T>(nullptr);

  // Fallback std::vectors used only when pinned allocation fails
  std::vector<T> flat_series_fallback;
  std::vector<T> h_matrix_fallback;

  T *h_flat_series   = nullptr;
  T *h_result_matrix = nullptr;

  if (h_pinned_series) {
    h_flat_series = h_pinned_series.get();
    std::memset(h_flat_series, 0, series_bytes);
  } else {
    flat_series_fallback.resize(N * max_L, T(0));
    h_flat_series = flat_series_fallback.data();
  }

  if (h_pinned_matrix) {
    h_result_matrix = h_pinned_matrix.get();
  } else {
    h_matrix_fallback.resize(matrix_elems);
    h_result_matrix = h_matrix_fallback.data();
  }

  // Flatten and convert to T
  for (size_t i = 0; i < N; ++i)
    for (size_t k = 0; k < series[i].size(); ++k)
      h_flat_series[i * max_L + k] = static_cast<T>(series[i][k]);

  // ---------------------------------------------------------------------------
  // Create CUDA stream and timing events
  // ---------------------------------------------------------------------------
  auto stream    = make_cuda_stream();
  auto evt_start = make_cuda_event();
  auto evt_end   = make_cuda_event();

  // Allocate device memory (RAII -- freed automatically)
  // Fix 1: No d_pair_i / d_pair_j — indices computed on-device via decode_pair()
  // Fix 2: Allocate NxN result matrix instead of per-pair distances
  auto d_series        = cuda_alloc<T>(N * max_L);
  auto d_lengths       = cuda_alloc<int>(N);
  auto d_result_matrix = cuda_alloc<T>(matrix_elems);

  const int N_series = static_cast<int>(N);

  // ---------------------------------------------------------------------------
  // Begin timed region: H2D transfers + kernel + D2H
  // ---------------------------------------------------------------------------
  CUDA_CHECK(cudaEventRecord(evt_start.get(), stream.get()));

  // Async H2D transfers (no pair index arrays needed — Fix 1)
  CUDA_CHECK(cudaMemcpyAsync(d_series.get(), h_flat_series,
                              series_bytes, cudaMemcpyHostToDevice, stream.get()));
  CUDA_CHECK(cudaMemcpyAsync(d_lengths.get(), lengths.data(),
                              N * sizeof(int), cudaMemcpyHostToDevice, stream.get()));

  // Zero-initialize NxN result matrix on device (diagonal = 0, Fix 2)
  CUDA_CHECK(cudaMemsetAsync(d_result_matrix.get(), 0, matrix_bytes, stream.get()));

  // ---------------------------------------------------------------------------
  // Kernel launch (on the same stream -- automatically waits for H2D)
  // ---------------------------------------------------------------------------
  if (max_L <= 32) {
    // Warp-level kernel: 8 pairs per block, 256 threads (8 warps)
    constexpr int pairs_per_block = PAIRS_PER_BLOCK;  // 8
    const int grid_size = static_cast<int>(
        (num_pairs + pairs_per_block - 1) / pairs_per_block);
    constexpr int block_size = pairs_per_block * 32;  // 256
    const size_t shared_mem = pairs_per_block * 2 * 32 * sizeof(T);

    dtw_warp_kernel<T><<<grid_size, block_size, shared_mem, stream.get()>>>(
        d_series.get(), d_lengths.get(), d_result_matrix.get(),
        N_series, static_cast<int>(max_L),
        static_cast<int>(num_pairs), use_squared_l2, band);
  } else if (max_L <= 128) {
    // Register-tiled kernel with TILE_W=4: 32 threads * 4 = 128 columns max
    constexpr int pairs_per_block = PAIRS_PER_BLOCK;
    constexpr int TILE_W = 4;
    const int grid_size = static_cast<int>(
        (num_pairs + pairs_per_block - 1) / pairs_per_block);
    constexpr int block_size = pairs_per_block * 32;
    // Series data + band arrays when banded (Fix 3)
    const size_t series_smem = pairs_per_block * 2 * max_L * sizeof(T);
    const size_t band_smem = (band >= 0) ? pairs_per_block * 2 * max_L * sizeof(int) : 0;
    const size_t shared_mem = series_smem + band_smem;

    dtw_regtile_kernel<T, TILE_W><<<grid_size, block_size, shared_mem, stream.get()>>>(
        d_series.get(), d_lengths.get(), d_result_matrix.get(),
        N_series, static_cast<int>(max_L),
        static_cast<int>(num_pairs), use_squared_l2, band);
  } else if (max_L <= 256) {
    // Register-tiled kernel with TILE_W=8: 32 threads * 8 = 256 columns max
    constexpr int pairs_per_block = PAIRS_PER_BLOCK;
    constexpr int TILE_W = 8;
    const int grid_size = static_cast<int>(
        (num_pairs + pairs_per_block - 1) / pairs_per_block);
    constexpr int block_size = pairs_per_block * 32;
    const size_t series_smem = pairs_per_block * 2 * max_L * sizeof(T);
    const size_t band_smem = (band >= 0) ? pairs_per_block * 2 * max_L * sizeof(int) : 0;
    const size_t shared_mem = series_smem + band_smem;

    dtw_regtile_kernel<T, TILE_W><<<grid_size, block_size, shared_mem, stream.get()>>>(
        d_series.get(), d_lengths.get(), d_result_matrix.get(),
        N_series, static_cast<int>(max_L),
        static_cast<int>(num_pairs), use_squared_l2, band);
  } else {
    // Wavefront kernel: shared memory and block size configuration
    const bool preload = (max_L <= 256);
    // L<=256: preload mode (2 series + 3 anti-diag buffers = 5)
    // 256<L<=1024: 3-buffer mode (3 anti-diag buffers)
    // L>1024: double-buffer mode (2 anti-diag buffers, saves occupancy)
    const size_t n_bufs = preload ? 5 : (max_L > 1024 ? 2 : 3);
    // Base shared memory for diag/series buffers
    size_t shared_mem = n_bufs * max_L * sizeof(T);
    // Fix 3: Add space for precomputed band boundary arrays (2 * max_L ints)
    if (band >= 0) {
      shared_mem += 2 * max_L * sizeof(int);
    }

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

    // Determine whether to use persistent mode
    auto gpu_cfg = query_gpu_config(device_id);
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, dtw_wavefront_kernel<T>, block_size, shared_mem);
    const int persistent_grid = gpu_cfg.sm_count * std::max(blocks_per_sm, 1);
    const bool use_persistent =
        (static_cast<int>(num_pairs) > persistent_grid * 4);

    if (use_persistent) {
      auto d_counter = cuda_alloc<int>(1);
      CUDA_CHECK(cudaMemsetAsync(d_counter.get(), 0, sizeof(int), stream.get()));

      dtw_wavefront_kernel<T><<<persistent_grid, block_size, shared_mem, stream.get()>>>(
          d_series.get(), d_lengths.get(), d_result_matrix.get(),
          N_series, static_cast<int>(max_L),
          static_cast<int>(num_pairs), use_squared_l2, band,
          d_counter.get());
    } else {
      // Non-persistent: one block per pair (original behavior)
      dtw_wavefront_kernel<T><<<static_cast<int>(num_pairs), block_size, shared_mem, stream.get()>>>(
          d_series.get(), d_lengths.get(), d_result_matrix.get(),
          N_series, static_cast<int>(max_L),
          static_cast<int>(num_pairs), use_squared_l2, band,
          nullptr);
    }
  }

  CUDA_CHECK(cudaGetLastError());

  // ---------------------------------------------------------------------------
  // Async D2H transfer: single contiguous NxN matrix (Fix 2 — no host fill loop)
  // ---------------------------------------------------------------------------
  CUDA_CHECK(cudaMemcpyAsync(h_result_matrix, d_result_matrix.get(),
                              matrix_bytes, cudaMemcpyDeviceToHost, stream.get()));

  // ---------------------------------------------------------------------------
  // End timed region and synchronize
  // ---------------------------------------------------------------------------
  CUDA_CHECK(cudaEventRecord(evt_end.get(), stream.get()));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, evt_start.get(), evt_end.get()));
  gpu_time_sec = static_cast<double>(elapsed_ms) / 1000.0;

  // Convert NxN matrix from T to double
  std::vector<double> result(matrix_elems);
  for (size_t i = 0; i < matrix_elems; ++i)
    result[i] = static_cast<double>(h_result_matrix[i]);
  return result;
}

// =========================================================================
// Templated LB_Keogh launch helper
// =========================================================================

/// Launch envelope + LB_Keogh kernels for a given compute type T.
/// Returns flat array of N*(N-1)/2 lower bounds as double.
/// d_series and d_lengths must already be uploaded; stream must be provided.
template <typename T>
std::vector<double> launch_lb_keogh_kernel(
    const T *d_series, const int *d_lengths,
    size_t N, size_t max_L, size_t num_pairs,
    int band, cudaStream_t stream, double &lb_time_sec)
{
  using dtwc::cuda::cuda_alloc;

  auto evt_start = dtwc::cuda::make_cuda_event();
  auto evt_end   = dtwc::cuda::make_cuda_event();

  const size_t env_elems = N * max_L;

  // Allocate envelope arrays on device
  auto d_upper = cuda_alloc<T>(env_elems);
  auto d_lower = cuda_alloc<T>(env_elems);

  CUDA_CHECK(cudaEventRecord(evt_start.get(), stream));

  // Launch envelope computation: one block per series
  {
    const int block_size = 256;
    const int grid_size = static_cast<int>(N);
    compute_envelopes_kernel<T><<<grid_size, block_size, 0, stream>>>(
        d_series, d_lengths,
        d_upper.get(), d_lower.get(),
        static_cast<int>(max_L), static_cast<int>(N), band);
  }

  // Allocate LB output array
  auto d_lb = cuda_alloc<T>(num_pairs);

  // Launch LB_Keogh computation: one thread per pair
  {
    const int block_size = 256;
    const int grid_size = static_cast<int>(
        (num_pairs + block_size - 1) / block_size);
    compute_lb_keogh_kernel<T><<<grid_size, block_size, 0, stream>>>(
        d_series, d_lengths,
        d_upper.get(), d_lower.get(),
        d_lb.get(),
        static_cast<int>(max_L), static_cast<int>(N),
        static_cast<int>(num_pairs));
  }

  CUDA_CHECK(cudaGetLastError());

  // Download LB values
  std::vector<T> h_lb(num_pairs);
  CUDA_CHECK(cudaMemcpyAsync(h_lb.data(), d_lb.get(),
                              num_pairs * sizeof(T), cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaEventRecord(evt_end.get(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, evt_start.get(), evt_end.get()));
  lb_time_sec = static_cast<double>(elapsed_ms) / 1000.0;

  // Convert to double
  std::vector<double> result(num_pairs);
  for (size_t i = 0; i < num_pairs; ++i)
    result[i] = static_cast<double>(h_lb[i]);
  return result;
}

/// Launch envelope + LB_Keogh kernels, handling series upload internally.
/// Standalone version that allocates and uploads series data.
template <typename T>
std::vector<double> launch_lb_keogh_standalone(
    const std::vector<std::vector<double>> &series,
    const std::vector<int> &lengths,
    size_t N, size_t max_L, size_t num_pairs,
    int band, double &lb_time_sec)
{
  using dtwc::cuda::cuda_alloc;
  using dtwc::cuda::make_cuda_stream;

  const size_t series_bytes = N * max_L * sizeof(T);

  // Flatten and convert series to T
  std::vector<T> flat(N * max_L, T(0));
  for (size_t i = 0; i < N; ++i)
    for (size_t k = 0; k < series[i].size(); ++k)
      flat[i * max_L + k] = static_cast<T>(series[i][k]);

  auto stream = make_cuda_stream();

  auto d_series  = cuda_alloc<T>(N * max_L);
  auto d_lengths = cuda_alloc<int>(N);

  CUDA_CHECK(cudaMemcpyAsync(d_series.get(), flat.data(),
                              series_bytes, cudaMemcpyHostToDevice, stream.get()));
  CUDA_CHECK(cudaMemcpyAsync(d_lengths.get(), lengths.data(),
                              N * sizeof(int), cudaMemcpyHostToDevice, stream.get()));

  return launch_lb_keogh_kernel<T>(
      d_series.get(), d_lengths.get(),
      N, max_L, num_pairs, band, stream.get(), lb_time_sec);
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

  // Fix 1: No pair index arrays — pairs are decoded on-device via decode_pair().
  // This eliminates 2 * num_pairs * sizeof(int) host allocation + H2D transfer.
  const size_t num_pairs = N * (N - 1) / 2;
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

  // ---------------------------------------------------------------------------
  // Phase 1 (optional): LB_Keogh pruning
  // ---------------------------------------------------------------------------
  // When use_lb_pruning is enabled and band >= 0, compute LB_Keogh for all
  // pairs on GPU. If skip_threshold > 0, pairs with LB > threshold are set
  // to INF and excluded from full DTW computation.
  std::vector<double> lb_values;
  const bool do_lb_pruning = opts.use_lb_pruning && (opts.band >= 0);

  if (do_lb_pruning) {
    if (use_fp32) {
      lb_values = launch_lb_keogh_standalone<float>(
          series, lengths, N, max_L, num_pairs, opts.band, result.lb_time_sec);
    } else {
      lb_values = launch_lb_keogh_standalone<double>(
          series, lengths, N, max_L, num_pairs, opts.band, result.lb_time_sec);
    }
  }

  // If we have a skip threshold, count prunable pairs and apply them
  // after DTW computation (or skip DTW for those pairs entirely via
  // post-processing the result matrix).
  const bool has_threshold = do_lb_pruning && (opts.skip_threshold > 0);

  // Launch DTW kernel — returns NxN matrix directly (Fix 2: no host-side fill loop)
  if (use_fp32) {
    result.matrix = launch_dtw_kernel<float>(
        series, lengths,
        N, max_L, num_pairs,
        opts.use_squared_l2, opts.band, opts.device_id, result.gpu_time_sec);
  } else {
    result.matrix = launch_dtw_kernel<double>(
        series, lengths,
        N, max_L, num_pairs,
        opts.use_squared_l2, opts.band, opts.device_id, result.gpu_time_sec);
  }

  // ---------------------------------------------------------------------------
  // Phase 2 (optional): Apply threshold pruning to the result matrix
  // ---------------------------------------------------------------------------
  // For pairs where LB > skip_threshold, overwrite the exact DTW distance
  // with infinity. This is a host-side post-process; the GPU computed all
  // pairs (future optimization: skip DTW kernel launch for pruned pairs by
  // building a compact pair list on GPU).
  if (has_threshold) {
    constexpr double INF = std::numeric_limits<double>::max();
    size_t pruned = 0;

    // Decode pair indices on host to apply threshold
    for (size_t k = 0; k < num_pairs; ++k) {
      if (lb_values[k] > opts.skip_threshold) {
        // Decode pair (i, j) from flat index
        const double Nd = static_cast<double>(N);
        const double kd = static_cast<double>(k);
        int i = static_cast<int>(
            std::floor(Nd - 0.5 - std::sqrt((Nd - 0.5) * (Nd - 0.5) - 2.0 * kd)));
        int row_start = i * (2 * static_cast<int>(N) - i - 1) / 2;
        if (row_start + (static_cast<int>(N) - i - 1) <= static_cast<int>(k)) {
          row_start += (static_cast<int>(N) - i - 1);
          ++i;
        }
        int j = i + 1 + (static_cast<int>(k) - row_start);

        result.matrix[i * N + j] = INF;
        result.matrix[j * N + i] = INF;
        ++pruned;
      }
    }
    result.pairs_pruned = pruned;
    result.pairs_computed = num_pairs - pruned;
  }

  if (opts.verbose) {
    std::cout << "CUDA DTW: " << num_pairs << " pairs"
              << (use_fp32 ? " [FP32]" : " [FP64]")
              << " in " << result.gpu_time_sec * 1000 << "ms";
    if (do_lb_pruning) {
      std::cout << " (LB_Keogh: " << result.lb_time_sec * 1000 << "ms";
      if (has_threshold) {
        std::cout << ", pruned " << result.pairs_pruned << "/" << num_pairs;
      }
      std::cout << ")";
    }
    std::cout << " on " << cuda_device_info(opts.device_id) << std::endl;
  }

  return result;
}

CUDALBResult compute_lb_keogh_cuda(
    const std::vector<std::vector<double>> &series,
    int band, int device_id)
{
  CUDALBResult result;
  const size_t N = series.size();
  result.n = N;

  if (N <= 1 || band < 0 || !cuda_available()) return result;

  CUDA_CHECK(cudaSetDevice(device_id));

  size_t max_L = 0;
  std::vector<int> lengths(N);
  for (size_t i = 0; i < N; ++i) {
    lengths[i] = static_cast<int>(series[i].size());
    max_L = std::max(max_L, series[i].size());
  }
  if (max_L == 0) return result;

  const size_t num_pairs = N * (N - 1) / 2;

  // Use FP64 for standalone LB computation (accuracy matters for pruning decisions)
  result.lb_values = launch_lb_keogh_standalone<double>(
      series, lengths, N, max_L, num_pairs, band, result.gpu_time_sec);

  return result;
}

// =========================================================================
// 1-vs-N / K-vs-N DTW kernels and host functions
// =========================================================================
//
// Dedicated kernels for computing DTW from K query series against all N
// target series. Key optimization: the query is loaded into shared memory
// once per block and reused, while each block processes a different target.
//
// Grid: dim3(N_targets, K_queries) or dim3(ceil(N/PPB), K) for warp/regtile
// Output: distances[query_idx * N + target_idx]

// ── Wavefront kernel for 1-vs-N (L > 256) ──────────────────────────────

template <typename T>
__global__ void dtw_one_vs_all_wavefront_kernel(
    const T *__restrict__ queries,         // [K * max_L] padded query series
    const int *__restrict__ query_lengths,  // [K] query lengths
    const T *__restrict__ all_series,      // [N * max_L] padded target series
    const int *__restrict__ lengths,        // [N] target lengths
    T *__restrict__ distances,             // [K * N] output distances
    int max_L, int N, int K, bool use_squared_l2, int band)
{
  const int target_idx = blockIdx.x;
  const int query_idx  = blockIdx.y;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  const T INF = (sizeof(T) == 4)
      ? static_cast<T>(3.402823466e+38f)
      : static_cast<T>(1.7976931348623157e+308);

  if (target_idx >= N || query_idx >= K) return;

  const int query_len  = query_lengths[query_idx];
  const int target_len = lengths[target_idx];

  if (query_len == 0 || target_len == 0) {
    if (tid == 0)
      distances[query_idx * N + target_idx] = INF;
    return;
  }

  const T *raw_query  = queries + static_cast<long long>(query_idx) * max_L;
  const T *raw_target = all_series + static_cast<long long>(target_idx) * max_L;

  // Orient: rows = short side, cols = long side
  const T *row_src = (query_len <= target_len) ? raw_query  : raw_target;
  const T *col_src = (query_len <= target_len) ? raw_target : raw_query;
  const int M     = min(query_len, target_len);
  const int N_len = max(query_len, target_len);

  // Shared memory layout (same as pairwise wavefront kernel)
  constexpr int PRELOAD_THRESHOLD = 256;
  const bool preload = (max_L <= PRELOAD_THRESHOLD);
  constexpr int DOUBLE_BUF_THRESHOLD = 1024;
  const bool use_double_buf = (max_L > DOUBLE_BUF_THRESHOLD) && !preload;

  extern __shared__ char smem_raw[];
  T *smem = reinterpret_cast<T *>(smem_raw);

  T *s_row_buf = nullptr;
  T *s_col_buf = nullptr;
  T *diag_buf[3];

  if (preload) {
    s_row_buf = smem;
    s_col_buf = smem + max_L;
    diag_buf[0] = smem + 2 * max_L;
    diag_buf[1] = smem + 3 * max_L;
    diag_buf[2] = smem + 4 * max_L;
  } else if (use_double_buf) {
    diag_buf[0] = smem;
    diag_buf[1] = smem + max_L;
    diag_buf[2] = nullptr;
  } else {
    diag_buf[0] = smem;
    diag_buf[1] = smem + max_L;
    diag_buf[2] = smem + 2 * max_L;
  }

  const T *s_row;
  const T *s_col;

  if (preload) {
    for (int t = tid; t < M; t += nthreads)
      s_row_buf[t] = row_src[t];
    for (int t = tid; t < N_len; t += nthreads)
      s_col_buf[t] = col_src[t];
    __syncthreads();
    s_row = s_row_buf;
    s_col = s_col_buf;
  } else {
    s_row = row_src;
    s_col = col_src;
  }

  // Band boundaries
  const bool use_band = (band >= 0) && (M > 1) && (N_len > band + 1);
  const double slope  = (M > 1) ? (double)(N_len - 1) / (double)(M - 1) : 0.0;
  const double window = (band >= 0) ? fmax((double)band, slope / 2.0) : 0.0;

  int *s_j_low = nullptr;
  int *s_j_high = nullptr;
  if (use_band) {
    T *band_base;
    if (preload)
      band_base = smem + 5 * max_L;
    else if (use_double_buf)
      band_base = smem + 2 * max_L;
    else
      band_base = smem + 3 * max_L;

    s_j_low  = reinterpret_cast<int *>(band_base);
    s_j_high = s_j_low + max_L;

    for (int t = tid; t < M; t += nthreads) {
      double center = slope * t;
      s_j_low[t]  = max(0, (int)ceil(round(100.0 * (center - window)) / 100.0));
      s_j_high[t] = min(N_len - 1, (int)floor(round(100.0 * (center + window)) / 100.0));
    }
    __syncthreads();
  }

  const int total_diags = M + N_len - 1;

  if (use_double_buf) {
    for (int k = 0; k < total_diags; ++k) {
      const int i_min = max(0, k - N_len + 1);
      const int i_max = min(k, M - 1);
      const int len_k = i_max - i_min + 1;
      const int i_min_k1 = max(0, (k - 1) - N_len + 1);
      const int i_min_k2 = max(0, (k - 2) - N_len + 1);

      T *cur  = diag_buf[k & 1];
      T *prev = diag_buf[(k & 1) ^ 1];

      constexpr int MAX_SI = 8;
      T cd[MAX_SI];
      for (int p = tid, s = 0; p < len_k && s < MAX_SI; p += nthreads, ++s) {
        int i = i_min + p;
        cd[s] = (k >= 2 && i > 0 && (k - i) > 0)
                    ? cur[(i - 1) - i_min_k2] : INF;
      }
      __syncthreads();

      for (int p = tid, s = 0; p < len_k && s < MAX_SI; p += nthreads, ++s) {
        int i = i_min + p, j = k - i;
        if (use_band && (j < s_j_low[i] || j > s_j_high[i])) {
          cur[p] = INF; continue;
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
        if (use_band && (j < s_j_low[i] || j > s_j_high[i])) {
          cur[p] = INF; continue;
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

  if (tid == 0) {
    int last_buf = use_double_buf ? ((total_diags - 1) & 1)
                                  : ((total_diags - 1) % 3);
    distances[query_idx * N + target_idx] = diag_buf[last_buf][0];
  }
}

// ── Warp kernel for 1-vs-N (L <= 32) ───────────────────────────────────

template <typename T>
__global__ void dtw_one_vs_all_warp_kernel(
    const T *__restrict__ queries,
    const int *__restrict__ query_lengths,
    const T *__restrict__ all_series,
    const int *__restrict__ lengths,
    T *__restrict__ distances,
    int max_L, int N, int K, bool use_squared_l2, int band)
{
  constexpr int PPB = PAIRS_PER_BLOCK;
  const int warp_id = threadIdx.x / 32;
  const int lane    = threadIdx.x % 32;
  const int target_idx = blockIdx.x * PPB + warp_id;
  const int query_idx  = blockIdx.y;

  const T INF = (sizeof(T) == 4)
      ? static_cast<T>(3.402823466e+38f)
      : static_cast<T>(1.7976931348623157e+308);

  if (target_idx >= N || query_idx >= K) return;

  const int qlen = query_lengths[query_idx];
  const int tlen = lengths[target_idx];

  if (qlen == 0 || tlen == 0) {
    if (lane == 0)
      distances[query_idx * N + target_idx] = INF;
    return;
  }

  const T *raw_query  = queries + static_cast<long long>(query_idx) * max_L;
  const T *raw_target = all_series + static_cast<long long>(target_idx) * max_L;

  const T *row_g = (qlen <= tlen) ? raw_query  : raw_target;
  const T *col_g = (qlen <= tlen) ? raw_target : raw_query;
  const int M     = min(qlen, tlen);
  const int N_len = max(qlen, tlen);

  extern __shared__ char smem_raw[];
  T *smem = reinterpret_cast<T *>(smem_raw);
  T *my_row = smem + warp_id * 64;
  T *my_col = smem + warp_id * 64 + 32;

  if (lane < M)     my_row[lane] = row_g[lane];
  if (lane < N_len) my_col[lane] = col_g[lane];
  __syncwarp();

  const bool use_band_flag = (band >= 0) && (M > 1) && (N_len > band + 1);
  const double slope  = (M > 1) ? (double)(N_len - 1) / (double)(M - 1) : 0.0;
  const double window = (band >= 0) ? fmax((double)band, slope / 2.0) : 0.0;

  int my_j_low = 0, my_j_high = N_len - 1;
  if (use_band_flag && lane < M) {
    double center = slope * lane;
    my_j_low  = max(0, (int)ceil(round(100.0 * (center - window)) / 100.0));
    my_j_high = min(N_len - 1, (int)floor(round(100.0 * (center + window)) / 100.0));
  }

  const unsigned FULL_MASK = 0xFFFFFFFF;
  T prev_val  = INF;
  T prev2_val = INF;
  const int total_diags = M + N_len - 1;

  for (int k = 0; k < total_diags; ++k) {
    const int i = lane;
    const int j = k - lane;

    T cost_above = __shfl_sync(FULL_MASK, prev_val, lane - 1);
    T cost_diag  = __shfl_sync(FULL_MASK, prev2_val, lane - 1);
    T cost_left  = prev_val;

    const bool valid = (i < M) && (j >= 0) && (j < N_len);
    T my_current = INF;

    if (valid) {
      bool in_band = true;
      if (use_band_flag && (j < my_j_low || j > my_j_high))
        in_band = false;

      if (in_band) {
        T diff = my_row[i] - my_col[j];
        T d = use_squared_l2 ? (diff * diff) : fabs(diff);

        if (i == 0 && j == 0) {
          my_current = d;
        } else {
          if (lane == 0) { cost_above = INF; cost_diag = INF; }
          if (j == 0) cost_left = INF;
          my_current = fmin(cost_diag, fmin(cost_above, cost_left)) + d;
        }
      }
    }

    prev2_val = prev_val;
    prev_val  = my_current;
  }

  if (lane == M - 1)
    distances[query_idx * N + target_idx] = prev_val;
}

// ── Register-tiled kernel for 1-vs-N (32 < L <= 256) ───────────────────

template <typename T, int TILE_W>
__global__ void dtw_one_vs_all_regtile_kernel(
    const T *__restrict__ queries,
    const int *__restrict__ query_lengths,
    const T *__restrict__ all_series,
    const int *__restrict__ lengths,
    T *__restrict__ distances,
    int max_L, int N, int K, bool use_squared_l2, int band)
{
  constexpr int WARP_SIZE = 32;
  constexpr unsigned FULL_MASK = 0xFFFFFFFF;
  constexpr int PPB = PAIRS_PER_BLOCK;

  const int warp_id = threadIdx.x / WARP_SIZE;
  const int lane    = threadIdx.x % WARP_SIZE;
  const int target_idx = blockIdx.x * PPB + warp_id;
  const int query_idx  = blockIdx.y;

  const T INF_VAL = (sizeof(T) == 4)
      ? static_cast<T>(3.402823466e+38f)
      : static_cast<T>(1.7976931348623157e+308);

  if (target_idx >= N || query_idx >= K) return;

  const int qlen = query_lengths[query_idx];
  const int tlen = lengths[target_idx];

  if (qlen == 0 || tlen == 0) {
    if (lane == 0)
      distances[query_idx * N + target_idx] = INF_VAL;
    return;
  }

  const T *raw_query  = queries + static_cast<long long>(query_idx) * max_L;
  const T *raw_target = all_series + static_cast<long long>(target_idx) * max_L;

  const T *row_g = (qlen <= tlen) ? raw_query  : raw_target;
  const T *col_g = (qlen <= tlen) ? raw_target : raw_query;
  const int M     = min(qlen, tlen);
  const int N_len = max(qlen, tlen);

  extern __shared__ char smem_raw[];
  T *smem = reinterpret_cast<T *>(smem_raw);
  T *my_row = smem + warp_id * 2 * max_L;
  T *my_col = my_row + max_L;

  for (int t = lane; t < M; t += WARP_SIZE)
    my_row[t] = row_g[t];
  for (int t = lane; t < N_len; t += WARP_SIZE)
    my_col[t] = col_g[t];
  __syncwarp();

  const bool use_band_flag = (band >= 0) && (M > 1) && (N_len > band + 1);
  const double slope  = (M > 1) ? (double)(N_len - 1) / (double)(M - 1) : 0.0;
  const double window = (band >= 0) ? fmax((double)band, slope / 2.0) : 0.0;

  int *my_j_low_arr = nullptr;
  int *my_j_high_arr = nullptr;
  if (use_band_flag) {
    char *band_base = smem_raw + PPB * 2 * max_L * sizeof(T)
                    + warp_id * 2 * max_L * sizeof(int);
    my_j_low_arr  = reinterpret_cast<int *>(band_base);
    my_j_high_arr = my_j_low_arr + max_L;

    for (int t = lane; t < M; t += WARP_SIZE) {
      double center = slope * t;
      my_j_low_arr[t]  = max(0, (int)ceil(round(100.0 * (center - window)) / 100.0));
      my_j_high_arr[t] = min(N_len - 1, (int)floor(round(100.0 * (center + window)) / 100.0));
    }
    __syncwarp();
  }

  const int col_start = lane * TILE_W;
  T col_val[TILE_W];
  for (int tw = 0; tw < TILE_W; ++tw) {
    int j = col_start + tw;
    col_val[tw] = (j < N_len) ? my_col[j] : T(0);
  }

  T penalty[TILE_W];
  T prev_penalty[TILE_W];
  for (int tw = 0; tw < TILE_W; ++tw) {
    penalty[tw]      = INF_VAL;
    prev_penalty[tw] = INF_VAL;
  }
  T prev_last = INF_VAL;

  const int num_col_threads = (N_len + TILE_W - 1) / TILE_W;
  const int total_steps = M + num_col_threads - 1;

  for (int step = 0; step < total_steps; ++step) {
    const int i = step - lane;
    const bool row_valid = (i >= 0) && (i < M) && (lane < num_col_threads);
    T row_val = (row_valid) ? my_row[i] : T(0);

    T penalty_from_left = __shfl_sync(FULL_MASK, penalty[TILE_W - 1], lane - 1);
    T diag_from_left = __shfl_sync(FULL_MASK, prev_last, lane - 1);

    if (lane == 0) {
      penalty_from_left = INF_VAL;
      diag_from_left    = INF_VAL;
    }

    if (row_valid) {
      T saved_prev_last = prev_penalty[TILE_W - 1];
      T left = penalty_from_left;
      T diag = diag_from_left;

      for (int tw = 0; tw < TILE_W; ++tw) {
        const int j = col_start + tw;
        if (j >= N_len) {
          penalty[tw] = INF_VAL;
          diag = prev_penalty[tw];
          left = INF_VAL;
          continue;
        }

        T above = prev_penalty[tw];

        bool in_band = true;
        if (use_band_flag && (j < my_j_low_arr[i] || j > my_j_high_arr[i]))
          in_band = false;

        if (!in_band) {
          diag = prev_penalty[tw];
          left = INF_VAL;
          penalty[tw] = INF_VAL;
          continue;
        }

        T diff = row_val - col_val[tw];
        T d = use_squared_l2 ? (diff * diff) : fabs(diff);

        T new_cost;
        if (i == 0 && j == 0) {
          new_cost = d;
        } else {
          T eff_above = (i == 0) ? INF_VAL : above;
          T eff_diag  = (i == 0 || j == 0) ? INF_VAL : diag;
          T eff_left  = (j == 0) ? INF_VAL : left;
          new_cost = fmin(eff_diag, fmin(eff_above, eff_left)) + d;
        }

        diag = prev_penalty[tw];
        left = new_cost;
        penalty[tw] = new_cost;
      }

      prev_last = saved_prev_last;
      for (int tw = 0; tw < TILE_W; ++tw)
        prev_penalty[tw] = penalty[tw];
    }
  }

  const int result_thread = (N_len - 1) / TILE_W;
  const int result_tw     = (N_len - 1) % TILE_W;
  T my_result = (lane == result_thread) ? penalty[result_tw] : INF_VAL;
  T final_result = __shfl_sync(FULL_MASK, my_result, result_thread);

  if (lane == 0)
    distances[query_idx * N + target_idx] = final_result;
}

// =========================================================================
// Templated kernel launch helper for 1-vs-N / K-vs-N (anonymous namespace)
// =========================================================================

namespace {

/// Launch the 1-vs-N DTW kernel for K queries against N targets.
/// Returns K*N flat distance array (row-major).
template <typename T>
std::vector<double> launch_one_vs_all_kernel(
    const std::vector<std::vector<double>> &queries_vec,
    const std::vector<int> &query_lengths_vec,
    const std::vector<std::vector<double>> &series,
    const std::vector<int> &lengths,
    size_t K, size_t N, size_t max_L,
    bool use_squared_l2, int band, int device_id, double &gpu_time_sec)
{
  using dtwc::cuda::cuda_alloc;
  using dtwc::cuda::pinned_alloc_nothrow;
  using dtwc::cuda::make_cuda_stream;
  using dtwc::cuda::make_cuda_event;
  using dtwc::cuda::PinnedPtr;

  const size_t series_bytes = N * max_L * sizeof(T);
  const size_t query_bytes  = K * max_L * sizeof(T);
  const size_t output_elems = K * N;
  const size_t output_bytes = output_elems * sizeof(T);

  // Pinned host memory for large buffers
  constexpr size_t PINNED_THRESHOLD = 256 * 1024;
  auto h_pinned_series = (series_bytes >= PINNED_THRESHOLD)
      ? pinned_alloc_nothrow<T>(N * max_L) : PinnedPtr<T>(nullptr);
  auto h_pinned_queries = (query_bytes >= PINNED_THRESHOLD)
      ? pinned_alloc_nothrow<T>(K * max_L) : PinnedPtr<T>(nullptr);
  auto h_pinned_output = (output_bytes >= PINNED_THRESHOLD)
      ? pinned_alloc_nothrow<T>(output_elems) : PinnedPtr<T>(nullptr);

  std::vector<T> series_fallback, queries_fallback, output_fallback;

  T *h_flat_series  = nullptr;
  T *h_flat_queries = nullptr;
  T *h_output       = nullptr;

  if (h_pinned_series) {
    h_flat_series = h_pinned_series.get();
    std::memset(h_flat_series, 0, series_bytes);
  } else {
    series_fallback.resize(N * max_L, T(0));
    h_flat_series = series_fallback.data();
  }

  if (h_pinned_queries) {
    h_flat_queries = h_pinned_queries.get();
    std::memset(h_flat_queries, 0, query_bytes);
  } else {
    queries_fallback.resize(K * max_L, T(0));
    h_flat_queries = queries_fallback.data();
  }

  if (h_pinned_output) {
    h_output = h_pinned_output.get();
  } else {
    output_fallback.resize(output_elems);
    h_output = output_fallback.data();
  }

  // Flatten series data
  for (size_t i = 0; i < N; ++i)
    for (size_t k = 0; k < series[i].size(); ++k)
      h_flat_series[i * max_L + k] = static_cast<T>(series[i][k]);

  // Flatten query data
  for (size_t i = 0; i < K; ++i)
    for (size_t k = 0; k < queries_vec[i].size(); ++k)
      h_flat_queries[i * max_L + k] = static_cast<T>(queries_vec[i][k]);

  // CUDA stream and events
  auto stream    = make_cuda_stream();
  auto evt_start = make_cuda_event();
  auto evt_end   = make_cuda_event();

  // Device memory (RAII)
  auto d_series    = cuda_alloc<T>(N * max_L);
  auto d_queries   = cuda_alloc<T>(K * max_L);
  auto d_lengths   = cuda_alloc<int>(N);
  auto d_qlengths  = cuda_alloc<int>(K);
  auto d_distances = cuda_alloc<T>(output_elems);

  CUDA_CHECK(cudaEventRecord(evt_start.get(), stream.get()));

  // Async H2D transfers
  CUDA_CHECK(cudaMemcpyAsync(d_series.get(), h_flat_series,
                              series_bytes, cudaMemcpyHostToDevice, stream.get()));
  CUDA_CHECK(cudaMemcpyAsync(d_queries.get(), h_flat_queries,
                              query_bytes, cudaMemcpyHostToDevice, stream.get()));
  CUDA_CHECK(cudaMemcpyAsync(d_lengths.get(), lengths.data(),
                              N * sizeof(int), cudaMemcpyHostToDevice, stream.get()));
  CUDA_CHECK(cudaMemcpyAsync(d_qlengths.get(), query_lengths_vec.data(),
                              K * sizeof(int), cudaMemcpyHostToDevice, stream.get()));

  // Zero-initialize output
  CUDA_CHECK(cudaMemsetAsync(d_distances.get(), 0, output_bytes, stream.get()));

  const int N_int = static_cast<int>(N);
  const int K_int = static_cast<int>(K);
  const int max_L_int = static_cast<int>(max_L);

  // Kernel dispatch based on max_L (same thresholds as pairwise kernels)
  if (max_L <= 32) {
    constexpr int ppb = PAIRS_PER_BLOCK;
    const int grid_x = static_cast<int>((N + ppb - 1) / ppb);
    dim3 grid(grid_x, K_int);
    constexpr int block_size = ppb * 32;
    const size_t shared_mem = ppb * 2 * 32 * sizeof(T);

    dtw_one_vs_all_warp_kernel<T><<<grid, block_size, shared_mem, stream.get()>>>(
        d_queries.get(), d_qlengths.get(), d_series.get(), d_lengths.get(),
        d_distances.get(), max_L_int, N_int, K_int, use_squared_l2, band);

  } else if (max_L <= 128) {
    constexpr int ppb = PAIRS_PER_BLOCK;
    constexpr int TILE_W = 4;
    const int grid_x = static_cast<int>((N + ppb - 1) / ppb);
    dim3 grid(grid_x, K_int);
    constexpr int block_size = ppb * 32;
    const size_t series_smem = ppb * 2 * max_L * sizeof(T);
    const size_t band_smem = (band >= 0) ? ppb * 2 * max_L * sizeof(int) : 0;
    const size_t shared_mem = series_smem + band_smem;

    dtw_one_vs_all_regtile_kernel<T, TILE_W><<<grid, block_size, shared_mem, stream.get()>>>(
        d_queries.get(), d_qlengths.get(), d_series.get(), d_lengths.get(),
        d_distances.get(), max_L_int, N_int, K_int, use_squared_l2, band);

  } else if (max_L <= 256) {
    constexpr int ppb = PAIRS_PER_BLOCK;
    constexpr int TILE_W = 8;
    const int grid_x = static_cast<int>((N + ppb - 1) / ppb);
    dim3 grid(grid_x, K_int);
    constexpr int block_size = ppb * 32;
    const size_t series_smem = ppb * 2 * max_L * sizeof(T);
    const size_t band_smem = (band >= 0) ? ppb * 2 * max_L * sizeof(int) : 0;
    const size_t shared_mem = series_smem + band_smem;

    dtw_one_vs_all_regtile_kernel<T, TILE_W><<<grid, block_size, shared_mem, stream.get()>>>(
        d_queries.get(), d_qlengths.get(), d_series.get(), d_lengths.get(),
        d_distances.get(), max_L_int, N_int, K_int, use_squared_l2, band);

  } else {
    // Wavefront kernel: one block per target, grid.y = K queries
    // max_L > 256 here: never preload, use 2 or 3 diag buffers
    const size_t n_bufs = (max_L > 1024) ? 2 : 3;
    size_t shared_mem = n_bufs * max_L * sizeof(T);
    if (band >= 0)
      shared_mem += 2 * max_L * sizeof(int);

    int block_size;
    if (max_L <= 512)      block_size = 128;
    else                   block_size = 256;

    if (shared_mem > 48 * 1024) {
      CUDA_CHECK(cudaFuncSetAttribute(dtw_one_vs_all_wavefront_kernel<T>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(shared_mem)));
    }

    dim3 grid(N_int, K_int);
    dtw_one_vs_all_wavefront_kernel<T><<<grid, block_size, shared_mem, stream.get()>>>(
        d_queries.get(), d_qlengths.get(), d_series.get(), d_lengths.get(),
        d_distances.get(), max_L_int, N_int, K_int, use_squared_l2, band);
  }

  CUDA_CHECK(cudaGetLastError());

  // D2H transfer
  CUDA_CHECK(cudaMemcpyAsync(h_output, d_distances.get(),
                              output_bytes, cudaMemcpyDeviceToHost, stream.get()));
  CUDA_CHECK(cudaEventRecord(evt_end.get(), stream.get()));
  CUDA_CHECK(cudaStreamSynchronize(stream.get()));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, evt_start.get(), evt_end.get()));
  gpu_time_sec = static_cast<double>(elapsed_ms) / 1000.0;

  // Convert to double
  std::vector<double> result(output_elems);
  for (size_t i = 0; i < output_elems; ++i)
    result[i] = static_cast<double>(h_output[i]);
  return result;
}

} // anonymous namespace

// =========================================================================
// Public API: compute_dtw_one_vs_all (by index)
// =========================================================================

CUDAOneVsNResult compute_dtw_one_vs_all(
    const std::vector<std::vector<double>> &series,
    size_t query_index,
    const CUDADistMatOptions &opts)
{
  CUDAOneVsNResult result;
  const size_t N = series.size();
  result.n = N;
  result.distances.resize(N, 0.0);

  if (N == 0 || !cuda_available()) return result;
  if (query_index >= N) {
    throw std::runtime_error("query_index " + std::to_string(query_index) +
                             " out of range [0, " + std::to_string(N) + ")");
  }

  CUDA_CHECK(cudaSetDevice(opts.device_id));

  size_t max_L = 0;
  std::vector<int> lengths(N);
  for (size_t i = 0; i < N; ++i) {
    lengths[i] = static_cast<int>(series[i].size());
    max_L = std::max(max_L, series[i].size());
  }
  if (max_L == 0) return result;

  std::vector<std::vector<double>> queries_vec = { series[query_index] };
  std::vector<int> query_lengths_vec = { static_cast<int>(series[query_index].size()) };

  bool use_fp32 = false;
  switch (opts.precision) {
    case CUDAPrecision::FP32: use_fp32 = true; break;
    case CUDAPrecision::FP64: use_fp32 = false; break;
    case CUDAPrecision::Auto:
    default: {
      auto gpu_cfg = query_gpu_config(opts.device_id);
      use_fp32 = (gpu_cfg.fp64_rate == FP64Rate::Slow);
      break;
    }
  }

  if (use_fp32) {
    result.distances = launch_one_vs_all_kernel<float>(
        queries_vec, query_lengths_vec, series, lengths,
        1, N, max_L, opts.use_squared_l2, opts.band, opts.device_id,
        result.gpu_time_sec);
  } else {
    result.distances = launch_one_vs_all_kernel<double>(
        queries_vec, query_lengths_vec, series, lengths,
        1, N, max_L, opts.use_squared_l2, opts.band, opts.device_id,
        result.gpu_time_sec);
  }

  if (opts.verbose) {
    std::cout << "CUDA 1-vs-" << N << " DTW"
              << (use_fp32 ? " [FP32]" : " [FP64]")
              << " in " << result.gpu_time_sec * 1000 << "ms on "
              << cuda_device_info(opts.device_id) << std::endl;
  }

  return result;
}

// =========================================================================
// Public API: compute_dtw_one_vs_all (external query)
// =========================================================================

CUDAOneVsNResult compute_dtw_one_vs_all(
    const std::vector<double> &query,
    const std::vector<std::vector<double>> &series,
    const CUDADistMatOptions &opts)
{
  CUDAOneVsNResult result;
  const size_t N = series.size();
  result.n = N;
  result.distances.resize(N, 0.0);

  if (N == 0 || !cuda_available()) return result;

  CUDA_CHECK(cudaSetDevice(opts.device_id));

  size_t max_L = query.size();
  std::vector<int> lengths(N);
  for (size_t i = 0; i < N; ++i) {
    lengths[i] = static_cast<int>(series[i].size());
    max_L = std::max(max_L, series[i].size());
  }
  if (max_L == 0) return result;

  std::vector<std::vector<double>> queries_vec = { query };
  std::vector<int> query_lengths_vec = { static_cast<int>(query.size()) };

  bool use_fp32 = false;
  switch (opts.precision) {
    case CUDAPrecision::FP32: use_fp32 = true; break;
    case CUDAPrecision::FP64: use_fp32 = false; break;
    case CUDAPrecision::Auto:
    default: {
      auto gpu_cfg = query_gpu_config(opts.device_id);
      use_fp32 = (gpu_cfg.fp64_rate == FP64Rate::Slow);
      break;
    }
  }

  if (use_fp32) {
    result.distances = launch_one_vs_all_kernel<float>(
        queries_vec, query_lengths_vec, series, lengths,
        1, N, max_L, opts.use_squared_l2, opts.band, opts.device_id,
        result.gpu_time_sec);
  } else {
    result.distances = launch_one_vs_all_kernel<double>(
        queries_vec, query_lengths_vec, series, lengths,
        1, N, max_L, opts.use_squared_l2, opts.band, opts.device_id,
        result.gpu_time_sec);
  }

  if (opts.verbose) {
    std::cout << "CUDA 1-vs-" << N << " DTW (external query)"
              << (use_fp32 ? " [FP32]" : " [FP64]")
              << " in " << result.gpu_time_sec * 1000 << "ms on "
              << cuda_device_info(opts.device_id) << std::endl;
  }

  return result;
}

// =========================================================================
// Public API: compute_dtw_k_vs_all
// =========================================================================

CUDAKVsNResult compute_dtw_k_vs_all(
    const std::vector<std::vector<double>> &series,
    const std::vector<size_t> &query_indices,
    const CUDADistMatOptions &opts)
{
  CUDAKVsNResult result;
  const size_t N = series.size();
  const size_t K = query_indices.size();
  result.n = N;
  result.k = K;
  result.distances.resize(K * N, 0.0);

  if (N == 0 || K == 0 || !cuda_available()) return result;

  for (size_t qi = 0; qi < K; ++qi) {
    if (query_indices[qi] >= N) {
      throw std::runtime_error("query_indices[" + std::to_string(qi) + "] = " +
                               std::to_string(query_indices[qi]) +
                               " out of range [0, " + std::to_string(N) + ")");
    }
  }

  CUDA_CHECK(cudaSetDevice(opts.device_id));

  size_t max_L = 0;
  std::vector<int> lengths(N);
  for (size_t i = 0; i < N; ++i) {
    lengths[i] = static_cast<int>(series[i].size());
    max_L = std::max(max_L, series[i].size());
  }
  if (max_L == 0) return result;

  std::vector<std::vector<double>> queries_vec(K);
  std::vector<int> query_lengths_vec(K);
  for (size_t qi = 0; qi < K; ++qi) {
    queries_vec[qi] = series[query_indices[qi]];
    query_lengths_vec[qi] = static_cast<int>(series[query_indices[qi]].size());
  }

  bool use_fp32 = false;
  switch (opts.precision) {
    case CUDAPrecision::FP32: use_fp32 = true; break;
    case CUDAPrecision::FP64: use_fp32 = false; break;
    case CUDAPrecision::Auto:
    default: {
      auto gpu_cfg = query_gpu_config(opts.device_id);
      use_fp32 = (gpu_cfg.fp64_rate == FP64Rate::Slow);
      break;
    }
  }

  if (use_fp32) {
    result.distances = launch_one_vs_all_kernel<float>(
        queries_vec, query_lengths_vec, series, lengths,
        K, N, max_L, opts.use_squared_l2, opts.band, opts.device_id,
        result.gpu_time_sec);
  } else {
    result.distances = launch_one_vs_all_kernel<double>(
        queries_vec, query_lengths_vec, series, lengths,
        K, N, max_L, opts.use_squared_l2, opts.band, opts.device_id,
        result.gpu_time_sec);
  }

  if (opts.verbose) {
    std::cout << "CUDA " << K << "-vs-" << N << " DTW"
              << (use_fp32 ? " [FP32]" : " [FP64]")
              << " in " << result.gpu_time_sec * 1000 << "ms on "
              << cuda_device_info(opts.device_id) << std::endl;
  }

  return result;
}

}  // namespace dtwc::cuda

#endif  // DTWC_HAS_CUDA
