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
#include "kernel_selection.hpp"
#include "launch_prep.hpp"
#include "../detail/decode_pair.hpp"

#ifdef DTWC_HAS_CUDA

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace dtwc::cuda {

// =========================================================================
// Device helper: decode flat upper-triangle pair index to (i, j)
// =========================================================================
//
// The decode is the SSOT dtwc::detail::decode_pair (dtwc/detail/decode_pair.hpp),
// shared with the host and marked __host__ __device__. It uses an FP64 seed +
// int64 correction, fixing the int32 overflow of the retired local copy.
using dtwc::detail::decode_pair;

namespace {

__device__ __forceinline__ bool fixed_band_contains(int i, int j, int band)
{
  if (band < 0) return true;
  return (i >= j) ? (i - j <= band) : (j - i <= band);
}

/// What the fill needs to know about a device.
struct DeviceLimits {
  bool slow_fp64;              ///< FP32 runs more than twice as fast as FP64
  int sm_count;                ///< multiprocessors, for the persistent grid
  size_t max_shared_per_block; ///< opt-in maximum, static + dynamic
};

/// The limits of every device, read once per process and read-only after.
/// @p device_id must have passed cudaSetDevice.
const DeviceLimits &device_limits(int device_id)
{
  static std::vector<DeviceLimits> limits;
  static cudaError_t read_error = cudaSuccess;
  static std::once_flag read_once;
  // The callable records an error rather than throwing it: where call_once
  // runs on glibc's pthread_once, a throw can hang the next caller on non-x86
  // targets such as aarch64 (GCC PR 66146).
  std::call_once(read_once, [] {
    const auto read = [](cudaDeviceAttr attribute, int device) {
      int value = 0;
      if (read_error == cudaSuccess)
        read_error = cudaDeviceGetAttribute(&value, attribute, device);
      return value;
    };
    int count = 0;
    read_error = cudaGetDeviceCount(&count);
    for (int d = 0; d < count; ++d)
      limits.push_back({
          read(cudaDevAttrSingleToDoublePrecisionPerfRatio, d) > 2,
          read(cudaDevAttrMultiProcessorCount, d),
          static_cast<size_t>(read(cudaDevAttrMaxSharedMemoryPerBlockOptin, d))});
  });
  CUDA_CHECK(read_error);
  return limits[static_cast<size_t>(device_id)];
}

/// Auto takes FP64 only where it runs at least half as fast as FP32 (the HPC
/// parts); a compute-capability table misread consumer Blackwell (sm_120).
bool resolve_fp32(CUDAPrecision precision, int device_id)
{
  switch (precision) {
  case CUDAPrecision::FP32:
    return true;
  case CUDAPrecision::FP64:
    return false;
  case CUDAPrecision::Auto:
    return device_limits(device_id).slow_fp64;
  }
  throw std::logic_error("resolve_fp32: unreachable CUDAPrecision");
}

/// C6: the queried opt-in shared-memory cap was computed and never read, so an
/// over-large request surfaced as a bare "invalid argument" from CUDA.
void require_shared_mem_fits(size_t shared_mem, int device_id, const char *what)
{
  const size_t cap = device_limits(device_id).max_shared_per_block;
  if (cap > 0 && shared_mem > cap)
    throw dtwc::DeviceError(
        std::string(what) + ": needs " + std::to_string(shared_mem)
        + " bytes of shared memory per block, but device "
        + std::to_string(device_id) + " allows at most " + std::to_string(cap)
        + ". Reduce the series length or the band.");
}

} // namespace

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
  constexpr int PRELOAD_THRESHOLD = 512;
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
  // Task 0.1: the double-buffer path caches each thread's cost-diagonal in a
  // fixed MAX_SI(8)-element register array, so an anti-diagonal longer than
  // blockDim.x(256)*MAX_SI = 2048 silently drops cells -> wrong DTW. Cap the
  // double-buffer path at 2048; longer series take the 3-buffer path (which
  // grid-strides every cell). The host mirrors this cap in n_bufs.
  constexpr int DOUBLE_BUF_MAX = 2048;
  const bool use_double_buf =
      (max_L > DOUBLE_BUF_THRESHOLD) && (max_L <= DOUBLE_BUF_MAX) && !preload;

  T *s_row_buf = nullptr;
  T *s_col_buf = nullptr;
  T *diag_buf[3]; // [0],[1] always used; [2] only in 3-buffer mode

  if (preload) {
    s_row_buf = smem;
    s_col_buf = smem + max_L;
    diag_buf[0] = smem + 2 * max_L;
    diag_buf[1] = smem + 3 * max_L;
    diag_buf[2] = smem + 4 * max_L; // 3-buffer always for preload
  } else if (use_double_buf) {
    diag_buf[0] = smem;
    diag_buf[1] = smem + max_L;
    diag_buf[2] = nullptr; // not used
  } else {
    diag_buf[0] = smem;
    diag_buf[1] = smem + max_L;
    diag_buf[2] = smem + 2 * max_L;
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

    std::int64_t si, sj;
    decode_pair(pid, N_series, si, sj);
    // Task 0.7: si/sj are int64 so the result_matrix[si*N_series+sj] writes
    // below are evaluated in 64-bit (the int32 index wrapped for N >= 46341).
    static_assert(sizeof(decltype(si * N_series + sj)) >= 8,
                  "matrix index must be 64-bit to avoid overflow at N>=46341");
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
          if (!fixed_band_contains(i, j, band)) {
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
          if (!fixed_band_contains(i, j, band)) {
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
  const int work_idx = blockIdx.x * PAIRS_PER_BLOCK + warp_id;  // global pair id

  const T INF = (sizeof(T) == 4)
      ? static_cast<T>(3.402823466e+38f)
      : static_cast<T>(1.7976931348623157e+308);

  if (work_idx >= num_pairs) return;

  std::int64_t si, sj;
  decode_pair(work_idx, N_series, si, sj);
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
      if (fixed_band_contains(i, j, band)) {
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
  const int work_idx = blockIdx.x * PAIRS_PER_BLOCK + warp_id;

  const T INF_VAL = (sizeof(T) == 4)
      ? static_cast<T>(3.402823466e+38f)
      : static_cast<T>(1.7976931348623157e+308);

  if (work_idx >= num_pairs) return;

  // Decode pair from flat upper-triangle index
  std::int64_t si, sj;
  decode_pair(work_idx, N_series, si, sj);
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

        if (!fixed_band_contains(i, j, band)) {
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

template <typename T>
struct CachedHostBuffer {
  size_t capacity = 0;
  PinnedPtr<T> pinned;
  std::vector<T> fallback;

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
        return pinned.get();
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
  size_t matrix_capacity = 0;
  size_t counter_capacity = 0;
  CachedHostBuffer<T> host_series;
  CachedHostBuffer<T> host_matrix;
  CudaPtr<T> d_series;
  CudaPtr<int> d_lengths;
  CudaPtr<T> d_result_matrix;
  CudaPtr<int> d_counter;
  CudaStream stream;
  CudaEvent evt_start;
  CudaEvent evt_end;

  void ensure_runtime(int new_device_id)
  {
    if (device_id != new_device_id) {
      d_series.reset();
      d_lengths.reset();
      d_result_matrix.reset();
      d_counter.reset();
      stream.reset();
      evt_start.reset();
      evt_end.reset();
      series_capacity = 0;
      length_capacity = 0;
      matrix_capacity = 0;
      counter_capacity = 0;
      device_id = new_device_id;
    }

    if (!stream)
      stream = make_cuda_stream();
    if (!evt_start)
      evt_start = make_cuda_event();
    if (!evt_end)
      evt_end = make_cuda_event();
  }
};

template <typename T>
DTWLaunchWorkspace<T> &get_dtw_launch_workspace(int device_id)
{
  thread_local DTWLaunchWorkspace<T> workspace;
  workspace.ensure_runtime(device_id);
  return workspace;
}

template <typename T>
void ensure_dtw_device_capacity(
    DTWLaunchWorkspace<T> &workspace,
    size_t series_elems,
    size_t length_elems,
    size_t matrix_elems)
{
  if (workspace.series_capacity < series_elems) {
    workspace.d_series = cuda_alloc<T>(series_elems);
    workspace.series_capacity = series_elems;
  }
  if (workspace.length_capacity < length_elems) {
    workspace.d_lengths = cuda_alloc<int>(length_elems);
    workspace.length_capacity = length_elems;
  }
  if (workspace.matrix_capacity < matrix_elems) {
    workspace.d_result_matrix = cuda_alloc<T>(matrix_elems);
    workspace.matrix_capacity = matrix_elems;
  }
}

template <typename T>
void flatten_series_buffer(
    T *dst,
    const std::vector<std::vector<double>> &series,
    size_t max_L)
{
  for (size_t i = 0; i < series.size(); ++i) {
    const auto &src = series[i];
    T *row_dst = dst + i * max_L;
    const size_t len = src.size();

    if constexpr (std::is_same_v<T, double>) {
      if (len > 0)
        std::memcpy(row_dst, src.data(), len * sizeof(double));
    } else {
      for (size_t k = 0; k < len; ++k)
        row_dst[k] = static_cast<T>(src[k]);
    }

    if (len < max_L)
      std::fill(row_dst + len, row_dst + max_L, T(0));
  }
}

template <typename T>
std::vector<double> convert_result_matrix(
    const T *src, size_t N, size_t matrix_elems)
{
  std::vector<double> result(matrix_elems);
  if constexpr (std::is_same_v<T, double>) {
    for (size_t i = 0; i < N; ++i) {
      const size_t row_offset = i * N;
      if (i > 0)
        std::memcpy(result.data() + row_offset, src + row_offset, i * sizeof(double));
      result[row_offset + i] = 0.0;
      if (i + 1 < N) {
        std::memcpy(result.data() + row_offset + i + 1,
                    src + row_offset + i + 1,
                    (N - i - 1) * sizeof(double));
      }
    }
  } else {
    for (size_t i = 0; i < N; ++i) {
      const size_t row_offset = i * N;
      for (size_t j = 0; j < i; ++j)
        result[row_offset + j] =
            dtwc::core::normalize_public_distance(src[row_offset + j]);
      result[row_offset + i] = 0.0;
      for (size_t j = i + 1; j < N; ++j)
        result[row_offset + j] =
            dtwc::core::normalize_public_distance(src[row_offset + j]);
    }
  }
  return result;
}

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
    bool use_squared_l2, int band, int device_id, double &gpu_time_sec,
    detail::KernelPath kernel_path)
{
  // Last line of defence. The public entry point applies the same guard
  // before it allocates the NxN result.
  detail::require_pair_count_fits(num_pairs, "launch_dtw_kernel");

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
  auto &workspace = get_dtw_launch_workspace<T>(device_id);
  T *h_flat_series = workspace.host_series.ensure(
      N * max_L, series_transfer >= PINNED_THRESHOLD);
  T *h_result_matrix = workspace.host_matrix.ensure(
      matrix_elems, matrix_bytes >= PINNED_THRESHOLD);
  flatten_series_buffer(h_flat_series, series, max_L);
  ensure_dtw_device_capacity(workspace, N * max_L, N, matrix_elems);

  const int N_series = static_cast<int>(N);
  auto stream = workspace.stream.get();

  // ---------------------------------------------------------------------------
  // Begin timed region: H2D transfers + kernel + D2H
  // ---------------------------------------------------------------------------
  CUDA_CHECK(cudaEventRecord(workspace.evt_start.get(), stream));

  // Async H2D transfers (no pair index arrays needed — Fix 1)
  CUDA_CHECK(cudaMemcpyAsync(workspace.d_series.get(), h_flat_series,
                              series_bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(workspace.d_lengths.get(), lengths.data(),
                              N * sizeof(int), cudaMemcpyHostToDevice, stream));

  // ---------------------------------------------------------------------------
  // Kernel launch (on the same stream -- automatically waits for H2D)
  // ---------------------------------------------------------------------------
  // The warp-family kernels run one pair per warp, PAIRS_PER_BLOCK warps per
  // block; each warp stages its pair's two series, `staged` samples apiece, in
  // shared memory.
  const auto launch_warp_family = [&](auto kernel, size_t staged) {
    const int grid_size = static_cast<int>(
        (num_pairs + PAIRS_PER_BLOCK - 1) / PAIRS_PER_BLOCK);
    const size_t shared_mem = PAIRS_PER_BLOCK * 2 * staged * sizeof(T);
    kernel<<<grid_size, PAIRS_PER_BLOCK * 32, shared_mem, stream>>>(
        workspace.d_series.get(), workspace.d_lengths.get(), workspace.d_result_matrix.get(),
        N_series, static_cast<int>(max_L),
        static_cast<int>(num_pairs), use_squared_l2, band);
  };

  if (kernel_path == detail::KernelPath::Warp) {
    launch_warp_family(dtw_warp_kernel<T>, 32);
  } else if (kernel_path == detail::KernelPath::RegTileW4) {
    launch_warp_family(dtw_regtile_kernel<T, 4>, max_L); // 32 lanes x 4 = 128 columns
  } else if (kernel_path == detail::KernelPath::RegTileW8) {
    launch_warp_family(dtw_regtile_kernel<T, 8>, max_L); // 32 lanes x 8 = 256 columns
  } else if (kernel_path == detail::KernelPath::Wavefront) {
    // Wavefront kernel: shared memory and block size configuration
    // Buffer-count policy (incl. the Task 0.1 cap at L>2048) lives in
    // detail::wavefront_buffer_count so both dispatch paths share it.
    const size_t n_bufs = detail::wavefront_buffer_count(max_L);
    if (max_L > 2048) {
      // Lock-free warn-once: exchange() is a single RMW, never a mutex.
      static std::atomic<bool> logged{ false };
      if (!logged.exchange(true, std::memory_order_relaxed)) {
        std::cerr << "[CUDA] max_L=" << max_L
                  << " > 2048: using the 3-buffer wavefront path "
                     "(the double-buffer register cache would drop cells).\n";
      }
    }
    const size_t shared_mem = n_bufs * max_L * sizeof(T);

    // Block size heuristic tuned for the anti-diagonal wavefront pattern.
    constexpr int block_size = 256;

    // A block's shared memory is these buffers plus the kernel's static
    // variables; beyond the 48 KiB any block may use, it must be opted in.
    cudaFuncAttributes kernel_attributes{};
    CUDA_CHECK(cudaFuncGetAttributes(&kernel_attributes, dtw_wavefront_kernel<T>));
    const size_t block_shared_mem = shared_mem + kernel_attributes.sharedSizeBytes;
    require_shared_mem_fits(block_shared_mem, device_id, "dtw_wavefront_kernel");
    if (block_shared_mem > 48 * 1024) {
      CUDA_CHECK(cudaFuncSetAttribute(dtw_wavefront_kernel<T>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(shared_mem)));
    }

    // Determine whether to use persistent mode
    int blocks_per_sm = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, dtw_wavefront_kernel<T>, block_size, shared_mem);
    const int persistent_grid =
        device_limits(device_id).sm_count * std::max(blocks_per_sm, 1);
    const bool use_persistent =
        (static_cast<int>(num_pairs) > persistent_grid * 4);

    if (use_persistent) {
      if (workspace.counter_capacity < 1) {
        workspace.d_counter = cuda_alloc<int>(1);
        workspace.counter_capacity = 1;
      }
      CUDA_CHECK(cudaMemsetAsync(workspace.d_counter.get(), 0, sizeof(int), stream));

      dtw_wavefront_kernel<T><<<persistent_grid, block_size, shared_mem, stream>>>(
          workspace.d_series.get(), workspace.d_lengths.get(), workspace.d_result_matrix.get(),
          N_series, static_cast<int>(max_L),
          static_cast<int>(num_pairs), use_squared_l2, band,
          workspace.d_counter.get());
    } else {
      // Non-persistent: one block per pair (original behavior)
      dtw_wavefront_kernel<T><<<static_cast<int>(num_pairs), block_size, shared_mem, stream>>>(
          workspace.d_series.get(), workspace.d_lengths.get(), workspace.d_result_matrix.get(),
          N_series, static_cast<int>(max_L),
          static_cast<int>(num_pairs), use_squared_l2, band,
          nullptr);
    }
  } else {
    throw std::logic_error("launch_dtw_kernel: unknown KernelPath");
  }

  CUDA_CHECK(cudaGetLastError());

  // ---------------------------------------------------------------------------
  // Async D2H transfer: single contiguous NxN matrix (Fix 2 — no host fill loop)
  // ---------------------------------------------------------------------------
  CUDA_CHECK(cudaMemcpyAsync(h_result_matrix, workspace.d_result_matrix.get(),
                              matrix_bytes, cudaMemcpyDeviceToHost, stream));

  // ---------------------------------------------------------------------------
  // End timed region and synchronize
  // ---------------------------------------------------------------------------
  CUDA_CHECK(cudaEventRecord(workspace.evt_end.get(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms,
                                  workspace.evt_start.get(),
                                  workspace.evt_end.get()));
  gpu_time_sec = static_cast<double>(elapsed_ms) / 1000.0;

  return convert_result_matrix(h_result_matrix, N, matrix_elems);
}

} // anonymous namespace

CUDADistMatResult compute_distance_matrix_cuda(
    const std::vector<std::vector<double>> &series,
    const CUDADistMatOptions &opts)
{
  validate_cuda_precision(opts.precision);
  const size_t N = series.size();
  // Both guards run before the N*N allocation: a missing device must not
  // answer with a zero matrix, and a pair count that does not fit the launch
  // geometry must not be narrowed to int.
  detail::require_cuda_device(cuda_available(), "compute_distance_matrix_cuda");
  detail::require_pair_count_fits(detail::upper_triangle_pairs(N),
                                  "compute_distance_matrix_cuda");

  CUDADistMatResult result;
  result.kernel_used = "none";
  result.n = N;
  result.matrix.resize(N * N, 0.0);

  if (N <= 1) return result;

  CUDA_CHECK(cudaSetDevice(opts.device_id));

  // Find max length for padding
  std::vector<int> lengths;
  const size_t max_L = detail::scan_series_lengths(series, lengths);

  if (max_L == 0) return result;

  const auto kernel_path = detail::select_kernel(max_L);
  result.kernel_used = std::string(detail::kernel_path_name(kernel_path));

  // Fix 1: No pair index arrays — pairs are decoded on-device via decode_pair().
  // This eliminates 2 * num_pairs * sizeof(int) host allocation + H2D transfer.
  const size_t num_pairs = detail::upper_triangle_pairs(N);
  result.pairs_computed = num_pairs;

  // Determine compute precision
  const bool use_fp32 = resolve_fp32(opts.precision, opts.device_id);
  result.matrix = use_fp32
      ? launch_dtw_kernel<float>(series, lengths, N, max_L, num_pairs,
                                 opts.use_squared_l2, opts.band, opts.device_id,
                                 result.gpu_time_sec, kernel_path)
      : launch_dtw_kernel<double>(series, lengths, N, max_L, num_pairs,
                                  opts.use_squared_l2, opts.band, opts.device_id,
                                  result.gpu_time_sec, kernel_path);

  if (opts.verbose) {
    std::cout << "CUDA DTW: " << num_pairs << " pairs"
              << (use_fp32 ? " [FP32]" : " [FP64]")
              << " in " << result.gpu_time_sec * 1000 << "ms"
              << " on " << cuda_device_info(opts.device_id) << std::endl;
  }

  return result;
}

}  // namespace dtwc::cuda

#endif  // DTWC_HAS_CUDA
