/**
 * @file cuda_dtw.cu
 * @brief CUDA implementation of batch DTW distance computation.
 *
 * @details Three kernel strategies:
 *   1. dtw_wavefront_kernel: anti-diagonal wavefront parallelism with
 *      shared-memory buffers. Used for series longer than 256; where the
 *      buffers do not fit a block's shared memory, the same kernel keeps them
 *      in global memory instead. Supports two scheduling modes:
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
#include <cmath>
#include <cstring>
#include <iostream>
#include <memory>
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
using dtwc::core::normalize_public_distance;

/// The slot of DistanceMatrix's packed lower triangle (row r holds columns
/// 0..r) that takes the distance of the pair decode_pair gives, si < sj. The
/// series are uploaded last first, so (si, sj) is the caller's pair
/// (N - 1 - si, N - 1 - sj): consecutive pairs take consecutive slots, from the
/// end of the triangle backwards, skipping one diagonal slot per row. A launch
/// of consecutive pairs therefore writes one contiguous span of the matrix.
__host__ __device__ inline std::int64_t packed_slot(std::int64_t si, std::int64_t sj,
                                                    std::int64_t N)
{
  const std::int64_t row = N - 1 - si;
  return row * (row + 1) / 2 + (N - 1 - sj);
}

/// What a wavefront kernel compiles. Preload (L <= kPreloadMaxLength): both
/// series and the three anti-diagonals in shared memory, and nothing else.
/// Shared: every shared-memory mode, chosen at run time. Global: three
/// anti-diagonals in global memory.
enum class Wavefront { Preload, Shared, Global };

// Declared for the device setup below; defined with the other kernels.
template <typename T, Wavefront Mode>
__global__ void dtw_wavefront_kernel(
    const T *__restrict__ all_series, const int *__restrict__ lengths,
    double *__restrict__ out, int N_series, int max_L, int num_pairs,
    bool use_squared_l2, int band, int *__restrict__ work_counter,
    std::int64_t first_pair, std::int64_t first_slot, T *__restrict__ scratch);

namespace {

__device__ __forceinline__ bool fixed_band_contains(int i, int j, int band)
{
  if (band < 0) return true;
  return (i >= j) ? (i - j <= band) : (j - i <= band);
}

/// What the fill needs to know about a device, and the device's one-time setup.
struct DeviceLimits {
  int compute_major = 0, compute_minor = 0; ///< compute capability, 8.0 at least
  bool slow_fp64 = false;                ///< FP32 runs more than twice as fast as FP64
  int sm_count = 0;                      ///< multiprocessors, for the persistent grid
  size_t max_shared_per_block = 0;       ///< opt-in maximum, static + dynamic
  size_t l2_bytes = 0;                   ///< L2 cache, which the global wavefront's slices fit
  size_t shared_per_sm = 0;              ///< an SM's shared memory, which picks the wavefront's route
  size_t reserved_shared_per_block = 0;  ///< the runtime's own shared memory in every block
  size_t wavefront_static_bytes[2] = {}; ///< the FP32 and FP64 wavefront kernels' own
  cudaError_t setup_error = cudaSuccess;
  std::once_flag set_up;
};

/// Opens the whole opt-in shared memory of a block to the wavefront kernel in
/// T on the current device, and records the kernel's static part.
template <typename T>
cudaError_t open_wavefront_shared_memory(DeviceLimits &device)
{
  const auto kernel = dtw_wavefront_kernel<T, Wavefront::Shared>;
  cudaFuncAttributes attributes{};
  const cudaError_t error = cudaFuncGetAttributes(&attributes, kernel);
  if (error != cudaSuccess) return error;
  device.wavefront_static_bytes[std::is_same_v<T, double>] = attributes.sharedSizeBytes;
  return cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(device.max_shared_per_block - attributes.sharedSizeBytes));
}

/// Every device's limits, read once per process, and each device's setup, run
/// once on its first fill. The wavefront kernels' dynamic shared-memory limit is
/// one value per kernel and device, shared by every host thread: opened fully
/// once, it is never set per launch, where one thread lowered it under another's
/// launch. Read-only afterwards. Neither callable throws: where call_once runs on
/// glibc's pthread_once, a throw can hang the next caller on non-x86 targets
/// such as aarch64 (GCC PR 66146). @p device_id must have passed cudaSetDevice.
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
    }
  });
  CUDA_CHECK(read_error);
  DeviceLimits &device = limits[static_cast<size_t>(device_id)];
  // Before the device's setup, which needs a kernel image the device can run.
  detail::require_compute_capability(device.compute_major, device.compute_minor, device_id);
  std::call_once(device.set_up, [&device] {
    device.setup_error = open_wavefront_shared_memory<float>(device);
    if (device.setup_error == cudaSuccess)
      device.setup_error = open_wavefront_shared_memory<double>(device);
  });
  CUDA_CHECK(device.setup_error);
  return device;
}

/// Auto takes FP64 only where it runs at least half as fast as FP32 (the HPC
/// parts); a compute-capability table misread consumer Blackwell (sm_120).
bool resolve_fp32(GpuPrecision precision, int device_id)
{
  switch (precision) {
  case GpuPrecision::FP32:
    return true;
  case GpuPrecision::FP64:
    return false;
  case GpuPrecision::Auto:
    return device_limits(device_id).slow_fp64;
  }
  throw std::logic_error("resolve_fp32: unreachable GpuPrecision");
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
//
// Wavefront::Preload compiles the preload mode alone: fewer registers, more
// blocks per SM. Wavefront::Global, for series whose buffers do not fit a
// block's shared memory, keeps the same 3 * max_L T's in the block's own slice
// of `scratch` (global memory, one slice per block of the grid) and runs only
// the 3-buffer mode, persistent. Every instantiation computes the same cells,
// and so the same distances.
//
// Every kernel computes the launch's pairs first_pair .. first_pair + num_pairs - 1
// and writes each distance, widened to the public double, to its packed_slot in
// `out`, which holds the launch's slots from first_slot on.

template <typename T, Wavefront Mode>
__global__ void dtw_wavefront_kernel(
    const T *__restrict__ all_series, // [N * max_L] padded, last series first
    const int *__restrict__ lengths,       // [N] actual lengths, same order
    double *__restrict__ out,         // the launch's packed slots
    int N_series, int max_L, int num_pairs, bool use_squared_l2, int band,
    int *__restrict__ work_counter,   // persistent mode when non-null
    std::int64_t first_pair, std::int64_t first_slot,
    T *__restrict__ scratch)          // [gridDim.x * 3 * max_L], Global only
{
  constexpr bool global_buffers = Mode == Wavefront::Global;
  const int tid = threadIdx.x;
  const int nthreads = blockDim.x;

  // Use the largest representable value for the compute type
  const T INF = (sizeof(T) == 4)
      ? static_cast<T>(3.402823466e+38f)    // FLT_MAX
      : static_cast<T>(1.7976931348623157e+308); // DBL_MAX

  // Preload threshold: series shorter than this are loaded into shared memory.
  // The host launches the Preload kernel up to it, so the Shared kernel never
  // takes this branch, yet compiling it still changes the kernel. FP32 keeps
  // it: its registers (68 instead of 50) hold the kernel at three blocks per SM
  // instead of four, and without it the FP32 fills at L 513-1024 and 2048 took
  // 4-23 % more time. FP64 compiles it out: 62 registers instead of 79, and its
  // fills at L 513-1024 took 0.84-0.89 of the time. On an RTX 4000 Ada
  // (.claude/baselines/2026-09-30-c2-cuda-route.md).
  constexpr int PRELOAD_THRESHOLD = static_cast<int>(detail::kPreloadMaxLength);
  const bool preload = Mode == Wavefront::Preload
      || (Mode == Wavefront::Shared && std::is_same_v<T, float> && max_L <= PRELOAD_THRESHOLD);

  // Shared memory layout:
  //   Preload mode:  [0..max_L) row_buf, [max_L..2*max_L) col_buf,
  //                  [2*max_L..5*max_L) 3 anti-diagonal buffers
  //   Non-preload:   [0..3*max_L) 3 anti-diagonal buffers
  extern __shared__ char smem_raw[];
  T *smem = global_buffers ? scratch + std::int64_t{ blockIdx.x } * 3 * max_L
                           : reinterpret_cast<T *>(smem_raw);

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
  const bool use_double_buf = !global_buffers
      && (max_L > DOUBLE_BUF_THRESHOLD) && (max_L <= DOUBLE_BUF_MAX) && !preload;

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
    decode_pair(first_pair + pid, N_series, si, sj);
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
      if (tid == 0)
        out[packed_slot(si, sj, N_series) - first_slot] = normalize_public_distance(INF);
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
      out[packed_slot(si, sj, N_series) - first_slot] = normalize_public_distance(dist);
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
    double *__restrict__ out,
    int N_series, int max_L, int num_pairs, bool use_squared_l2, int band,
    std::int64_t first_pair, std::int64_t first_slot)
{
  const int warp_id = threadIdx.x / 32;       // which warp within block [0..7]
  const int lane    = threadIdx.x % 32;       // lane within warp [0..31]
  const int work_idx = blockIdx.x * PAIRS_PER_BLOCK + warp_id;  // pair within the launch

  const T INF = (sizeof(T) == 4)
      ? static_cast<T>(3.402823466e+38f)
      : static_cast<T>(1.7976931348623157e+308);

  if (work_idx >= num_pairs) return;

  std::int64_t si, sj;
  decode_pair(first_pair + work_idx, N_series, si, sj);
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
    if (lane == 0)
      out[packed_slot(si, sj, N_series) - first_slot] = normalize_public_distance(INF);
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
  if (lane == M - 1)
    out[packed_slot(si, sj, N_series) - first_slot] = normalize_public_distance(prev_val);
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
    double *__restrict__ out,
    int N_series, int max_L, int num_pairs, bool use_squared_l2, int band,
    std::int64_t first_pair, std::int64_t first_slot)
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
  decode_pair(first_pair + work_idx, N_series, si, sj);
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
    if (lane == 0)
      out[packed_slot(si, sj, N_series) - first_slot] = normalize_public_distance(INF_VAL);
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

  if (lane == 0)
    out[packed_slot(si, sj, N_series) - first_slot] = normalize_public_distance(final_result);
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
  size_t out_capacity = 0;
  size_t counter_capacity = 0;
  size_t scratch_capacity = 0;
  CachedHostBuffer<T> host_series;
  CudaPtr<T> d_series;
  CudaPtr<int> d_lengths;
  CudaPtr<double> d_out; ///< one launch's packed slots
  CudaPtr<int> d_counter;
  CudaPtr<T> d_scratch; ///< the global wavefront's anti-diagonals, a slice per block
  CudaStream stream;
  CudaEvent evt_start;
  CudaEvent evt_end;

  void ensure_runtime(int new_device_id)
  {
    if (device_id != new_device_id) {
      d_series.reset();
      d_lengths.reset();
      d_out.reset();
      d_counter.reset();
      d_scratch.reset();
      stream.reset();
      evt_start.reset();
      evt_end.reset();
      series_capacity = 0;
      length_capacity = 0;
      out_capacity = 0;
      counter_capacity = 0;
      scratch_capacity = 0;
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
    size_t out_elems,
    size_t scratch_elems)
{
  if (workspace.series_capacity < series_elems) {
    workspace.d_series = cuda_alloc<T>(series_elems);
    workspace.series_capacity = series_elems;
  }
  if (workspace.length_capacity < length_elems) {
    workspace.d_lengths = cuda_alloc<int>(length_elems);
    workspace.length_capacity = length_elems;
  }
  if (workspace.out_capacity < out_elems) {
    workspace.d_out = cuda_alloc<double>(out_elems);
    workspace.out_capacity = out_elems;
  }
  if (workspace.scratch_capacity < scratch_elems) {
    workspace.d_scratch = cuda_alloc<T>(scratch_elems);
    workspace.scratch_capacity = scratch_elems;
  }
}

/// The series in upload order, the last first (see packed_slot): each padded
/// to max_L in @p dst, and its length in @p lengths.
template <typename T>
void flatten_series_buffer(
    T *dst, std::vector<int> &lengths,
    const std::vector<std::vector<double>> &series,
    size_t max_L)
{
  const size_t N = series.size();
  lengths.resize(N);
  for (size_t d = 0; d < N; ++d) {
    const auto &src = series[N - 1 - d];
    T *row_dst = dst + d * max_L;
    const size_t len = src.size();
    lengths[d] = static_cast<int>(len);

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

/// Fill @p out with the distance of every pair, computed in T. The pairs run
/// in launches of at most kMaxPairsPerLaunch consecutive pairs, whose first
/// pair is an int64 offset; a launch writes its span of packed slots on the
/// device, and the span is copied as it is into out.raw(), on the heap or
/// mapped, so the device holds one launch's output whatever N is. No launch
/// writes a diagonal slot: the host zeroes the diagonal.
///
/// Pair indices are decoded on the device (decode_pair), so no pair list is
/// built or uploaded. GPU timing is measured with CUDA events around the
/// uploads, the launches and the copies.
template <typename T>
void launch_dtw_kernel(
    const std::vector<std::vector<double>> &series,
    size_t N, size_t max_L,
    bool use_squared_l2, int band, int device_id, double &gpu_time_sec,
    detail::KernelPath kernel_path, core::DistanceMatrix &out)
{
  const auto n = static_cast<std::int64_t>(N);
  const std::int64_t num_pairs = n * (n - 1) / 2;
  const std::int64_t chunk = std::min(num_pairs, detail::kMaxPairsPerLaunch);

  // The wavefront loops persistent blocks over a launch that has many more
  // pairs than fit on the device at once, one block per pair otherwise. A
  // shared-memory block holds wavefront_buffer_count's diagonal buffers, which
  // select_kernel sized so that three fit an SM above L = 2048, within the
  // opt-in maximum that device_limits opened once;
  // up to kPreloadMaxLength its kernel is the preload mode compiled alone: 40
  // registers instead of 68 (FP32), so up to 6 blocks per SM instead of 3, and
  // 11.5-18.5 % less time at L 257-512 in FP32 and FP64
  // (.claude/baselines/2026-09-30-c1-cuda-long-series.md). The global-memory
  // wavefront always runs persistent, each block in its own slice of scratch,
  // on at most one block per pair of a launch.
  const bool global = kernel_path == detail::KernelPath::WavefrontGlobal;
  const bool wavefront = global || kernel_path == detail::KernelPath::Wavefront;
  constexpr int block_size = 256;
  const size_t wavefront_shared_mem =
      global ? 0 : detail::wavefront_buffer_count(max_L) * max_L * sizeof(T);
  const auto shared_kernel = max_L <= detail::kPreloadMaxLength
      ? dtw_wavefront_kernel<T, Wavefront::Preload>
      : dtw_wavefront_kernel<T, Wavefront::Shared>;
  const auto &device = device_limits(device_id);
  int persistent_grid = 0;
  if (wavefront) {
    int blocks_per_sm = 0;
    if (global)
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm, dtw_wavefront_kernel<T, Wavefront::Global>, block_size, 0));
    else
      CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &blocks_per_sm, shared_kernel, block_size, wavefront_shared_mem));
    persistent_grid = device.sm_count * std::max(blocks_per_sm, 1);
  }
  // A global block's slice holds three anti-diagonals of max_L values. The grid
  // holds no more slices than fit the L2 cache, and a block per SM at least:
  // past the L2 their traffic goes to DRAM, which halved the FP32 fill at
  // L = 20,000 on an RTX 4000 Ada (.claude/baselines/2026-09-30-c1-cuda-long-series.md).
  const auto l2_slices = static_cast<std::int64_t>(device.l2_bytes / (3 * max_L * sizeof(T)));
  const int global_grid =
      global ? static_cast<int>(std::min(std::min<std::int64_t>(chunk, persistent_grid),
                                         std::max<std::int64_t>(device.sm_count, l2_slices)))
             : 0;

  // Every device buffer the launches use is allocated before the caller's
  // matrix, so a device that cannot hold them refuses the fill first. A
  // launch's span holds its pairs and one diagonal slot per row it crosses.
  auto &workspace = get_dtw_launch_workspace<T>(device_id);
  ensure_dtw_device_capacity(workspace, N * max_L, N, static_cast<size_t>(chunk) + N,
                             static_cast<size_t>(global_grid) * 3 * max_L);
  if (wavefront && workspace.counter_capacity < 1) {
    workspace.d_counter = cuda_alloc<int>(1);
    workspace.counter_capacity = 1;
  }

  // The flattened series go through pinned memory where the budget allows, so
  // the upload is a true asynchronous copy; pageable memory otherwise.
  constexpr size_t PINNED_THRESHOLD = 256 * 1024;
  const size_t series_bytes = N * max_L * sizeof(T);
  T *h_flat_series = workspace.host_series.ensure(
      N * max_L, series_bytes >= PINNED_THRESHOLD);
  std::vector<int> lengths;
  flatten_series_buffer(h_flat_series, lengths, series, max_L);

  // Every refusal is behind us and every buffer is allocated: size the
  // caller's matrix. One already N x N, as a mapped one is, keeps its storage.
  if (out.size() != N) out.resize(N);

  const int N_series = static_cast<int>(N);
  auto stream = workspace.stream.get();

  CUDA_CHECK(cudaEventRecord(workspace.evt_start.get(), stream));
  CUDA_CHECK(cudaMemcpyAsync(workspace.d_series.get(), h_flat_series,
                              series_bytes, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(workspace.d_lengths.get(), lengths.data(),
                              N * sizeof(int), cudaMemcpyHostToDevice, stream));

  // The warp-family kernels run one pair per warp, PAIRS_PER_BLOCK warps per
  // block; each warp stages its pair's two series, `staged` samples apiece, in
  // shared memory.
  const auto launch_warp_family = [&](auto kernel, size_t staged, std::int64_t first,
                                      int count, std::int64_t first_slot) {
    const int grid_size = static_cast<int>(
        (std::int64_t{ count } + PAIRS_PER_BLOCK - 1) / PAIRS_PER_BLOCK);
    const size_t shared_mem = PAIRS_PER_BLOCK * 2 * staged * sizeof(T);
    kernel<<<grid_size, PAIRS_PER_BLOCK * 32, shared_mem, stream>>>(
        workspace.d_series.get(), workspace.d_lengths.get(), workspace.d_out.get(),
        N_series, static_cast<int>(max_L), count, use_squared_l2, band, first, first_slot);
  };

  // The launches take the matrix from the top down, each the range of slots
  // [first_slot, end): its pairs' slots and the diagonal slots among and above
  // them; the last range starts at slot 0. The ranges tile the matrix and the
  // copies are the only writes to it. The device buffer is zeroed first, so
  // every slot a copy writes is a distance or a diagonal 0: a fill that stops
  // between launches leaves copied ranges that are right and the rest untouched.
  std::int64_t end = n * (n + 1) / 2;
  for (std::int64_t first = 0; first < num_pairs; first += chunk) {
    const int count = static_cast<int>(std::min(chunk, num_pairs - first));
    std::int64_t first_slot = 0;
    if (first + count < num_pairs) {
      std::int64_t si = 0, sj = 0;
      decode_pair(first + count - 1, n, si, sj);
      first_slot = packed_slot(si, sj, n);
    }
    const auto span = static_cast<size_t>(end - first_slot);
    CUDA_CHECK(cudaMemsetAsync(workspace.d_out.get(), 0, span * sizeof(double), stream));

    if (kernel_path == detail::KernelPath::Warp) {
      launch_warp_family(dtw_warp_kernel<T>, 32, first, count, first_slot);
    } else if (kernel_path == detail::KernelPath::RegTileW4) {
      launch_warp_family(dtw_regtile_kernel<T, 4>, max_L, first, count, first_slot); // 32 lanes x 4 = 128 columns
    } else if (kernel_path == detail::KernelPath::RegTileW8) {
      launch_warp_family(dtw_regtile_kernel<T, 8>, max_L, first, count, first_slot); // 32 lanes x 8 = 256 columns
    } else if (kernel_path == detail::KernelPath::Wavefront) {
      const bool persistent = count > persistent_grid * 4;
      if (persistent)
        CUDA_CHECK(cudaMemsetAsync(workspace.d_counter.get(), 0, sizeof(int), stream));
      shared_kernel<<<persistent ? persistent_grid : count, block_size, wavefront_shared_mem,
                      stream>>>(
          workspace.d_series.get(), workspace.d_lengths.get(), workspace.d_out.get(),
          N_series, static_cast<int>(max_L), count, use_squared_l2, band,
          persistent ? workspace.d_counter.get() : nullptr, first, first_slot, nullptr);
    } else if (global) {
      CUDA_CHECK(cudaMemsetAsync(workspace.d_counter.get(), 0, sizeof(int), stream));
      dtw_wavefront_kernel<T, Wavefront::Global><<<global_grid, block_size, 0, stream>>>(
          workspace.d_series.get(), workspace.d_lengths.get(), workspace.d_out.get(),
          N_series, static_cast<int>(max_L), count, use_squared_l2, band,
          workspace.d_counter.get(), first, first_slot, workspace.d_scratch.get());
    } else {
      throw std::logic_error("launch_dtw_kernel: unknown KernelPath");
    }
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpyAsync(out.raw() + first_slot, workspace.d_out.get(),
                                span * sizeof(double), cudaMemcpyDeviceToHost, stream));
    end = first_slot;
  }

  CUDA_CHECK(cudaEventRecord(workspace.evt_end.get(), stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));

  float elapsed_ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms,
                                  workspace.evt_start.get(),
                                  workspace.evt_end.get()));
  gpu_time_sec = static_cast<double>(elapsed_ms) / 1000.0;
}

} // anonymous namespace

CUDADistMatResult compute_distance_matrix_cuda(
    const std::vector<std::vector<double>> &series,
    const CUDADistMatOptions &opts, core::DistanceMatrix &out)
{
  const size_t N = series.size();
  // Before `out` is touched: a missing device must not answer with a zero matrix.
  detail::require_cuda_device(cuda_available(), "compute_distance_matrix_cuda");

  CUDADistMatResult result;
  result.kernel_used = "none";
  result.n = N;
  if (N <= 1) {
    if (out.size() != N) out.resize(N);
    if (N == 1) out.set(0, 0, 0.0);
    return result;
  }

  CUDA_CHECK(cudaSetDevice(opts.device_id));

  size_t max_L = 0;
  for (const auto &s : series) max_L = std::max(max_L, s.size());

  // An all-zero matrix would read as N identical series.
  if (max_L == 0)
    throw dtwc::InvalidInput("compute_distance_matrix_cuda: every series is "
                             "empty, so there is no distance to compute.");

  const bool use_fp32 = resolve_fp32(opts.precision, opts.device_id);
  const auto &device = device_limits(opts.device_id);
  const auto kernel_path = detail::select_kernel(
      max_L, use_fp32 ? sizeof(float) : sizeof(double), device.shared_per_sm,
      device.wavefront_static_bytes[use_fp32 ? 0 : 1] + device.reserved_shared_per_block);
  result.kernel_used = std::string(detail::kernel_path_name(kernel_path));
  result.pairs_computed = detail::upper_triangle_pairs(N);

  if (use_fp32)
    launch_dtw_kernel<float>(series, N, max_L, opts.use_squared_l2, opts.band,
                             opts.device_id, result.gpu_time_sec, kernel_path, out);
  else
    launch_dtw_kernel<double>(series, N, max_L, opts.use_squared_l2, opts.band,
                              opts.device_id, result.gpu_time_sec, kernel_path, out);

  if (opts.verbose) {
    std::cout << "CUDA DTW: " << result.pairs_computed << " pairs"
              << (use_fp32 ? " [FP32]" : " [FP64]")
              << " in " << result.gpu_time_sec * 1000 << "ms"
              << " on " << cuda_device_info(opts.device_id) << std::endl;
  }

  return result;
}

}  // namespace dtwc::cuda

#endif  // DTWC_HAS_CUDA
