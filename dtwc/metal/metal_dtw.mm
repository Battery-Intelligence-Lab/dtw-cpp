/**
 * @file metal_dtw.mm
 * @brief Metal GPU implementation of batch DTW distance computation.
 *
 * @details Five DTW kernel variants share the same pairwise-distance API:
 *            - dtw_wavefront             anti-diagonal, threadgroup memory
 *            - dtw_wavefront_global      anti-diagonal, device memory (long series)
 *            - dtw_banded_row            row-major, one thread per pair (tight band)
 *            - dtw_regtile_w4 / _w8      register-tile, SIMD-group per pair
 *
 *          Algorithmic lineage:
 *            - Register-tile + warp-shuffle cost propagation: Schmidt & Hundt
 *              (2020), "cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-
 *              Enabled GPUs", Euro-Par 2020, LNCS 12247 pp. 597-612
 *              [https://doi.org/10.1007/978-3-030-57675-2_37]. Reference
 *              implementation ported via dtwc/cuda/cuda_dtw.cu.
 *            - Sakoe-Chiba band constraint: Sakoe & Chiba (1978),
 *              IEEE TASSP 26(1) pp. 43-49.
 *
 * @author Volkan Kumtepeli
 * @date 2026-04-12
 */

#import "metal_dtw.hpp"

#ifdef DTWC_HAS_METAL

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>
#include <type_traits>

#include "../detail/decode_pair.hpp"

namespace dtwc::metal {

// ---------------------------------------------------------------------------
// MSL kernel source (compiled at runtime via newLibraryWithSource:)
// ---------------------------------------------------------------------------
// Embedding as a raw string keeps the build simple — no xcrun metal step,
// no metallib artifact to locate at runtime. Apple's runtime shader
// compiler caches compiled libraries internally, so the one-time cost is
// amortized across all kernel dispatches in a process.
// This MSL source holds only the kernels. The metal_stdlib include, the
// `using namespace metal;`, and the SSOT decode_pair() are prepended at
// library-compile time from dtwc::detail::kDecodePairMSL (see context()), so
// the decode lives in one place across host, CUDA, and Metal.
static NSString *const kDTWMetalKernelSource = @R"METAL(
// The slot of pair (i, j), i < j, in the packed lower triangle the kernels
// write: DistanceMatrix's layout, row j holding columns 0..j.
static inline long packed_slot(long i, long j)
{
  return j * (j + 1) / 2 + i;
}

// Anti-diagonal wavefront DTW, one threadgroup per pair, in two kernels that
// differ only in where the three rotating diagonals live: dtw_wavefront keeps
// them in threadgroup memory, dtw_wavefront_global in device memory for series
// whose 3*max_L*sizeof(float) exceeds the threadgroup-memory cap (32 KB on
// M1/M2/M3 -> max_L ~2730). Both run this body: Ptr is the diagonals' pointer
// type and Scope the memory its barriers order.
//
// Buffers:
//   0: all_series  — N_series * max_L FP32, padded; row s starts at s*max_L
//   1: lengths     — N_series int32
//   2: out_matrix  — N_series * (N_series + 1) / 2 FP32, the packed lower triangle
//   3: N_series    — int32
//   4: max_L       — int32 (row pitch of all_series)
//   5: band        — int32 (Sakoe-Chiba band width; -1 = unbounded)
//   6: use_sq_l2   — int32 (0 = |a-b|, 1 = (a-b)^2)
//   7: scratch     — dtw_wavefront_global: 3 * max_L FP32 per threadgroup
//   8: pair_offset — int64 (base for chunked dispatch)
// Threadgroup memory:
//   0: smem        — dtw_wavefront: 3 * max_L float
template <mem_flags Scope, typename Ptr>
static inline void dtw_wavefront_body(
    device const float *all_series, device const int *lengths,
    device float *out_matrix, int N_series, int max_L, int band, int use_sq_l2,
    long pair_offset, Ptr diagonals, uint tid, uint pid, uint ntids)
{
  // 64-bit num_pairs/index math: int32 overflowed N*(N-1) at N >= 46341.
  const long num_pairs = (long)N_series * (N_series - 1) / 2;
  const long pair_idx  = (long)pid + pair_offset;
  if (pair_idx >= num_pairs) return;

  long a_idx, b_idx;
  decode_pair(pair_idx, N_series, a_idx, b_idx);

  const int La = lengths[a_idx];
  const int Lb = lengths[b_idx];
  const float INF = 3.402823466e+38f; // FLT_MAX
  // No warping path: an empty series, or a length gap the band cannot bridge (as the CUDA kernels).
  if (La <= 0 || Lb <= 0 || (band >= 0 && (La - Lb > band || Lb - La > band))) {
    if (tid == 0) out_matrix[packed_slot(a_idx, b_idx)] = INF;
    return;
  }
  const device float *a = all_series + a_idx * max_L;
  const device float *b = all_series + b_idx * max_L;

  const int K = La + Lb - 1;          // number of anti-diagonals

  // Three rotating buffers, each of length max_L:
  //   d[k % 3][i]  -> DTW cost on anti-diagonal k, row i
  Ptr d0 = diagonals + 0 * max_L;
  Ptr d1 = diagonals + 1 * max_L;
  Ptr d2 = diagonals + 2 * max_L;

  // Initialize the three buffers so boundary reads are INF before any
  // anti-diagonal has been written.
  for (int i = (int)tid; i < max_L; i += (int)ntids) {
    d0[i] = INF;
    d1[i] = INF;
    d2[i] = INF;
  }
  threadgroup_barrier(Scope);

  for (int k = 0; k < K; ++k) {
    // Rotate: current = d[k%3], prev = d[(k-1)%3], prev2 = d[(k-2)%3].
    Ptr cur   = (k % 3 == 0) ? d0 : ((k % 3 == 1) ? d1 : d2);
    Ptr prev  = (k % 3 == 0) ? d2 : ((k % 3 == 1) ? d0 : d1);
    Ptr prev2 = (k % 3 == 0) ? d1 : ((k % 3 == 1) ? d2 : d0);

    int i_lo = max(0, k - Lb + 1);
    int i_hi = min(La - 1, k);

    // Band clip: |i - j| = |2i - k| <= band  ->  i in [(k-band+1)/2, (k+band)/2].
    // Widen before k +/- band so an INT_MAX full-coverage request cannot wrap.
    if (band >= 0) {
      const long band_lo = ((long)k - (long)band + 1L) / 2L;
      const long band_hi = ((long)k + (long)band) / 2L;
      if (band_lo > (long)i_lo) i_lo = (int)band_lo;
      if (band_hi < (long)i_hi) i_hi = (int)band_hi;
    }
    const int diag_len = i_hi - i_lo + 1;

    if (diag_len > 0) {
      for (int idx = (int)tid; idx < diag_len; idx += (int)ntids) {
        const int i = i_lo + idx;
        const int j = k - i;

        float diff = a[i] - b[j];
        float cost = use_sq_l2 ? (diff * diff) : fabs(diff);

        float best;
        if (i == 0 && j == 0) {
          best = 0.0f; // origin
        } else {
          float cdiag = (i > 0 && j > 0) ? prev2[i - 1] : INF;
          float cup   = (i > 0)          ? prev[i - 1]  : INF;
          float cleft = (j > 0)          ? prev[i]      : INF;
          best = min(cdiag, min(cup, cleft));
        }

        cur[i] = cost + best;
      }
    }

    // INF-fill the band-adjacent positions so the next diagonal's in-band
    // reads of out-of-band cells see INF instead of stale data from 3
    // diagonals ago (3-buffer rotation). Both edges need stamping because
    // the band can shift by 1 either way per diagonal.
    if ((int)tid == 0 && band >= 0) {
      if (i_lo - 1 >= 0 && i_lo - 1 < max_L) cur[i_lo - 1] = INF;
      if (i_hi + 1 >= 0 && i_hi + 1 < max_L) cur[i_hi + 1] = INF;
    }

    threadgroup_barrier(Scope);
  }

  // Final answer: cell (La-1, Lb-1) lives on the last anti-diagonal,
  // written to d[(K-1) % 3][La-1]. Thread 0 stores the result.
  if (tid == 0) {
    const int last = K - 1;
    Ptr cur = (last % 3 == 0) ? d0 : ((last % 3 == 1) ? d1 : d2);
    out_matrix[packed_slot(a_idx, b_idx)] = cur[La - 1];
  }
}

kernel void dtw_wavefront(
    device const float*   all_series [[buffer(0)]],
    device const int*     lengths    [[buffer(1)]],
    device float*         out_matrix [[buffer(2)]],
    constant int&         N_series   [[buffer(3)]],
    constant int&         max_L      [[buffer(4)]],
    constant int&         band       [[buffer(5)]],
    constant int&         use_sq_l2  [[buffer(6)]],
    constant long&        pair_offset [[buffer(8)]],
    threadgroup float*    smem       [[threadgroup(0)]],
    uint tid   [[thread_position_in_threadgroup]],
    uint pid   [[threadgroup_position_in_grid]],
    uint ntids [[threads_per_threadgroup]])
{
  dtw_wavefront_body<mem_flags::mem_threadgroup>(
      all_series, lengths, out_matrix, N_series, max_L, band, use_sq_l2,
      pair_offset, smem, tid, pid, ntids);
}

kernel void dtw_wavefront_global(
    device const float*   all_series [[buffer(0)]],
    device const int*     lengths    [[buffer(1)]],
    device float*         out_matrix [[buffer(2)]],
    constant int&         N_series   [[buffer(3)]],
    constant int&         max_L      [[buffer(4)]],
    constant int&         band       [[buffer(5)]],
    constant int&         use_sq_l2  [[buffer(6)]],
    device float*         scratch    [[buffer(7)]],
    constant long&        pair_offset [[buffer(8)]],
    uint tid   [[thread_position_in_threadgroup]],
    uint pid   [[threadgroup_position_in_grid]],
    uint ntids [[threads_per_threadgroup]])
{
  // The slice is indexed by threadgroup position, not by pair, so every chunk
  // of a chunked dispatch reuses the same scratch.
  dtw_wavefront_body<mem_flags::mem_device>(
      all_series, lengths, out_matrix, N_series, max_L, band, use_sq_l2,
      pair_offset, scratch + (size_t)pid * 3 * max_L, tid, pid, ntids);
}

// Row-major banded DTW: one thread per pair, no intra-threadgroup barriers.
//
// For tight Sakoe-Chiba bands the anti-diagonal wavefront kernel is barrier-
// bound (2*La threadgroup_barriers dominate compute). This kernel trades
// within-pair parallelism for across-pair parallelism: each thread iterates
// row-by-row with two rolling buffers of length (2*band+1) in device memory.
//
// Relative-index scheme: cell (i, j) with |i - j| <= band is stored at
//   r = j - i + band   in [0, 2*band].
// Dependencies at row i, col j:
//   D[i-1][j-1]  -> prev[r]     (relative index j-1-(i-1)+band = r)
//   D[i-1][j]    -> prev[r+1]   (relative index j-(i-1)+band   = r+1)
//   D[i][j-1]    -> cur[r-1]    (relative index j-1-i+band     = r-1)
// Out-of-band cells stay INF (explicit fill each row after the j loop), so
// reads of prev[r+1] at the band edge naturally return INF.
//
// Memory layout is COALESCED across threads in a SIMD group:
//   scratch[ row_half * W*stride + r * stride + gid ]
// where stride = total threads dispatched. Threads in one SIMD group reading
// `prev[r]` all hit one 128-byte cache line, avoiding the 32x bandwidth
// penalty of a naive per-thread-strip layout.
kernel void dtw_banded_row(
    device const float*   all_series  [[buffer(0)]],
    device const int*     lengths     [[buffer(1)]],
    device float*         out_matrix  [[buffer(2)]],
    constant int&         N_series    [[buffer(3)]],
    constant int&         max_L       [[buffer(4)]],
    constant int&         band        [[buffer(5)]],
    constant int&         use_sq_l2   [[buffer(6)]],
    device float*         scratch     [[buffer(7)]],
    constant long&        pair_offset [[buffer(8)]],
    constant int&         stride      [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
  // 64-bit num_pairs/index math.
  const long num_pairs = (long)N_series * (N_series - 1) / 2;
  const long pair_idx = (long)gid + pair_offset;
  if (pair_idx >= num_pairs) return;

  long a_idx, b_idx;
  decode_pair(pair_idx, N_series, a_idx, b_idx);

  const int La = lengths[a_idx];
  const int Lb = lengths[b_idx];
  const float INF = 3.402823466e+38f;
  // No warping path: an empty series, or a length gap the band (> 0 on this route) cannot bridge.
  if (La <= 0 || Lb <= 0 || La - Lb > band || Lb - La > band) {
    out_matrix[packed_slot(a_idx, b_idx)] = INF;
    return;
  }
  const device float *a = all_series + a_idx * max_L;
  const device float *b = all_series + b_idx * max_L;

  const int W = 2 * band + 1;                // strip width

  // Interleaved coalesced layout: this thread's prev[r] is scratch[0 + r*stride + gid];
  // cur[r] is scratch[W*stride + r*stride + gid]. Pointer arithmetic:
  device float *prev = scratch + gid;
  device float *cur  = scratch + (size_t)W * stride + gid;

  // Initial prev row: nothing computed yet, all INF.
  for (int r = 0; r < W; ++r) prev[r * stride] = INF;

  // Track previous row's band bounds so we can INF-fill only the delta
  // between adjacent rows' bands (2 cells maximum) instead of all W cells.
  int prev_r_lo = 0;
  int prev_r_hi = -1; // sentinel: row i=-1 had no in-band cells

  for (int i = 0; i < La; ++i) {
    const int j_lo = max(0, i - band);
    const int j_hi = min(Lb - 1, i + band);
    const int r_lo = j_lo - i + band;        // 0..band
    const int r_hi = j_hi - i + band;        // band..W-1

    // INF-fill only cells in cur that WON'T be written by the j-loop but
    // WILL be read by row i+1. Cells [prev_r_lo..prev_r_hi] of prev were
    // valid; after the upcoming swap, cur becomes prev. So for row i+1,
    // any read at r in [r_lo_next..r_hi_next] (for cdiag/cup) touches
    // positions [r_lo_next..r_hi_next+1] of prev(=cur-after-swap). We
    // simply INF-fill cells outside [r_lo, r_hi] of cur that were set by
    // some earlier iteration (up to prev_r_lo..prev_r_hi).
    for (int r = prev_r_lo; r < r_lo; ++r) cur[r * stride] = INF;
    for (int r = r_hi + 1; r <= prev_r_hi; ++r) cur[r * stride] = INF;

    // Register-rotated read of prev: hold prev[r*stride] and prev[(r+1)*stride]
    // in registers, shift one position each iteration. One device load per cell.
    float prev_r, prev_rp1;
    if (i == 0) {
      prev_r   = INF;
      prev_rp1 = INF;
    } else {
      prev_r   = (r_lo     < W) ? prev[r_lo * stride]       : INF;
      prev_rp1 = (r_lo + 1 < W) ? prev[(r_lo + 1) * stride] : INF;
    }
    float last_cur = INF; // D[i][j-1] from previous j iteration

    for (int j = j_lo; j <= j_hi; ++j) {
      const int r = j - i + band;            // 0..W-1

      const float diff = a[i] - b[j];
      const float cost = use_sq_l2 ? (diff * diff) : fabs(diff);

      float best;
      if (i == 0 && j == 0) {
        best = 0.0f;
      } else {
        const float cdiag = (i > 0 && j > 0)         ? prev_r   : INF;
        const float cup   = (i > 0 && r + 1 < W)     ? prev_rp1 : INF;
        const float cleft = (j > 0 && r > 0)         ? last_cur : INF;
        best = min(cdiag, min(cup, cleft));
      }
      last_cur = cost + best;
      cur[r * stride] = last_cur;

      // Shift register window for next j (r+1): prev_r <- prev_rp1;
      // prev_rp1 <- prev[(r+2)*stride] (INF if out of band in prev row).
      prev_r   = prev_rp1;
      prev_rp1 = (r + 2 < W) ? prev[(r + 2) * stride] : INF;
    }

    // Swap prev <-> cur for next row.
    device float *tmp = prev; prev = cur; cur = tmp;
    prev_r_lo = r_lo;
    prev_r_hi = r_hi;
  }

  // After the final iteration we swapped; the completed row is now `prev`.
  // Target cell (La-1, Lb-1) lives at relative index (Lb-1)-(La-1)+band.
  const int r_final = (Lb - 1) - (La - 1) + band;
  out_matrix[packed_slot(a_idx, b_idx)] =
      (r_final >= 0 && r_final < W) ? prev[r_final * stride] : INF;
}

// ---------------------------------------------------------------------------
// Register-tile DTW for short/medium series (max_L <= 256, unbanded).
//
// Adapted from Schmidt & Hundt, "cuDTW++: Ultra-Fast Dynamic Time Warping on
// CUDA-Enabled GPUs" (Euro-Par 2020, LNCS 12247), translated from the CUDA
// reference in dtwc/cuda/cuda_dtw.cu (`dtw_regtile_kernel`).
//
// One SIMD-group (32 threads) per pair, `pairs_per_tg` warps per threadgroup.
// Each thread holds TILE_W columns in registers; 32 threads cover 32*TILE_W
// columns total. Left-neighbor values are fetched via simd_shuffle_up (MSL's
// analogue of CUDA `__shfl_sync(..., lane - 1)` from the cuDTW++ regtile
// kernel).
//
// Wavefront timing: at step s, thread t processes row i = s - t. Thread t-1
// processed row i at step s-1, so at step s its penalty[TILE_W-1] holds
// cost[i][col_start-1] — fetched via simd_shuffle_up(penalty[TILE_W-1], 1).
//
// Unbanded-only for this pass — banded regtile adds register pressure that
// is deferred.

template <int TILE_W>
static float dtw_regtile_compute(
    threadgroup const float *my_row,
    threadgroup const float *my_col,
    int M, int N_len,
    uint simd_lane,
    int use_sq_l2)
{
  const float INF = 3.402823466e+38f;
  const int col_start = (int)simd_lane * TILE_W;

  float col_val[TILE_W];
  #pragma unroll
  for (int tw = 0; tw < TILE_W; ++tw) {
    int j = col_start + tw;
    col_val[tw] = (j < N_len) ? my_col[j] : 0.0f;
  }

  float dp[TILE_W];
  #pragma unroll
  for (int tw = 0; tw < TILE_W; ++tw) {
    dp[tw] = INF;
  }
  float prev_last = INF; // saved dp[TILE_W-1] from previous row

  const int num_col_threads = (N_len + TILE_W - 1) / TILE_W;
  const int total_steps = M + num_col_threads - 1;

  for (int step = 0; step < total_steps; ++step) {
    const int i = step - (int)simd_lane;
    const bool row_valid =
        (i >= 0) && (i < M) && ((int)simd_lane < num_col_threads);

    const float row_val = row_valid ? my_row[i] : 0.0f;

    // Cross-lane communication (all lanes must participate in shuffles).
    float penalty_from_left = simd_shuffle_up(dp[TILE_W - 1], 1);
    float diag_from_left    = simd_shuffle_up(prev_last, 1);
    if (simd_lane == 0) {
      penalty_from_left = INF; // no left neighbor for lane 0
      diag_from_left    = INF;
    }

    if (row_valid) {
      const float saved_prev_last = dp[TILE_W - 1];

      float left = penalty_from_left; // cost[i][col_start - 1]
      float diag = diag_from_left;    // cost[i-1][col_start - 1]

      #pragma unroll
      for (int tw = 0; tw < TILE_W; ++tw) {
        const int j = col_start + tw;
        const float above = dp[tw];
        float new_cost = INF;
        if (j < N_len) {
          const float diff = row_val - col_val[tw];
          const float d    = use_sq_l2 ? (diff * diff) : fabs(diff);
          if (i == 0 && j == 0) {
            new_cost = d;
          } else {
            const float eff_above = (i == 0)           ? INF : above;
            const float eff_diag  = (i == 0 || j == 0) ? INF : diag;
            const float eff_left  = (j == 0)           ? INF : left;
            new_cost = min(eff_diag, min(eff_above, eff_left)) + d;
          }
        }
        diag = above;
        left = new_cost;
        dp[tw] = new_cost;
      }

      prev_last = saved_prev_last;
    }
  }

  // Result is cost[M-1][N_len-1].
  const int result_thread = (N_len - 1) / TILE_W;
  const int result_tw     = (N_len - 1) % TILE_W;
  const float my_result =
      ((int)simd_lane == result_thread) ? dp[result_tw] : INF;
  return simd_shuffle(my_result, (ushort)result_thread);
}

template <int TILE_W>
static void dtw_regtile_kernel_body(
    device const float *all_series,
    device const int   *lengths,
    device float       *out_matrix,
    int  N_series,
    int  max_L,
    int  use_sq_l2,
    long pair_offset,
    int  pairs_per_tg,
    threadgroup float  *smem,
    uint simd_lane,
    uint simd_id,
    uint tg_idx)
{
  // 64-bit num_pairs/index math.
  const long num_pairs = (long)N_series * (N_series - 1) / 2;
  const long pair_idx  = (long)tg_idx * (long)pairs_per_tg + (long)simd_id + pair_offset;
  if (pair_idx >= num_pairs) return;

  // Lane 0 decodes the pair and broadcasts it. MSL's SIMD-group functions take no 64-bit type, so the
  // two series indices cross as int, which holds them because N_series is an int.
  int si0 = 0, sj0 = 0;
  if (simd_lane == 0) {
    long a_idx, b_idx;
    decode_pair(pair_idx, N_series, a_idx, b_idx);
    si0 = (int)a_idx;
    sj0 = (int)b_idx;
  }
  const long si = simd_broadcast_first(si0);
  const long sj = simd_broadcast_first(sj0);
  const int ni = lengths[si];
  const int nj = lengths[sj];
  const device float *x = all_series + si * max_L;
  const device float *y = all_series + sj * max_L;

  // Orient: rows = short side, columns = long side.
  const device float *row_g = (ni <= nj) ? x : y;
  const device float *col_g = (ni <= nj) ? y : x;
  const int M     = min(ni, nj);
  const int N_len = max(ni, nj);

  const float INF = 3.402823466e+38f;

  if (M == 0 || N_len == 0) {
    if (simd_lane == 0) out_matrix[packed_slot(si, sj)] = INF;
    return;
  }

  // Per-warp smem slice: 2 * max_L floats (row + col).
  threadgroup float *my_row = smem + (int)simd_id * 2 * max_L;
  threadgroup float *my_col = my_row + max_L;

  for (int t = (int)simd_lane; t < M; t += 32)
    my_row[t] = row_g[t];
  for (int t = (int)simd_lane; t < N_len; t += 32)
    my_col[t] = col_g[t];
  simdgroup_barrier(mem_flags::mem_threadgroup);

  const float final_result =
      dtw_regtile_compute<TILE_W>(my_row, my_col, M, N_len, simd_lane, use_sq_l2);

  if (simd_lane == 0) out_matrix[packed_slot(si, sj)] = final_result;
}

kernel void dtw_regtile_w4(
    device const float*   all_series   [[buffer(0)]],
    device const int*     lengths      [[buffer(1)]],
    device float*         out_matrix   [[buffer(2)]],
    constant int&         N_series     [[buffer(3)]],
    constant int&         max_L        [[buffer(4)]],
    constant int&         use_sq_l2    [[buffer(6)]],
    constant long&        pair_offset  [[buffer(8)]],
    constant int&         pairs_per_tg [[buffer(9)]],
    threadgroup float*    smem         [[threadgroup(0)]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint tg_idx    [[threadgroup_position_in_grid]])
{
  dtw_regtile_kernel_body<4>(all_series, lengths, out_matrix, N_series, max_L,
                             use_sq_l2, pair_offset, pairs_per_tg, smem,
                             simd_lane, simd_id, tg_idx);
}

kernel void dtw_regtile_w8(
    device const float*   all_series   [[buffer(0)]],
    device const int*     lengths      [[buffer(1)]],
    device float*         out_matrix   [[buffer(2)]],
    constant int&         N_series     [[buffer(3)]],
    constant int&         max_L        [[buffer(4)]],
    constant int&         use_sq_l2    [[buffer(6)]],
    constant long&        pair_offset  [[buffer(8)]],
    constant int&         pairs_per_tg [[buffer(9)]],
    threadgroup float*    smem         [[threadgroup(0)]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint tg_idx    [[threadgroup_position_in_grid]])
{
  dtw_regtile_kernel_body<8>(all_series, lengths, out_matrix, N_series, max_L,
                             use_sq_l2, pair_offset, pairs_per_tg, smem,
                             simd_lane, simd_id, tg_idx);
}

)METAL";

// ---------------------------------------------------------------------------
// Lazy-initialized Metal context (device, queue, pipeline).
// ---------------------------------------------------------------------------
struct MetalContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pipeline = nil;            // threadgroup-memory wavefront
  id<MTLComputePipelineState> pipeline_global = nil;     // device-memory wavefront
  id<MTLComputePipelineState> pipeline_banded_row = nil; // row-major banded (tight-band)
  id<MTLComputePipelineState> pipeline_regtile_w4 = nil; // register-tile, max_L <= 128
  id<MTLComputePipelineState> pipeline_regtile_w8 = nil; // register-tile, max_L <= 256
  bool initialized = false;
  bool init_failed = false;
  std::string init_error;
};

static MetalContext &context()
{
  static MetalContext ctx;
  static std::once_flag once;
  std::call_once(once, []() {
    @autoreleasepool {
      ctx.device = MTLCreateSystemDefaultDevice();
      if (!ctx.device) {
        ctx.init_failed = true;
        ctx.init_error = "MTLCreateSystemDefaultDevice returned nil";
        return;
      }
      [ctx.device retain]; // we hold it for the process lifetime

      ctx.queue = [ctx.device newCommandQueue];
      if (!ctx.queue) {
        ctx.init_failed = true;
        ctx.init_error = "newCommandQueue failed";
        return;
      }

      NSError *err = nil;
      // Prepend the SSOT decode (metal_stdlib include + using + decode_pair)
      // so the kernels below can call it. stringByAppendingString avoids the
      // '%' format pitfalls of stringWithFormat (kernels contain `k % 3`).
      NSString *metalSource =
          [[NSString stringWithUTF8String:dtwc::detail::kDecodePairMSL]
              stringByAppendingString:kDTWMetalKernelSource];
      id<MTLLibrary> lib = [ctx.device newLibraryWithSource:metalSource
                                                   options:nil
                                                     error:&err];
      if (!lib) {
        ctx.init_failed = true;
        ctx.init_error = err ? [[err localizedDescription] UTF8String]
                             : "newLibraryWithSource failed";
        return;
      }

      // Every pipeline is built by this one helper, so each failure names
      // its kernel function.
      auto make_pipeline = [&](NSString *name,
                               id<MTLComputePipelineState> &out) -> bool {
        id<MTLFunction> fn = [lib newFunctionWithName:name];
        if (!fn) {
          ctx.init_failed = true;
          ctx.init_error = std::string("kernel function ") +
                           [name UTF8String] + " not found";
          return false;
        }
        out = [ctx.device newComputePipelineStateWithFunction:fn error:&err];
        [fn release];
        if (!out) {
          ctx.init_failed = true;
          ctx.init_error = std::string("kernel function ") + [name UTF8String] + ": "
                           + (err ? [[err localizedDescription] UTF8String]
                                  : "newComputePipelineStateWithFunction failed");
          return false;
        }
        return true;
      };

      // In order, stopping at the first failure; the library is released
      // either way.
      const bool built =
          make_pipeline(@"dtw_wavefront", ctx.pipeline)
          && make_pipeline(@"dtw_wavefront_global", ctx.pipeline_global)
          && make_pipeline(@"dtw_banded_row", ctx.pipeline_banded_row)
          && make_pipeline(@"dtw_regtile_w4", ctx.pipeline_regtile_w4)
          && make_pipeline(@"dtw_regtile_w8", ctx.pipeline_regtile_w8);
      [lib release];
      if (!built) return;

      ctx.initialized = true;
    }
  });
  return ctx;
}

bool metal_available()
{
  auto &ctx = context();
  return ctx.initialized;
}

std::string metal_device_info()
{
  auto &ctx = context();
  if (!ctx.initialized) {
    return std::string("Metal unavailable: ") +
           (ctx.init_error.empty() ? "unknown" : ctx.init_error);
  }
  @autoreleasepool {
    NSString *name = [ctx.device name];
    uint64_t reg = ctx.device.registryID;
    uint64_t mem = ctx.device.recommendedMaxWorkingSetSize; // bytes
    char buf[256];
    std::snprintf(buf, sizeof(buf),
        "%s (registryID=0x%llx, max_working_set=%.2f GB)",
        name ? [name UTF8String] : "unknown-gpu",
        (unsigned long long)reg,
        (double)mem / (1024.0 * 1024.0 * 1024.0));
    return std::string(buf);
  }
}

// ---------------------------------------------------------------------------
// Main entry: pairwise distance matrix.
// ---------------------------------------------------------------------------
namespace {

// The Objective-C objects one compute_distance_matrix_metal call creates. This
// file is built without ARC: the destructor releases each, so every exit, a
// throw included, releases each exactly once (a message to nil does nothing).
// The pool replaces @autoreleasepool, which is not drained when an exception
// leaves it and so would strand the command buffers and encoders the chunk
// loop autoreleases; releasing a pool drains it.
struct CallObjects
{
  NSAutoreleasePool *pool = [[NSAutoreleasePool alloc] init];
  id<MTLBuffer> series = nil;
  id<MTLBuffer> lengths = nil;
  id<MTLBuffer> out = nil;
  id<MTLBuffer> scratch[2] = { nil, nil };

  CallObjects() = default;
  CallObjects(const CallObjects &) = delete;
  CallObjects &operator=(const CallObjects &) = delete;
  ~CallObjects()
  {
    [scratch[1] release];
    [scratch[0] release];
    [out release];
    [lengths release];
    [series release];
    [pool release];
  }
};

} // namespace

MetalDistMatResult compute_distance_matrix_metal(
    const std::vector<std::vector<double>> &series,
    const MetalDistMatOptions &opts, core::DistanceMatrix &out)
{
  return compute_distance_matrix_metal(
      series.size(),
      [&series](std::size_t i) { return std::span<const double>(series[i]); },
      opts,
      [](std::size_t count, const std::function<void(std::size_t)> &task) {
        for (std::size_t i = 0; i < count; ++i) task(i);
      },
      out);
}

MetalDistMatResult compute_distance_matrix_metal(
    std::size_t n_series, const SeriesAt &series,
    const MetalDistMatOptions &opts, const ParallelFor &parallel,
    core::DistanceMatrix &out)
{
  validate_metal_precision(opts.precision);
  auto &ctx = context();
  // As on CUDA: an unavailable backend is a typed DeviceError, for
  // every N and before the result is allocated, never a zero-filled result
  // that reads as a valid answer.
  if (!ctx.initialized)
    throw dtwc::DeviceError("Metal backend unavailable ("
                            + (ctx.init_error.empty()
                                 ? std::string("unknown") : ctx.init_error)
                            + "): no usable Metal GPU was detected. No CPU "
                              "fallback was attempted.");

  MetalDistMatResult result;
  const size_t N = n_series;
  result.n = N;
  if (N <= 1) {
    if (out.size() != N) out.resize(N);
    if (N == 1) out.set(0, 0, 0.0);
    return result;
  }

  // Find max length and build padded FP32 input buffer in parallel.
  int max_L = 0;
  std::vector<int> lengths(N);
  for (size_t s = 0; s < N; ++s) {
    lengths[s] = static_cast<int>(series(s).size());
    if (lengths[s] > max_L) max_L = lengths[s];
  }
  // An all-zero matrix would read as N identical series.
  if (max_L == 0)
    throw dtwc::InvalidInput("compute_distance_matrix_metal: every series is "
                             "empty, so there is no distance to compute.");

  const size_t num_pairs = N * (N - 1) / 2;
  result.pairs_computed = num_pairs;

  {
    CallObjects owned;
    auto t0 = std::chrono::steady_clock::now();

    // Upload series (FP32, padded) via caller's parallel loop.
    const size_t series_bytes = N * max_L * sizeof(float);
    owned.series = [ctx.device
        newBufferWithLength:series_bytes
                    options:MTLResourceStorageModeShared];
    if (!owned.series) throw dtwc::DeviceError("Metal: series buffer allocation failed");
    float *series_ptr = static_cast<float *>([owned.series contents]);
    parallel(N, [&](size_t s) {
      float *dst_row = series_ptr + s * static_cast<size_t>(max_L);
      const auto src = series(s);
      const int len = lengths[s];
      for (int k = 0; k < len; ++k) {
        dst_row[k] = static_cast<float>(src[static_cast<size_t>(k)]);
      }
      if (len < max_L) {
        std::memset(dst_row + len, 0, static_cast<size_t>(max_L - len) * sizeof(float));
      }
    });

    // Upload lengths
    owned.lengths = [ctx.device
        newBufferWithBytes:lengths.data()
                    length:N * sizeof(int)
                   options:MTLResourceStorageModeShared];
    if (!owned.lengths) throw dtwc::DeviceError("Metal: lengths buffer allocation failed");

    // Output: the packed lower triangle in FP32, widened into `out` on the
    // host. Zeroed, so the diagonal slots, which no kernel writes, read 0.
    const size_t slots = core::packed_size(N);
    owned.out = [ctx.device
        newBufferWithLength:slots * sizeof(float)
                    options:MTLResourceStorageModeShared];
    if (!owned.out) throw dtwc::DeviceError("Metal: output buffer allocation failed");
    std::memset([owned.out contents], 0, slots * sizeof(float));

    // Scalar args
    const int N_int = static_cast<int>(N);
    const int requested_band = opts.band;
    const int use_sq_l2 = opts.use_squared_l2 ? 1 : 0;

    // Choose kernel variant based on workload:
    //   banded-row kernel: tight Sakoe-Chiba band (band > 0, band*20 < max_L).
    //                      One thread per pair, row-major iteration, no barriers.
    //                      Wins on tight bands where the wavefront's 2*La
    //                      threadgroup barriers dominate.
    //   regtile kernels:   short/medium unbanded (max_L <= 256). SIMD-shuffle
    //                      cross-lane communication, ~4x fewer threadgroup
    //                      barriers than wavefront on the same work.
    //   threadgroup kernel: 3 * max_L * 4 <= device cap (32KB on M1/M2/M3 -> max_L <= 2730)
    //   global kernel:     uses device memory for the 3 anti-diagonal buffers
    const NSUInteger tg_mem_len = 3 * (NSUInteger)max_L * sizeof(float);
    const NSUInteger tg_mem_cap = ctx.device.maxThreadgroupMemoryLength;
    // Row-major one-thread-per-pair wins only when the band is tight enough
    // that wavefront barrier overhead dominates. Empirically (M2 Max), a
    // ~5%-of-length band is the crossover: at band/L < 1/20 the row-major
    // kernel beats the wavefront; wider bands put too much sequential work
    // on a single thread. Cap at 512 to avoid huge per-thread scratch.
    const bool use_banded_row =
        (requested_band > 0) && (requested_band <= 512)
        && (static_cast<std::int64_t>(requested_band) * 20
            < static_cast<std::int64_t>(max_L));
    // A non-negative band covering the largest possible |i-j| in the batch is
    // exactly unbounded. Normalize before regtile routing so full-coverage
    // bands (e.g. INT_MAX) also benefit from register-tiling, while preserving
    // banded_row's positive strip width.
    const int band =
        (!use_banded_row && requested_band >= 0
         && requested_band >= max_L - 1)
          ? -1 : requested_band;
    // Register-tile path: unbanded (or normalized-unbanded) with SIMD width 32.
    // TILE_W=4 covers max_L in [1, 128], TILE_W=8 covers (128, 256].
    const int regtile_tile_w = (max_L <= 128) ? 4 : 8;
    id<MTLComputePipelineState> regtile_pipe =
        (regtile_tile_w == 4) ? ctx.pipeline_regtile_w4 : ctx.pipeline_regtile_w8;
    const bool use_regtile =
        !use_banded_row && (band == -1)
        && (max_L > 0) && (max_L <= 256)
        && (regtile_pipe.threadExecutionWidth == 32)
        && (regtile_pipe.maxTotalThreadsPerThreadgroup >= 32);
    const bool use_global =
        !use_banded_row && !use_regtile && (tg_mem_len > tg_mem_cap);

    id<MTLComputePipelineState> pipeline;
    if (use_banded_row) {
      pipeline = ctx.pipeline_banded_row;
      result.kernel_used = "banded_row";
    } else if (use_regtile) {
      pipeline = regtile_pipe;
      result.kernel_used = (regtile_tile_w == 4) ? "regtile_w4" : "regtile_w8";
    } else if (use_global) {
      pipeline = ctx.pipeline_global;
      result.kernel_used = "wavefront_global";
    } else {
      pipeline = ctx.pipeline;
      result.kernel_used = "wavefront";
    }

    // Chunk pairs across multiple command buffers. macOS's GPU watchdog
    // kills compute that holds the GPU for more than ~2 s per command buffer
    // (error: kIOGPUCommandBufferCallbackErrorImpactingInteractivity).
    // Rough rule: keep each dispatch bounded in total cell count.
    //
    // Global-memory kernel is slower per-pair than threadgroup, so chunk more
    // aggressively for long series. Banded-row touches only band*La cells
    // per pair, so cells-per-pair accounting uses band*L, not L*L.
    const size_t cells_budget = 5e9; // ~5 billion DTW cells per command buffer
    const size_t cells_per_pair = use_banded_row
        ? (size_t)(2 * band + 1) * (size_t)max_L
        : (size_t)max_L * (size_t)max_L;
    size_t chunk = std::max<size_t>(1, cells_budget / cells_per_pair);
    if (chunk > num_pairs) chunk = num_pairs;

    const size_t num_chunks = (num_pairs + chunk - 1) / chunk;
    const size_t scratch_slots = (num_chunks > 1) ? 2 : 1;
    // Allocate scratch sized for one chunk (double-buffered when multiple chunks exist).
    size_t scratch_bytes = 0;
    if (use_global) {
      scratch_bytes = chunk * 3ULL * (size_t)max_L * sizeof(float);
    } else if (use_banded_row) {
      // Coalesced layout: 2 rolling row-strips of (2*band+1) floats; each
      // row stripe has `stride` floats (stride == total threads dispatched).
      // Stride is computed below once tg_size is known — defer allocation.
      scratch_bytes = 0;
    }
    if (scratch_bytes > 0) {
      owned.scratch[0] = [ctx.device newBufferWithLength:scratch_bytes
                                                 options:MTLResourceStorageModePrivate];
      if (!owned.scratch[0]) {
        throw dtwc::DeviceError(
            "Metal: scratch allocation failed (" + std::to_string(scratch_bytes)
            + " bytes for max_L=" + std::to_string(max_L) + ", chunk="
            + std::to_string(chunk) + "). No CPU fallback was attempted.");
      }
      if (scratch_slots == 2) {
        owned.scratch[1] = [ctx.device newBufferWithLength:scratch_bytes
                                                   options:MTLResourceStorageModePrivate];
      }
    }

    // Pick threads-per-threadgroup.
    //  - banded-row: one thread per pair, so tg_size = simd width (32) for
    //    cheap dispatch without intra-threadgroup cooperation.
    //  - regtile:    up to 8 warps per threadgroup, 1 warp per pair.
    //                Actual pairs_per_tg = tg_size / simd is passed to the MSL
    //                kernel so any hardware maxTotalThreadsPerThreadgroup clamp
    //                never skips pairs.
    //  - wavefront kernels: cooperate within a pair, use up to max_L threads
    //    rounded to a multiple of the simd width.
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    const NSUInteger simd = pipeline.threadExecutionWidth;
    NSUInteger pairs_per_tg = 1;
    NSUInteger tg_size;
    if (use_banded_row) {
      tg_size = simd;
      if (tg_size > max_threads) tg_size = max_threads;
    } else if (use_regtile) {
      const NSUInteger max_pairs_threads = std::max<NSUInteger>(1, max_threads / simd);
      const NSUInteger max_pairs_smem =
          std::max<NSUInteger>(1, tg_mem_cap / (2 * (NSUInteger)max_L * sizeof(float)));
      pairs_per_tg = std::min<NSUInteger>(8, std::min(max_pairs_threads, max_pairs_smem));
      tg_size = simd * pairs_per_tg;
    } else {
      tg_size = std::min((NSUInteger)max_L, max_threads);
      if (tg_size == 0) tg_size = 1;
      if (tg_size > simd) tg_size = (tg_size / simd) * simd;
    }
    const NSUInteger regtile_smem_len =
        use_regtile ? (pairs_per_tg * 2 * (NSUInteger)max_L * sizeof(float)) : 0;

    // Banded-row: allocate coalesced scratch sized by total thread count.
    // Grid is 1D of (ceil(chunk/tg_size) * tg_size) threads.
    int banded_stride = 0;
    if (use_banded_row) {
      const size_t ntgs = (chunk + tg_size - 1) / tg_size;
      banded_stride = static_cast<int>(ntgs * tg_size);
      scratch_bytes =
          2ULL * (size_t)(2 * band + 1) * (size_t)banded_stride * sizeof(float);
      owned.scratch[0] = [ctx.device newBufferWithLength:scratch_bytes
                                                 options:MTLResourceStorageModePrivate];
      if (!owned.scratch[0]) {
        throw dtwc::DeviceError(
            "Metal: banded-row scratch allocation failed ("
            + std::to_string(scratch_bytes) + " bytes). No CPU fallback was attempted.");
      }
      if (scratch_slots == 2) {
        owned.scratch[1] = [ctx.device newBufferWithLength:scratch_bytes
                                                   options:MTLResourceStorageModePrivate];
      }
    }

    // A command buffer that failed, whichever chunk it carried, is a typed
    // error: its pairs would otherwise stay 0 in the output and read as
    // identical series. The buffers are autoreleased; the pool in `owned`
    // keeps them until the call returns.
    auto check = [](id<MTLCommandBuffer> cmd) {
      if (cmd.error) {
        NSString *desc = [cmd.error localizedDescription];
        throw dtwc::DeviceError(std::string("Metal kernel failed: ") +
                                (desc ? [desc UTF8String] : "unknown"));
      }
    };
    const bool dual_scratch = owned.scratch[1] != nil;
    std::vector<id<MTLCommandBuffer>> in_flight;
    size_t chunk_idx = 0;
    for (size_t off = 0; off < num_pairs; off += chunk, ++chunk_idx) {
      if ((use_global || use_banded_row) && dual_scratch && chunk_idx >= 2) {
        id<MTLCommandBuffer> prev_cmd = in_flight[chunk_idx - 2];
        [prev_cmd waitUntilCompleted];
        check(prev_cmd);
      }

      // The kernels read buffer(8) as a 64-bit `long`: typed here, never narrowed.
      const std::int64_t pair_offset = static_cast<std::int64_t>(off);
      const size_t this_chunk = std::min(chunk, num_pairs - off);

      id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:pipeline];
      [enc setBuffer:owned.series  offset:0 atIndex:0];
      [enc setBuffer:owned.lengths offset:0 atIndex:1];
      [enc setBuffer:owned.out     offset:0 atIndex:2];
      [enc setBytes:&N_int     length:sizeof(int) atIndex:3];
      [enc setBytes:&max_L     length:sizeof(int) atIndex:4];
      [enc setBytes:&band      length:sizeof(int) atIndex:5];
      [enc setBytes:&use_sq_l2 length:sizeof(int) atIndex:6];
      if (use_global || use_banded_row) {
        id<MTLBuffer> cur_scratch = dual_scratch ? owned.scratch[chunk_idx % 2] : owned.scratch[0];
        [enc setBuffer:cur_scratch offset:0 atIndex:7];
      } else if (use_regtile) {
        [enc setThreadgroupMemoryLength:regtile_smem_len atIndex:0];
      } else {
        [enc setThreadgroupMemoryLength:tg_mem_len atIndex:0];
      }
      [enc setBytes:&pair_offset length:sizeof(pair_offset) atIndex:8];
      if (use_banded_row) {
        [enc setBytes:&banded_stride length:sizeof(int) atIndex:9];
      } else if (use_regtile) {
        const int pairs_per_tg_arg = static_cast<int>(pairs_per_tg);
        [enc setBytes:&pairs_per_tg_arg length:sizeof(int) atIndex:9];
      }

      // Grid geometry:
      //  - banded-row: one thread per pair, dispatch ceil(chunk / tg_size) tgs.
      //  - regtile:    pairs_per_tg warps per tg, dispatch ceil(chunk/pairs_per_tg) tgs.
      //  - wavefront:  one threadgroup per pair.
      MTLSize grid, tg;
      if (use_banded_row) {
        const size_t ntgs = (this_chunk + tg_size - 1) / tg_size;
        grid = MTLSizeMake(ntgs, 1, 1);
        tg   = MTLSizeMake(tg_size, 1, 1);
      } else if (use_regtile) {
        const size_t ntgs = (this_chunk + pairs_per_tg - 1) / pairs_per_tg;
        grid = MTLSizeMake(ntgs, 1, 1);
        tg   = MTLSizeMake(tg_size, 1, 1);
      } else {
        grid = MTLSizeMake(this_chunk, 1, 1);
        tg   = MTLSizeMake(tg_size, 1, 1);
      }
      [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [enc endEncoding];
      [cmd commit];

      if ((use_global || use_banded_row) && !dual_scratch) {
        [cmd waitUntilCompleted];
        check(cmd);
      } else {
        in_flight.push_back(cmd);
      }
    }
    for (id<MTLCommandBuffer> cmd : in_flight) {
      [cmd waitUntilCompleted];
      check(cmd);
    }

    // Every refusal is behind us: size the caller's matrix (one already N x N,
    // as a mapped one is, keeps its storage) and copy the packed FP32 triangle
    // into it, slot for slot: widened in parallel for a float64 matrix, as it
    // is for a float32 one.
    if (out.size() != N) out.resize(N);
    const float *packed = static_cast<const float *>([owned.out contents]);
    out.visit([&](auto *dst) {
      using V = std::remove_pointer_t<decltype(dst)>;
      if (slots >= 4096) {
        constexpr size_t kCopyChunks = 128;
        parallel(kCopyChunks, [&](size_t c) {
          const size_t t0 = slots * c / kCopyChunks;
          const size_t t1 = slots * (c + 1) / kCopyChunks;
          for (size_t t = t0; t < t1; ++t)
            dst[t] = static_cast<V>(dtwc::core::normalize_public_distance(packed[t]));
        });
      } else {
        for (size_t t = 0; t < slots; ++t)
          dst[t] = static_cast<V>(dtwc::core::normalize_public_distance(packed[t]));
      }
    });

    auto t1 = std::chrono::steady_clock::now();
    result.gpu_time_sec =
        std::chrono::duration<double>(t1 - t0).count();
  }

  if (opts.verbose) {
    std::cout << "Metal DTW: " << num_pairs << " pairs in "
              << (result.gpu_time_sec * 1000.0) << " ms";
    std::cout << " on " << metal_device_info() << std::endl;
  }
  // kernel_used is set during dispatch to reflect the actual pipeline chosen.

  return result;
}

} // namespace dtwc::metal

#endif // DTWC_HAS_METAL
