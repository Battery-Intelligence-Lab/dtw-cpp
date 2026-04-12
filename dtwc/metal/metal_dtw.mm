/**
 * @file metal_dtw.mm
 * @brief Metal GPU implementation of batch DTW distance computation.
 *
 * @details Anti-diagonal wavefront kernel: one threadgroup per DTW pair.
 *          Threads within a threadgroup cooperate on cells along the
 *          current anti-diagonal (cells with i+j=k are independent).
 *          Three rotating threadgroup-memory buffers hold anti-diagonals
 *          k, k-1, k-2.
 *
 *          This is the initial scaffolded port of the CUDA wavefront
 *          kernel (dtwc/cuda/cuda_dtw.cu). It covers the pairwise distance
 *          matrix path only; LB_Keogh pruning, warp-shuffle, register-tile,
 *          and 1-vs-all/k-vs-all variants are follow-on work.
 *
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

namespace dtwc::metal {

// ---------------------------------------------------------------------------
// MSL kernel source (compiled at runtime via newLibraryWithSource:)
// ---------------------------------------------------------------------------
// Embedding as a raw string keeps the build simple — no xcrun metal step,
// no metallib artifact to locate at runtime. Apple's runtime shader
// compiler caches compiled libraries internally, so the one-time cost is
// amortized across all kernel dispatches in a process.
static NSString *const kDTWMetalSource = @R"METAL(
#include <metal_stdlib>
using namespace metal;

// Decode flat upper-triangle pair index k into (i, j) row-column pair.
// Matches CPU enumeration: k=0 -> (0,1), k=1 -> (0,2), ...
static inline void decode_pair(int k, int N, thread int &i, thread int &j)
{
  float Nf = float(N);
  float kf = float(k);
  i = int(floor(Nf - 0.5f - sqrt((Nf - 0.5f) * (Nf - 0.5f) - 2.0f * kf)));
  int row_start = i * (2 * N - i - 1) / 2;
  if (row_start + (N - i - 1) <= k) {
    row_start += (N - i - 1);
    ++i;
  }
  j = i + 1 + (k - row_start);
}

// Anti-diagonal wavefront DTW (one threadgroup per pair).
//
// Buffers:
//   0: all_series  — N_series * max_L FP32, padded; row s starts at s*max_L
//   1: lengths     — N_series int32
//   2: out_matrix  — N_series * N_series FP32 (row-major, symmetric)
//   3: N_series    — int32
//   4: max_L       — int32 (row pitch of all_series)
//   5: band        — int32 (Sakoe-Chiba band width; -1 = unbounded)
//   6: use_sq_l2   — int32 (0 = |a-b|, 1 = (a-b)^2)
// Threadgroup memory:
//   0: smem        — 3 * max_L float (3 rotating anti-diagonal buffers)
kernel void dtw_wavefront(
    device const float*   all_series [[buffer(0)]],
    device const int*     lengths    [[buffer(1)]],
    device float*         out_matrix [[buffer(2)]],
    constant int&         N_series   [[buffer(3)]],
    constant int&         max_L      [[buffer(4)]],
    constant int&         band       [[buffer(5)]],
    constant int&         use_sq_l2  [[buffer(6)]],
    constant int&         pair_offset [[buffer(8)]],
    threadgroup float*    smem       [[threadgroup(0)]],
    uint tid   [[thread_position_in_threadgroup]],
    uint pid   [[threadgroup_position_in_grid]],
    uint ntids [[threads_per_threadgroup]])
{
  const int num_pairs = N_series * (N_series - 1) / 2;
  const int real_pid = (int)pid + pair_offset;
  if (real_pid >= num_pairs) return;

  int a_idx, b_idx;
  decode_pair(real_pid, N_series, a_idx, b_idx);

  const int La = lengths[a_idx];
  const int Lb = lengths[b_idx];
  const device float *a = all_series + a_idx * max_L;
  const device float *b = all_series + b_idx * max_L;

  const float INF = 3.402823466e+38f; // FLT_MAX
  const int K = La + Lb - 1;          // number of anti-diagonals

  // Three rotating buffers, each of length max_L:
  //   d[k % 3][i]  -> DTW cost on anti-diagonal k, row i
  threadgroup float *d0 = smem + 0 * max_L;
  threadgroup float *d1 = smem + 1 * max_L;
  threadgroup float *d2 = smem + 2 * max_L;

  // Initialize the three buffers so boundary reads are INF before any
  // anti-diagonal has been written.
  for (int i = (int)tid; i < max_L; i += (int)ntids) {
    d0[i] = INF;
    d1[i] = INF;
    d2[i] = INF;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int k = 0; k < K; ++k) {
    // Rotate: current = d[k%3], prev = d[(k-1)%3], prev2 = d[(k-2)%3].
    threadgroup float *cur  = (k % 3 == 0) ? d0 : ((k % 3 == 1) ? d1 : d2);
    threadgroup float *prev = (k % 3 == 0) ? d2 : ((k % 3 == 1) ? d0 : d1);
    threadgroup float *prev2 = (k % 3 == 0) ? d1 : ((k % 3 == 1) ? d2 : d0);

    int i_lo = max(0, k - Lb + 1);
    int i_hi = min(La - 1, k);

    // Band clip: |i - j| = |2i - k| <= band  ->  i in [(k-band+1)/2, (k+band)/2].
    if (band >= 0) {
      const int band_lo = (k - band + 1) / 2;  // ceil((k-band)/2) when k-band>=0
      const int band_hi = (k + band) / 2;      // floor((k+band)/2)
      i_lo = max(i_lo, band_lo);
      i_hi = min(i_hi, band_hi);
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

    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // Final answer: cell (La-1, Lb-1) lives on the last anti-diagonal,
  // written to d[(K-1) % 3][La-1]. Thread 0 stores the result (symmetric).
  if (tid == 0) {
    const int last = K - 1;
    threadgroup float *cur = (last % 3 == 0) ? d0 : ((last % 3 == 1) ? d1 : d2);
    float result = cur[La - 1];
    out_matrix[a_idx * N_series + b_idx] = result;
    out_matrix[b_idx * N_series + a_idx] = result;
  }
}

// Anti-diagonal wavefront DTW with device-memory scratch buffers.
// For series whose 3*max_L*sizeof(float) exceeds the threadgroup-memory cap
// (32 KB on M1/M2/M3 -> max_L ~2730). Same algorithm, just reading/writing
// to `scratch` (unified memory on Apple Silicon) instead of threadgroup memory.
// Layout: scratch[pid * 3 * max_L + band_idx * max_L + row_idx].
kernel void dtw_wavefront_global(
    device const float*   all_series [[buffer(0)]],
    device const int*     lengths    [[buffer(1)]],
    device float*         out_matrix [[buffer(2)]],
    constant int&         N_series   [[buffer(3)]],
    constant int&         max_L      [[buffer(4)]],
    constant int&         band       [[buffer(5)]],
    constant int&         use_sq_l2  [[buffer(6)]],
    device float*         scratch    [[buffer(7)]],
    constant int&         pair_offset [[buffer(8)]],
    uint tid   [[thread_position_in_threadgroup]],
    uint pid   [[threadgroup_position_in_grid]],
    uint ntids [[threads_per_threadgroup]])
{
  const int num_pairs = N_series * (N_series - 1) / 2;
  const int real_pid = (int)pid + pair_offset;
  if (real_pid >= num_pairs) return;

  int a_idx, b_idx;
  decode_pair(real_pid, N_series, a_idx, b_idx);

  const int La = lengths[a_idx];
  const int Lb = lengths[b_idx];
  const device float *a = all_series + a_idx * max_L;
  const device float *b = all_series + b_idx * max_L;

  const float INF = 3.402823466e+38f;
  const int K = La + Lb - 1;

  // Per-threadgroup slice of scratch: 3 buffers of max_L floats each.
  // Indexed by local threadgroup position (not real_pid) so that chunked
  // dispatches can reuse the same scratch region.
  device float *my_scratch = scratch + (size_t)pid * 3 * max_L;
  device float *d0 = my_scratch + 0 * max_L;
  device float *d1 = my_scratch + 1 * max_L;
  device float *d2 = my_scratch + 2 * max_L;

  for (int i = (int)tid; i < max_L; i += (int)ntids) {
    d0[i] = INF;
    d1[i] = INF;
    d2[i] = INF;
  }
  threadgroup_barrier(mem_flags::mem_device);

  for (int k = 0; k < K; ++k) {
    device float *cur   = (k % 3 == 0) ? d0 : ((k % 3 == 1) ? d1 : d2);
    device float *prev  = (k % 3 == 0) ? d2 : ((k % 3 == 1) ? d0 : d1);
    device float *prev2 = (k % 3 == 0) ? d1 : ((k % 3 == 1) ? d2 : d0);

    int i_lo = max(0, k - Lb + 1);
    int i_hi = min(La - 1, k);
    if (band >= 0) {
      const int band_lo = (k - band + 1) / 2;
      const int band_hi = (k + band) / 2;
      i_lo = max(i_lo, band_lo);
      i_hi = min(i_hi, band_hi);
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
          best = 0.0f;
        } else {
          float cdiag = (i > 0 && j > 0) ? prev2[i - 1] : INF;
          float cup   = (i > 0)          ? prev[i - 1]  : INF;
          float cleft = (j > 0)          ? prev[i]      : INF;
          best = min(cdiag, min(cup, cleft));
        }

        cur[i] = cost + best;
      }
    }

    // INF-fill band-adjacent cells to prevent stale-data reads on subsequent
    // diagonals (see dtw_wavefront comment).
    if ((int)tid == 0 && band >= 0) {
      if (i_lo - 1 >= 0 && i_lo - 1 < max_L) cur[i_lo - 1] = INF;
      if (i_hi + 1 >= 0 && i_hi + 1 < max_L) cur[i_hi + 1] = INF;
    }

    threadgroup_barrier(mem_flags::mem_device);
  }

  if (tid == 0) {
    const int last = K - 1;
    device float *cur = (last % 3 == 0) ? d0 : ((last % 3 == 1) ? d1 : d2);
    float result = cur[La - 1];
    out_matrix[a_idx * N_series + b_idx] = result;
    out_matrix[b_idx * N_series + a_idx] = result;
  }
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
    constant int&         pair_offset [[buffer(8)]],
    constant int&         stride      [[buffer(9)]],
    uint gid [[thread_position_in_grid]])
{
  const int num_pairs = N_series * (N_series - 1) / 2;
  const int real_pid = (int)gid + pair_offset;
  if (real_pid >= num_pairs) return;

  int a_idx, b_idx;
  decode_pair(real_pid, N_series, a_idx, b_idx);

  const int La = lengths[a_idx];
  const int Lb = lengths[b_idx];
  const device float *a = all_series + a_idx * max_L;
  const device float *b = all_series + b_idx * max_L;

  const int W = 2 * band + 1;                // strip width
  const float INF = 3.402823466e+38f;

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
  float result = (r_final >= 0 && r_final < W) ? prev[r_final * stride] : INF;
  out_matrix[a_idx * N_series + b_idx] = result;
  out_matrix[b_idx * N_series + a_idx] = result;
}

// ---------------------------------------------------------------------------
// K-vs-N DTW: one threadgroup per (query k, target j) pair. Structure mirrors
// `dtw_wavefront` / `dtw_wavefront_global` but replaces the upper-triangle
// decode with explicit q = pid / N, t = pid % N, and reads the query and
// target from two separate buffers.
// ---------------------------------------------------------------------------
struct KVNParams {
  int N_target;
  int max_L;
  int band;
  int use_sq_l2;
  int pair_offset;
  int num_pairs;
};

kernel void dtw_kvn_wavefront(
    device const float*   queries    [[buffer(0)]],
    device const float*   targets    [[buffer(1)]],
    device const int*     q_lengths  [[buffer(2)]],
    device const int*     t_lengths  [[buffer(3)]],
    device float*         out_matrix [[buffer(4)]],
    constant KVNParams&   p          [[buffer(5)]],
    threadgroup float*    smem       [[threadgroup(0)]],
    uint tid   [[thread_position_in_threadgroup]],
    uint pid   [[threadgroup_position_in_grid]],
    uint ntids [[threads_per_threadgroup]])
{
  const int N_target = p.N_target;
  const int max_L    = p.max_L;
  const int band     = p.band;
  const int use_sq_l2 = p.use_sq_l2;
  const int pair_offset = p.pair_offset;
  const int num_pairs = p.num_pairs;

  const int real_pid = (int)pid + pair_offset;
  if (real_pid >= num_pairs) return;

  const int q_idx = real_pid / N_target;
  const int t_idx = real_pid - q_idx * N_target;

  const int La = q_lengths[q_idx];
  const int Lb = t_lengths[t_idx];
  const device float *a = queries + q_idx * max_L;
  const device float *b = targets + t_idx * max_L;

  const float INF = 3.402823466e+38f;
  const int K = La + Lb - 1;

  threadgroup float *d0 = smem + 0 * max_L;
  threadgroup float *d1 = smem + 1 * max_L;
  threadgroup float *d2 = smem + 2 * max_L;

  for (int i = (int)tid; i < max_L; i += (int)ntids) {
    d0[i] = INF;
    d1[i] = INF;
    d2[i] = INF;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int k = 0; k < K; ++k) {
    threadgroup float *cur  = (k % 3 == 0) ? d0 : ((k % 3 == 1) ? d1 : d2);
    threadgroup float *prev = (k % 3 == 0) ? d2 : ((k % 3 == 1) ? d0 : d1);
    threadgroup float *prev2 = (k % 3 == 0) ? d1 : ((k % 3 == 1) ? d2 : d0);

    int i_lo = max(0, k - Lb + 1);
    int i_hi = min(La - 1, k);
    if (band >= 0) {
      const int band_lo = (k - band + 1) / 2;
      const int band_hi = (k + band) / 2;
      i_lo = max(i_lo, band_lo);
      i_hi = min(i_hi, band_hi);
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
          best = 0.0f;
        } else {
          float cdiag = (i > 0 && j > 0) ? prev2[i - 1] : INF;
          float cup   = (i > 0)          ? prev[i - 1]  : INF;
          float cleft = (j > 0)          ? prev[i]      : INF;
          best = min(cdiag, min(cup, cleft));
        }
        cur[i] = cost + best;
      }
    }
    if ((int)tid == 0 && band >= 0) {
      if (i_lo - 1 >= 0 && i_lo - 1 < max_L) cur[i_lo - 1] = INF;
      if (i_hi + 1 >= 0 && i_hi + 1 < max_L) cur[i_hi + 1] = INF;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (tid == 0) {
    const int last = K - 1;
    threadgroup float *cur = (last % 3 == 0) ? d0 : ((last % 3 == 1) ? d1 : d2);
    out_matrix[q_idx * N_target + t_idx] = cur[La - 1];
  }
}

// K-vs-N with device-memory scratch (max_L > threadgroup cap).
kernel void dtw_kvn_wavefront_global(
    device const float*   queries    [[buffer(0)]],
    device const float*   targets    [[buffer(1)]],
    device const int*     q_lengths  [[buffer(2)]],
    device const int*     t_lengths  [[buffer(3)]],
    device float*         out_matrix [[buffer(4)]],
    constant KVNParams&   p          [[buffer(5)]],
    device float*         scratch    [[buffer(6)]],
    uint tid   [[thread_position_in_threadgroup]],
    uint pid   [[threadgroup_position_in_grid]],
    uint ntids [[threads_per_threadgroup]])
{
  const int N_target = p.N_target;
  const int max_L    = p.max_L;
  const int band     = p.band;
  const int use_sq_l2 = p.use_sq_l2;
  const int pair_offset = p.pair_offset;
  const int num_pairs = p.num_pairs;

  const int real_pid = (int)pid + pair_offset;
  if (real_pid >= num_pairs) return;

  const int q_idx = real_pid / N_target;
  const int t_idx = real_pid - q_idx * N_target;

  const int La = q_lengths[q_idx];
  const int Lb = t_lengths[t_idx];
  const device float *a = queries + q_idx * max_L;
  const device float *b = targets + t_idx * max_L;

  const float INF = 3.402823466e+38f;
  const int K = La + Lb - 1;

  device float *my_scratch = scratch + (size_t)pid * 3 * max_L;
  device float *d0 = my_scratch + 0 * max_L;
  device float *d1 = my_scratch + 1 * max_L;
  device float *d2 = my_scratch + 2 * max_L;

  for (int i = (int)tid; i < max_L; i += (int)ntids) {
    d0[i] = INF;
    d1[i] = INF;
    d2[i] = INF;
  }
  threadgroup_barrier(mem_flags::mem_device);

  for (int k = 0; k < K; ++k) {
    device float *cur   = (k % 3 == 0) ? d0 : ((k % 3 == 1) ? d1 : d2);
    device float *prev  = (k % 3 == 0) ? d2 : ((k % 3 == 1) ? d0 : d1);
    device float *prev2 = (k % 3 == 0) ? d1 : ((k % 3 == 1) ? d2 : d0);

    int i_lo = max(0, k - Lb + 1);
    int i_hi = min(La - 1, k);
    if (band >= 0) {
      const int band_lo = (k - band + 1) / 2;
      const int band_hi = (k + band) / 2;
      i_lo = max(i_lo, band_lo);
      i_hi = min(i_hi, band_hi);
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
          best = 0.0f;
        } else {
          float cdiag = (i > 0 && j > 0) ? prev2[i - 1] : INF;
          float cup   = (i > 0)          ? prev[i - 1]  : INF;
          float cleft = (j > 0)          ? prev[i]      : INF;
          best = min(cdiag, min(cup, cleft));
        }
        cur[i] = cost + best;
      }
    }
    if ((int)tid == 0 && band >= 0) {
      if (i_lo - 1 >= 0 && i_lo - 1 < max_L) cur[i_lo - 1] = INF;
      if (i_hi + 1 >= 0 && i_hi + 1 < max_L) cur[i_hi + 1] = INF;
    }
    threadgroup_barrier(mem_flags::mem_device);
  }

  if (tid == 0) {
    const int last = K - 1;
    device float *cur = (last % 3 == 0) ? d0 : ((last % 3 == 1) ? d1 : d2);
    out_matrix[q_idx * N_target + t_idx] = cur[La - 1];
  }
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
  id<MTLComputePipelineState> pipeline_kvn = nil;        // K-vs-N threadgroup-memory
  id<MTLComputePipelineState> pipeline_kvn_global = nil; // K-vs-N device-memory
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
      id<MTLLibrary> lib = [ctx.device newLibraryWithSource:kDTWMetalSource
                                                   options:nil
                                                     error:&err];
      if (!lib) {
        ctx.init_failed = true;
        ctx.init_error = err ? [[err localizedDescription] UTF8String]
                             : "newLibraryWithSource failed";
        return;
      }

      id<MTLFunction> fn = [lib newFunctionWithName:@"dtw_wavefront"];
      if (!fn) {
        ctx.init_failed = true;
        ctx.init_error = "kernel function dtw_wavefront not found";
        [lib release];
        return;
      }

      ctx.pipeline = [ctx.device newComputePipelineStateWithFunction:fn
                                                               error:&err];
      [fn release];
      if (!ctx.pipeline) {
        ctx.init_failed = true;
        ctx.init_error = err ? [[err localizedDescription] UTF8String]
                             : "newComputePipelineStateWithFunction failed";
        [lib release];
        return;
      }

      // Second pipeline: device-memory variant for long series.
      id<MTLFunction> fn_global =
          [lib newFunctionWithName:@"dtw_wavefront_global"];
      if (!fn_global) {
        ctx.init_failed = true;
        ctx.init_error = "kernel function dtw_wavefront_global not found";
        [lib release];
        return;
      }
      ctx.pipeline_global =
          [ctx.device newComputePipelineStateWithFunction:fn_global error:&err];
      [fn_global release];
      if (!ctx.pipeline_global) {
        ctx.init_failed = true;
        ctx.init_error = err ? [[err localizedDescription] UTF8String]
                             : "newComputePipelineStateWithFunction (global) failed";
        [lib release];
        return;
      }

      // Third pipeline: row-major banded kernel for tight Sakoe-Chiba bands.
      id<MTLFunction> fn_banded_row =
          [lib newFunctionWithName:@"dtw_banded_row"];
      if (!fn_banded_row) {
        ctx.init_failed = true;
        ctx.init_error = "kernel function dtw_banded_row not found";
        [lib release];
        return;
      }
      ctx.pipeline_banded_row =
          [ctx.device newComputePipelineStateWithFunction:fn_banded_row error:&err];
      [fn_banded_row release];
      if (!ctx.pipeline_banded_row) {
        ctx.init_failed = true;
        ctx.init_error = err ? [[err localizedDescription] UTF8String]
                             : "newComputePipelineStateWithFunction (banded_row) failed";
        [lib release];
        return;
      }

      // Fourth/fifth pipelines: K-vs-N wavefront (threadgroup + global).
      id<MTLFunction> fn_kvn = [lib newFunctionWithName:@"dtw_kvn_wavefront"];
      if (!fn_kvn) {
        ctx.init_failed = true;
        ctx.init_error = "kernel function dtw_kvn_wavefront not found";
        [lib release];
        return;
      }
      ctx.pipeline_kvn =
          [ctx.device newComputePipelineStateWithFunction:fn_kvn error:&err];
      [fn_kvn release];
      if (!ctx.pipeline_kvn) {
        ctx.init_failed = true;
        ctx.init_error = err ? [[err localizedDescription] UTF8String]
                             : "newComputePipelineStateWithFunction (kvn) failed";
        [lib release];
        return;
      }

      id<MTLFunction> fn_kvn_g =
          [lib newFunctionWithName:@"dtw_kvn_wavefront_global"];
      if (!fn_kvn_g) {
        ctx.init_failed = true;
        ctx.init_error = "kernel function dtw_kvn_wavefront_global not found";
        [lib release];
        return;
      }
      ctx.pipeline_kvn_global =
          [ctx.device newComputePipelineStateWithFunction:fn_kvn_g error:&err];
      [fn_kvn_g release];
      [lib release];
      if (!ctx.pipeline_kvn_global) {
        ctx.init_failed = true;
        ctx.init_error = err ? [[err localizedDescription] UTF8String]
                             : "newComputePipelineStateWithFunction (kvn_global) failed";
        return;
      }

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
MetalDistMatResult compute_distance_matrix_metal(
    const std::vector<std::vector<double>> &series,
    const MetalDistMatOptions &opts)
{
  MetalDistMatResult result;
  const size_t N = series.size();
  result.n = N;
  result.matrix.assign(N * N, 0.0);

  if (N <= 1) return result;

  auto &ctx = context();
  if (!ctx.initialized) {
    if (opts.verbose) {
      std::cerr << "[Metal] Backend unavailable: " << ctx.init_error << '\n';
    }
    return result;
  }

  // Find max length and build padded FP32 input buffer.
  int max_L = 0;
  std::vector<int> lengths(N);
  for (size_t s = 0; s < N; ++s) {
    lengths[s] = static_cast<int>(series[s].size());
    if (lengths[s] > max_L) max_L = lengths[s];
  }
  if (max_L == 0) return result;

  const size_t num_pairs = N * (N - 1) / 2;
  result.pairs_computed = num_pairs;

  if (opts.precision == MetalPrecision::FP64 && opts.verbose) {
    std::cerr << "[Metal] FP64 not implemented; using FP32.\n";
  }

  @autoreleasepool {
    auto t0 = std::chrono::steady_clock::now();

    // Upload series (FP32, padded)
    const size_t series_bytes = N * max_L * sizeof(float);
    id<MTLBuffer> buf_series = [ctx.device
        newBufferWithLength:series_bytes
                    options:MTLResourceStorageModeShared];
    if (!buf_series) throw std::runtime_error("Metal: series buffer allocation failed");
    float *series_ptr = static_cast<float *>([buf_series contents]);
    std::memset(series_ptr, 0, series_bytes);
    for (size_t s = 0; s < N; ++s) {
      for (int k = 0; k < lengths[s]; ++k) {
        series_ptr[s * max_L + k] = static_cast<float>(series[s][k]);
      }
    }

    // Upload lengths
    id<MTLBuffer> buf_lengths = [ctx.device
        newBufferWithBytes:lengths.data()
                    length:N * sizeof(int)
                   options:MTLResourceStorageModeShared];
    if (!buf_lengths) throw std::runtime_error("Metal: lengths buffer allocation failed");

    // Output matrix (FP32 on device; promoted to double on host).
    id<MTLBuffer> buf_out = [ctx.device
        newBufferWithLength:N * N * sizeof(float)
                    options:MTLResourceStorageModeShared];
    if (!buf_out) throw std::runtime_error("Metal: output buffer allocation failed");
    std::memset([buf_out contents], 0, N * N * sizeof(float));

    // Scalar args
    const int N_int = static_cast<int>(N);
    const int band = opts.band;
    const int use_sq_l2 = opts.use_squared_l2 ? 1 : 0;

    // Choose kernel variant based on workload:
    //   banded-row kernel: tight Sakoe-Chiba band (band > 0, band*4 < max_L).
    //                      One thread per pair, row-major iteration, no barriers.
    //                      Wins on tight bands where the wavefront's 2*La
    //                      threadgroup barriers dominate.
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
        (band > 0) && (band * 20 < max_L) && (band <= 512);
    const bool use_global = !use_banded_row && (tg_mem_len > tg_mem_cap);

    id<MTLComputePipelineState> pipeline;
    if (use_banded_row)   pipeline = ctx.pipeline_banded_row;
    else if (use_global)  pipeline = ctx.pipeline_global;
    else                  pipeline = ctx.pipeline;

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

    // Allocate scratch sized for one chunk (reused across chunks).
    id<MTLBuffer> buf_scratch = nil;
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
      buf_scratch = [ctx.device newBufferWithLength:scratch_bytes
                                            options:MTLResourceStorageModePrivate];
      if (!buf_scratch) {
        [buf_out release];
        [buf_lengths release];
        [buf_series release];
        if (opts.verbose) {
          std::cerr << "[Metal] scratch allocation failed (" << scratch_bytes
                    << " bytes for max_L=" << max_L
                    << ", chunk=" << chunk << "); falling back to CPU.\n";
        }
        result.matrix.clear();
        result.matrix.resize(N * N, 0.0);
        result.pairs_computed = 0;
        return result;
      }
    }

    // Pick threads-per-threadgroup.
    //  - banded-row: one thread per pair, so tg_size = simd width (32) for
    //    cheap dispatch without intra-threadgroup cooperation.
    //  - wavefront kernels: cooperate within a pair, use up to max_L threads
    //    rounded to a multiple of the simd width.
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    const NSUInteger simd = pipeline.threadExecutionWidth;
    NSUInteger tg_size;
    if (use_banded_row) {
      tg_size = simd;
      if (tg_size > max_threads) tg_size = max_threads;
    } else {
      tg_size = std::min((NSUInteger)max_L, max_threads);
      if (tg_size == 0) tg_size = 1;
      if (tg_size > simd) tg_size = (tg_size / simd) * simd;
    }

    // Banded-row: allocate coalesced scratch sized by total thread count.
    // Grid is 1D of (ceil(chunk/tg_size) * tg_size) threads.
    int banded_stride = 0;
    if (use_banded_row) {
      const size_t ntgs = (chunk + tg_size - 1) / tg_size;
      banded_stride = static_cast<int>(ntgs * tg_size);
      scratch_bytes =
          2ULL * (size_t)(2 * band + 1) * (size_t)banded_stride * sizeof(float);
      buf_scratch = [ctx.device newBufferWithLength:scratch_bytes
                                            options:MTLResourceStorageModePrivate];
      if (!buf_scratch) {
        [buf_out release];
        [buf_lengths release];
        [buf_series release];
        if (opts.verbose) {
          std::cerr << "[Metal] banded-row scratch alloc failed ("
                    << scratch_bytes << " bytes); CPU fallback.\n";
        }
        result.matrix.clear();
        result.matrix.resize(N * N, 0.0);
        result.pairs_computed = 0;
        return result;
      }
    }

    id<MTLCommandBuffer> last_cmd = nil;
    for (size_t off = 0; off < num_pairs; off += chunk) {
      const int pair_offset = static_cast<int>(off);
      const size_t this_chunk = std::min(chunk, num_pairs - off);

      id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:pipeline];
      [enc setBuffer:buf_series  offset:0 atIndex:0];
      [enc setBuffer:buf_lengths offset:0 atIndex:1];
      [enc setBuffer:buf_out     offset:0 atIndex:2];
      [enc setBytes:&N_int     length:sizeof(int) atIndex:3];
      [enc setBytes:&max_L     length:sizeof(int) atIndex:4];
      [enc setBytes:&band      length:sizeof(int) atIndex:5];
      [enc setBytes:&use_sq_l2 length:sizeof(int) atIndex:6];
      if (use_global || use_banded_row) {
        [enc setBuffer:buf_scratch offset:0 atIndex:7];
      } else {
        [enc setThreadgroupMemoryLength:tg_mem_len atIndex:0];
      }
      [enc setBytes:&pair_offset length:sizeof(int) atIndex:8];
      if (use_banded_row) {
        [enc setBytes:&banded_stride length:sizeof(int) atIndex:9];
      }

      // Grid geometry:
      //  - banded-row: one thread per pair, dispatch ceil(chunk / tg_size) tgs.
      //  - wavefront:  one threadgroup per pair.
      MTLSize grid, tg;
      if (use_banded_row) {
        const size_t ntgs = (this_chunk + tg_size - 1) / tg_size;
        grid = MTLSizeMake(ntgs, 1, 1);
        tg   = MTLSizeMake(tg_size, 1, 1);
      } else {
        grid = MTLSizeMake(this_chunk, 1, 1);
        tg   = MTLSizeMake(tg_size, 1, 1);
      }
      [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [enc endEncoding];
      [cmd commit];
      last_cmd = cmd;

      // When scratch is reused across chunks we must wait for each chunk
      // before launching the next (otherwise the next chunk clobbers
      // in-flight scratch). Applies to both global and banded-row kernels.
      if (use_global || use_banded_row) {
        [cmd waitUntilCompleted];
        if (cmd.error) {
          NSString *desc = [cmd.error localizedDescription];
          throw std::runtime_error(std::string("Metal kernel failed: ") +
                                   (desc ? [desc UTF8String] : "unknown"));
        }
      }
    }
    if (last_cmd && !use_global && !use_banded_row) {
      [last_cmd waitUntilCompleted];
      if (last_cmd.error) {
        NSString *desc = [last_cmd.error localizedDescription];
        throw std::runtime_error(std::string("Metal kernel failed: ") +
                                 (desc ? [desc UTF8String] : "unknown"));
      }
    }

    // Copy result back (FP32 -> FP64 for API compatibility with CUDA path).
    const float *out_ptr = static_cast<const float *>([buf_out contents]);
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        result.matrix[i * N + j] = static_cast<double>(out_ptr[i * N + j]);
      }
    }

    [buf_out release];
    [buf_lengths release];
    [buf_series release];
    if (buf_scratch) [buf_scratch release];

    auto t1 = std::chrono::steady_clock::now();
    result.gpu_time_sec =
        std::chrono::duration<double>(t1 - t0).count();
  }

  if (opts.verbose) {
    std::cout << "Metal DTW: " << num_pairs << " pairs in "
              << (result.gpu_time_sec * 1000.0) << " ms on "
              << metal_device_info() << std::endl;
  }
  // Record which kernel path was taken so tests / benchmarks can assert on it.
  if (opts.band > 0 && opts.band * 20 < max_L && opts.band <= 512) {
    result.kernel_used = "banded_row";
  } else if (3u * (unsigned)max_L * sizeof(float)
             > (unsigned)ctx.device.maxThreadgroupMemoryLength) {
    result.kernel_used = "wavefront_global";
  } else {
    result.kernel_used = "wavefront";
  }

  return result;
}

// ===========================================================================
// K-vs-N implementation
// ===========================================================================

namespace {

// Core dispatch: K queries against N targets -> K*N distance matrix.
MetalKVsNResult compute_kvn_impl(
    const std::vector<std::vector<double>> &queries,
    const std::vector<std::vector<double>> &targets,
    const MetalDistMatOptions &opts)
{
  MetalKVsNResult result;
  const size_t Kq = queries.size();
  const size_t N  = targets.size();
  result.k = Kq;
  result.n = N;
  result.distances.assign(Kq * N, 0.0);

  if (Kq == 0 || N == 0) return result;

  auto &ctx = context();
  if (!ctx.initialized) {
    if (opts.verbose) {
      std::cerr << "[Metal] Backend unavailable: " << ctx.init_error << '\n';
    }
    return result;
  }

  // Shared max_L across queries and targets. Users with very different query
  // vs target lengths pay padding cost; the CUDA path has the same behavior.
  int max_L = 0;
  std::vector<int> q_len(Kq), t_len(N);
  for (size_t k = 0; k < Kq; ++k) {
    q_len[k] = static_cast<int>(queries[k].size());
    if (q_len[k] > max_L) max_L = q_len[k];
  }
  for (size_t j = 0; j < N; ++j) {
    t_len[j] = static_cast<int>(targets[j].size());
    if (t_len[j] > max_L) max_L = t_len[j];
  }
  if (max_L == 0) return result;

  const size_t num_pairs = Kq * N;

  if (opts.precision == MetalPrecision::FP64 && opts.verbose) {
    std::cerr << "[Metal] FP64 not implemented; using FP32.\n";
  }

  @autoreleasepool {
    auto t0 = std::chrono::steady_clock::now();

    auto upload_series = [&](const std::vector<std::vector<double>> &ser,
                             size_t count) -> id<MTLBuffer> {
      const size_t bytes = count * max_L * sizeof(float);
      id<MTLBuffer> buf = [ctx.device newBufferWithLength:bytes
                                                  options:MTLResourceStorageModeShared];
      if (!buf) return nil;
      float *p = static_cast<float *>([buf contents]);
      std::memset(p, 0, bytes);
      for (size_t s = 0; s < count; ++s) {
        for (size_t k = 0; k < ser[s].size(); ++k) {
          p[s * max_L + k] = static_cast<float>(ser[s][k]);
        }
      }
      return buf;
    };

    id<MTLBuffer> buf_queries = upload_series(queries, Kq);
    id<MTLBuffer> buf_targets = upload_series(targets, N);
    if (!buf_queries || !buf_targets) {
      throw std::runtime_error("Metal: K-vs-N series buffer allocation failed");
    }

    id<MTLBuffer> buf_qlen = [ctx.device newBufferWithBytes:q_len.data()
                                                     length:Kq * sizeof(int)
                                                    options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_tlen = [ctx.device newBufferWithBytes:t_len.data()
                                                     length:N * sizeof(int)
                                                    options:MTLResourceStorageModeShared];
    if (!buf_qlen || !buf_tlen) {
      throw std::runtime_error("Metal: K-vs-N lengths buffer allocation failed");
    }

    id<MTLBuffer> buf_out = [ctx.device newBufferWithLength:Kq * N * sizeof(float)
                                                    options:MTLResourceStorageModeShared];
    if (!buf_out) throw std::runtime_error("Metal: K-vs-N output buffer allocation failed");
    std::memset([buf_out contents], 0, Kq * N * sizeof(float));

    struct KVNParams {
      int N_target;
      int max_L;
      int band;
      int use_sq_l2;
      int pair_offset;
      int num_pairs;
    };
    KVNParams params{};
    params.N_target    = static_cast<int>(N);
    params.max_L       = max_L;
    params.band        = opts.band;
    params.use_sq_l2   = opts.use_squared_l2 ? 1 : 0;
    params.pair_offset = 0;
    params.num_pairs   = static_cast<int>(num_pairs);

    const NSUInteger tg_mem_len = 3 * (NSUInteger)max_L * sizeof(float);
    const NSUInteger tg_mem_cap = ctx.device.maxThreadgroupMemoryLength;
    const bool use_global = (tg_mem_len > tg_mem_cap);
    id<MTLComputePipelineState> pipeline =
        use_global ? ctx.pipeline_kvn_global : ctx.pipeline_kvn;

    // Chunk pairs across command buffers (same watchdog budget as the NxN path).
    const size_t cells_budget = 5e9;
    const size_t cells_per_pair = (size_t)max_L * (size_t)max_L;
    size_t chunk = std::max<size_t>(1, cells_budget / cells_per_pair);
    if (chunk > num_pairs) chunk = num_pairs;

    id<MTLBuffer> buf_scratch = nil;
    if (use_global) {
      const size_t scratch_bytes =
          chunk * 3ULL * (size_t)max_L * sizeof(float);
      buf_scratch = [ctx.device newBufferWithLength:scratch_bytes
                                            options:MTLResourceStorageModePrivate];
      if (!buf_scratch) {
        [buf_out release];
        [buf_tlen release];
        [buf_qlen release];
        [buf_targets release];
        [buf_queries release];
        if (opts.verbose) {
          std::cerr << "[Metal] K-vs-N scratch alloc failed; CPU fallback.\n";
        }
        return result;
      }
    }

    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger tg_size = std::min((NSUInteger)max_L, max_threads);
    if (tg_size == 0) tg_size = 1;
    const NSUInteger simd = pipeline.threadExecutionWidth;
    if (tg_size > simd) tg_size = (tg_size / simd) * simd;

    id<MTLCommandBuffer> last_cmd = nil;
    for (size_t off = 0; off < num_pairs; off += chunk) {
      params.pair_offset = static_cast<int>(off);
      const size_t this_chunk = std::min(chunk, num_pairs - off);

      id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:pipeline];
      [enc setBuffer:buf_queries offset:0 atIndex:0];
      [enc setBuffer:buf_targets offset:0 atIndex:1];
      [enc setBuffer:buf_qlen    offset:0 atIndex:2];
      [enc setBuffer:buf_tlen    offset:0 atIndex:3];
      [enc setBuffer:buf_out     offset:0 atIndex:4];
      [enc setBytes:&params length:sizeof(KVNParams) atIndex:5];
      if (use_global) {
        [enc setBuffer:buf_scratch offset:0 atIndex:6];
      } else {
        [enc setThreadgroupMemoryLength:tg_mem_len atIndex:0];
      }

      MTLSize grid = MTLSizeMake(this_chunk, 1, 1);
      MTLSize tg   = MTLSizeMake(tg_size, 1, 1);
      [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [enc endEncoding];
      [cmd commit];
      last_cmd = cmd;

      if (use_global) {
        [cmd waitUntilCompleted];
        if (cmd.error) {
          NSString *desc = [cmd.error localizedDescription];
          throw std::runtime_error(std::string("Metal K-vs-N kernel failed: ") +
                                   (desc ? [desc UTF8String] : "unknown"));
        }
      }
    }
    if (last_cmd && !use_global) {
      [last_cmd waitUntilCompleted];
      if (last_cmd.error) {
        NSString *desc = [last_cmd.error localizedDescription];
        throw std::runtime_error(std::string("Metal K-vs-N kernel failed: ") +
                                 (desc ? [desc UTF8String] : "unknown"));
      }
    }

    const float *out_ptr = static_cast<const float *>([buf_out contents]);
    for (size_t i = 0; i < Kq * N; ++i) {
      result.distances[i] = static_cast<double>(out_ptr[i]);
    }

    [buf_out release];
    [buf_tlen release];
    [buf_qlen release];
    [buf_targets release];
    [buf_queries release];
    if (buf_scratch) [buf_scratch release];

    auto t1 = std::chrono::steady_clock::now();
    result.gpu_time_sec = std::chrono::duration<double>(t1 - t0).count();
  }

  result.kernel_used = (3u * (unsigned)max_L * sizeof(float)
                        > (unsigned)ctx.device.maxThreadgroupMemoryLength)
                           ? "kvn_wavefront_global"
                           : "kvn_wavefront";
  if (opts.verbose) {
    std::cout << "Metal K-vs-N: " << num_pairs << " pairs in "
              << (result.gpu_time_sec * 1000.0) << " ms ("
              << result.kernel_used << ")\n";
  }
  return result;
}

} // anonymous namespace

MetalKVsNResult compute_dtw_k_vs_all_metal(
    const std::vector<std::vector<double>> &series,
    const std::vector<size_t> &query_indices,
    const MetalDistMatOptions &opts)
{
  std::vector<std::vector<double>> queries;
  queries.reserve(query_indices.size());
  for (size_t qi : query_indices) {
    if (qi >= series.size()) {
      throw std::out_of_range("compute_dtw_k_vs_all_metal: query_indices out of range");
    }
    queries.push_back(series[qi]);
  }
  return compute_kvn_impl(queries, series, opts);
}

MetalKVsNResult compute_dtw_k_vs_all_metal(
    const std::vector<std::vector<double>> &queries,
    const std::vector<std::vector<double>> &targets,
    const MetalDistMatOptions &opts)
{
  return compute_kvn_impl(queries, targets, opts);
}

MetalOneVsNResult compute_dtw_one_vs_all_metal(
    const std::vector<std::vector<double>> &series,
    size_t query_index,
    const MetalDistMatOptions &opts)
{
  if (query_index >= series.size()) {
    throw std::out_of_range("compute_dtw_one_vs_all_metal: query_index out of range");
  }
  auto kvn = compute_kvn_impl({series[query_index]}, series, opts);
  MetalOneVsNResult out;
  out.distances  = std::move(kvn.distances);
  out.n          = kvn.n;
  out.gpu_time_sec = kvn.gpu_time_sec;
  out.kernel_used  = std::move(kvn.kernel_used);
  return out;
}

MetalOneVsNResult compute_dtw_one_vs_all_metal(
    const std::vector<double> &query,
    const std::vector<std::vector<double>> &targets,
    const MetalDistMatOptions &opts)
{
  auto kvn = compute_kvn_impl({query}, targets, opts);
  MetalOneVsNResult out;
  out.distances  = std::move(kvn.distances);
  out.n          = kvn.n;
  out.gpu_time_sec = kvn.gpu_time_sec;
  out.kernel_used  = std::move(kvn.kernel_used);
  return out;
}

} // namespace dtwc::metal

#endif // DTWC_HAS_METAL
