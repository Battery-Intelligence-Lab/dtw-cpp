/**
 * @file metal_dtw.mm
 * @brief Metal GPU implementation of batch DTW distance computation.
 *
 * @details Five DTW kernel variants share the same pairwise-distance API:
 *            - dtw_wavefront             anti-diagonal, threadgroup memory
 *            - dtw_wavefront_global      anti-diagonal, device memory (long series)
 *            - dtw_banded_row            row-major, one thread per pair (tight band)
 *            - dtw_regtile_w4 / _w8      register-tile, SIMD-group per pair
 *          Three LB_Keogh kernels gate the DTW dispatch when pruning is on:
 *            - compute_envelopes, compute_lb_keogh, compact_active_pairs.
 *
 *          Algorithmic lineage:
 *            - Register-tile + warp-shuffle cost propagation: Schmidt & Hundt
 *              (2020), "cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-
 *              Enabled GPUs", Euro-Par 2020, LNCS 12247 pp. 597-612
 *              [https://doi.org/10.1007/978-3-030-57675-2_37]. Reference
 *              implementation ported via dtwc/cuda/cuda_dtw.cu.
 *            - LB_Keogh lower bound: Keogh & Ratanamahatana (2005), "Exact
 *              Indexing of Dynamic Time Warping", KAIS 7(3) pp. 358-386.
 *            - Sakoe-Chiba band constraint: Sakoe & Chiba (1978),
 *              IEEE TASSP 26(1) pp. 43-49.
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
//   8: pair_offset — int32 (base for chunked dispatch)
//  10: pair_indices — [[optional]] int32 buffer mapping work_idx -> real pair id
//                    (nonempty only when has_pair_indices != 0, used for LB_Keogh pruning)
//  11: has_pair_indices — int32 flag (0 = ignore buffer(10))
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
    device const int*     pair_indices [[buffer(10)]],
    constant int&         has_pair_indices [[buffer(11)]],
    threadgroup float*    smem       [[threadgroup(0)]],
    uint tid   [[thread_position_in_threadgroup]],
    uint pid   [[threadgroup_position_in_grid]],
    uint ntids [[threads_per_threadgroup]])
{
  const int num_pairs = N_series * (N_series - 1) / 2;
  const int work_idx  = (int)pid + pair_offset;
  if (work_idx >= num_pairs) return;
  // Pruning path: resolve the work index through the compacted pair list so
  // we only touch active pairs (pruned pairs have their +∞ already stamped
  // by compact_active_pairs).
  const int real_pid = has_pair_indices ? pair_indices[work_idx] : work_idx;

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
    device const int*     pair_indices [[buffer(10)]],
    constant int&         has_pair_indices [[buffer(11)]],
    uint tid   [[thread_position_in_threadgroup]],
    uint pid   [[threadgroup_position_in_grid]],
    uint ntids [[threads_per_threadgroup]])
{
  const int num_pairs = N_series * (N_series - 1) / 2;
  const int work_idx  = (int)pid + pair_offset;
  if (work_idx >= num_pairs) return;
  const int real_pid = has_pair_indices ? pair_indices[work_idx] : work_idx;

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

// ---------------------------------------------------------------------------
// Register-tile DTW for short/medium series (max_L <= 256, unbanded).
//
// Adapted from Schmidt & Hundt, "cuDTW++: Ultra-Fast Dynamic Time Warping on
// CUDA-Enabled GPUs" (Euro-Par 2020, LNCS 12247), translated from the CUDA
// reference in dtwc/cuda/cuda_dtw.cu (`dtw_regtile_kernel`, lines 543-780).
//
// One SIMD-group (32 threads) per pair, PAIRS_PER_TG warps per threadgroup.
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
constant int PAIRS_PER_TG = 8;

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
  for (int tw = 0; tw < TILE_W; ++tw) {
    int j = col_start + tw;
    col_val[tw] = (j < N_len) ? my_col[j] : 0.0f;
  }

  float penalty[TILE_W];
  float prev_penalty[TILE_W];
  for (int tw = 0; tw < TILE_W; ++tw) {
    penalty[tw] = INF;
    prev_penalty[tw] = INF;
  }
  float prev_last = INF; // saved prev_penalty[TILE_W-1] before overwrite

  const int num_col_threads = (N_len + TILE_W - 1) / TILE_W;
  const int total_steps = M + num_col_threads - 1;

  for (int step = 0; step < total_steps; ++step) {
    const int i = step - (int)simd_lane;
    const bool row_valid =
        (i >= 0) && (i < M) && ((int)simd_lane < num_col_threads);

    const float row_val = row_valid ? my_row[i] : 0.0f;

    // Cross-lane communication (all lanes must participate in shuffles).
    float penalty_from_left = simd_shuffle_up(penalty[TILE_W - 1], 1);
    float diag_from_left    = simd_shuffle_up(prev_last, 1);
    if (simd_lane == 0) {
      penalty_from_left = INF; // no left neighbor for lane 0
      diag_from_left    = INF;
    }

    if (row_valid) {
      const float saved_prev_last = prev_penalty[TILE_W - 1];

      float left = penalty_from_left; // cost[i][col_start - 1]
      float diag = diag_from_left;    // cost[i-1][col_start - 1]

      for (int tw = 0; tw < TILE_W; ++tw) {
        const int j = col_start + tw;
        if (j >= N_len) {
          penalty[tw] = INF;
          diag = prev_penalty[tw];
          left = INF;
          continue;
        }

        const float above = prev_penalty[tw];
        const float diff  = row_val - col_val[tw];
        const float d     = use_sq_l2 ? (diff * diff) : fabs(diff);

        float new_cost;
        if (i == 0 && j == 0) {
          new_cost = d;
        } else {
          const float eff_above = (i == 0)              ? INF : above;
          const float eff_diag  = (i == 0 || j == 0)    ? INF : diag;
          const float eff_left  = (j == 0)              ? INF : left;
          new_cost = min(eff_diag, min(eff_above, eff_left)) + d;
        }

        diag = prev_penalty[tw];
        left = new_cost;
        penalty[tw] = new_cost;
      }

      prev_last = saved_prev_last;
      for (int tw = 0; tw < TILE_W; ++tw) {
        prev_penalty[tw] = penalty[tw];
      }
    }
    // When !row_valid, penalty/prev_penalty/prev_last are untouched; other
    // lanes' shuffles continue to see stable values from our last valid step.
  }

  // Result is cost[M-1][N_len-1].
  const int result_thread = (N_len - 1) / TILE_W;
  const int result_tw     = (N_len - 1) % TILE_W;
  const float my_result =
      ((int)simd_lane == result_thread) ? penalty[result_tw] : INF;
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
    int  pair_offset,
    threadgroup float  *smem,
    uint simd_lane,
    uint simd_id,
    uint tg_idx)
{
  const int num_pairs = N_series * (N_series - 1) / 2;
  const int work_idx  = (int)tg_idx * PAIRS_PER_TG + (int)simd_id;
  const int real_pid  = work_idx + pair_offset;
  if (real_pid >= num_pairs) return;

  int si, sj;
  decode_pair(real_pid, N_series, si, sj);
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
    if (simd_lane == 0) {
      out_matrix[si * N_series + sj] = INF;
      out_matrix[sj * N_series + si] = INF;
    }
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

  if (simd_lane == 0) {
    out_matrix[si * N_series + sj] = final_result;
    out_matrix[sj * N_series + si] = final_result;
  }
}

kernel void dtw_regtile_w4(
    device const float*   all_series [[buffer(0)]],
    device const int*     lengths    [[buffer(1)]],
    device float*         out_matrix [[buffer(2)]],
    constant int&         N_series   [[buffer(3)]],
    constant int&         max_L      [[buffer(4)]],
    constant int&         use_sq_l2  [[buffer(6)]],
    constant int&         pair_offset [[buffer(8)]],
    threadgroup float*    smem       [[threadgroup(0)]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint tg_idx    [[threadgroup_position_in_grid]])
{
  dtw_regtile_kernel_body<4>(all_series, lengths, out_matrix, N_series, max_L,
                             use_sq_l2, pair_offset, smem,
                             simd_lane, simd_id, tg_idx);
}

kernel void dtw_regtile_w8(
    device const float*   all_series [[buffer(0)]],
    device const int*     lengths    [[buffer(1)]],
    device float*         out_matrix [[buffer(2)]],
    constant int&         N_series   [[buffer(3)]],
    constant int&         max_L      [[buffer(4)]],
    constant int&         use_sq_l2  [[buffer(6)]],
    constant int&         pair_offset [[buffer(8)]],
    threadgroup float*    smem       [[threadgroup(0)]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simd_id   [[simdgroup_index_in_threadgroup]],
    uint tg_idx    [[threadgroup_position_in_grid]])
{
  dtw_regtile_kernel_body<8>(all_series, lengths, out_matrix, N_series, max_L,
                             use_sq_l2, pair_offset, smem,
                             simd_lane, simd_id, tg_idx);
}

// ---------------------------------------------------------------------------
// LB_Keogh pipeline — three cooperating kernels:
//   1. compute_envelopes      — sliding min/max per series
//   2. compute_lb_keogh       — symmetric LB per pair
//   3. compact_active_pairs   — threshold filter, stamp INF for pruned pairs
//
// Algorithm: Keogh & Ratanamahatana (2005), "Exact Indexing of Dynamic Time
// Warping", Knowledge and Information Systems 7(3), 358-386. Symmetric
// variant LB = max(LB(j|env_i), LB(i|env_j)) — the tighter of the two single-
// direction bounds, see Rakthanmanon et al. (2012) "Searching and Mining
// Trillions of Time Series Subsequences under DTW", KDD '12.
//
// Ported from the CUDA pipeline at dtwc/cuda/cuda_dtw.cu:785-910 (itself
// inspired by cuDTW++, Schmidt & Hundt 2020).
// ---------------------------------------------------------------------------

// One threadgroup per series. Each thread covers ceil(max_L / ntids) positions.
// Brute-force O(band) scan per position; simple and cache-friendly for the
// small bands used in practice.
kernel void compute_envelopes(
    device const float*   all_series      [[buffer(0)]],
    device const int*     lengths         [[buffer(1)]],
    device float*         upper_envelopes [[buffer(2)]],
    device float*         lower_envelopes [[buffer(3)]],
    constant int&         max_L           [[buffer(4)]],
    constant int&         N_series        [[buffer(5)]],
    constant int&         env_band        [[buffer(6)]],
    uint tid   [[thread_position_in_threadgroup]],
    uint pid   [[threadgroup_position_in_grid]],
    uint ntids [[threads_per_threadgroup]])
{
  const int series_idx = (int)pid;
  if (series_idx >= N_series) return;

  const int L = lengths[series_idx];
  const device float *series = all_series + series_idx * max_L;
  device float *upper = upper_envelopes + series_idx * max_L;
  device float *lower = lower_envelopes + series_idx * max_L;

  const int w = (env_band >= 0) ? env_band : 0;

  for (int k = (int)tid; k < L; k += (int)ntids) {
    const int lo = (k >= w) ? k - w : 0;
    const int hi = (k + w + 1 < L) ? k + w + 1 : L;

    float max_val = series[lo];
    float min_val = series[lo];
    for (int j = lo + 1; j < hi; ++j) {
      const float v = series[j];
      max_val = max(max_val, v);
      min_val = min(min_val, v);
    }
    upper[k] = max_val;
    lower[k] = min_val;
  }

  // Zero-fill padding past the valid series range.
  for (int k = L + (int)tid; k < max_L; k += (int)ntids) {
    upper[k] = 0.0f;
    lower[k] = 0.0f;
  }
}

// One thread per upper-triangle pair. Writes symmetric LB =
// max(LB_Keogh(query=j, env=i), LB_Keogh(query=i, env=j)) to lb_values[pid].
kernel void compute_lb_keogh(
    device const float*   all_series      [[buffer(0)]],
    device const int*     lengths         [[buffer(1)]],
    device const float*   upper_envelopes [[buffer(2)]],
    device const float*   lower_envelopes [[buffer(3)]],
    device float*         lb_values       [[buffer(4)]],
    constant int&         N_series        [[buffer(5)]],
    constant int&         max_L           [[buffer(6)]],
    constant int&         num_pairs       [[buffer(7)]],
    uint gid [[thread_position_in_grid]])
{
  const int pid = (int)gid;
  if (pid >= num_pairs) return;

  int si, sj;
  decode_pair(pid, N_series, si, sj);

  const int Li = lengths[si];
  const int Lj = lengths[sj];
  const int n  = min(Li, Lj);

  const device float *series_i = all_series + si * max_L;
  const device float *series_j = all_series + sj * max_L;
  const device float *upper_i  = upper_envelopes + si * max_L;
  const device float *lower_i  = lower_envelopes + si * max_L;
  const device float *upper_j  = upper_envelopes + sj * max_L;
  const device float *lower_j  = lower_envelopes + sj * max_L;

  float lb1 = 0.0f;
  float lb2 = 0.0f;
  for (int k = 0; k < n; ++k) {
    const float vj = series_j[k];
    const float vi = series_i[k];
    lb1 += max(0.0f, max(vj - upper_i[k], lower_i[k] - vj));
    lb2 += max(0.0f, max(vi - upper_j[k], lower_j[k] - vi));
  }
  lb_values[pid] = max(lb1, lb2);
}

// One thread per pair. Partitions pairs into active (lb <= threshold, appended
// to active_pairs via atomic counter) and pruned (+∞ stamped into
// result_matrix at both (si, sj) and (sj, si)).
kernel void compact_active_pairs(
    device const float*   lb_values     [[buffer(0)]],
    device int*           active_pairs  [[buffer(1)]],
    device atomic_int*    active_count  [[buffer(2)]],
    device float*         result_matrix [[buffer(3)]],
    constant int&         N_series      [[buffer(4)]],
    constant int&         num_pairs     [[buffer(5)]],
    constant float&       threshold     [[buffer(6)]],
    uint gid [[thread_position_in_grid]])
{
  const int pid = (int)gid;
  if (pid >= num_pairs) return;

  if (lb_values[pid] <= threshold) {
    const int slot =
        atomic_fetch_add_explicit(active_count, 1, memory_order_relaxed);
    active_pairs[slot] = pid;
    return;
  }

  const float INF = 3.402823466e+38f;
  int si, sj;
  decode_pair(pid, N_series, si, sj);
  result_matrix[si * N_series + sj] = INF;
  result_matrix[sj * N_series + si] = INF;
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
  id<MTLComputePipelineState> pipeline_kvn = nil;        // K-vs-N threadgroup-memory
  id<MTLComputePipelineState> pipeline_kvn_global = nil; // K-vs-N device-memory
  id<MTLComputePipelineState> pipeline_envelopes = nil;  // LB_Keogh envelope builder
  id<MTLComputePipelineState> pipeline_lb_keogh = nil;   // LB_Keogh pairwise
  id<MTLComputePipelineState> pipeline_compact = nil;    // Threshold + compaction
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

      // Register-tile pipelines (unbanded, max_L <= 256).
      {
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
            ctx.init_error = err ? [[err localizedDescription] UTF8String]
                                 : "newComputePipelineStateWithFunction failed";
            return false;
          }
          return true;
        };

        if (!make_pipeline(@"dtw_regtile_w4", ctx.pipeline_regtile_w4)) {
          [lib release];
          return;
        }
        if (!make_pipeline(@"dtw_regtile_w8", ctx.pipeline_regtile_w8)) {
          [lib release];
          return;
        }
        if (!make_pipeline(@"compute_envelopes", ctx.pipeline_envelopes)) {
          [lib release];
          return;
        }
        if (!make_pipeline(@"compute_lb_keogh", ctx.pipeline_lb_keogh)) {
          [lib release];
          return;
        }
        if (!make_pipeline(@"compact_active_pairs", ctx.pipeline_compact)) {
          [lib release];
          return;
        }
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
        (band > 0) && (band * 20 < max_L) && (band <= 512);
    // Register-tile path: unbanded only for this pass. TILE_W=4 covers
    // max_L in [1, 128] (32 lanes * 4 cols = 128), TILE_W=8 covers (128, 256].
    const bool use_regtile =
        !use_banded_row && (band == -1) && (max_L > 0) && (max_L <= 256);
    const int regtile_tile_w = (max_L <= 128) ? 4 : 8;
    const bool use_global =
        !use_banded_row && !use_regtile && (tg_mem_len > tg_mem_cap);

    id<MTLComputePipelineState> pipeline;
    if (use_banded_row)
      pipeline = ctx.pipeline_banded_row;
    else if (use_regtile)
      pipeline = (regtile_tile_w == 4) ? ctx.pipeline_regtile_w4
                                       : ctx.pipeline_regtile_w8;
    else if (use_global)
      pipeline = ctx.pipeline_global;
    else
      pipeline = ctx.pipeline;

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
    //  - regtile:    PAIRS_PER_TG (8) warps per threadgroup, 1 warp per pair.
    //                tg_size = 8 * simd = 256 threads. Threadgroup memory holds
    //                8 * 2 * max_L floats of series data (16 KB at max_L=256).
    //  - wavefront kernels: cooperate within a pair, use up to max_L threads
    //    rounded to a multiple of the simd width.
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    const NSUInteger simd = pipeline.threadExecutionWidth;
    const NSUInteger kPairsPerTG = 8; // must match PAIRS_PER_TG in MSL
    NSUInteger tg_size;
    if (use_banded_row) {
      tg_size = simd;
      if (tg_size > max_threads) tg_size = max_threads;
    } else if (use_regtile) {
      tg_size = simd * kPairsPerTG;
      if (tg_size > max_threads) tg_size = (max_threads / simd) * simd;
    } else {
      tg_size = std::min((NSUInteger)max_L, max_threads);
      if (tg_size == 0) tg_size = 1;
      if (tg_size > simd) tg_size = (tg_size / simd) * simd;
    }
    const NSUInteger regtile_smem_len =
        use_regtile ? (kPairsPerTG * 2 * (NSUInteger)max_L * sizeof(float)) : 0;

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

    // -----------------------------------------------------------------------
    // Optional LB_Keogh pre-pass: compute envelopes, pairwise lower bounds,
    // and compact pairs whose LB <= threshold. Pruned pairs get +∞ stamped
    // into the result matrix here; the DTW dispatch below then runs only on
    // the survivor list (passed via pair_indices[work_idx]).
    //
    // Only wavefront / wavefront_global kernels support pair_indices in this
    // pass. If user requested LB but selected banded_row/regtile, we silently
    // disable LB (with a verbose-mode warning).
    // -----------------------------------------------------------------------
    const bool pipeline_uses_pair_indices = !use_banded_row && !use_regtile;
    const bool lb_requested = opts.use_lb_keogh;
    bool lb_active = lb_requested && pipeline_uses_pair_indices && num_pairs > 0;
    if (lb_requested && !lb_active && opts.verbose) {
      std::cerr << "[Metal] LB_Keogh requested but current kernel path does "
                   "not support pruning (requires wavefront / wavefront_global); "
                   "disabling pruning.\n";
    }

    id<MTLBuffer> buf_pair_indices = nil;
    int has_pair_indices = 0;
    size_t effective_pairs = num_pairs;

    if (lb_active) {
      int env_band = opts.lb_envelope_band;
      if (env_band < 0) {
        env_band = (opts.band > 0) ? opts.band : std::max(1, max_L / 10);
      }
      const int num_pairs_int = static_cast<int>(num_pairs);
      const float threshold_f32 = static_cast<float>(opts.lb_threshold);

      const size_t env_bytes = (size_t)N * (size_t)max_L * sizeof(float);
      id<MTLBuffer> buf_upper = [ctx.device
          newBufferWithLength:env_bytes options:MTLResourceStorageModePrivate];
      id<MTLBuffer> buf_lower = [ctx.device
          newBufferWithLength:env_bytes options:MTLResourceStorageModePrivate];
      id<MTLBuffer> buf_lb = [ctx.device
          newBufferWithLength:num_pairs * sizeof(float)
                      options:MTLResourceStorageModePrivate];
      buf_pair_indices = [ctx.device
          newBufferWithLength:num_pairs * sizeof(int)
                      options:MTLResourceStorageModePrivate];
      id<MTLBuffer> buf_active_count = [ctx.device
          newBufferWithLength:sizeof(int)
                      options:MTLResourceStorageModeShared];
      if (!buf_upper || !buf_lower || !buf_lb || !buf_pair_indices ||
          !buf_active_count) {
        if (buf_upper) [buf_upper release];
        if (buf_lower) [buf_lower release];
        if (buf_lb)    [buf_lb release];
        if (buf_pair_indices) {
          [buf_pair_indices release];
          buf_pair_indices = nil;
        }
        if (buf_active_count) [buf_active_count release];
        if (opts.verbose) {
          std::cerr << "[Metal] LB_Keogh buffer allocation failed; "
                       "falling back to unpruned DTW.\n";
        }
        lb_active = false;
      } else {
        *static_cast<int *>([buf_active_count contents]) = 0;

        // 1. Envelopes (N threadgroups, threads cooperate on elements).
        {
          id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
          id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
          [enc setComputePipelineState:ctx.pipeline_envelopes];
          [enc setBuffer:buf_series  offset:0 atIndex:0];
          [enc setBuffer:buf_lengths offset:0 atIndex:1];
          [enc setBuffer:buf_upper   offset:0 atIndex:2];
          [enc setBuffer:buf_lower   offset:0 atIndex:3];
          [enc setBytes:&max_L    length:sizeof(int) atIndex:4];
          [enc setBytes:&N_int    length:sizeof(int) atIndex:5];
          [enc setBytes:&env_band length:sizeof(int) atIndex:6];
          const NSUInteger env_max =
              ctx.pipeline_envelopes.maxTotalThreadsPerThreadgroup;
          NSUInteger env_tg = std::min<NSUInteger>(128, env_max);
          if (env_tg > (NSUInteger)max_L && max_L > 0)
            env_tg = (NSUInteger)max_L;
          if (env_tg == 0) env_tg = 1;
          [enc dispatchThreadgroups:MTLSizeMake(N, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(env_tg, 1, 1)];
          [enc endEncoding];
          [cmd commit];
          [cmd waitUntilCompleted];
          if (cmd.error) {
            NSString *desc = [cmd.error localizedDescription];
            throw std::runtime_error(
                std::string("Metal envelopes kernel failed: ") +
                (desc ? [desc UTF8String] : "unknown"));
          }
        }

        // 2. Pairwise LB_Keogh (one thread per pair).
        {
          id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
          id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
          [enc setComputePipelineState:ctx.pipeline_lb_keogh];
          [enc setBuffer:buf_series  offset:0 atIndex:0];
          [enc setBuffer:buf_lengths offset:0 atIndex:1];
          [enc setBuffer:buf_upper   offset:0 atIndex:2];
          [enc setBuffer:buf_lower   offset:0 atIndex:3];
          [enc setBuffer:buf_lb      offset:0 atIndex:4];
          [enc setBytes:&N_int          length:sizeof(int) atIndex:5];
          [enc setBytes:&max_L          length:sizeof(int) atIndex:6];
          [enc setBytes:&num_pairs_int  length:sizeof(int) atIndex:7];
          const NSUInteger lb_max =
              ctx.pipeline_lb_keogh.maxTotalThreadsPerThreadgroup;
          const NSUInteger lb_tg = std::min<NSUInteger>(256, lb_max);
          const NSUInteger lb_ntgs = (num_pairs + lb_tg - 1) / lb_tg;
          [enc dispatchThreadgroups:MTLSizeMake(lb_ntgs, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(lb_tg, 1, 1)];
          [enc endEncoding];
          [cmd commit];
          [cmd waitUntilCompleted];
          if (cmd.error) {
            NSString *desc = [cmd.error localizedDescription];
            throw std::runtime_error(
                std::string("Metal LB_Keogh kernel failed: ") +
                (desc ? [desc UTF8String] : "unknown"));
          }
        }

        // 3. Compact (stamp +∞ for pruned, atomic-append active pids).
        {
          id<MTLCommandBuffer> cmd = [ctx.queue commandBuffer];
          id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
          [enc setComputePipelineState:ctx.pipeline_compact];
          [enc setBuffer:buf_lb           offset:0 atIndex:0];
          [enc setBuffer:buf_pair_indices offset:0 atIndex:1];
          [enc setBuffer:buf_active_count offset:0 atIndex:2];
          [enc setBuffer:buf_out          offset:0 atIndex:3];
          [enc setBytes:&N_int         length:sizeof(int)   atIndex:4];
          [enc setBytes:&num_pairs_int length:sizeof(int)   atIndex:5];
          [enc setBytes:&threshold_f32 length:sizeof(float) atIndex:6];
          const NSUInteger ct_max =
              ctx.pipeline_compact.maxTotalThreadsPerThreadgroup;
          const NSUInteger ct_tg = std::min<NSUInteger>(256, ct_max);
          const NSUInteger ct_ntgs = (num_pairs + ct_tg - 1) / ct_tg;
          [enc dispatchThreadgroups:MTLSizeMake(ct_ntgs, 1, 1)
              threadsPerThreadgroup:MTLSizeMake(ct_tg, 1, 1)];
          [enc endEncoding];
          [cmd commit];
          [cmd waitUntilCompleted];
          if (cmd.error) {
            NSString *desc = [cmd.error localizedDescription];
            throw std::runtime_error(
                std::string("Metal compact kernel failed: ") +
                (desc ? [desc UTF8String] : "unknown"));
          }
        }

        const int active_count =
            *static_cast<int *>([buf_active_count contents]);
        effective_pairs = (size_t)active_count;
        has_pair_indices = 1;
        result.pairs_pruned = num_pairs - effective_pairs;
        result.pairs_computed = effective_pairs;

        [buf_upper release];
        [buf_lower release];
        [buf_lb release];
        [buf_active_count release];
        // buf_pair_indices kept live for the DTW dispatch below.
      }
    }

    // For wavefront kernels we always bind a pair_indices buffer — either the
    // active-pairs list (when pruning) or a 1-int dummy (when not).
    if (pipeline_uses_pair_indices && !buf_pair_indices) {
      buf_pair_indices = [ctx.device
          newBufferWithLength:sizeof(int)
                      options:MTLResourceStorageModePrivate];
      if (!buf_pair_indices) {
        throw std::runtime_error(
            "Metal: pair_indices dummy buffer allocation failed");
      }
    }

    // Trim chunk to effective_pairs so the final chunk never overshoots.
    if (chunk > effective_pairs && effective_pairs > 0) {
      chunk = effective_pairs;
    }

    id<MTLCommandBuffer> last_cmd = nil;
    for (size_t off = 0; off < effective_pairs; off += chunk) {
      const int pair_offset = static_cast<int>(off);
      const size_t this_chunk = std::min(chunk, effective_pairs - off);

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
      } else if (use_regtile) {
        [enc setThreadgroupMemoryLength:regtile_smem_len atIndex:0];
      } else {
        [enc setThreadgroupMemoryLength:tg_mem_len atIndex:0];
      }
      [enc setBytes:&pair_offset length:sizeof(int) atIndex:8];
      if (use_banded_row) {
        [enc setBytes:&banded_stride length:sizeof(int) atIndex:9];
      }
      if (pipeline_uses_pair_indices) {
        [enc setBuffer:buf_pair_indices offset:0 atIndex:10];
        [enc setBytes:&has_pair_indices length:sizeof(int) atIndex:11];
      }

      // Grid geometry:
      //  - banded-row: one thread per pair, dispatch ceil(chunk / tg_size) tgs.
      //  - regtile:    kPairsPerTG warps per tg, dispatch ceil(chunk/kPairsPerTG) tgs.
      //  - wavefront:  one threadgroup per pair.
      MTLSize grid, tg;
      if (use_banded_row) {
        const size_t ntgs = (this_chunk + tg_size - 1) / tg_size;
        grid = MTLSizeMake(ntgs, 1, 1);
        tg   = MTLSizeMake(tg_size, 1, 1);
      } else if (use_regtile) {
        const size_t ntgs = (this_chunk + kPairsPerTG - 1) / kPairsPerTG;
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
    if (buf_pair_indices) [buf_pair_indices release];

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
  const bool lbl_banded_row =
      (opts.band > 0) && (opts.band * 20 < max_L) && (opts.band <= 512);
  const bool lbl_regtile =
      !lbl_banded_row && (opts.band == -1) && (max_L > 0) && (max_L <= 256);
  if (lbl_banded_row) {
    result.kernel_used = "banded_row";
  } else if (lbl_regtile) {
    result.kernel_used = (max_L <= 128) ? "regtile_w4" : "regtile_w8";
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
