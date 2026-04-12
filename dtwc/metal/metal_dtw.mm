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
    // Safe because out-of-band cells in prev/prev2 buffers are INF from init,
    // and the 3-buffer rotation is such that reads by in-band cells only touch
    // prev-diagonals' in-band cells (proof: (i,j) in band at k implies
    // (i-1,j-1), (i-1,j), (i,j-1) are in band at k-2 and k-1 respectively).
    if (band >= 0) {
      const int band_lo = (k - band + 1) / 2;  // ceil((k-band)/2) when k-band>=0
      const int band_hi = (k + band) / 2;      // floor((k+band)/2)
      i_lo = max(i_lo, band_lo);
      i_hi = min(i_hi, band_hi);
    }
    const int diag_len = i_hi - i_lo + 1;
    if (diag_len <= 0) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      continue;
    }

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
    if (diag_len <= 0) {
      threadgroup_barrier(mem_flags::mem_device);
      continue;
    }

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
)METAL";

// ---------------------------------------------------------------------------
// Lazy-initialized Metal context (device, queue, pipeline).
// ---------------------------------------------------------------------------
struct MetalContext {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pipeline = nil;        // threadgroup-memory wavefront
  id<MTLComputePipelineState> pipeline_global = nil; // device-memory wavefront
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
      [lib release];
      if (!ctx.pipeline_global) {
        ctx.init_failed = true;
        ctx.init_error = err ? [[err localizedDescription] UTF8String]
                             : "newComputePipelineStateWithFunction (global) failed";
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

    // Choose kernel variant based on threadgroup-memory requirement:
    //   threadgroup kernel:  3 * max_L * 4 <= device cap (32KB on M1/M2/M3 -> max_L <= 2730)
    //   global kernel:       uses device memory for the 3 anti-diagonal buffers
    const NSUInteger tg_mem_len = 3 * (NSUInteger)max_L * sizeof(float);
    const NSUInteger tg_mem_cap = ctx.device.maxThreadgroupMemoryLength;
    const bool use_global = (tg_mem_len > tg_mem_cap);

    id<MTLComputePipelineState> pipeline =
        use_global ? ctx.pipeline_global : ctx.pipeline;

    // Chunk pairs across multiple command buffers. macOS's GPU watchdog
    // kills compute that holds the GPU for more than ~2 s per command buffer
    // (error: kIOGPUCommandBufferCallbackErrorImpactingInteractivity).
    // Rough rule: keep each dispatch bounded in total cell count.
    //
    // Global-memory kernel is slower per-pair than threadgroup, so chunk more
    // aggressively for long series.
    const size_t cells_budget = 5e9; // ~5 billion DTW cells per command buffer
    const size_t cells_per_pair = (size_t)max_L * (size_t)max_L;
    size_t chunk = std::max<size_t>(1, cells_budget / cells_per_pair);
    if (chunk > num_pairs) chunk = num_pairs;

    // Allocate scratch sized for one chunk (reused across chunks).
    id<MTLBuffer> buf_scratch = nil;
    if (use_global) {
      const size_t scratch_bytes =
          chunk * 3ULL * (size_t)max_L * sizeof(float);
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

    // Pick threads-per-threadgroup: at most max_L, at most kernel's hint.
    const NSUInteger max_threads = pipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger tg_size = std::min((NSUInteger)max_L, max_threads);
    if (tg_size == 0) tg_size = 1;
    const NSUInteger simd = pipeline.threadExecutionWidth;
    if (tg_size > simd) tg_size = (tg_size / simd) * simd;

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
      if (use_global) {
        [enc setBuffer:buf_scratch offset:0 atIndex:7];
      } else {
        [enc setThreadgroupMemoryLength:tg_mem_len atIndex:0];
      }
      [enc setBytes:&pair_offset length:sizeof(int) atIndex:8];

      MTLSize grid = MTLSizeMake(this_chunk, 1, 1);
      MTLSize tg   = MTLSizeMake(tg_size, 1, 1);
      [enc dispatchThreadgroups:grid threadsPerThreadgroup:tg];
      [enc endEncoding];
      [cmd commit];
      last_cmd = cmd;

      // When the global kernel reuses a single scratch region across chunks,
      // we must wait for each chunk before starting the next (otherwise the
      // next chunk clobbers in-flight scratch).
      if (use_global) {
        [cmd waitUntilCompleted];
        if (cmd.error) {
          NSString *desc = [cmd.error localizedDescription];
          throw std::runtime_error(std::string("Metal kernel failed: ") +
                                   (desc ? [desc UTF8String] : "unknown"));
        }
      }
    }
    if (last_cmd && !use_global) {
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

  return result;
}

} // namespace dtwc::metal

#endif // DTWC_HAS_METAL
