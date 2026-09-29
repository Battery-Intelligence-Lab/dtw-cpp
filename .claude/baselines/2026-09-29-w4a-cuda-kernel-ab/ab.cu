// W4a: CUDA kernel A/B driver (measurement only; lives outside the repository).
//
// The library's kernels come from dtwc/cuda/cuda_dtw.cu, included verbatim and
// compiled with build-cuda's nvcc flags, so every timed kernel is library code.
// dtw_wavefront_kernel_3buf (wavefront_3buf.inc) is a sed clone of
// dtw_wavefront_kernel whose one change is DOUBLE_BUF_MAX = 0: the wavefront as it
// would be with the double-buffer mode deleted. Its launch sizes shared memory for
// 3 buffers (5 in preload mode, L <= 512).
//
// Build: build.bat (after make_clone.sh). Usage: ab f32|f64 N L REPS
// Prints "lib" rows (public entry, forced paths, checked only), "run" rows (one
// per timed launch) and "path" rows (per-path summary).

#include "cuda/cuda_dtw.cu"

namespace dtwc::cuda {
#include "wavefront_3buf.inc"
}

#include "support/deterministic_series.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

template <typename T>
void w4a_oracle(const T *flat, int N, int L, T *out);

namespace {

using namespace dtwc::cuda;

enum class Path { Warp, RegTileW4, RegTileW8, Wavefront, Wavefront3Buf };
constexpr Path kAllPaths[] = { Path::Warp, Path::RegTileW4, Path::RegTileW8,
                               Path::Wavefront, Path::Wavefront3Buf };

const char *path_name(Path p)
{
  switch (p) {
  case Path::Warp: return "warp";
  case Path::RegTileW4: return "regtile_w4";
  case Path::RegTileW8: return "regtile_w8";
  case Path::Wavefront: return "wavefront";
  case Path::Wavefront3Buf: return "wavefront_3buf";
  }
  return "?";
}

bool accepts(Path p, int L)
{
  switch (p) {
  case Path::Warp: return L <= 32;
  case Path::RegTileW4: return L <= 128;
  case Path::RegTileW8: return L <= 256;
  default: return true;
  }
}

Path auto_path(int L)
{
  switch (detail::auto_kernel(static_cast<std::size_t>(L))) {
  case detail::KernelPath::Warp: return Path::Warp;
  case detail::KernelPath::RegTileW4: return Path::RegTileW4;
  case detail::KernelPath::RegTileW8: return Path::RegTileW8;
  case detail::KernelPath::Wavefront: return Path::Wavefront;
  }
  std::abort();
}

void check(cudaError_t e, const char *what)
{
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA error in %s: %s\n", what, cudaGetErrorString(e));
    std::exit(2);
  }
}

struct LaunchInfo {
  int grid = 0;
  std::size_t shmem = 0;
  int n_bufs = 0;
  bool persistent = false;
  int blocks_per_sm = 0;
  int regs = 0;
  std::size_t local_bytes = 0;
};

template <typename K>
void fill_attrs(K kernel, int block, std::size_t shmem, LaunchInfo &info)
{
  cudaFuncAttributes a{};
  check(cudaFuncGetAttributes(&a, kernel), "cudaFuncGetAttributes");
  info.regs = a.numRegs;
  info.local_bytes = a.localSizeBytes;
  check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&info.blocks_per_sm, kernel, block, shmem),
        "occupancy");
}

// One launch of path p, geometry as in launch_dtw_kernel (cuda_dtw.cu). The two
// events bracket the kernel alone; memsets and copies stay outside them.
template <typename T>
float timed_launch(Path p, const T *d_series, const int *d_lengths, T *d_result,
                   int *d_counter, int N, int max_L, int num_pairs, cudaStream_t s,
                   cudaEvent_t e0, cudaEvent_t e1, LaunchInfo &info)
{
  const bool sq = false;  // library default metric: L1
  const int band = -1;    // full band
  const int device_id = 0;
  constexpr int block = PAIRS_PER_BLOCK * 32;  // 256 for every path

  if (p == Path::Warp || p == Path::RegTileW4 || p == Path::RegTileW8) {
    const int grid = (num_pairs + PAIRS_PER_BLOCK - 1) / PAIRS_PER_BLOCK;
    const std::size_t shmem = (p == Path::Warp)
        ? PAIRS_PER_BLOCK * 2 * 32 * sizeof(T)
        : PAIRS_PER_BLOCK * 2 * static_cast<std::size_t>(max_L) * sizeof(T);
    info.grid = grid;
    info.shmem = shmem;
    if (p == Path::Warp) {
      fill_attrs(dtw_warp_kernel<T>, block, shmem, info);
      check(cudaEventRecord(e0, s), "record e0");
      dtw_warp_kernel<T><<<grid, block, shmem, s>>>(d_series, d_lengths, d_result, N, max_L,
                                                     num_pairs, sq, band);
    } else if (p == Path::RegTileW4) {
      fill_attrs(dtw_regtile_kernel<T, 4>, block, shmem, info);
      check(cudaEventRecord(e0, s), "record e0");
      dtw_regtile_kernel<T, 4><<<grid, block, shmem, s>>>(d_series, d_lengths, d_result, N,
                                                           max_L, num_pairs, sq, band);
    } else {
      fill_attrs(dtw_regtile_kernel<T, 8>, block, shmem, info);
      check(cudaEventRecord(e0, s), "record e0");
      dtw_regtile_kernel<T, 8><<<grid, block, shmem, s>>>(d_series, d_lengths, d_result, N,
                                                           max_L, num_pairs, sq, band);
    }
    check(cudaEventRecord(e1, s), "record e1");
  } else {
    auto kernel = (p == Path::Wavefront) ? dtw_wavefront_kernel<T> : dtw_wavefront_kernel_3buf<T>;
    const std::size_t n_bufs = (p == Path::Wavefront)
        ? detail::wavefront_buffer_count(static_cast<std::size_t>(max_L))
        : (max_L <= 512 ? 5 : 3);
    const std::size_t shmem = n_bufs * static_cast<std::size_t>(max_L) * sizeof(T);
    require_shared_mem_fits(shmem, device_id, "w4a wavefront");
    // The library opts in when dynamic > 48 KiB. The clone also counts its 16 B of
    // static shared memory (s_pid): at FP64 L = 2048 its 3 buffers are exactly 48 KiB.
    cudaFuncAttributes attrs{};
    check(cudaFuncGetAttributes(&attrs, kernel), "cudaFuncGetAttributes");
    const std::size_t static_smem = (p == Path::Wavefront) ? 0 : attrs.sharedSizeBytes;
    if (shmem + static_smem > 48 * 1024)
      check(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 static_cast<int>(shmem)), "smem attr");
    // Mechanism test only (unset in the registered A/B): a shared-memory carveout
    // preference in percent, which fixes how much of the SM's L1 is left as cache.
    if (const char *carve = std::getenv("W4A_CARVEOUT"))
      check(cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
                                 std::atoi(carve)), "carveout attr");
    fill_attrs(kernel, block, shmem, info);
    const int persistent_grid = query_gpu_config(device_id).sm_count * std::max(info.blocks_per_sm, 1);
    info.persistent = num_pairs > persistent_grid * 4;
    info.n_bufs = static_cast<int>(n_bufs);
    info.shmem = shmem;
    if (info.persistent) {
      info.grid = persistent_grid;
      check(cudaMemsetAsync(d_counter, 0, sizeof(int), s), "counter memset");
      check(cudaEventRecord(e0, s), "record e0");
      kernel<<<persistent_grid, block, shmem, s>>>(d_series, d_lengths, d_result, N, max_L,
                                                   num_pairs, sq, band, d_counter);
    } else {
      info.grid = num_pairs;
      check(cudaEventRecord(e0, s), "record e0");
      kernel<<<num_pairs, block, shmem, s>>>(d_series, d_lengths, d_result, N, max_L,
                                             num_pairs, sq, band, nullptr);
    }
    check(cudaEventRecord(e1, s), "record e1");
  }
  check(cudaGetLastError(), "launch");
  check(cudaStreamSynchronize(s), "sync");
  float ms = 0.0f;
  check(cudaEventElapsedTime(&ms, e0, e1), "elapsed");
  return ms;
}

// Off-diagonal entries that differ bitwise from the oracle.
template <typename T>
long long mismatches(const std::vector<T> &got, const std::vector<T> &oracle, int N, double &max_abs)
{
  long long bad = 0;
  max_abs = 0.0;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      if (i == j) continue;
      const std::size_t k = std::size_t(i) * N + j;
      if (std::memcmp(&got[k], &oracle[k], sizeof(T)) != 0) {
        ++bad;
        const double d = std::abs(double(got[k]) - double(oracle[k]));
        if (!(d <= max_abs)) max_abs = d;  // NaN (unwritten entry) propagates
      }
    }
  return bad;
}

double median(std::vector<float> v)
{
  std::sort(v.begin(), v.end());
  const std::size_t n = v.size();
  return (n % 2) ? v[n / 2] : 0.5 * (double(v[n / 2 - 1]) + double(v[n / 2]));
}

template <typename T>
int run(const char *prec, int N, int L, int reps)
{
  check(cudaSetDevice(0), "set device");
  const auto &cfg = query_gpu_config(0);
  std::printf("# device %s cc %d.%d sm %d fp64_rate %s\n", cfg.device_name.c_str(),
              cfg.compute_major, cfg.compute_minor, cfg.sm_count,
              cfg.fp64_rate == FP64Rate::Slow ? "Slow" : "Full");

  // bench_cuda_dtw's data: benchmark_series_set(N, L, 200), values in [-1, 1).
  const auto series = dtwc::test_support::benchmark_series_set(std::size_t(N), std::size_t(L), 200u);
  std::vector<T> flat(std::size_t(N) * L);
  for (int i = 0; i < N; ++i)
    for (int k = 0; k < L; ++k)
      flat[std::size_t(i) * L + k] = static_cast<T>(series[i][k]);  // as flatten_series_buffer
  const std::vector<int> lengths(N, L);
  const int num_pairs = N * (N - 1) / 2;
  const double cells = double(num_pairs) * L * L;

  std::vector<T> oracle(std::size_t(N) * N, T(0));
  const auto t0 = std::chrono::steady_clock::now();
  w4a_oracle<T>(flat.data(), N, L, oracle.data());
  const double oracle_s =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  std::printf("# oracle dtwFull_L<%s> %d pairs in %.1f s\n", prec, num_pairs, oracle_s);

  // Public entry with each override: correctness of the library's own dispatch.
  const dtwc::KernelOverride overrides[] = { dtwc::KernelOverride::Auto,
                                             dtwc::KernelOverride::RegTile,
                                             dtwc::KernelOverride::Wavefront };
  const char *override_names[] = { "Auto", "RegTile", "Wavefront" };
  for (int o = 0; o < 3; ++o) {
    CUDADistMatOptions opts;
    opts.precision = std::is_same_v<T, float> ? CUDAPrecision::FP32 : CUDAPrecision::FP64;
    opts.kernel_override = overrides[o];
    const auto res = compute_distance_matrix_cuda(series, opts);
    long long bad = 0;
    for (int i = 0; i < N; ++i)
      for (int j = 0; j < N; ++j) {
        const double want = (i == j) ? 0.0
            : dtwc::core::normalize_public_distance(oracle[std::size_t(i) * N + j]);
        if (std::memcmp(&res.matrix[std::size_t(i) * N + j], &want, sizeof(double)) != 0) ++bad;
      }
    std::printf("lib,%s,%d,%d,%s,%s,%d,%.4f,%lld\n", prec, N, L, override_names[o],
                res.kernel_used.c_str(), int(res.kernel_override_fell_back),
                res.gpu_time_sec * 1e3, bad);
  }

  T *d_series = nullptr, *d_result = nullptr;
  int *d_lengths = nullptr, *d_counter = nullptr;
  check(cudaMalloc(&d_series, flat.size() * sizeof(T)), "malloc series");
  check(cudaMalloc(&d_lengths, N * sizeof(int)), "malloc lengths");
  check(cudaMalloc(&d_result, std::size_t(N) * N * sizeof(T)), "malloc result");
  check(cudaMalloc(&d_counter, sizeof(int)), "malloc counter");
  check(cudaMemcpy(d_series, flat.data(), flat.size() * sizeof(T), cudaMemcpyHostToDevice), "h2d");
  check(cudaMemcpy(d_lengths, lengths.data(), N * sizeof(int), cudaMemcpyHostToDevice), "h2d");
  cudaStream_t s;
  cudaEvent_t e0, e1;
  check(cudaStreamCreate(&s), "stream");
  check(cudaEventCreate(&e0), "event");
  check(cudaEventCreate(&e1), "event");

  std::vector<Path> paths;
  for (Path p : kAllPaths)
    if (accepts(p, L)) paths.push_back(p);
  const Path auto_p = auto_path(L);
  const std::size_t np = paths.size();
  std::vector<std::vector<float>> times(np);
  std::vector<long long> bad(np, 0);
  std::vector<double> worst(np, 0.0);
  std::vector<LaunchInfo> info(np);
  std::vector<T> host(std::size_t(N) * N);

  auto one = [&](std::size_t k) {
    check(cudaMemsetAsync(d_result, 0xFF, host.size() * sizeof(T), s), "result memset");  // NaN
    const float ms = timed_launch<T>(paths[k], d_series, d_lengths, d_result, d_counter, N, L,
                                     num_pairs, s, e0, e1, info[k]);
    check(cudaMemcpy(host.data(), d_result, host.size() * sizeof(T), cudaMemcpyDeviceToHost), "d2h");
    double max_abs = 0.0;
    bad[k] += mismatches(host, oracle, N, max_abs);
    if (!(max_abs <= worst[k])) worst[k] = max_abs;
    return ms;
  };

  for (std::size_t k = 0; k < np; ++k) one(k);  // warm-up, checked, untimed
  for (int r = 0; r < reps; ++r)
    for (std::size_t q = 0; q < np; ++q) {
      const std::size_t k = (q + std::size_t(r)) % np;  // rotate the order each repetition
      const float ms = one(k);
      times[k].push_back(ms);
      std::printf("run,%s,%d,%d,%s,%d,%.4f\n", prec, N, L, path_name(paths[k]), r, ms);
    }

  double auto_median = 0.0;
  for (std::size_t k = 0; k < np; ++k)
    if (paths[k] == auto_p) auto_median = median(times[k]);
  std::printf("# path,prec,N,L,path,auto,median_ms,min_ms,max_ms,gcell_s,ratio_to_auto,"
              "mismatches,max_abs_err,grid,shmem,n_bufs,persistent,blocks_per_sm,regs,local_bytes\n");
  for (std::size_t k = 0; k < np; ++k) {
    const double med = median(times[k]);
    const auto [mn, mx] = std::minmax_element(times[k].begin(), times[k].end());
    std::printf("path,%s,%d,%d,%s,%d,%.4f,%.4f,%.4f,%.1f,%.4f,%lld,%g,%d,%zu,%d,%d,%d,%d,%zu\n",
                prec, N, L, path_name(paths[k]), int(paths[k] == auto_p), med, double(*mn),
                double(*mx), cells / (med * 1e-3) / 1e9, med / auto_median, bad[k], worst[k],
                info[k].grid, info[k].shmem, info[k].n_bufs, int(info[k].persistent),
                info[k].blocks_per_sm, info[k].regs, info[k].local_bytes);
  }
  cudaFree(d_series);
  cudaFree(d_lengths);
  cudaFree(d_result);
  cudaFree(d_counter);
  return 0;
}

} // namespace

int main(int argc, char **argv)
{
  if (argc != 5) {
    std::fprintf(stderr, "usage: ab f32|f64 N L REPS\n");
    return 1;
  }
  const std::string prec = argv[1];
  const int N = std::atoi(argv[2]), L = std::atoi(argv[3]), reps = std::atoi(argv[4]);
  if (prec == "f32") return run<float>("f32", N, L, reps);
  if (prec == "f64") return run<double>("f64", N, L, reps);
  std::fprintf(stderr, "precision must be f32 or f64\n");
  return 1;
}
