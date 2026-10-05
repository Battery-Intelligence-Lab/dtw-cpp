// GC: CLARA's assignment of N series of length L to k medoids (series 0, N/k,
// 2N/k, ...), on the GPU (compute_medoid_distances_cuda; the rate is its
// CUDA-event time: each block's upload, launch and download) or on the CPU
// (fast_clara's loop: the Problem's bound DTW function, run_openmp over the
// series, the nearest medoid kept); and the fill of N series
// (compute_distance_matrix_cuda, CUDA-event time), the GPU's reference rate.
// One warm-up, then REPS timed runs; medians.
// Usage: clara_assign_bench <fill32|fill64|rect32|rect64|cpu> <N> <L> <k> <REPS>
// Series: bench_cuda_dtw's benchmark_series_set(N, L, 200). Scratch tool; not part of the library.
#include <dtwc.hpp>
#include <cuda/cuda_dtw.cuh>

#include "support/deterministic_series.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

struct Run {
  double wall = 0, gpu = 0;
  std::uint64_t hash = 14695981039346656037ull; // FNV-1a 64 over the labels and the nearest distances
};

void mix(std::uint64_t &hash, const void *data, std::size_t bytes)
{
  const auto *b = static_cast<const unsigned char *>(data);
  for (std::size_t i = 0; i < bytes; ++i) hash = (hash ^ b[i]) * 1099511628211ull;
}

double median(std::vector<double> v)
{
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}

} // namespace

int main(int argc, char **argv)
{
  if (argc != 6) {
    std::fprintf(stderr, "usage: clara_assign_bench <fill32|fill64|rect32|rect64|cpu> <N> <L> <k> <REPS>\n");
    return 1;
  }
  const std::string mode = argv[1];
  const std::size_t N = std::strtoull(argv[2], nullptr, 10);
  const std::size_t L = std::strtoull(argv[3], nullptr, 10);
  const std::size_t k = std::strtoull(argv[4], nullptr, 10);
  const int reps = std::atoi(argv[5]);
  const bool fp32 = mode.back() == '2';

  auto series = dtwc::test_support::benchmark_series_set(N, L, 200u);
  std::vector<std::size_t> medoid_index(k);
  std::vector<std::vector<double>> medoids;
  for (std::size_t m = 0; m < k; ++m) {
    medoid_index[m] = m * N / k;
    medoids.push_back(series[medoid_index[m]]);
  }
  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = fp32 ? dtwc::GpuPrecision::FP32 : dtwc::GpuPrecision::FP64;

  // The CPU's assignment, as fast_clara's: a point at 0 from itself, the
  // nearest medoid kept (strict <, lower slot first, a medoid serving itself).
  dtwc::Problem prob("gc");
  if (mode == "cpu") {
    std::vector<std::string> names(N);
    prob.set_data(dtwc::Data(std::vector<std::vector<double>>(series), std::move(names)));
  }
  std::vector<std::size_t> labels(N);
  std::vector<double> best(N);
  const auto assign = [&](std::size_t p, auto distance) {
    double best_d = std::numeric_limits<double>::max();
    std::size_t label = 0;
    for (std::size_t m = 0; m < k; ++m) {
      const double d = p == medoid_index[m] ? 0.0 : distance(m);
      if (m == 0 || d < best_d || (d == best_d && p == medoid_index[m])) { best_d = d; label = m; }
    }
    labels[p] = label;
    best[p] = best_d;
  };

  const auto once = [&]() {
    Run run;
    const auto t0 = std::chrono::steady_clock::now();
    if (mode == "fill32" || mode == "fill64") {
      dtwc::core::DistanceMatrix matrix;
      run.gpu = dtwc::cuda::compute_distance_matrix_cuda(series, opts, matrix).gpu_time_sec;
      mix(run.hash, matrix.raw(), matrix.packed_count() * sizeof(double));
    } else if (mode == "rect32" || mode == "rect64") {
      run.gpu = dtwc::cuda::compute_medoid_distances_cuda(
                    series, medoids, opts,
                    [&](std::size_t first, std::size_t count, std::span<const double> d) {
                      auto task = [&](std::size_t i) {
                        assign(first + i, [&](std::size_t m) { return d[i * k + m]; });
                      };
                      dtwc::run_openmp(task, count);
                    })
                    .gpu_time_sec;
    } else {
      const auto &dtw = prob.dtw_function();
      auto task = [&](std::size_t p) {
        const auto x = prob.series(p);
        assign(p, [&](std::size_t m) { return dtw(x, prob.series(medoid_index[m])); });
      };
      dtwc::run_openmp(task, N);
    }
    run.wall = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (mode.rfind("fill", 0) != 0) {
      mix(run.hash, labels.data(), labels.size() * sizeof(labels[0]));
      mix(run.hash, best.data(), best.size() * sizeof(best[0]));
    }
    return run;
  };

  const bool fill = mode.rfind("fill", 0) == 0;
  const double pairs = fill ? double(N) * double(N - 1) / 2 : double(N) * double(k);
  (void)once(); // warm-up: the CUDA context, this thread's buffers, the CPU's caches
  std::vector<double> wall, gpu;
  std::uint64_t hash = 0;
  for (int r = 0; r < reps; ++r) {
    const auto run = once();
    wall.push_back(run.wall);
    gpu.push_back(run.gpu);
    hash = run.hash;
    std::printf("run,%s,%zu,%zu,%zu,%d,wall %.4f s,gpu %.4f s\n", argv[1], N, L, k, r, run.wall, run.gpu);
  }
  const double t = mode == "cpu" ? median(wall) : median(gpu);
  std::printf("median,%s,N %zu,L %zu,k %zu,wall %.4f s,gpu %.4f s,%.1f kpairs/s,%.1f Gcell/s,"
              "min %.4f,max %.4f,fnv1a64 %016llx\n",
              argv[1], N, L, k, median(wall), median(gpu), pairs / t * 1e-3,
              pairs * double(L) * double(L) / t * 1e-9, mode == "cpu" ? *std::min_element(wall.begin(), wall.end()) : *std::min_element(gpu.begin(), gpu.end()),
              mode == "cpu" ? *std::max_element(wall.begin(), wall.end()) : *std::max_element(gpu.begin(), gpu.end()),
              static_cast<unsigned long long>(hash));
  return 0;
}
