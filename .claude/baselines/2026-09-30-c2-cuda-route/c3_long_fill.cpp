// C1: Problem::fill_distance_matrix of the same N x L series on the GPU (gpu: precision Auto;
// gpu64: FP64) or the CPU, one warm-up fill on the GPU, then REPS timed fills.
// Usage: long_fill <gpu|gpu64|cpu> <N> <L> <REPS> [sql2]  (C3: sql2 = the squared L2 metric).
// Series: bench_cuda_dtw's benchmark_series_set(N, L, 200). Scratch tool; not part of the library.
#include <dtwc.hpp>

#include "support/deterministic_series.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

int main(int argc, char **argv)
{
  if (argc != 5 && !(argc == 6 && std::string(argv[5]) == "sql2")) {
    std::fprintf(stderr, "usage: long_fill <gpu|gpu64|cpu> <N> <L> <REPS> [sql2]\n");
    return 1;
  }
  const std::string device = argv[1];
  const bool gpu = device == "gpu" || device == "gpu64";
  const std::size_t N = std::strtoull(argv[2], nullptr, 10);
  const std::size_t L = std::strtoull(argv[3], nullptr, 10);
  const int reps = std::atoi(argv[4]);

  auto series = dtwc::test_support::benchmark_series_set(N, L, 200u);
  std::vector<std::string> names;
  for (std::size_t i = 0; i < N; ++i) names.push_back("s" + std::to_string(i));
  dtwc::Problem prob("c1");
  prob.set_data(dtwc::Data(std::move(series), std::move(names)));
  if (argc == 6) prob.set_metric(dtwc::core::MetricType::SquaredL2);
  if (gpu) {
    prob.set_device(dtwc::Device::GPU);
    if (device == "gpu64") prob.set_cuda_settings({ 0, dtwc::GpuPrecision::FP64 });
    prob.fill_distance_matrix(); // warm-up: the CUDA context and this thread's buffers
  }

  const double cells = double(N) * double(N - 1) / 2 * double(L) * double(L);
  std::vector<double> seconds;
  for (int r = 0; r < reps; ++r) {
    prob.refresh_distance_matrix();
    const auto t0 = std::chrono::steady_clock::now();
    prob.fill_distance_matrix();
    seconds.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    std::printf("run,%s,%zu,%zu,%d,%.4f\n", argv[1], N, L, r, seconds.back());
  }
  std::sort(seconds.begin(), seconds.end());
  const double median = seconds[seconds.size() / 2];
  const auto &m = std::as_const(prob).distance_matrix();
  std::printf("median,%s,%zu,%zu,%.4f s,%.1f Gcell/s,min %.4f,max %.4f,d(0,1)=%.17g,d(N-1,N-2)=%.17g\n",
              argv[1], N, L, median, cells / median * 1e-9, seconds.front(), seconds.back(),
              m.get(0, 1), m.get(N - 1, N - 2));
  // C3: FNV-1a 64 over the bytes of every packed slot, so two builds' matrices
  // compare whole (after the timed fills; the timing above is C1's unchanged).
  std::uint64_t hash = 14695981039346656037ull;
  const auto *bytes = reinterpret_cast<const unsigned char *>(m.raw());
  for (std::size_t b = 0; b < m.packed_count() * sizeof(double); ++b)
    hash = (hash ^ bytes[b]) * 1099511628211ull;
  std::printf("matrix,%s,%zu,%zu,slots %zu,fnv1a64 %016llx\n", argv[1], N, L, m.packed_count(),
              static_cast<unsigned long long>(hash));
  return 0;
}
