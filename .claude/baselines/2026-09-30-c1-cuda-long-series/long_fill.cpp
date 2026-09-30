// C1: Problem::fill_distance_matrix of the same N x L series on the GPU (precision Auto) or the
// CPU, one warm-up fill on the GPU, then REPS timed fills. Usage: long_fill <gpu|cpu> <N> <L> <REPS>.
// Series: bench_cuda_dtw's benchmark_series_set(N, L, 200). Scratch tool; not part of the library.
#include <dtwc.hpp>

#include "support/deterministic_series.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

int main(int argc, char **argv)
{
  if (argc != 5) {
    std::fprintf(stderr, "usage: long_fill <gpu|cpu> <N> <L> <REPS>\n");
    return 1;
  }
  const bool gpu = std::string(argv[1]) == "gpu";
  const std::size_t N = std::strtoull(argv[2], nullptr, 10);
  const std::size_t L = std::strtoull(argv[3], nullptr, 10);
  const int reps = std::atoi(argv[4]);

  auto series = dtwc::test_support::benchmark_series_set(N, L, 200u);
  std::vector<std::string> names;
  for (std::size_t i = 0; i < N; ++i) names.push_back("s" + std::to_string(i));
  dtwc::Problem prob("c1");
  prob.set_data(dtwc::Data(std::move(series), std::move(names)));
  if (gpu) {
    prob.set_device(dtwc::Device::GPU);
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
  return 0;
}
