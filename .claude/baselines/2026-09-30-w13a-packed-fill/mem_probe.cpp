// Peak device and host memory of one Problem::fill_distance_matrix on device gpu.
// Usage: mem_probe <N> <L> <fp64 0|1>. Scratch tool for W13a's memory figure; not part of the library.
#include <dtwc.hpp>

#include <cuda_runtime.h>
#define NOMINMAX
#include <windows.h>
#include <psapi.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <thread>
#include <vector>

static PROCESS_MEMORY_COUNTERS_EX host_counters()
{
  PROCESS_MEMORY_COUNTERS_EX c{};
  GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS *>(&c), sizeof(c));
  return c;
}

int main(int argc, char **argv)
{
  const size_t N = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 20000;
  const size_t L = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 1000;
  const bool fp64 = argc > 3 && std::atoi(argv[3]) != 0;
  const double MB = 1024.0 * 1024.0;

  std::vector<std::vector<double>> series(N, std::vector<double>(L));
  std::mt19937_64 rng(42);
  std::normal_distribution<double> step(0.0, 1.0);
  for (auto &s : series) {
    double acc = 0.0;
    for (auto &v : s) { acc += step(rng); v = acc; }
  }
  std::vector<std::string> names(N);
  for (size_t i = 0; i < N; ++i) names[i] = "s" + std::to_string(i);

  dtwc::Problem prob("mem_probe");
  prob.set_data(dtwc::Data(std::move(series), std::move(names)));
  prob.set_device(dtwc::Device::GPU);
  prob.set_cuda_settings({ 0, fp64 ? dtwc::GpuPrecision::FP64 : dtwc::GpuPrecision::FP32 });

  cudaFree(nullptr); // create the context before the baseline reading
  size_t free0 = 0, total = 0;
  cudaMemGetInfo(&free0, &total);
  const auto before = host_counters();

  std::atomic<bool> done{ false };
  size_t min_free = free0;
  std::thread sampler([&] {
    while (!done.load()) {
      size_t f = 0, t = 0;
      if (cudaMemGetInfo(&f, &t) == cudaSuccess) min_free = std::min(min_free, f);
      std::this_thread::sleep_for(std::chrono::milliseconds(2));
    }
  });
  const auto t0 = std::chrono::steady_clock::now();
  prob.fill_distance_matrix();
  const auto t1 = std::chrono::steady_clock::now();
  done = true;
  sampler.join();
  const auto after = host_counters();

  std::printf("N=%zu L=%zu precision=%s fill_s=%.3f\n", N, L, fp64 ? "FP64" : "FP32",
              std::chrono::duration<double>(t1 - t0).count());
  std::printf("device: total_MB=%.1f free_before_MB=%.1f min_free_during_MB=%.1f peak_fill_MB=%.1f\n",
              total / MB, free0 / MB, min_free / MB, (free0 - min_free) / MB);
  std::printf("host: private_before_MB=%.1f peak_private_MB=%.1f peak_private_minus_before_MB=%.1f "
              "working_set_before_MB=%.1f peak_working_set_MB=%.1f\n",
              before.PrivateUsage / MB, after.PeakPagefileUsage / MB,
              (after.PeakPagefileUsage - before.PrivateUsage) / MB, before.WorkingSetSize / MB,
              after.PeakWorkingSetSize / MB);
  std::printf("check: d(0,1)=%.17g d(N-1,N-2)=%.17g\n", prob.dist_by_ind(0, 1),
              prob.dist_by_ind(static_cast<int>(N - 1), static_cast<int>(N - 2)));
  return 0;
}
