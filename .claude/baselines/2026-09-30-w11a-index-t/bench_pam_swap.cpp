// Scratch PAM swap benchmark for W11a (not in the repo): FasterPAM on a cached
// matrix. Prints, per (N, k), the deterministic counters (sweeps, cost, medoids
// hash) and the wall-clock of each repetition (advisory: shared machine).
// Written so the same source compiles before and after the index_t change.
#include "Problem.hpp"
#include "algorithms/fast_pam.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace {

dtwc::Data make_data(std::size_t n, std::size_t len)
{
  std::vector<std::vector<double>> series(n);
  std::vector<std::string> names(n);
  std::uint64_t s = 0x9E3779B97F4A7C15ull;
  for (std::size_t i = 0; i < n; ++i) {
    const double phase = double(i % 17) * 0.37, freq = 1.0 + double(i % 5) * 0.25;
    series[i].resize(len);
    for (std::size_t t = 0; t < len; ++t) {
      s = s * 6364136223846793005ull + 1442695040888963407ull;
      const double noise = double(s >> 11) * 0x1.0p-53 - 0.5;
      series[i][t] = std::sin(freq * double(t) * 0.3 + phase) + 0.2 * noise;
    }
    names[i] = std::to_string(i);
  }
  return dtwc::Data(std::move(series), std::move(names));
}

} // namespace

int main(int argc, char **argv)
{
  const int reps = argc > 1 ? std::atoi(argv[1]) : 5;
  const std::size_t len = 16;
  const std::size_t Ns[] = { 2000, 4000 };
  const int ks[] = { 10, 50 };
  for (const std::size_t N : Ns) {
    dtwc::Problem prob("bench");
    prob.set_data(make_data(N, len));
    const auto t0 = std::chrono::steady_clock::now();
    prob.fill_distance_matrix();
    const double fill_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    std::printf("N=%zu L=%zu fill %.1f ms\n", N, len, fill_ms);
    for (const int k : ks) {
      std::vector<double> ms;
      for (int r = 0; r < reps; ++r) {
        const auto a = std::chrono::steady_clock::now();
        const auto res = dtwc::fast_pam_seeded(prob, k, 42, 100);
        ms.push_back(
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - a).count());
        if (r == 0) {
          std::uint64_t h = 1469598103934665603ull;
          for (const auto m : res.medoid_indices) h = (h ^ std::uint64_t(m)) * 1099511628211ull;
          for (const auto l : res.labels) h = (h ^ std::uint64_t(l)) * 1099511628211ull;
          std::printf("  k=%d sweeps=%d converged=%d cost=%.17g hash=%016llx\n", k,
                      int(res.iterations), int(res.converged), res.total_cost,
                      static_cast<unsigned long long>(h));
        }
      }
      std::sort(ms.begin(), ms.end());
      std::printf("  k=%d ms: min %.1f median %.1f max %.1f (reps %d)\n", k, ms.front(),
                  ms[ms.size() / 2], ms.back(), reps);
    }
  }
}
