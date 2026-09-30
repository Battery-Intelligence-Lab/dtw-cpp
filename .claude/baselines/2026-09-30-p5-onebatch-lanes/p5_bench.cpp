// Scratch (never committed): one OneBatchPAM call per round on seeded random walks.
//   p5_bench <N> <k> <L> <f32:0|1> <metric:0=L1|1=SqL2> <oddfrac_eighths:0..8> <band> <batch> <rounds>
// Prints, per round: whole-call seconds, hash of labels+medoids+cost bits, evals. The instrumented
// library prints the table fill seconds and the table hash on stderr.
#include "Problem.hpp"
#include "algorithms/one_batch_pam.hpp"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

template <typename T>
static dtwc::Data walks(int n, int len, int odd_eighths, std::uint64_t seed)
{
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> step;
  std::vector<std::vector<T>> rows;
  std::vector<std::string> names;
  for (int i = 0; i < n; ++i) {
    const int li = (int(rng() % 8) < odd_eighths) ? len + 3 : len;
    std::vector<T> v(li);
    double w = 0;
    for (auto &e : v) e = T(w += step(rng));
    rows.push_back(std::move(v));
    names.push_back("s" + std::to_string(i));
  }
  return dtwc::Data(std::move(rows), std::move(names));
}

int main(int argc, char **argv)
{
  if (argc < 10) return 2;
  const int N = std::atoi(argv[1]), k = std::atoi(argv[2]), L = std::atoi(argv[3]);
  const bool f32 = std::atoi(argv[4]) != 0;
  const int metric = std::atoi(argv[5]), odd = std::atoi(argv[6]), band = std::atoi(argv[7]);
  const int batch = std::atoi(argv[8]), rounds = std::atoi(argv[9]);
  for (int r = 0; r < rounds; ++r) {
    dtwc::Problem prob("p5");
    prob.set_data(f32 ? walks<float>(N, L, odd, 20260930) : walks<double>(N, L, odd, 20260930));
    prob.set_band(band);
    prob.set_metric(metric ? dtwc::core::MetricType::SquaredL2 : dtwc::core::MetricType::L1);
    dtwc::algorithms::OneBatchPAMOptions o;
    o.n_clusters = k;
    o.batch_size = batch;
    dtwc::algorithms::OneBatchPAMStats st;
    const auto t0 = std::chrono::steady_clock::now();
    const auto res = dtwc::algorithms::one_batch_pam(prob, o, &st);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::uint64_t h = 1469598103934665603ull;
    auto mix = [&](const void *p, std::size_t n) {
      for (std::size_t i = 0; i < n; ++i) { h ^= static_cast<const unsigned char *>(p)[i]; h *= 1099511628211ull; }
    };
    for (auto l : res.labels) mix(&l, sizeof l);
    for (auto m : res.medoid_indices) mix(&m, sizeof m);
    mix(&res.total_cost, sizeof res.total_cost);
    mix(&st.estimated_objective, sizeof st.estimated_objective);
    std::printf("call_s=%.6f hash=%016llx cost=%.17g evals=%llu m=%zu\n", s, (unsigned long long)h,
                res.total_cost, (unsigned long long)st.distance_evaluations, st.batch_size);
    std::fflush(stdout);
  }
  return 0;
}
