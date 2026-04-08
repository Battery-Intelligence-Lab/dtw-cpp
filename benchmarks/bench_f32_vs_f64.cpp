/**
 * @file bench_f32_vs_f64.cpp
 * @brief Benchmark: float32 vs float64 DTW computation speed.
 */

#include <dtwc.hpp>

#include <benchmark/benchmark.h>

#include <random>
#include <vector>

namespace {

template <typename T>
std::vector<std::vector<T>> make_random_series(int n, int len, unsigned seed = 42)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<T> dist(T(0), T(10));
  std::vector<std::vector<T>> out(n);
  for (auto &s : out) {
    s.resize(len);
    for (auto &v : s) v = dist(rng);
  }
  return out;
}

// Full DTW (no band)
template <typename T>
static void BM_dtw_full(benchmark::State &state)
{
  const int len = static_cast<int>(state.range(0));
  auto series = make_random_series<T>(200, len);
  int pair = 0;
  for (auto _ : state) {
    int i = (pair * 2) % 200;
    int j = (pair * 2 + 1) % 200;
    auto d = dtwc::dtwFull<T>(series[i], series[j]);
    benchmark::DoNotOptimize(d);
    ++pair;
  }
  state.SetItemsProcessed(state.iterations());
}

// Banded DTW
template <typename T>
static void BM_dtw_banded(benchmark::State &state)
{
  const int len = static_cast<int>(state.range(0));
  const int band = static_cast<int>(state.range(1));
  auto series = make_random_series<T>(200, len);
  int pair = 0;
  for (auto _ : state) {
    int i = (pair * 2) % 200;
    int j = (pair * 2 + 1) % 200;
    auto d = dtwc::dtwBanded<T>(series[i], series[j], band);
    benchmark::DoNotOptimize(d);
    ++pair;
  }
  state.SetItemsProcessed(state.iterations());
}

// Full DTW
BENCHMARK(BM_dtw_full<double>)->Arg(500)->Arg(2000)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_dtw_full<float>)->Arg(500)->Arg(2000)->Unit(benchmark::kMillisecond);

// Banded DTW
BENCHMARK(BM_dtw_banded<double>)->Args({500, 50})->Args({2000, 200})->Args({8000, 500})->Unit(benchmark::kMillisecond);
BENCHMARK(BM_dtw_banded<float>)->Args({500, 50})->Args({2000, 200})->Args({8000, 500})->Unit(benchmark::kMillisecond);

} // namespace

BENCHMARK_MAIN();
