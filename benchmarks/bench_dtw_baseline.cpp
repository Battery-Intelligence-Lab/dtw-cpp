/**
 * @file bench_dtw_baseline.cpp
 * @brief Baseline microbenchmarks for DTW distance computations.
 *
 * @details Captures performance of dtwFull, dtwFull_L, dtwBanded, and
 *          fillDistanceMatrix before any optimisation work begins.
 *          Uses Google Benchmark with deterministic random data (fixed seeds).
 *
 * @date 28 Mar 2026
 */

#include <benchmark/benchmark.h>
#include <dtwc.hpp>

#include <vector>
#include <random>
#include <string>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a random time series of a given length using a fixed seed.
static std::vector<double> random_series(size_t len, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> s(len);
  for (auto &v : s)
    v = dist(rng);
  return s;
}

/// Build a dtwc::Data object with N random series of length L.
static dtwc::Data make_random_data(int N, int L, unsigned base_seed = 100)
{
  std::vector<std::vector<dtwc::data_t>> vecs;
  std::vector<std::string> names;
  vecs.reserve(N);
  names.reserve(N);
  for (int i = 0; i < N; ++i) {
    vecs.push_back(random_series(static_cast<size_t>(L), base_seed + i));
    names.push_back("s" + std::to_string(i));
  }
  return dtwc::Data(std::move(vecs), std::move(names));
}

// ---------------------------------------------------------------------------
// BM_dtwFull — full O(n*m) DTW for varying series lengths
// ---------------------------------------------------------------------------
static void BM_dtwFull(benchmark::State &state)
{
  const auto len = static_cast<size_t>(state.range(0));
  auto x = random_series(len, 42);
  auto y = random_series(len, 43);
  for (auto _ : state) {
    benchmark::DoNotOptimize(dtwc::dtwFull<double>(x, y));
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
  state.SetComplexityN(static_cast<int64_t>(len));
}
BENCHMARK(BM_dtwFull)
  ->Arg(100)
  ->Arg(500)
  ->Arg(1000)
  ->Arg(4000)
  ->Unit(benchmark::kMicrosecond)
  ->Complexity();

// ---------------------------------------------------------------------------
// BM_dtwFull_L — linear-space (light) DTW
// ---------------------------------------------------------------------------
static void BM_dtwFull_L(benchmark::State &state)
{
  const auto len = static_cast<size_t>(state.range(0));
  auto x = random_series(len, 42);
  auto y = random_series(len, 43);
  for (auto _ : state) {
    benchmark::DoNotOptimize(dtwc::dtwFull_L<double>(x, y));
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
  state.SetComplexityN(static_cast<int64_t>(len));
}
BENCHMARK(BM_dtwFull_L)
  ->Arg(100)
  ->Arg(500)
  ->Arg(1000)
  ->Arg(4000)
  ->Arg(8000)
  ->Unit(benchmark::kMicrosecond)
  ->Complexity();

// ---------------------------------------------------------------------------
// BM_dtwBanded — Sakoe-Chiba banded DTW with varying band widths
// ---------------------------------------------------------------------------
static void BM_dtwBanded(benchmark::State &state)
{
  const auto len = static_cast<size_t>(state.range(0));
  const int band = static_cast<int>(state.range(1));
  auto x = random_series(len, 42);
  auto y = random_series(len, 43);
  for (auto _ : state) {
    benchmark::DoNotOptimize(dtwc::dtwBanded<double>(x, y, band));
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
  state.SetComplexityN(static_cast<int64_t>(len));
}
BENCHMARK(BM_dtwBanded)
  ->Args({1000, 10})
  ->Args({1000, 50})
  ->Args({1000, 100})
  ->Args({4000, 50})
  ->Args({4000, 100})
  ->Unit(benchmark::kMicrosecond)
  ->Complexity();

// ---------------------------------------------------------------------------
// BM_dtwBanded_fullFallback — banded with band=-1 (falls back to full)
// ---------------------------------------------------------------------------
static void BM_dtwBanded_fullFallback(benchmark::State &state)
{
  const auto len = static_cast<size_t>(state.range(0));
  auto x = random_series(len, 42);
  auto y = random_series(len, 43);
  for (auto _ : state) {
    benchmark::DoNotOptimize(dtwc::dtwBanded<double>(x, y, -1));
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(BM_dtwBanded_fullFallback)
  ->Arg(500)
  ->Arg(1000)
  ->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM_fillDistanceMatrix — end-to-end distance matrix build
// Args: (N_series, series_length, band)  where band=-1 means full DTW
// ---------------------------------------------------------------------------
static void BM_fillDistanceMatrix(benchmark::State &state)
{
  const int N = static_cast<int>(state.range(0));
  const int L = static_cast<int>(state.range(1));
  const int band = static_cast<int>(state.range(2));

  for (auto _ : state) {
    state.PauseTiming();
    dtwc::Problem prob("bench");
    prob.set_data(make_random_data(N, L));
    prob.band = band;
    state.ResumeTiming();

    prob.fillDistanceMatrix();
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations())
                          * static_cast<int64_t>(N) * (N - 1) / 2);
}
BENCHMARK(BM_fillDistanceMatrix)
  ->Args({20, 100, -1})
  ->Args({50, 100, -1})
  ->Args({20, 500, -1})
  ->Args({50, 500, -1})
  ->Args({50, 500, 10})
  ->Args({50, 500, 50})
  ->Args({50, 1000, 50})
  ->Unit(benchmark::kMillisecond);
