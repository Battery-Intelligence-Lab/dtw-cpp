/**
 * @file bench_dtw_baseline.cpp
<<<<<<< HEAD
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
=======
 * @brief Baseline microbenchmarks for DTW distance computation.
 *
 * @details Provides benchmarks for dtwFull, dtwFull_L, and dtwBanded,
 * plus a roofline-aware benchmark that reports cells/sec, FLOP/sec,
 * and bytes/sec to help characterize whether DTW is compute- or
 * memory-bound at various series lengths.
 *
 * Build with:
 *   cmake .. -DCMAKE_BUILD_TYPE=Release -DDTWC_BUILD_BENCHMARK=ON
 *   cmake --build . --target bench_dtw_baseline
 *
 * Run with:
 *   ./bench_dtw_baseline --benchmark_format=console
 */

#include <benchmark/benchmark.h>
#include "../dtwc/warping.hpp"

#include <vector>
#include <random>
#include <cstddef>

namespace {

/// Generate a random time series of length n using a given seed.
std::vector<double> random_series(std::size_t n, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> v(n);
  for (auto &val : v)
    val = dist(rng);
  return v;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// BM_dtwFull: full cost-matrix DTW
// ---------------------------------------------------------------------------
static void BM_dtwFull(benchmark::State &state)
{
  const auto n = static_cast<std::size_t>(state.range(0));
  auto x = random_series(n, 42);
  auto y = random_series(n, 43);

>>>>>>> e59a35c (Fix documentation errors and add roofline-aware benchmarks)
  for (auto _ : state) {
    benchmark::DoNotOptimize(dtwc::dtwFull<double>(x, y));
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
<<<<<<< HEAD
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
=======
  state.counters["series_len"] = benchmark::Counter(
      static_cast<double>(n), benchmark::Counter::kDefaults);
}
BENCHMARK(BM_dtwFull)->Arg(50)->Arg(100)->Arg(200)->Arg(500)->Arg(1000);

// ---------------------------------------------------------------------------
// BM_dtwFull_L: memory-efficient (light) DTW
// ---------------------------------------------------------------------------
static void BM_dtwFull_L(benchmark::State &state)
{
  const auto n = static_cast<std::size_t>(state.range(0));
  auto x = random_series(n, 42);
  auto y = random_series(n, 43);

>>>>>>> e59a35c (Fix documentation errors and add roofline-aware benchmarks)
  for (auto _ : state) {
    benchmark::DoNotOptimize(dtwc::dtwFull_L<double>(x, y));
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
<<<<<<< HEAD
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
=======
  state.counters["series_len"] = benchmark::Counter(
      static_cast<double>(n), benchmark::Counter::kDefaults);
}
BENCHMARK(BM_dtwFull_L)->Arg(50)->Arg(100)->Arg(200)->Arg(500)->Arg(1000)->Arg(2000);

// ---------------------------------------------------------------------------
// BM_dtwBanded: banded DTW with Sakoe-Chiba constraint
// ---------------------------------------------------------------------------
static void BM_dtwBanded(benchmark::State &state)
{
  const auto n = static_cast<std::size_t>(state.range(0));
  const auto band = static_cast<int>(state.range(1));
  auto x = random_series(n, 42);
  auto y = random_series(n, 43);

>>>>>>> e59a35c (Fix documentation errors and add roofline-aware benchmarks)
  for (auto _ : state) {
    benchmark::DoNotOptimize(dtwc::dtwBanded<double>(x, y, band));
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
<<<<<<< HEAD
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
=======
  state.counters["series_len"] = benchmark::Counter(
      static_cast<double>(n), benchmark::Counter::kDefaults);
  state.counters["band"] = benchmark::Counter(
      static_cast<double>(band), benchmark::Counter::kDefaults);
}
BENCHMARK(BM_dtwBanded)
    ->Args({500, 10})
    ->Args({500, 50})
    ->Args({1000, 10})
    ->Args({1000, 50})
    ->Args({1000, 100})
    ->Args({2000, 50});

// ---------------------------------------------------------------------------
// BM_dtw_roofline: measure bytes/sec and FLOP/sec to characterize bottleneck
// ---------------------------------------------------------------------------
static void BM_dtw_roofline(benchmark::State &state)
{
  const auto n = static_cast<std::size_t>(state.range(0));
  auto x = random_series(n, 42);
  auto y = random_series(n, 43);

  for (auto _ : state) {
    benchmark::DoNotOptimize(dtwc::dtwFull_L<double>(x, y));
  }

  // DTW inner loop per cell: 1 subtract, 1 abs, 2 min comparisons, 1 add = ~5 FLOPs
  // Memory per cell: read 3 neighbors (24 bytes) + 2 series values (16 bytes) = ~40 bytes
  const double cells = static_cast<double>(n) * static_cast<double>(n);
  state.counters["cells"] = benchmark::Counter(
      cells, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["FLOP"] = benchmark::Counter(
      5.0 * cells, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["bytes"] = benchmark::Counter(
      40.0 * cells, benchmark::Counter::kIsIterationInvariantRate);
}
BENCHMARK(BM_dtw_roofline)->Arg(100)->Arg(500)->Arg(1000)->Arg(2000);

BENCHMARK_MAIN();
>>>>>>> e59a35c (Fix documentation errors and add roofline-aware benchmarks)
