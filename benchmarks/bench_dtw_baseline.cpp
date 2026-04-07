/**
 * @file bench_dtw_baseline.cpp
 * @brief Baseline microbenchmarks for DTW distance computations.
 *
 * @details Captures performance of dtwFull, dtwFull_L, dtwBanded, and
 *          fillDistanceMatrix before any optimisation work begins.
 *          Uses Google Benchmark with deterministic random data (fixed seeds).
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include <benchmark/benchmark.h>
#include <dtwc.hpp>
#include <core/lower_bound_impl.hpp>
#include <core/z_normalize.hpp>

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

static void configure_problem_variant(dtwc::Problem &prob, int variant_code)
{
  dtwc::core::DTWVariantParams params;
  switch (variant_code) {
  case 1:
    params.variant = dtwc::core::DTWVariant::WDTW;
    params.wdtw_g = 0.05;
    break;
  case 2:
    params.variant = dtwc::core::DTWVariant::DDTW;
    break;
  case 3:
    params.variant = dtwc::core::DTWVariant::ADTW;
    params.adtw_penalty = 1.0;
    break;
  case 0:
  default:
    params.variant = dtwc::core::DTWVariant::Standard;
    break;
  }
  prob.set_variant(params);
}

static const char *variant_name(int variant_code)
{
  switch (variant_code) {
  case 1: return "WDTW";
  case 2: return "DDTW";
  case 3: return "ADTW";
  case 0:
  default: return "Standard";
  }
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
// BM_wdtwBanded_g — WDTW banded path that recomputes weights every call
// ---------------------------------------------------------------------------
static void BM_wdtwBanded_g(benchmark::State &state)
{
  const auto len = static_cast<size_t>(state.range(0));
  const int band = static_cast<int>(state.range(1));
  constexpr double g = 0.05;
  auto x = random_series(len, 42);
  auto y = random_series(len, 43);
  for (auto _ : state) {
    benchmark::DoNotOptimize(dtwc::wdtwBanded<double>(x, y, band, g));
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(BM_wdtwBanded_g)
  ->Args({1000, 10})
  ->Args({1000, 50})
  ->Args({4000, 50})
  ->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM_wdtwBanded_precomputed — WDTW banded path with reused weight vector
// ---------------------------------------------------------------------------
static void BM_wdtwBanded_precomputed(benchmark::State &state)
{
  const auto len = static_cast<size_t>(state.range(0));
  const int band = static_cast<int>(state.range(1));
  constexpr double g = 0.05;
  auto x = random_series(len, 42);
  auto y = random_series(len, 43);
  const int max_dev = static_cast<int>(std::max(x.size(), y.size()));
  const auto weights = dtwc::wdtw_weights<double>(max_dev, g);
  for (auto _ : state) {
    benchmark::DoNotOptimize(dtwc::wdtwBanded<double>(x, y, weights, band));
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(BM_wdtwBanded_precomputed)
  ->Args({1000, 10})
  ->Args({1000, 50})
  ->Args({4000, 50})
  ->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM_ddtwBanded — DDTW banded path including derivative preprocessing
// ---------------------------------------------------------------------------
static void BM_ddtwBanded(benchmark::State &state)
{
  const auto len = static_cast<size_t>(state.range(0));
  const int band = static_cast<int>(state.range(1));
  auto x = random_series(len, 42);
  auto y = random_series(len, 43);
  for (auto _ : state) {
    benchmark::DoNotOptimize(dtwc::ddtwBanded<double>(x, y, band));
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(BM_ddtwBanded)
  ->Args({1000, 10})
  ->Args({1000, 50})
  ->Args({4000, 50})
  ->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM_adtwBanded — ADTW banded path
// ---------------------------------------------------------------------------
static void BM_adtwBanded(benchmark::State &state)
{
  const auto len = static_cast<size_t>(state.range(0));
  const int band = static_cast<int>(state.range(1));
  constexpr double penalty = 1.0;
  auto x = random_series(len, 42);
  auto y = random_series(len, 43);
  for (auto _ : state) {
    benchmark::DoNotOptimize(dtwc::adtwBanded<double>(x, y, band, penalty));
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(BM_adtwBanded)
  ->Args({1000, 10})
  ->Args({1000, 50})
  ->Args({4000, 50})
  ->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM_wdtwBanded_mv_ndim1 — MV WDTW ndim==1 scalar fallback path
// ---------------------------------------------------------------------------
static void BM_wdtwBanded_mv_ndim1(benchmark::State &state)
{
  const auto len = static_cast<size_t>(state.range(0));
  const int band = static_cast<int>(state.range(1));
  constexpr double g = 0.05;
  auto x = random_series(len, 42);
  auto y = random_series(len, 43);
  for (auto _ : state) {
    benchmark::DoNotOptimize(
      dtwc::wdtwBanded_mv<double>(x.data(), x.size(), y.data(), y.size(), 1, band, g));
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(BM_wdtwBanded_mv_ndim1)
  ->Args({1000, 10})
  ->Args({1000, 50})
  ->Args({4000, 50})
  ->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM_adtwBanded_mv_ndim1 — MV ADTW ndim==1 scalar fallback path
// ---------------------------------------------------------------------------
static void BM_adtwBanded_mv_ndim1(benchmark::State &state)
{
  const auto len = static_cast<size_t>(state.range(0));
  const int band = static_cast<int>(state.range(1));
  constexpr double penalty = 1.0;
  auto x = random_series(len, 42);
  auto y = random_series(len, 43);
  for (auto _ : state) {
    benchmark::DoNotOptimize(
      dtwc::adtwBanded_mv<double>(x.data(), x.size(), y.data(), y.size(), 1, band, penalty));
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(BM_adtwBanded_mv_ndim1)
  ->Args({1000, 10})
  ->Args({1000, 50})
  ->Args({4000, 50})
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
  ->Args({100, 100, -1})
  ->Args({20, 500, -1})
  ->Args({50, 500, -1})
  ->Args({100, 500, -1})
  ->Args({50, 500, 10})
  ->Args({50, 500, 50})
  ->Args({50, 1000, 50})
  ->Args({100, 1000, -1})
  ->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// BM_fillDistanceMatrix_variant — end-to-end Problem path by variant
// Args: (N_series, series_length, band, variant_code)
// ---------------------------------------------------------------------------
static void BM_fillDistanceMatrix_variant(benchmark::State &state)
{
  const int N = static_cast<int>(state.range(0));
  const int L = static_cast<int>(state.range(1));
  const int band = static_cast<int>(state.range(2));
  const int variant = static_cast<int>(state.range(3));
  state.SetLabel(variant_name(variant));

  for (auto _ : state) {
    state.PauseTiming();
    dtwc::Problem prob("bench_variant");
    prob.set_data(make_random_data(N, L));
    prob.band = band;
    configure_problem_variant(prob, variant);
    state.ResumeTiming();

    prob.fillDistanceMatrix();
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations())
                          * static_cast<int64_t>(N) * (N - 1) / 2);
}
BENCHMARK(BM_fillDistanceMatrix_variant)
  ->Args({50, 500, 50, 0})
  ->Args({50, 500, 50, 1})
  ->Args({50, 500, 50, 2})
  ->Args({50, 500, 50, 3})
  ->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// BM_lb_keogh — LB_Keogh lower bound for varying lengths
// ---------------------------------------------------------------------------
static void BM_lb_keogh(benchmark::State &state)
{
  const auto len = static_cast<size_t>(state.range(0));
  auto query = random_series(len, 42);
  auto candidate = random_series(len, 43);

  std::vector<double> upper(len), lower(len);
  dtwc::core::compute_envelopes(candidate.data(), len, 10, upper.data(), lower.data());

  for (auto _ : state) {
    benchmark::DoNotOptimize(
      dtwc::core::lb_keogh(query.data(), len, upper.data(), lower.data()));
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
  state.SetComplexityN(static_cast<int64_t>(len));
}
BENCHMARK(BM_lb_keogh)
  ->Arg(100)
  ->Arg(500)
  ->Arg(1000)
  ->Arg(4000)
  ->Arg(8000)
  ->Unit(benchmark::kNanosecond)
  ->Complexity();

// ---------------------------------------------------------------------------
// BM_z_normalize — z-normalization for varying lengths
// ---------------------------------------------------------------------------
static void BM_z_normalize(benchmark::State &state)
{
  const auto len = static_cast<size_t>(state.range(0));
  auto series_data = random_series(len, 42);

  for (auto _ : state) {
    state.PauseTiming();
    auto copy = series_data; // fresh copy each iteration
    state.ResumeTiming();
    dtwc::core::z_normalize(copy.data(), copy.size());
    benchmark::DoNotOptimize(copy.data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
  state.SetComplexityN(static_cast<int64_t>(len));
}
BENCHMARK(BM_z_normalize)
  ->Arg(100)
  ->Arg(500)
  ->Arg(1000)
  ->Arg(4000)
  ->Arg(8000)
  ->Unit(benchmark::kNanosecond)
  ->Complexity();

// ---------------------------------------------------------------------------
// BM_compute_envelopes — envelope computation for varying lengths
// ---------------------------------------------------------------------------
static void BM_compute_envelopes(benchmark::State &state)
{
  const auto len = static_cast<size_t>(state.range(0));
  const int band = static_cast<int>(state.range(1));
  auto series = random_series(len, 42);
  std::vector<double> upper(len), lower(len);

  for (auto _ : state) {
    dtwc::core::compute_envelopes(series.data(), len, band, upper.data(), lower.data());
    benchmark::DoNotOptimize(upper.data());
    benchmark::DoNotOptimize(lower.data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(BM_compute_envelopes)
  ->Args({1000, 10})
  ->Args({1000, 50})
  ->Args({4000, 50})
  ->Unit(benchmark::kNanosecond);

// ---------------------------------------------------------------------------
// BM_dtwFull_L_SquaredL2 — linear-space DTW with SquaredL2 metric
// ---------------------------------------------------------------------------
static void BM_dtwFull_L_SquaredL2(benchmark::State &state)
{
  const auto len = static_cast<size_t>(state.range(0));
  auto x = random_series(len, 42);
  auto y = random_series(len, 43);
  for (auto _ : state) {
    benchmark::DoNotOptimize(
      dtwc::dtwFull_L<double>(x.data(), x.size(), y.data(), y.size(), -1.0,
                              dtwc::core::MetricType::SquaredL2));
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
  state.SetComplexityN(static_cast<int64_t>(len));
}
BENCHMARK(BM_dtwFull_L_SquaredL2)
  ->Arg(100)
  ->Arg(500)
  ->Arg(1000)
  ->Arg(4000)
  ->Unit(benchmark::kMicrosecond)
  ->Complexity();

// ---------------------------------------------------------------------------
// BM_dtwBanded_earlyAbandon — banded DTW with and without early abandon
// ---------------------------------------------------------------------------
static void BM_dtwBanded_earlyAbandon(benchmark::State &state)
{
  const auto len = static_cast<size_t>(state.range(0));
  const int band = static_cast<int>(state.range(1));
  auto x = random_series(len, 42);
  auto y = random_series(len, 43);
  // Compute a tight upper bound to make early abandon effective
  double ub = dtwc::dtwBanded<double>(x, y, band);
  for (auto _ : state) {
    benchmark::DoNotOptimize(dtwc::dtwBanded<double>(x, y, band, ub * 1.01));
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
}
BENCHMARK(BM_dtwBanded_earlyAbandon)
  ->Args({1000, 50})
  ->Args({4000, 50})
  ->Args({4000, 100})
  ->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// BM_dtwFull_L_mv_ndim2 — multivariate DTW ndim=2
// ---------------------------------------------------------------------------
static void BM_dtwFull_L_mv_ndim2(benchmark::State &state)
{
  const auto steps = static_cast<size_t>(state.range(0));
  auto x = random_series(steps * 2, 42);
  auto y = random_series(steps * 2, 43);
  for (auto _ : state) {
    benchmark::DoNotOptimize(
      dtwc::dtwFull_L_mv<double>(x.data(), steps, y.data(), steps, 2));
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()));
  state.SetComplexityN(static_cast<int64_t>(steps));
}
BENCHMARK(BM_dtwFull_L_mv_ndim2)
  ->Arg(100)
  ->Arg(500)
  ->Arg(1000)
  ->Arg(4000)
  ->Unit(benchmark::kMicrosecond)
  ->Complexity();

