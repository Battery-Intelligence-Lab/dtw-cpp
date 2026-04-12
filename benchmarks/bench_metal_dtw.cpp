/**
 * @file bench_metal_dtw.cpp
 * @brief Metal GPU benchmarks for the DTW distance matrix on Apple Silicon.
 *
 * @details Mirrors bench_cuda_dtw.cpp but targets the Metal backend. Measures
 *          GPU throughput and compares against the multi-threaded CPU path
 *          so we can publish Mac Studio / M-series numbers.
 *
 * @date 2026-04-12
 */

#include <benchmark/benchmark.h>
#include <dtwc.hpp>

#ifdef DTWC_HAS_METAL
#include <metal/metal_dtw.hpp>
#endif

#include <limits>
#include <random>
#include <string>
#include <vector>

static std::vector<double> random_series(size_t len, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> s(len);
  for (auto &v : s) v = dist(rng);
  return s;
}

static std::vector<std::vector<double>> make_series_set(int N, int L,
                                                        unsigned base_seed = 300)
{
  std::vector<std::vector<double>> vecs;
  vecs.reserve(N);
  for (int i = 0; i < N; ++i)
    vecs.push_back(random_series(static_cast<size_t>(L), base_seed + i));
  return vecs;
}

#ifdef DTWC_HAS_METAL

static void BM_metal_distanceMatrix(benchmark::State &state)
{
  const int N = static_cast<int>(state.range(0));
  const int L = static_cast<int>(state.range(1));
  auto series = make_series_set(N, L);

  // Warm-up (first call lazy-compiles the MSL library).
  dtwc::metal::MetalDistMatOptions opts;
  opts.band = -1;
  (void)dtwc::metal::compute_distance_matrix_metal(series, opts);

  for (auto _ : state) {
    auto r = dtwc::metal::compute_distance_matrix_metal(series, opts);
    benchmark::DoNotOptimize(r.matrix.data());
  }

  const int64_t pairs = static_cast<int64_t>(N) * (N - 1) / 2;
  state.counters["pairs"] = static_cast<double>(pairs);
  state.counters["N"] = N;
  state.counters["L"] = L;
}

BENCHMARK(BM_metal_distanceMatrix)
    ->Args({50, 500})
    ->Args({100, 500})
    ->Args({100, 1000})
    ->Args({200, 500})
    ->Args({50, 2500}) // close to threadgroup memory cap (2730 max on M2/M3)
    ->Args({10, 10000}) // global-memory kernel territory
    ->Args({30, 10000})
    ->Args({75, 500})   // 75-series scaling sweep
    ->Args({75, 1000})
    ->Args({75, 2500})
    ->Unit(benchmark::kMillisecond);

// Banded variant: same sizes, band = 10% of length. Reduces work
// proportionally since cells outside the Sakoe-Chiba band are INF.
static void BM_metal_distanceMatrix_banded(benchmark::State &state)
{
  const int N = static_cast<int>(state.range(0));
  const int L = static_cast<int>(state.range(1));
  const int band = std::max(1, L / 10);
  auto series = make_series_set(N, L);

  dtwc::metal::MetalDistMatOptions opts;
  opts.band = band;
  (void)dtwc::metal::compute_distance_matrix_metal(series, opts);

  for (auto _ : state) {
    auto r = dtwc::metal::compute_distance_matrix_metal(series, opts);
    benchmark::DoNotOptimize(r.matrix.data());
  }

  const int64_t pairs = static_cast<int64_t>(N) * (N - 1) / 2;
  state.counters["pairs"] = static_cast<double>(pairs);
  state.counters["N"] = N;
  state.counters["L"] = L;
  state.counters["band"] = band;
}

BENCHMARK(BM_metal_distanceMatrix_banded)
    ->Args({30, 10000})
    ->Args({75, 2500})
    ->Args({75, 10000})
    ->Unit(benchmark::kMillisecond);

// Tight-band variant: band = 1% of length. Typical "real DTW" band size.
static void BM_metal_distanceMatrix_tightband(benchmark::State &state)
{
  const int N = static_cast<int>(state.range(0));
  const int L = static_cast<int>(state.range(1));
  const int band = std::max(1, L / 100);
  auto series = make_series_set(N, L);

  dtwc::metal::MetalDistMatOptions opts;
  opts.band = band;
  (void)dtwc::metal::compute_distance_matrix_metal(series, opts);

  for (auto _ : state) {
    auto r = dtwc::metal::compute_distance_matrix_metal(series, opts);
    benchmark::DoNotOptimize(r.matrix.data());
  }
  state.counters["N"] = N;
  state.counters["L"] = L;
  state.counters["band"] = band;
}

BENCHMARK(BM_metal_distanceMatrix_tightband)
    ->Args({75, 10000})
    ->Unit(benchmark::kMillisecond);

// Unbanded 75×10000 on Metal — user wants the GPU number too.
BENCHMARK(BM_metal_distanceMatrix)
    ->Args({75, 10000})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);

// Register-tile kernel coverage: short/medium unbanded (max_L <= 256).
// Reports the regtile kernel's throughput on its target workload range so the
// 2-4x over wavefront expectation can be verified against the baseline numbers
// above. Uses N=100 to fill many threadgroups with work.
static void BM_metal_regtile_short(benchmark::State &state)
{
  const int N = static_cast<int>(state.range(0));
  const int L = static_cast<int>(state.range(1));
  auto series = make_series_set(N, L);

  dtwc::metal::MetalDistMatOptions opts;
  opts.band = -1;
  (void)dtwc::metal::compute_distance_matrix_metal(series, opts); // warm-up

  for (auto _ : state) {
    auto r = dtwc::metal::compute_distance_matrix_metal(series, opts);
    benchmark::DoNotOptimize(r.matrix.data());
  }

  const int64_t pairs = static_cast<int64_t>(N) * (N - 1) / 2;
  state.counters["pairs"] = static_cast<double>(pairs);
  state.counters["N"] = N;
  state.counters["L"] = L;
}

BENCHMARK(BM_metal_regtile_short)
    ->Args({100, 64})
    ->Args({100, 128})
    ->Args({100, 192})
    ->Args({100, 256})
    ->Unit(benchmark::kMillisecond);

// LB_Keogh pruning path on a k-medoids-shape workload.
// Compares:
//   - baseline wavefront (N x N)
//   - LB_Keogh with a permissive threshold (all active, overhead only)
//   - LB_Keogh with a strict threshold (most pruned on random data)
// Reports the wall time with pruning counters so the cost/benefit can be read
// off the JSON. Tuned so 3*max_L*4 exceeds threadgroup cap -> wavefront_global.
static void BM_metal_lb_keogh_permissive(benchmark::State &state)
{
  const int N = static_cast<int>(state.range(0));
  const int L = static_cast<int>(state.range(1));
  auto series = make_series_set(N, L);

  dtwc::metal::MetalDistMatOptions opts;
  opts.enable_lb_keogh = true;
  opts.lb_threshold = std::numeric_limits<double>::infinity();
  opts.lb_envelope_band = std::max(1, L / 10);
  (void)dtwc::metal::compute_distance_matrix_metal(series, opts); // warm-up

  size_t last_pruned = 0;
  for (auto _ : state) {
    auto r = dtwc::metal::compute_distance_matrix_metal(series, opts);
    last_pruned = r.pairs_pruned;
    benchmark::DoNotOptimize(r.matrix.data());
  }
  const int64_t pairs = static_cast<int64_t>(N) * (N - 1) / 2;
  state.counters["N"] = N;
  state.counters["L"] = L;
  state.counters["pairs_total"] = static_cast<double>(pairs);
  state.counters["pairs_pruned"] = static_cast<double>(last_pruned);
}

BENCHMARK(BM_metal_lb_keogh_permissive)
    ->Args({100, 1000})
    ->Args({200, 1000})
    ->Unit(benchmark::kMillisecond);

static void BM_metal_lb_keogh_strict(benchmark::State &state)
{
  const int N = static_cast<int>(state.range(0));
  const int L = static_cast<int>(state.range(1));
  auto series = make_series_set(N, L);

  dtwc::metal::MetalDistMatOptions opts;
  opts.enable_lb_keogh = true;
  opts.lb_threshold = 0.0; // prunes nearly everything on random data
  opts.lb_envelope_band = std::max(1, L / 10);
  (void)dtwc::metal::compute_distance_matrix_metal(series, opts);

  size_t last_pruned = 0;
  for (auto _ : state) {
    auto r = dtwc::metal::compute_distance_matrix_metal(series, opts);
    last_pruned = r.pairs_pruned;
    benchmark::DoNotOptimize(r.matrix.data());
  }
  const int64_t pairs = static_cast<int64_t>(N) * (N - 1) / 2;
  state.counters["N"] = N;
  state.counters["L"] = L;
  state.counters["pairs_total"] = static_cast<double>(pairs);
  state.counters["pairs_pruned"] = static_cast<double>(last_pruned);
}

BENCHMARK(BM_metal_lb_keogh_strict)
    ->Args({100, 1000})
    ->Args({200, 1000})
    ->Unit(benchmark::kMillisecond);

// Report achieved FLOPs vs Apple GPU theoretical peak for each size.
// Ops per DTW cell (approximate): 1 sub + 1 abs + 3 min + 1 add = ~6 FP32.
static void BM_metal_flops(benchmark::State &state)
{
  const int N = static_cast<int>(state.range(0));
  const int L = static_cast<int>(state.range(1));
  auto series = make_series_set(N, L);

  dtwc::metal::MetalDistMatOptions opts;
  (void)dtwc::metal::compute_distance_matrix_metal(series, opts); // warm-up

  for (auto _ : state) {
    auto r = dtwc::metal::compute_distance_matrix_metal(series, opts);
    benchmark::DoNotOptimize(r.matrix.data());
  }

  const int64_t pairs = static_cast<int64_t>(N) * (N - 1) / 2;
  const int64_t cells = pairs * static_cast<int64_t>(L) * static_cast<int64_t>(L);
  const double ops = static_cast<double>(cells) * 6.0;
  const double wall = state.iterations() > 0
      ? state.iterations() * std::chrono::duration<double>(
            std::chrono::nanoseconds(static_cast<int64_t>(state.iterations() > 0
                ? 0 : 0))).count()
      : 0.0;
  (void)wall;
  state.counters["N"] = N;
  state.counters["L"] = L;
  state.counters["cells_1e9"] = static_cast<double>(cells) / 1e9;
  state.counters["ops_1e9"] = ops / 1e9;
  state.counters["GFLOPS"] = benchmark::Counter(
      ops, benchmark::Counter::kIsIterationInvariantRate);
  state.counters["cells_per_sec_1e9"] = benchmark::Counter(
      static_cast<double>(cells), benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_metal_flops)
    ->Args({50, 500})
    ->Args({100, 1000})
    ->Args({50, 2500})
    ->Unit(benchmark::kMillisecond);

#endif // DTWC_HAS_METAL

// CPU reference at matching sizes so the JSON output has an apples-to-apples
// comparison row. Uses the library's fillDistanceMatrix via Problem.
static void BM_cpu_distanceMatrix(benchmark::State &state)
{
  const int N = static_cast<int>(state.range(0));
  const int L = static_cast<int>(state.range(1));
  auto series = make_series_set(N, L);

  std::vector<std::vector<dtwc::data_t>> vecs = series;
  std::vector<std::string> names(N);
  for (int i = 0; i < N; ++i) names[i] = "s" + std::to_string(i);

  for (auto _ : state) {
    auto vecs_copy = vecs;
    auto names_copy = names;
    dtwc::Data data{std::move(vecs_copy), std::move(names_copy)};
    dtwc::Problem prob;
    prob.set_data(std::move(data));
    prob.band = -1;
    prob.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
    prob.fillDistanceMatrix();
    benchmark::ClobberMemory();
  }

  const int64_t pairs = static_cast<int64_t>(N) * (N - 1) / 2;
  state.counters["pairs"] = static_cast<double>(pairs);
  state.counters["N"] = N;
  state.counters["L"] = L;
}

BENCHMARK(BM_cpu_distanceMatrix)
    ->Args({50, 500})
    ->Args({100, 500})
    ->Args({100, 1000})
    ->Args({200, 500})
    ->Args({10, 10000})
    ->Args({30, 10000})
    ->Args({75, 500})
    ->Args({75, 1000})
    ->Args({75, 2500})
    ->Unit(benchmark::kMillisecond);

// Fixed band=100 (user-requested: <4s CPU for 75×10000).
static void BM_cpu_distanceMatrix_b100(benchmark::State &state)
{
  const int N = static_cast<int>(state.range(0));
  const int L = static_cast<int>(state.range(1));
  const int band = 100;
  auto series = make_series_set(N, L);

  std::vector<std::vector<dtwc::data_t>> vecs = series;
  std::vector<std::string> names(N);
  for (int i = 0; i < N; ++i) names[i] = "s" + std::to_string(i);

  for (auto _ : state) {
    auto vecs_copy = vecs;
    auto names_copy = names;
    dtwc::Data data{std::move(vecs_copy), std::move(names_copy)};
    dtwc::Problem prob;
    prob.set_data(std::move(data));
    prob.band = band;
    prob.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
    prob.fillDistanceMatrix();
    benchmark::ClobberMemory();
  }
  state.counters["N"] = N;
  state.counters["L"] = L;
  state.counters["band"] = band;
}

BENCHMARK(BM_cpu_distanceMatrix_b100)
    ->Args({75, 10000})
    ->Unit(benchmark::kMillisecond);

// Unbanded 75-series 10k — user target: <100 s CPU.
BENCHMARK(BM_cpu_distanceMatrix)
    ->Args({75, 10000})
    ->Unit(benchmark::kMillisecond)
    ->Iterations(1); // one pass is enough; it's long

static void BM_cpu_distanceMatrix_banded(benchmark::State &state)
{
  const int N = static_cast<int>(state.range(0));
  const int L = static_cast<int>(state.range(1));
  const int band = std::max(1, L / 10);
  auto series = make_series_set(N, L);

  std::vector<std::vector<dtwc::data_t>> vecs = series;
  std::vector<std::string> names(N);
  for (int i = 0; i < N; ++i) names[i] = "s" + std::to_string(i);

  for (auto _ : state) {
    auto vecs_copy = vecs;
    auto names_copy = names;
    dtwc::Data data{std::move(vecs_copy), std::move(names_copy)};
    dtwc::Problem prob;
    prob.set_data(std::move(data));
    prob.band = band;
    prob.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
    prob.fillDistanceMatrix();
    benchmark::ClobberMemory();
  }

  const int64_t pairs = static_cast<int64_t>(N) * (N - 1) / 2;
  state.counters["pairs"] = static_cast<double>(pairs);
  state.counters["N"] = N;
  state.counters["L"] = L;
  state.counters["band"] = band;
}

BENCHMARK(BM_cpu_distanceMatrix_banded)
    ->Args({30, 10000})
    ->Args({75, 2500})
    ->Args({75, 10000})
    ->Unit(benchmark::kMillisecond);
