/**
 * @file bench_cuda_dtw.cpp
 * @brief GPU benchmarks for CUDA DTW distance matrix computation.
 *
 * @details Measures GPU throughput (pairs/sec), GPU vs CPU speedup,
 *          and scaling with N (number of series) and L (series length).
 *          Uses Google Benchmark with deterministic random data.
 *
 * @author Volkan Kumtepeli
 * @date 01 Apr 2026
 */

#include <benchmark/benchmark.h>
#include <dtwc.hpp>

#ifdef DTWC_HAS_CUDA
#include <cuda/cuda_dtw.cuh>
#endif

#include <vector>
#include <random>
#include <string>
#include <iostream>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<double> random_series(size_t len, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> s(len);
  for (auto &v : s)
    v = dist(rng);
  return s;
}

static std::vector<std::vector<double>> make_series_set(int N, int L,
                                                         unsigned base_seed = 200)
{
  std::vector<std::vector<double>> vecs;
  vecs.reserve(N);
  for (int i = 0; i < N; ++i)
    vecs.push_back(random_series(static_cast<size_t>(L), base_seed + i));
  return vecs;
}

static std::vector<std::vector<double>> make_pruning_friendly_series_set(int N, int L)
{
  std::vector<std::vector<double>> vecs;
  vecs.reserve(N);
  for (int i = 0; i < N; ++i) {
    const double family_bias = (i % 2 == 0) ? -150.0 : 150.0;
    std::vector<double> s(static_cast<size_t>(L));
    for (int k = 0; k < L; ++k) {
      s[static_cast<size_t>(k)] = family_bias + 0.05 * k + 0.001 * i;
    }
    vecs.push_back(std::move(s));
  }
  return vecs;
}

static dtwc::Data make_random_data(int N, int L, unsigned base_seed = 200)
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

#ifdef DTWC_HAS_CUDA

// ---------------------------------------------------------------------------
// BM_cuda_distanceMatrix — GPU distance matrix (varying N and L)
// Args: (N_series, series_length)
// ---------------------------------------------------------------------------
static void BM_cuda_distanceMatrix(benchmark::State &state)
{
  const int N = static_cast<int>(state.range(0));
  const int L = static_cast<int>(state.range(1));
  auto series = make_series_set(N, L);
  const int64_t num_pairs = static_cast<int64_t>(N) * (N - 1) / 2;

  dtwc::cuda::CUDADistMatOptions opts;
  opts.verbose = false;

  // Warm-up: first CUDA call has driver overhead
  { auto warmup = dtwc::cuda::compute_distance_matrix_cuda(series, opts); }

  for (auto _ : state) {
    auto result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
    benchmark::DoNotOptimize(result.matrix.data());
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_pairs);
  state.counters["pairs"] = benchmark::Counter(
      static_cast<double>(num_pairs), benchmark::Counter::kDefaults);
  state.counters["pairs/sec"] = benchmark::Counter(
      static_cast<double>(num_pairs), benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_cuda_distanceMatrix)
  ->Args({20, 100})
  ->Args({50, 100})
  ->Args({100, 100})
  ->Args({20, 500})
  ->Args({50, 500})
  ->Args({100, 500})
  ->Args({50, 1000})
  ->Args({100, 1000})
  ->Args({200, 500})
  ->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// BM_cuda_vs_cpu — side-by-side comparison at same problem size
// Args: (N_series, series_length)
// ---------------------------------------------------------------------------
static void BM_cpu_distanceMatrix(benchmark::State &state)
{
  const int N = static_cast<int>(state.range(0));
  const int L = static_cast<int>(state.range(1));
  const int64_t num_pairs = static_cast<int64_t>(N) * (N - 1) / 2;

  for (auto _ : state) {
    state.PauseTiming();
    dtwc::Problem prob("bench");
    prob.set_data(make_random_data(N, L));
    state.ResumeTiming();

    prob.fillDistanceMatrix();
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_pairs);
  state.counters["pairs/sec"] = benchmark::Counter(
      static_cast<double>(num_pairs), benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_cpu_distanceMatrix)
  ->Args({20, 100})
  ->Args({50, 100})
  ->Args({100, 100})
  ->Args({20, 500})
  ->Args({50, 500})
  ->Args({100, 500})
  ->Args({50, 1000})
  ->Args({100, 1000})
  ->Args({200, 500})
  ->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// BM_cuda_scaling_N — fix L, vary N to see pair-parallelism scaling
// ---------------------------------------------------------------------------
static void BM_cuda_scaling_N(benchmark::State &state)
{
  const int N = static_cast<int>(state.range(0));
  const int L = 500;
  auto series = make_series_set(N, L);
  const int64_t num_pairs = static_cast<int64_t>(N) * (N - 1) / 2;

  dtwc::cuda::CUDADistMatOptions opts;
  { auto warmup = dtwc::cuda::compute_distance_matrix_cuda(series, opts); }

  for (auto _ : state) {
    auto result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
    benchmark::DoNotOptimize(result.matrix.data());
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_pairs);
  state.counters["pairs/sec"] = benchmark::Counter(
      static_cast<double>(num_pairs), benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_cuda_scaling_N)
  ->Arg(10)
  ->Arg(20)
  ->Arg(50)
  ->Arg(100)
  ->Arg(200)
  ->Arg(500)
  ->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// BM_cuda_scaling_L — fix N, vary L to see per-pair cost scaling
// ---------------------------------------------------------------------------
static void BM_cuda_scaling_L(benchmark::State &state)
{
  const int N = 50;
  const int L = static_cast<int>(state.range(0));
  auto series = make_series_set(N, L);
  const int64_t num_pairs = static_cast<int64_t>(N) * (N - 1) / 2;

  dtwc::cuda::CUDADistMatOptions opts;
  { auto warmup = dtwc::cuda::compute_distance_matrix_cuda(series, opts); }

  for (auto _ : state) {
    auto result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
    benchmark::DoNotOptimize(result.matrix.data());
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_pairs);
  state.counters["pairs/sec"] = benchmark::Counter(
      static_cast<double>(num_pairs), benchmark::Counter::kIsIterationInvariantRate);
  state.counters["cells/sec"] = benchmark::Counter(
      static_cast<double>(num_pairs) * L * L,
      benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_cuda_scaling_L)
  ->Arg(100)
  ->Arg(250)
  ->Arg(500)
  ->Arg(1000)
  ->Arg(2000)
  ->Arg(4000)
  ->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// BM_cuda_structuredDistanceMatrix — same structured data as pruning benchmark,
// but without LB pruning. This is the fair baseline for the pruning path.
// Args: (N_series, series_length)
// ---------------------------------------------------------------------------
static void BM_cuda_structuredDistanceMatrix(benchmark::State &state)
{
  const int N = static_cast<int>(state.range(0));
  const int L = static_cast<int>(state.range(1));
  auto series = make_pruning_friendly_series_set(N, L);
  const int64_t num_pairs = static_cast<int64_t>(N) * (N - 1) / 2;

  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = 10;
  opts.verbose = false;

  { auto warmup = dtwc::cuda::compute_distance_matrix_cuda(series, opts); }

  for (auto _ : state) {
    auto result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
    benchmark::DoNotOptimize(result.matrix.data());
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_pairs);
  state.counters["pairs"] = benchmark::Counter(
      static_cast<double>(num_pairs), benchmark::Counter::kDefaults);
}

BENCHMARK(BM_cuda_structuredDistanceMatrix)
  ->Args({50, 500})
  ->Args({100, 500})
  ->Args({100, 1000})
  ->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// BM_cuda_prunedDistanceMatrix — full matrix with real device-side pruning
// Args: (N_series, series_length)
// ---------------------------------------------------------------------------
static void BM_cuda_prunedDistanceMatrix(benchmark::State &state)
{
  const int N = static_cast<int>(state.range(0));
  const int L = static_cast<int>(state.range(1));
  auto series = make_pruning_friendly_series_set(N, L);
  const int64_t num_pairs = static_cast<int64_t>(N) * (N - 1) / 2;

  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = 10;
  opts.use_lb_keogh = true;
  opts.lb_threshold = 50.0;
  opts.verbose = false;

  dtwc::cuda::CUDADistMatResult last;
  { last = dtwc::cuda::compute_distance_matrix_cuda(series, opts); }

  for (auto _ : state) {
    last = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
    benchmark::DoNotOptimize(last.matrix.data());
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_pairs);
  state.counters["pairs"] = benchmark::Counter(
      static_cast<double>(num_pairs), benchmark::Counter::kDefaults);
  state.counters["active_pairs"] = benchmark::Counter(
      static_cast<double>(last.pairs_computed), benchmark::Counter::kDefaults);
  state.counters["pruned_pairs"] = benchmark::Counter(
      static_cast<double>(last.pairs_pruned), benchmark::Counter::kDefaults);
}

BENCHMARK(BM_cuda_prunedDistanceMatrix)
  ->Args({50, 500})
  ->Args({100, 500})
  ->Args({100, 1000})
  ->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// BM_cuda_oneVsAll — repeated query search
// Args: (N_series, series_length)
// ---------------------------------------------------------------------------
static void BM_cuda_oneVsAll(benchmark::State &state)
{
  const int N = static_cast<int>(state.range(0));
  const int L = static_cast<int>(state.range(1));
  auto series = make_series_set(N, L);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.verbose = false;

  { auto warmup = dtwc::cuda::compute_dtw_one_vs_all(series, 0, opts); }

  for (auto _ : state) {
    auto result = dtwc::cuda::compute_dtw_one_vs_all(series, 0, opts);
    benchmark::DoNotOptimize(result.distances.data());
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * N);
}

BENCHMARK(BM_cuda_oneVsAll)
  ->Args({50, 500})
  ->Args({100, 500})
  ->Args({100, 1000})
  ->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// BM_cuda_kVsAll — batched multi-query search
// Args: (N_series, series_length, K_queries)
// ---------------------------------------------------------------------------
static void BM_cuda_kVsAll(benchmark::State &state)
{
  const int N = static_cast<int>(state.range(0));
  const int L = static_cast<int>(state.range(1));
  const int K = static_cast<int>(state.range(2));
  auto series = make_series_set(N, L);
  std::vector<size_t> query_indices;
  query_indices.reserve(static_cast<size_t>(K));
  for (int i = 0; i < K; ++i)
    query_indices.push_back(static_cast<size_t>(i));

  dtwc::cuda::CUDADistMatOptions opts;
  opts.verbose = false;

  { auto warmup = dtwc::cuda::compute_dtw_k_vs_all(series, query_indices, opts); }

  for (auto _ : state) {
    auto result = dtwc::cuda::compute_dtw_k_vs_all(series, query_indices, opts);
    benchmark::DoNotOptimize(result.distances.data());
  }

  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * K * N);
}

BENCHMARK(BM_cuda_kVsAll)
  ->Args({50, 500, 4})
  ->Args({100, 500, 4})
  ->Args({100, 1000, 8})
  ->Unit(benchmark::kMillisecond);

#endif // DTWC_HAS_CUDA

// If CUDA is not available, provide a placeholder so the binary still links
#ifndef DTWC_HAS_CUDA
static void BM_cuda_not_available(benchmark::State &state)
{
  for (auto _ : state) {}
  state.SkipWithMessage("CUDA not available in this build");
}
BENCHMARK(BM_cuda_not_available);
#endif
