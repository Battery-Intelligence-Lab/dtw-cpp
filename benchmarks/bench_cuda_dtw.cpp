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

#include "../tests/support/deterministic_series.hpp"

#ifdef DTWC_HAS_CUDA
#include <cuda/cuda_dtw.cuh>
#endif

#include <vector>
#include <string>
#include <iostream>

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<std::vector<double>> make_series_set(int N, int L,
                                                         unsigned base_seed = 200)
{
  return dtwc::test_support::benchmark_series_set(
    static_cast<std::size_t>(N), static_cast<std::size_t>(L), base_seed);
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
  auto vecs = dtwc::test_support::benchmark_series_set(
    static_cast<std::size_t>(N), static_cast<std::size_t>(L), base_seed);
  std::vector<std::string> names;
  names.reserve(N);
  for (int i = 0; i < N; ++i)
    names.push_back("s" + std::to_string(i));
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
  dtwc::core::DistanceMatrix matrix;
  (void)dtwc::cuda::compute_distance_matrix_cuda(series, opts, matrix); // warm-up

  for (auto _ : state) {
    (void)dtwc::cuda::compute_distance_matrix_cuda(series, opts, matrix);
    benchmark::DoNotOptimize(matrix.raw());
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

    prob.fill_distance_matrix();
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
  dtwc::core::DistanceMatrix matrix;
  (void)dtwc::cuda::compute_distance_matrix_cuda(series, opts, matrix); // warm-up

  for (auto _ : state) {
    (void)dtwc::cuda::compute_distance_matrix_cuda(series, opts, matrix);
    benchmark::DoNotOptimize(matrix.raw());
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
  dtwc::core::DistanceMatrix matrix;
  (void)dtwc::cuda::compute_distance_matrix_cuda(series, opts, matrix); // warm-up

  for (auto _ : state) {
    (void)dtwc::cuda::compute_distance_matrix_cuda(series, opts, matrix);
    benchmark::DoNotOptimize(matrix.raw());
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

  dtwc::core::DistanceMatrix matrix;
  (void)dtwc::cuda::compute_distance_matrix_cuda(series, opts, matrix); // warm-up

  for (auto _ : state) {
    (void)dtwc::cuda::compute_distance_matrix_cuda(series, opts, matrix);
    benchmark::DoNotOptimize(matrix.raw());
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
// BM_cuda_fill — the public fill: Problem::fill_distance_matrix on device gpu,
// from the series to the filled packed matrix (upload, kernels, transfer, the
// host's share). Args: (N_series, series_length, fp64). N is sized so one fill
// takes at least a second on the RTX 4000 Ada.
// ---------------------------------------------------------------------------
static void BM_cuda_fill(benchmark::State &state)
{
  const int N = static_cast<int>(state.range(0));
  const int L = static_cast<int>(state.range(1));
  const bool fp64 = state.range(2) != 0;

  dtwc::Problem prob("bench");
  prob.set_data(make_random_data(N, L));
  prob.set_device(dtwc::Device::GPU);
  prob.set_gpu_precision(fp64 ? dtwc::GpuPrecision::FP64 : dtwc::GpuPrecision::FP32);
  prob.fill_distance_matrix(); // warm-up: the CUDA context and this thread's buffers

  for (auto _ : state) {
    state.PauseTiming();
    prob.refresh_distance_matrix();
    state.ResumeTiming();
    prob.fill_distance_matrix();
  }

  const double cells = static_cast<double>(N) * (N - 1) / 2 * L * L;
  state.counters["Gcell/s"] = benchmark::Counter(
      cells * 1e-9, benchmark::Counter::kIsIterationInvariantRate);
}

BENCHMARK(BM_cuda_fill)
  ->ArgNames({ "N", "L", "fp64" })
  ->Args({ 9000, 100, 0 })
  ->Args({ 1100, 500, 0 })
  ->Args({ 330, 2000, 0 })
  ->Args({ 2700, 100, 1 })
  ->Args({ 520, 500, 1 })
  ->Args({ 140, 2000, 1 })
  ->Iterations(1)
  ->UseRealTime()
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

// Custom main: inject GPU device info into Google Benchmark's JSON context so
// benchmark results include hardware specs (more valuable than the auto-filled
// host_name, which we strip post-run).
int main(int argc, char **argv)
{
  benchmark::Initialize(&argc, argv);
#ifdef DTWC_HAS_CUDA
  benchmark::AddCustomContext("gpu_backend", "cuda");
  benchmark::AddCustomContext("gpu_available",
      dtwc::cuda::cuda_available() ? "true" : "false");
  if (dtwc::cuda::cuda_available())
    benchmark::AddCustomContext("gpu_device_info", dtwc::cuda::cuda_device_info(0));
#else
  benchmark::AddCustomContext("gpu_backend", "cuda");
  benchmark::AddCustomContext("gpu_available", "false");
#endif
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
