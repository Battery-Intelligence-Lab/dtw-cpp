/**
 * @file bench_mmap_access.cpp
 * @brief Mmap vs dense access pattern benchmarks for distance matrices.
 *
 * @details Compares DenseDistanceMatrix and MmapDistanceMatrix across seven
 *          access patterns: fill, random get, sequential scan, open/read
 *          latency, medoid access, series layout, and CLARA subsample.
 *          All benchmarks always compile (llfio is a required dependency).
 *
 * @author Volkan Kumtepeli
 * @date 08 Apr 2026
 */

#include <benchmark/benchmark.h>
#include <dtwc.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <utility>
#include <vector>

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

/// Return a temp directory path for mmap benchmark files.
static std::filesystem::path temp_mmap_path(const std::string &name)
{
  auto dir = std::filesystem::temp_directory_path() / "dtwc_bench";
  std::filesystem::create_directories(dir);
  return dir / name;
}

// ---------------------------------------------------------------------------
// 1. BM_fill_dense / BM_fill_mmap — Distance matrix fill
// ---------------------------------------------------------------------------

static void BM_fill_dense(benchmark::State &state)
{
  constexpr int N = 5000, L = 25, band = 10;

  for (auto _ : state) {
    state.PauseTiming();
    dtwc::Problem prob("bench_fill_dense");
    prob.set_data(make_random_data(N, L));
    prob.band = band;
    // Force BruteForce for fair comparison with mmap (Pruned is dense-only)
    prob.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
    state.ResumeTiming();

    prob.fillDistanceMatrix();
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations())
                          * static_cast<int64_t>(N) * (N - 1) / 2);
}
BENCHMARK(BM_fill_dense)->Unit(benchmark::kMillisecond);

static void BM_fill_mmap(benchmark::State &state)
{
  constexpr int N = 5000, L = 25, band = 10;
  static int counter = 0;
  auto data = make_random_data(N, L);
  std::vector<std::filesystem::path> cleanup_paths;

  for (auto _ : state) {
    state.PauseTiming();
    auto path = temp_mmap_path("fill_mmap_" + std::to_string(counter++) + ".cache");
    dtwc::Problem prob("bench_fill_mmap");
    prob.set_data(data);
    prob.band = band;
    // Pruned strategy calls dense_distance_matrix() which is incompatible with mmap;
    // force BruteForce which dispatches correctly via visit_distmat.
    prob.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
    prob.use_mmap_distance_matrix(path);
    cleanup_paths.push_back(path);
    state.ResumeTiming();

    prob.fillDistanceMatrix();
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations())
                          * static_cast<int64_t>(N) * (N - 1) / 2);

  // Clean up after all iterations (prob already destroyed at this point)
  for (const auto &p : cleanup_paths)
    std::filesystem::remove(p);
}
BENCHMARK(BM_fill_mmap)->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// 2. BM_random_get_dense / BM_random_get_mmap — Random tri_index lookups
// ---------------------------------------------------------------------------

static void BM_random_get_dense(benchmark::State &state)
{
  constexpr size_t N = 5000;
  constexpr size_t num_lookups = 500000;

  // Pre-fill dense matrix with random values
  dtwc::core::DenseDistanceMatrix dm(N);
  {
    std::mt19937 rng(77);
    std::uniform_real_distribution<double> ddist(0.0, 100.0);
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j <= i; ++j)
        dm.set(i, j, ddist(rng));
  }

  // Pre-generate random (i,j) pairs
  std::vector<std::pair<size_t, size_t>> pairs(num_lookups);
  {
    std::mt19937 rng(88);
    std::uniform_int_distribution<size_t> idist(0, N - 1);
    for (auto &p : pairs)
      p = { idist(rng), idist(rng) };
  }

  for (auto _ : state) {
    double sum = 0.0;
    for (const auto &[i, j] : pairs)
      sum += dm.get(i, j);
    benchmark::DoNotOptimize(sum);
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_lookups);
}
BENCHMARK(BM_random_get_dense)->Unit(benchmark::kMillisecond);

static void BM_random_get_mmap(benchmark::State &state)
{
  constexpr size_t N = 5000;
  constexpr size_t num_lookups = 500000;

  auto path = temp_mmap_path("random_get_mmap.cache");

  // Pre-fill mmap matrix with same random values
  {
    dtwc::core::MmapDistanceMatrix dm(path, N);
    std::mt19937 rng(77);
    std::uniform_real_distribution<double> ddist(0.0, 100.0);
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j <= i; ++j)
        dm.set(i, j, ddist(rng));
    dm.sync();
  }

  // Reopen for benchmark
  auto dm = dtwc::core::MmapDistanceMatrix::open(path);

  // Pre-generate random (i,j) pairs
  std::vector<std::pair<size_t, size_t>> pairs(num_lookups);
  {
    std::mt19937 rng(88);
    std::uniform_int_distribution<size_t> idist(0, N - 1);
    for (auto &p : pairs)
      p = { idist(rng), idist(rng) };
  }

  for (auto _ : state) {
    double sum = 0.0;
    for (const auto &[i, j] : pairs)
      sum += dm.get(i, j);
    benchmark::DoNotOptimize(sum);
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * num_lookups);

  // Cleanup
  std::filesystem::remove(path);
}
BENCHMARK(BM_random_get_mmap)->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// 3. BM_sequential_scan_dense / BM_sequential_scan_mmap — Raw pointer scan
// ---------------------------------------------------------------------------

static void BM_sequential_scan_dense(benchmark::State &state)
{
  constexpr size_t N = 5000;

  dtwc::core::DenseDistanceMatrix dm(N);
  {
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> ddist(0.0, 100.0);
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j <= i; ++j)
        dm.set(i, j, ddist(rng));
  }

  const size_t count = dm.packed_count();
  for (auto _ : state) {
    double sum = 0.0;
    const double *ptr = dm.raw();
    for (size_t i = 0; i < count; ++i)
      sum += ptr[i];
    benchmark::DoNotOptimize(sum);
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations())
                          * static_cast<int64_t>(count) * 8);
}
BENCHMARK(BM_sequential_scan_dense)->Unit(benchmark::kMillisecond);

static void BM_sequential_scan_mmap(benchmark::State &state)
{
  constexpr size_t N = 5000;

  auto path = temp_mmap_path("seq_scan_mmap.cache");
  {
    dtwc::core::MmapDistanceMatrix dm(path, N);
    std::mt19937 rng(99);
    std::uniform_real_distribution<double> ddist(0.0, 100.0);
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j <= i; ++j)
        dm.set(i, j, ddist(rng));
    dm.sync();
  }

  auto dm = dtwc::core::MmapDistanceMatrix::open(path);
  const size_t count = dm.packed_count();

  for (auto _ : state) {
    double sum = 0.0;
    const double *ptr = dm.raw();
    for (size_t i = 0; i < count; ++i)
      sum += ptr[i];
    benchmark::DoNotOptimize(sum);
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations())
                          * static_cast<int64_t>(count) * 8);

  std::filesystem::remove(path);
}
BENCHMARK(BM_sequential_scan_mmap)->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// 4. BM_open_mmap / BM_read_binary_to_vector — Startup latency
// ---------------------------------------------------------------------------

static void BM_open_mmap(benchmark::State &state)
{
  constexpr size_t N = 5000;

  auto path = temp_mmap_path("open_latency.cache");
  {
    dtwc::core::MmapDistanceMatrix dm(path, N);
    std::mt19937 rng(55);
    std::uniform_real_distribution<double> ddist(0.0, 100.0);
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j <= i; ++j)
        dm.set(i, j, ddist(rng));
    dm.sync();
  }

  const auto file_sz = std::filesystem::file_size(path);

  for (auto _ : state) {
    auto dm = dtwc::core::MmapDistanceMatrix::open(path);
    benchmark::DoNotOptimize(dm.raw());
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations())
                          * static_cast<int64_t>(file_sz));

  std::filesystem::remove(path);
}
BENCHMARK(BM_open_mmap)->Unit(benchmark::kMicrosecond);

static void BM_read_binary_to_vector(benchmark::State &state)
{
  constexpr size_t N = 5000;

  auto path = temp_mmap_path("read_vec_latency.cache");
  {
    dtwc::core::MmapDistanceMatrix dm(path, N);
    std::mt19937 rng(55);
    std::uniform_real_distribution<double> ddist(0.0, 100.0);
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j <= i; ++j)
        dm.set(i, j, ddist(rng));
    dm.sync();
  }

  const auto file_sz = std::filesystem::file_size(path);

  for (auto _ : state) {
    std::ifstream ifs(path, std::ios::binary);
    std::vector<double> vec(file_sz / sizeof(double));
    ifs.read(reinterpret_cast<char *>(vec.data()), static_cast<std::streamsize>(file_sz));
    benchmark::DoNotOptimize(vec.data());
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations())
                          * static_cast<int64_t>(file_sz));

  std::filesystem::remove(path);
}
BENCHMARK(BM_read_binary_to_vector)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// 5. BM_medoid_access_dense / BM_medoid_access_mmap — Realistic FastPAM access
// ---------------------------------------------------------------------------

static void BM_medoid_access_dense(benchmark::State &state)
{
  constexpr size_t N = 5000;
  constexpr size_t k = 10;

  dtwc::core::DenseDistanceMatrix dm(N);
  {
    std::mt19937 rng(111);
    std::uniform_real_distribution<double> ddist(0.0, 100.0);
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j <= i; ++j)
        dm.set(i, j, ddist(rng));
  }

  // Pick k medoids deterministically
  std::vector<size_t> medoids(k);
  std::iota(medoids.begin(), medoids.end(), 0);

  for (auto _ : state) {
    double total = 0.0;
    for (size_t p = k; p < N; ++p) {
      double min_d = std::numeric_limits<double>::max();
      for (size_t m = 0; m < k; ++m) {
        double d = dm.get(p, medoids[m]);
        if (d < min_d) min_d = d;
      }
      total += min_d;
    }
    benchmark::DoNotOptimize(total);
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations())
                          * static_cast<int64_t>(N - k) * static_cast<int64_t>(k));
}
BENCHMARK(BM_medoid_access_dense)->Unit(benchmark::kMillisecond);

static void BM_medoid_access_mmap(benchmark::State &state)
{
  constexpr size_t N = 5000;
  constexpr size_t k = 10;

  auto path = temp_mmap_path("medoid_access_mmap.cache");
  {
    dtwc::core::MmapDistanceMatrix dm(path, N);
    std::mt19937 rng(111);
    std::uniform_real_distribution<double> ddist(0.0, 100.0);
    for (size_t i = 0; i < N; ++i)
      for (size_t j = 0; j <= i; ++j)
        dm.set(i, j, ddist(rng));
    dm.sync();
  }

  auto dm = dtwc::core::MmapDistanceMatrix::open(path);

  std::vector<size_t> medoids(k);
  std::iota(medoids.begin(), medoids.end(), 0);

  for (auto _ : state) {
    double total = 0.0;
    for (size_t p = k; p < N; ++p) {
      double min_d = std::numeric_limits<double>::max();
      for (size_t m = 0; m < k; ++m) {
        double d = dm.get(p, medoids[m]);
        if (d < min_d) min_d = d;
      }
      total += min_d;
    }
    benchmark::DoNotOptimize(total);
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations())
                          * static_cast<int64_t>(N - k) * static_cast<int64_t>(k));

  std::filesystem::remove(path);
}
BENCHMARK(BM_medoid_access_mmap)->Unit(benchmark::kMillisecond);

// ---------------------------------------------------------------------------
// 6. BM_series_contiguous_flat / BM_series_vec_of_vec — Series access layout
// ---------------------------------------------------------------------------

static void BM_series_vec_of_vec(benchmark::State &state)
{
  constexpr size_t N = 5000;
  constexpr size_t L = 25;

  std::vector<std::vector<double>> series(N);
  {
    std::mt19937 rng(200);
    std::uniform_real_distribution<double> ddist(-1.0, 1.0);
    for (auto &s : series) {
      s.resize(L);
      for (auto &v : s) v = ddist(rng);
    }
  }

  for (auto _ : state) {
    double sum = 0.0;
    for (const auto &s : series)
      for (double v : s)
        sum += v;
    benchmark::DoNotOptimize(sum);
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations())
                          * static_cast<int64_t>(N * L * 8));
}
BENCHMARK(BM_series_vec_of_vec)->Unit(benchmark::kMicrosecond);

static void BM_series_contiguous_flat(benchmark::State &state)
{
  constexpr size_t N = 5000;
  constexpr size_t L = 25;

  std::vector<double> flat(N * L);
  std::vector<size_t> offsets(N);
  {
    std::mt19937 rng(200);
    std::uniform_real_distribution<double> ddist(-1.0, 1.0);
    for (size_t i = 0; i < N; ++i) {
      offsets[i] = i * L;
      for (size_t j = 0; j < L; ++j)
        flat[i * L + j] = ddist(rng);
    }
  }

  for (auto _ : state) {
    double sum = 0.0;
    for (size_t i = 0; i < N; ++i) {
      const double *ptr = flat.data() + offsets[i];
      for (size_t j = 0; j < L; ++j)
        sum += ptr[j];
    }
    benchmark::DoNotOptimize(sum);
  }
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations())
                          * static_cast<int64_t>(N * L * 8));
}
BENCHMARK(BM_series_contiguous_flat)->Unit(benchmark::kMicrosecond);

// ---------------------------------------------------------------------------
// 7. BM_clara_copy / BM_clara_view — CLARA subsample overhead
// ---------------------------------------------------------------------------

static void BM_clara_copy(benchmark::State &state)
{
  constexpr size_t N = 5000;
  constexpr size_t L = 25;
  constexpr size_t S = 100;

  std::vector<std::vector<double>> series(N);
  {
    std::mt19937 rng(300);
    std::uniform_real_distribution<double> ddist(-1.0, 1.0);
    for (auto &s : series) {
      s.resize(L);
      for (auto &v : s) v = ddist(rng);
    }
  }

  // Pick 100 random sample indices
  std::vector<size_t> indices(S);
  {
    std::mt19937 rng(400);
    std::uniform_int_distribution<size_t> idist(0, N - 1);
    for (auto &idx : indices) idx = idist(rng);
  }

  for (auto _ : state) {
    std::vector<std::vector<double>> sub;
    sub.reserve(S);
    for (size_t idx : indices)
      sub.push_back(series[idx]);
    benchmark::DoNotOptimize(sub.data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * S);
}
BENCHMARK(BM_clara_copy)->Unit(benchmark::kMicrosecond);

static void BM_clara_view(benchmark::State &state)
{
  constexpr size_t N = 5000;
  constexpr size_t L = 25;
  constexpr size_t S = 100;

  std::vector<double> flat(N * L);
  std::vector<size_t> offsets(N);
  {
    std::mt19937 rng(300);
    std::uniform_real_distribution<double> ddist(-1.0, 1.0);
    for (size_t i = 0; i < N; ++i) {
      offsets[i] = i * L;
      for (size_t j = 0; j < L; ++j)
        flat[i * L + j] = ddist(rng);
    }
  }

  std::vector<size_t> indices(S);
  {
    std::mt19937 rng(400);
    std::uniform_int_distribution<size_t> idist(0, N - 1);
    for (auto &idx : indices) idx = idist(rng);
  }

  for (auto _ : state) {
    std::vector<std::pair<const double *, size_t>> views;
    views.reserve(S);
    for (size_t idx : indices)
      views.emplace_back(flat.data() + offsets[idx], L);
    benchmark::DoNotOptimize(views.data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) * S);
}
BENCHMARK(BM_clara_view)->Unit(benchmark::kMicrosecond);
