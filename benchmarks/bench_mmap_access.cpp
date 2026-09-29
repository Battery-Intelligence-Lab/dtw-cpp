/**
 * @file bench_mmap_access.cpp
 * @brief Read and write probe of core::DistanceMatrix, on the heap and mapped.
 *
 * @details get and set are the same inline code for both storages, so the probe
 *          compares only where the doubles live: the heap, or a mapped `.dtwm`
 *          file behind the page cache. Argument 0 is the heap, 1 the mapped file.
 *          A build without llfio reports the mapped cases as errors.
 *
 * @author Volkan Kumtepeli
 * @date 29 Sep 2026
 */

#include <benchmark/benchmark.h>
#include <core/distance_matrix.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <optional>
#include <random>
#include <vector>

namespace {

using dtwc::core::DistanceMatrix;
namespace fs = std::filesystem;

constexpr std::size_t N = 4000; // 8 million doubles, 64 MB

/// An N x N matrix with every pair set, on the heap or mapped to a fresh file
/// that is removed with it.
struct Filled
{
  fs::path path = fs::temp_directory_path() / "dtwc_bench_mmap_access.dtwm";
  std::optional<DistanceMatrix> m;

  explicit Filled(benchmark::State &state)
  {
    fs::remove(path);
    try {
      m.emplace(state.range(0) == 1 ? DistanceMatrix::map(path, N, {}) : DistanceMatrix(N));
    } catch (const std::exception &e) {
      state.SkipWithError(e.what());
      return;
    }
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = 0; j <= i; ++j)
        m->set(i, j, static_cast<double>(i + j));
  }
  ~Filled()
  {
    m.reset(); // unmap before the file goes
    std::error_code ignored;
    fs::remove(path, ignored);
  }
};

void BM_set_all(benchmark::State &state)
{
  Filled f(state);
  if (!f.m) return;
  for (auto _ : state) {
    for (std::size_t i = 0; i < N; ++i)
      for (std::size_t j = 0; j <= i; ++j)
        f.m->set(i, j, static_cast<double>(i));
    benchmark::ClobberMemory();
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(f.m->packed_count()));
}

void BM_get_row(benchmark::State &state) // a row per medoid, as the assignment loops read
{
  Filled f(state);
  if (!f.m) return;
  for (auto _ : state) {
    double sum = 0.0;
    for (std::size_t i = 0; i < N; i += 97)
      for (std::size_t j = 0; j < N; ++j)
        sum += f.m->get(i, j);
    benchmark::DoNotOptimize(sum);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(((N + 96) / 97) * N));
}

void BM_get_random(benchmark::State &state)
{
  Filled f(state);
  if (!f.m) return;
  std::vector<std::size_t> index(1 << 16);
  std::mt19937_64 rng(42);
  for (auto &k : index) k = rng() % N;
  for (auto _ : state) {
    double sum = 0.0;
    for (std::size_t k = 0; k + 1 < index.size(); ++k)
      sum += f.m->get(index[k], index[k + 1]);
    benchmark::DoNotOptimize(sum);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(index.size() - 1));
}

} // namespace

BENCHMARK(BM_set_all)->Arg(0)->Arg(1)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_get_row)->Arg(0)->Arg(1)->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_get_random)->Arg(0)->Arg(1)->Unit(benchmark::kMicrosecond);
