/**
 * @file bench_mpi_dtw.cpp
 * @brief MPI benchmarks for distributed DTW distance matrix computation.
 *
 * @details Standalone benchmark (not Google Benchmark) because MPI requires
 *          MPI_Init/MPI_Finalize bracketing. Run with:
 *            mpiexec -n <P> ./bench_mpi_dtw [N] [L] [band]
 *
 *          Measures:
 *          - Wall-clock time for distributed distance matrix computation
 *          - Scaling efficiency across different rank counts
 *          - Comparison with serial (single-process) computation
 *
 * @date 01 Apr 2026
 */

#include <dtwc.hpp>

#ifdef DTWC_HAS_MPI
#include <mpi/mpi_distance_matrix.hpp>
#include <mpi.h>
#endif

#include <vector>
#include <random>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <cmath>

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
                                                         unsigned base_seed = 300)
{
  std::vector<std::vector<double>> vecs;
  vecs.reserve(N);
  for (int i = 0; i < N; ++i)
    vecs.push_back(random_series(static_cast<size_t>(L), base_seed + i));
  return vecs;
}

struct BenchConfig {
  int N = 50;       // number of series
  int L = 500;      // series length
  int band = -1;    // Sakoe-Chiba band (-1 = full)
  int repeats = 3;  // number of timing repeats
};

#ifdef DTWC_HAS_MPI

static double compute_serial_reference(
    const std::vector<std::vector<double>> &series, int band)
{
  const size_t N = series.size();
  auto start = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      double d;
      if (band >= 0)
        d = dtwc::dtwBanded<double>(series[i], series[j], band);
      else
        d = dtwc::dtwFull_L<double>(series[i], series[j]);
      (void)d;
    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  return std::chrono::duration<double>(end - start).count();
}

static void run_mpi_benchmark(const BenchConfig &cfg)
{
  auto [rank, world_size] = dtwc::mpi::mpi_rank_and_size();
  auto series = make_series_set(cfg.N, cfg.L);
  const size_t num_pairs = static_cast<size_t>(cfg.N) * (cfg.N - 1) / 2;

  dtwc::mpi::MPIDistMatOptions opts;
  opts.band = cfg.band;
  opts.verbose = false;

  // Warm-up
  { auto warmup = dtwc::mpi::compute_distance_matrix_mpi(series, opts); }
  MPI_Barrier(MPI_COMM_WORLD);

  // Timed runs
  double total_time = 0.0;
  double min_time = 1e30, max_time = 0.0;

  for (int r = 0; r < cfg.repeats; ++r) {
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    auto result = dtwc::mpi::compute_distance_matrix_mpi(series, opts);
    auto end = std::chrono::high_resolution_clock::now();

    double elapsed = std::chrono::duration<double>(end - start).count();
    total_time += elapsed;
    min_time = std::min(min_time, elapsed);
    max_time = std::max(max_time, elapsed);
  }

  double avg_time = total_time / cfg.repeats;

  // Serial reference (only rank 0)
  double serial_time = 0.0;
  if (rank == 0) {
    serial_time = compute_serial_reference(series, cfg.band);
  }

  if (rank == 0) {
    double pairs_per_sec = static_cast<double>(num_pairs) / avg_time;
    double cells_per_sec = pairs_per_sec * cfg.L * cfg.L;

    std::cout << "=== MPI DTW Benchmark ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Config:       N=" << cfg.N << ", L=" << cfg.L
              << ", band=" << cfg.band
              << ", ranks=" << world_size << std::endl;
    std::cout << "Total pairs:  " << num_pairs << std::endl;
    std::cout << "MPI time:     " << avg_time * 1000.0 << " ms"
              << " (min=" << min_time * 1000.0
              << ", max=" << max_time * 1000.0 << ")" << std::endl;
    std::cout << "Serial time:  " << serial_time * 1000.0 << " ms" << std::endl;
    std::cout << "Speedup:      " << serial_time / avg_time << "x"
              << " (ideal " << world_size << "x)" << std::endl;
    std::cout << "Efficiency:   "
              << (serial_time / avg_time / world_size) * 100.0 << "%" << std::endl;
    std::cout << "Throughput:   " << pairs_per_sec << " pairs/sec" << std::endl;
    std::cout << "              " << cells_per_sec / 1e9
              << " Gcells/sec" << std::endl;
    std::cout << std::endl;
  }
}

#endif // DTWC_HAS_MPI

// Suppress unused function warning when MPI is disabled
namespace { struct dummy_ref { void operator()() {
  (void)random_series; (void)make_series_set;
}};
}

int main(int argc, char *argv[])
{
#ifdef DTWC_HAS_MPI
  MPI_Init(&argc, &argv);

  // Benchmark suite: multiple configurations
  struct { int N; int L; int band; const char* label; } configs[] = {
    {  20, 100,  -1, "Small (20x100, full)" },
    {  50, 100,  -1, "Medium-N (50x100, full)" },
    {  50, 500,  -1, "Medium (50x500, full)" },
    { 100, 500,  -1, "Large-N (100x500, full)" },
    { 200, 500,  -1, "XL-N (200x500, full)" },
    {  50, 1000, -1, "Long-L (50x1000, full)" },
    {  50, 500,  50, "Banded (50x500, band=50)" },
    { 100, 1000, -1, "Large (100x1000, full)" },
  };

  auto [rank, _ws] = dtwc::mpi::mpi_rank_and_size();
  if (rank == 0) {
    std::cout << "MPI DTW Distance Matrix Benchmark" << std::endl;
    std::cout << "Ranks: " << _ws << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    std::cout << std::endl;
  }

  // Parse optional override from command line: N L [band]
  if (argc >= 3) {
    BenchConfig cfg;
    cfg.N = std::atoi(argv[1]);
    cfg.L = std::atoi(argv[2]);
    if (argc >= 4) cfg.band = std::atoi(argv[3]);
    run_mpi_benchmark(cfg);
  } else {
    for (auto &c : configs) {
      BenchConfig cfg;
      cfg.N = c.N;
      cfg.L = c.L;
      cfg.band = c.band;
      if (rank == 0)
        std::cout << "--- " << c.label << " ---" << std::endl;
      run_mpi_benchmark(cfg);
    }
  }

  MPI_Finalize();
#else
  (void)argc; (void)argv;
  std::cout << "MPI not available in this build." << std::endl;
  return 1;
#endif
  return 0;
}
