/**
 * @file unit_test_mpi.cpp
 * @brief Tests for MPI distributed distance matrix computation.
 *
 * Run with: mpiexec -n 4 ./bin/unit_test_mpi
 * Also works with -n 1 (single process) and -n 2.
 *
 * This test has its own main() because MPI requires MPI_Init/MPI_Finalize
 * bracketing, which is incompatible with Catch2's main.
 */

#ifdef DTWC_HAS_MPI

#include <mpi.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <string>

#include <dtwc.hpp>

// ---- Minimal test framework (Catch2 doesn't mix with MPI main) ----

static int g_test_count = 0;
static int g_pass_count = 0;
static int g_fail_count = 0;

#define MPI_CHECK(cond, msg)                                                     \
  do {                                                                           \
    g_test_count++;                                                              \
    if (cond) {                                                                  \
      g_pass_count++;                                                            \
    } else {                                                                     \
      g_fail_count++;                                                            \
      std::cerr << "FAIL [rank " << g_rank << "]: " << (msg) << std::endl;      \
    }                                                                            \
  } while (0)

static int g_rank = 0;

// ---- Helpers ----

static std::vector<double> make_series(size_t len, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  std::vector<double> s(len);
  for (auto &v : s)
    v = dist(rng);
  return s;
}

// ---- Tests ----

void test_basic_properties()
{
  // Generate identical data on all ranks (same seeds)
  const size_t N = 10;
  const size_t L = 50;
  std::vector<std::vector<double>> series;
  for (size_t i = 0; i < N; ++i)
    series.push_back(make_series(L, 100 + static_cast<unsigned>(i)));

  auto result = dtwc::mpi::compute_distance_matrix_mpi(series);

  MPI_CHECK(result.n == N, "n matches input size");
  MPI_CHECK(result.matrix.size() == N * N, "matrix has N*N elements");

  // Diagonal must be zero
  for (size_t i = 0; i < N; ++i)
    MPI_CHECK(result.matrix[i * N + i] == 0.0,
              "diagonal zero at i=" + std::to_string(i));

  // Symmetry
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i + 1; j < N; ++j)
      MPI_CHECK(std::abs(result.matrix[i * N + j] - result.matrix[j * N + i]) < 1e-12,
                "symmetry at (" + std::to_string(i) + "," + std::to_string(j) + ")");

  // Non-negative
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      MPI_CHECK(result.matrix[i * N + j] >= 0.0,
                "non-negative at (" + std::to_string(i) + "," + std::to_string(j) + ")");

  // Off-diagonal must be positive (distinct random series)
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i + 1; j < N; ++j)
      MPI_CHECK(result.matrix[i * N + j] > 0.0,
                "positive off-diagonal at (" + std::to_string(i) + "," + std::to_string(j) + ")");
}

void test_matches_serial()
{
  const size_t N = 8;
  const size_t L = 30;
  std::vector<std::vector<double>> series;
  for (size_t i = 0; i < N; ++i)
    series.push_back(make_series(L, 200 + static_cast<unsigned>(i)));

  // MPI-distributed result
  auto mpi_result = dtwc::mpi::compute_distance_matrix_mpi(series);

  // Serial reference (every rank computes independently — same result)
  std::vector<double> serial_matrix(N * N, 0.0);
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i + 1; j < N; ++j) {
      const double d = dtwc::dtwFull_L<double>(series[i], series[j]);
      serial_matrix[i * N + j] = d;
      serial_matrix[j * N + i] = d;
    }

  // Compare element-wise
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      MPI_CHECK(std::abs(mpi_result.matrix[i * N + j] - serial_matrix[i * N + j]) < 1e-10,
                "MPI==serial at (" + std::to_string(i) + "," + std::to_string(j) + ")");
}

void test_banded_matches_serial()
{
  const size_t N = 6;
  const size_t L = 40;
  const int band = 5;
  std::vector<std::vector<double>> series;
  for (size_t i = 0; i < N; ++i)
    series.push_back(make_series(L, 300 + static_cast<unsigned>(i)));

  dtwc::mpi::MPIDistMatOptions opts;
  opts.band = band;
  auto mpi_result = dtwc::mpi::compute_distance_matrix_mpi(series, opts);

  // Serial banded reference
  std::vector<double> serial_matrix(N * N, 0.0);
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i + 1; j < N; ++j) {
      const double d = dtwc::dtwBanded<double>(series[i], series[j], band);
      serial_matrix[i * N + j] = d;
      serial_matrix[j * N + i] = d;
    }

  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      MPI_CHECK(std::abs(mpi_result.matrix[i * N + j] - serial_matrix[i * N + j]) < 1e-10,
                "banded MPI==serial at (" + std::to_string(i) + "," + std::to_string(j) + ")");
}

void test_single_series()
{
  std::vector<std::vector<double>> series = { make_series(20, 42) };
  auto result = dtwc::mpi::compute_distance_matrix_mpi(series);
  MPI_CHECK(result.n == 1, "single series n==1");
  MPI_CHECK(result.matrix.size() == 1, "single series matrix size==1");
  MPI_CHECK(result.matrix[0] == 0.0, "single series distance==0");
}

void test_two_series()
{
  std::vector<std::vector<double>> series = {
    make_series(25, 1),
    make_series(25, 2)
  };
  auto result = dtwc::mpi::compute_distance_matrix_mpi(series);
  MPI_CHECK(result.n == 2, "two series n==2");
  MPI_CHECK(result.matrix[0] == 0.0, "d(0,0)==0");
  MPI_CHECK(result.matrix[3] == 0.0, "d(1,1)==0");
  MPI_CHECK(result.matrix[1] > 0.0, "d(0,1)>0");
  MPI_CHECK(std::abs(result.matrix[1] - result.matrix[2]) < 1e-12, "d(0,1)==d(1,0)");

  // Verify against serial
  const double d_serial = dtwc::dtwFull_L<double>(series[0], series[1]);
  MPI_CHECK(std::abs(result.matrix[1] - d_serial) < 1e-10, "matches serial for 2 series");
}

void test_all_ranks_agree()
{
  const size_t N = 7;
  const size_t L = 35;
  std::vector<std::vector<double>> series;
  for (size_t i = 0; i < N; ++i)
    series.push_back(make_series(L, 500 + static_cast<unsigned>(i)));

  auto result = dtwc::mpi::compute_distance_matrix_mpi(series);

  // Compute a checksum of the matrix
  double local_sum = 0.0;
  for (const auto &v : result.matrix)
    local_sum += v;

  // All ranks should have the same checksum
  double global_min = 0.0, global_max = 0.0;
  MPI_Allreduce(&local_sum, &global_min, 1, MPI_DOUBLE, MPI_MIN, MPI_COMM_WORLD);
  MPI_Allreduce(&local_sum, &global_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);

  MPI_CHECK(std::abs(global_max - global_min) < 1e-10,
            "all ranks agree on matrix (sum diff=" + std::to_string(global_max - global_min) + ")");
}

// ---- Main ----

int main(int argc, char *argv[])
{
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);

  test_basic_properties();
  test_matches_serial();
  test_banded_matches_serial();
  test_single_series();
  test_two_series();
  test_all_ranks_agree();

  if (g_rank == 0) {
    std::cout << "\n=== MPI Tests: " << g_pass_count << "/" << g_test_count
              << " passed, " << g_fail_count << " failed ===" << std::endl;
  }

  MPI_Finalize();
  return g_fail_count > 0 ? 1 : 0;
}

#else // !DTWC_HAS_MPI

#include <iostream>

int main()
{
  std::cout << "MPI not enabled (DTWC_ENABLE_MPI=OFF). Skipping MPI tests." << std::endl;
  return 0;
}

#endif // DTWC_HAS_MPI
