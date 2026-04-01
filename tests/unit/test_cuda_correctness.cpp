/**
 * @file test_cuda_correctness.cpp
 * @brief CUDA DTW correctness tests: compare GPU distance matrix against CPU reference.
 *
 * @date 01 Apr 2026
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <dtwc.hpp>

#ifdef DTWC_HAS_CUDA
#include <cuda/cuda_dtw.cuh>
#endif

#include <random>
#include <vector>

using Catch::Matchers::WithinRel;

#ifndef DTWC_HAS_CUDA

TEST_CASE("CUDA not available", "[cuda]")
{
  SKIP("DTWC_HAS_CUDA not defined; CUDA tests skipped");
}

#else // DTWC_HAS_CUDA

namespace {

/// Generate N random series of the given length using a fixed seed.
std::vector<std::vector<double>> generate_random_series(
    size_t n, size_t length, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);

  std::vector<std::vector<double>> series(n);
  for (auto &s : series) {
    s.resize(length);
    for (auto &v : s)
      v = dist(rng);
  }
  return series;
}

/// Compute the full NxN CPU distance matrix using dtwFull_L (L1 metric).
std::vector<double> cpu_distance_matrix(
    const std::vector<std::vector<double>> &series)
{
  const size_t N = series.size();
  std::vector<double> mat(N * N, 0.0);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      double d = dtwc::dtwFull_L<double>(series[i], series[j]);
      mat[i * N + j] = d;
      mat[j * N + i] = d;
    }
  }
  return mat;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Element-wise GPU vs CPU comparison helpers
// ---------------------------------------------------------------------------

TEST_CASE("test_gpu_matches_cpu_small", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 50;
  auto series = generate_random_series(N, L, /*seed=*/42);

  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  REQUIRE(gpu_result.matrix.size() == N * N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_gpu_matches_cpu_medium", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 20;
  constexpr size_t L = 200;
  auto series = generate_random_series(N, L, /*seed=*/123);

  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  REQUIRE(gpu_result.matrix.size() == N * N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_gpu_matches_cpu_large", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 50;
  constexpr size_t L = 500;
  auto series = generate_random_series(N, L, /*seed=*/9999);

  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  REQUIRE(gpu_result.matrix.size() == N * N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

// ---------------------------------------------------------------------------
// Structural properties of the distance matrix
// ---------------------------------------------------------------------------

TEST_CASE("test_gpu_symmetry", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 15;
  constexpr size_t L = 80;
  auto series = generate_random_series(N, L, /*seed=*/7777);

  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series);

  REQUIRE(gpu_result.n == N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE(gpu_result.matrix[i * N + j] == gpu_result.matrix[j * N + i]);
    }
  }
}

TEST_CASE("test_gpu_diagonal_zero", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 15;
  constexpr size_t L = 80;
  auto series = generate_random_series(N, L, /*seed=*/5555);

  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series);

  REQUIRE(gpu_result.n == N);

  for (size_t i = 0; i < N; ++i) {
    INFO("i=" << i);
    REQUIRE(gpu_result.matrix[i * N + i] == 0.0);
  }
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

TEST_CASE("test_gpu_single_series", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  auto series = generate_random_series(1, 30, /*seed=*/11);

  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series);

  REQUIRE(gpu_result.n == 1);
  REQUIRE(gpu_result.matrix.size() == 1);
  REQUIRE(gpu_result.matrix[0] == 0.0);
}

TEST_CASE("test_gpu_two_identical", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  auto series = generate_random_series(1, 60, /*seed=*/22);
  // Duplicate the single series so we have two identical ones.
  series.push_back(series[0]);

  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series);

  REQUIRE(gpu_result.n == 2);
  REQUIRE(gpu_result.matrix.size() == 4);

  // Diagonal must be zero.
  REQUIRE(gpu_result.matrix[0 * 2 + 0] == 0.0);
  REQUIRE(gpu_result.matrix[1 * 2 + 1] == 0.0);

  // Off-diagonal: identical series should have zero distance.
  REQUIRE(gpu_result.matrix[0 * 2 + 1] == 0.0);
  REQUIRE(gpu_result.matrix[1 * 2 + 0] == 0.0);
}

#endif // DTWC_HAS_CUDA
