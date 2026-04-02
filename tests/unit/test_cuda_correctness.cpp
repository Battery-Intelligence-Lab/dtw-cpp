/**
 * @file test_cuda_correctness.cpp
 * @brief CUDA DTW correctness tests: compare GPU distance matrix against CPU reference.
 *
 * @date 01 Apr 2026
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <dtwc.hpp>

#ifdef DTWC_HAS_CUDA
#include <cuda/cuda_dtw.cuh>
// gpu_config.cuh is an internal header — tested indirectly via compute_distance_matrix_cuda()
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

/// Compute the NxN CPU banded distance matrix using dtwBanded (L1 metric).
std::vector<double> cpu_banded_distance_matrix(
    const std::vector<std::vector<double>> &series, int band)
{
  const size_t N = series.size();
  std::vector<double> mat(N * N, 0.0);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      double d = dtwc::dtwBanded<double>(series[i], series[j], band);
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

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
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

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
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

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
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

// ---------------------------------------------------------------------------
// Banded DTW: GPU vs CPU comparison
// ---------------------------------------------------------------------------

TEST_CASE("test_gpu_banded_matches_cpu_small", "[cuda][banded]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 50;
  constexpr int band = 5;
  auto series = generate_random_series(N, L, /*seed=*/42);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = band;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;

  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  auto cpu_mat    = cpu_banded_distance_matrix(series, band);

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

TEST_CASE("test_gpu_banded_matches_cpu_medium", "[cuda][banded]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 15;
  constexpr size_t L = 200;
  constexpr int band = 20;
  auto series = generate_random_series(N, L, /*seed=*/123);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = band;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;

  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  auto cpu_mat    = cpu_banded_distance_matrix(series, band);

  REQUIRE(gpu_result.n == N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_gpu_banded_narrow_band", "[cuda][banded]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 8;
  constexpr size_t L = 100;
  constexpr int band = 2;
  auto series = generate_random_series(N, L, /*seed=*/555);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = band;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;

  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  auto cpu_mat    = cpu_banded_distance_matrix(series, band);

  REQUIRE(gpu_result.n == N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_gpu_banded_negative_is_full_dtw", "[cuda][banded]")
{
  // band < 0 should produce full (unconstrained) DTW, identical to default
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 50;
  auto series = generate_random_series(N, L, /*seed=*/42);

  dtwc::cuda::CUDADistMatOptions opts_full;
  opts_full.band = -1;

  dtwc::cuda::CUDADistMatOptions opts_default;
  // default band is -1

  auto gpu_full    = dtwc::cuda::compute_distance_matrix_cuda(series, opts_full);
  auto gpu_default = dtwc::cuda::compute_distance_matrix_cuda(series, opts_default);

  REQUIRE(gpu_full.n == N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE(gpu_full.matrix[i * N + j] == gpu_default.matrix[j * N + i]);
    }
  }
}

TEST_CASE("test_gpu_banded_wide_band_equals_full", "[cuda][banded]")
{
  // A band wider than series length should give same result as full DTW
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 8;
  constexpr size_t L = 30;
  constexpr int band = 500; // much wider than L
  auto series = generate_random_series(N, L, /*seed=*/777);

  dtwc::cuda::CUDADistMatOptions opts_banded;
  opts_banded.band = band;
  opts_banded.precision = dtwc::cuda::CUDAPrecision::FP64;

  auto gpu_banded = dtwc::cuda::compute_distance_matrix_cuda(series, opts_banded);
  auto cpu_full   = cpu_distance_matrix(series);

  REQUIRE(gpu_banded.n == N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_banded.matrix[i * N + j],
                   WithinRel(cpu_full[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_gpu_banded_unequal_lengths", "[cuda][banded]")
{
  // Test banded DTW with series of different lengths
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  std::mt19937 rng(314);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);

  std::vector<std::vector<double>> series(6);
  // Varying lengths: 30, 50, 40, 60, 35, 55
  const size_t lens[] = {30, 50, 40, 60, 35, 55};
  for (size_t s = 0; s < 6; ++s) {
    series[s].resize(lens[s]);
    for (auto &v : series[s]) v = dist(rng);
  }

  constexpr int band = 8;

  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = band;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;

  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  auto cpu_mat    = cpu_banded_distance_matrix(series, band);

  REQUIRE(gpu_result.n == 6);

  for (size_t i = 0; i < 6; ++i) {
    for (size_t j = 0; j < 6; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * 6 + j],
                   WithinRel(cpu_mat[i * 6 + j], 1e-10));
    }
  }
}

// ---------------------------------------------------------------------------
// Warp-level kernel tests (short series, L <= 32)
// ---------------------------------------------------------------------------

TEST_CASE("test_warp_kernel_short_series_L8", "[cuda][warp]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 20;
  constexpr size_t L = 8;
  auto series = generate_random_series(N, L, /*seed=*/1001);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_warp_kernel_short_series_L16", "[cuda][warp]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 15;
  constexpr size_t L = 16;
  auto series = generate_random_series(N, L, /*seed=*/2002);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_warp_kernel_short_series_L32", "[cuda][warp]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 20;
  constexpr size_t L = 32;
  auto series = generate_random_series(N, L, /*seed=*/3003);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_warp_kernel_short_series_L1", "[cuda][warp]")
{
  // Edge case: single-element series through the warp kernel
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  std::vector<std::vector<double>> series = {{3.0}, {7.0}, {1.0}, {5.0}};
  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);

  const size_t N = series.size();
  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      double expected = std::abs(series[i][0] - series[j][0]);
      INFO("i=" << i << " j=" << j);
      REQUIRE(gpu_result.matrix[i * N + j] == Catch::Approx(expected));
      REQUIRE(gpu_result.matrix[j * N + i] == Catch::Approx(expected));
    }
  }
}

TEST_CASE("test_warp_kernel_variable_short_lengths", "[cuda][warp]")
{
  // Variable lengths all <= 32 to exercise the warp kernel
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  std::mt19937 rng(4004);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);

  std::vector<std::vector<double>> series(8);
  const size_t lens[] = {5, 10, 15, 20, 25, 30, 8, 12};
  for (size_t s = 0; s < 8; ++s) {
    series[s].resize(lens[s]);
    for (auto &v : series[s]) v = dist(rng);
  }

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  const size_t N = series.size();
  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_warp_kernel_banded_short_series", "[cuda][warp][banded]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 12;
  constexpr size_t L = 20;
  constexpr int band = 3;
  auto series = generate_random_series(N, L, /*seed=*/5005);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = band;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;

  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  auto cpu_mat    = cpu_banded_distance_matrix(series, band);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_warp_kernel_fp32_short_series", "[cuda][warp][fp32]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 15;
  constexpr size_t L = 24;
  auto series = generate_random_series(N, L, /*seed=*/6006);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP32;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-4));
    }
  }
}

TEST_CASE("test_warp_kernel_many_pairs_short_series", "[cuda][warp]")
{
  // Stress test: many pairs to ensure multi-block dispatch works
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 50;   // 1225 pairs, ~154 blocks of 8
  constexpr size_t L = 16;
  auto series = generate_random_series(N, L, /*seed=*/7007);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);

  REQUIRE(gpu_result.n == N);
  REQUIRE(gpu_result.pairs_computed == N * (N - 1) / 2);

  // Spot-check 25 pairs against CPU
  for (size_t k = 0; k < 25; ++k) {
    size_t i = k;
    size_t j = N - 1 - k;
    double cpu_d = dtwc::dtwFull_L<double>(series[i], series[j]);
    double gpu_d = gpu_result.matrix[i * N + j];
    INFO("k=" << k << " i=" << i << " j=" << j);
    REQUIRE_THAT(gpu_d, WithinRel(cpu_d, 1e-10));
  }

  // Symmetry check on a sample
  for (size_t i = 0; i < 10; ++i)
    for (size_t j = i + 1; j < 10; ++j)
      REQUIRE(gpu_result.matrix[i * N + j] == gpu_result.matrix[j * N + i]);
}

TEST_CASE("test_warp_kernel_squared_l2_short_series", "[cuda][warp]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 20;
  auto series = generate_random_series(N, L, /*seed=*/8008);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.use_squared_l2 = true;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      double cpu_d = dtwc::dtwFull_L<double>(series[i], series[j], -1.0,
                                              dtwc::core::MetricType::SquaredL2);
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_d, 1e-10));
    }
  }
}

// ---------------------------------------------------------------------------
// FP32 precision tests: looser tolerance due to single-precision accumulation
// ---------------------------------------------------------------------------

TEST_CASE("test_gpu_fp32_matches_cpu_small", "[cuda][fp32]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 50;
  auto series = generate_random_series(N, L, /*seed=*/42);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP32;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  REQUIRE(gpu_result.matrix.size() == N * N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      // FP32 accumulation: relative tolerance ~1e-5
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-4));
    }
  }
}

TEST_CASE("test_gpu_fp32_matches_cpu_medium", "[cuda][fp32]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 15;
  constexpr size_t L = 200;
  auto series = generate_random_series(N, L, /*seed=*/123);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP32;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-4));
    }
  }
}

TEST_CASE("test_gpu_fp32_symmetry", "[cuda][fp32]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 12;
  constexpr size_t L = 80;
  auto series = generate_random_series(N, L, /*seed=*/3333);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP32;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);

  REQUIRE(gpu_result.n == N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE(gpu_result.matrix[i * N + j] == gpu_result.matrix[j * N + i]);
    }
  }
}

TEST_CASE("test_gpu_fp32_identical_series_zero_distance", "[cuda][fp32]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  auto series = generate_random_series(1, 60, /*seed=*/22);
  series.push_back(series[0]);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP32;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);

  REQUIRE(gpu_result.n == 2);
  REQUIRE(gpu_result.matrix[0 * 2 + 1] == 0.0);
  REQUIRE(gpu_result.matrix[1 * 2 + 0] == 0.0);
}

TEST_CASE("test_gpu_fp32_banded_matches_cpu", "[cuda][fp32][banded]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 50;
  constexpr int band = 5;
  auto series = generate_random_series(N, L, /*seed=*/42);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = band;
  opts.precision = dtwc::cuda::CUDAPrecision::FP32;

  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  auto cpu_mat    = cpu_banded_distance_matrix(series, band);

  REQUIRE(gpu_result.n == N);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-4));
    }
  }
}

// ---------------------------------------------------------------------------
// GPU config detection
// ---------------------------------------------------------------------------

// GPU config (query_gpu_config) is an internal API tested indirectly
// through compute_distance_matrix_cuda's precision auto-detection.
// Direct testing would require linking cudart to the test binary.

// ---------------------------------------------------------------------------
// Squared-L2 metric: GPU vs CPU
// ---------------------------------------------------------------------------

TEST_CASE("GPU matches CPU with squared-L2 metric", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 50;
  auto series = generate_random_series(N, L, /*seed=*/500);

  // GPU with squared L2
  dtwc::cuda::CUDADistMatOptions opts;
  opts.use_squared_l2 = true;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);

  REQUIRE(gpu_result.n == N);
  REQUIRE(gpu_result.matrix.size() == N * N);

  // CPU reference with squared L2
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      double cpu_d = dtwc::dtwFull_L<double>(series[i], series[j], -1.0,
                                              dtwc::core::MetricType::SquaredL2);
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_d, 1e-10));
      REQUIRE_THAT(gpu_result.matrix[j * N + i],
                   WithinRel(cpu_d, 1e-10));
    }
  }
}

// ---------------------------------------------------------------------------
// Variable-length series
// ---------------------------------------------------------------------------

TEST_CASE("GPU handles variable-length series", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  // Series with different lengths
  std::vector<std::vector<double>> series;
  std::mt19937 rng(600);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (int len : {30, 50, 80, 100, 60, 40}) {
    std::vector<double> s(static_cast<size_t>(len));
    for (auto &v : s) v = dist(rng);
    series.push_back(std::move(s));
  }

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  const size_t N = series.size();

  REQUIRE(gpu_result.n == N);

  // Compare with CPU
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      double cpu_d = dtwc::dtwFull_L<double>(series[i], series[j]);
      double gpu_d = gpu_result.matrix[i * N + j];
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_d, WithinRel(cpu_d, 1e-10));
    }
  }

  // Also check symmetry
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      REQUIRE(gpu_result.matrix[i * N + j] == gpu_result.matrix[j * N + i]);
}

// ---------------------------------------------------------------------------
// Edge case: single pair (N=2)
// ---------------------------------------------------------------------------

TEST_CASE("GPU single pair (N=2)", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  std::vector<std::vector<double>> series = {{1.0, 2.0, 3.0}, {4.0, 5.0, 6.0}};
  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);

  double cpu_d = dtwc::dtwFull_L<double>(series[0], series[1]);
  REQUIRE(result.n == 2);
  REQUIRE(result.matrix[0 * 2 + 1] == Catch::Approx(cpu_d).epsilon(1e-10));
  REQUIRE(result.matrix[1 * 2 + 0] == Catch::Approx(cpu_d).epsilon(1e-10));
  REQUIRE(result.matrix[0] == 0.0);
  REQUIRE(result.matrix[3] == 0.0);
}

// ---------------------------------------------------------------------------
// Edge case: length-1 series
// ---------------------------------------------------------------------------

TEST_CASE("GPU length-1 series", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  std::vector<std::vector<double>> series = {{3.0}, {7.0}, {1.0}};
  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);

  REQUIRE(result.n == 3);
  // DTW of single-element series = |a - b| (L1 metric)
  REQUIRE(result.matrix[0 * 3 + 1] == Catch::Approx(4.0));
  REQUIRE(result.matrix[0 * 3 + 2] == Catch::Approx(2.0));
  REQUIRE(result.matrix[1 * 3 + 2] == Catch::Approx(6.0));
  // Symmetry
  REQUIRE(result.matrix[1 * 3 + 0] == Catch::Approx(4.0));
  REQUIRE(result.matrix[2 * 3 + 0] == Catch::Approx(2.0));
  REQUIRE(result.matrix[2 * 3 + 1] == Catch::Approx(6.0));
  // Diagonal
  REQUIRE(result.matrix[0] == 0.0);
  REQUIRE(result.matrix[4] == 0.0);
  REQUIRE(result.matrix[8] == 0.0);
}

// ---------------------------------------------------------------------------
// Register-tiled kernel tests (medium series, 32 < L <= 256)
// ---------------------------------------------------------------------------

TEST_CASE("test_regtile_kernel_L33", "[cuda][regtile]")
{
  // Just above warp kernel threshold — exercises TILE_W=4 with minimal columns
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 33;
  auto series = generate_random_series(N, L, /*seed=*/10001);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_regtile_kernel_L64", "[cuda][regtile]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 12;
  constexpr size_t L = 64;
  auto series = generate_random_series(N, L, /*seed=*/10002);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_regtile_kernel_L128", "[cuda][regtile]")
{
  // Boundary: exactly at TILE_W=4 maximum (32*4=128)
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 128;
  auto series = generate_random_series(N, L, /*seed=*/10003);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_regtile_kernel_L129", "[cuda][regtile]")
{
  // Just above TILE_W=4 — exercises TILE_W=8
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 8;
  constexpr size_t L = 129;
  auto series = generate_random_series(N, L, /*seed=*/10004);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_regtile_kernel_L256", "[cuda][regtile]")
{
  // Boundary: exactly at TILE_W=8 maximum (32*8=256)
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 8;
  constexpr size_t L = 256;
  auto series = generate_random_series(N, L, /*seed=*/10005);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_regtile_kernel_variable_lengths", "[cuda][regtile]")
{
  // Variable lengths spanning the regtile range
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  std::mt19937 rng(10006);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);

  std::vector<std::vector<double>> series(8);
  const size_t lens[] = {35, 50, 70, 100, 40, 80, 60, 90};
  for (size_t s = 0; s < 8; ++s) {
    series[s].resize(lens[s]);
    for (auto &v : series[s]) v = dist(rng);
  }

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  const size_t N = series.size();
  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_regtile_kernel_banded_L100", "[cuda][regtile][banded]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 100;
  constexpr int band = 10;
  auto series = generate_random_series(N, L, /*seed=*/10007);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = band;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;

  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  auto cpu_mat    = cpu_banded_distance_matrix(series, band);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-10));
    }
  }
}

TEST_CASE("test_regtile_kernel_fp32_L100", "[cuda][regtile][fp32]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 100;
  auto series = generate_random_series(N, L, /*seed=*/10008);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP32;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  auto cpu_mat    = cpu_distance_matrix(series);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_mat[i * N + j], 1e-4));
    }
  }
}

TEST_CASE("test_regtile_kernel_squared_l2_L80", "[cuda][regtile]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 10;
  constexpr size_t L = 80;
  auto series = generate_random_series(N, L, /*seed=*/10009);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.use_squared_l2 = true;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);

  REQUIRE(gpu_result.n == N);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      double cpu_d = dtwc::dtwFull_L<double>(series[i], series[j], -1.0,
                                              dtwc::core::MetricType::SquaredL2);
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(gpu_result.matrix[i * N + j],
                   WithinRel(cpu_d, 1e-10));
    }
  }
}

// ---------------------------------------------------------------------------
// Larger stress test
// ---------------------------------------------------------------------------

TEST_CASE("GPU stress test (100 series x 200 length)", "[cuda]")
{
  if (!dtwc::cuda::cuda_available()) { SKIP("No CUDA device"); return; }

  constexpr size_t N = 100;
  constexpr size_t L = 200;
  auto series = generate_random_series(N, L, /*seed=*/700);

  dtwc::cuda::CUDADistMatOptions opts;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto gpu_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  REQUIRE(gpu_result.n == N);
  REQUIRE(gpu_result.pairs_computed == N * (N - 1) / 2);

  // Spot-check 20 pairs against CPU
  for (size_t k = 0; k < 20; ++k) {
    size_t i = k;
    size_t j = N - 1 - k;
    double cpu_d = dtwc::dtwFull_L<double>(series[i], series[j]);
    double gpu_d = gpu_result.matrix[i * N + j];
    INFO("k=" << k << " i=" << i << " j=" << j);
    REQUIRE_THAT(gpu_d, WithinRel(cpu_d, 1e-10));
  }

  // Verify diagonal is zero
  for (size_t i = 0; i < N; ++i)
    REQUIRE(gpu_result.matrix[i * N + i] == 0.0);

  // Verify symmetry on a sample
  for (size_t i = 0; i < 10; ++i)
    for (size_t j = i + 1; j < 10; ++j)
      REQUIRE(gpu_result.matrix[i * N + j] == gpu_result.matrix[j * N + i]);
}

#endif // DTWC_HAS_CUDA
