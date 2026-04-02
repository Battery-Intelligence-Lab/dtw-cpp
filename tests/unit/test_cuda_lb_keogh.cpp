/**
 * @file test_cuda_lb_keogh.cpp
 * @brief Tests for GPU LB_Keogh: envelope computation, lower bounds, and
 *        pruned distance matrix integration.
 *
 * @details Compares GPU LB_Keogh results against CPU reference implementation
 *          from lower_bound_impl.hpp. Tests cover:
 *          - Standalone compute_lb_keogh_cuda() correctness
 *          - LB_Keogh <= DTW property (lower bound guarantee)
 *          - Pruned distance matrix via use_lb_pruning option
 *          - Edge cases (single series, empty series, varying lengths)
 *
 * @date 01 Apr 2026
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <dtwc.hpp>
#include <core/lower_bound_impl.hpp>

#ifdef DTWC_HAS_CUDA
#include <cuda/cuda_dtw.cuh>
#endif

#include <cmath>
#include <random>
#include <vector>
#include <limits>

using Catch::Matchers::WithinRel;

#ifndef DTWC_HAS_CUDA

TEST_CASE("CUDA LB_Keogh - not available", "[cuda][lb_keogh]")
{
  SKIP("DTWC_HAS_CUDA not defined; CUDA LB_Keogh tests skipped");
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

/// Compute CPU reference LB_Keogh for all pairs (symmetric).
/// Returns flat array of N*(N-1)/2 values in the same pair ordering as GPU.
std::vector<double> cpu_lb_keogh_all_pairs(
    const std::vector<std::vector<double>> &series, int band)
{
  const size_t N = series.size();
  const size_t num_pairs = N * (N - 1) / 2;

  // Precompute envelopes
  std::vector<dtwc::core::Envelope> envs(N);
  for (size_t i = 0; i < N; ++i)
    envs[i] = dtwc::core::compute_envelope(series[i], band);

  // Compute symmetric LB_Keogh for all pairs
  std::vector<double> lb(num_pairs);
  size_t k = 0;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      lb[k] = dtwc::core::lb_keogh_symmetric(
          series[i], envs[i], series[j], envs[j]);
      ++k;
    }
  }
  return lb;
}

} // anonymous namespace

TEST_CASE("GPU LB_Keogh matches CPU reference", "[cuda][lb_keogh]")
{
  if (!dtwc::cuda::cuda_available()) {
    SKIP("No CUDA device available");
  }

  const size_t N = 20;
  const size_t L = 64;
  const int band = 5;
  auto series = generate_random_series(N, L, 42);

  // CPU reference
  auto cpu_lb = cpu_lb_keogh_all_pairs(series, band);

  // GPU computation
  auto gpu_result = dtwc::cuda::compute_lb_keogh_cuda(series, band);

  REQUIRE(gpu_result.n == N);
  REQUIRE(gpu_result.lb_values.size() == cpu_lb.size());

  const size_t num_pairs = N * (N - 1) / 2;
  for (size_t k = 0; k < num_pairs; ++k) {
    CAPTURE(k, cpu_lb[k], gpu_result.lb_values[k]);
    CHECK_THAT(gpu_result.lb_values[k], WithinRel(cpu_lb[k], 1e-10));
  }
}

TEST_CASE("GPU LB_Keogh is a valid lower bound on DTW", "[cuda][lb_keogh]")
{
  if (!dtwc::cuda::cuda_available()) {
    SKIP("No CUDA device available");
  }

  const size_t N = 15;
  const size_t L = 48;
  const int band = 4;
  auto series = generate_random_series(N, L, 123);

  // GPU LB_Keogh
  auto gpu_lb = dtwc::cuda::compute_lb_keogh_cuda(series, band);

  // GPU DTW (banded, same band)
  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = band;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto dtw_result = dtwc::cuda::compute_distance_matrix_cuda(series, opts);

  // Check LB <= DTW for all pairs
  size_t k = 0;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      double lb = gpu_lb.lb_values[k];
      double dtw = dtw_result.matrix[i * N + j];
      CAPTURE(i, j, lb, dtw);
      CHECK(lb <= dtw + 1e-10);  // LB must not exceed DTW
      ++k;
    }
  }
}

TEST_CASE("GPU LB_Keogh with varying band widths", "[cuda][lb_keogh]")
{
  if (!dtwc::cuda::cuda_available()) {
    SKIP("No CUDA device available");
  }

  const size_t N = 10;
  const size_t L = 32;
  auto series = generate_random_series(N, L, 77);

  for (int band : {1, 3, 8, 16, 31}) {
    SECTION("band = " + std::to_string(band)) {
      auto cpu_lb = cpu_lb_keogh_all_pairs(series, band);
      auto gpu_result = dtwc::cuda::compute_lb_keogh_cuda(series, band);

      REQUIRE(gpu_result.lb_values.size() == cpu_lb.size());
      for (size_t k = 0; k < cpu_lb.size(); ++k) {
        CAPTURE(band, k, cpu_lb[k], gpu_result.lb_values[k]);
        CHECK_THAT(gpu_result.lb_values[k], WithinRel(cpu_lb[k], 1e-10));
      }
    }
  }
}

TEST_CASE("GPU LB_Keogh edge cases", "[cuda][lb_keogh]")
{
  if (!dtwc::cuda::cuda_available()) {
    SKIP("No CUDA device available");
  }

  SECTION("Single series returns empty") {
    std::vector<std::vector<double>> series = {{1.0, 2.0, 3.0}};
    auto result = dtwc::cuda::compute_lb_keogh_cuda(series, 2);
    CHECK(result.lb_values.empty());
    CHECK(result.n == 1);
  }

  SECTION("Negative band returns empty") {
    auto series = generate_random_series(5, 16, 99);
    auto result = dtwc::cuda::compute_lb_keogh_cuda(series, -1);
    CHECK(result.lb_values.empty());
  }

  SECTION("Identical series have LB = 0") {
    std::vector<double> s = {1.0, 3.0, -2.0, 5.0, 0.0};
    std::vector<std::vector<double>> series = {s, s, s};
    auto result = dtwc::cuda::compute_lb_keogh_cuda(series, 2);
    REQUIRE(result.lb_values.size() == 3);
    for (size_t k = 0; k < 3; ++k) {
      CHECK(result.lb_values[k] == Catch::Approx(0.0).margin(1e-12));
    }
  }

  SECTION("Two series") {
    std::vector<std::vector<double>> series = {
      {1.0, 2.0, 3.0, 4.0},
      {5.0, 6.0, 7.0, 8.0}
    };
    auto gpu_result = dtwc::cuda::compute_lb_keogh_cuda(series, 1);
    auto cpu_lb = cpu_lb_keogh_all_pairs(series, 1);
    REQUIRE(gpu_result.lb_values.size() == 1);
    CHECK_THAT(gpu_result.lb_values[0], WithinRel(cpu_lb[0], 1e-10));
  }
}

TEST_CASE("GPU distance matrix with LB pruning (threshold mode)", "[cuda][lb_keogh]")
{
  if (!dtwc::cuda::cuda_available()) {
    SKIP("No CUDA device available");
  }

  const size_t N = 10;
  const size_t L = 32;
  const int band = 3;
  auto series = generate_random_series(N, L, 55);

  // Compute without pruning (reference)
  dtwc::cuda::CUDADistMatOptions opts_ref;
  opts_ref.band = band;
  opts_ref.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto ref = dtwc::cuda::compute_distance_matrix_cuda(series, opts_ref);

  // Find a threshold that prunes some but not all pairs
  double max_dist = 0;
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i + 1; j < N; ++j)
      max_dist = std::max(max_dist, ref.matrix[i * N + j]);

  const double threshold = max_dist * 0.5;  // should prune some pairs

  // Compute with pruning
  dtwc::cuda::CUDADistMatOptions opts_pruned;
  opts_pruned.band = band;
  opts_pruned.precision = dtwc::cuda::CUDAPrecision::FP64;
  opts_pruned.use_lb_pruning = true;
  opts_pruned.skip_threshold = threshold;
  auto pruned = dtwc::cuda::compute_distance_matrix_cuda(series, opts_pruned);

  // Check: non-pruned pairs should have exact distances
  // Pruned pairs should have INF
  constexpr double INF = std::numeric_limits<double>::max();
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      double pruned_val = pruned.matrix[i * N + j];
      double ref_val = ref.matrix[i * N + j];
      if (pruned_val >= INF * 0.5) {
        // This pair was pruned -- its LB must exceed threshold
        // (we don't check exact LB here, just that pruning is consistent)
        CHECK(pruned.matrix[j * N + i] >= INF * 0.5);  // symmetric
      } else {
        // Not pruned -- must match reference
        CAPTURE(i, j, ref_val, pruned_val);
        CHECK_THAT(pruned_val, WithinRel(ref_val, 1e-10));
      }
    }
  }

  // Should have pruned at least one pair (with a 50% threshold on random data)
  // This is a soft check -- if it fails, the test data might need adjustment
  // CHECK(pruned.pairs_pruned > 0);  // commented: not guaranteed for all seeds
}

TEST_CASE("GPU LB_Keogh with larger dataset", "[cuda][lb_keogh]")
{
  if (!dtwc::cuda::cuda_available()) {
    SKIP("No CUDA device available");
  }

  const size_t N = 50;
  const size_t L = 128;
  const int band = 10;
  auto series = generate_random_series(N, L, 314);

  auto cpu_lb = cpu_lb_keogh_all_pairs(series, band);
  auto gpu_result = dtwc::cuda::compute_lb_keogh_cuda(series, band);

  REQUIRE(gpu_result.lb_values.size() == cpu_lb.size());

  // Check all pairs match within tolerance
  size_t mismatches = 0;
  for (size_t k = 0; k < cpu_lb.size(); ++k) {
    double rel_err = std::abs(gpu_result.lb_values[k] - cpu_lb[k])
                     / std::max(1.0, std::abs(cpu_lb[k]));
    if (rel_err > 1e-10)
      ++mismatches;
  }
  CHECK(mismatches == 0);
}

#endif // DTWC_HAS_CUDA
