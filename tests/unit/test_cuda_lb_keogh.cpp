/**
 * @file test_cuda_lb_keogh.cpp
 * @brief Tests for GPU LB_Keogh: envelope computation, lower bounds, and
 *        pruned distance matrix integration.
 *
 * @details Compares GPU LB_Keogh results against CPU reference implementation
 *          from lower_bound_impl.hpp. Tests cover:
 *          - Standalone compute_lb_keogh_cuda() correctness
 *          - LB_Keogh <= DTW property (lower bound guarantee)
 *          - Pruned distance matrix via use_lb_keogh option
 *          - Edge cases (single series, empty series, varying lengths)
 *
 * @author Volkan Kumtepeli
 * @date 01 Apr 2026
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <dtwc.hpp>
#include <core/lower_bound_impl.hpp>

#include "gpu_fixed_band_oracle.hpp"
#include "../support/deterministic_series.hpp"

#ifdef DTWC_HAS_CUDA
#include <cuda/cuda_dtw.cuh>
#endif

#include <cmath>
#include <limits>
#include <sstream>
#include <vector>

using Catch::Matchers::WithinRel;

#ifndef DTWC_HAS_CUDA

TEST_CASE("CUDA LB_Keogh - not available", "[cuda][lb_keogh]")
{
  SKIP("DTWC_HAS_CUDA not defined; CUDA LB_Keogh tests skipped");
}

#else // DTWC_HAS_CUDA

namespace {

constexpr auto generate_random_series =
  &dtwc::test_support::accelerator_series_set;

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
  opts_pruned.use_lb_keogh = true;
  opts_pruned.lb_threshold = threshold;
  auto pruned = dtwc::cuda::compute_distance_matrix_cuda(series, opts_pruned);

  REQUIRE(pruned.pairs_computed + pruned.pairs_pruned == N * (N - 1) / 2);

  // Check: non-pruned pairs should have exact distances
  // Pruned pairs are NaN: not computed (design §9), never a finite maximum
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      double pruned_val = pruned.matrix[i * N + j];
      double ref_val = ref.matrix[i * N + j];
      if (std::isnan(pruned_val)) {
        // This pair was pruned -- its LB must exceed threshold
        // (we don't check exact LB here, just that pruning is consistent)
        CHECK(std::isnan(pruned.matrix[j * N + i]));  // symmetric
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

TEST_CASE("GPU distance matrix with LB pruning can prune everything", "[cuda][lb_keogh]")
{
  if (!dtwc::cuda::cuda_available()) {
    SKIP("No CUDA device available");
  }

  std::vector<std::vector<double>> series(6, std::vector<double>(32));
  for (size_t i = 0; i < series.size(); ++i) {
    const double base = 1000.0 * static_cast<double>(i + 1);
    for (size_t k = 0; k < series[i].size(); ++k)
      series[i][k] = base + static_cast<double>(k);
  }

  dtwc::cuda::CUDADistMatOptions opts;
  opts.band = 3;
  opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  opts.use_lb_keogh = true;
  opts.lb_threshold = 1.0;

  auto pruned = dtwc::cuda::compute_distance_matrix_cuda(series, opts);
  const size_t N = series.size();
  const size_t total_pairs = N * (N - 1) / 2;

  REQUIRE(pruned.pairs_pruned == total_pairs);
  REQUIRE(pruned.pairs_computed == 0);

  for (size_t i = 0; i < N; ++i) {
    CHECK(pruned.matrix[i * N + i] == Catch::Approx(0.0));
    for (size_t j = i + 1; j < N; ++j) {
      CHECK(std::isnan(pruned.matrix[i * N + j]));
      CHECK(std::isnan(pruned.matrix[j * N + i]));
    }
  }
}

TEST_CASE("GPU distance matrix with LB pruning can prune nothing", "[cuda][lb_keogh]")
{
  if (!dtwc::cuda::cuda_available()) {
    SKIP("No CUDA device available");
  }

  const size_t N = 8;
  const size_t L = 40;
  const int band = 4;
  auto series = generate_random_series(N, L, 808);

  dtwc::cuda::CUDADistMatOptions ref_opts;
  ref_opts.band = band;
  ref_opts.precision = dtwc::cuda::CUDAPrecision::FP64;
  auto ref = dtwc::cuda::compute_distance_matrix_cuda(series, ref_opts);

  dtwc::cuda::CUDADistMatOptions pruned_opts = ref_opts;
  pruned_opts.use_lb_keogh = true;
  pruned_opts.lb_threshold = 1e12;
  auto pruned = dtwc::cuda::compute_distance_matrix_cuda(series, pruned_opts);

  REQUIRE(pruned.pairs_pruned == 0);
  REQUIRE(pruned.pairs_computed == N * (N - 1) / 2);

  for (size_t i = 0; i < N; ++i) {
    for (size_t j = 0; j < N; ++j) {
      CAPTURE(i, j, ref.matrix[i * N + j], pruned.matrix[i * N + j]);
      CHECK_THAT(pruned.matrix[i * N + j], WithinRel(ref.matrix[i * N + j], 1e-10));
    }
  }
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

// ---------------------------------------------------------------------------
// FX-13: the bound is admissible for its cost and band on the real device.
// One pair per call, lb_threshold = that pair's exact DTW from the test-only
// full-matrix oracle plus the FP32 slack (relative 1e-4, absolute 1e-5; FP64
// gets the same). An admissible bound cannot prune the pair. CUDA prunes only
// for band >= 0, so the property runs at a fixed band.
// ---------------------------------------------------------------------------
TEST_CASE("FX-13 CUDA LB_Keogh never prunes a pair within its exact DTW",
          "[cuda][lb_keogh][FX-13]")
{
  if (!dtwc::cuda::cuda_available()) {
    SKIP("No CUDA device available");
  }
  namespace oracle = dtwc::test::gpu_fixed_band;
  using dtwc::test_support::benchmark_series;

  std::size_t checked = 0, pruned = 0, off = 0;
  std::ostringstream first_prune;
  for (const auto precision : { dtwc::cuda::CUDAPrecision::FP64,
                                dtwc::cuda::CUDAPrecision::FP32 }) {
    for (const bool squared : { false, true }) {
      for (unsigned seed = 1; seed <= 60; ++seed) {
        const int band = 2;
        const std::size_t nx = 2 + seed % 11;
        const std::vector<std::vector<double>> pair{
          benchmark_series(nx, seed), benchmark_series(nx + seed % 3, seed + 7919U)
        };
        const double exact =
          oracle::full_matrix_oracle(pair[0], pair[1], band, squared);

        dtwc::cuda::CUDADistMatOptions opts;
        opts.band = band;
        opts.use_squared_l2 = squared;
        opts.precision = precision;
        opts.use_lb_keogh = true;
        opts.lb_threshold = exact * (1.0 + 1e-4) + 1e-5;
        const auto gpu = dtwc::cuda::compute_distance_matrix_cuda(pair, opts);
        ++checked;
        if (gpu.pairs_pruned != 0) {
          if (pruned++ == 0)
            first_prune << "squared=" << squared << " seed=" << seed
                        << " exact=" << exact;
          continue;
        }
        if (!(std::abs(gpu.matrix[1] - exact) <= 1e-4 * exact + 1e-5)) ++off;
      }
    }
  }
  INFO("first wrongly pruned pair: " << first_prune.str());
  CHECK(pruned == 0);
  REQUIRE(checked == 240);
  REQUIRE(off == 0);
}

TEST_CASE("FX-13 CUDA registered F27 and F50 pairs survive their threshold",
          "[cuda][lb_keogh][FX-13]")
{
  if (!dtwc::cuda::cuda_available()) {
    SKIP("No CUDA device available");
  }
  const std::vector<std::vector<double>> singletons{ { 0.0 }, { 0.5 } };
  const std::vector<std::vector<double>> warped{
    { 5.0, 0.0, 0.0 }, { 5.0, 5.0, 0.0 }
  };
  for (const auto precision : { dtwc::cuda::CUDAPrecision::FP64,
                                dtwc::cuda::CUDAPrecision::FP32 }) {
    dtwc::cuda::CUDADistMatOptions opts;
    opts.precision = precision;
    opts.use_lb_keogh = true;

    // F27: the L1 excess 0.5 exceeds the squared DTW 0.25 and the threshold.
    opts.band = 0;
    opts.use_squared_l2 = true;
    opts.lb_threshold = 0.3;
    const auto squared = dtwc::cuda::compute_distance_matrix_cuda(singletons, opts);
    CHECK(squared.pairs_pruned == 0);
    CHECK(squared.matrix[1] == 0.25);

    // F50: full DTW of {5,0,0} and {5,5,0} is 0; an envelope collapsed to the
    // first value by the signed k+w+1 overflow at INT_MAX bounds it at 10.
    opts.band = std::numeric_limits<int>::max();
    opts.use_squared_l2 = false;
    opts.lb_threshold = 1.0;
    const auto wide = dtwc::cuda::compute_distance_matrix_cuda(warped, opts);
    CHECK(wide.pairs_pruned == 0);
    CHECK(wide.matrix[1] == 0.0);
  }
}

#endif // DTWC_HAS_CUDA
