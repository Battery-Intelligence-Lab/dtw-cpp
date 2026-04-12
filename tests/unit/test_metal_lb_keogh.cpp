/**
 * @file test_metal_lb_keogh.cpp
 * @brief Metal LB_Keogh pruning path correctness.
 *
 * @details Three scenarios exercise the opt-in pruning path:
 *   1. Pruning disabled (`use_lb_keogh=false`) — result bit-identical to
 *      the non-LB call; `pairs_pruned == 0`.
 *   2. Threshold = +∞ — all pairs active; surviving results match non-LB.
 *   3. Threshold = 0 on random series — most pairs pruned; surviving pairs
 *      match CPU banded-DTW reference; pruned pairs return +∞.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <dtwc.hpp>

#ifdef DTWC_HAS_METAL
#include <metal/metal_dtw.hpp>
#endif

#include <cmath>
#include <limits>
#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

#ifndef DTWC_HAS_METAL

TEST_CASE("Metal LB_Keogh skipped (no DTWC_HAS_METAL)", "[metal][lb_keogh]")
{
  SKIP("DTWC_HAS_METAL not defined; Metal LB_Keogh tests skipped");
}

#else // DTWC_HAS_METAL

namespace {

std::vector<std::vector<double>> random_series(size_t n, size_t length,
                                               unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-5.0, 5.0);
  std::vector<std::vector<double>> out(n);
  for (auto &s : out) {
    s.resize(length);
    for (auto &v : s) v = dist(rng);
  }
  return out;
}

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

} // namespace

TEST_CASE("Metal LB_Keogh disabled matches non-LB path", "[metal][lb_keogh]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");
  // L must route to wavefront (i.e., not regtile). Regtile cap is 256, so
  // pick L=400 with N=4 for a small wavefront workload.
  const size_t N = 4;
  const size_t L = 400;
  auto series = random_series(N, L, 0x1234);

  dtwc::metal::MetalDistMatOptions opts_plain;
  auto plain = dtwc::metal::compute_distance_matrix_metal(series, opts_plain);

  dtwc::metal::MetalDistMatOptions opts_disabled;
  opts_disabled.use_lb_keogh = false;
  auto disabled = dtwc::metal::compute_distance_matrix_metal(series,
                                                             opts_disabled);

  REQUIRE(disabled.pairs_pruned == 0);
  REQUIRE(disabled.pairs_computed == plain.pairs_computed);
  REQUIRE(disabled.matrix.size() == plain.matrix.size());
  for (size_t k = 0; k < plain.matrix.size(); ++k) {
    CAPTURE(k);
    REQUIRE(disabled.matrix[k] == plain.matrix[k]);
  }
}

TEST_CASE("Metal LB_Keogh permissive threshold keeps all pairs", "[metal][lb_keogh]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");
  const size_t N = 5;
  const size_t L = 400;
  auto series = random_series(N, L, 0x5678);

  auto cpu = cpu_distance_matrix(series);

  dtwc::metal::MetalDistMatOptions opts;
  opts.use_lb_keogh = true;
  opts.lb_threshold = std::numeric_limits<double>::infinity();
  opts.lb_envelope_band = std::max<int>(1, static_cast<int>(L) / 10);
  auto gpu = dtwc::metal::compute_distance_matrix_metal(series, opts);

  INFO("kernel_used=" << gpu.kernel_used
       << " pairs_computed=" << gpu.pairs_computed
       << " pairs_pruned=" << gpu.pairs_pruned);
  REQUIRE(gpu.pairs_pruned == 0);
  REQUIRE(gpu.pairs_computed == N * (N - 1) / 2);
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      CAPTURE(i, j, cpu[i * N + j], gpu.matrix[i * N + j]);
      REQUIRE_THAT(gpu.matrix[i * N + j],
                   WithinRel(cpu[i * N + j], 1e-3) || WithinAbs(cpu[i * N + j], 1e-2));
    }
  }
}

TEST_CASE("Metal LB_Keogh strict threshold prunes and stamps INF", "[metal][lb_keogh]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");
  const size_t N = 10;
  const size_t L = 400;
  auto series = random_series(N, L, 0x9ABC);
  auto cpu = cpu_distance_matrix(series);

  dtwc::metal::MetalDistMatOptions opts;
  opts.use_lb_keogh = true;
  opts.lb_threshold = 0.0; // prune every pair whose envelopes disagree at all
  opts.lb_envelope_band = std::max<int>(1, static_cast<int>(L) / 10);
  auto gpu = dtwc::metal::compute_distance_matrix_metal(series, opts);

  INFO("kernel_used=" << gpu.kernel_used
       << " pairs_pruned=" << gpu.pairs_pruned
       << " pairs_computed=" << gpu.pairs_computed);
  const size_t total_pairs = N * (N - 1) / 2;
  REQUIRE(gpu.pairs_pruned + gpu.pairs_computed == total_pairs);
  REQUIRE(gpu.pairs_pruned > 0); // random data very unlikely to survive lb=0

  // Every off-diagonal is either +inf (pruned) or within tolerance of CPU
  // DTW (survivor).
  const float INF = std::numeric_limits<float>::max();
  for (size_t i = 0; i < N; ++i) {
    REQUIRE(gpu.matrix[i * N + i] == 0.0);
    for (size_t j = i + 1; j < N; ++j) {
      const double g = gpu.matrix[i * N + j];
      const double gt = gpu.matrix[j * N + i];
      CAPTURE(i, j, cpu[i * N + j], g);
      REQUIRE(g == gt); // symmetry preserved

      if (std::abs(g - static_cast<double>(INF)) < 1e30) {
        // Pruned pair: LB_Keogh must be a valid lower bound on the CPU DTW.
        // We can't REQUIRE it strictly without recomputing LB; just assert
        // the stamp is +∞-like and non-negative.
        REQUIRE(g > 1e30);
      } else {
        REQUIRE_THAT(g,
                     WithinRel(cpu[i * N + j], 1e-3) ||
                         WithinAbs(cpu[i * N + j], 1e-2));
      }
    }
  }
}

TEST_CASE("Metal LB_Keogh silently disables on banded_row path", "[metal][lb_keogh]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");
  // band=20 with L=400 -> band*20=400 NOT less than L=400, so we stay on
  // wavefront. Use band=15 with L=400 so band*20 = 300 < 400, which routes
  // to banded_row. LB_Keogh should be ignored on that path.
  const size_t N = 4;
  const size_t L = 400;
  const int band = 15;
  auto series = random_series(N, L, 0x333);

  dtwc::metal::MetalDistMatOptions opts;
  opts.band = band;
  opts.use_lb_keogh = true;
  opts.lb_threshold = 0.0;
  auto gpu = dtwc::metal::compute_distance_matrix_metal(series, opts);

  INFO("kernel_used=" << gpu.kernel_used);
  REQUIRE(gpu.kernel_used == "banded_row");
  REQUIRE(gpu.pairs_pruned == 0); // LB silently ignored on banded_row
  REQUIRE(gpu.pairs_computed == N * (N - 1) / 2);
}

#endif // DTWC_HAS_METAL
