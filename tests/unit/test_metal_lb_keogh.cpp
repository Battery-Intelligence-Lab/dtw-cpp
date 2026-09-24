/**
 * @file test_metal_lb_keogh.cpp
 * @brief Metal LB_Keogh pruning path correctness.
 *
 * @details Three scenarios exercise the opt-in pruning path:
 *   1. Pruning disabled (`use_lb_keogh=false`) — result bit-identical to
 *      the non-LB call; `pairs_pruned == 0`.
 *   2. Threshold = +∞ — all pairs active; surviving results match non-LB.
 *   3. Threshold = 0 on random series — most pairs pruned; surviving pairs
 *      match CPU banded-DTW reference; pruned pairs are NaN (not computed).
 *
 *   FX-13 adds the admissibility property (a pair whose exact DTW is within
 *   the threshold is never pruned, for L1 and squared L2, full and banded
 *   DTW), NaN for pruned pairs, and the typed error for an envelope narrower
 *   than the DTW window.
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <dtwc.hpp>

#include "gpu_fixed_band_oracle.hpp"
#include "../support/deterministic_series.hpp"

#ifdef DTWC_HAS_METAL
#include <metal/metal_dtw.hpp>
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <sstream>
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
  return dtwc::test_support::symmetric_zero_diagonal_matrix(
    series,
    [](const auto &left, const auto &right) {
      return dtwc::dtwFull_L<double>(left, right);
    });
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

TEST_CASE("Metal LB_Keogh strict threshold prunes and stamps NaN",
          "[metal][lb_keogh]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");
  const size_t N = 10;
  const size_t L = 400;
  auto series = random_series(N, L, 0x9ABC);
  auto cpu = cpu_distance_matrix(series);

  dtwc::metal::MetalDistMatOptions opts;
  opts.use_lb_keogh = true;
  opts.lb_threshold = 0.0; // prune every pair whose envelopes disagree at all
  auto gpu = dtwc::metal::compute_distance_matrix_metal(series, opts);

  INFO("kernel_used=" << gpu.kernel_used
       << " pairs_pruned=" << gpu.pairs_pruned
       << " pairs_computed=" << gpu.pairs_computed);
  const size_t total_pairs = N * (N - 1) / 2;
  REQUIRE(gpu.pairs_pruned + gpu.pairs_computed == total_pairs);
  REQUIRE(gpu.pairs_pruned > 0); // random data very unlikely to survive lb=0

  // Every off-diagonal is either NaN (pruned: not computed, design §9) or
  // within tolerance of CPU DTW (survivor).
  size_t nan_count = 0;
  for (size_t i = 0; i < N; ++i) {
    REQUIRE(gpu.matrix[i * N + i] == 0.0);
    for (size_t j = i + 1; j < N; ++j) {
      const double g = gpu.matrix[i * N + j];
      const double gt = gpu.matrix[j * N + i];
      CAPTURE(i, j, cpu[i * N + j], g);

      if (std::isnan(g)) {
        REQUIRE(std::isnan(gt)); // symmetry preserved
        ++nan_count;
      } else {
        REQUIRE(g == gt); // symmetry preserved
        REQUIRE_THAT(g,
                     WithinRel(cpu[i * N + j], 1e-3) ||
                         WithinAbs(cpu[i * N + j], 1e-2));
      }
    }
  }
  REQUIRE(nan_count == gpu.pairs_pruned);
}

TEST_CASE("Metal compute_lb_keogh_metal matches CPU reference", "[metal][lb_keogh]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");
  const size_t N = 8;
  const size_t L = 96;
  const int band = 8;
  auto series = random_series(N, L, 0xBEEF);

  auto gpu = dtwc::metal::compute_lb_keogh_metal(series, band);
  REQUIRE(gpu.n == N);
  REQUIRE(gpu.lb_values.size() == N * (N - 1) / 2);

  // CPU reference using core::lb_keogh_symmetric.
  std::vector<dtwc::core::Envelope> envs(N);
  for (size_t i = 0; i < N; ++i)
    envs[i] = dtwc::core::compute_envelope(series[i], band);

  size_t k = 0;
  for (size_t i = 0; i < N; ++i) {
    for (size_t j = i + 1; j < N; ++j) {
      const double cpu = dtwc::core::lb_keogh_symmetric(
          series[i], envs[i], series[j], envs[j]);
      const double g = gpu.lb_values[k++];
      CAPTURE(i, j, cpu, g);
      // FP32 math on GPU vs FP64 CPU — allow relative slack.
      REQUIRE_THAT(g, WithinRel(cpu, 1e-3) || WithinAbs(cpu, 1e-3));
    }
  }
}

TEST_CASE("Metal compute_lb_keogh_metal edge cases", "[metal][lb_keogh]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");

  SECTION("band < 0 returns empty") {
    auto series = random_series(4, 32, 1);
    auto r = dtwc::metal::compute_lb_keogh_metal(series, -1);
    CHECK(r.lb_values.empty());
  }

  SECTION("N <= 1 returns empty") {
    std::vector<std::vector<double>> single{{1.0, 2.0, 3.0}};
    auto r = dtwc::metal::compute_lb_keogh_metal(single, 2);
    CHECK(r.lb_values.empty());
    CHECK(r.n == 1);
  }

  SECTION("Identical series yield LB = 0") {
    std::vector<double> s = {1.0, 3.0, -2.0, 5.0, 0.0};
    std::vector<std::vector<double>> series = {s, s, s};
    auto r = dtwc::metal::compute_lb_keogh_metal(series, 2);
    REQUIRE(r.lb_values.size() == 3);
    for (size_t i = 0; i < 3; ++i) CHECK(r.lb_values[i] == 0.0);
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

// ---------------------------------------------------------------------------
// FX-13: the bound is admissible for its cost and band on the real device.
//
// One pair per call; lb_threshold is that pair's exact DTW from the test-only
// full-matrix oracle (never the code under test) plus the registered FP32
// slack (relative 1e-4, absolute 1e-5). An admissible bound, LB <= DTW <=
// threshold, cannot prune the pair; an inadmissible one does. Wavefront is
// forced because only that route runs the LB stage (F30), and lb_time_sec
// proves the stage ran. Values in [-1, 1) make excesses below 1 common, where
// an unsquared excess exceeds the squared cost (F27); band -1 is full DTW,
// which a narrow envelope under-covers (F28). `witnesses` counts pairs the
// pre-FX-13 bound (L1 units, radius max(1, max_L/10) for full DTW) would have
// pruned, so a green run is not vacuous.
// ---------------------------------------------------------------------------
TEST_CASE("FX-13 Metal LB_Keogh never prunes a pair within its exact DTW",
          "[metal][lb_keogh][FX-13]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");
  namespace oracle = dtwc::test::gpu_fixed_band;
  using dtwc::test_support::benchmark_series;

  std::size_t checked = 0, pruned = 0, off = 0, witnesses = 0;
  std::ostringstream first_prune;
  for (const bool squared : { false, true }) {
    for (const int band : { -1, 2 }) {
      for (unsigned seed = 1; seed <= 60; ++seed) {
        const std::size_t nx = 2 + seed % 11;
        const std::size_t ny = (band < 0) ? 2 + (seed / 11) % 11 : nx + seed % 3;
        const std::vector<std::vector<double>> pair{
          benchmark_series(nx, seed), benchmark_series(ny, seed + 7919U)
        };
        const double exact =
          oracle::full_matrix_oracle(pair[0], pair[1], band, squared);
        const double threshold = exact * (1.0 + 1e-4) + 1e-5;

        const int old_radius = (band < 0)
          ? std::max(1, static_cast<int>(std::max(nx, ny)) / 10) : band;
        const auto old_env_x = dtwc::core::compute_envelope(pair[0], old_radius);
        const auto old_env_y = dtwc::core::compute_envelope(pair[1], old_radius);
        if (dtwc::core::lb_keogh_symmetric(pair[0], old_env_x, pair[1], old_env_y)
            > threshold)
          ++witnesses;

        dtwc::metal::MetalDistMatOptions opts;
        opts.band = band;
        opts.use_squared_l2 = squared;
        opts.use_lb_keogh = true;
        opts.lb_threshold = threshold;
        opts.kernel_override = dtwc::KernelOverride::Wavefront;
        const auto gpu = dtwc::metal::compute_distance_matrix_metal(pair, opts);
        REQUIRE(gpu.kernel_used == "wavefront");
        REQUIRE(gpu.lb_time_sec > 0.0);
        ++checked;
        if (gpu.pairs_pruned != 0) {
          if (pruned++ == 0)
            first_prune << "squared=" << squared << " band=" << band
                        << " seed=" << seed << " exact=" << exact;
          continue;
        }
        if (!(std::abs(gpu.matrix[1] - exact) <= 1e-4 * exact + 1e-5)) ++off;
      }
    }
  }
  INFO("first wrongly pruned pair: " << first_prune.str());
  CHECK(pruned == 0);
  REQUIRE(checked == 240);
  REQUIRE(witnesses > 0);
  REQUIRE(off == 0);
}

TEST_CASE("FX-13 Metal pruned pairs are NaN; the registered F27/F28/F50 pairs survive",
          "[metal][lb_keogh][FX-13]")
{
  if (!dtwc::metal::metal_available()) SKIP("Metal unavailable");
  dtwc::metal::MetalDistMatOptions opts; // band -1: full DTW
  opts.use_lb_keogh = true;
  opts.kernel_override = dtwc::KernelOverride::Wavefront;

  SECTION("a pruned pair is NaN, never a finite maximum") {
    // Global envelopes: LB(0,1) = 0.5 <= 1 = threshold; LB(0,2) = 30 and
    // LB(1,2) = 29.5 exceed it. DTW(0,1) = 0.5 exactly.
    const std::vector<std::vector<double>> series{
      { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 0.5 }, { 10.0, 10.0, 10.0 }
    };
    opts.lb_threshold = 1.0;
    const auto gpu = dtwc::metal::compute_distance_matrix_metal(series, opts);
    REQUIRE(gpu.pairs_pruned == 2);
    REQUIRE(gpu.pairs_computed == 1);
    for (std::size_t i = 0; i < 3; ++i) REQUIRE(gpu.matrix[i * 3 + i] == 0.0);
    REQUIRE(gpu.matrix[0 * 3 + 1] == 0.5);
    REQUIRE(gpu.matrix[1 * 3 + 0] == 0.5);
    for (const std::size_t k : { 0 * 3 + 2, 2 * 3 + 0, 1 * 3 + 2, 2 * 3 + 1 }) {
      CAPTURE(k, gpu.matrix[k]);
      REQUIRE(std::isnan(gpu.matrix[k]));
    }
  }

  SECTION("F27: squared L2 squares each excess") {
    // The L1 excess 0.5 exceeds the squared DTW 0.25 and the threshold 0.3.
    const std::vector<std::vector<double>> series{ { 0.0 }, { 0.5 } };
    opts.band = 0;
    opts.use_squared_l2 = true;
    opts.lb_threshold = 0.3;
    const auto gpu = dtwc::metal::compute_distance_matrix_metal(series, opts);
    REQUIRE(gpu.pairs_pruned == 0);
    REQUIRE(gpu.matrix[1] == 0.25);
  }

  const std::vector<std::vector<double>> warped{
    { 0, 0, 0, 0, 1, 1, 1, 1, 1, 1 }, { 0, 0, 0, 0, 0, 0, 1, 1, 1, 1 }
  };

  SECTION("F28: full DTW uses the whole-series envelope") {
    // True DTW is 0; the old radius-1 default bound was 1 > 0.5.
    opts.lb_threshold = 0.5;
    const auto gpu = dtwc::metal::compute_distance_matrix_metal(warped, opts);
    REQUIRE(gpu.pairs_pruned == 0);
    REQUIRE(gpu.matrix[1] == 0.0);
  }

  SECTION("F50: an INT_MAX radius builds the global envelope, not an overflowed one") {
    // Full DTW of {5,0,0} and {5,5,0} is 0; an envelope collapsed to the first
    // value (the signed k+w+1 overflow) bounds it at 10. The host normalizes an
    // INT_MAX band to full DTW, so only an explicit INT_MAX envelope radius and
    // the standalone entry point hand INT_MAX to the kernel's clamp.
    const std::vector<std::vector<double>> series{ { 5, 0, 0 }, { 5, 5, 0 } };
    constexpr int int_max = std::numeric_limits<int>::max();
    opts.band = int_max;
    opts.lb_threshold = 1.0;
    for (const int radius : { -1, int_max }) {
      CAPTURE(radius);
      opts.lb_envelope_band = radius;
      const auto gpu = dtwc::metal::compute_distance_matrix_metal(series, opts);
      REQUIRE(gpu.pairs_pruned == 0);
      REQUIRE(gpu.matrix[1] == 0.0);
    }
    const auto lb = dtwc::metal::compute_lb_keogh_metal(series, int_max);
    REQUIRE(lb.lb_values.size() == 1);
    REQUIRE(lb.lb_values[0] == 0.0);
  }

  SECTION("an envelope narrower than the DTW window is a typed error") {
    opts.lb_threshold = 0.5;
    opts.lb_envelope_band = 8; // full DTW of length-10 series needs >= 9
    REQUIRE_THROWS_AS(dtwc::metal::compute_distance_matrix_metal(warped, opts),
                      dtwc::InvalidInput);
    opts.band = 3;
    opts.lb_envelope_band = 2;
    REQUIRE_THROWS_AS(dtwc::metal::compute_distance_matrix_metal(warped, opts),
                      dtwc::InvalidInput);
    opts.lb_envelope_band = 3; // covers the window: accepted
    REQUIRE_NOTHROW(dtwc::metal::compute_distance_matrix_metal(warped, opts));
  }
}

#endif // DTWC_HAS_METAL
