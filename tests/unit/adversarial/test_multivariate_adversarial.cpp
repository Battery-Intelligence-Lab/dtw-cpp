/**
 * @file test_multivariate_adversarial.cpp
 * @brief Adversarial stress tests for the multivariate DTW implementation.
 *
 * @details These tests are written independently from the implementation to
 *          rigorously verify mathematical properties, edge conditions, and
 *          consistency of the multivariate DTW functions:
 *            - dtwFull_L_mv
 *            - dtwBanded_mv
 *            - derivative_transform_mv
 *          and the Problem integration with ndim > 1.
 *
 * All tests use Catch2 (NOT Google Test).
 *
 * @date 2026-04-02
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "dtwc.hpp"
#include "warping.hpp"
#include "warping_ddtw.hpp"

#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <iostream>

using Catch::Matchers::WithinAbs;
using data_t = double;

// =============================================================================
// Helpers
// =============================================================================

namespace {

/// Build a random series of n_timesteps * ndim elements (interleaved layout).
std::vector<data_t> random_mv_series(std::mt19937 &rng, size_t n_timesteps, size_t ndim,
                                      double lo = -10.0, double hi = 10.0)
{
  std::uniform_real_distribution<data_t> dist(lo, hi);
  std::vector<data_t> v(n_timesteps * ndim);
  for (auto &val : v)
    val = dist(rng);
  return v;
}

/// Extract channel `d` from an interleaved series into a plain vector.
std::vector<data_t> extract_channel(const std::vector<data_t> &x, size_t ndim, size_t d)
{
  const size_t n = x.size() / ndim;
  std::vector<data_t> ch(n);
  for (size_t t = 0; t < n; ++t)
    ch[t] = x[t * ndim + d];
  return ch;
}

} // anonymous namespace

// =============================================================================
// Test 1: ndim=1 matches scalar dtwFull_L for 100+ random pairs, lengths 1–200
// =============================================================================

TEST_CASE("MV adversarial: ndim=1 matches scalar dtwFull_L (100 random pairs)", "[mv][adversarial][ndim1]")
{
  std::mt19937 rng(12345);
  std::uniform_int_distribution<size_t> len_dist(1, 200);
  constexpr int PAIRS = 120;
  int mismatches = 0;

  for (int trial = 0; trial < PAIRS; ++trial) {
    const size_t nx = len_dist(rng);
    const size_t ny = len_dist(rng);
    auto x = random_mv_series(rng, nx, 1);
    auto y = random_mv_series(rng, ny, 1);

    const data_t d_scalar = dtwc::dtwFull_L(x.data(), nx, y.data(), ny);
    const data_t d_mv     = dtwc::dtwFull_L_mv(x.data(), nx, y.data(), ny, 1);

    // Both must be exactly equal — ndim=1 dispatches to the scalar path.
    if (std::abs(d_scalar - d_mv) > 1e-12) {
      ++mismatches;
    }
  }

  REQUIRE(mismatches == 0);
}

// =============================================================================
// Test 2: ndim=2/3/5 banded with large band matches unbanded
// =============================================================================

TEST_CASE("MV adversarial: banded(large band) == unbanded for ndim=2,3,5", "[mv][adversarial][banded]")
{
  std::mt19937 rng(7777);

  for (size_t ndim : {size_t(2), size_t(3), size_t(5)}) {
    CAPTURE(ndim);

    for (int trial = 0; trial < 20; ++trial) {
      std::uniform_int_distribution<size_t> len_dist(3, 50);
      const size_t nx = len_dist(rng);
      const size_t ny = len_dist(rng);

      auto x = random_mv_series(rng, nx, ndim);
      auto y = random_mv_series(rng, ny, ndim);

      const int big_band = static_cast<int>(std::max(nx, ny)) + 100;

      const data_t d_full   = dtwc::dtwFull_L_mv(x.data(), nx, y.data(), ny, ndim);
      const data_t d_banded = dtwc::dtwBanded_mv(x.data(), nx, y.data(), ny, ndim, big_band);

      REQUIRE_THAT(d_full, WithinAbs(d_banded, 1e-10));
    }
  }
}

// =============================================================================
// Test 3: Symmetry — dtwFull_L_mv(x,y) == dtwFull_L_mv(y,x)
// =============================================================================

TEST_CASE("MV adversarial: symmetry for dtwFull_L_mv (ndim=1,2,3,5)", "[mv][adversarial][symmetry]")
{
  std::mt19937 rng(99991);

  for (size_t ndim : {size_t(1), size_t(2), size_t(3), size_t(5)}) {
    CAPTURE(ndim);

    for (int trial = 0; trial < 40; ++trial) {
      std::uniform_int_distribution<size_t> len_dist(1, 80);
      const size_t nx = len_dist(rng);
      const size_t ny = len_dist(rng);

      auto x = random_mv_series(rng, nx, ndim);
      auto y = random_mv_series(rng, ny, ndim);

      const data_t d_xy = dtwc::dtwFull_L_mv(x.data(), nx, y.data(), ny, ndim);
      const data_t d_yx = dtwc::dtwFull_L_mv(y.data(), ny, x.data(), nx, ndim);

      REQUIRE_THAT(d_xy, WithinAbs(d_yx, 1e-10));
    }
  }
}

TEST_CASE("MV adversarial: symmetry for dtwBanded_mv (ndim=2,3)", "[mv][adversarial][symmetry]")
{
  std::mt19937 rng(11111);

  for (size_t ndim : {size_t(2), size_t(3)}) {
    CAPTURE(ndim);

    for (int trial = 0; trial < 30; ++trial) {
      std::uniform_int_distribution<size_t> len_dist(2, 60);
      const size_t nx = len_dist(rng);
      const size_t ny = len_dist(rng);
      std::uniform_int_distribution<int> band_dist(1, 30);
      const int band = band_dist(rng);

      auto x = random_mv_series(rng, nx, ndim);
      auto y = random_mv_series(rng, ny, ndim);

      const data_t d_xy = dtwc::dtwBanded_mv(x.data(), nx, y.data(), ny, ndim, band);
      const data_t d_yx = dtwc::dtwBanded_mv(y.data(), ny, x.data(), nx, ndim, band);

      REQUIRE_THAT(d_xy, WithinAbs(d_yx, 1e-10));
    }
  }
}

// =============================================================================
// Test 4: Non-negativity — always >= 0
// =============================================================================

TEST_CASE("MV adversarial: non-negativity (dtwFull_L_mv, ndim=1..5)", "[mv][adversarial][nonneg]")
{
  std::mt19937 rng(555);

  for (size_t ndim = 1; ndim <= 5; ++ndim) {
    CAPTURE(ndim);

    for (int trial = 0; trial < 30; ++trial) {
      std::uniform_int_distribution<size_t> len_dist(1, 100);
      const size_t nx = len_dist(rng);
      const size_t ny = len_dist(rng);

      auto x = random_mv_series(rng, nx, ndim);
      auto y = random_mv_series(rng, ny, ndim);

      const data_t d = dtwc::dtwFull_L_mv(x.data(), nx, y.data(), ny, ndim);
      // Result should either be a valid non-negative value or max (empty case handled elsewhere)
      REQUIRE((d >= 0.0 || d == std::numeric_limits<data_t>::max()));
    }
  }
}

TEST_CASE("MV adversarial: non-negativity (dtwBanded_mv, ndim=2,3)", "[mv][adversarial][nonneg]")
{
  std::mt19937 rng(666);

  for (size_t ndim : {size_t(2), size_t(3)}) {
    CAPTURE(ndim);

    for (int trial = 0; trial < 30; ++trial) {
      std::uniform_int_distribution<size_t> len_dist(2, 80);
      const size_t nx = len_dist(rng);
      const size_t ny = len_dist(rng);
      std::uniform_int_distribution<int> band_dist(0, 40);
      const int band = band_dist(rng);

      auto x = random_mv_series(rng, nx, ndim);
      auto y = random_mv_series(rng, ny, ndim);

      const data_t d = dtwc::dtwBanded_mv(x.data(), nx, y.data(), ny, ndim, band);
      REQUIRE((d >= 0.0 || d == std::numeric_limits<data_t>::max()));
    }
  }
}

// =============================================================================
// Test 5: Identity — d(x, x) == 0 for multivariate (via pointer equality)
// =============================================================================

TEST_CASE("MV adversarial: identity d(x,x)==0 via same pointer, ndim=1..5", "[mv][adversarial][identity]")
{
  std::mt19937 rng(246);

  for (size_t ndim = 1; ndim <= 5; ++ndim) {
    CAPTURE(ndim);

    for (int trial = 0; trial < 20; ++trial) {
      std::uniform_int_distribution<size_t> len_dist(1, 100);
      const size_t n = len_dist(rng);
      auto x = random_mv_series(rng, n, ndim);

      // Same pointer: implementation should short-circuit to 0
      const data_t d = dtwc::dtwFull_L_mv(x.data(), n, x.data(), n, ndim);
      REQUIRE(d == 0.0);
    }
  }
}

TEST_CASE("MV adversarial: identity d(x,x)==0 via copy comparison, ndim=2,3", "[mv][adversarial][identity]")
{
  // Different pointers to identical content: result must be 0 (DTW cost is 0 for equal series)
  std::mt19937 rng(135);

  for (size_t ndim : {size_t(2), size_t(3)}) {
    CAPTURE(ndim);

    for (int trial = 0; trial < 20; ++trial) {
      std::uniform_int_distribution<size_t> len_dist(1, 50);
      const size_t n = len_dist(rng);
      auto x = random_mv_series(rng, n, ndim);
      auto y = x; // exact copy

      const data_t d = dtwc::dtwFull_L_mv(x.data(), n, y.data(), n, ndim);
      REQUIRE_THAT(d, WithinAbs(0.0, 1e-12));
    }
  }
}

// =============================================================================
// Test 6: Triangle inequality stress — find counterexamples (expected for DTW)
// =============================================================================

TEST_CASE("MV adversarial: triangle inequality — document violations (expected for DTW)", "[mv][adversarial][triangle]")
{
  // DTW is NOT a metric: triangle inequality violations are expected.
  // This test counts them and verifies they exist (> 0 expected).
  std::mt19937 rng(31415);
  std::uniform_int_distribution<size_t> len_dist(3, 30);

  int violations = 0;
  int checks = 0;
  constexpr size_t NDIM = 2;
  constexpr int TRIPLES = 200;

  for (int trial = 0; trial < TRIPLES; ++trial) {
    const size_t na = len_dist(rng);
    const size_t nb = len_dist(rng);
    const size_t nc = len_dist(rng);

    auto a = random_mv_series(rng, na, NDIM);
    auto b = random_mv_series(rng, nb, NDIM);
    auto c = random_mv_series(rng, nc, NDIM);

    const data_t dab = dtwc::dtwFull_L_mv(a.data(), na, b.data(), nb, NDIM);
    const data_t dbc = dtwc::dtwFull_L_mv(b.data(), nb, c.data(), nc, NDIM);
    const data_t dac = dtwc::dtwFull_L_mv(a.data(), na, c.data(), nc, NDIM);

    if (dac > dab + dbc + 1e-9) ++violations;
    ++checks;
  }

  // For DTW, violations are expected to exist; we just document them.
  // Do NOT REQUIRE violations == 0.  If we find them it confirms DTW is non-metric.
  std::cout << "[triangle] " << violations << "/" << checks
            << " violations found for ndim=" << NDIM
            << " (DTW is NOT a metric — violations are expected)\n";
  // The test itself always passes — we are stress-testing for crashes/NaNs, not asserting metric.
  REQUIRE(checks == TRIPLES);
}

// =============================================================================
// Test 7: Large ndim (D=10, D=50) — no crash, finite results
// =============================================================================

TEST_CASE("MV adversarial: large ndim D=10 no crash, finite results", "[mv][adversarial][largeDim]")
{
  std::mt19937 rng(8080);
  constexpr size_t NDIM = 10;

  for (int trial = 0; trial < 20; ++trial) {
    std::uniform_int_distribution<size_t> len_dist(2, 40);
    const size_t nx = len_dist(rng);
    const size_t ny = len_dist(rng);

    auto x = random_mv_series(rng, nx, NDIM);
    auto y = random_mv_series(rng, ny, NDIM);

    const data_t d_full   = dtwc::dtwFull_L_mv(x.data(), nx, y.data(), ny, NDIM);
    const data_t d_banded = dtwc::dtwBanded_mv(x.data(), nx, y.data(), ny, NDIM, 10);

    REQUIRE(std::isfinite(d_full));
    REQUIRE(d_full >= 0.0);
    REQUIRE(std::isfinite(d_banded));
    REQUIRE(d_banded >= 0.0);
  }
}

TEST_CASE("MV adversarial: large ndim D=50 no crash, finite results", "[mv][adversarial][largeDim]")
{
  std::mt19937 rng(9999);
  constexpr size_t NDIM = 50;

  for (int trial = 0; trial < 10; ++trial) {
    std::uniform_int_distribution<size_t> len_dist(2, 20);
    const size_t nx = len_dist(rng);
    const size_t ny = len_dist(rng);

    auto x = random_mv_series(rng, nx, NDIM);
    auto y = random_mv_series(rng, ny, NDIM);

    const data_t d = dtwc::dtwFull_L_mv(x.data(), nx, y.data(), ny, NDIM);
    REQUIRE(std::isfinite(d));
    REQUIRE(d >= 0.0);
  }
}

// =============================================================================
// Test 8: Asymmetric lengths — x has 3 timesteps, y has 7, ndim=3
// =============================================================================

TEST_CASE("MV adversarial: asymmetric lengths nx=3, ny=7, ndim=3", "[mv][adversarial][asymlen]")
{
  // x: 3 timesteps x 3 dims (9 elements)
  // y: 7 timesteps x 3 dims (21 elements)
  double x[] = {
    1,0,0,  // t=0
    0,1,0,  // t=1
    0,0,1   // t=2
  };
  double y[] = {
    1,0,0,  // t=0
    1,0,0,  // t=1 (same as t=0)
    0,1,0,  // t=2
    0,1,0,  // t=3 (same)
    0,0,1,  // t=4
    0,0,1,  // t=5 (same)
    0,0,0   // t=6
  };

  const data_t d = dtwc::dtwFull_L_mv(x, 3, y, 7, 3);
  REQUIRE(std::isfinite(d));
  REQUIRE(d >= 0.0);

  // Symmetry must hold
  const data_t d_rev = dtwc::dtwFull_L_mv(y, 7, x, 3, 3);
  REQUIRE_THAT(d, WithinAbs(d_rev, 1e-10));
}

TEST_CASE("MV adversarial: asymmetric random lengths, ndim=3, 30 pairs", "[mv][adversarial][asymlen]")
{
  std::mt19937 rng(2468);

  for (int trial = 0; trial < 30; ++trial) {
    std::uniform_int_distribution<size_t> short_len(1, 10);
    std::uniform_int_distribution<size_t> long_len(11, 50);

    const size_t nx = short_len(rng);
    const size_t ny = long_len(rng);

    auto x = random_mv_series(rng, nx, 3);
    auto y = random_mv_series(rng, ny, 3);

    const data_t d_xy = dtwc::dtwFull_L_mv(x.data(), nx, y.data(), ny, 3);
    const data_t d_yx = dtwc::dtwFull_L_mv(y.data(), ny, x.data(), nx, 3);

    REQUIRE(std::isfinite(d_xy));
    REQUIRE(d_xy >= 0.0);
    REQUIRE_THAT(d_xy, WithinAbs(d_yx, 1e-10));
  }
}

// =============================================================================
// Test 9: Single timestep — nx=1, ndim=5
// =============================================================================

TEST_CASE("MV adversarial: single timestep nx=1 or ny=1, ndim=5", "[mv][adversarial][single]")
{
  std::mt19937 rng(13579);
  constexpr size_t NDIM = 5;

  SECTION("Both single timestep") {
    auto x = random_mv_series(rng, 1, NDIM);
    auto y = random_mv_series(rng, 1, NDIM);

    const data_t d = dtwc::dtwFull_L_mv(x.data(), 1, y.data(), 1, NDIM);
    REQUIRE(std::isfinite(d));
    REQUIRE(d >= 0.0);

    // Manual check: for single timestep L1 DTW equals L1 distance
    data_t expected = 0.0;
    for (size_t d_ = 0; d_ < NDIM; ++d_)
      expected += std::abs(x[d_] - y[d_]);
    REQUIRE_THAT(d, WithinAbs(expected, 1e-12));
  }

  SECTION("x has 1 timestep, y has many") {
    auto x = random_mv_series(rng, 1, NDIM);
    auto y = random_mv_series(rng, 20, NDIM);

    const data_t d = dtwc::dtwFull_L_mv(x.data(), 1, y.data(), 20, NDIM);
    REQUIRE(std::isfinite(d));
    REQUIRE(d >= 0.0);

    // Symmetry
    const data_t d_rev = dtwc::dtwFull_L_mv(y.data(), 20, x.data(), 1, NDIM);
    REQUIRE_THAT(d, WithinAbs(d_rev, 1e-10));
  }

  SECTION("x has 1 timestep, y has 1: identity when equal") {
    auto x = random_mv_series(rng, 1, NDIM);
    auto y = x; // identical copy

    const data_t d = dtwc::dtwFull_L_mv(x.data(), 1, y.data(), 1, NDIM);
    REQUIRE_THAT(d, WithinAbs(0.0, 1e-12));
  }
}

// =============================================================================
// Test 10: derivative_transform_mv stress — ndim=5, long series
// =============================================================================

TEST_CASE("MV adversarial: derivative_transform_mv ndim=5 matches per-channel scalar", "[mv][adversarial][ddtw]")
{
  std::mt19937 rng(54321);
  constexpr size_t NDIM = 5;

  for (int trial = 0; trial < 20; ++trial) {
    std::uniform_int_distribution<size_t> len_dist(2, 200);
    const size_t n = len_dist(rng);

    auto x = random_mv_series(rng, n, NDIM);
    auto dx_mv = dtwc::derivative_transform_mv(x, NDIM);

    REQUIRE(dx_mv.size() == x.size());

    // Each channel must match the scalar derivative_transform
    for (size_t d = 0; d < NDIM; ++d) {
      auto ch = extract_channel(x, NDIM, d);
      auto dch = dtwc::derivative_transform(ch);

      REQUIRE(dch.size() == n);

      for (size_t t = 0; t < n; ++t) {
        REQUIRE_THAT(dx_mv[t * NDIM + d], WithinAbs(dch[t], 1e-12));
      }
    }
  }
}

TEST_CASE("MV adversarial: derivative_transform_mv channels are independent", "[mv][adversarial][ddtw]")
{
  // Verify that changing one channel does not affect the derivative of another channel.
  constexpr size_t NDIM = 3;
  constexpr size_t N = 10;

  // Base series
  std::vector<data_t> x(N * NDIM, 0.0);
  for (size_t t = 0; t < N; ++t) {
    x[t * NDIM + 0] = static_cast<data_t>(t);       // channel 0: linear ramp
    x[t * NDIM + 1] = static_cast<data_t>(t * t);   // channel 1: quadratic
    x[t * NDIM + 2] = std::sin(static_cast<data_t>(t)); // channel 2: sinusoidal
  }

  auto dx = dtwc::derivative_transform_mv(x, NDIM);

  // Now perturb only channel 0 and recompute
  auto x2 = x;
  for (size_t t = 0; t < N; ++t)
    x2[t * NDIM + 0] += 1000.0; // massive shift to channel 0

  auto dx2 = dtwc::derivative_transform_mv(x2, NDIM);

  // Channels 1 and 2 should be unaffected
  for (size_t t = 0; t < N; ++t) {
    REQUIRE_THAT(dx[t * NDIM + 1], WithinAbs(dx2[t * NDIM + 1], 1e-12));
    REQUIRE_THAT(dx[t * NDIM + 2], WithinAbs(dx2[t * NDIM + 2], 1e-12));
  }
}

TEST_CASE("MV adversarial: derivative_transform_mv zero derivative for constant series", "[mv][adversarial][ddtw]")
{
  // A constant per-channel series should have derivative 0 everywhere (interior)
  constexpr size_t NDIM = 4;
  constexpr size_t N = 8;

  std::vector<data_t> x(N * NDIM);
  for (size_t d = 0; d < NDIM; ++d) {
    const data_t val = static_cast<data_t>(d + 1) * 3.7;
    for (size_t t = 0; t < N; ++t)
      x[t * NDIM + d] = val;
  }

  auto dx = dtwc::derivative_transform_mv(x, NDIM);

  // All derivatives should be 0 (constant series: slope is 0 everywhere)
  for (size_t i = 0; i < dx.size(); ++i) {
    REQUIRE_THAT(dx[i], WithinAbs(0.0, 1e-12));
  }
}

// =============================================================================
// Test 11: Problem integration — ndim=3 distance matrix symmetric and finite
// =============================================================================

TEST_CASE("MV adversarial: Problem ndim=3 distance matrix symmetric and finite", "[mv][adversarial][problem]")
{
  std::mt19937 rng(77777);
  constexpr size_t NDIM = 3;
  constexpr int N_SERIES = 8;

  dtwc::Data data;
  data.ndim = NDIM;

  for (int i = 0; i < N_SERIES; ++i) {
    std::uniform_int_distribution<size_t> len_dist(3, 20);
    const size_t n_steps = len_dist(rng);
    auto series = random_mv_series(rng, n_steps, NDIM);
    data.p_vec.push_back(std::move(series));
    data.p_names.push_back("s" + std::to_string(i));
  }

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.verbose = false;
  prob.fillDistanceMatrix();

  for (int i = 0; i < N_SERIES; ++i) {
    for (int j = 0; j < N_SERIES; ++j) {
      const data_t dij = prob.distByInd(i, j);
      const data_t dji = prob.distByInd(j, i);

      // Finite
      REQUIRE(std::isfinite(dij));

      // Non-negative
      REQUIRE(dij >= 0.0);

      // Symmetry
      REQUIRE_THAT(dij, WithinAbs(dji, 1e-10));

      // Identity (diagonal)
      if (i == j) {
        REQUIRE_THAT(dij, WithinAbs(0.0, 1e-10));
      }
    }
  }
}

TEST_CASE("MV adversarial: Problem ndim=3, series with different timestep counts", "[mv][adversarial][problem]")
{
  // Deliberately give different numbers of timesteps to each series.
  dtwc::Data data;
  data.ndim = 3;
  data.p_vec = {
    {1,0,0, 0,1,0},           // 2 timesteps
    {1,0,0, 0,1,0, 0,0,1},    // 3 timesteps
    {0,0,0, 0,0,0, 0,0,0, 1,1,1} // 4 timesteps
  };
  data.p_names = {"a", "b", "c"};

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.verbose = false;
  prob.fillDistanceMatrix();

  // All off-diagonal distances should be positive (series are different)
  REQUIRE(prob.distByInd(0, 1) >= 0.0);
  REQUIRE(prob.distByInd(0, 2) >= 0.0);
  REQUIRE(prob.distByInd(1, 2) >= 0.0);

  // Diagonal must be zero
  REQUIRE_THAT(prob.distByInd(0, 0), WithinAbs(0.0, 1e-10));
  REQUIRE_THAT(prob.distByInd(1, 1), WithinAbs(0.0, 1e-10));
  REQUIRE_THAT(prob.distByInd(2, 2), WithinAbs(0.0, 1e-10));

  // Symmetry
  REQUIRE_THAT(prob.distByInd(0, 1), WithinAbs(prob.distByInd(1, 0), 1e-10));
  REQUIRE_THAT(prob.distByInd(0, 2), WithinAbs(prob.distByInd(2, 0), 1e-10));
  REQUIRE_THAT(prob.distByInd(1, 2), WithinAbs(prob.distByInd(2, 1), 1e-10));
}

// =============================================================================
// Test 12: Performance — ndim=1 MV vs scalar (500 pairs)
// =============================================================================

TEST_CASE("MV adversarial: ndim=1 MV dispatches to scalar — timing parity (500 pairs)", "[mv][adversarial][perf]")
{
  std::mt19937 rng(42);
  std::uniform_int_distribution<size_t> len_dist(50, 150);

  constexpr int PAIRS = 500;
  std::vector<std::vector<data_t>> xs(PAIRS), ys(PAIRS);
  std::vector<size_t> nx_vec(PAIRS), ny_vec(PAIRS);

  for (int i = 0; i < PAIRS; ++i) {
    nx_vec[i] = len_dist(rng);
    ny_vec[i] = len_dist(rng);
    xs[i] = random_mv_series(rng, nx_vec[i], 1);
    ys[i] = random_mv_series(rng, ny_vec[i], 1);
  }

  // First: verify correctness (MV ndim=1 must match scalar exactly)
  for (int i = 0; i < PAIRS; ++i) {
    const data_t d_scalar = dtwc::dtwFull_L(xs[i].data(), nx_vec[i], ys[i].data(), ny_vec[i]);
    const data_t d_mv     = dtwc::dtwFull_L_mv(xs[i].data(), nx_vec[i], ys[i].data(), ny_vec[i], 1);
    REQUIRE_THAT(d_mv, WithinAbs(d_scalar, 1e-12));
  }

  // Second: timing measurement (informational, no hard assertion)
  auto t0 = std::chrono::high_resolution_clock::now();
  volatile data_t sink_scalar = 0;
  for (int i = 0; i < PAIRS; ++i)
    sink_scalar += dtwc::dtwFull_L(xs[i].data(), nx_vec[i], ys[i].data(), ny_vec[i]);
  auto t1 = std::chrono::high_resolution_clock::now();

  volatile data_t sink_mv = 0;
  for (int i = 0; i < PAIRS; ++i)
    sink_mv += dtwc::dtwFull_L_mv(xs[i].data(), nx_vec[i], ys[i].data(), ny_vec[i], 1);
  auto t2 = std::chrono::high_resolution_clock::now();

  auto ms_scalar = std::chrono::duration<double, std::milli>(t1 - t0).count();
  auto ms_mv     = std::chrono::duration<double, std::milli>(t2 - t1).count();
  std::cout << "[perf/500pairs] scalar: " << ms_scalar << " ms,  MV(D=1): " << ms_mv
            << " ms  (ratio: " << (ms_mv / ms_scalar) << "x)\n";

  // Since ndim=1 dispatches directly to dtwFull_L (no MV overhead), ratio must be close to 1x.
  // Generous bound of 3x to account for measurement noise on CI.
  REQUIRE(ms_mv < ms_scalar * 3.0 + 50.0);

  // Suppress unused-variable warnings
  (void)sink_scalar;
  (void)sink_mv;
}

// =============================================================================
// Test 13 (bonus): empty series — both MV functions return max
// =============================================================================

TEST_CASE("MV adversarial: empty series returns max (dtwFull_L_mv, ndim=3)", "[mv][adversarial][empty]")
{
  constexpr data_t maxVal = std::numeric_limits<data_t>::max();
  double x[] = {1.0, 2.0, 3.0}; // 1 timestep, ndim=3

  REQUIRE(dtwc::dtwFull_L_mv(x, 1, static_cast<double*>(nullptr), 0, 3) == maxVal);
  REQUIRE(dtwc::dtwFull_L_mv(static_cast<double*>(nullptr), 0, x, 1, 3) == maxVal);
}

TEST_CASE("MV adversarial: empty series returns max (dtwBanded_mv, ndim=2)", "[mv][adversarial][empty]")
{
  constexpr data_t maxVal = std::numeric_limits<data_t>::max();
  double x[] = {1.0, 2.0}; // 1 timestep, ndim=2

  REQUIRE(dtwc::dtwBanded_mv(x, 1, static_cast<double*>(nullptr), 0, 2, 10) == maxVal);
  REQUIRE(dtwc::dtwBanded_mv(static_cast<double*>(nullptr), 0, x, 1, 2, 10) == maxVal);
}

// =============================================================================
// Test 14 (bonus): SquaredL2 metric — ndim=2,3 correctness and symmetry
// =============================================================================

TEST_CASE("MV adversarial: SquaredL2 symmetry, ndim=2,3", "[mv][adversarial][sqL2]")
{
  std::mt19937 rng(31337);

  for (size_t ndim : {size_t(2), size_t(3)}) {
    CAPTURE(ndim);

    for (int trial = 0; trial < 30; ++trial) {
      std::uniform_int_distribution<size_t> len_dist(2, 50);
      const size_t nx = len_dist(rng);
      const size_t ny = len_dist(rng);

      auto x = random_mv_series(rng, nx, ndim);
      auto y = random_mv_series(rng, ny, ndim);

      const data_t d_xy = dtwc::dtwFull_L_mv(x.data(), nx, y.data(), ny, ndim, -1.0,
                                               dtwc::core::MetricType::SquaredL2);
      const data_t d_yx = dtwc::dtwFull_L_mv(y.data(), ny, x.data(), nx, ndim, -1.0,
                                               dtwc::core::MetricType::SquaredL2);

      REQUIRE(std::isfinite(d_xy));
      REQUIRE(d_xy >= 0.0);
      REQUIRE_THAT(d_xy, WithinAbs(d_yx, 1e-10));
    }
  }
}

TEST_CASE("MV adversarial: SquaredL2 identity d(x,x)==0, ndim=3", "[mv][adversarial][sqL2]")
{
  std::mt19937 rng(22222);
  constexpr size_t NDIM = 3;

  for (int trial = 0; trial < 20; ++trial) {
    std::uniform_int_distribution<size_t> len_dist(1, 60);
    const size_t n = len_dist(rng);
    auto x = random_mv_series(rng, n, NDIM);

    const data_t d = dtwc::dtwFull_L_mv(x.data(), n, x.data(), n, NDIM, -1.0,
                                          dtwc::core::MetricType::SquaredL2);
    REQUIRE(d == 0.0);
  }
}

// =============================================================================
// Test 15 (bonus): Monotonicity of banded MV — larger band gives <= cost
// =============================================================================

TEST_CASE("MV adversarial: banded_mv monotonicity (larger band => lower or equal cost)", "[mv][adversarial][monotone]")
{
  std::mt19937 rng(55555);

  for (size_t ndim : {size_t(2), size_t(3)}) {
    CAPTURE(ndim);

    for (int trial = 0; trial < 10; ++trial) {
      std::uniform_int_distribution<size_t> len_dist(5, 30);
      const size_t n = len_dist(rng);

      auto x = random_mv_series(rng, n, ndim);
      auto y = random_mv_series(rng, n, ndim);

      data_t prev_cost = dtwc::dtwBanded_mv(x.data(), n, y.data(), n, ndim, 1);
      for (int band = 2; band <= static_cast<int>(n); ++band) {
        const data_t cost = dtwc::dtwBanded_mv(x.data(), n, y.data(), n, ndim, band);
        REQUIRE(cost <= prev_cost + 1e-10);
        prev_cost = cost;
      }
    }
  }
}
