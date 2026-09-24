/**
 * @file unit_test_lower_bounds.cpp
 * @brief Unit tests for LB_Keogh and LB_Kim lower bound functions.
 *
 * @details Verifies correctness of lower-bound computations:
 *   - Envelope construction for known series
 *   - LB_Keogh and LB_Kim are always <= actual DTW distance
 *   - Identical series yield LB == 0
 *   - Edge cases (single element, empty series)
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include <core/lower_bound_impl.hpp>
#include <warping.hpp>

#include "../gpu_fixed_band_oracle.hpp"
#include "../../support/deterministic_series.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <span>
#include <sstream>
#include <vector>
#include <random>

using Catch::Matchers::WithinAbs;
using data_t = double;


// ---------------------------------------------------------------------------
// Test 1: Envelope correctness for a known series with band=1
// ---------------------------------------------------------------------------
TEST_CASE("compute_envelopes known series band=1", "[lower_bounds][envelopes]")
{
  // Series:  {1, 3, 2, 4, 1}
  // band = 1 means window [i-1, i+1]
  //
  // i=0: window [0,1] -> values {1,3} -> upper=3, lower=1
  // i=1: window [0,2] -> values {1,3,2} -> upper=3, lower=1
  // i=2: window [1,3] -> values {3,2,4} -> upper=4, lower=2
  // i=3: window [2,4] -> values {2,4,1} -> upper=4, lower=1
  // i=4: window [3,4] -> values {4,1} -> upper=4, lower=1

  const std::vector<data_t> series = { 1.0, 3.0, 2.0, 4.0, 1.0 };
  std::vector<data_t> upper(5), lower(5);

  dtwc::core::compute_envelopes(series, 1, upper, lower);

  // Upper envelope
  REQUIRE_THAT(upper[0], WithinAbs(3.0, 1e-15));
  REQUIRE_THAT(upper[1], WithinAbs(3.0, 1e-15));
  REQUIRE_THAT(upper[2], WithinAbs(4.0, 1e-15));
  REQUIRE_THAT(upper[3], WithinAbs(4.0, 1e-15));
  REQUIRE_THAT(upper[4], WithinAbs(4.0, 1e-15));

  // Lower envelope
  REQUIRE_THAT(lower[0], WithinAbs(1.0, 1e-15));
  REQUIRE_THAT(lower[1], WithinAbs(1.0, 1e-15));
  REQUIRE_THAT(lower[2], WithinAbs(2.0, 1e-15));
  REQUIRE_THAT(lower[3], WithinAbs(1.0, 1e-15));
  REQUIRE_THAT(lower[4], WithinAbs(1.0, 1e-15));
}

// ---------------------------------------------------------------------------
// Test 2: Envelope with band=0 (each position is its own window)
// ---------------------------------------------------------------------------
TEST_CASE("compute_envelopes band=0 identity", "[lower_bounds][envelopes]")
{
  const std::vector<data_t> series = { 5.0, 2.0, 8.0, 1.0 };
  std::vector<data_t> upper(4), lower(4);

  dtwc::core::compute_envelopes(series, 0, upper, lower);

  for (std::size_t i = 0; i < series.size(); ++i) {
    REQUIRE_THAT(upper[i], WithinAbs(series[i], 1e-15));
    REQUIRE_THAT(lower[i], WithinAbs(series[i], 1e-15));
  }
}

// ---------------------------------------------------------------------------
// Test 3: LB_Keogh == 0 for identical series
// ---------------------------------------------------------------------------
TEST_CASE("lb_keogh identical series gives zero", "[lower_bounds][lb_keogh]")
{
  const std::vector<data_t> series = { 1.0, 3.0, 2.0, 4.0, 1.0 };
  std::vector<data_t> upper, lower;

  dtwc::core::compute_envelopes(series, 1, upper, lower);
  const data_t lb = dtwc::core::lb_keogh(series, upper, lower);

  REQUIRE_THAT(lb, WithinAbs(0.0, 1e-15));
}

// ---------------------------------------------------------------------------
// Test 4: LB_Kim == 0 for identical series
// ---------------------------------------------------------------------------
TEST_CASE("lb_kim identical series gives zero", "[lower_bounds][lb_kim]")
{
  const std::vector<data_t> series = { 1.0, 3.0, 2.0, 4.0, 1.0 };
  const data_t lb = dtwc::core::lb_kim(series, series);

  REQUIRE_THAT(lb, WithinAbs(0.0, 1e-15));
}

// ---------------------------------------------------------------------------
// Test 5: LB_Keogh <= DTW for random series (property test, 10 pairs)
// ---------------------------------------------------------------------------
TEST_CASE("lb_keogh <= DTW for random series", "[lower_bounds][lb_keogh][property]")
{
  std::mt19937 rng(42); // NOLINT(cert-msc51-cpp): fixed seed keeps this property test reproducible.
  std::uniform_real_distribution<data_t> dist(-10.0, 10.0);

  constexpr int N_pairs = 10;
  constexpr int series_len = 20;
  constexpr int band = 3;

  for (int p = 0; p < N_pairs; ++p) {
    std::vector<data_t> x(series_len), y(series_len);
    for (int i = 0; i < series_len; ++i) {
      x[i] = dist(rng);
      y[i] = dist(rng);
    }

    // Compute envelopes of y, then LB_Keogh(x, envelope_of_y)
    std::vector<data_t> upper, lower;
    dtwc::core::compute_envelopes(y, band, upper, lower);
    const data_t lb = dtwc::core::lb_keogh(x, upper, lower);

    // Compute banded DTW for comparison
    const data_t dtw_dist = dtwc::dtwBanded<data_t>(x, y, band);

    INFO("Pair " << p << ": LB_Keogh=" << lb << " DTW=" << dtw_dist);
    REQUIRE(lb <= dtw_dist + 1e-10); // allow tiny floating-point tolerance
  }
}

// ---------------------------------------------------------------------------
// Test 6: LB_Kim <= DTW for random series (property test, 10 pairs)
// ---------------------------------------------------------------------------
TEST_CASE("lb_kim <= DTW for random series", "[lower_bounds][lb_kim][property]")
{
  std::mt19937 rng(123); // NOLINT(cert-msc51-cpp): fixed seed keeps this property test reproducible.
  std::uniform_real_distribution<data_t> dist(-10.0, 10.0);

  constexpr int N_pairs = 10;
  constexpr int series_len = 20;

  for (int p = 0; p < N_pairs; ++p) {
    std::vector<data_t> x(series_len), y(series_len);
    for (int i = 0; i < series_len; ++i) {
      x[i] = dist(rng);
      y[i] = dist(rng);
    }

    const data_t lb = dtwc::core::lb_kim(x, y);
    const data_t dtw_dist = dtwc::dtwFull<data_t>(x, y);

    INFO("Pair " << p << ": LB_Kim=" << lb << " DTW=" << dtw_dist);
    REQUIRE(lb <= dtw_dist + 1e-10);
  }
}

// ---------------------------------------------------------------------------
// Test 7: LB_Keogh with query inside envelope gives 0
// ---------------------------------------------------------------------------
TEST_CASE("lb_keogh query within envelope gives zero", "[lower_bounds][lb_keogh]")
{
  // Candidate series
  const std::vector<data_t> candidate = { 1.0, 5.0, 3.0, 7.0, 2.0 };
  std::vector<data_t> upper, lower;
  dtwc::core::compute_envelopes(candidate, 2, upper, lower);

  // Construct a query that is strictly within the envelope at every point
  std::vector<data_t> query(candidate.size());
  for (std::size_t i = 0; i < candidate.size(); ++i) {
    query[i] = (upper[i] + lower[i]) / 2.0; // midpoint is always within
  }

  const data_t lb = dtwc::core::lb_keogh(query, upper, lower);
  REQUIRE_THAT(lb, WithinAbs(0.0, 1e-15));
}

// ---------------------------------------------------------------------------
// Test 8: LB_Kim with single-element series
// ---------------------------------------------------------------------------
TEST_CASE("lb_kim single element series", "[lower_bounds][lb_kim]")
{
  const std::vector<data_t> x = { 5.0 };
  const std::vector<data_t> y = { 3.0 };

  const data_t lb = dtwc::core::lb_kim(x, y);
  // For single-element series, LB_Kim = |5 - 3| = 2
  // (first and last are same element; min/max branch skipped for length < 2)
  REQUIRE_THAT(lb, WithinAbs(2.0, 1e-15));

  // Also verify <= DTW
  const data_t dtw_dist = dtwc::dtwFull<data_t>(x, y);
  REQUIRE(lb <= dtw_dist + 1e-10);
}

// ---------------------------------------------------------------------------
// Test 9: LB_Kim with empty series returns 0
// ---------------------------------------------------------------------------
TEST_CASE("lb_kim empty series returns zero", "[lower_bounds][lb_kim]")
{
  const std::vector<data_t> empty;
  const std::vector<data_t> x = { 1.0, 2.0, 3.0 };

  REQUIRE_THAT(dtwc::core::lb_kim(empty, x), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(dtwc::core::lb_kim(x, empty), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(dtwc::core::lb_kim(empty, empty), WithinAbs(0.0, 1e-15));
}

// ---------------------------------------------------------------------------
// Test 10: LB_Kim <= DTW for known simple case with exact expected values
// ---------------------------------------------------------------------------
TEST_CASE("lb_kim <= dtw for known simple case", "[lower_bounds][lb_kim]")
{
  // x = {1, 2, 3}, y = {3, 4, 5, 6, 7}
  // DTW(x,y) = 13 (from existing warping tests)
  const std::vector<data_t> x = { 1.0, 2.0, 3.0 };
  const std::vector<data_t> y = { 3.0, 4.0, 5.0, 6.0, 7.0 };

  const data_t lb = dtwc::core::lb_kim(x, y);
  const data_t dtw_dist = dtwc::dtwFull<data_t>(x, y);

  REQUIRE_THAT(dtw_dist, WithinAbs(13.0, 1e-15));
  REQUIRE(lb <= dtw_dist + 1e-10);
  // LB_Kim: max(|1-3|, |3-7|, |1-3|, |3-7|) = max(2, 4, 2, 4) = 4
  REQUIRE_THAT(lb, WithinAbs(4.0, 1e-15));
}

// ---------------------------------------------------------------------------
// Test 11: Envelope with large band covers entire series
// ---------------------------------------------------------------------------
TEST_CASE("compute_envelopes large band covers full series", "[lower_bounds][envelopes]")
{
  const std::vector<data_t> series = { 3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0 };
  std::vector<data_t> upper, lower;

  // band = 100 >> series length, so every position sees all values
  dtwc::core::compute_envelopes(series, 100, upper, lower);

  const data_t series_min = *std::min_element(series.begin(), series.end());
  const data_t series_max = *std::max_element(series.begin(), series.end());

  for (std::size_t i = 0; i < series.size(); ++i) {
    REQUIRE_THAT(upper[i], WithinAbs(series_max, 1e-15));
    REQUIRE_THAT(lower[i], WithinAbs(series_min, 1e-15));
  }
}

// ---------------------------------------------------------------------------
// A5: band < 0 means UNBANDED DTW, not radius 0.
//
// lb_enhanced/lb_webb clamp the window with `max(band, 0)`, which collapses the
// elastic arms onto the diagonal. A diagonal cell is only forced when the band
// pins the path to it; an unbanded path may step around it entirely, so the
// clamp produces an INADMISSIBLE bound.
//
// Counterexample (L1, n = 4):
//   A = [0, 5, 0, 0], B = [0, 0, 5, 0]
//   full DTW path (0,0)->(0,1)->(1,2)->(2,3)->(3,3) costs 0,
//   while the clamped bound charges |A[1]-B[1]| + |A[2]-B[2]| = 10.
// ---------------------------------------------------------------------------

TEST_CASE("lb_enhanced is admissible for band < 0", "[lower_bounds][lb_enhanced][admissibility]")
{
  const std::vector<data_t> A = { 0.0, 5.0, 0.0, 0.0 };
  const std::vector<data_t> B = { 0.0, 0.0, 5.0, 0.0 };

  const data_t true_dtw = dtwc::dtwFull_L<data_t>(A, B);
  REQUIRE_THAT(true_dtw, WithinAbs(0.0, 1e-12));

  const auto env_B = dtwc::core::compute_envelope(B, -1);
  const double lb = dtwc::core::lb_enhanced(
    std::span<const double>(A), std::span<const double>(B), env_B, -1, 2);

  REQUIRE(lb <= true_dtw + 1e-12);
}

TEST_CASE("lb_webb is admissible for band < 0", "[lower_bounds][lb_webb][admissibility]")
{
  const std::vector<data_t> A = { 0.0, 5.0, 0.0, 0.0 };
  const std::vector<data_t> B = { 0.0, 0.0, 5.0, 0.0 };

  const data_t true_dtw = dtwc::dtwFull_L<data_t>(A, B);
  REQUIRE_THAT(true_dtw, WithinAbs(0.0, 1e-12));

  const auto ea = dtwc::core::compute_webb_envelope(A, -1);
  const auto eb = dtwc::core::compute_webb_envelope(B, -1);
  const double lb = dtwc::core::lb_webb(
    std::span<const double>(A), ea, std::span<const double>(B), eb, -1);

  REQUIRE(lb <= true_dtw + 1e-12);
}

// ---------------------------------------------------------------------------
// F46 `.lower` size gap: the public entry points validated only `env.upper`
// and then indexed `lower` / `ul` / `lu`, reading out of bounds when the
// arrays disagree. Every array an entry point reads is now checked.
//
// LB_Keogh keeps its D2-derived prefix semantics: it includes the first
// min(query.size(), env.upper.size()) rows, because the unequal-length prefix
// theorem shows that dropping the remaining rows discards only nonnegative
// terms. The size check therefore demands coverage of that prefix, not
// equality; only an array shorter than the prefix forces the trivial 0.
// LB_Enhanced and LB_Webb have no prefix theorem and keep exact equality.
//
// Fixture (both cases): A = [1,2,3,4] is the query, B = [4,3,2,1] the
// candidate, band 1, so the centred window [i-1, i+1] gives
//   env_B.upper = [4, 4, 3, 2],  env_B.lower = [3, 2, 1, 1].
// ---------------------------------------------------------------------------

TEST_CASE("lb_keogh truncates to the covered envelope prefix",
          "[lower_bounds][envelopes][sizes]")
{
  const std::vector<data_t> A = { 1.0, 2.0, 3.0, 4.0 };
  const std::vector<data_t> B = { 4.0, 3.0, 2.0, 1.0 };

  auto env_B = dtwc::core::compute_envelope(B, 1);
  REQUIRE(env_B.upper == std::vector<data_t>{ 4.0, 4.0, 3.0, 2.0 });
  REQUIRE(env_B.lower == std::vector<data_t>{ 3.0, 2.0, 1.0, 1.0 });

  env_B.upper.pop_back(); // upper.size() == 3, lower.size() == 4

  // n = min(4, 3) = 3, and lower covers [0, 3), so the prefix bound is taken:
  //   i=0: 1 < lower[0]=3 -> 2      i=1: 2 in [2,4] -> 0
  //   i=2: 3 in [1,3]     -> 0      total = 2
  CHECK(dtwc::core::lb_keogh(std::span<const double>(A), env_B) == 2.0);
  CHECK(dtwc::core::lb_keogh(A, env_B) == 2.0);
}

TEST_CASE("lower-bound entry points reject a ragged envelope",
          "[lower_bounds][envelopes][sizes]")
{
  const std::vector<data_t> A = { 1.0, 2.0, 3.0, 4.0 };
  const std::vector<data_t> B = { 4.0, 3.0, 2.0, 1.0 };

  auto env_B = dtwc::core::compute_envelope(B, 1);
  env_B.lower.pop_back(); // upper.size() == 4, lower.size() == 3

  // n = min(4, 4) = 4 but lower stops at 3, so indexing it over the prefix
  // would read out of bounds; the trivially admissible 0 is returned instead.

  CHECK(dtwc::core::lb_keogh(std::span<const double>(A), env_B) == 0.0);
  CHECK(dtwc::core::lb_keogh(A, env_B) == 0.0);
  CHECK(dtwc::core::lb_enhanced(
          std::span<const double>(A), std::span<const double>(B), env_B, 1, 2)
        == 0.0);

  auto ea = dtwc::core::compute_webb_envelope(A, 1);
  auto eb = dtwc::core::compute_webb_envelope(B, 1);
  eb.lu.pop_back(); // upper.size() == 4, lu.size() == 3
  CHECK(dtwc::core::lb_webb(
          std::span<const double>(A), ea, std::span<const double>(B), eb, 1)
        == 0.0);
}

// ---------------------------------------------------------------------------
// FX-13: every LB_Keogh is admissible for its cost and band.
//
// Oracle: the plain full-matrix DP of gpu_fixed_band_oracle.hpp under the
// literal |i-j| <= band predicate (every cell for band < 0); it shares no
// production DTW or envelope code. Both directions over the min(n, m) prefix
// (D2) and their maximum, for L1 and unrooted squared L2, full DTW and fixed
// bands. Values in [-1, 1) make excesses below 1 common, where an unsquared
// excess exceeds the squared cost it claims to bound (F27); full DTW is where a
// radius-zero envelope overshoots (F46). The witness counts prove the inputs
// can see each defect, so a green run is not vacuous.
// ---------------------------------------------------------------------------
TEST_CASE("FX-13 LB_Keogh <= exact DTW for L1 and squared L2 (full and banded)",
          "[lower_bounds][lb_keogh][property][FX-13]")
{
  namespace oracle = dtwc::test::gpu_fixed_band;
  using dtwc::test_support::benchmark_series;

  std::size_t checked = 0, violations = 0;
  std::size_t radius_zero_witnesses = 0, l1_unit_witnesses = 0;
  std::ostringstream first_violation;
  for (const bool squared : { false, true }) {
    for (const int band : { -1, 0, 2 }) {
      for (unsigned seed = 1; seed <= 200; ++seed) {
        const auto x = benchmark_series(1 + seed % 9, seed);
        const auto y = benchmark_series(1 + (seed / 9) % 9, seed + 7919U);
        const double exact = oracle::full_matrix_oracle(x, y, band, squared);

        std::vector<data_t> ux, lx, uy, ly, ux0, lx0, uy0, ly0;
        dtwc::core::compute_envelopes(x, band, ux, lx);
        dtwc::core::compute_envelopes(y, band, uy, ly);
        dtwc::core::compute_envelopes(x, 0, ux0, lx0); // pre-FX-13 band < 0
        dtwc::core::compute_envelopes(y, 0, uy0, ly0);

        const std::size_t n = std::min(x.size(), y.size());
        const auto l1 = [n](const auto &q, const auto &u, const auto &l) {
          return dtwc::core::lb_keogh(q.data(), n, u.data(), l.data());
        };
        const auto sq = [n](const auto &q, const auto &u, const auto &l) {
          return dtwc::core::lb_keogh_squared(q.data(), n, u.data(), l.data());
        };
        const auto bound_of = [&](const auto &f, const auto &ex, const auto &ey,
                                  const auto &fx, const auto &fy) {
          return std::max(f(x, ey, fy), f(y, ex, fx));
        };
        const double l1_bound = bound_of(l1, ux, uy, lx, ly);
        const double bound = squared ? bound_of(sq, ux, uy, lx, ly) : l1_bound;

        ++checked;
        if (!(bound <= exact * (1.0 + 1e-12) + 1e-12) && violations++ == 0)
          first_violation << "squared=" << squared << " band=" << band
                          << " seed=" << seed << " bound=" << bound
                          << " exact=" << exact;
        if (band < 0) {
          const double radius_zero = squared ? bound_of(sq, ux0, uy0, lx0, ly0)
                                             : bound_of(l1, ux0, uy0, lx0, ly0);
          if (radius_zero > exact) ++radius_zero_witnesses;
        }
        if (squared && l1_bound > exact) ++l1_unit_witnesses;
      }
    }
  }
  INFO("first violation: " << first_violation.str());
  CHECK(violations == 0);
  REQUIRE(checked == 1200);
  REQUIRE(radius_zero_witnesses > 0);
  REQUIRE(l1_unit_witnesses > 0);
}

TEST_CASE("FX-13 a negative band builds the global envelope (scalar and multivariate)",
          "[lower_bounds][envelopes][FX-13]")
{
  const std::vector<data_t> scalar = { 3.0, 1.0, 4.0, 1.0, 5.0 };
  const auto env = dtwc::core::compute_envelope(scalar, -1);
  CHECK(env.upper == std::vector<data_t>(5, 5.0));
  CHECK(env.lower == std::vector<data_t>(5, 1.0));

  // Channel 0 = {3, 1, 4, 1, 5}, channel 1 = {-2, 7, 0, -9, 6}, interleaved.
  const std::vector<data_t> mv = { 3, -2, 1, 7, 4, 0, 1, -9, 5, 6 };
  std::vector<data_t> upper(mv.size()), lower(mv.size());
  dtwc::core::compute_envelopes_mv(mv.data(), 5, 2, -1, upper.data(), lower.data());
  for (std::size_t t = 0; t < 5; ++t) {
    CHECK(upper[2 * t] == 5.0);
    CHECK(lower[2 * t] == 1.0);
    CHECK(upper[2 * t + 1] == 7.0);
    CHECK(lower[2 * t + 1] == -9.0);
  }
}
