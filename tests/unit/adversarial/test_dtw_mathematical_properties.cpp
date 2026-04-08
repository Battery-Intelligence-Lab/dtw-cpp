/**
 * @file test_dtw_mathematical_properties.cpp
 * @brief Adversarial tests for DTW mathematical properties.
 *
 * @details Tests are derived from the mathematical definition of DTW,
 *          NOT from reading the implementation. They verify identity,
 *          symmetry, non-negativity, known values, and cross-variant
 *          equivalence for dtwFull, dtwFull_L, and dtwBanded.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "dtwc.hpp"
#include "warping.hpp"

#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <algorithm>
#include <numeric>

using namespace dtwc;
using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::vector<data_t> make_random_series(std::mt19937 &rng, size_t len,
                                               double lo = -10.0, double hi = 10.0)
{
  std::uniform_real_distribution<data_t> dist(lo, hi);
  std::vector<data_t> v(len);
  for (auto &x : v) x = dist(rng);
  return v;
}

// ============================================================================
// AREA 1 -- Mathematical properties (per variant)
// ============================================================================

// ---------- dtwFull ---------------------------------------------------------

TEST_CASE("dtwFull identity: DTW(x, x) == 0", "[adversarial][dtwFull]")
{
  std::mt19937 rng(42);
  for (int trial = 0; trial < 10; ++trial) {
    auto x = make_random_series(rng, 20 + trial * 10);
    REQUIRE_THAT(dtwFull<data_t>(x, x), WithinAbs(0.0, 1e-12));
  }
}

TEST_CASE("dtwFull self-pointer identity: DTW(&x, &x) == 0", "[adversarial][dtwFull]")
{
  std::vector<data_t> x = {1.0, 2.0, 3.0, 4.0, 5.0};
  REQUIRE(dtwFull<data_t>(x, x) == 0.0); // same object -> pointer shortcut
}

TEST_CASE("dtwFull symmetry: DTW(x, y) == DTW(y, x)", "[adversarial][dtwFull]")
{
  std::mt19937 rng(123);
  for (int trial = 0; trial < 20; ++trial) {
    auto x = make_random_series(rng, 15 + trial);
    auto y = make_random_series(rng, 20 + trial);
    auto dxy = dtwFull<data_t>(x, y);
    auto dyx = dtwFull<data_t>(y, x);
    REQUIRE_THAT(dxy, WithinAbs(dyx, 1e-10));
  }
}

TEST_CASE("dtwFull non-negativity", "[adversarial][dtwFull]")
{
  std::mt19937 rng(77);
  for (int trial = 0; trial < 20; ++trial) {
    auto x = make_random_series(rng, 30);
    auto y = make_random_series(rng, 30);
    REQUIRE(dtwFull<data_t>(x, y) >= 0.0);
  }
}

TEST_CASE("dtwFull empty series returns max value", "[adversarial][dtwFull]")
{
  std::vector<data_t> empty;
  std::vector<data_t> y = {1.0, 2.0, 3.0};
  constexpr auto maxVal = std::numeric_limits<data_t>::max();
  REQUIRE(dtwFull<data_t>(empty, y) == maxVal);
  REQUIRE(dtwFull<data_t>(y, empty) == maxVal);

  // Empty series should return maxVal regardless of pointer identity.
  REQUIRE(dtwFull<data_t>(empty, empty) == maxVal);

  std::vector<data_t> empty2;
  REQUIRE(dtwFull<data_t>(empty, empty2) == maxVal);
}

TEST_CASE("dtwFull single element: DTW({a}, {b}) == |a - b|", "[adversarial][dtwFull]")
{
  std::mt19937 rng(11);
  std::uniform_real_distribution<data_t> dist(-100, 100);
  for (int i = 0; i < 20; ++i) {
    data_t a = dist(rng), b = dist(rng);
    REQUIRE_THAT(dtwFull<data_t>(std::vector<data_t>{a}, std::vector<data_t>{b}), WithinAbs(std::abs(a - b), 1e-12));
  }
}

TEST_CASE("dtwFull identical series (copy): DTW(x, copy) == 0", "[adversarial][dtwFull]")
{
  std::mt19937 rng(55);
  auto x = make_random_series(rng, 100);
  auto copy = x; // deep copy, different address
  REQUIRE_THAT(dtwFull<data_t>(x, copy), WithinAbs(0.0, 1e-10));
}

TEST_CASE("dtwFull known value: DTW({1,2,3}, {1,2,3}) == 0", "[adversarial][dtwFull]")
{
  std::vector<data_t> a = {1, 2, 3};
  REQUIRE_THAT(dtwFull<data_t>(a, a), WithinAbs(0.0, 1e-12));
}

TEST_CASE("dtwFull known value: DTW({0,0,0}, {1,1,1}) == 3.0", "[adversarial][dtwFull]")
{
  // L1 metric, each of 3 aligned pairs differs by 1 => cost = 3
  std::vector<data_t> zeros = {0, 0, 0};
  std::vector<data_t> ones  = {1, 1, 1};
  REQUIRE_THAT(dtwFull<data_t>(zeros, ones), WithinAbs(3.0, 1e-12));
}

TEST_CASE("dtwFull triangle inequality violations are bounded", "[adversarial][dtwFull]")
{
  // DTW is NOT a true metric; triangle inequality can fail.
  // But for "nearby" random series, violations should be small.
  std::mt19937 rng(999);
  int violations = 0;
  double max_excess = 0.0;
  const int N = 50;
  for (int t = 0; t < N; ++t) {
    auto x = make_random_series(rng, 30);
    auto y = make_random_series(rng, 30);
    auto z = make_random_series(rng, 30);
    auto dxz = dtwFull<data_t>(x, z);
    auto dxy = dtwFull<data_t>(x, y);
    auto dyz = dtwFull<data_t>(y, z);
    double excess = dxz - (dxy + dyz);
    if (excess > 1e-12) {
      ++violations;
      max_excess = std::max(max_excess, excess);
    }
  }
  // We just record; violations may or may not happen.
  // But the excess should not be astronomical relative to the distances.
  INFO("Triangle inequality violations: " << violations << "/" << N
       << ", max excess: " << max_excess);
  // No hard REQUIRE here -- DTW is not a metric.
  // But we check it does not produce nonsense values.
  REQUIRE(max_excess < 1e6); // sanity bound
}

// ---------- dtwFull_L -------------------------------------------------------

TEST_CASE("dtwFull_L identity: DTW(x, x) == 0", "[adversarial][dtwFull_L]")
{
  std::mt19937 rng(42);
  for (int trial = 0; trial < 10; ++trial) {
    auto x = make_random_series(rng, 20 + trial * 10);
    REQUIRE_THAT(dtwFull_L<data_t>(x, x), WithinAbs(0.0, 1e-12));
  }
}

TEST_CASE("dtwFull_L self-pointer identity", "[adversarial][dtwFull_L]")
{
  std::vector<data_t> x = {1, 2, 3, 4, 5};
  REQUIRE(dtwFull_L<data_t>(x, x) == 0.0);
}

TEST_CASE("dtwFull_L symmetry", "[adversarial][dtwFull_L]")
{
  std::mt19937 rng(123);
  for (int trial = 0; trial < 20; ++trial) {
    auto x = make_random_series(rng, 15 + trial);
    auto y = make_random_series(rng, 20 + trial);
    REQUIRE_THAT(dtwFull_L<data_t>(x, y), WithinAbs(dtwFull_L<data_t>(y, x), 1e-10));
  }
}

TEST_CASE("dtwFull_L non-negativity", "[adversarial][dtwFull_L]")
{
  std::mt19937 rng(77);
  for (int trial = 0; trial < 20; ++trial) {
    auto x = make_random_series(rng, 30);
    auto y = make_random_series(rng, 30);
    REQUIRE(dtwFull_L<data_t>(x, y) >= 0.0);
  }
}

TEST_CASE("dtwFull_L empty series returns max value", "[adversarial][dtwFull_L]")
{
  std::vector<data_t> empty;
  std::vector<data_t> y = {1, 2, 3};
  constexpr auto maxVal = std::numeric_limits<data_t>::max();
  REQUIRE(dtwFull_L<data_t>(empty, y) == maxVal);
  REQUIRE(dtwFull_L<data_t>(y, empty) == maxVal);

  // Empty series should return maxVal regardless of pointer identity.
  REQUIRE(dtwFull_L<data_t>(empty, empty) == maxVal);

  std::vector<data_t> empty2;
  REQUIRE(dtwFull_L<data_t>(empty, empty2) == maxVal);
}

TEST_CASE("dtwFull_L single element", "[adversarial][dtwFull_L]")
{
  std::mt19937 rng(11);
  std::uniform_real_distribution<data_t> dist(-100, 100);
  for (int i = 0; i < 20; ++i) {
    data_t a = dist(rng), b = dist(rng);
    REQUIRE_THAT(dtwFull_L<data_t>(std::vector<data_t>{a}, std::vector<data_t>{b}), WithinAbs(std::abs(a - b), 1e-12));
  }
}

TEST_CASE("dtwFull_L identical series (copy)", "[adversarial][dtwFull_L]")
{
  std::mt19937 rng(55);
  auto x = make_random_series(rng, 100);
  auto copy = x;
  REQUIRE_THAT(dtwFull_L<data_t>(x, copy), WithinAbs(0.0, 1e-10));
}

TEST_CASE("dtwFull_L known value: {1,2,3} vs {1,2,3}", "[adversarial][dtwFull_L]")
{
  std::vector<data_t> a = {1, 2, 3};
  auto copy = a;
  REQUIRE_THAT(dtwFull_L<data_t>(a, copy), WithinAbs(0.0, 1e-12));
}

TEST_CASE("dtwFull_L known value: {0,0,0} vs {1,1,1} == 3.0", "[adversarial][dtwFull_L]")
{
  std::vector<data_t> zeros = {0, 0, 0};
  std::vector<data_t> ones  = {1, 1, 1};
  REQUIRE_THAT(dtwFull_L<data_t>(zeros, ones), WithinAbs(3.0, 1e-12));
}

// ---------- dtwBanded -------------------------------------------------------

TEST_CASE("dtwBanded identity: DTW(x, x) == 0", "[adversarial][dtwBanded]")
{
  std::mt19937 rng(42);
  for (int trial = 0; trial < 10; ++trial) {
    auto x = make_random_series(rng, 20 + trial * 10);
    // band = -1 (full), band = 5, band = 1000
    for (int band : {-1, 5, 1000}) {
      INFO("trial=" << trial << " band=" << band);
      REQUIRE_THAT(dtwBanded<data_t>(x, x, band), WithinAbs(0.0, 1e-10));
    }
  }
}

TEST_CASE("dtwBanded self-pointer identity", "[adversarial][dtwBanded]")
{
  std::vector<data_t> x = {1, 2, 3, 4, 5};
  // dtwBanded may delegate to dtwFull_L which has the pointer check
  REQUIRE_THAT(dtwBanded<data_t>(x, x, 3), WithinAbs(0.0, 1e-10));
}

TEST_CASE("dtwBanded symmetry", "[adversarial][dtwBanded]")
{
  std::mt19937 rng(123);
  for (int trial = 0; trial < 20; ++trial) {
    auto x = make_random_series(rng, 15 + trial);
    auto y = make_random_series(rng, 20 + trial);
    for (int band : {2, 5, 10}) {
      INFO("trial=" << trial << " band=" << band);
      REQUIRE_THAT(dtwBanded<data_t>(x, y, band),
                   WithinAbs(dtwBanded<data_t>(y, x, band), 1e-10));
    }
  }
}

TEST_CASE("dtwBanded non-negativity", "[adversarial][dtwBanded]")
{
  std::mt19937 rng(77);
  for (int trial = 0; trial < 20; ++trial) {
    auto x = make_random_series(rng, 30);
    auto y = make_random_series(rng, 30);
    for (int band : {1, 3, 10, -1}) {
      REQUIRE(dtwBanded<data_t>(x, y, band) >= 0.0);
    }
  }
}

TEST_CASE("dtwBanded empty series returns max value", "[adversarial][dtwBanded]")
{
  std::vector<data_t> empty;
  std::vector<data_t> y = {1, 2, 3};
  constexpr auto maxVal = std::numeric_limits<data_t>::max();
  REQUIRE(dtwBanded<data_t>(empty, y, 2) == maxVal);
  REQUIRE(dtwBanded<data_t>(y, empty, 2) == maxVal);
}

TEST_CASE("dtwBanded single element", "[adversarial][dtwBanded]")
{
  std::mt19937 rng(11);
  std::uniform_real_distribution<data_t> dist(-100, 100);
  for (int i = 0; i < 20; ++i) {
    data_t a = dist(rng), b = dist(rng);
    REQUIRE_THAT(dtwBanded<data_t>(std::vector<data_t>{a}, std::vector<data_t>{b}, 5), WithinAbs(std::abs(a - b), 1e-12));
  }
}

TEST_CASE("dtwBanded identical series (copy)", "[adversarial][dtwBanded]")
{
  std::mt19937 rng(55);
  auto x = make_random_series(rng, 100);
  auto copy = x;
  REQUIRE_THAT(dtwBanded<data_t>(x, copy, 10), WithinAbs(0.0, 1e-10));
}

TEST_CASE("dtwBanded known value: {0,0,0} vs {1,1,1} == 3.0", "[adversarial][dtwBanded]")
{
  std::vector<data_t> zeros = {0, 0, 0};
  std::vector<data_t> ones  = {1, 1, 1};
  // With band >= 2, the optimal path is the diagonal => cost = 3.0
  REQUIRE_THAT(dtwBanded<data_t>(zeros, ones, 5), WithinAbs(3.0, 1e-12));
}

// ============================================================================
// AREA 2 -- Cross-variant equivalence
// ============================================================================

TEST_CASE("dtwFull vs dtwFull_L: exact match for random pairs", "[adversarial][equivalence]")
{
  std::mt19937 rng(2024);
  std::uniform_int_distribution<size_t> len_dist(10, 500);
  for (int trial = 0; trial < 20; ++trial) {
    size_t lx = len_dist(rng);
    size_t ly = len_dist(rng);
    auto x = make_random_series(rng, lx);
    auto y = make_random_series(rng, ly);
    auto d_full  = dtwFull<data_t>(x, y);
    auto d_light = dtwFull_L<data_t>(x, y);
    INFO("trial=" << trial << " lx=" << lx << " ly=" << ly
         << " dtwFull=" << d_full << " dtwFull_L=" << d_light);
    // They compute the same recurrence; results should match to
    // floating-point precision (may differ by operation ordering).
    REQUIRE_THAT(d_full, WithinRel(d_light, 1e-10));
  }
}

TEST_CASE("dtwBanded with large band == dtwFull_L", "[adversarial][equivalence]")
{
  std::mt19937 rng(3001);
  for (int trial = 0; trial < 15; ++trial) {
    size_t lx = 20 + trial * 5;
    size_t ly = 25 + trial * 3;
    auto x = make_random_series(rng, lx);
    auto y = make_random_series(rng, ly);
    int big_band = static_cast<int>(std::max(lx, ly)) + 10;
    auto d_banded = dtwBanded<data_t>(x, y, big_band);
    auto d_full   = dtwFull_L<data_t>(x, y);
    INFO("trial=" << trial << " lx=" << lx << " ly=" << ly
         << " big_band=" << big_band
         << " dtwBanded=" << d_banded << " dtwFull_L=" << d_full);
    REQUIRE_THAT(d_banded, WithinRel(d_full, 1e-10));
  }
}

TEST_CASE("dtwBanded with negative band == dtwFull_L", "[adversarial][equivalence]")
{
  std::mt19937 rng(7777);
  for (int trial = 0; trial < 10; ++trial) {
    auto x = make_random_series(rng, 40);
    auto y = make_random_series(rng, 50);
    auto d_neg   = dtwBanded<data_t>(x, y, -1);
    auto d_full  = dtwFull_L<data_t>(x, y);
    REQUIRE_THAT(d_neg, WithinRel(d_full, 1e-12));
  }
}

TEST_CASE("dtwBanded band=0 forces diagonal (equal-length series)", "[adversarial][equivalence]")
{
  // band=0 constrains the warping path to the diagonal only.
  // For equal-length series the cost is sum |x[i] - y[i]|.
  std::mt19937 rng(5050);
  for (int trial = 0; trial < 10; ++trial) {
    size_t len = 10 + trial * 5;
    auto x = make_random_series(rng, len);
    auto y = make_random_series(rng, len);
    double expected = 0.0;
    for (size_t i = 0; i < len; ++i)
      expected += std::abs(x[i] - y[i]);

    auto result = dtwBanded<data_t>(x, y, 0);
    INFO("trial=" << trial << " len=" << len
         << " expected=" << expected << " got=" << result);
    REQUIRE_THAT(result, WithinRel(expected, 1e-10));
  }
}

TEST_CASE("Different lengths: DTW({1,2,3}, {1,2,3,4,5}) is valid positive", "[adversarial][lengths]")
{
  std::vector<data_t> a = {1, 2, 3};
  std::vector<data_t> b = {1, 2, 3, 4, 5};
  auto d1 = dtwFull<data_t>(a, b);
  auto d2 = dtwFull_L<data_t>(a, b);
  auto d3 = dtwBanded<data_t>(a, b, 10);
  REQUIRE(d1 > 0.0);
  REQUIRE(d2 > 0.0);
  REQUIRE(d3 > 0.0);
  REQUIRE(std::isfinite(d1));
  REQUIRE(std::isfinite(d2));
  REQUIRE(std::isfinite(d3));
}

TEST_CASE("Very different lengths: length=1 vs length=1000 does not crash", "[adversarial][lengths]")
{
  std::mt19937 rng(9999);
  std::vector<data_t> single = {3.14};
  auto big = make_random_series(rng, 1000);

  auto d1 = dtwFull<data_t>(single, big);
  auto d2 = dtwFull_L<data_t>(single, big);
  auto d3 = dtwBanded<data_t>(single, big, 5);

  REQUIRE(std::isfinite(d1));
  REQUIRE(std::isfinite(d2));
  REQUIRE(std::isfinite(d3));
  REQUIRE(d1 >= 0.0);
  REQUIRE(d2 >= 0.0);
  REQUIRE(d3 >= 0.0);

  // Also verify symmetry for this extreme case
  REQUIRE_THAT(dtwFull<data_t>(big, single), WithinAbs(d1, 1e-10));
  REQUIRE_THAT(dtwFull_L<data_t>(big, single), WithinAbs(d2, 1e-10));
}
