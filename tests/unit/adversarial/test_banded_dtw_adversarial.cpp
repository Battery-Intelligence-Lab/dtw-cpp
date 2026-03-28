/**
 * @file test_banded_dtw_adversarial.cpp
 * @brief Adversarial tests for banded DTW boundary conditions and rolling buffer correctness.
 *
 * Tests are derived from the Sakoe-Chiba band specification:
 *   H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization for spoken
 *   word recognition". IEEE Trans. Acoustics, Speech, Signal Processing, 26(1), 43-49 (1978).
 *
 * @date 28 Mar 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>
#include <random>
#include <cmath>
#include <numeric>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using data_t = double;

namespace {

// Generate a random series of given length using the provided RNG.
std::vector<data_t> random_series(std::mt19937 &rng, int len, double lo = -10.0, double hi = 10.0)
{
  std::uniform_real_distribution<data_t> dist(lo, hi);
  std::vector<data_t> v(static_cast<size_t>(len));
  for (auto &val : v)
    val = dist(rng);
  return v;
}

} // anonymous namespace

// ===========================================================================
// Area 1: Banded DTW Boundary Conditions
// ===========================================================================

TEST_CASE("Banded DTW: band < 0 falls back to full DTW", "[dtwBanded][boundary]")
{
  std::mt19937 rng(42);

  SECTION("Short equal-length series") {
    auto x = random_series(rng, 20);
    auto y = random_series(rng, 20);
    const auto banded = dtwc::dtwBanded<data_t>(x, y, -1);
    const auto full = dtwc::dtwFull_L<data_t>(x, y);
    REQUIRE_THAT(banded, WithinAbs(full, 1e-12));
  }

  SECTION("Unequal-length series") {
    auto x = random_series(rng, 15);
    auto y = random_series(rng, 30);
    const auto banded = dtwc::dtwBanded<data_t>(x, y, -1);
    const auto full = dtwc::dtwFull_L<data_t>(x, y);
    REQUIRE_THAT(banded, WithinAbs(full, 1e-12));
  }

  SECTION("band = -100 also falls back") {
    auto x = random_series(rng, 10);
    auto y = random_series(rng, 10);
    const auto banded = dtwc::dtwBanded<data_t>(x, y, -100);
    const auto full = dtwc::dtwFull_L<data_t>(x, y);
    REQUIRE_THAT(banded, WithinAbs(full, 1e-12));
  }
}

TEST_CASE("Banded DTW: band = 0 forces diagonal alignment for equal-length series", "[dtwBanded][boundary]")
{
  // With band=0 and equal lengths, the only valid path is the diagonal.
  // Cost should equal sum of |x[i] - y[i]|.
  // NOTE: The implementation falls back to dtwFull_L when m_long <= (band+1),
  // i.e., when length <= 1. For length >= 2, band=0 is used.

  SECTION("Equal-length series, length 2") {
    // For length=2, m_long=2, band=0 => m_long <= band+1 => 2 <= 1 is FALSE.
    // So banding IS applied. Diagonal path cost = |x[0]-y[0]| + |x[1]-y[1]|.
    std::vector<data_t> x{1.0, 5.0};
    std::vector<data_t> y{3.0, 2.0};
    const auto result = dtwc::dtwBanded<data_t>(x, y, 0);
    const data_t diagonal_cost = std::abs(1.0 - 3.0) + std::abs(5.0 - 2.0); // 2 + 3 = 5
    REQUIRE_THAT(result, WithinAbs(diagonal_cost, 1e-12));
  }

  SECTION("Equal-length series, length 5") {
    std::vector<data_t> x{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<data_t> y{5.0, 4.0, 3.0, 2.0, 1.0};
    const auto result = dtwc::dtwBanded<data_t>(x, y, 0);
    // Diagonal cost: |1-5| + |2-4| + |3-3| + |4-2| + |5-1| = 4+2+0+2+4 = 12
    const data_t diagonal_cost = 12.0;
    REQUIRE_THAT(result, WithinAbs(diagonal_cost, 1e-12));
  }
}

TEST_CASE("Banded DTW: band >= max(len) equals full DTW", "[dtwBanded][boundary]")
{
  std::mt19937 rng(42);

  for (int trial = 0; trial < 10; ++trial) {
    const int len_x = 5 + (trial * 3);
    const int len_y = 8 + (trial * 2);
    auto x = random_series(rng, len_x);
    auto y = random_series(rng, len_y);

    const auto banded = dtwc::dtwBanded<data_t>(x, y, 10000);
    const auto full = dtwc::dtwFull_L<data_t>(x, y);
    REQUIRE_THAT(banded, WithinAbs(full, 1e-10));
  }
}

TEST_CASE("Banded DTW: larger band gives <= cost (monotonicity)", "[dtwBanded][boundary]")
{
  // More paths available with larger band => cost can only decrease or stay the same.
  std::mt19937 rng(42);

  SECTION("Systematic band sweep") {
    auto x = random_series(rng, 50);
    auto y = random_series(rng, 50);

    data_t prev_cost = dtwc::dtwBanded<data_t>(x, y, 0);
    for (int band = 1; band <= 50; ++band) {
      const auto cost = dtwc::dtwBanded<data_t>(x, y, band);
      REQUIRE(cost <= prev_cost + 1e-12); // allow tiny floating-point noise
      prev_cost = cost;
    }
  }

  SECTION("Unequal lengths") {
    auto x = random_series(rng, 30);
    auto y = random_series(rng, 60);

    data_t prev_cost = dtwc::dtwBanded<data_t>(x, y, 0);
    for (int band = 1; band <= 60; ++band) {
      const auto cost = dtwc::dtwBanded<data_t>(x, y, band);
      REQUIRE(cost <= prev_cost + 1e-12);
      prev_cost = cost;
    }
  }
}

TEST_CASE("Banded DTW: band=1 allows +/-1 diagonal deviation", "[dtwBanded][boundary]")
{
  // Construct series where optimal full DTW path needs exactly +/-1 deviation from diagonal.
  // x = [0, 0, 1, 1, 0]
  // y = [0, 1, 1, 0, 0]
  // The optimal alignment shifts y[1]=1 to match x[2]=1, which is a +1 deviation.
  // Band=1 should allow this. Band=0 (diagonal only) should give higher cost.
  std::vector<data_t> x{0.0, 0.0, 1.0, 1.0, 0.0};
  std::vector<data_t> y{0.0, 1.0, 1.0, 0.0, 0.0};

  const auto cost_band0 = dtwc::dtwBanded<data_t>(x, y, 0);
  const auto cost_band1 = dtwc::dtwBanded<data_t>(x, y, 1);
  const auto cost_full = dtwc::dtwFull_L<data_t>(x, y);

  // Band=1 should be at least as good as band=0
  REQUIRE(cost_band1 <= cost_band0 + 1e-12);

  // Band=1 should match full DTW for this gentle shift
  // (full DTW path only needs +/-1 deviation)
  REQUIRE_THAT(cost_band1, WithinAbs(cost_full, 1e-12));
}

TEST_CASE("Banded DTW: unequal lengths", "[dtwBanded][boundary]")
{
  SECTION("Short vs long") {
    std::vector<data_t> x{1.0, 2.0, 3.0};
    std::vector<data_t> y{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

    // Should not crash, should return finite value
    const auto result = dtwc::dtwBanded<data_t>(x, y, 2);
    REQUIRE(std::isfinite(result));
    REQUIRE(result >= 0.0);
  }

  SECTION("Length ratio 1:10") {
    std::mt19937 rng(42);
    auto x = random_series(rng, 10);
    auto y = random_series(rng, 100);

    const auto result = dtwc::dtwBanded<data_t>(x, y, 5);
    REQUIRE(std::isfinite(result));
    REQUIRE(result >= 0.0);
  }
}

TEST_CASE("Banded DTW: very short series with band", "[dtwBanded][boundary]")
{
  // Single-element series
  std::vector<data_t> x{1.0};
  std::vector<data_t> y{2.0};
  const auto result = dtwc::dtwBanded<data_t>(x, y, 10);
  REQUIRE_THAT(result, WithinAbs(1.0, 1e-12));

  // Single-element, same value
  std::vector<data_t> a{5.0};
  std::vector<data_t> b{5.0};
  REQUIRE_THAT(dtwc::dtwBanded<data_t>(a, b, 10), WithinAbs(0.0, 1e-12));

  // Single-element vs multi-element
  std::vector<data_t> c{0.0};
  std::vector<data_t> d{1.0, 2.0, 3.0};
  const auto result2 = dtwc::dtwBanded<data_t>(c, d, 10);
  // Full DTW of {0} vs {1,2,3}: cost = |0-1| + |0-2| + |0-3| = 1+2+3 = 6
  REQUIRE_THAT(result2, WithinAbs(6.0, 1e-12));
}

// ===========================================================================
// Area 2: Rolling Buffer Memory Correctness
// ===========================================================================

TEST_CASE("Rolling buffer vs full matrix: random pairs with large band", "[dtwBanded][buffer]")
{
  // When band covers the full matrix, dtwBanded must equal dtwFull_L.
  std::mt19937 rng(42);

  for (int trial = 0; trial < 50; ++trial) {
    std::uniform_int_distribution<int> len_dist(2, 100);
    const int len_x = len_dist(rng);
    const int len_y = len_dist(rng);
    auto x = random_series(rng, len_x);
    auto y = random_series(rng, len_y);

    const int big_band = std::max(len_x, len_y) + 10;
    const auto banded = dtwc::dtwBanded<data_t>(x, y, big_band);
    const auto full = dtwc::dtwFull_L<data_t>(x, y);

    REQUIRE_THAT(banded, WithinAbs(full, 1e-10));
  }
}

TEST_CASE("Large series with small band: no crash, reasonable memory", "[dtwBanded][buffer]")
{
  // N=5000, band=5. The rolling buffer should handle this without excessive memory.
  std::mt19937 rng(42);
  auto x = random_series(rng, 5000);
  auto y = random_series(rng, 5000);

  const auto result = dtwc::dtwBanded<data_t>(x, y, 5);
  REQUIRE(std::isfinite(result));
  REQUIRE(result >= 0.0);
}

TEST_CASE("Repeated calls with different sizes: thread-local buffer reuse", "[dtwBanded][buffer]")
{
  // Call dtwBanded with varying sizes to exercise thread-local buffer resizing.
  std::mt19937 rng(42);

  // Call 1: 100x100, band=5
  auto x1 = random_series(rng, 100);
  auto y1 = random_series(rng, 100);
  const auto r1 = dtwc::dtwBanded<data_t>(x1, y1, 5);
  REQUIRE(std::isfinite(r1));
  REQUIRE(r1 >= 0.0);

  // Call 2: 50x50, band=10
  auto x2 = random_series(rng, 50);
  auto y2 = random_series(rng, 50);
  const auto r2 = dtwc::dtwBanded<data_t>(x2, y2, 10);
  REQUIRE(std::isfinite(r2));
  REQUIRE(r2 >= 0.0);

  // Call 3: 200x200, band=3
  auto x3 = random_series(rng, 200);
  auto y3 = random_series(rng, 200);
  const auto r3 = dtwc::dtwBanded<data_t>(x3, y3, 3);
  REQUIRE(std::isfinite(r3));
  REQUIRE(r3 >= 0.0);

  // Verify correctness by comparing with full DTW (large band)
  const auto r1_full = dtwc::dtwBanded<data_t>(x1, y1, 200);
  const auto r1_ref = dtwc::dtwFull_L<data_t>(x1, y1);
  REQUIRE_THAT(r1_full, WithinAbs(r1_ref, 1e-10));

  const auto r2_full = dtwc::dtwBanded<data_t>(x2, y2, 200);
  const auto r2_ref = dtwc::dtwFull_L<data_t>(x2, y2);
  REQUIRE_THAT(r2_full, WithinAbs(r2_ref, 1e-10));

  const auto r3_full = dtwc::dtwBanded<data_t>(x3, y3, 200);
  const auto r3_ref = dtwc::dtwFull_L<data_t>(x3, y3);
  REQUIRE_THAT(r3_full, WithinAbs(r3_ref, 1e-10));
}

TEST_CASE("Symmetry with band: dtwBanded(x,y,band) == dtwBanded(y,x,band)", "[dtwBanded][buffer]")
{
  std::mt19937 rng(42);

  SECTION("Equal-length series, various bands") {
    auto x = random_series(rng, 40);
    auto y = random_series(rng, 40);

    for (int band : {0, 1, 2, 5, 10, 20, 100}) {
      const auto xy = dtwc::dtwBanded<data_t>(x, y, band);
      const auto yx = dtwc::dtwBanded<data_t>(y, x, band);
      REQUIRE_THAT(xy, WithinAbs(yx, 1e-12));
    }
  }

  SECTION("Unequal-length series, various bands") {
    auto x = random_series(rng, 25);
    auto y = random_series(rng, 50);

    for (int band : {0, 1, 3, 10, 30, 100}) {
      const auto xy = dtwc::dtwBanded<data_t>(x, y, band);
      const auto yx = dtwc::dtwBanded<data_t>(y, x, band);
      REQUIRE_THAT(xy, WithinAbs(yx, 1e-12));
    }
  }
}

TEST_CASE("Non-negativity with band: dtwBanded(x,y,band) >= 0", "[dtwBanded][buffer]")
{
  std::mt19937 rng(42);

  for (int trial = 0; trial < 30; ++trial) {
    std::uniform_int_distribution<int> len_dist(1, 100);
    std::uniform_int_distribution<int> band_dist(0, 50);
    const int len_x = len_dist(rng);
    const int len_y = len_dist(rng);
    const int band = band_dist(rng);

    auto x = random_series(rng, len_x);
    auto y = random_series(rng, len_y);

    const auto result = dtwc::dtwBanded<data_t>(x, y, band);
    REQUIRE(result >= 0.0);
  }

  // Identity: distance to self is zero
  auto x = random_series(rng, 50);
  REQUIRE_THAT(dtwc::dtwBanded<data_t>(x, x, 5), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(dtwc::dtwBanded<data_t>(x, x, 0), WithinAbs(0.0, 1e-12));
}
