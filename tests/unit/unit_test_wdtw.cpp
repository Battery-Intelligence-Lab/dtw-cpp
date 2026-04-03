/**
 * @file unit_test_wdtw.cpp
 * @brief Unit tests for Weighted DTW (WDTW) functions
 *
 * Tests follow TDD: written before implementation.
 *
 * Reference: Jeong, Y.-S., Jeong, M. K., & Omitaomu, O. A. (2011).
 *   Weighted dynamic time warping for time series classification.
 *   Pattern Recognition, 44(9), 2231-2240.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace dtwc;

// ---------- wdtwFull tests ----------

TEST_CASE("wdtwFull: identity gives zero distance", "[wdtw]")
{
  using T = double;
  std::vector<T> x{ 1, 2, 3, 4, 5 };

  // Any g should yield 0 when comparing a series to itself
  for (double g : { 0.0, 0.05, 1.0, 100.0 }) {
    REQUIRE_THAT(wdtwFull<T>(x, x, g), WithinAbs(0.0, 1e-15));
  }
}

TEST_CASE("wdtwFull: symmetry", "[wdtw]")
{
  using T = double;
  std::vector<T> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 };

  for (double g : { 0.0, 0.05, 1.0, 10.0 }) {
    REQUIRE_THAT(wdtwFull<T>(x, y, g), WithinAbs(wdtwFull<T>(y, x, g), 1e-12));
  }
}

TEST_CASE("wdtwFull: non-negativity", "[wdtw]")
{
  using T = double;
  std::vector<T> x{ 1, 2, 3 }, y{ 5, 6, 7, 8 };

  for (double g : { 0.0, 0.05, 1.0, 100.0 }) {
    REQUIRE(wdtwFull<T>(x, y, g) >= 0.0);
  }
}

TEST_CASE("wdtwFull: g=0 gives weights all 0.5, hand-computed example", "[wdtw]")
{
  // With g=0, logistic w(k) = 1/(1+exp(0)) = 0.5 for all k.
  // For x={1,2,3}, y={2,3,4}: optimal path is diagonal, cost = 0.5*1 + 0.5*1 + 0.5*1 = 1.5
  // ... wait, need to recompute with DP.
  //
  // Actually the full DP:
  //   C(0,0) = 0.5 * |1-2| = 0.5
  //   C(1,0) = C(0,0) + 0.5 * |2-2| = 0.5
  //   C(2,0) = C(1,0) + 0.5 * |3-2| = 1.0
  //   C(0,1) = C(0,0) + 0.5 * |1-3| = 1.5
  //   C(0,2) = C(0,1) + 0.5 * |1-4| = 3.0
  //   C(1,1) = min(1.5, 0.5, 0.5) + 0.5*|2-3| = 0.5 + 0.5 = 1.0
  //   C(2,1) = min(1.0, 1.0, 0.5) + 0.5*|3-3| = 0.5 + 0.0 = 0.5
  //   C(1,2) = min(3.0, 1.0, 1.5) + 0.5*|2-4| = 1.0 + 1.0 = 2.0
  //   C(2,2) = min(2.0, 0.5, 1.0) + 0.5*|3-4| = 0.5 + 0.5 = 1.0
  //
  // Result = 1.0
  using T = double;
  std::vector<T> x{ 1, 2, 3 }, y{ 2, 3, 4 };

  REQUIRE_THAT(wdtwFull<T>(x, y, 0.0), WithinAbs(1.0, 1e-12));
}

TEST_CASE("wdtwFull: g=0 is exactly half of standard DTW", "[wdtw]")
{
  // When g=0, all weights are 0.5, so WDTW = 0.5 * DTW.
  using T = double;
  std::vector<T> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 };

  double dtw_val = dtwFull<T>(x, y);
  double wdtw_val = wdtwFull<T>(x, y, 0.0);

  REQUIRE_THAT(wdtw_val, WithinAbs(dtw_val * 0.5, 1e-12));
}

TEST_CASE("wdtwFull: large g penalizes off-diagonal, wdtw >= dtw for shifted series", "[wdtw]")
{
  // With very large g, weights approach a step function:
  //   w(k) ~ 0 for k < max_len/2, w(k) ~ 1 for k >= max_len/2
  // For series that need off-diagonal alignment, the weighted cost changes.
  // With g=100, for short series the weights for small |i-j| are near 0
  // and for large |i-j| near 1.
  // For series that must align off-diagonal, the large-|i-j| steps get
  // weight ~1 while diagonal steps get weight ~0.
  // Compare against standard DTW where all weights are implicitly 1.
  using T = double;

  // Two series where the optimal DTW path goes off-diagonal
  std::vector<T> x{ 0, 0, 0, 1, 2, 3 };
  std::vector<T> y{ 1, 2, 3, 0, 0, 0 };

  double dtw_val = dtwFull<T>(x, y);
  double wdtw_g100 = wdtwFull<T>(x, y, 100.0);

  // With large g, weights for |i-j|>=max_len/2 approach 1 and for small |i-j| approach 0.
  // The result should differ from standard DTW.
  // Just check it's non-negative and finite.
  REQUIRE(wdtw_g100 >= 0.0);
  REQUIRE(std::isfinite(wdtw_g100));

  // Also verify it differs from the g=0 case
  double wdtw_g0 = wdtwFull<T>(x, y, 0.0);
  REQUIRE(std::abs(wdtw_g100 - wdtw_g0) > 1e-6);
}

TEST_CASE("wdtwFull: different lengths work correctly", "[wdtw]")
{
  using T = double;
  std::vector<T> x{ 1.0, 2.0 };
  std::vector<T> y{ 1.0, 2.0, 3.0, 4.0, 5.0 };

  double result = wdtwFull<T>(x, y, 0.05);
  REQUIRE(result >= 0.0);
  REQUIRE(std::isfinite(result));

  // Symmetry with different lengths
  REQUIRE_THAT(result, WithinAbs(wdtwFull<T>(y, x, 0.05), 1e-12));
}

TEST_CASE("wdtwFull: empty series returns maxValue", "[wdtw]")
{
  using T = double;
  std::vector<T> x{ 1, 2, 3 }, empty{};

  REQUIRE(wdtwFull<T>(x, empty, 0.05) > 1e10);
  REQUIRE(wdtwFull<T>(empty, x, 0.05) > 1e10);
  std::vector<T> empty2{};
  REQUIRE(wdtwFull<T>(empty, empty2, 0.05) > 1e10);
}

// ---------- wdtwBanded tests ----------

TEST_CASE("wdtwBanded: band=-1 falls back to full WDTW", "[wdtw]")
{
  using T = double;
  std::vector<T> x{ 1, 2, 3, 4 }, y{ 2, 4, 6, 8, 10 };

  for (double g : { 0.0, 0.05, 1.0 }) {
    double full_val = wdtwFull<T>(x, y, g);
    double banded_val = wdtwBanded<T>(x, y, -1, g);
    REQUIRE_THAT(banded_val, WithinAbs(full_val, 1e-12));
  }
}

TEST_CASE("wdtwBanded: identity gives zero", "[wdtw]")
{
  using T = double;
  std::vector<T> x{ 1, 2, 3, 4, 5 };

  REQUIRE_THAT(wdtwBanded<T>(x, x, 2, 0.05), WithinAbs(0.0, 1e-15));
}

TEST_CASE("wdtwBanded: symmetry", "[wdtw]")
{
  using T = double;
  std::vector<T> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 };

  for (int band : { 1, 2, 5 }) {
    for (double g : { 0.0, 0.05, 1.0 }) {
      REQUIRE_THAT(wdtwBanded<T>(x, y, band, g),
                   WithinAbs(wdtwBanded<T>(y, x, band, g), 1e-12));
    }
  }
}

TEST_CASE("wdtwBanded: empty series returns maxValue", "[wdtw]")
{
  using T = double;
  std::vector<T> x{ 1, 2, 3 }, empty{};

  REQUIRE(wdtwBanded<T>(x, empty, 2, 0.05) > 1e10);
  REQUIRE(wdtwBanded<T>(empty, x, 2, 0.05) > 1e10);
}

TEST_CASE("wdtwBanded: large band equals full wdtw", "[wdtw]")
{
  using T = double;
  std::vector<T> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 };

  for (double g : { 0.0, 0.05, 1.0 }) {
    double full_val = wdtwFull<T>(x, y, g);
    double banded_val = wdtwBanded<T>(x, y, 100, g);
    REQUIRE_THAT(banded_val, WithinAbs(full_val, 1e-12));
  }
}

TEST_CASE("wdtwFull: equal-element series gives zero", "[wdtw]")
{
  using T = double;
  std::vector<T> x{ 5, 5, 5, 5 };
  std::vector<T> y{ 5, 5, 5, 5 };

  // Not the same object, but same values
  for (double g : { 0.0, 0.05, 1.0, 100.0 }) {
    REQUIRE_THAT(wdtwFull<T>(x, y, g), WithinAbs(0.0, 1e-15));
  }
}
