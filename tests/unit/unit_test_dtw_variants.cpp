/**
 * @file unit_test_dtw_variants.cpp
 * @brief Unit tests comparing DTW variants (full, light, banded).
 *
 * Tests identity, symmetry, banded-vs-full relationships,
 * diagonal-only band, different-length series, and empty series.
 *
 * @date 28 Mar 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc;


// ---------------------------------------------------------------------------
// 1. Identity: dtw(x, x) == 0 for all three variants
// ---------------------------------------------------------------------------
TEST_CASE("DTW of identical series is zero", "[Phase1][dtw_variants]")
{
  using dt = double;
  std::vector<dt> x{ 1, 3, 5, 2, 4 };

  SECTION("dtwFull")
  {
    REQUIRE_THAT(dtwFull<dt>(x, x), WithinAbs(0.0, 1e-15));
  }
  SECTION("dtwFull_L")
  {
    REQUIRE_THAT(dtwFull_L<dt>(x, x), WithinAbs(0.0, 1e-15));
  }
  SECTION("dtwBanded with default band (-1, i.e. full)")
  {
    REQUIRE_THAT(dtwBanded<dt>(x, x), WithinAbs(0.0, 1e-15));
  }
  SECTION("dtwBanded with band=2")
  {
    REQUIRE_THAT(dtwBanded<dt>(x, x, 2), WithinAbs(0.0, 1e-15));
  }
}

// ---------------------------------------------------------------------------
// 2. Symmetry: dtw(x, y) == dtw(y, x)
// ---------------------------------------------------------------------------
TEST_CASE("DTW is symmetric", "[Phase1][dtw_variants]")
{
  using dt = double;
  std::vector<dt> x{ 1, 2, 3, 4, 5 };
  std::vector<dt> y{ 2, 4, 6, 8, 10 };

  SECTION("dtwFull")
  {
    REQUIRE_THAT(dtwFull<dt>(x, y), WithinAbs(dtwFull<dt>(y, x), 1e-15));
  }
  SECTION("dtwFull_L")
  {
    REQUIRE_THAT(dtwFull_L<dt>(x, y), WithinAbs(dtwFull_L<dt>(y, x), 1e-15));
  }
  SECTION("dtwBanded (full)")
  {
    REQUIRE_THAT(dtwBanded<dt>(x, y), WithinAbs(dtwBanded<dt>(y, x), 1e-15));
  }
  SECTION("dtwBanded (band=1)")
  {
    REQUIRE_THAT(dtwBanded<dt>(x, y, 1), WithinAbs(dtwBanded<dt>(y, x, 1), 1e-15));
  }
}

// ---------------------------------------------------------------------------
// 3. Banded DTW >= Full DTW (band constrains the search, so it is an upper
//    bound or equal).
// ---------------------------------------------------------------------------
TEST_CASE("Banded DTW >= Full DTW", "[Phase1][dtw_variants]")
{
  using dt = double;
  std::vector<dt> x{ 1, 2, 3, 4, 5, 6, 7, 8 };
  std::vector<dt> y{ 8, 7, 6, 5, 4, 3, 2, 1 };

  double full_dist = dtwFull<dt>(x, y);

  for (int band : { 1, 2, 3, 4 }) {
    double banded_dist = dtwBanded<dt>(x, y, band);
    // Banded distance must be >= full distance (or equal within tolerance).
    REQUIRE(banded_dist >= full_dist - 1e-10);
  }
}

// ---------------------------------------------------------------------------
// 4. Very wide band gives same result as full DTW
// ---------------------------------------------------------------------------
TEST_CASE("Wide band equals full DTW", "[Phase1][dtw_variants]")
{
  using dt = double;
  std::vector<dt> x{ 1, 3, 5, 7, 9, 11, 13 };
  std::vector<dt> y{ 2, 4, 6, 8, 10 };

  double full_dist = dtwFull<dt>(x, y);
  double wide_band_dist = dtwBanded<dt>(x, y, 100);

  REQUIRE_THAT(wide_band_dist, WithinAbs(full_dist, 1e-12));
}

// ---------------------------------------------------------------------------
// 5. Band=0 for equal-length series should give the Euclidean (L1) alignment
//    along the diagonal only: sum of |x[i]-y[i]|
// ---------------------------------------------------------------------------
TEST_CASE("Band=0 for equal-length series gives diagonal-only L1 cost", "[Phase1][dtw_variants]")
{
  using dt = double;
  std::vector<dt> x{ 1, 2, 3, 4, 5 };
  std::vector<dt> y{ 5, 4, 3, 2, 1 };

  // Diagonal-only cost: sum |x[i]-y[i]| = 4+2+0+2+4 = 12
  double diagonal_cost = 0.0;
  for (size_t i = 0; i < x.size(); ++i)
    diagonal_cost += std::abs(x[i] - y[i]);

  double banded_dist = dtwBanded<dt>(x, y, 0);
  REQUIRE_THAT(banded_dist, WithinAbs(diagonal_cost, 1e-12));
}

// ---------------------------------------------------------------------------
// 6. dtwFull and dtwFull_L give the same result
// ---------------------------------------------------------------------------
TEST_CASE("dtwFull and dtwFull_L agree", "[Phase1][dtw_variants]")
{
  using dt = double;

  SECTION("Equal-length series")
  {
    std::vector<dt> x{ 1, 3, 5, 2, 4 };
    std::vector<dt> y{ 2, 4, 6, 3, 5 };
    REQUIRE_THAT(dtwFull<dt>(x, y), WithinAbs(dtwFull_L<dt>(x, y), 1e-12));
  }

  SECTION("Different-length series")
  {
    std::vector<dt> x{ 1, 2, 3 };
    std::vector<dt> y{ 3, 4, 5, 6, 7 };
    REQUIRE_THAT(dtwFull<dt>(x, y), WithinAbs(dtwFull_L<dt>(x, y), 1e-12));
  }
}

// ---------------------------------------------------------------------------
// 7. Different-length series: result is finite and positive
// ---------------------------------------------------------------------------
TEST_CASE("DTW of different-length series is finite and positive", "[Phase1][dtw_variants]")
{
  using dt = double;
  std::vector<dt> short_s{ 1, 2, 3 };
  std::vector<dt> long_s{ 10, 20, 30, 40, 50 };

  double d_full = dtwFull<dt>(short_s, long_s);
  double d_L = dtwFull_L<dt>(short_s, long_s);
  double d_banded = dtwBanded<dt>(short_s, long_s);

  REQUIRE(std::isfinite(d_full));
  REQUIRE(d_full > 0.0);
  REQUIRE(std::isfinite(d_L));
  REQUIRE(d_L > 0.0);
  REQUIRE(std::isfinite(d_banded));
  REQUIRE(d_banded > 0.0);
}

// ---------------------------------------------------------------------------
// 8. Empty series: returns max/infinity
// ---------------------------------------------------------------------------
TEST_CASE("DTW with empty series returns very large value", "[Phase1][dtw_variants]")
{
  using dt = double;
  std::vector<dt> x{ 1, 2, 3 };
  std::vector<dt> empty;

  // All variants should return a very large value for empty input.
  REQUIRE(dtwFull<dt>(x, empty) > 1e10);
  REQUIRE(dtwFull<dt>(empty, x) > 1e10);
  REQUIRE(dtwFull_L<dt>(x, empty) > 1e10);
  REQUIRE(dtwFull_L<dt>(empty, x) > 1e10);
  REQUIRE(dtwBanded<dt>(x, empty) > 1e10);
  REQUIRE(dtwBanded<dt>(empty, x) > 1e10);
}

// ---------------------------------------------------------------------------
// 9. Single-element series
// ---------------------------------------------------------------------------
TEST_CASE("DTW of single-element series", "[Phase1][dtw_variants]")
{
  using dt = double;
  std::vector<dt> a{ 5.0 };
  std::vector<dt> b{ 8.0 };

  // Distance between single-element series is |5 - 8| = 3.
  REQUIRE_THAT(dtwFull<dt>(a, b), WithinAbs(3.0, 1e-15));
  REQUIRE_THAT(dtwFull_L<dt>(a, b), WithinAbs(3.0, 1e-15));
  REQUIRE_THAT(dtwBanded<dt>(a, b), WithinAbs(3.0, 1e-15));
}

// ---------------------------------------------------------------------------
// 10. Known hand-computed example
//     x = {1, 2, 3}, y = {2, 2, 2, 3, 4}
//     Full cost matrix (L1 distance):
//     C(0,0)=1  C(0,1)=2  C(0,2)=3  C(0,3)=5  C(0,4)=8
//     C(1,0)=1  C(1,1)=1  C(1,2)=1  C(1,3)=2  C(1,4)=4
//     C(2,0)=2  C(2,1)=2  C(2,2)=2  C(2,3)=1  C(2,4)=2
//     Answer: C(2,4) = 2
// ---------------------------------------------------------------------------
TEST_CASE("DTW known hand-computed example", "[Phase1][dtw_variants]")
{
  using dt = double;
  std::vector<dt> x{ 1, 2, 3 };
  std::vector<dt> y{ 2, 2, 2, 3, 4 };

  REQUIRE_THAT(dtwFull<dt>(x, y), WithinAbs(2.0, 1e-15));
  REQUIRE_THAT(dtwFull_L<dt>(x, y), WithinAbs(2.0, 1e-15));
  // With default band (-1), banded should equal full.
  REQUIRE_THAT(dtwBanded<dt>(x, y), WithinAbs(2.0, 1e-15));
}

// ---------------------------------------------------------------------------
// 11. Monotone shift: shifting y by constant c increases DTW by at most N*c
// ---------------------------------------------------------------------------
TEST_CASE("Shifting series by constant increases DTW predictably", "[Phase1][dtw_variants]")
{
  using dt = double;
  std::vector<dt> x{ 1, 2, 3, 4, 5 };
  std::vector<dt> y = x; // identical first

  REQUIRE_THAT(dtwFull<dt>(x, y), WithinAbs(0.0, 1e-15));

  // Shift y by +10.
  for (auto &v : y) v += 10.0;
  double dist = dtwFull<dt>(x, y);

  // The DTW cost for equal-length identical-shape series shifted by c
  // is exactly N * c (the optimal alignment is the diagonal).
  double expected = x.size() * 10.0;
  REQUIRE_THAT(dist, WithinAbs(expected, 1e-12));
}

// ---------------------------------------------------------------------------
// 12. Increasing band width monotonically decreases (or maintains) DTW cost
// ---------------------------------------------------------------------------
TEST_CASE("Increasing band width monotonically decreases DTW cost", "[Phase1][dtw_variants]")
{
  using dt = double;
  // Series that require time-shifting for a good alignment.
  std::vector<dt> x{ 0, 0, 1, 2, 1, 0, 0, 0, 0, 0 };
  std::vector<dt> y{ 0, 0, 0, 0, 0, 0, 1, 2, 1, 0 };

  double prev_dist = std::numeric_limits<double>::max();
  for (int band = 0; band <= 10; ++band) {
    double dist = dtwBanded<dt>(x, y, band);
    REQUIRE(dist <= prev_dist + 1e-10);
    prev_dist = dist;
  }

  // The widest band should equal full DTW.
  double full_dist = dtwFull<dt>(x, y);
  REQUIRE_THAT(prev_dist, WithinAbs(full_dist, 1e-12));
}
