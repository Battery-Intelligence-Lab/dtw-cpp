/**
 * @file unit_test_missing_dtw.cpp
 * @brief Unit tests for DTW with missing data (NaN-aware, DTW-AROW).
 *
 * @details Tests for dtwMissing, dtwMissing_L, and dtwMissing_banded functions
 * which handle NaN values in time series by treating missing pairs as zero cost.
 *
 * Reference: Yurtman, Soenen, Meert & Blockeel (2023), "Estimating DTW Distance
 *            Between Time Series with Missing Data", ECML-PKDD 2023, LNCS 14173.
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>   // for NAN, std::nan
#include <limits>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

// ===========================================================================
// dtwMissing_L — no missing data (should match standard DTW)
// ===========================================================================

TEST_CASE("dtwMissing_L: no NaN matches standard DTW", "[missing_dtw]")
{
  std::vector<double> x{ 1, 2, 3, 4, 5 };
  std::vector<double> y{ 2, 3, 4 };

  const auto missing_result = dtwMissing_L<double>(x, y);
  const auto standard_result = dtwFull_L<double>(x, y);

  REQUIRE_THAT(missing_result, WithinAbs(standard_result, 1e-12));
}

TEST_CASE("dtwMissing_L: no NaN matches standard DTW (equal length)", "[missing_dtw]")
{
  std::vector<double> x{ 1, 3, 5 };
  std::vector<double> y{ 2, 4, 6 };

  const auto missing_result = dtwMissing_L<double>(x, y);
  const auto standard_result = dtwFull_L<double>(x, y);

  REQUIRE_THAT(missing_result, WithinAbs(standard_result, 1e-12));
}

TEST_CASE("dtwMissing_L: no NaN matches standard DTW (SquaredL2)", "[missing_dtw]")
{
  std::vector<double> x{ 1, 2, 3 };
  std::vector<double> y{ 2, 4, 5 };

  const auto missing_result = dtwMissing_L<double>(x, y, -1, core::MetricType::SquaredL2);
  const auto standard_result = dtwFull_L<double>(x, y, -1, core::MetricType::SquaredL2);

  REQUIRE_THAT(missing_result, WithinAbs(standard_result, 1e-12));
}

// ===========================================================================
// dtwMissing_L — all missing data
// ===========================================================================

TEST_CASE("dtwMissing_L: all NaN in both series gives zero", "[missing_dtw]")
{
  std::vector<double> x{ NaN, NaN, NaN };
  std::vector<double> y{ NaN, NaN };

  REQUIRE_THAT(dtwMissing_L<double>(x, y), WithinAbs(0.0, 1e-15));
}

TEST_CASE("dtwMissing_L: one series entirely NaN gives zero", "[missing_dtw]")
{
  std::vector<double> x{ 1, 2, 3, 4, 5 };
  std::vector<double> y_nan{ NaN, NaN, NaN };

  REQUIRE_THAT(dtwMissing_L<double>(x, y_nan), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(dtwMissing_L<double>(y_nan, x), WithinAbs(0.0, 1e-15));
}

// ===========================================================================
// dtwMissing_L — partial missing data
// ===========================================================================

TEST_CASE("dtwMissing_L: single missing at start of x", "[missing_dtw]")
{
  // x = {NaN, 2, 3}, y = {1, 2, 3}
  // cost(NaN, 1) = 0
  // Remaining path through present values should give less than or equal to
  // standard DTW on the original
  std::vector<double> x{ NaN, 2, 3 };
  std::vector<double> y{ 1, 2, 3 };

  const auto result = dtwMissing_L<double>(x, y);
  REQUIRE(result >= 0.0);
  // With NaN at x[0], cost(x[0], y[j]) = 0 for any j.
  // Hand-compute: C(0,0)=0, C(1,0)=0+|2-1|=1, C(2,0)=1+|3-1|=3
  // C(0,1)=0+0=0, C(0,2)=0+0=0
  // C(1,1)=min(0,1,0)+|2-2|=0, C(1,2)=min(0,0,0)+|2-3|=1
  // C(2,1)=min(0,1,0)+|3-2|=1, C(2,2)=min(0,1,1)+|3-3|=0
  REQUIRE_THAT(result, WithinAbs(0.0, 1e-12));
}

TEST_CASE("dtwMissing_L: single missing in the middle", "[missing_dtw]")
{
  // x = {1, NaN, 3}, y = {1, 2, 3}
  std::vector<double> x{ 1, NaN, 3 };
  std::vector<double> y{ 1, 2, 3 };

  const auto result = dtwMissing_L<double>(x, y);
  REQUIRE(result >= 0.0);
  // Hand-compute:
  // C(0,0)=|1-1|=0, C(1,0)=0+0=0, C(2,0)=0+|3-1|=2
  // C(0,1)=0+|1-2|=1, C(0,2)=1+|1-3|=3
  // C(1,1)=min(0,0,1)+0=0, C(1,2)=min(1,0,0)+0=0
  // C(2,1)=min(0,0,0)+|3-2|=1, C(2,2)=min(0,0,1)+|3-3|=0
  REQUIRE_THAT(result, WithinAbs(0.0, 1e-12));
}

TEST_CASE("dtwMissing_L: single missing at end", "[missing_dtw]")
{
  // x = {1, 2, NaN}, y = {1, 2, 3}
  std::vector<double> x{ 1, 2, NaN };
  std::vector<double> y{ 1, 2, 3 };

  const auto result = dtwMissing_L<double>(x, y);
  REQUIRE(result >= 0.0);
  // C(0,0)=0, C(1,0)=0+|2-1|=1, C(2,0)=1+0=1
  // C(0,1)=0+|1-2|=1, C(0,2)=1+|1-3|=3
  // C(1,1)=min(0,1,1)+|2-2|=0, C(1,2)=min(1,0,0)+|2-3|=1
  // C(2,1)=min(0,1,0)+0=0, C(2,2)=min(0,0,1)+0=0
  REQUIRE_THAT(result, WithinAbs(0.0, 1e-12));
}

TEST_CASE("dtwMissing_L: both series have NaN at different positions", "[missing_dtw]")
{
  // x = {1, NaN, 3, 4}, y = {NaN, 2, 3, 4}
  std::vector<double> x{ 1, NaN, 3, 4 };
  std::vector<double> y{ NaN, 2, 3, 4 };

  const auto result = dtwMissing_L<double>(x, y);
  REQUIRE(result >= 0.0);
  // Symmetry holds since cost function is symmetric
  const auto reverse = dtwMissing_L<double>(y, x);
  REQUIRE_THAT(result, WithinAbs(reverse, 1e-12));
}

// ===========================================================================
// dtwMissing_L — symmetry
// ===========================================================================

TEST_CASE("dtwMissing_L: symmetry with no NaN", "[missing_dtw]")
{
  std::vector<double> x{ 1, 3, 5, 2 };
  std::vector<double> y{ 2, 4, 6 };

  REQUIRE_THAT(dtwMissing_L<double>(x, y), WithinAbs(dtwMissing_L<double>(y, x), 1e-12));
}

TEST_CASE("dtwMissing_L: symmetry with NaN", "[missing_dtw]")
{
  std::vector<double> x{ 1, NaN, 5, 2 };
  std::vector<double> y{ NaN, 4, 6 };

  REQUIRE_THAT(dtwMissing_L<double>(x, y), WithinAbs(dtwMissing_L<double>(y, x), 1e-12));
}

// ===========================================================================
// dtwMissing_L — edge cases
// ===========================================================================

TEST_CASE("dtwMissing_L: empty vectors return maxValue", "[missing_dtw]")
{
  std::vector<double> x{ 1, 2, 3 };
  std::vector<double> empty{};

  REQUIRE(dtwMissing_L<double>(x, empty) > 1e10);
  REQUIRE(dtwMissing_L<double>(empty, x) > 1e10);
}

TEST_CASE("dtwMissing_L: identical series gives zero", "[missing_dtw]")
{
  std::vector<double> x{ 1, 2, 3, 4, 5 };
  REQUIRE_THAT(dtwMissing_L<double>(x, x), WithinAbs(0.0, 1e-15));
}

TEST_CASE("dtwMissing_L: single-element series", "[missing_dtw]")
{
  std::vector<double> x{ 5.0 };
  std::vector<double> y{ 3.0 };
  std::vector<double> y_nan{ NaN };

  REQUIRE_THAT(dtwMissing_L<double>(x, y), WithinAbs(2.0, 1e-12));
  REQUIRE_THAT(dtwMissing_L<double>(x, y_nan), WithinAbs(0.0, 1e-15));
}

TEST_CASE("dtwMissing_L: non-negativity", "[missing_dtw]")
{
  std::vector<double> x{ 1, NaN, 3 };
  std::vector<double> y{ 5, 6, NaN, 8 };

  REQUIRE(dtwMissing_L<double>(x, y) >= 0.0);
}

// ===========================================================================
// dtwMissing_L — early abandon
// ===========================================================================

TEST_CASE("dtwMissing_L: early abandon triggers correctly", "[missing_dtw]")
{
  std::vector<double> x{ 1, 2, 3, 4, 5 };
  std::vector<double> y{ 100, 200, 300, 400, 500 };
  constexpr double maxValue = std::numeric_limits<double>::max();

  // With a small threshold, early abandon should kick in
  const auto result = dtwMissing_L<double>(x, y, 10.0);
  REQUIRE(result == maxValue);

  // With no threshold, should return the actual distance
  const auto full = dtwMissing_L<double>(x, y, -1.0);
  REQUIRE(full < maxValue);
}

// ===========================================================================
// dtwMissing — full matrix version
// ===========================================================================

TEST_CASE("dtwMissing: matches dtwMissing_L for no NaN", "[missing_dtw]")
{
  std::vector<double> x{ 1, 2, 3, 4 };
  std::vector<double> y{ 2, 4, 5 };

  const auto full = dtwMissing<double>(x, y);
  const auto light = dtwMissing_L<double>(x, y);

  REQUIRE_THAT(full, WithinAbs(light, 1e-12));
}

TEST_CASE("dtwMissing: matches dtwMissing_L with NaN", "[missing_dtw]")
{
  std::vector<double> x{ 1, NaN, 3, 4 };
  std::vector<double> y{ NaN, 2, 3 };

  const auto full = dtwMissing<double>(x, y);
  const auto light = dtwMissing_L<double>(x, y);

  REQUIRE_THAT(full, WithinAbs(light, 1e-12));
}

TEST_CASE("dtwMissing: all NaN gives zero", "[missing_dtw]")
{
  std::vector<double> x{ NaN, NaN };
  std::vector<double> y{ NaN, NaN, NaN };

  REQUIRE_THAT(dtwMissing<double>(x, y), WithinAbs(0.0, 1e-15));
}

// ===========================================================================
// dtwMissing_banded — banded version
// ===========================================================================

TEST_CASE("dtwMissing_banded: negative band falls back to dtwMissing_L", "[missing_dtw]")
{
  std::vector<double> x{ 1, NaN, 3 };
  std::vector<double> y{ 2, 4, NaN };

  const auto full = dtwMissing_L<double>(x, y);
  const auto banded = dtwMissing_banded<double>(x, y, -1);

  REQUIRE_THAT(banded, WithinAbs(full, 1e-12));
}

TEST_CASE("dtwMissing_banded: large band matches unbanded", "[missing_dtw]")
{
  std::vector<double> x{ 1, NaN, 3, 4, 5 };
  std::vector<double> y{ 2, 4, NaN, 6, 7, 8 };

  const auto full = dtwMissing_L<double>(x, y);
  const auto banded = dtwMissing_banded<double>(x, y, 100);

  REQUIRE_THAT(banded, WithinAbs(full, 1e-12));
}

TEST_CASE("dtwMissing_banded: no NaN matches standard dtwBanded", "[missing_dtw]")
{
  std::vector<double> x{ 1, 2, 3, 4, 5 };
  std::vector<double> y{ 2, 3, 4, 5, 6, 7, 8 };
  int band = 2;

  const auto missing_result = dtwMissing_banded<double>(x, y, band);
  const auto standard_result = dtwBanded<double>(x, y, band);

  REQUIRE_THAT(missing_result, WithinAbs(standard_result, 1e-12));
}

TEST_CASE("dtwMissing_banded: symmetry", "[missing_dtw]")
{
  std::vector<double> x{ 1, NaN, 3, 4, 5 };
  std::vector<double> y{ 2, 4, NaN, 6, 7 };
  int band = 2;

  REQUIRE_THAT(dtwMissing_banded<double>(x, y, band),
               WithinAbs(dtwMissing_banded<double>(y, x, band), 1e-12));
}

TEST_CASE("dtwMissing_banded: all NaN gives zero", "[missing_dtw]")
{
  std::vector<double> x{ NaN, NaN, NaN, NaN, NaN };
  std::vector<double> y{ NaN, NaN, NaN, NaN, NaN };

  REQUIRE_THAT(dtwMissing_banded<double>(x, y, 2), WithinAbs(0.0, 1e-15));
}

TEST_CASE("dtwMissing_banded: empty vectors", "[missing_dtw]")
{
  std::vector<double> x{ 1, 2, 3 };
  std::vector<double> empty{};

  REQUIRE(dtwMissing_banded<double>(x, empty, 2) > 1e10);
  REQUIRE(dtwMissing_banded<double>(empty, x, 2) > 1e10);
}

TEST_CASE("dtwMissing_banded: single element", "[missing_dtw]")
{
  std::vector<double> x{ 5.0 };
  std::vector<double> y{ 1, 2, 3, 4, 5, 6, 7, 8 };

  // Single-element short side falls back to dtwMissing_L
  const auto banded = dtwMissing_banded<double>(x, y, 2);
  const auto full = dtwMissing_L<double>(x, y);

  REQUIRE_THAT(banded, WithinAbs(full, 1e-12));
}

// ===========================================================================
// dtwMissing_L — hand-computed values with specific missing patterns
// ===========================================================================

TEST_CASE("dtwMissing_L: hand-computed with NaN in y", "[missing_dtw]")
{
  // x = {1, 2, 3}, y = {1, NaN, 3}
  // C(0,0) = |1-1| = 0
  // C(1,0) = 0 + |2-1| = 1
  // C(2,0) = 1 + |3-1| = 3
  // C(0,1) = 0 + 0 = 0   (y[1] is NaN)
  // C(0,2) = 0 + |1-3| = 2
  // C(1,1) = min(0, 1, 0) + 0 = 0   (y[1] is NaN)
  // C(1,2) = min(0, 0, 0) + |2-3| = 1
  // C(2,1) = min(1, 0, 0) + 0 = 0   (y[1] is NaN)
  // C(2,2) = min(0, 0, 1) + |3-3| = 0
  std::vector<double> x{ 1, 2, 3 };
  std::vector<double> y{ 1, NaN, 3 };

  REQUIRE_THAT(dtwMissing_L<double>(x, y), WithinAbs(0.0, 1e-12));
}

TEST_CASE("dtwMissing_L: hand-computed with multiple NaN", "[missing_dtw]")
{
  // x = {10, NaN}, y = {NaN, 10}
  // C(0,0) = 0 (x[0]=10, y[0]=NaN -> 0)
  // C(1,0) = 0 + 0 = 0 (x[1]=NaN)
  // C(0,1) = 0 + |10-10| = 0
  // C(1,1) = min(0, 0, 0) + 0 = 0 (x[1]=NaN)
  std::vector<double> x{ 10, NaN };
  std::vector<double> y{ NaN, 10 };

  REQUIRE_THAT(dtwMissing_L<double>(x, y), WithinAbs(0.0, 1e-12));
}

TEST_CASE("dtwMissing_L: hand-computed nonzero result with NaN", "[missing_dtw]")
{
  // x = {1, 2}, y = {NaN, 5}
  // C(0,0) = 0 (y[0]=NaN)
  // C(1,0) = 0 + 0 = 0 (y[0]=NaN)
  // C(0,1) = 0 + |1-5| = 4
  // C(1,1) = min(0, 0, 4) + |2-5| = 0 + 3 = 3
  std::vector<double> x{ 1, 2 };
  std::vector<double> y{ NaN, 5 };

  REQUIRE_THAT(dtwMissing_L<double>(x, y), WithinAbs(3.0, 1e-12));
}

// ===========================================================================
// dtwMissing_L — missing less than or equal to standard DTW
// ===========================================================================

TEST_CASE("dtwMissing_L: adding NaN does not increase distance", "[missing_dtw]")
{
  // Replace a value with NaN: the distance should decrease or stay the same
  // because NaN pairs cost 0 instead of the actual distance.
  std::vector<double> x_full{ 1, 5, 3 };
  std::vector<double> x_nan{ 1, NaN, 3 };
  std::vector<double> y{ 1, 2, 3 };

  const auto dist_full = dtwMissing_L<double>(x_full, y);
  const auto dist_nan = dtwMissing_L<double>(x_nan, y);

  // The NaN version should be <= the full version because we zero out a
  // cost that was previously > 0.
  REQUIRE(dist_nan <= dist_full + 1e-12);
}

// ===========================================================================
// dtwMissing_banded with SquaredL2 metric
// ===========================================================================

TEST_CASE("dtwMissing_banded: SquaredL2 no NaN matches standard", "[missing_dtw]")
{
  std::vector<double> x{ 1, 2, 3, 4, 5 };
  std::vector<double> y{ 2, 3, 4, 5, 6, 7, 8 };
  int band = 3;
  auto metric = core::MetricType::SquaredL2;

  const auto missing_result = dtwMissing_banded<double>(x, y, band, -1, metric);
  const auto standard_result = dtwBanded<double>(x, y, band, -1, metric);

  REQUIRE_THAT(missing_result, WithinAbs(standard_result, 1e-12));
}
