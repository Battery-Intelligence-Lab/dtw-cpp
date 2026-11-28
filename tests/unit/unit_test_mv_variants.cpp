/**
 * @file unit_test_mv_variants.cpp
 * @brief Unit tests for multivariate WDTW and ADTW wrapper overloads.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <limits>
#include <vector>

using Catch::Matchers::WithinAbs;

TEST_CASE("WDTW MV: ndim=1 matches scalar wdtwBanded with weights", "[mv][wdtw]")
{
  std::vector<double> x = {1, 3, 4, 2, 5};
  std::vector<double> y = {2, 4, 3, 5, 1};
  const int max_dev = static_cast<int>(std::max(x.size(), y.size())) - 1;
  auto w = dtwc::wdtw_weights<double>(max_dev, 0.05);
  double d_scalar = dtwc::wdtwBanded(x, y, w);
  double d_mv = dtwc::wdtwBanded_mv(x.data(), 5, y.data(), 5, 1, -1, 0.05);
  REQUIRE_THAT(d_mv, WithinAbs(d_scalar, 1e-10));
}

TEST_CASE("WDTW MV: ndim=2 symmetry", "[mv][wdtw]")
{
  double x[] = {1, 2, 3, 4, 5, 6};
  double y[] = {6, 5, 4, 3, 2, 1};
  double d1 = dtwc::wdtwBanded_mv(x, 3, y, 3, 2);
  double d2 = dtwc::wdtwBanded_mv(y, 3, x, 3, 2);
  REQUIRE_THAT(d1, WithinAbs(d2, 1e-10));
}

TEST_CASE("WDTW MV: empty series returns max", "[mv][wdtw]")
{
  double x[] = {1, 2, 3, 4};
  constexpr double maxVal = std::numeric_limits<double>::max();
  REQUIRE(dtwc::wdtwBanded_mv(x, 2, static_cast<double *>(nullptr), 0, 2) == maxVal);
  REQUIRE(dtwc::wdtwBanded_mv(static_cast<double *>(nullptr), 0, x, 2, 2) == maxVal);
}

TEST_CASE("WDTW MV banded: ndim=1 matches scalar wdtwBanded", "[mv][wdtw]")
{
  std::vector<double> x = {1, 3, 4, 2, 5};
  std::vector<double> y = {2, 4, 3, 5, 1};
  double d_scalar = dtwc::wdtwBanded(x, y, 2, 0.05);
  double d_mv = dtwc::wdtwBanded_mv(x.data(), 5, y.data(), 5, 1, 2, 0.05);
  REQUIRE_THAT(d_mv, WithinAbs(d_scalar, 1e-10));
}

TEST_CASE("ADTW MV: ndim=1 matches scalar adtwBanded, band -1", "[mv][adtw]")
{
  std::vector<double> x = {1, 3, 4, 2, 5};
  std::vector<double> y = {2, 4, 3, 5, 1};
  double d_scalar = dtwc::adtwBanded(x, y, -1, 1.0);
  double d_mv = dtwc::adtwBanded_mv(x.data(), 5, y.data(), 5, 1, -1, 1.0);
  REQUIRE_THAT(d_mv, WithinAbs(d_scalar, 1e-10));
}

TEST_CASE("ADTW MV: penalty increases distance for unequal-length series", "[mv][adtw]")
{
  // Unequal lengths force non-diagonal steps, so penalty has a direct effect
  double x[] = {0, 0, 1, 1, 2, 2};  // 3 timesteps
  double y[] = {0, 0, 2, 2};         // 2 timesteps
  double d_low = dtwc::adtwBanded_mv(x, 3, y, 2, 2, -1, 0.0);
  double d_high = dtwc::adtwBanded_mv(x, 3, y, 2, 2, -1, 10.0);
  REQUIRE(d_high > d_low);
}

TEST_CASE("ADTW MV banded: ndim=1 matches scalar adtwBanded", "[mv][adtw]")
{
  std::vector<double> x = {1, 3, 4, 2, 5};
  std::vector<double> y = {2, 4, 3, 5, 1};
  double d_scalar = dtwc::adtwBanded(x, y, 2, 1.0);
  double d_mv = dtwc::adtwBanded_mv(x.data(), 5, y.data(), 5, 1, 2, 1.0);
  REQUIRE_THAT(d_mv, WithinAbs(d_scalar, 1e-10));
}

// Banded MV is a first-class path: adtwBanded_mv and wdtwBanded_mv used to fall
// back to the unbanded MV variant. Band restrictions are honoured and must give a
// different result from the unbanded MV when the band constrains the warping path.
TEST_CASE("ADTW MV banded actually restricts the warping path", "[mv][adtw][banded]")
{
  // Shifted-peak pattern: x has a peak at step 2, y has the same peak at step 8.
  // Unbanded DTW warps heavily to align the peaks; band=1 forbids that warp.
  std::vector<double> x = {0,0, 0,0, 9,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0};
  std::vector<double> y = {0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 9,0, 0,0};
  const double penalty = 0.5;

  const double d_unbanded = dtwc::adtwBanded_mv<double>(x.data(), 10, y.data(), 10, 2, -1, penalty);
  const double d_banded1  = dtwc::adtwBanded_mv<double>(x.data(), 10, y.data(), 10, 2, 1, penalty);

  INFO("d_unbanded=" << d_unbanded << " d_banded1=" << d_banded1);
  REQUIRE(d_banded1 > d_unbanded);
}

TEST_CASE("WDTW MV banded actually restricts the warping path", "[mv][wdtw][banded]")
{
  std::vector<double> x = {0,0, 0,0, 9,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0};
  std::vector<double> y = {0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 9,0, 0,0};
  const double g = 0.05;

  const double d_unbanded = dtwc::wdtwBanded_mv<double>(x.data(), 10, y.data(), 10, 2, -1, g);
  const double d_banded1  = dtwc::wdtwBanded_mv<double>(x.data(), 10, y.data(), 10, 2, 1, g);

  INFO("d_unbanded=" << d_unbanded << " d_banded1=" << d_banded1);
  REQUIRE(d_banded1 > d_unbanded);
}
