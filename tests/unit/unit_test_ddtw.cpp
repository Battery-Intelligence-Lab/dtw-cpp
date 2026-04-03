/**
 * @file unit_test_ddtw.cpp
 * @brief Unit tests for Derivative DTW (DDTW) functions
 *
 * @details Tests the derivative transform and DDTW distance functions.
 *          DDTW preprocesses series with a derivative transform, then runs
 *          standard DTW on the derivative series, capturing shape rather
 *          than amplitude.
 *
 * Reference: E. J. Keogh and M. J. Pazzani, "Derivative Dynamic Time Warping,"
 *            SIAM International Conference on Data Mining, 2001.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include <dtwc.hpp>
#include <warping_ddtw.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

// ---------------------------------------------------------------------------
// derivative_transform tests
// ---------------------------------------------------------------------------

TEST_CASE("derivative_transform of known sequence", "[ddtw][derivative_transform]")
{
  // x = {1, 3, 5, 2, 4}
  // x'[0] = x[1] - x[0] = 2
  // x'[1] = ((x[1]-x[0]) + (x[2]-x[0])/2) / 2 = (2 + 2) / 2 = 2
  // x'[2] = ((x[2]-x[1]) + (x[3]-x[1])/2) / 2 = (2 + (-0.5)) / 2 = 0.75
  // x'[3] = ((x[3]-x[2]) + (x[4]-x[2])/2) / 2 = (-3 + (-0.5)) / 2 = -1.75
  // x'[4] = x[4] - x[3] = 2
  std::vector<double> x{ 1, 3, 5, 2, 4 };
  auto dx = derivative_transform(x);

  REQUIRE(dx.size() == 5);
  REQUIRE_THAT(dx[0], WithinAbs(2.0, 1e-12));
  REQUIRE_THAT(dx[1], WithinAbs(2.0, 1e-12));
  REQUIRE_THAT(dx[2], WithinAbs(0.75, 1e-12));
  REQUIRE_THAT(dx[3], WithinAbs(-1.75, 1e-12));
  REQUIRE_THAT(dx[4], WithinAbs(2.0, 1e-12));
}

TEST_CASE("derivative_transform of length-1 series", "[ddtw][derivative_transform]")
{
  std::vector<double> x{ 42.0 };
  auto dx = derivative_transform(x);

  REQUIRE(dx.size() == 1);
  REQUIRE_THAT(dx[0], WithinAbs(0.0, 1e-15));
}

TEST_CASE("derivative_transform of length-2 series", "[ddtw][derivative_transform]")
{
  // x'[0] = x[1] - x[0] = 7 - 3 = 4
  // x'[1] = x[1] - x[0] = 7 - 3 = 4
  std::vector<double> x{ 3.0, 7.0 };
  auto dx = derivative_transform(x);

  REQUIRE(dx.size() == 2);
  REQUIRE_THAT(dx[0], WithinAbs(4.0, 1e-15));
  REQUIRE_THAT(dx[1], WithinAbs(4.0, 1e-15));
}

TEST_CASE("derivative_transform of empty series", "[ddtw][derivative_transform]")
{
  std::vector<double> x{};
  auto dx = derivative_transform(x);
  REQUIRE(dx.empty());
}

TEST_CASE("derivative_transform of constant series", "[ddtw][derivative_transform]")
{
  std::vector<double> x{ 5.0, 5.0, 5.0, 5.0 };
  auto dx = derivative_transform(x);

  REQUIRE(dx.size() == 4);
  for (auto v : dx)
    REQUIRE_THAT(v, WithinAbs(0.0, 1e-15));
}

TEST_CASE("derivative_transform of linear series", "[ddtw][derivative_transform]")
{
  // x = {2, 4, 6, 8, 10}, slope = 2 everywhere
  // x'[0] = 4 - 2 = 2
  // x'[i] = ((x[i]-x[i-1]) + (x[i+1]-x[i-1])/2) / 2 = (2 + 2) / 2 = 2 for interior
  // x'[4] = 10 - 8 = 2
  std::vector<double> x{ 2, 4, 6, 8, 10 };
  auto dx = derivative_transform(x);

  REQUIRE(dx.size() == 5);
  for (auto v : dx)
    REQUIRE_THAT(v, WithinAbs(2.0, 1e-12));
}

// ---------------------------------------------------------------------------
// ddtwBanded tests
// ---------------------------------------------------------------------------

TEST_CASE("ddtw self-distance is zero", "[ddtw][ddtwBanded]")
{
  std::vector<double> x{ 1.0, 3.0, 5.0, 2.0, 4.0 };

  REQUIRE_THAT(ddtwBanded(x, x, 2), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(ddtwBanded(x, x, -1), WithinAbs(0.0, 1e-12));
}

TEST_CASE("ddtw symmetry", "[ddtw][ddtwBanded]")
{
  std::vector<double> x{ 1, 3, 5, 2, 4 };
  std::vector<double> y{ 2, 4, 1, 5, 3 };

  double d1 = ddtwBanded(x, y, -1);
  double d2 = ddtwBanded(y, x, -1);
  REQUIRE_THAT(d1, WithinAbs(d2, 1e-12));

  // Also with banding
  double d3 = ddtwBanded(x, y, 2);
  double d4 = ddtwBanded(y, x, 2);
  REQUIRE_THAT(d3, WithinAbs(d4, 1e-12));
}

TEST_CASE("ddtw non-negativity", "[ddtw][ddtwBanded]")
{
  std::vector<double> x{ 1, 3, 5, 2, 4 };
  std::vector<double> y{ 10, 20, 30, 40, 50 };

  REQUIRE(ddtwBanded(x, y, -1) >= 0.0);
  REQUIRE(ddtwBanded(x, y, 2) >= 0.0);
}

TEST_CASE("ddtw equivalence to manual derivative + dtwBanded", "[ddtw][ddtwBanded]")
{
  std::vector<double> x{ 1, 3, 5, 2, 4 };
  std::vector<double> y{ 2, 4, 1, 5, 3 };

  auto dx = derivative_transform(x);
  auto dy = derivative_transform(y);

  // Full DTW (band = -1)
  double manual = dtwBanded<double>(dx, dy, -1);
  double ddtw_val = ddtwBanded(x, y, -1);
  REQUIRE_THAT(ddtw_val, WithinAbs(manual, 1e-12));

  // Banded DTW
  int band = 2;
  double manual_banded = dtwBanded<double>(dx, dy, band);
  double ddtw_banded = ddtwBanded(x, y, band);
  REQUIRE_THAT(ddtw_banded, WithinAbs(manual_banded, 1e-12));
}

TEST_CASE("ddtw with band=-1 uses full DTW on derivatives", "[ddtw][ddtwBanded]")
{
  std::vector<double> x{ 1, 2, 3, 4, 5 };
  std::vector<double> y{ 5, 4, 3, 2, 1 };

  auto dx = derivative_transform(x);
  auto dy = derivative_transform(y);

  double expected = dtwFull_L<double>(dx, dy);
  double result = ddtwBanded(x, y, -1);
  REQUIRE_THAT(result, WithinAbs(expected, 1e-12));
}

// ---------------------------------------------------------------------------
// ddtwFull_L tests
// ---------------------------------------------------------------------------

TEST_CASE("ddtwFull_L self-distance is zero", "[ddtw][ddtwFull_L]")
{
  std::vector<double> x{ 1.0, 3.0, 5.0, 2.0, 4.0 };
  REQUIRE_THAT(ddtwFull_L(x, x), WithinAbs(0.0, 1e-12));
}

TEST_CASE("ddtwFull_L symmetry", "[ddtw][ddtwFull_L]")
{
  std::vector<double> x{ 1, 3, 5, 2, 4 };
  std::vector<double> y{ 2, 4, 1, 5, 3 };

  REQUIRE_THAT(ddtwFull_L(x, y), WithinAbs(ddtwFull_L(y, x), 1e-12));
}

TEST_CASE("ddtwFull_L equivalence to manual derivative + dtwFull_L", "[ddtw][ddtwFull_L]")
{
  std::vector<double> x{ 1, 3, 5, 2, 4 };
  std::vector<double> y{ 2, 4, 1, 5, 3 };

  auto dx = derivative_transform(x);
  auto dy = derivative_transform(y);

  double expected = dtwFull_L<double>(dx, dy);
  double result = ddtwFull_L(x, y);
  REQUIRE_THAT(result, WithinAbs(expected, 1e-12));
}

TEST_CASE("ddtwFull_L agrees with ddtwBanded band=-1", "[ddtw][ddtwFull_L]")
{
  std::vector<double> x{ 1, 3, 5, 2, 4 };
  std::vector<double> y{ 2, 4, 1, 5, 3 };

  REQUIRE_THAT(ddtwFull_L(x, y), WithinAbs(ddtwBanded(x, y, -1), 1e-12));
}

TEST_CASE("ddtw different-length series", "[ddtw][ddtwBanded]")
{
  std::vector<double> x{ 1, 2, 3 };
  std::vector<double> y{ 1, 2, 3, 4, 5 };

  auto dx = derivative_transform(x);
  auto dy = derivative_transform(y);

  double expected = dtwBanded<double>(dx, dy, -1);
  double result = ddtwBanded(x, y, -1);
  REQUIRE_THAT(result, WithinAbs(expected, 1e-12));

  // Symmetry with different lengths
  REQUIRE_THAT(ddtwBanded(x, y, -1), WithinAbs(ddtwBanded(y, x, -1), 1e-12));
}
