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

#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

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

TEST_CASE("ddtw equivalence to manual derivative + dtwBanded", "[ddtw][ddtwBanded]")
{
  std::vector<double> x{ 1, 3, 5, 2, 4 };
  std::vector<double> y{ 2, 4, 1, 5, 3 };

  auto dx = derivative_transform(x);
  auto dy = derivative_transform(y);

  REQUIRE_THAT(ddtwBanded(x, y, -1), WithinAbs(dtwBanded<double>(dx, dy, -1), 1e-12));
  REQUIRE_THAT(ddtwBanded(x, y, 2), WithinAbs(dtwBanded<double>(dx, dy, 2), 1e-12));
}

TEST_CASE("ddtwFull_L equivalence to manual derivative + dtwFull_L", "[ddtw][ddtwFull_L]")
{
  std::vector<double> x{ 1, 3, 5, 2, 4 };
  std::vector<double> y{ 2, 4, 1, 5, 3 };

  auto dx = derivative_transform(x);
  auto dy = derivative_transform(y);

  REQUIRE_THAT(ddtwFull_L(x, y), WithinAbs(dtwFull_L<double>(dx, dy), 1e-12));
}
