/*
 * unit_test_warping.cpp
 *
 * Unit test file for time warping functions
 *  Created on: 03 Dec 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;

using namespace dtwc;

TEST_CASE("dtwFull_test", "[dtwFull]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 }, z{ 1, 2, 3 };

  // Zero distance between same vectors:
  REQUIRE_THAT(dtwFull<data_t>(x, x), WithinAbs(0, 1e-15));
  REQUIRE_THAT(dtwFull<data_t>(x, z), WithinAbs(0, 1e-15));
  REQUIRE_THAT(dtwFull<data_t>(z, x), WithinAbs(0, 1e-15));

  // Some distance between others: 13
  REQUIRE_THAT(dtwFull<data_t>(x, y), WithinAbs(13, 1e-15));
  REQUIRE_THAT(dtwFull<data_t>(y, x), WithinAbs(13, 1e-15));
}