/**
 * @file unit_test_Range.cpp
 * @brief Unit tests for the dtwc::Range class and its dtwc::Index.
 *
 * Covers default construction, single- and double-parameter construction,
 * range-based iteration correctness, and the Index arithmetic that iteration
 * does not use (decrement, difference, offsets, subscript, equality).
 *
 * @date 31 Dec 2023
 * @authors Volkan Kumtepeli, Becky Perriment
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using namespace dtwc;

TEST_CASE("Range class functionality", "[Range]")
{
  using namespace dtwc;

  SECTION("Default constructor")
  {
    Range range;
    REQUIRE(*range.begin() == 0);
    REQUIRE(*range.end() == 0);
  }

  SECTION("Single parameter constructor")
  {
    Range range(5);
    REQUIRE(*range.begin() == 0);
    REQUIRE(*range.end() == 5);
  }

  SECTION("Double parameter constructor")
  {
    Range range(2, 5);
    REQUIRE(*range.begin() == 2);
    REQUIRE(*range.end() == 5);
  }

  SECTION("Iteration over Range")
  {
    size_t sum = 0;
    for (auto idx : Range(0, 3))
      sum += idx;

    REQUIRE(sum == 3); // 0 + 1 + 2
  }
}

TEST_CASE("Index arithmetic and comparison", "[Index]")
{
  REQUIRE(*(--Index(3)) == 2);
  REQUIRE((Index(10) - Index(5)) == 5);
  REQUIRE(*(Index(5) + 5) == 10);
  REQUIRE(*(Index(10) - 5) == 5);
  REQUIRE(Index(10)[2] == 12);
  REQUIRE(Index(5) == Index(5));
  REQUIRE_FALSE(Index(5) == Index(10));
}
