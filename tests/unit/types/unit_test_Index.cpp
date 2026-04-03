/**
 * @file unit_test_Index.cpp
 * @brief Unit tests for the dtwc::Index class.
 *
 * Covers default and parameterised construction, pre-increment/decrement,
 * arithmetic operators (+, -, difference), subscript indexing, and equality.
 *
 * @date 31 Dec 2023
 * @authors Volkan Kumtepeli, Becky Perriment
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;

using namespace dtwc;

TEST_CASE("Index class tests", "[Index]")
{
  using namespace dtwc;

  SECTION("Default constructor")
  {
    Index idx;
    REQUIRE(*idx == 0);
  }

  SECTION("Parameterized constructor")
  {
    Index idx(5);
    REQUIRE(*idx == 5);
  }

  SECTION("Pre-increment operator")
  {
    Index idx(3);
    ++idx;
    REQUIRE(*idx == 4);
  }

  SECTION("Pre-decrement operator")
  {
    Index idx(3);
    --idx;
    REQUIRE(*idx == 2);
  }

  SECTION("Difference operator")
  {
    Index idx1(10), idx2(5);
    REQUIRE((idx1 - idx2) == 5);
  }

  SECTION("Addition operator")
  {
    Index idx(5);
    auto idx2 = idx + 5;
    REQUIRE(*idx2 == 10);
  }

  SECTION("Subtraction operator")
  {
    Index idx(10);
    auto idx2 = idx - 5;
    REQUIRE(*idx2 == 5);
  }

  SECTION("Indexing operator")
  {
    Index idx(10);
    REQUIRE(idx[2] == 12);
  }

  SECTION("Equality operator")
  {
    Index idx1(5), idx2(5), idx3(10);
    REQUIRE(idx1 == idx2);
    REQUIRE_FALSE(idx1 == idx3);
  }
}