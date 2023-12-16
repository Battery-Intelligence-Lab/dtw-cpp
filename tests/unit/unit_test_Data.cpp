/*
 * unit_test_Data.cpp
 *
 * Unit test file for time Data class
 *  Created on: 16 Dec 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>
#include <string>

using Catch::Matchers::WithinAbs;

using namespace dtwc;

TEST_CASE("Data class functionality", "[Data]")
{
  SECTION("Default constructor creates empty Data object")
  {
    Data data;
    REQUIRE(data.size() == 0);
    REQUIRE(data.p_vec.empty());
    REQUIRE(data.p_names.empty());
  }

  SECTION("Parameterized constructor sets data correctly")
  {
    std::vector<std::vector<data_t>> testVec = { { 1, 2, 3 }, { 4, 5, 6 } };
    std::vector<std::string> testNames = { "First", "Second" };

    Data data(std::move(testVec), std::move(testNames));

    REQUIRE(data.size() == 2);
    REQUIRE(data.p_vec.size() == 2);
    REQUIRE(data.p_names.size() == 2);
    REQUIRE(data.p_vec[0].size() == 3);
    REQUIRE(data.p_vec[1].size() == 3);
    REQUIRE(data.p_names[0] == "First");
    REQUIRE(data.p_names[1] == "Second");
  }

  SECTION("Data size corresponds to number of elements")
  {
    Data data;
    data.p_vec = { { 1 }, { 2, 3 }, { 4, 5, 6 } };
    data.p_names = { "A", "B", "C" };
    data.Nb = static_cast<int>(data.p_vec.size());

    REQUIRE(data.size() == 3);
  }

  SECTION("Constructor throws assertion error with mismatched vector sizes")
  {
    std::vector<std::vector<data_t>> testVec = { { 1, 2 }, { 3, 4 } };
    std::vector<std::string> testNames = { "One" };

    REQUIRE_THROWS_AS(Data(std::move(testVec), std::move(testNames)), std::exception);
  }
}