/*
 * unit_test_DataLoader.cpp
 *
 * Unit test file for time DataLoader class
 *  Created on: 29 Dec 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <sstream>
#include <thread>

using Catch::Matchers::WithinAbs;

using namespace dtwc;

TEST_CASE("DataLoader class functionality", "[DataLoader]")
{
  SECTION("DataLoader Default Constructor")
  {
    // Test if the DataLoader is constructed with default parameters
    DataLoader loader;

    REQUIRE(loader.startColumn() == 0);
    REQUIRE(loader.startRow() == 0);
    REQUIRE(loader.n_data() == -1);
    REQUIRE(loader.delimiter() == ',');
    REQUIRE(loader.path() == ".");
  }

  SECTION("DataLoader Path Constructor")
  {
    // Test if the DataLoader is constructed with the correct path
    DataLoader loader1("test.csv"), loader2("test.tsv");

    REQUIRE(loader1.startColumn() == 0);
    REQUIRE(loader1.startRow() == 0);
    REQUIRE(loader1.n_data() == -1);
    REQUIRE(loader1.delimiter() == ',');
    REQUIRE(loader1.path() == "test.csv");

    REQUIRE(loader2.startColumn() == 0);
    REQUIRE(loader2.startRow() == 0);
    REQUIRE(loader2.n_data() == -1);
    REQUIRE(loader2.delimiter() == '\t');
    REQUIRE(loader2.path() == "test.tsv");
  }

  SECTION("DataLoader Path and Ndata Constructor")
  {
    // Test if DataLoader is constructed with correct path and Ndata
    int Ndata = 100;
    DataLoader loader1("test.csv", Ndata), loader2("test.tsv", Ndata);

    REQUIRE(loader1.startColumn() == 0);
    REQUIRE(loader1.startRow() == 0);
    REQUIRE(loader1.n_data() == Ndata);
    REQUIRE(loader1.delimiter() == ',');
    REQUIRE(loader1.path() == "test.csv");

    REQUIRE(loader2.startColumn() == 0);
    REQUIRE(loader2.startRow() == 0);
    REQUIRE(loader2.n_data() == Ndata);
    REQUIRE(loader2.delimiter() == '\t');
    REQUIRE(loader2.path() == "test.tsv");
  }

  SECTION("Method Chaining")
  {
    fs::path testPath = "test.csv";
    DataLoader loader;
    loader.startColumn(1).startRow(3).n_data(100).path(testPath).verbosity(1);
    // Test if the method chaining correctly sets the properties
    REQUIRE(loader.startColumn() == 1);
    REQUIRE(loader.startRow() == 3);
    REQUIRE(loader.n_data() == 100);
    REQUIRE(loader.delimiter() == ',');
    REQUIRE(loader.path() == testPath);
    REQUIRE(loader.verbosity() == 1);
  }
}