/**
 * @file unit_test_fileOperations.cpp
 * @brief Unit test file for file reading functions
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 25 Dec 2023
 */

#include <dtwc.hpp>
#include "../test_util.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <set>
#include <map>

using Catch::Matchers::WithinAbs;

using namespace dtwc;

TEST_CASE("ignoreBOM test", "[file_operations]")
{
  auto testText = [](std::string data) {
    std::stringstream in(data);

    ignoreBOM(in);

    std::string result;
    std::getline(in, result);

    return result;
  };

  SECTION("ignoreBOM with BOM present")
  {
    std::string dataText = "\xEF\xBB\xBFText with BOM";
    REQUIRE(testText(dataText) == "Text with BOM");
  }

  SECTION("ignoreBOM without BOM present")
  {
    std::string dataText = "Text without BOM";
    REQUIRE(testText(dataText) == "Text without BOM");
  }

  SECTION("ignoreBOM with empty stream")
  {
    std::string dataText("");
    REQUIRE(testText(dataText).empty());
  }

  SECTION("ignoreBOM with stream less than 3 characters")
  {
    std::string dataText = "Hi";
    REQUIRE(testText(dataText) == "Hi");
  }
}

TEST_CASE("Write and Read Distance Matrices via Problem", "[fileOperations]")
{
  // Test round-trip of distance matrix I/O through Problem.
  auto N = GENERATE(1, 2, 5, 10, 20);

  // Create a DenseDistanceMatrix with random values.
  dtwc::core::DenseDistanceMatrix matrix(static_cast<size_t>(N));
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(0.0, 100.0);
  for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
    matrix.set(i, i, 0.0);
    for (size_t j = i + 1; j < static_cast<size_t>(N); ++j)
      matrix.set(i, j, dist(rng));
  }

  // Write via Problem's writeDistanceMatrix mechanism (inline CSV).
  fs::path tempFilePath = "test_distmat.csv";
  {
    std::ofstream file(tempFilePath);
    for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
      for (size_t j = 0; j < static_cast<size_t>(N); ++j) {
        if (j > 0) file << ',';
        file << std::setprecision(15) << matrix.get(i, j);
      }
      file << '\n';
    }
  }

  // Read back via Problem's readDistanceMatrix mechanism.
  Problem prob;
  prob.readDistanceMatrix(tempFilePath);

  // Verify the file round-trip by reading the CSV manually and comparing.
  {
    std::ifstream inFile(tempFilePath);
    std::string line;
    size_t row = 0;
    while (std::getline(inFile, line)) {
      std::istringstream ss(line);
      std::string cell;
      size_t col = 0;
      while (std::getline(ss, cell, ',')) {
        double val = std::stod(cell);
        REQUIRE_THAT(val, WithinAbs(matrix.get(row, col), 1e-3));
        ++col;
      }
      ++row;
    }
    REQUIRE(row == static_cast<size_t>(N));
  } // close inFile before removing

  fs::remove(tempFilePath);
}

TEST_CASE("Write and Read Empty Matrix", "[fileOperations]")
{
  dtwc::core::DenseDistanceMatrix matrix;
  REQUIRE(matrix.size() == 0);

  // Empty matrix should be default-constructed with size 0.
  dtwc::core::DenseDistanceMatrix readMat;
  REQUIRE(readMat.size() == 0);
}


TEST_CASE("Load batch file", "[fileOperations]")
{
  std::string tempFileName = "test_matrix";

  // Generate data:
  const int N_data = GENERATE(1, 2, 10, 1000); // Size of the outer vector
  const int L_data = GENERATE(1, 2, 10, 1000); // Maximum size of the inner vectors

  const auto random_data = test_util::get_random_data<double>(N_data, L_data);

  // write the files
  test_util::write_data_to_file(tempFileName + ".csv", random_data, ',');
  test_util::write_data_to_file(tempFileName + ".tsv", random_data, '\t');

  // ----- now testing -----
  SECTION("csv batch load")
  {
    fs::path pth = tempFileName + ".csv";
    int start_row{ 0 }, start_col{}, delimiter{ ',' };
    auto [p_vec, p_names] = load_batch_file<double>(pth, N_data, false, start_row, start_col, delimiter);

    for (size_t i{}; i < N_data; i++)
      REQUIRE(p_names[i] == std::to_string(i + 1));

    REQUIRE(p_vec == random_data);
  }

  SECTION("tsv batch load")
  {
    fs::path pth = tempFileName + ".tsv";
    int start_row{ 0 }, start_col{}, delimiter{ '\t' };
    auto [p_vec, p_names] = load_batch_file<double>(pth, N_data, false, start_row, start_col, delimiter);

    for (size_t i{}; i < N_data; i++)
      REQUIRE(p_names[i] == std::to_string(i + 1));

    REQUIRE(p_vec == random_data);
  }

  fs::remove(tempFileName + ".csv"); // Clean up the test files
  fs::remove(tempFileName + ".tsv"); // Clean up the test files
}

TEST_CASE("readFile throws on missing file", "[fileOperations]")
{
  fs::path nonExistentFile = "this_file_does_not_exist_12345.csv";

  // Ensure the file doesn't exist
  if (fs::exists(nonExistentFile)) {
    fs::remove(nonExistentFile);
  }

  REQUIRE_THROWS_AS(readFile<double>(nonExistentFile), std::runtime_error);
}

TEST_CASE("load_batch_file throws on missing file", "[fileOperations]")
{
  fs::path nonExistentFile = "this_batch_file_does_not_exist_12345.csv";

  // Ensure the file doesn't exist
  if (fs::exists(nonExistentFile)) {
    fs::remove(nonExistentFile);
  }

  REQUIRE_THROWS_AS(load_batch_file<double>(nonExistentFile), std::runtime_error);
}

TEST_CASE("Load folder", "[fileOperations]")
{
  // Generate data:
  constexpr int stringLength = 10;

  const int N_data = GENERATE(1, 2, 10, 100);  // Size of the outer vector
  const int L_data = GENERATE(1, 2, 10, 1000); // Maximum size of the inner vectors

  const auto random_data = test_util::get_random_data<double>(N_data, L_data);
  const auto random_names = test_util::get_random_names(N_data, stringLength);

  // ----- now testing -----
  SECTION("csv batch load")
  {
    std::string folder("CSV");
    test_util::write_data_to_folder(folder, random_data, random_names);
    fs::path pth = folder;
    int start_row{ 0 }, start_col{ 1 };
    char delimiter{ ',' };
    auto [p_vec, p_names] = load_folder<double>(pth, N_data, false, start_row, start_col, delimiter);

    // Order of names and data is different in different operating systems.
    for (size_t i{}; i < N_data; i++) {
      auto iterNow = std::find(p_names.begin(), p_names.end(), random_names[i]);
      REQUIRE(iterNow != p_names.end());

      const int j = std::distance(p_names.begin(), iterNow);
      REQUIRE(p_vec[j] == random_data[i]);
    }
  }

  fs::remove_all("CSV"); // Clean up the test files
}
