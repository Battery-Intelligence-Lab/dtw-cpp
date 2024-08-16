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
#include <armadillo>
#include <fstream>
#include <random>
#include <algorithm>
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

TEST_CASE("Write and Read Distance Matrices", "[fileOperations]")
{
  // Write and read different length matrices.
  auto M = GENERATE(1, 2, 3, 5, 10, 20, 50, 100);
  auto N = GENERATE(1, 2, 3, 5, 10, 20, 50, 100);

  Problem::distMat_t matrix(M, N, arma::fill::randu);
  fs::path tempFilePath = "test_matrix.csv";

  dtwc::writeMatrix(matrix, tempFilePath);

  Problem::distMat_t readMat;

  dtwc::readMatrix(readMat, tempFilePath);

  REQUIRE(arma::approx_equal(readMat, matrix, "absdiff", 1e-3));
  fs::remove(tempFilePath); // Clean up the test file
}

TEST_CASE("Write and Read Empty Matrix", "[fileOperations]")
{
  Problem::distMat_t matrix;
  fs::path tempFilePath = "test_matrix.csv";

  dtwc::writeMatrix(matrix, tempFilePath);

  REQUIRE(matrix.n_rows == 0);
  REQUIRE(matrix.n_cols == 0);

  Problem::distMat_t readMat;

  dtwc::readMatrix(readMat, tempFilePath);

  REQUIRE(readMat.n_rows == 0);
  REQUIRE(readMat.n_cols == 0);

  fs::remove(tempFilePath); // Clean up the test file
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
