/*
 * unit_test_fileOperations.cpp
 *
 * Unit test file for file reading functions
 *  Created on: 25 Dec 2023
 *   Author(s): Volkan Kumtepeli, Becky Perriment
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include <string>
#include <sstream>
#include <fstream>
#include <random>
#include <algorithm>
#include <iterator>

using Catch::Matchers::WithinAbs;

using namespace dtwc;
using namespace Eigen;

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

TEST_CASE("Write and Read Eigen Matrices", "[fileOperations]")
{
  // Write and read different length matrices.
  auto M = GENERATE(1, 2, 3, 5, 10, 20, 50, 100);
  auto N = GENERATE(1, 2, 3, 5, 10, 20, 50, 100);

  Eigen::ArrayXXd matrix = MatrixXd::Random(M, N);
  fs::path tempFilePath = "test_matrix.csv";

  dtwc::writeMatrix(matrix, tempFilePath);

  Eigen::Array<data_t, Eigen::Dynamic, Eigen::Dynamic> readMat;

  dtwc::readMatrix(readMat, tempFilePath);

  REQUIRE(readMat.isApprox(matrix, 1e-3));
  fs::remove(tempFilePath); // Clean up the test file
}

TEST_CASE("Write and Read Empty Matrix", "[fileOperations]")
{
  Eigen::ArrayXXd matrix;
  fs::path tempFilePath = "test_matrix.csv";

  dtwc::writeMatrix(matrix, tempFilePath);

  REQUIRE(matrix.rows() == 0);
  REQUIRE(matrix.cols() == 0);

  Eigen::Array<data_t, Eigen::Dynamic, Eigen::Dynamic> readMat;

  dtwc::readMatrix(readMat, tempFilePath);

  REQUIRE(readMat.rows() == 0);
  REQUIRE(readMat.cols() == 0);

  fs::remove(tempFilePath); // Clean up the test file
}

TEST_CASE("Load batch file", "[fileOperations]")
{
  std::string tempFileName = "test_matrix";

  // Generate data:
  const int N_data = GENERATE(1, 2, 10, 1000); // Size of the outer vector
  const int L_data = GENERATE(1, 2, 10, 1000); // Maximum size of the inner vectors

  std::vector<std::vector<double>> random_data;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_int_distribution<> dis(0, L_data);


  for (int i = 0; i < N_data; ++i) {
    int innerSize = dis(gen); // Random size for the inner vector
    std::vector<double> innerVector;

    for (int j = 0; j < innerSize; ++j)
      innerVector.push_back(dis(gen)); // Generate random number

    random_data.push_back(std::move(innerVector));
  }

  // write the files
  {
    std::ofstream out_csv(tempFileName + ".csv", std::ios_base::out);
    std::ofstream out_tsv(tempFileName + ".tsv", std::ios_base::out);


    for (const auto &innerVector : random_data) {
      for (size_t i = 0; i < innerVector.size(); i++) {
        if (i != 0) {
          out_csv << ',';
          out_tsv << '\t';
        }

        out_csv << innerVector[i];
        out_tsv << innerVector[i];
      }

      out_csv << '\n';
      out_tsv << '\n';
    }
  } // Auto close at the end thanks to the destructor.


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
