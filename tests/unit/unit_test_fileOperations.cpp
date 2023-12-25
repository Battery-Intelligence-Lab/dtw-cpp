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

using Catch::Matchers::WithinAbs;

using namespace dtwc;
using namespace Eigen;

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