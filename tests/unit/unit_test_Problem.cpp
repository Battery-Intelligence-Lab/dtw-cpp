/**
 * @file unit_test_Problem.cpp
 * @brief Unit test file for Problem class
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 03 Dec 2023
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;

using namespace dtwc;

TEST_CASE("dtwFull_test", "[dtwFull]")
{
  dtwc::Clock clk; // Create a clock object
  std::string probName = "DTW_kMeans_results";

  auto Nc = 3; // Number of clusters

  constexpr int N_repetition = 5;
  constexpr int maxIter = 100;
  constexpr int Ndata_max = 10;

  dtwc::DataLoader dl{ settings::dataPath / "dummy", Ndata_max };
  dl.startColumn(1).startRow(1); // Since dummy files are in Pandas format skip first row/column.

  dtwc::Problem prob{ probName, dl }; // Create a problem.

  prob.set_numberOfClusters(Nc); // Nc = number of clusters.


  REQUIRE(prob.cluster_size() == Nc);
  REQUIRE(prob.name == probName);

  // prob.cluster_by_kMedoidsPAM_repetetive(N_repetition, maxIter);
}

TEST_CASE("dtwFull_L_test", "[dtwFull_L]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 }, z{ 1, 2, 3 }, empty{};
  constexpr double ground_truth = 13;

  // Zero distance between same vectors:
  REQUIRE_THAT(dtwFull_L<data_t>(x, x), WithinAbs(0, 1e-15));
  REQUIRE_THAT(dtwFull_L<data_t>(x, z), WithinAbs(0, 1e-15));
  REQUIRE_THAT(dtwFull_L<data_t>(z, x), WithinAbs(0, 1e-15));

  // Some distance between others: 13
  REQUIRE_THAT(dtwFull_L<data_t>(x, y), WithinAbs(ground_truth, 1e-15));
  REQUIRE_THAT(dtwFull_L<data_t>(y, x), WithinAbs(ground_truth, 1e-15));

  // Empty vector should give infinite cost.
  REQUIRE(dtwFull_L<data_t>(x, empty) > 1e10);
  REQUIRE(dtwFull_L<data_t>(empty, x) > 1e10);
}

TEST_CASE("dtwBanded_test", "[dtwBanded]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 }, z{ 1, 2, 3 }, empty{};
  constexpr double ground_truth = 13;

  // Zero distance between same vectors:
  REQUIRE_THAT(dtwBanded<data_t>(x, x), WithinAbs(0, 1e-15));
  REQUIRE_THAT(dtwBanded<data_t>(x, z), WithinAbs(0, 1e-15));
  REQUIRE_THAT(dtwBanded<data_t>(z, x), WithinAbs(0, 1e-15));

  // Some distance between others with too large band, should be same as unbanded.
  int band = 100;
  REQUIRE_THAT(dtwBanded<data_t>(x, y, band), WithinAbs(ground_truth, 1e-15));
  REQUIRE_THAT(dtwBanded<data_t>(y, x, band), WithinAbs(ground_truth, 1e-15));

  // Banded distance:
  band = 2;
  REQUIRE_THAT(dtwBanded<data_t>(x, y, band), WithinAbs(ground_truth, 1e-15));
  REQUIRE_THAT(dtwBanded<data_t>(y, x, band), WithinAbs(ground_truth, 1e-15));

  // Empty vector should give infinite cost.
  REQUIRE(dtwBanded<data_t>(x, empty) > 1e10);
  REQUIRE(dtwBanded<data_t>(empty, x) > 1e10);
}