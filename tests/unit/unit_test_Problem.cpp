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

#include <chrono>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

using Catch::Matchers::WithinAbs;

using namespace dtwc;

// Use compile-time test data directory if available, otherwise fall back to relative path
#ifndef DTWC_TEST_DATA_DIR
#define DTWC_TEST_DATA_DIR "./data"
#endif

// Initialize settings::paths::data from compile-time definition for tests
struct TestPathInitializer {
  TestPathInitializer() {
    dtwc::settings::paths::set_data_path(DTWC_TEST_DATA_DIR);
  }
};
static TestPathInitializer testPathInit;

TEST_CASE("dtwFull_test", "[dtwFull]")
{
  dtwc::Clock clk; // Create a clock object
  std::string probName = "DTW_kMeans_results";

  auto Nc = 3; // Number of clusters

  constexpr int n_repetitions = 5;
  constexpr int max_iter = 100;
  constexpr int Ndata_max = 10;

  dtwc::DataLoader dl{ settings::paths::data / "dummy", Ndata_max };
  dl.start_column(1).start_row(1); // Since dummy files are in Pandas format skip first row/column.

  dtwc::Problem prob{ probName, dl }; // Create a problem.

  prob.set_n_clusters(Nc); // Nc = number of clusters.

  REQUIRE(prob.n_clusters() == Nc);
  REQUIRE(prob.name() == probName);

  // prob.cluster_by_kMedoidsLloyd_repetetive(n_repetitions, max_iter);
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

TEST_CASE("cluster_and_process completes when the silhouette is undefined",
          "[Problem][silhouette][degenerate]")
{
  // Regression: write_silhouettes() called scores::silhouette() unguarded, so
  // the shipped cluster_and_process() path aborted with an exception after
  // already writing the distance matrix and cluster files whenever fewer than
  // two clusters are realised (k = 1 here). Warn and skip the file instead.
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto out = std::filesystem::temp_directory_path()
                 / ("dtwc_cap_k1_" + std::to_string(nonce));
  std::filesystem::create_directories(out);

  std::vector<std::vector<data_t>> series{
    { 0.0, 1.0, 2.0 }, { 0.0, 2.0, 4.0 }, { 5.0, 5.0, 5.0 }
  };
  std::vector<std::string> names{ "a", "b", "c" };

  Problem prob{ "cap_k1" };
  prob.set_data(Data(std::move(series), std::move(names)));
  prob.set_output_folder(out);
  prob.set_n_clusters(1);
  prob.set_max_iter(10);

  REQUIRE_NOTHROW(prob.cluster_and_process());
  REQUIRE(std::filesystem::exists(out / "cap_k1_Nc_1.csv"));
  REQUIRE_FALSE(std::filesystem::exists(out / "cap_k1_silhouettes_Nc_1.csv"));

  std::error_code ec;
  std::filesystem::remove_all(out, ec);
}

TEST_CASE("write_silhouettes propagates a corrupt labelling",
          "[Problem][silhouette][taxonomy]")
{
  // Audit 2026-09-02, D2: the skip-and-warn guard caught every InvalidInput, so
  // a clusters_ind that disagrees with the data ("labels disagree") was
  // downgraded to a warning and the caller reported success with no file.
  // Only the undefined-score case (UndefinedScore) may be skipped.
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const auto out = std::filesystem::temp_directory_path()
                 / ("dtwc_sil_corrupt_" + std::to_string(nonce));
  std::filesystem::create_directories(out);

  std::vector<std::vector<data_t>> series{
    { 0.0, 1.0, 2.0 }, { 0.0, 2.0, 4.0 }, { 5.0, 5.0, 5.0 }
  };
  std::vector<std::string> names{ "a", "b", "c" };

  Problem prob{ "sil_corrupt" };
  prob.set_data(Data(std::move(series), std::move(names)));
  prob.set_output_folder(out);
  prob.set_n_clusters(2);
  prob.centroids_ind = { 0, 2 };

  // (a) clusters_ind of the wrong length: a data/labelling error, must throw.
  prob.clusters_ind = { 0, 1 };
  REQUIRE_THROWS_AS(prob.write_silhouettes(), dtwc::InvalidInput);
  REQUIRE_FALSE(std::filesystem::exists(out / "sil_corrupt_silhouettes_Nc_2.csv"));

  // (b) an out-of-range label: likewise.
  prob.clusters_ind = { 0, 1, 7 };
  REQUIRE_THROWS_AS(prob.write_silhouettes(), dtwc::InvalidInput);

  // (c) a valid labelling that realises only one cluster: still skipped.
  prob.clusters_ind = { 0, 0, 0 };
  REQUIRE_NOTHROW(prob.write_silhouettes());
  REQUIRE_FALSE(std::filesystem::exists(out / "sil_corrupt_silhouettes_Nc_2.csv"));

  // Non-vacuity: the two types really are distinguishable at the throw site.
  prob.clusters_ind = { 0, 1 };
  try {
    (void)scores::silhouette(prob);
    FAIL("silhouette accepted a clusters_ind of the wrong length");
  } catch (const dtwc::UndefinedScore &) {
    FAIL("a corrupt labelling was reported as an undefined score");
  } catch (const dtwc::InvalidInput &) {
    SUCCEED("a corrupt labelling raises the propagating InvalidInput");
  }
  prob.clusters_ind = { 0, 0, 0 };
  REQUIRE_THROWS_AS(scores::silhouette(prob), dtwc::UndefinedScore);

  std::error_code ec;
  std::filesystem::remove_all(out, ec);
}
