/**
 * @file unit_test_Problem.cpp
 * @brief Unit test file for Problem class
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 03 Dec 2023
 */

#include <dtwc.hpp>

#include "../support/scratch_directory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

using namespace dtwc;

TEST_CASE("cluster_and_process completes when the silhouette is undefined",
          "[Problem][silhouette][degenerate]")
{
  // Regression: write_silhouettes() called scores::silhouette() unguarded, so
  // the shipped cluster_and_process() path aborted with an exception after
  // already writing the distance matrix and cluster files whenever fewer than
  // two clusters are realised (k = 1 here). Warn and skip the file instead.
  const dtwc::test_support::ScratchDirectory out{ "cap_k1" };

  std::vector<std::vector<data_t>> series{
    { 0.0, 1.0, 2.0 }, { 0.0, 2.0, 4.0 }, { 5.0, 5.0, 5.0 }
  };
  std::vector<std::string> names{ "a", "b", "c" };

  Problem prob{ "cap_k1" };
  prob.set_data(Data(std::move(series), std::move(names)));
  prob.set_output_folder(out.path);
  prob.set_n_clusters(1);
  prob.set_max_iter(10);

  REQUIRE_NOTHROW(prob.cluster_and_process());
  REQUIRE(std::filesystem::exists(out.path / "cap_k1_Nc_1.csv"));
  REQUIRE_FALSE(std::filesystem::exists(out.path / "cap_k1_silhouettes_Nc_1.csv"));
}

TEST_CASE("write_silhouettes propagates a corrupt labelling",
          "[Problem][silhouette][taxonomy]")
{
  // Audit 2026-09-02, D2: the skip-and-warn guard caught every InvalidInput, so
  // a clusters_ind that disagrees with the data ("labels disagree") was
  // downgraded to a warning and the caller reported success with no file.
  // Only the undefined-score case (UndefinedScore) may be skipped.
  const dtwc::test_support::ScratchDirectory out{ "sil_corrupt" };

  std::vector<std::vector<data_t>> series{
    { 0.0, 1.0, 2.0 }, { 0.0, 2.0, 4.0 }, { 5.0, 5.0, 5.0 }
  };
  std::vector<std::string> names{ "a", "b", "c" };

  Problem prob{ "sil_corrupt" };
  prob.set_data(Data(std::move(series), std::move(names)));
  prob.set_output_folder(out.path);
  prob.set_n_clusters(2);
  prob.centroids_ind = { 0, 2 };

  // (a) clusters_ind of the wrong length: a data/labelling error, must throw.
  prob.clusters_ind = { 0, 1 };
  REQUIRE_THROWS_AS(prob.write_silhouettes(), dtwc::InvalidInput);
  REQUIRE_FALSE(std::filesystem::exists(out.path / "sil_corrupt_silhouettes_Nc_2.csv"));

  // (b) an out-of-range label: likewise.
  prob.clusters_ind = { 0, 1, 7 };
  REQUIRE_THROWS_AS(prob.write_silhouettes(), dtwc::InvalidInput);

  // (c) a valid labelling that realises only one cluster: still skipped.
  prob.clusters_ind = { 0, 0, 0 };
  REQUIRE_NOTHROW(prob.write_silhouettes());
  REQUIRE_FALSE(std::filesystem::exists(out.path / "sil_corrupt_silhouettes_Nc_2.csv"));

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
}
