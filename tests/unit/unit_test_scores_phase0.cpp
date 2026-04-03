/**
 * @file unit_test_scores_phase0.cpp
 * @brief Unit tests for the Davies-Bouldin Index fix (Phase 0)
 *
 * Verifies that daviesBouldinIndex returns a positive value for
 * a clustered dataset, not always 0 as the buggy version did.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>

#include <vector>
#include <string>

using namespace dtwc;

/**
 * Helper: build a Problem with explicit time-series data and manually
 * assigned clusters/centroids so we can test scoring functions without
 * running the full clustering pipeline.
 */
static Problem make_clustered_problem()
{
  // Two well-separated clusters:
  //   Cluster 0: series 0, 1, 2  (values near 1)
  //   Cluster 1: series 3, 4, 5  (values near 10)
  std::vector<std::vector<data_t>> vecs = {
    { 1.0, 1.0, 1.0 }, // 0 - cluster 0
    { 1.1, 1.0, 0.9 }, // 1 - cluster 0
    { 0.9, 1.1, 1.0 }, // 2 - cluster 0
    { 10.0, 10.0, 10.0 }, // 3 - cluster 1
    { 10.1, 10.0, 9.9 },  // 4 - cluster 1
    { 9.9, 10.1, 10.0 },  // 5 - cluster 1
  };
  std::vector<std::string> names = { "a", "b", "c", "d", "e", "f" };

  Data data(std::move(vecs), std::move(names));

  Problem prob("dbi_test");
  prob.set_data(std::move(data));
  prob.set_numberOfClusters(2);

  // Manually assign clusters: first 3 -> cluster 0, last 3 -> cluster 1
  prob.clusters_ind = { 0, 0, 0, 1, 1, 1 };
  // Medoids: series 0 for cluster 0, series 3 for cluster 1
  prob.centroids_ind = { 0, 3 };

  return prob;
}

TEST_CASE("DBI returns positive value for well-separated clusters", "[scores][dbi]")
{
  auto prob = make_clustered_problem();
  double dbi = scores::daviesBouldinIndex(prob);

  // The old buggy code always returned 0. A correct implementation must
  // return a positive value for non-trivial clusters.
  REQUIRE(dbi > 0.0);

  // For well-separated clusters the DBI should be small (good clustering),
  // but definitely not zero.
  REQUIRE(dbi < 10.0); // sanity upper bound
}

TEST_CASE("DBI throws when not clustered", "[scores][dbi]")
{
  Problem prob("empty_test");
  // centroids_ind is empty -> should throw
  REQUIRE_THROWS_AS(scores::daviesBouldinIndex(prob), std::runtime_error);
}

TEST_CASE("DBI is smaller for well-separated clusters than for overlapping", "[scores][dbi]")
{
  // Well-separated clusters
  auto prob_good = make_clustered_problem();
  double dbi_good = scores::daviesBouldinIndex(prob_good);

  // Same data but deliberately bad clustering: mix near-1 and near-10 series
  auto prob_bad = make_clustered_problem();
  prob_bad.clusters_ind = { 0, 1, 0, 1, 0, 1 };
  prob_bad.centroids_ind = { 0, 1 }; // medoids: series 0 (value~1) and series 1 (value~1)

  double dbi_bad = scores::daviesBouldinIndex(prob_bad);

  // Bad clustering should have a higher (worse) DBI than good clustering
  REQUIRE(dbi_bad > dbi_good);
}
