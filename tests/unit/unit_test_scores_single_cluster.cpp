/**
 * @file unit_test_scores_single_cluster.cpp
 * @brief R4(a): Nc<2 guard for the Davies-Bouldin and Dunn indices.
 *
 * The adversarial audit (handoff-2026-06-01:25) found that davies_bouldin
 * and dunn lacked an Nc<2 guard. With a single cluster the results are
 * mathematically undefined and silently wrong rather than erroring:
 *   - DBI: the max_{j!=i} loop finds no second cluster, so the index collapses
 *          to 0 (looks like a "perfect" clustering).
 *   - Dunn: there are no inter-cluster pairs, so min_inter stays at
 *          numeric_limits<double>::max() and the ratio is a meaningless huge
 *          value (or +inf).
 * The fix rejects Nc < 2 with dtwc::InvalidInput (the project taxonomy; it
 * derives from std::runtime_error).
 *
 * These tests exercise the LIVE public entry points dtwc::scores::davies_bouldin
 * and dtwc::scores::dunn (declared in scores.hpp) on a clustered (non-empty
 * centroids) but single-cluster problem, so the new Nc<2 guard fires — NOT the
 * pre-existing "cluster first" (empty centroids) runtime_error guard.
 *
 * @author Volkan Kumtepeli
 * @date 06 Jul 2026
 */

#include <dtwc.hpp>
#include <scores.hpp>

#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <string>
#include <vector>

using namespace dtwc;

namespace {

/// Build a fully-clustered problem with exactly ONE cluster (Nc == 1).
/// centroids_ind is non-empty, so the "cluster first" guard is satisfied and
/// only the new Nc<2 guard can fire.
Problem make_single_cluster_problem()
{
  std::vector<std::vector<data_t>> vecs = {
    { 1.0, 1.0, 1.0 },
    { 1.1, 1.0, 0.9 },
    { 0.9, 1.1, 1.0 },
  };
  std::vector<std::string> names = { "a", "b", "c" };

  Data data(std::move(vecs), std::move(names));
  Problem prob("single_cluster");
  prob.set_data(std::move(data));
  prob.set_n_clusters(1);
  prob.clusters_ind = { 0, 0, 0 }; // all points in the one cluster
  prob.centroids_ind = { 0 };      // clustered (non-empty) but a single medoid
  return prob;
}

} // namespace

TEST_CASE("DBI: single cluster (Nc<2) throws InvalidInput, not silent 0",
          "[scores][dbi][r4]")
{
  auto prob = make_single_cluster_problem();
  REQUIRE(prob.n_clusters() == 1);
  REQUIRE_THROWS_AS(scores::davies_bouldin(prob), dtwc::InvalidInput);
}

TEST_CASE("Dunn: single cluster (Nc<2) throws InvalidInput, not silent inf",
          "[scores][dunn][r4]")
{
  auto prob = make_single_cluster_problem();
  REQUIRE(prob.n_clusters() == 1);
  REQUIRE_THROWS_AS(scores::dunn(prob), dtwc::InvalidInput);
}

TEST_CASE("DBI/Dunn: empty-centroids problem still throws the 'cluster first' error",
          "[scores][dbi][dunn][r4]")
{
  // No centroids set at all: the pre-existing runtime_error guard must still
  // win (the Nc<2 guard must not shadow the "cluster first" precondition).
  Problem prob("unclustered");
  std::vector<std::vector<data_t>> vecs = { { 1.0, 2.0 }, { 3.0, 4.0 } };
  std::vector<std::string> names = { "a", "b" };
  prob.set_data(Data(std::move(vecs), std::move(names)));

  REQUIRE_THROWS_AS(scores::davies_bouldin(prob), dtwc::InvalidInput);
  REQUIRE_THROWS_AS(scores::dunn(prob), dtwc::InvalidInput);
}
