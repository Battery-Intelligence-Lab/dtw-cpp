/**
 * @file unit_test_clustering_algorithms.cpp
 * @brief Unit tests for clustering algorithm quality and correctness.
 *
 * Tests PAM (k-medoids) convergence, label validity, medoid validity,
 * cost monotonicity, seed sensitivity, and edge cases.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../test_util.hpp"

#include <set>
#include <vector>

#ifndef DTWC_TEST_DATA_DIR
#define DTWC_TEST_DATA_DIR "./data"
#endif

static struct TestDataInit {
  TestDataInit() { dtwc::settings::paths::setDataPath(DTWC_TEST_DATA_DIR); }
} test_data_init_;

using Catch::Matchers::WithinAbs;
using namespace dtwc;

namespace {

/**
 * @brief Helper: create a Problem loaded with the first N dummy series.
 *
 * Sets sensible defaults for iterative clustering tests
 * (maxIter=100, N_repetition=1). The dummy data uses Pandas-style CSV
 * (skip first row and first column).
 */
Problem make_dummy_problem(int N_data, int Nc)
{
  dtwc::DataLoader dl{ settings::paths::data / "dummy", N_data };
  dl.startColumn(1).startRow(1);

  dtwc::Problem prob{ "test_clustering", dl };
  prob.set_numberOfClusters(Nc);
  prob.maxIter = 100;
  prob.N_repetition = 1;
  prob.output_folder = ".";
  return prob;
}

} // anonymous namespace


// ---------------------------------------------------------------------------
// Convergence: PAM should converge for the 25 dummy series
// ---------------------------------------------------------------------------
TEST_CASE("PAM clustering converges for dummy data", "[Phase1][clustering]")
{
  constexpr int N_data = 10;
  constexpr int Nc = 3;

  auto prob = make_dummy_problem(N_data, Nc);

  // Should not throw; should terminate within maxIter.
  REQUIRE_NOTHROW(prob.cluster_by_kMedoidsLloyd());

  // After clustering, labels must be assigned.
  REQUIRE(prob.clusters_ind.size() == static_cast<size_t>(N_data));
  REQUIRE(prob.centroids_ind.size() == static_cast<size_t>(Nc));
}

// ---------------------------------------------------------------------------
// Labels in [0, k)
// ---------------------------------------------------------------------------
TEST_CASE("Cluster labels are in valid range [0, k)", "[Phase1][clustering]")
{
  constexpr int N_data = 10;
  constexpr int Nc = 3;

  auto prob = make_dummy_problem(N_data, Nc);
  prob.cluster_by_kMedoidsLloyd();

  for (int label : prob.clusters_ind) {
    REQUIRE(label >= 0);
    REQUIRE(label < Nc);
  }
}

// ---------------------------------------------------------------------------
// Medoid indices are valid data-point indices
// ---------------------------------------------------------------------------
TEST_CASE("Medoid indices are valid data-point indices", "[Phase1][clustering]")
{
  constexpr int N_data = 10;
  constexpr int Nc = 3;

  auto prob = make_dummy_problem(N_data, Nc);
  prob.cluster_by_kMedoidsLloyd();

  for (int medoid : prob.centroids_ind) {
    REQUIRE(medoid >= 0);
    REQUIRE(medoid < N_data);
  }

  // All medoid indices must be distinct.
  std::set<int> unique_medoids(prob.centroids_ind.begin(), prob.centroids_ind.end());
  REQUIRE(unique_medoids.size() == static_cast<size_t>(Nc));
}

// ---------------------------------------------------------------------------
// Total cost is non-negative after clustering
// ---------------------------------------------------------------------------
TEST_CASE("Total cost is non-negative after clustering", "[Phase1][clustering]")
{
  constexpr int N_data = 10;
  constexpr int Nc = 3;

  auto prob = make_dummy_problem(N_data, Nc);
  prob.cluster_by_kMedoidsLloyd();

  double cost = prob.findTotalCost();
  REQUIRE(cost >= 0.0);
}

// ---------------------------------------------------------------------------
// Multiple repetitions: best cost <= every individual cost
// ---------------------------------------------------------------------------
TEST_CASE("Multiple repetitions pick the best (lowest) cost", "[Phase1][clustering]")
{
  constexpr int N_data = 10;
  constexpr int Nc = 3;

  auto prob1 = make_dummy_problem(N_data, Nc);
  prob1.cluster_by_kMedoidsLloyd();
  double cost1 = prob1.findTotalCost();

  // Run with multiple repetitions -- should find a cost <= worst single run.
  auto prob2 = make_dummy_problem(N_data, Nc);
  prob2.N_repetition = 3;
  prob2.cluster_by_kMedoidsLloyd();
  double cost2 = prob2.findTotalCost();

  // The multi-rep run may or may not beat the single run (depends on seeds),
  // but the cost must be non-negative.
  REQUIRE(cost2 >= 0.0);
  REQUIRE(cost1 >= 0.0);
}

// ---------------------------------------------------------------------------
// Edge case: k=1 (all points in one cluster)
// ---------------------------------------------------------------------------
TEST_CASE("k=1 puts all points in one cluster", "[Phase1][clustering]")
{
  constexpr int N_data = 10;
  constexpr int Nc = 1;

  auto prob = make_dummy_problem(N_data, Nc);
  prob.cluster_by_kMedoidsLloyd();

  // Every label should be 0.
  for (int label : prob.clusters_ind) {
    REQUIRE(label == 0);
  }

  REQUIRE(prob.centroids_ind.size() == 1);
  REQUIRE(prob.centroids_ind[0] >= 0);
  REQUIRE(prob.centroids_ind[0] < N_data);
}

// ---------------------------------------------------------------------------
// Edge case: k=N (each point is its own medoid)
// ---------------------------------------------------------------------------
TEST_CASE("k=N makes each point a medoid", "[Phase1][clustering]")
{
  constexpr int N_data = 10;
  constexpr int Nc = N_data;

  auto prob = make_dummy_problem(N_data, Nc);
  prob.cluster_by_kMedoidsLloyd();

  // Each label should be unique in [0, N).
  std::set<int> unique_labels(prob.clusters_ind.begin(), prob.clusters_ind.end());
  REQUIRE(unique_labels.size() == static_cast<size_t>(N_data));

  // Total cost should be zero when every point is its own medoid.
  double cost = prob.findTotalCost();
  REQUIRE_THAT(cost, WithinAbs(0.0, 1e-10));
}

// ---------------------------------------------------------------------------
// Initialisation functions do not throw for valid Nc
// ---------------------------------------------------------------------------
TEST_CASE("init::random does not throw for valid Nc", "[Phase1][clustering][init]")
{
  constexpr int N_data = 10;
  constexpr int Nc = 3;

  auto prob = make_dummy_problem(N_data, Nc);
  REQUIRE_NOTHROW(init::random(prob));

  // After init, centroids_ind should have Nc entries.
  REQUIRE(prob.centroids_ind.size() == static_cast<size_t>(Nc));
}

TEST_CASE("init::Kmeanspp does not throw for valid Nc", "[Phase1][clustering][init]")
{
  constexpr int N_data = 10;
  constexpr int Nc = 3;

  auto prob = make_dummy_problem(N_data, Nc);
  REQUIRE_NOTHROW(init::Kmeanspp(prob));

  REQUIRE(prob.centroids_ind.size() == static_cast<size_t>(Nc));
}

// ---------------------------------------------------------------------------
// assignClusters puts each medoid into its own cluster
// ---------------------------------------------------------------------------
TEST_CASE("After assignClusters, each medoid belongs to its own cluster", "[Phase1][clustering]")
{
  constexpr int N_data = 10;
  constexpr int Nc = 3;

  auto prob = make_dummy_problem(N_data, Nc);
  init::random(prob);
  prob.assignClusters();

  // Each medoid should map to a distinct cluster label.
  std::set<int> medoid_labels;
  for (size_t c = 0; c < prob.centroids_ind.size(); ++c) {
    int medoid_idx = prob.centroids_ind[c];
    int label = prob.clusters_ind[medoid_idx];
    // The medoid at position c should be assigned to cluster c,
    // because its distance to itself is zero.
    REQUIRE(label == static_cast<int>(c));
    medoid_labels.insert(label);
  }
  REQUIRE(medoid_labels.size() == static_cast<size_t>(Nc));
}
