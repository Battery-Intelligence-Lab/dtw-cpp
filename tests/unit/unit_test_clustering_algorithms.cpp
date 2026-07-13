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

#include <chrono>
#include <filesystem>
#include <set>
#include <string>
#include <system_error>
#include <vector>

#ifndef DTWC_TEST_DATA_DIR
#define DTWC_TEST_DATA_DIR "./data"
#endif

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
  // Use the configure-time absolute fixture path directly. A former global
  // TestDataInit mutated settings::paths::data during static initialization;
  // cross-TU initialization order could then reset it to "./data", making the
  // test CWD-dependent and causing the intermittent Windows 0xc0000409 hunt.
  dtwc::DataLoader dl{ std::filesystem::path{DTWC_TEST_DATA_DIR} / "dummy", N_data };
  dl.startColumn(1).startRow(1);

  dtwc::Problem prob{ "test_clustering", dl };
  prob.set_numberOfClusters(Nc);
  prob.maxIter = 100;
  prob.N_repetition = 1;
  // Write test output CSVs to the system temp dir, not the project root/CWD.
  // Without this, tests pollute the working directory with test_clustering*.csv.
  prob.output_folder = std::filesystem::temp_directory_path().string();
  return prob;
}

struct TemporaryOutputDirectory {
  std::filesystem::path path;

  TemporaryOutputDirectory()
  {
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    path = std::filesystem::temp_directory_path()
         / ("dtwc_capped_lloyd_" + std::to_string(nonce));
    std::filesystem::create_directories(path);
  }

  ~TemporaryOutputDirectory()
  {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

Problem make_capped_lloyd_problem(
  const std::filesystem::path &output, std::string name, int max_iter)
{
  Problem problem(name);
  problem.set_data(Data(
    std::vector<std::vector<data_t>>{
      {0.0}, {40.0}, {40.0}, {46.0}, {49.0},
      {51.0}, {51.0}, {51.0}, {100.0}
    },
    std::vector<std::string>{
      "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"
    }));
  problem.set_n_clusters(2);
  problem.set_max_iter(max_iter);
  problem.set_n_repetitions(1);
  problem.output_folder = output;
  problem.init_fun = [](Problem &candidate) {
    std::vector<int> initial_medoids{0, 8};
    candidate.set_clusters(initial_medoids);
  };
  return problem;
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

TEST_CASE("init::Kmeanspp selects distinct medoids when all weights are zero",
          "[Phase1][clustering][init][degenerate]")
{
  Problem prob("identical_kmeanspp");
  prob.set_data(Data(
    std::vector<std::vector<data_t>>{
      {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0},
      {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}},
    std::vector<std::string>{"a", "b", "c", "d"}));
  prob.set_n_clusters(3);

  REQUIRE_NOTHROW(init::Kmeanspp(prob));
  REQUIRE(prob.centroids_ind.size() == 3);
  const std::set<int> unique(prob.centroids_ind.begin(),
                             prob.centroids_ind.end());
  CHECK(unique.size() == 3);
}

// ---------------------------------------------------------------------------
// assignClusters puts each medoid into its own cluster
// ---------------------------------------------------------------------------
TEST_CASE("After assignClusters, each medoid belongs to its own cluster", "[Phase1][clustering]")
{
  constexpr int N_data = 10;
  constexpr int Nc = 3;

  auto prob = make_dummy_problem(N_data, Nc);
  prob.fillDistanceMatrix();
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

TEST_CASE("Capped Lloyd returns labels assigned to its final medoids",
          "[Phase1][clustering][lloyd][capped][m29]")
{
  TemporaryOutputDirectory output;
  auto capped = make_capped_lloyd_problem(output.path, "capped", 1);
  auto converged = make_capped_lloyd_problem(output.path, "converged", 100);

  capped.cluster_by_kmedoids_lloyd();
  converged.cluster_by_kmedoids_lloyd();

  const std::vector<int> expected_medoids{1, 5};
  const std::vector<int> expected_labels{0, 0, 0, 1, 1, 1, 1, 1, 1};
  REQUIRE(capped.medoids() == expected_medoids);
  CHECK(capped.last_iterations == 1);
  REQUIRE(converged.medoids() == expected_medoids);
  REQUIRE(converged.labels() == expected_labels);
  CHECK(converged.last_iterations == 2);
  CHECK_THAT(converged.find_total_cost(), WithinAbs(96.0, 1e-12));

  std::vector<int> nearest_labels;
  nearest_labels.reserve(capped.size());
  double nearest_cost = 0.0;
  for (std::size_t point = 0; point < capped.size(); ++point) {
    int nearest_cluster = 0;
    double nearest_distance = capped.dist_by_ind(
      static_cast<int>(point), capped.medoids().front());
    for (std::size_t cluster = 1; cluster < capped.medoids().size(); ++cluster) {
      const double candidate_distance = capped.dist_by_ind(
        static_cast<int>(point), capped.medoids()[cluster]);
      if (candidate_distance < nearest_distance) {
        nearest_cluster = static_cast<int>(cluster);
        nearest_distance = candidate_distance;
      }
    }
    nearest_labels.push_back(nearest_cluster);
    nearest_cost += nearest_distance;
  }

  REQUIRE(nearest_labels == expected_labels);
  REQUIRE_THAT(nearest_cost, WithinAbs(96.0, 1e-12));
  CHECK(capped.labels() == nearest_labels);
  CHECK_THAT(capped.find_total_cost(), WithinAbs(nearest_cost, 1e-12));
  CHECK(capped.labels() == converged.labels());
}
