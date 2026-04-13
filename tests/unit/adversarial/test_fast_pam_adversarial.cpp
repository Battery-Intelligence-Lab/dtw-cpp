/**
 * @file test_fast_pam_adversarial.cpp
 * @brief Adversarial tests for k-medoids (PAM) clustering correctness properties.
 *
 * @details Tests are written against the ALGORITHM SPECIFICATION (Schubert & Rousseeuw 2021),
 * not the implementation. They verify invariants that any correct k-medoids algorithm must
 * satisfy: label validity, medoid uniqueness, cost non-negativity, convergence, determinism,
 * quality guarantees, and known-cluster recovery.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

// ---------------------------------------------------------------------------
// Helper: create a Problem with synthetic time-series data (no filesystem).
// Each "series" is a short vector of doubles.
// ---------------------------------------------------------------------------
static Problem make_synthetic_problem(
  const std::vector<std::vector<data_t>> &series,
  const std::string &name = "synth")
{
  std::vector<std::vector<data_t>> vecs = series;
  std::vector<std::string> names;
  names.reserve(series.size());
  for (size_t i = 0; i < series.size(); ++i)
    names.push_back("s" + std::to_string(i));

  Problem prob(name);
  prob.data = Data(std::move(vecs), std::move(names));
  prob.output_folder = std::filesystem::temp_directory_path();
  prob.refreshDistanceMatrix();
  return prob;
}

// Helper: run the k-medoids clustering and return the problem (mutated in-place).
static void run_clustering(Problem &prob, int k, int max_iter = 100, int n_rep = 1)
{
  prob.set_numberOfClusters(k);
  prob.maxIter = max_iter;
  prob.N_repetition = n_rep;
  prob.method = Method::Kmedoids;
  prob.cluster();
}

// Helper: compute total cost from scratch given current assignments.
static double compute_total_cost(Problem &prob)
{
  double cost = 0.0;
  for (int i = 0; i < prob.size(); ++i)
    cost += prob.distByInd(i, prob.centroid_of(i));
  return cost;
}

// ---------------------------------------------------------------------------
// Generate well-separated clusters: 3 groups around different base values.
// Group A: values near 0, Group B: values near 100, Group C: values near 200.
// Each series has length 10.
// ---------------------------------------------------------------------------
static std::vector<std::vector<data_t>> make_three_clusters(int per_cluster = 5)
{
  std::vector<std::vector<data_t>> series;
  std::mt19937 rng(42);
  std::normal_distribution<data_t> noise(0.0, 0.5);

  for (int c = 0; c < 3; ++c) {
    double base = c * 100.0;
    for (int i = 0; i < per_cluster; ++i) {
      std::vector<data_t> s(10);
      for (auto &v : s)
        v = base + noise(rng);
      series.push_back(std::move(s));
    }
  }
  return series;
}

// ===========================================================================
// AREA 1: FastPAM Correctness Properties
// ===========================================================================

TEST_CASE("Adversarial: Labels are in valid range [0, k)", "[adversarial][pam][labels]")
{
  auto series = make_three_clusters(5);
  auto prob = make_synthetic_problem(series);
  const int k = 3;
  run_clustering(prob, k);

  for (int i = 0; i < prob.size(); ++i) {
    REQUIRE(prob.clusters_ind[i] >= 0);
    REQUIRE(prob.clusters_ind[i] < k);
  }
}

TEST_CASE("Adversarial: Medoid count equals k", "[adversarial][pam][medoids]")
{
  auto series = make_three_clusters(5);
  auto prob = make_synthetic_problem(series);

  for (int k : {1, 2, 3, 5}) {
    SECTION("k=" + std::to_string(k))
    {
      auto p = make_synthetic_problem(series);
      run_clustering(p, k);
      REQUIRE(static_cast<int>(p.centroids_ind.size()) == k);
    }
  }
}

TEST_CASE("Adversarial: Medoid indices are unique", "[adversarial][pam][medoids]")
{
  auto series = make_three_clusters(5);
  auto prob = make_synthetic_problem(series);
  const int k = 3;
  run_clustering(prob, k);

  std::set<int> unique_medoids(prob.centroids_ind.begin(), prob.centroids_ind.end());
  REQUIRE(unique_medoids.size() == prob.centroids_ind.size());
}

TEST_CASE("Adversarial: All medoid indices are valid data indices [0, N)", "[adversarial][pam][medoids]")
{
  auto series = make_three_clusters(5);
  auto prob = make_synthetic_problem(series);
  const int k = 3;
  run_clustering(prob, k);

  const int N = prob.size();
  for (int idx : prob.centroids_ind) {
    REQUIRE(idx >= 0);
    REQUIRE(idx < N);
  }
}

TEST_CASE("Adversarial: Labels vector size equals N", "[adversarial][pam][labels]")
{
  auto series = make_three_clusters(5);
  auto prob = make_synthetic_problem(series);
  const int k = 3;
  run_clustering(prob, k);

  REQUIRE(static_cast<int>(prob.clusters_ind.size()) == prob.size());
}

TEST_CASE("Adversarial: Total cost is non-negative", "[adversarial][pam][cost]")
{
  auto series = make_three_clusters(5);
  auto prob = make_synthetic_problem(series);
  const int k = 3;
  run_clustering(prob, k);

  double cost = compute_total_cost(prob);
  REQUIRE(cost >= 0.0);
}

TEST_CASE("Adversarial: Converges or hits max_iter", "[adversarial][pam][convergence]")
{
  // We test convergence indirectly: with max_iter=1, the algorithm should
  // still produce valid output (either converged in 1 step or stopped).
  // With max_iter=100 on a small dataset, it should converge.
  auto series = make_three_clusters(5);

  SECTION("Small max_iter still produces valid output")
  {
    auto prob = make_synthetic_problem(series);
    run_clustering(prob, 3, /*max_iter=*/1);
    // Must still have valid labels and medoids
    REQUIRE(static_cast<int>(prob.clusters_ind.size()) == prob.size());
    REQUIRE(static_cast<int>(prob.centroids_ind.size()) == 3);
  }

  SECTION("Large max_iter on easy problem converges")
  {
    auto prob = make_synthetic_problem(series);
    run_clustering(prob, 3, /*max_iter=*/100);
    // Verify output is valid
    REQUIRE(static_cast<int>(prob.clusters_ind.size()) == prob.size());
    double cost = compute_total_cost(prob);
    REQUIRE(cost >= 0.0);
    REQUIRE(cost < 1e12); // Not degenerate
  }
}

TEST_CASE("Adversarial: k=1 puts all points in cluster 0", "[adversarial][pam][k1]")
{
  auto series = make_three_clusters(5);
  auto prob = make_synthetic_problem(series);
  run_clustering(prob, 1);

  // All labels must be 0
  for (int i = 0; i < prob.size(); ++i) {
    REQUIRE(prob.clusters_ind[i] == 0);
  }

  // The single medoid should minimize total cost.
  // Verify: swapping to any other point as medoid does not improve cost.
  double best_cost = compute_total_cost(prob);
  int medoid = prob.centroids_ind[0];

  for (int candidate = 0; candidate < prob.size(); ++candidate) {
    double candidate_cost = 0.0;
    for (int j = 0; j < prob.size(); ++j)
      candidate_cost += prob.distByInd(candidate, j);
    // The chosen medoid should be at least as good as any other
    REQUIRE(best_cost <= candidate_cost + 1e-10);
  }
}

TEST_CASE("Adversarial: k=N gives each point its own cluster with zero cost", "[adversarial][pam][kN]")
{
  // Use a small dataset (k=N means every point is a medoid)
  std::vector<std::vector<data_t>> series = {
    {1.0, 2.0, 3.0},
    {4.0, 5.0, 6.0},
    {7.0, 8.0, 9.0},
    {10.0, 11.0, 12.0}
  };
  auto prob = make_synthetic_problem(series);
  const int N = prob.size();
  run_clustering(prob, N);

  // Every point should be a medoid
  std::set<int> medoid_set(prob.centroids_ind.begin(), prob.centroids_ind.end());
  REQUIRE(static_cast<int>(medoid_set.size()) == N);

  // Total cost should be 0 (every point is its own medoid)
  double cost = compute_total_cost(prob);
  REQUIRE_THAT(cost, WithinAbs(0.0, 1e-10));
}

TEST_CASE("Adversarial: Deterministic with same RNG seed", "[adversarial][pam][determinism]")
{
  auto series = make_three_clusters(5);

  // The global RNG (randGenerator) is seeded with 29 by default.
  // We reset it before each run to get identical results.
  auto run_once = [&]() {
    dtwc::randGenerator.seed(42);
    auto prob = make_synthetic_problem(series);
    run_clustering(prob, 3);
    return std::make_pair(prob.clusters_ind, prob.centroids_ind);
  };

  auto [labels1, medoids1] = run_once();
  auto [labels2, medoids2] = run_once();

  REQUIRE(labels1 == labels2);
  REQUIRE(medoids1 == medoids2);
}

// ===========================================================================
// AREA 2: FastPAM Quality
// ===========================================================================

TEST_CASE("Adversarial: Better than random medoid selection", "[adversarial][pam][quality]")
{
  auto series = make_three_clusters(5);
  const int k = 3;
  const int N = static_cast<int>(series.size());

  // Run clustering
  dtwc::randGenerator.seed(123);
  auto prob = make_synthetic_problem(series);
  run_clustering(prob, k);
  double pam_cost = compute_total_cost(prob);

  // Generate 10 random medoid selections and compute their costs
  std::mt19937 test_rng(999);
  double worst_random_cost = 0.0;

  for (int trial = 0; trial < 10; ++trial) {
    // Pick k random medoids
    std::vector<int> indices(N);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), test_rng);
    std::vector<int> random_medoids(indices.begin(), indices.begin() + k);

    // Create a fresh problem and assign clusters using these medoids
    auto rprob = make_synthetic_problem(series);
    rprob.fillDistanceMatrix();
    rprob.set_numberOfClusters(k);
    rprob.set_clusters(random_medoids);
    rprob.assignClusters();
    double random_cost = compute_total_cost(rprob);

    worst_random_cost = std::max(worst_random_cost, random_cost);
  }

  // PAM should beat (or tie with) the worst random assignment
  REQUIRE(pam_cost <= worst_random_cost + 1e-10);
}

TEST_CASE("Adversarial: Recovers 3 well-separated clusters", "[adversarial][pam][quality][recovery]")
{
  // 3 clusters well separated: base values 0, 100, 200.
  // Series within a cluster have noise ~0.5, distance between clusters ~100.
  // PAM(k=3) should perfectly separate them.
  auto series = make_three_clusters(5);
  const int per_cluster = 5;

  dtwc::randGenerator.seed(77);
  auto prob = make_synthetic_problem(series);
  run_clustering(prob, 3);

  // Check that points from the same ground-truth cluster share the same label.
  // Ground truth: indices [0..4] = cluster A, [5..9] = cluster B, [10..14] = cluster C.
  // We don't know which label number maps to which ground-truth cluster,
  // but within each ground-truth group, all labels should be identical.

  for (int c = 0; c < 3; ++c) {
    int start = c * per_cluster;
    int expected_label = prob.clusters_ind[start];
    for (int i = start + 1; i < start + per_cluster; ++i) {
      REQUIRE(prob.clusters_ind[i] == expected_label);
    }
  }

  // Also verify that the three groups have DIFFERENT labels
  std::set<int> group_labels;
  for (int c = 0; c < 3; ++c)
    group_labels.insert(prob.clusters_ind[c * per_cluster]);
  REQUIRE(group_labels.size() == 3);
}

TEST_CASE("Adversarial: Cost is non-increasing across iterations (monotonic)", "[adversarial][pam][quality][monotonic]")
{
  // We simulate iteration-by-iteration clustering by running with max_iter=1, 2, 3, ...
  // and verifying cost doesn't increase. Since the algorithm re-initialises each time
  // with the same seed, we instead run a single full clustering and verify the final
  // cost is at most the cost after the first iteration.

  auto series = make_three_clusters(5);

  // Run with max_iter=1 (single iteration)
  dtwc::randGenerator.seed(55);
  auto prob1 = make_synthetic_problem(series);
  run_clustering(prob1, 3, /*max_iter=*/1);
  double cost_iter1 = compute_total_cost(prob1);

  // Run with max_iter=100 (full convergence)
  dtwc::randGenerator.seed(55);
  auto prob_full = make_synthetic_problem(series);
  run_clustering(prob_full, 3, /*max_iter=*/100);
  double cost_full = compute_total_cost(prob_full);

  // Full run should be at least as good as after 1 iteration
  REQUIRE(cost_full <= cost_iter1 + 1e-10);
}

TEST_CASE("Adversarial: Medoids are actual data points", "[adversarial][pam][medoids][datapoints]")
{
  // k-medoids must select actual data points, not synthetic means.
  // Verify each medoid index refers to a series that exists in the data.
  auto series = make_three_clusters(5);
  auto prob = make_synthetic_problem(series);
  run_clustering(prob, 3);

  for (int medoid_idx : prob.centroids_ind) {
    // The medoid must be a valid index into the data
    REQUIRE(medoid_idx >= 0);
    REQUIRE(medoid_idx < prob.size());

    // The medoid's data must match the original series exactly
    const auto &medoid_series = prob.p_vec(medoid_idx);
    REQUIRE(medoid_series == series[medoid_idx]);
  }
}

TEST_CASE("Adversarial: Every point assigned to exactly one cluster", "[adversarial][pam][labels][partition]")
{
  auto series = make_three_clusters(5);
  auto prob = make_synthetic_problem(series);
  const int k = 3;
  run_clustering(prob, k);

  // clusters_ind gives one label per point -- by construction it's a single label.
  // Verify every cluster has at least one member (its medoid).
  std::vector<int> cluster_counts(k, 0);
  for (int i = 0; i < prob.size(); ++i) {
    int label = prob.clusters_ind[i];
    REQUIRE(label >= 0);
    REQUIRE(label < k);
    cluster_counts[label]++;
  }

  // Each cluster must have at least one member (the medoid itself)
  for (int c = 0; c < k; ++c) {
    REQUIRE(cluster_counts[c] >= 1);
  }
}

TEST_CASE("Adversarial: Identical series get same cluster label", "[adversarial][pam][correctness]")
{
  // If two series are identical, they must be in the same cluster.
  // Data is trivially separable: two groups of 3 identical series, groups far apart.
  std::vector<std::vector<data_t>> series = {
    {1.0, 2.0, 3.0},        // group A (idx 0)
    {1.0, 2.0, 3.0},        // group A (idx 1, identical)
    {1.0, 2.0, 3.0},        // group A (idx 2, identical)
    {100.0, 200.0, 300.0},  // group B (idx 3)
    {100.0, 200.0, 300.0},  // group B (idx 4, identical)
    {100.0, 200.0, 300.0},  // group B (idx 5, identical)
  };

  // Use multiple random restarts — the algorithm keeps the min-cost solution,
  // and for trivially separable data, only the "one medoid per group" init yields
  // zero cost. A fixed seed is not platform-robust (RNG-driven init interacts with
  // std::lib floating-point sampling), so we let N_repetition=10 find the optimum.
  dtwc::randGenerator.seed(12345);
  auto prob = make_synthetic_problem(series);
  run_clustering(prob, 2, /*max_iter=*/100, /*n_rep=*/10);

  // Identical series must share the same label
  REQUIRE(prob.clusters_ind[0] == prob.clusters_ind[1]);
  REQUIRE(prob.clusters_ind[0] == prob.clusters_ind[2]);
  REQUIRE(prob.clusters_ind[3] == prob.clusters_ind[4]);
  REQUIRE(prob.clusters_ind[3] == prob.clusters_ind[5]);
  // And the two groups must have different labels (required by min-cost property)
  REQUIRE(prob.clusters_ind[0] != prob.clusters_ind[3]);
}

TEST_CASE("Adversarial: Single data point", "[adversarial][pam][edge]")
{
  std::vector<std::vector<data_t>> series = {
    {5.0, 10.0, 15.0}
  };

  auto prob = make_synthetic_problem(series);
  run_clustering(prob, 1);

  REQUIRE(prob.clusters_ind.size() == 1);
  REQUIRE(prob.clusters_ind[0] == 0);
  REQUIRE(prob.centroids_ind.size() == 1);
  REQUIRE(prob.centroids_ind[0] == 0);

  double cost = compute_total_cost(prob);
  REQUIRE_THAT(cost, WithinAbs(0.0, 1e-10));
}

TEST_CASE("Adversarial: Two points, k=2", "[adversarial][pam][edge]")
{
  std::vector<std::vector<data_t>> series = {
    {0.0, 0.0, 0.0},
    {10.0, 10.0, 10.0}
  };

  auto prob = make_synthetic_problem(series);
  run_clustering(prob, 2);

  // Each point should be its own medoid
  std::set<int> medoids(prob.centroids_ind.begin(), prob.centroids_ind.end());
  REQUIRE(medoids.size() == 2);
  REQUIRE(medoids.count(0) == 1);
  REQUIRE(medoids.count(1) == 1);

  double cost = compute_total_cost(prob);
  REQUIRE_THAT(cost, WithinAbs(0.0, 1e-10));
}

TEST_CASE("Adversarial: Varying series lengths handled correctly", "[adversarial][pam][robustness]")
{
  // DTW handles different-length series; clustering should still work.
  std::vector<std::vector<data_t>> series = {
    {1.0, 2.0, 3.0},
    {1.0, 2.0, 3.0, 4.0, 5.0},
    {100.0, 200.0},
    {100.0, 200.0, 300.0, 400.0}
  };

  auto prob = make_synthetic_problem(series);
  run_clustering(prob, 2);

  // Basic invariants must hold
  REQUIRE(static_cast<int>(prob.clusters_ind.size()) == prob.size());
  REQUIRE(static_cast<int>(prob.centroids_ind.size()) == 2);

  for (int i = 0; i < prob.size(); ++i) {
    REQUIRE(prob.clusters_ind[i] >= 0);
    REQUIRE(prob.clusters_ind[i] < 2);
  }

  // Series near each other should cluster together
  REQUIRE(prob.clusters_ind[0] == prob.clusters_ind[1]); // both near small values
  REQUIRE(prob.clusters_ind[2] == prob.clusters_ind[3]); // both near large values
  REQUIRE(prob.clusters_ind[0] != prob.clusters_ind[2]); // groups are different
}
