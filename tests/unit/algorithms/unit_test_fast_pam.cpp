/**
 * @file unit_test_fast_pam.cpp
 * @brief Unit tests for FastPAM1 k-medoids clustering algorithm.
 *
 * @details Tests verify correctness, convergence, and quality of FastPAM
 * against basic invariants and compared to Lloyd-style k-medoids.
 *
 * @date 28 Mar 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <set>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

// ---------------------------------------------------------------------------
// Helper: build a Problem with N synthetic time series.
// Creates distinct patterns so clustering is meaningful.
// ---------------------------------------------------------------------------
static Problem make_synthetic_problem(int N)
{
  std::vector<std::vector<data_t>> vecs;
  std::vector<std::string> names;

  for (int i = 0; i < N; ++i) {
    // Create time series with group structure:
    //   Group 0: i in [0, N/3)       -> low baseline, gentle slope
    //   Group 1: i in [N/3, 2*N/3)   -> medium baseline, steeper slope
    //   Group 2: i in [2*N/3, N)     -> high baseline, negative slope
    int group = (i * 3) / N;
    double baseline = group * 50.0;
    double slope = (group == 2) ? -2.0 : (group + 1) * 1.5;
    double noise_offset = i * 0.1;  // deterministic per-series offset

    std::vector<data_t> ts;
    int len = 20 + (i % 5);  // varying lengths
    for (int j = 0; j < len; ++j) {
      ts.push_back(baseline + slope * j + noise_offset);
    }
    vecs.push_back(std::move(ts));
    names.push_back("ts_" + std::to_string(i));
  }

  Data data(std::move(vecs), std::move(names));
  Problem prob("fast_pam_test");
  prob.set_data(std::move(data));
  return prob;
}

// ---------------------------------------------------------------------------
// Helper: build a small Problem for edge-case tests.
// ---------------------------------------------------------------------------
static Problem make_small_problem(int N = 5)
{
  std::vector<std::vector<data_t>> vecs;
  std::vector<std::string> names;

  for (int i = 0; i < N; ++i) {
    std::vector<data_t> ts;
    for (int j = 0; j <= i + 2; ++j) {
      ts.push_back(static_cast<data_t>(i * 10 + j));
    }
    vecs.push_back(std::move(ts));
    names.push_back("ts_" + std::to_string(i));
  }

  Data data(std::move(vecs), std::move(names));
  Problem prob("fast_pam_small");
  prob.set_data(std::move(data));
  return prob;
}

// ---------------------------------------------------------------------------
// Helper: compute Lloyd-style cost for comparison.
// Runs Problem's existing cluster_by_kMedoidsPAM which is Lloyd iteration.
// ---------------------------------------------------------------------------
static double lloyd_cost(int N, int k)
{
  Problem prob = make_synthetic_problem(N);
  prob.set_numberOfClusters(k);
  prob.N_repetition = 5;
  prob.maxIter = 100;
  prob.fillDistanceMatrix();
  prob.init_fun = init::Kmeanspp;
  prob.cluster_by_kMedoidsPAM();
  return prob.findTotalCost();
}


// ===========================================================================
// Test 1: FastPAM converges on synthetic data.
// ===========================================================================
TEST_CASE("FastPAM converges on synthetic data", "[fast_pam][convergence]")
{
  constexpr int N = 15;
  constexpr int k = 3;

  Problem prob = make_synthetic_problem(N);
  auto result = fast_pam(prob, k);

  REQUIRE(result.converged);
  REQUIRE(result.iterations < 100);
  REQUIRE(result.total_cost > 0.0);
}

// ===========================================================================
// Test 2: FastPAM finds cost <= Lloyd iteration.
// ===========================================================================
TEST_CASE("FastPAM cost <= Lloyd cost", "[fast_pam][quality]")
{
  constexpr int N = 15;
  constexpr int k = 3;

  // FastPAM uses the same K-means++ init, and both use the same global RNG.
  // To make a fair comparison, we reset the RNG seed before each.
  dtwc::randGenerator.seed(42);
  Problem prob_fp = make_synthetic_problem(N);
  double fp_cost = fast_pam(prob_fp, k).total_cost;

  dtwc::randGenerator.seed(42);
  double ll_cost = lloyd_cost(N, k);

  // FastPAM should find an equal or better (lower) cost than Lloyd.
  // Allow a small tolerance for floating-point arithmetic.
  REQUIRE(fp_cost <= ll_cost + 1e-9);
}

// ===========================================================================
// Test 3: Medoid indices are valid.
// ===========================================================================
TEST_CASE("FastPAM medoid indices are valid", "[fast_pam][medoids]")
{
  constexpr int N = 12;
  constexpr int k = 3;

  Problem prob = make_synthetic_problem(N);
  auto result = fast_pam(prob, k);

  REQUIRE(result.medoid_indices.size() == static_cast<size_t>(k));

  // All medoid indices must be in [0, N).
  for (int m : result.medoid_indices) {
    REQUIRE(m >= 0);
    REQUIRE(m < N);
  }

  // All medoid indices must be distinct.
  std::set<int> unique_medoids(result.medoid_indices.begin(), result.medoid_indices.end());
  REQUIRE(unique_medoids.size() == static_cast<size_t>(k));
}

// ===========================================================================
// Test 4: Labels are in [0, k).
// ===========================================================================
TEST_CASE("FastPAM labels are in valid range", "[fast_pam][labels]")
{
  constexpr int N = 12;
  constexpr int k = 3;

  Problem prob = make_synthetic_problem(N);
  auto result = fast_pam(prob, k);

  REQUIRE(result.labels.size() == static_cast<size_t>(N));

  for (int label : result.labels) {
    REQUIRE(label >= 0);
    REQUIRE(label < k);
  }
}

// ===========================================================================
// Test 5: Convergence flag is set when iterations < max_iter.
// ===========================================================================
TEST_CASE("FastPAM convergence flag matches iteration count", "[fast_pam][convergence_flag]")
{
  constexpr int N = 10;
  constexpr int k = 2;

  Problem prob = make_small_problem(N);
  auto result = fast_pam(prob, k, 200);

  if (result.converged) {
    REQUIRE(result.iterations < 200);
  } else {
    REQUIRE(result.iterations == 200);
  }
}

// ===========================================================================
// Test 6: k=1 produces a single cluster with all points.
// ===========================================================================
TEST_CASE("FastPAM k=1 assigns all points to one cluster", "[fast_pam][k1]")
{
  constexpr int N = 8;
  constexpr int k = 1;

  Problem prob = make_small_problem(N);
  auto result = fast_pam(prob, k);

  REQUIRE(result.medoid_indices.size() == 1);
  REQUIRE(result.labels.size() == static_cast<size_t>(N));

  for (int label : result.labels) {
    REQUIRE(label == 0);
  }

  REQUIRE(result.converged);
}

// ===========================================================================
// Test 7: k=N makes every point a medoid.
// ===========================================================================
TEST_CASE("FastPAM k=N makes every point a medoid", "[fast_pam][kN]")
{
  constexpr int N = 5;
  constexpr int k = N;

  Problem prob = make_small_problem(N);
  auto result = fast_pam(prob, k);

  REQUIRE(result.medoid_indices.size() == static_cast<size_t>(N));
  REQUIRE_THAT(result.total_cost, WithinAbs(0.0, 1e-10));

  // Every point should be a medoid.
  std::set<int> medoid_set(result.medoid_indices.begin(), result.medoid_indices.end());
  REQUIRE(medoid_set.size() == static_cast<size_t>(N));
}

// ===========================================================================
// Test 8: Each medoid is assigned to its own cluster.
// ===========================================================================
TEST_CASE("FastPAM medoids are assigned to their own cluster", "[fast_pam][self_assignment]")
{
  constexpr int N = 12;
  constexpr int k = 3;

  Problem prob = make_synthetic_problem(N);
  auto result = fast_pam(prob, k);

  // For each medoid m at index medoid_indices[c], its label should be c.
  for (int c = 0; c < k; ++c) {
    int medoid_point = result.medoid_indices[c];
    REQUIRE(result.labels[medoid_point] == c);
  }
}

// ===========================================================================
// Test 9: Total cost matches sum of nearest-medoid distances.
// ===========================================================================
TEST_CASE("FastPAM total_cost matches recomputed cost", "[fast_pam][cost_consistency]")
{
  constexpr int N = 10;
  constexpr int k = 2;

  Problem prob = make_small_problem(N);
  auto result = fast_pam(prob, k);

  // Recompute total cost from labels and medoid_indices.
  double recomputed_cost = 0.0;
  for (int p = 0; p < N; ++p) {
    int medoid = result.medoid_indices[result.labels[p]];
    recomputed_cost += prob.distByInd(p, medoid);
  }

  REQUIRE_THAT(result.total_cost, WithinAbs(recomputed_cost, 1e-10));
}

// ===========================================================================
// Test 10: Invalid inputs throw.
// ===========================================================================
TEST_CASE("FastPAM throws on invalid inputs", "[fast_pam][errors]")
{
  SECTION("k = 0 throws")
  {
    Problem prob = make_small_problem(5);
    REQUIRE_THROWS_AS(fast_pam(prob, 0), std::runtime_error);
  }

  SECTION("k > N throws")
  {
    Problem prob = make_small_problem(5);
    REQUIRE_THROWS_AS(fast_pam(prob, 10), std::runtime_error);
  }

  SECTION("empty problem throws")
  {
    Problem prob("empty");
    REQUIRE_THROWS_AS(fast_pam(prob, 1), std::runtime_error);
  }
}

// ===========================================================================
// Test 11: Does not modify Problem's centroids_ind or clusters_ind.
// ===========================================================================
TEST_CASE("FastPAM does not modify Problem state", "[fast_pam][no_side_effects]")
{
  constexpr int N = 8;
  constexpr int k = 2;

  Problem prob = make_small_problem(N);

  // Set up some initial state on prob.
  prob.set_numberOfClusters(3);
  auto orig_centroids = prob.centroids_ind;
  auto orig_clusters = prob.clusters_ind;

  auto result = fast_pam(prob, k);

  // prob's internal state should be restored.
  REQUIRE(prob.centroids_ind == orig_centroids);
  REQUIRE(prob.clusters_ind == orig_clusters);
  REQUIRE(prob.cluster_size() == 3);
}
