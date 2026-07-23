/**
 * @file unit_test_fast_pam.cpp
 * @brief Unit tests for FastPAM1 k-medoids clustering algorithm.
 *
 * @details Tests verify correctness, convergence, and quality of FastPAM
 * against basic invariants and compared to Lloyd-style k-medoids.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <set>
#include <string>
#include <system_error>
#include <vector>

#ifndef DTWC_TEST_DATA_DIR
#define DTWC_TEST_DATA_DIR "./data"
#endif

static struct TestDataInitFP {
  TestDataInitFP() {
    dtwc::settings::paths::setDataPath(DTWC_TEST_DATA_DIR);
    // Route Lloyd/FastPAM CSV output to a per-run temp dir so the test never
    // pollutes the repo root or the build tree (previously `.`, which was
    // CWD-dependent and leaked files when run from the repo root).
    const auto out = std::filesystem::temp_directory_path() / "dtwc_fast_pam_test";
    std::error_code ec;
    std::filesystem::create_directories(out, ec);
    dtwc::settings::paths::setResultsPath(out);
  }
} test_data_init_fp_;

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
// Runs Problem's existing cluster_by_kMedoidsLloyd which is Lloyd iteration.
// ---------------------------------------------------------------------------
static double lloyd_cost(int N, int k)
{
  Problem prob = make_synthetic_problem(N);
  prob.set_numberOfClusters(k);
  prob.N_repetition = 5;
  prob.maxIter = 100;
  prob.fillDistanceMatrix();
  prob.init_fun = init::Kmeanspp;
  prob.cluster_by_kMedoidsLloyd();
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

TEST_CASE("seeded FastPAM BUILD samples proportional to k-median distance",
          "[fast_pam][seeded][initialization]")
{
  // Conditional on first medoid 0, the remaining singleton-series distances
  // are 1 and 3. K-median++ therefore selects point 2 with probability 3/4;
  // incorrectly squaring the weights would move that probability to 9/10.
  Problem prob("fast_pam_d_sampling");
  prob.set_data(Data(std::vector<std::vector<data_t>>{{0.0}, {1.0}, {3.0}},
                     std::vector<std::string>{"zero", "one", "three"}));

  int conditioned = 0;
  int selected_far = 0;
  constexpr std::uint64_t seed_count = 4096;
  for (std::uint64_t seed = 0; seed < seed_count; ++seed) {
    // max_iter=0 observes the deterministic seeded BUILD result before SWAP.
    const auto result = fast_pam_seeded(prob, 2, seed, 0);
    if (result.medoid_indices.front() != 0) continue;
    ++conditioned;
    if (result.medoid_indices.back() == 2) ++selected_far;
  }

  REQUIRE(conditioned > 1000);
  const double far_fraction = static_cast<double>(selected_far) / conditioned;
  CAPTURE(conditioned, selected_far, far_fraction);
  REQUIRE(far_fraction > 0.70);
  REQUIRE(far_fraction < 0.80);
}

TEST_CASE("seeded FastPAM translates negative Soft-DTW sampling weights",
          "[fast_pam][seeded][softdtw]")
{
  Problem prob("fast_pam_softdtw_sampling");
  prob.set_data(Data(
    std::vector<std::vector<data_t>>{
      {0.0, 0.1, 0.0, 0.2}, {0.2, 0.1, 0.3, 0.2},
      {10.0, 10.2, 9.9, 10.1}, {9.8, 10.0, 10.1, 9.9}},
    std::vector<std::string>{"a", "b", "c", "d"}));
  core::DTWVariantParams params;
  params.variant = core::DTWVariant::SoftDTW;
  params.sdtw_gamma = 0.7;
  prob.set_variant(params);

  // Raw Soft-DTW may be negative off diagonal. That is valid objective input,
  // but cannot be passed directly to a weighted random sampler.
  REQUIRE(prob.dist_by_ind(0, 1) < 0.0);
  const auto result = fast_pam_seeded(prob, 2, 42, 20);
  CHECK(result.labels.size() == 4);
  CHECK(result.medoid_indices.size() == 2);
  CHECK(std::isfinite(result.total_cost));
}

TEST_CASE("seeded FastPAM completes the medoid set when all weights are zero",
          "[fast_pam][seeded][degenerate]")
{
  // Exercises dtwc::fast_pam_seeded. Identical series drive every BUILD sampling
  // weight to exactly zero, so core::distance_sampling_weights returns
  // total == 0 and fast_pam.cpp:495 must complete the distinct medoid set
  // deterministically instead of constructing an invalid weighted distribution.
  // The k-means++ siblings of this case are in unit_test_clustering_algorithms.cpp.
  Problem prob("fast_pam_identical_series");
  prob.set_data(Data(
    std::vector<std::vector<data_t>>{
      {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0},
      {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}},
    std::vector<std::string>{"a", "b", "c", "d"}));

  // max_iter=0 observes the deterministic seeded BUILD result before SWAP.
  const auto result = fast_pam_seeded(prob, 3, 7, 0);

  REQUIRE(result.medoid_indices.size() == 3);
  const std::set<int> unique(result.medoid_indices.begin(),
                             result.medoid_indices.end());
  CHECK(unique.size() == 3);
  // The fallback fills upward from index 0, so whichever index the seed drew
  // first, both 0 and 1 must be completed into the set. This holds for all four
  // possible first draws and is therefore independent of the seed.
  CHECK(unique.count(0) == 1);
  CHECK(unique.count(1) == 1);
  CHECK(result.total_cost == 0.0);
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
    recomputed_cost += prob.dist_by_ind(p, medoid);
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
// Test 11: 2.0 (Task 1.6) write-back — fast_pam stores the result INTO prob.
// (1.x asserted non-mutation; API contract §2.5 moves the binding auto-wire into
//  core, so pure-C++ users get prob.centroids_ind/clusters_ind/n_clusters set.)
// ===========================================================================
TEST_CASE("FastPAM writes result back into Problem", "[fast_pam][write_back]")
{
  constexpr int N = 8;
  constexpr int k = 2;

  Problem prob = make_small_problem(N);

  // Set up some unrelated initial state on prob (k=3) to prove fast_pam overwrites it.
  prob.set_n_clusters(3);

  auto result = fast_pam(prob, k);

  // prob now holds the fast_pam result (labels/medoids/k), with NO manual wiring.
  REQUIRE(prob.n_clusters() == k);
  REQUIRE(prob.centroids_ind == result.medoid_indices);
  REQUIRE(prob.clusters_ind == result.labels);
  REQUIRE(static_cast<int>(prob.centroids_ind.size()) == k);
  REQUIRE(static_cast<int>(prob.clusters_ind.size()) == N);
}
