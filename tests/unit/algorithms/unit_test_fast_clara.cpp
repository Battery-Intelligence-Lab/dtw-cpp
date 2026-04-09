/**
 * @file unit_test_fast_clara.cpp
 * @brief Unit tests for FastCLARA scalable k-medoids clustering algorithm.
 *
 * @details Tests verify correctness, reproducibility, edge cases, and quality
 * of FastCLARA compared to single-subsample PAM.
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 */

#include <dtwc.hpp>
#include <algorithms/fast_clara.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <limits>
#include <set>
#include <string>
#include <vector>

#ifndef DTWC_TEST_DATA_DIR
#define DTWC_TEST_DATA_DIR "./data"
#endif

static struct TestDataInitCLARA {
  TestDataInitCLARA() {
    dtwc::settings::paths::setDataPath(DTWC_TEST_DATA_DIR);
    dtwc::settings::paths::setResultsPath(".");
  }
} test_data_init_clara_;

using Catch::Matchers::WithinAbs;
using namespace dtwc;

// ---------------------------------------------------------------------------
// Helper: build a Problem with N synthetic time series (3 groups).
// ---------------------------------------------------------------------------
static Problem make_clara_problem(int N)
{
  std::vector<std::vector<data_t>> vecs;
  std::vector<std::string> names;

  for (int i = 0; i < N; ++i) {
    int group = (i * 3) / N;
    double baseline = group * 50.0;
    double slope = (group == 2) ? -2.0 : (group + 1) * 1.5;
    double noise_offset = i * 0.1;

    std::vector<data_t> ts;
    int len = 20 + (i % 5);
    for (int j = 0; j < len; ++j) {
      ts.push_back(baseline + slope * j + noise_offset);
    }
    vecs.push_back(std::move(ts));
    names.push_back("ts_" + std::to_string(i));
  }

  Data data(std::move(vecs), std::move(names));
  Problem prob("fast_clara_test");
  prob.set_data(std::move(data));
  return prob;
}

// ===========================================================================
// Test 1: CLARA produces valid labels in [0, k).
// ===========================================================================
TEST_CASE("FastCLARA produces valid labels", "[fast_clara][labels]")
{
  constexpr int N = 100;
  constexpr int k = 3;

  Problem prob = make_clara_problem(N);

  algorithms::CLARAOptions opts;
  opts.n_clusters = k;
  opts.n_samples = 3;
  opts.random_seed = 42;

  auto result = algorithms::fast_clara(prob, opts);

  REQUIRE(result.labels.size() == static_cast<size_t>(N));
  for (int label : result.labels) {
    REQUIRE(label >= 0);
    REQUIRE(label < k);
  }
}

// ===========================================================================
// Test 2: Medoid indices are valid and distinct.
// ===========================================================================
TEST_CASE("FastCLARA medoid indices are valid", "[fast_clara][medoids]")
{
  constexpr int N = 100;
  constexpr int k = 3;

  Problem prob = make_clara_problem(N);

  algorithms::CLARAOptions opts;
  opts.n_clusters = k;
  opts.n_samples = 3;
  opts.random_seed = 42;

  auto result = algorithms::fast_clara(prob, opts);

  REQUIRE(result.medoid_indices.size() == static_cast<size_t>(k));

  for (int m : result.medoid_indices) {
    REQUIRE(m >= 0);
    REQUIRE(m < N);
  }

  // All medoid indices must be distinct.
  std::set<int> unique_medoids(result.medoid_indices.begin(), result.medoid_indices.end());
  REQUIRE(unique_medoids.size() == static_cast<size_t>(k));
}

// ===========================================================================
// Test 3: Each medoid is assigned to its own cluster.
// ===========================================================================
TEST_CASE("FastCLARA medoids are assigned to their own cluster", "[fast_clara][self_assignment]")
{
  constexpr int N = 60;
  constexpr int k = 3;

  Problem prob = make_clara_problem(N);

  algorithms::CLARAOptions opts;
  opts.n_clusters = k;
  opts.n_samples = 3;
  opts.random_seed = 42;

  auto result = algorithms::fast_clara(prob, opts);

  for (int c = 0; c < k; ++c) {
    int medoid_point = result.medoid_indices[c];
    REQUIRE(result.labels[medoid_point] == c);
  }
}

// ===========================================================================
// Test 4: Total cost matches recomputed cost from labels and medoids.
// ===========================================================================
TEST_CASE("FastCLARA total_cost matches recomputed cost", "[fast_clara][cost_consistency]")
{
  constexpr int N = 50;
  constexpr int k = 3;

  Problem prob = make_clara_problem(N);

  algorithms::CLARAOptions opts;
  opts.n_clusters = k;
  opts.n_samples = 3;
  opts.random_seed = 42;

  auto result = algorithms::fast_clara(prob, opts);

  // Recompute total cost from labels and medoid_indices.
  double recomputed_cost = 0.0;
  for (int p = 0; p < N; ++p) {
    int medoid = result.medoid_indices[result.labels[p]];
    recomputed_cost += prob.distByInd(p, medoid);
  }

  REQUIRE_THAT(result.total_cost, WithinAbs(recomputed_cost, 1e-10));
}

// ===========================================================================
// Test 5: Reproducibility -- same seed produces same result.
// ===========================================================================
TEST_CASE("FastCLARA is reproducible with same seed", "[fast_clara][reproducibility]")
{
  constexpr int N = 60;
  constexpr int k = 3;

  algorithms::CLARAOptions opts;
  opts.n_clusters = k;
  opts.n_samples = 3;
  opts.random_seed = 123;

  // Reset the global RNG before each run to ensure determinism,
  // since fast_pam uses the global randGenerator for Kmeanspp init.
  dtwc::randGenerator.seed(42);
  Problem prob1 = make_clara_problem(N);
  auto result1 = algorithms::fast_clara(prob1, opts);

  dtwc::randGenerator.seed(42);
  Problem prob2 = make_clara_problem(N);
  auto result2 = algorithms::fast_clara(prob2, opts);

  REQUIRE(result1.labels == result2.labels);
  REQUIRE(result1.medoid_indices == result2.medoid_indices);
  REQUIRE_THAT(result1.total_cost, WithinAbs(result2.total_cost, 1e-10));
}

// ===========================================================================
// Test 6: sample_size >= N falls back to FastPAM on full data.
// ===========================================================================
TEST_CASE("FastCLARA falls back to FastPAM when sample_size >= N", "[fast_clara][fallback]")
{
  constexpr int N = 15;
  constexpr int k = 3;

  // Run CLARA with sample_size >= N.
  Problem prob_clara = make_clara_problem(N);
  algorithms::CLARAOptions opts;
  opts.n_clusters = k;
  opts.sample_size = N + 10; // Larger than N.
  opts.n_samples = 1;
  opts.random_seed = 42;

  // Reset the global RNG to a known state for FastPAM init consistency.
  dtwc::randGenerator.seed(42);
  auto clara_result = algorithms::fast_clara(prob_clara, opts);

  // Run FastPAM directly.
  Problem prob_pam = make_clara_problem(N);
  dtwc::randGenerator.seed(42);
  auto pam_result = fast_pam(prob_pam, k, 100);

  // Results should be identical (same algorithm, same seed).
  REQUIRE(clara_result.medoid_indices == pam_result.medoid_indices);
  REQUIRE(clara_result.labels == pam_result.labels);
  REQUIRE_THAT(clara_result.total_cost, WithinAbs(pam_result.total_cost, 1e-10));
}

// ===========================================================================
// Test 7: CLARA result is no worse than a single subsample PAM.
// ===========================================================================
TEST_CASE("FastCLARA with multiple samples is no worse than single sample", "[fast_clara][quality]")
{
  constexpr int N = 80;
  constexpr int k = 3;

  // Single subsample.
  Problem prob1 = make_clara_problem(N);
  algorithms::CLARAOptions opts1;
  opts1.n_clusters = k;
  opts1.n_samples = 1;
  opts1.random_seed = 42;
  auto result1 = algorithms::fast_clara(prob1, opts1);

  // Multiple subsamples (should be at least as good).
  Problem prob5 = make_clara_problem(N);
  algorithms::CLARAOptions opts5;
  opts5.n_clusters = k;
  opts5.n_samples = 5;
  opts5.random_seed = 42;
  auto result5 = algorithms::fast_clara(prob5, opts5);

  // The first subsample uses the same seed, so n_samples=5 tries
  // that same subsample PLUS 4 more. Cost should be <= (best of 5).
  REQUIRE(result5.total_cost <= result1.total_cost + 1e-10);
}

// ===========================================================================
// Test 8: k=1 with CLARA gives a single cluster.
// ===========================================================================
TEST_CASE("FastCLARA k=1 assigns all points to one cluster", "[fast_clara][k1]")
{
  constexpr int N = 30;
  constexpr int k = 1;

  Problem prob = make_clara_problem(N);

  algorithms::CLARAOptions opts;
  opts.n_clusters = k;
  opts.n_samples = 2;
  opts.random_seed = 42;

  auto result = algorithms::fast_clara(prob, opts);

  REQUIRE(result.medoid_indices.size() == 1);
  for (int label : result.labels) {
    REQUIRE(label == 0);
  }
}

// ===========================================================================
// Test 9: Invalid inputs throw.
// ===========================================================================
TEST_CASE("FastCLARA throws on invalid inputs", "[fast_clara][errors]")
{
  SECTION("k = 0 throws")
  {
    Problem prob = make_clara_problem(10);
    algorithms::CLARAOptions opts;
    opts.n_clusters = 0;
    REQUIRE_THROWS_AS(algorithms::fast_clara(prob, opts), std::runtime_error);
  }

  SECTION("k > N throws")
  {
    Problem prob = make_clara_problem(5);
    algorithms::CLARAOptions opts;
    opts.n_clusters = 10;
    REQUIRE_THROWS_AS(algorithms::fast_clara(prob, opts), std::runtime_error);
  }

  SECTION("empty problem throws")
  {
    Problem prob("empty");
    algorithms::CLARAOptions opts;
    opts.n_clusters = 1;
    REQUIRE_THROWS_AS(algorithms::fast_clara(prob, opts), std::runtime_error);
  }

  SECTION("n_samples = 0 throws")
  {
    Problem prob = make_clara_problem(10);
    algorithms::CLARAOptions opts;
    opts.n_clusters = 2;
    opts.n_samples = 0;
    REQUIRE_THROWS_AS(algorithms::fast_clara(prob, opts), std::runtime_error);
  }
}

// ===========================================================================
// Test 10: Auto sample_size default (Schubert & Rousseeuw 2021 formula).
// ===========================================================================
TEST_CASE("FastCLARA auto sample_size uses improved formula", "[fast_clara][auto_sample_size]")
{
  // With N=100, k=5: max(40+10, min(100, 150)) = max(50, 100) = 100 -> fallback to FastPAM.
  // With N=200, k=5: max(50, min(200, 150)) = max(50, 150) = 150 < 200, so CLARA path.
  // The test just verifies it runs without error and produces a valid result.
  constexpr int N = 100;
  constexpr int k = 5;

  Problem prob = make_clara_problem(N);

  algorithms::CLARAOptions opts;
  opts.n_clusters = k;
  opts.sample_size = -1; // Auto.
  opts.n_samples = 2;
  opts.random_seed = 42;

  auto result = algorithms::fast_clara(prob, opts);

  REQUIRE(result.labels.size() == static_cast<size_t>(N));
  REQUIRE(result.medoid_indices.size() == static_cast<size_t>(k));
  REQUIRE(result.total_cost > 0.0);
}

// ===========================================================================
// Test 10b: Auto sample_size with large N uses improved formula (not old 40+2k).
// ===========================================================================
TEST_CASE("FastCLARA auto sample_size with large N uses Schubert formula", "[fast_clara][auto_sample_size_large]")
{
  // With N=1000, k=5: max(40+10, min(1000, 150)) = max(50, 150) = 150.
  // Old formula would give 50. New formula gives 150.
  // Both should produce valid results; this test confirms the new formula runs.
  constexpr int N = 200;
  constexpr int k = 5;

  Problem prob = make_clara_problem(N);

  algorithms::CLARAOptions opts;
  opts.n_clusters = k;
  opts.sample_size = -1; // Auto — triggers new formula.
  opts.n_samples = 2;
  opts.random_seed = 42;

  auto result = algorithms::fast_clara(prob, opts);

  REQUIRE(result.labels.size() == static_cast<size_t>(N));
  REQUIRE(result.medoid_indices.size() == static_cast<size_t>(k));
  REQUIRE(result.total_cost > 0.0);
}

// ===========================================================================
// Test 11: Different seeds produce different results.
// ===========================================================================
TEST_CASE("FastCLARA different seeds can produce different results", "[fast_clara][seed_variation]")
{
  constexpr int N = 80;
  constexpr int k = 3;

  Problem prob1 = make_clara_problem(N);
  algorithms::CLARAOptions opts1;
  opts1.n_clusters = k;
  opts1.n_samples = 1;
  opts1.sample_size = 30; // Small sample to amplify randomness.
  opts1.random_seed = 1;
  auto result1 = algorithms::fast_clara(prob1, opts1);

  Problem prob2 = make_clara_problem(N);
  algorithms::CLARAOptions opts2 = opts1;
  opts2.random_seed = 999;
  auto result2 = algorithms::fast_clara(prob2, opts2);

  // We cannot guarantee different results (they MIGHT converge to the same),
  // but at minimum both should be valid.
  REQUIRE(result1.labels.size() == static_cast<size_t>(N));
  REQUIRE(result2.labels.size() == static_cast<size_t>(N));
  REQUIRE(result1.total_cost > 0.0);
  REQUIRE(result2.total_cost > 0.0);
}

// ===========================================================================
// Test 12: FastCLARA propagates ndim to sub-problem.
// ===========================================================================
TEST_CASE("FastCLARA: propagates ndim to sub-problem", "[clara][mv]")
{
  dtwc::Data data;
  data.ndim = 2;
  // 10 series, 3 timesteps x 2 features each
  for (int i = 0; i < 10; ++i) {
    data.p_vec.push_back({double(i), double(i + 1), double(i + 2), double(i + 3), double(i + 4), double(i + 5)});
    data.p_names.push_back("s" + std::to_string(i));
  }

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.verbose = false;

  dtwc::algorithms::CLARAOptions opts;
  opts.n_clusters = 2;
  opts.n_samples = 2;
  opts.sample_size = 6;

  auto result = dtwc::algorithms::fast_clara(prob, opts);
  REQUIRE(result.labels.size() == 10);
  REQUIRE(result.medoid_indices.size() == 2);
  REQUIRE(result.total_cost >= 0.0);
}

// ===========================================================================
// Test 13: FastCLARA propagates missing_strategy to sub-problem.
// ===========================================================================
TEST_CASE("FastCLARA: propagates missing_strategy", "[clara][missing]")
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  dtwc::Data data;
  for (int i = 0; i < 10; ++i) {
    std::vector<double> series = {double(i), double(i + 1), double(i + 2)};
    if (i % 3 == 0) series[1] = nan;  // Some series have NaN
    data.p_vec.push_back(std::move(series));
    data.p_names.push_back("s" + std::to_string(i));
  }

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.missing_strategy = dtwc::core::MissingStrategy::ZeroCost;
  prob.verbose = false;

  dtwc::algorithms::CLARAOptions opts;
  opts.n_clusters = 2;
  opts.n_samples = 2;
  opts.sample_size = 6;

  // Should NOT throw — missing_strategy propagated to sub-problem.
  REQUIRE_NOTHROW(dtwc::algorithms::fast_clara(prob, opts));
}

// ===========================================================================
// Test 14: Improved sample size formula with large k.
// ===========================================================================
TEST_CASE("FastCLARA: improved sample size formula", "[clara]")
{
  // For k=70: old formula = 40+140=180, new = max(180, min(N, 800))
  // With N=1000: sample_size should be 800, not 180.
  dtwc::Data data;
  for (int i = 0; i < 1000; ++i) {
    data.p_vec.push_back({double(i), double(i + 1)});
    data.p_names.push_back("s" + std::to_string(i));
  }

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.verbose = false;

  dtwc::algorithms::CLARAOptions opts;
  opts.n_clusters = 70;
  opts.n_samples = 1;
  opts.sample_size = -1;  // auto

  // Should complete (the larger sample gives better results).
  auto result = dtwc::algorithms::fast_clara(prob, opts);
  REQUIRE(result.labels.size() == 1000);
  REQUIRE(result.medoid_indices.size() == 70);
}

// ===========================================================================
// Test 15: FastCLARA with float32 data uses f32 view-mode subsample.
// ===========================================================================
TEST_CASE("FastCLARA with float32 data", "[fast_clara][float32]")
{
  constexpr int N = 60;
  constexpr int k = 3;

  // Build float32 data with 3 distinct groups
  std::vector<std::vector<float>> vecs;
  std::vector<std::string> names;
  for (int i = 0; i < N; ++i) {
    int group = (i * 3) / N;
    float baseline = static_cast<float>(group * 50);
    float slope = static_cast<float>((group == 2) ? -2.0 : (group + 1) * 1.5);
    float noise_offset = static_cast<float>(i * 0.1);

    std::vector<float> ts;
    int len = 20 + (i % 5);
    for (int j = 0; j < len; ++j)
      ts.push_back(baseline + slope * j + noise_offset);
    vecs.push_back(std::move(ts));
    names.push_back("ts_" + std::to_string(i));
  }

  Data data(std::move(vecs), std::move(names));
  REQUIRE(data.is_f32());

  Problem prob("f32_clara");
  prob.set_data(std::move(data));

  algorithms::CLARAOptions opts;
  opts.n_clusters = k;
  opts.n_samples = 3;
  opts.random_seed = 42;

  auto result = algorithms::fast_clara(prob, opts);

  REQUIRE(result.labels.size() == static_cast<size_t>(N));
  REQUIRE(result.medoid_indices.size() == static_cast<size_t>(k));
  REQUIRE(result.total_cost > 0.0);

  // All labels valid
  for (int label : result.labels) {
    REQUIRE(label >= 0);
    REQUIRE(label < k);
  }

  // All medoids distinct and valid
  std::set<int> unique_medoids(result.medoid_indices.begin(), result.medoid_indices.end());
  REQUIRE(unique_medoids.size() == static_cast<size_t>(k));
}
