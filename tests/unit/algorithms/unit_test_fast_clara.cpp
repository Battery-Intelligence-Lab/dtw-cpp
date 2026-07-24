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
#include <algorithms/detail/fast_clara_plan.hpp>
#include <algorithms/fast_clara.hpp>
#include <error.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <array>
#include <filesystem>
#include <limits>
#include <set>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifndef DTWC_TEST_DATA_DIR
#define DTWC_TEST_DATA_DIR "./data"
#endif

static struct TestDataInitCLARA
{
  TestDataInitCLARA()
  {
    dtwc::settings::paths::setDataPath(DTWC_TEST_DATA_DIR);
    // Route CSV output to a per-run temp dir so the test doesn't pollute the
    // repo root or build tree (CWD-dependent otherwise).
    const auto out = std::filesystem::temp_directory_path() / "dtwc_fast_clara_test";
    std::error_code ec;
    std::filesystem::create_directories(out, ec);
    dtwc::settings::paths::setResultsPath(out);
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

  // CLARA owns a complete invocation-local seed schedule. Deliberately give the
  // legacy Tier-2 engine different states: neither run may consume it, and the
  // results must still agree.
  const auto legacy_rng_original = dtwc::randGenerator;
  dtwc::randGenerator.seed(17);
  const auto legacy_rng_before_1 = dtwc::randGenerator;
  Problem prob1 = make_clara_problem(N);
  auto result1 = algorithms::fast_clara(prob1, opts);
  CHECK(dtwc::randGenerator == legacy_rng_before_1);

  dtwc::randGenerator.seed(8675309);
  const auto legacy_rng_before_2 = dtwc::randGenerator;
  Problem prob2 = make_clara_problem(N);
  auto result2 = algorithms::fast_clara(prob2, opts);
  CHECK(dtwc::randGenerator == legacy_rng_before_2);
  dtwc::randGenerator = legacy_rng_original;

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
  // A full-data request is one seeded PAM invocation, regardless of the CLARA
  // repetition count: there is no subsampling left to repeat.
  opts.n_samples = 3;
  opts.random_seed = 42;

  const auto legacy_rng_original = dtwc::randGenerator;
  dtwc::randGenerator.seed(314159);
  const auto legacy_rng_before = dtwc::randGenerator;
  auto clara_result = algorithms::fast_clara(prob_clara, opts);
  CHECK(dtwc::randGenerator == legacy_rng_before);
  dtwc::randGenerator = legacy_rng_original;

  // Run the exact invocation-local FastPAM fallback oracle directly.
  Problem prob_pam = make_clara_problem(N);
  auto pam_result = fast_pam_seeded(prob_pam, k, opts.random_seed, 100);

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
    REQUIRE_THROWS_WITH(
      algorithms::fast_clara(prob, opts),
      "fast_clara: n_samples must be positive.");
  }

  SECTION("negative n_samples throws")
  {
    Problem prob = make_clara_problem(10);
    algorithms::CLARAOptions opts;
    opts.n_clusters = 2;
    opts.n_samples = -1;
    REQUIRE_THROWS_AS(algorithms::fast_clara(prob, opts), InvalidInput);
  }

  SECTION("sample_size accepts only -1 or a positive value")
  {
    Problem prob = make_clara_problem(10);
    algorithms::CLARAOptions opts;
    opts.n_clusters = 2;
    opts.sample_size = GENERATE(0, -2, std::numeric_limits<int>::min());
    REQUIRE_THROWS_WITH(
      algorithms::fast_clara(prob, opts),
      "fast_clara: sample_size must be -1 or a positive integer.");
  }

  SECTION("max_iter must be positive")
  {
    Problem prob = make_clara_problem(10);
    algorithms::CLARAOptions opts;
    opts.n_clusters = 2;
    opts.max_iter = 0;
    REQUIRE_THROWS_WITH(
      algorithms::fast_clara(prob, opts),
      "fast_clara: max_iter must be positive.");
  }
}

TEST_CASE("FastCLARA dimension planning is overflow-safe before allocation",
          "[fast_clara][errors][overflow]")
{
  algorithms::CLARAOptions opts;
  opts.n_clusters = 300'000'000;
  opts.sample_size = -1;
  opts.n_samples = 1;
  opts.max_iter = 1;

  // 10*k+100 overflows a 32-bit int. The mathematical auto size is capped by
  // N and must remain exactly INT_MAX, not wrap to the smaller 40+2*k arm.
  const auto plan = algorithms::detail::resolve_clara_plan(
    std::numeric_limits<int>::max(), opts, "fast_clara");
  REQUIRE(plan.n_points == std::numeric_limits<int>::max());
  REQUIRE(plan.sample_size == std::numeric_limits<int>::max());

  opts.n_clusters = std::numeric_limits<int>::max();
  const auto maximal_k_plan = algorithms::detail::resolve_clara_plan(
    std::numeric_limits<int>::max(), opts, "fast_clara");
  REQUIRE(maximal_k_plan.sample_size == std::numeric_limits<int>::max());

  REQUIRE_THROWS_WITH(
    algorithms::detail::validate_streaming_clara_plan(
      maximal_k_plan, "fast_clara"),
    "fast_clara: sample_size resolves to N, but the Parquet dataset exceeds "
    "ram_limit_bytes; use sample_size < N or raise the RAM limit for the "
    "single full-data PAM fallback.");

  REQUIRE_THROWS_WITH(
    algorithms::detail::resolve_clara_plan(
      static_cast<std::int64_t>(std::numeric_limits<int>::max()) + 1,
      opts,
      "fast_clara"),
    "fast_clara: N exceeds the int-indexed clustering result limit.");
}

TEST_CASE("FastCLARA forced streaming validates its route before reader I/O",
          "[fast_clara][errors][streaming]")
{
  algorithms::CLARAOptions opts;
  opts.n_clusters = 2;
  opts.force_parquet_streaming = true;
  opts.ram_limit_bytes = 4096;
  opts.parquet_path = "never_opened.parquet";

  SECTION("a limit is required")
  {
    opts.ram_limit_bytes = 0;
    Problem settings_only{"clara_missing_stream_limit"};
    REQUIRE_THROWS_WITH(
      algorithms::fast_clara(settings_only, opts),
      "fast_clara: force_parquet_streaming requires ram_limit_bytes and "
      "parquet_path.");
  }

  SECTION("a Parquet path is required")
  {
    opts.parquet_path.clear();
    Problem settings_only{"clara_missing_stream_path"};
    REQUIRE_THROWS_WITH(
      algorithms::fast_clara(settings_only, opts),
      "fast_clara: force_parquet_streaming requires ram_limit_bytes and "
      "parquet_path.");
  }

  SECTION("resident series are rejected")
  {
    Problem resident = make_clara_problem(3);
    REQUIRE_THROWS_WITH(
      algorithms::fast_clara(resident, opts),
      "fast_clara: force_parquet_streaming requires a settings-only Problem "
      "without resident series.");
  }

#ifndef DTWC_HAS_PARQUET
  SECTION("the missing capability is loud")
  {
    Problem settings_only{"clara_missing_parquet"};
    REQUIRE_THROWS_WITH(
      algorithms::fast_clara(settings_only, opts),
      "fast_clara: force_parquet_streaming requires a build with Parquet "
      "support.");
  }
#endif
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
    data.p_vec.push_back({ double(i), double(i + 1), double(i + 2), double(i + 3), double(i + 4), double(i + 5) });
    data.p_names.push_back("s" + std::to_string(i));
  }

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.set_verbose(false);

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
    std::vector<double> series = { double(i), double(i + 1), double(i + 2) };
    if (i % 3 == 0) series[1] = nan; // Some series have NaN
    data.p_vec.push_back(std::move(series));
    data.p_names.push_back("s" + std::to_string(i));
  }

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.missing_strategy = dtwc::core::MissingStrategy::ZeroCost;
  prob.set_verbose(false);

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
    data.p_vec.push_back({ double(i), double(i + 1) });
    data.p_names.push_back("s" + std::to_string(i));
  }

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.set_verbose(false);

  dtwc::algorithms::CLARAOptions opts;
  opts.n_clusters = 70;
  opts.n_samples = 1;
  opts.sample_size = -1; // auto

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
  opts.sample_size = 30;
  opts.random_seed = 42;

  auto result = algorithms::fast_clara(prob, opts);

  REQUIRE(result.labels.size() == static_cast<size_t>(N));
  REQUIRE(result.medoid_indices.size() == static_cast<size_t>(k));
  REQUIRE(result.total_cost > 0.0);
  REQUIRE(prob.dense_distance_matrix().size() == 0);
  REQUIRE(prob.dense_distance_matrix().packed_count() == 0);

  // All labels valid
  for (int label : result.labels) {
    REQUIRE(label >= 0);
    REQUIRE(label < k);
  }

  // All medoids distinct and valid
  std::set<int> unique_medoids(result.medoid_indices.begin(), result.medoid_indices.end());
  REQUIRE(unique_medoids.size() == static_cast<size_t>(k));
}

// ===========================================================================
// Task 0.11 / Phase 8: both paths use the portable seeded selection map.
//
// Bug (2026-06-01 audit): the in-RAM subsample used std::mt19937 + std::shuffle
// while the chunked (Parquet) path uses std::mt19937_64 + std::sample, so for the
// same seed the two paths drew DIFFERENT subsamples and returned DIFFERENT
// medoids. The Phase-8 map also removes the standard-library dependence from
// both paths while retaining their shared sorted-sample contract.
//
// This test pins that contract WITHOUT needing Parquet, using a deterministic
// oracle (no RNG-order guessing):
//   * series i is the constant vector [i, i, ..., i] (length L), so for equal
//     length series DTW(i, j) = L * |i - j|.
//   * with k = 1 the single medoid is, for ANY Kmeanspp init, the point that
//     minimises total distance = the MEDIAN of the sampled indices (fast_pam's
//     k=1 swap converges to it — verified against fast_pam.cpp: second_dist is
//     +inf for k=1, so each swap delta is (total dist to candidate) − (to medoid),
//     minimised by the median).
//   * stable selection over [0, N) returns a SORTED subset, so an odd-sized
//     sample's median sits at position sample_size/2.
// Expected medoids are literal portable-v1 fingerprints below; recomputing
// them with the production sampler here would make the oracle tautological.
//
// Both historical vendor schedules (mt19937+shuffle and mt19937_64+std::sample)
// select different medians, so this literal portable-v1 oracle fails before the
// Phase-8 repair. Six seeds make an accidental collision negligible (~1e-10).
// ===========================================================================
TEST_CASE("FastCLARA in-RAM uses the portable seeded sample contract",
          "[fast_clara][task0_11][seed]")
{
  constexpr int N = 200;
  constexpr int L = 8;
  constexpr int sample_size = 51; // odd -> unique median at index 25 of the sample

  // Constant series: series i has value i, so DTW(i, j) = L * |i - j|.
  std::vector<std::vector<data_t>> vecs;
  std::vector<std::string> names;
  for (int i = 0; i < N; ++i) {
    vecs.emplace_back(static_cast<size_t>(L), static_cast<data_t>(i));
    names.push_back("s" + std::to_string(i));
  }

  constexpr std::array expected_medians{
    std::pair{ 1u, 103 }, std::pair{ 7u, 86 }, std::pair{ 42u, 121 }, std::pair{ 123u, 108 }, std::pair{ 999u, 106 }, std::pair{ 2024u, 105 }
  };
  for (const auto [seed, expected_medoid] : expected_medians) {
    // Fresh problem per seed (no cached-matrix carry-over between seeds).
    std::vector<std::vector<data_t>> v = vecs;
    std::vector<std::string> nm = names;
    Problem prob("clara_seed_" + std::to_string(seed));
    prob.set_data(Data(std::move(v), std::move(nm)));
    prob.set_verbose(false);

    algorithms::CLARAOptions opts;
    opts.n_clusters = 1;
    opts.sample_size = sample_size; // < N -> CLARA path (not the FastPAM fallback)
    opts.n_samples = 1;
    opts.random_seed = seed;

    dtwc::randGenerator.seed(42); // k=1 is init-independent; reseed for hygiene
    auto result = algorithms::fast_clara(prob, opts);

    REQUIRE(result.medoid_indices.size() == 1);
    INFO("seed=" << seed << " expected median index=" << expected_medoid
                 << " got=" << result.medoid_indices[0]);
    REQUIRE(result.medoid_indices[0] == expected_medoid);
  }
}

// ===========================================================================
// Task 0.11 — Test B: parallel in-RAM assignment stays deterministic + correct.
//
// assign_all_points() is OpenMP-parallel over N points and calls the serially
// bound DTW dispatcher directly, so workers share no parent-cache writes. A
// race would corrupt labels / total_cost non-deterministically. N > 64 forces
// the parallel branch. Results must be reproducible and self-consistent
// (total_cost == cost recomputed from the returned labels + medoids). total_cost
// uses a serial index-ordered reduction, so it is deterministic despite threads.
// ===========================================================================
TEST_CASE("FastCLARA parallel in-RAM assignment is deterministic and consistent",
          "[fast_clara][task0_11][parallel]")
{
  constexpr int N = 200; // > 64 -> exercises the OpenMP assign branch
  constexpr int k = 3;

  algorithms::CLARAOptions opts;
  opts.n_clusters = k;
  opts.n_samples = 3;
  opts.sample_size = 60; // < N -> CLARA path, so assign_all_points runs
  opts.random_seed = 7;

  dtwc::randGenerator.seed(42);
  Problem prob1 = make_clara_problem(N);
  auto r1 = algorithms::fast_clara(prob1, opts);

  dtwc::randGenerator.seed(42);
  Problem prob2 = make_clara_problem(N);
  auto r2 = algorithms::fast_clara(prob2, opts);

  // Reproducible across runs (deterministic parallel assignment + serial sum).
  REQUIRE(r1.labels == r2.labels);
  REQUIRE(r1.medoid_indices == r2.medoid_indices);
  REQUIRE_THAT(r1.total_cost, WithinAbs(r2.total_cost, 1e-12));

  // Valid + self-consistent: total_cost equals cost recomputed from labels.
  REQUIRE(r1.labels.size() == static_cast<size_t>(N));
  double recomputed = 0.0;
  const auto &distance = prob1.dtw_function();
  for (int p = 0; p < N; ++p) {
    REQUIRE(r1.labels[p] >= 0);
    REQUIRE(r1.labels[p] < k);
    const int medoid = r1.medoid_indices[r1.labels[p]];
    recomputed += p == medoid ? 0.0 : distance(prob1.series(p), prob1.series(medoid));
  }
  REQUIRE_THAT(r1.total_cost, WithinAbs(recomputed, 1e-9));
  REQUIRE(prob1.dense_distance_matrix().size() == 0);
  REQUIRE(prob1.dense_distance_matrix().packed_count() == 0);
}

TEST_CASE("FastCLARA leaves an existing parent cache byte-stable",
          "[fast_clara][matrix_free][cache_state]")
{
  constexpr int N = 40;
  Problem cached = make_clara_problem(N);
  Problem fresh = make_clara_problem(N);

  auto &matrix = cached.dense_distance_matrix();
  matrix.resize(N);
  matrix.set(0, 1, 12345.0);
  matrix.set(2, 3, 67890.0);

  algorithms::CLARAOptions opts;
  opts.n_clusters = 3;
  opts.sample_size = 20;
  opts.n_samples = 2;
  opts.random_seed = 42;

  const auto cached_result = algorithms::fast_clara(cached, opts);
  const auto fresh_result = algorithms::fast_clara(fresh, opts);

  REQUIRE(cached_result.labels == fresh_result.labels);
  REQUIRE(cached_result.medoid_indices == fresh_result.medoid_indices);
  REQUIRE(cached_result.total_cost == fresh_result.total_cost);
  REQUIRE(matrix.size() == N);
  REQUIRE(matrix.packed_count() == static_cast<std::size_t>(N) * (N + 1) / 2);
  REQUIRE(matrix.count_computed() == 2);
  REQUIRE(matrix.get(0, 1) == 12345.0);
  REQUIRE(matrix.get(2, 3) == 67890.0);
}
