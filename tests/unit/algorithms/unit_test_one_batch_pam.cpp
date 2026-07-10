/**
 * @file unit_test_one_batch_pam.cpp
 * @brief Correctness, work-bound, and registered quality bands for OneBatchPAM.
 *
 * Registered bands (before execution):
 *   HARD-WORK: actual distance calls / N^2 <= (m+k)/N and the Problem's full
 *              distance matrix remains unfilled.
 *   HARD-QUALITY: on separated synthetic data, objective <= 1.05 * FasterPAM.
 *   HARD-STATE: deterministic for a fixed seed; result is written to Problem.
 *   ADVISORY-50K [.] bench: N=50,000, calls <=10% of N^2 and objective within
 *              5% of the exact separated-line/FasterPAM oracle.
 */

#include <dtwc.hpp>
#include <algorithms/one_batch_pam.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace dtwc;

namespace {

Problem make_problem(int n, int groups, int length = 8)
{
  std::vector<std::vector<data_t>> series;
  std::vector<std::string> names;
  series.reserve(static_cast<std::size_t>(n));
  names.reserve(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) {
    const int group = (i * groups) / n;
    const double within = static_cast<double>(i % (n / groups)) * 0.015;
    std::vector<data_t> values(static_cast<std::size_t>(length));
    for (int t = 0; t < length; ++t)
      values[static_cast<std::size_t>(t)] = group * 100.0 + within + t * (group + 1) * 0.03;
    series.push_back(std::move(values));
    names.push_back("s" + std::to_string(i));
  }
  Problem problem("one_batch_test");
  problem.set_data(Data(std::move(series), std::move(names)));
  return problem;
}

void require_valid(const core::ClusteringResult& result, int n, int k)
{
  REQUIRE(result.labels.size() == static_cast<std::size_t>(n));
  REQUIRE(result.medoid_indices.size() == static_cast<std::size_t>(k));
  REQUIRE(std::set<int>(result.medoid_indices.begin(), result.medoid_indices.end()).size()
          == static_cast<std::size_t>(k));
  for (int label : result.labels) REQUIRE(label >= 0); 
  for (int label : result.labels) REQUIRE(label < k);
  for (int medoid : result.medoid_indices) REQUIRE(medoid >= 0);
  for (int medoid : result.medoid_indices) REQUIRE(medoid < n);
}

template <typename Function>
std::string capture_stderr(Function&& function)
{
  std::ostringstream captured;
  std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
  try {
    function();
  } catch (...) {
    std::cerr.rdbuf(previous);
    throw;
  }
  std::cerr.rdbuf(previous);
  return captured.str();
}

} // namespace

TEST_CASE("OneBatchPAM stays within its fixed-batch distance budget",
          "[one_batch_pam][work]")
{
  constexpr int n = 240;
  constexpr int k = 4;
  constexpr int m = 72;
  auto problem = make_problem(n, k);

  algorithms::OneBatchPAMOptions options;
  options.n_clusters = k;
  options.batch_size = m;
  options.random_seed = 19;
  algorithms::OneBatchPAMStats stats;
  const auto result = algorithms::one_batch_pam(problem, options, &stats);

  require_valid(result, n, k);
  REQUIRE(stats.batch_size == m);
  REQUIRE(stats.distance_evaluations <= static_cast<std::uint64_t>(n) * (m + k));
  REQUIRE(stats.full_matrix_fraction <= static_cast<double>(m + k) / n);
  REQUIRE_FALSE(problem.is_distance_matrix_filled());
  REQUIRE(problem.labels() == result.labels);
  REQUIRE(problem.medoids() == result.medoid_indices);
}

TEST_CASE("OneBatchPAM is reproducible and within five percent of FasterPAM",
          "[one_batch_pam][quality][reproducibility]")
{
  constexpr int n = 240;
  constexpr int k = 4;
  auto p1 = make_problem(n, k);
  auto p2 = make_problem(n, k);
  auto oracle_problem = make_problem(n, k);

  algorithms::OneBatchPAMOptions options;
  options.n_clusters = k;
  options.batch_size = 96;
  options.random_seed = 7;
  const auto first = algorithms::one_batch_pam(p1, options);
  const auto second = algorithms::one_batch_pam(p2, options);
  const auto oracle = fast_pam(oracle_problem, k);

  REQUIRE(first.medoid_indices == second.medoid_indices);
  REQUIRE(first.labels == second.labels);
  REQUIRE(first.total_cost == second.total_cost);
  REQUIRE(first.total_cost <= oracle.total_cost * 1.05 + 1e-9);
}

TEST_CASE("OneBatchPAM finite-maximum debiasing uses actual Dmax below one",
          "[one_batch_pam][debiasing][regression]")
{
  SECTION("Dmax below one uses the actual finite table maximum") {
    Problem problem("one_batch_small_scale");
    problem.set_data(Data(std::vector<std::vector<data_t>>{{0.0}, {0.01}, {0.1}},
                          std::vector<std::string>{"zero", "near", "far"}));

    // Find a deterministic seed for this standard-library implementation
    // whose first shuffle selects the two near points.  Replaying the same
    // engine in one_batch_pam selects the identical fixed batch, while keeping
    // the regression portable across standard-library shuffle algorithms.
    std::uint64_t counterexample_seed = 0;
    for (; counterexample_seed < 1024; ++counterexample_seed) {
      std::vector<int> permutation{0, 1, 2};
      std::mt19937_64 rng(counterexample_seed);
      std::shuffle(permutation.begin(), permutation.end(), rng);
      if ((permutation[0] == 0 && permutation[1] == 1)
          || (permutation[0] == 1 && permutation[1] == 0))
        break;
    }
    REQUIRE(counterexample_seed < 1024);

    algorithms::OneBatchPAMOptions options;
    options.n_clusters = 1;
    options.batch_size = 2;
    options.random_seed = counterexample_seed;
    options.weighting = algorithms::OneBatchWeighting::NearestNeighbor;

    const auto result = algorithms::one_batch_pam(problem, options);

    // The experiment-code finite-max correction normalizes the fixed table by
    // Dmax=0.1, so the far nonsampled point cannot win merely because all
    // off-diagonal distances are numerically below 1.
    REQUIRE(result.medoid_indices == std::vector<int>{0});
    REQUIRE(std::abs(result.total_cost - 0.11) <= 1e-12);
  }

  SECTION("an all-zero table uses a finite normalization fallback") {
    Problem problem("one_batch_zero_scale");
    problem.set_data(Data(std::vector<std::vector<data_t>>{{0.0}, {0.0}, {0.0}},
                          std::vector<std::string>{"a", "b", "c"}));

    algorithms::OneBatchPAMOptions options;
    options.n_clusters = 1;
    options.batch_size = 2;
    options.random_seed = 0;
    options.weighting = algorithms::OneBatchWeighting::NearestNeighbor;
    algorithms::OneBatchPAMStats stats;

    const auto result = algorithms::one_batch_pam(problem, options, &stats);

    require_valid(result, 3, 1);
    REQUIRE(result.total_cost == 0.0);
    REQUIRE(std::isfinite(stats.estimated_objective));
  }
}

TEST_CASE("OneBatchPAM relative tolerance scales below unit cost",
          "[one_batch_pam][tolerance][regression]")
{
  Problem problem("one_batch_subunit_tolerance");
  problem.set_data(Data(std::vector<std::vector<data_t>>{
                          {0.0}, {0.01}, {0.02}, {1.0}},
                        std::vector<std::string>{"zero", "one", "two", "far"}));

  algorithms::OneBatchPAMOptions options;
  options.n_clusters = 2;
  options.batch_size = 4;
  options.max_iter = 1;
  options.relative_tolerance = 0.2;
  options.random_seed = 0;
  options.weighting = algorithms::OneBatchWeighting::Uniform;
  algorithms::OneBatchPAMStats stats;

  const auto result = algorithms::one_batch_pam(problem, options, &stats);

  // The initial {3,2} medoids cost 0.03. Replacing 2 by 1 lowers that to
  // 0.02: an absolute gain of 0.01 and a 33.3% relative improvement. A 20%
  // threshold must therefore accept the swap even though the objective is <1.
  REQUIRE(result.medoid_indices == std::vector<int>{3, 1});
  REQUIRE(std::abs(result.total_cost - 0.02) <= 1e-12);
  REQUIRE(stats.accepted_swaps == 1);
  REQUIRE(std::abs(stats.estimated_objective - 0.02) <= 1e-12);
}

TEST_CASE("OneBatchPAM handles k=1, k=N, and invalid options",
          "[one_batch_pam][edge]")
{
  SECTION("k=1") {
    auto problem = make_problem(30, 1);
    algorithms::OneBatchPAMOptions options;
    options.n_clusters = 1;
    options.batch_size = 12;
    const auto result = algorithms::one_batch_pam(problem, options);
    require_valid(result, 30, 1);
    REQUIRE(result.converged);
  }
  SECTION("k=N") {
    auto problem = make_problem(12, 3);
    algorithms::OneBatchPAMOptions options;
    options.n_clusters = 12;
    const auto result = algorithms::one_batch_pam(problem, options);
    REQUIRE(result.total_cost == 0.0);
    REQUIRE(result.labels == result.medoid_indices);
  }
  SECTION("invalid") {
    auto problem = make_problem(12, 3);
    algorithms::OneBatchPAMOptions options;
    options.n_clusters = 0;
    REQUIRE_THROWS_AS(algorithms::one_batch_pam(problem, options), InvalidInput);
    options.n_clusters = 2;
    options.batch_size = 0;
    REQUIRE_THROWS_AS(algorithms::one_batch_pam(problem, options), InvalidInput);
  }
}

TEST_CASE("OneBatchPAM reports explicit batch sizes raised to the cluster count",
          "[one_batch_pam][loudness][options]")
{
  constexpr auto expected =
    "[dtwc] warning: one_batch_pam requested batch_size=2, but n_clusters=4 "
    "requires batch_size >= 4; using effective batch_size=4. Set batch_size "
    "to at least n_clusters to avoid this adjustment.\n";

  SECTION("an explicit undersized batch reports every corrected invocation") {
    auto first_problem = make_problem(12, 4);
    auto second_problem = make_problem(12, 4);
    algorithms::OneBatchPAMOptions options;
    options.n_clusters = 4;
    options.batch_size = 2;
    options.max_iter = 1;
    algorithms::OneBatchPAMStats first_stats;
    algorithms::OneBatchPAMStats second_stats;

    const std::string stderr_output = capture_stderr([&] {
      algorithms::one_batch_pam(first_problem, options, &first_stats);
      algorithms::one_batch_pam(second_problem, options, &second_stats);
    });

    REQUIRE(stderr_output == std::string(expected) + expected);
    REQUIRE(first_stats.batch_size == 4);
    REQUIRE(second_stats.batch_size == 4);
  }

  SECTION("automatic batch selection remains silent") {
    // N=256 selects m=180 automatically; k=181 exercises the same m >= k
    // correction without turning an internal auto-policy choice into noise.
    auto problem = make_problem(256, 181);
    algorithms::OneBatchPAMOptions options;
    options.n_clusters = 181;
    options.batch_size = -1;
    options.max_iter = 1;
    algorithms::OneBatchPAMStats stats;

    REQUIRE(capture_stderr([&] {
      algorithms::one_batch_pam(problem, options, &stats);
    }).empty());
    REQUIRE(stats.batch_size == 181);
  }

  SECTION("an explicit batch at least as large as k remains silent") {
    auto problem = make_problem(12, 4);
    algorithms::OneBatchPAMOptions options;
    options.n_clusters = 4;
    options.batch_size = 4;
    options.max_iter = 1;

    REQUIRE(capture_stderr([&] { algorithms::one_batch_pam(problem, options); }).empty());
  }
}

TEST_CASE("OneBatchPAM 50k registered scaling and quality band",
          "[.][one_batch_pam][bench][50k]")
{
  constexpr int n = 50000;
  constexpr int k = 5;
  auto problem = make_problem(n, k, 1);
  algorithms::OneBatchPAMOptions options;
  options.n_clusters = k;
  options.batch_size = 256;
  options.random_seed = 42;
  algorithms::OneBatchPAMStats stats;
  const auto result = algorithms::one_batch_pam(problem, options, &stats);

  // On this separated one-dimensional construction, the exact group medians
  // are the FasterPAM fixed point. Compute that oracle directly without
  // materialising the infeasible 50k-by-50k matrix.
  const int group_size = n / k;
  double oracle_cost = 0.0;
  for (int group = 0; group < k; ++group) {
    const int median_offset = (group_size - 1) / 2;
    for (int offset = 0; offset < group_size; ++offset)
      oracle_cost += std::abs(offset - median_offset) * 0.015;
  }
  REQUIRE(result.total_cost <= oracle_cost * 1.05 + 1e-9);
  REQUIRE(stats.full_matrix_fraction <= 0.10);
}
