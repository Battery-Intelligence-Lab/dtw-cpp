/**
 * @file unit_test_one_batch_pam.cpp
 * @brief Correctness, work-bound, and registered quality bands for OneBatchPAM.
 *
 * Registered bands (before execution):
 *   HARD-WORK: actual distance calls / N^2 <= (m+k)/N and the Problem's full
 *              distance matrix remains unfilled.
 *   HARD-QUALITY: on separated synthetic data, objective <= 1.05 * FasterPAM.
 *   HARD-STATE: deterministic for a fixed seed; result is written to Problem.
 *   ADVISORY-50K [.] bench: N=50,000 genuinely warped series of lengths
 *              64..128, calls <=0.52198956% of N^2 and objective within 5%
 *              of an exhaustive exact 100-profile medoid oracle.
 */

#include <dtwc.hpp>
#include <algorithms/one_batch_pam.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <vector>

using namespace dtwc;

namespace {

void require_valid(const core::ClusteringResult &result, int n, int k);

constexpr int warped_groups = 5;
constexpr int warped_variants = 100;
constexpr int warped_min_length = 64;
constexpr int warped_max_length = 128;
constexpr int warped_band = 8;
constexpr int warped_batch = 256;
constexpr double warped_group_separation = 1000.0;
constexpr double warped_profile_abs_bound = 3.1;

std::vector<data_t> warped_profile(int variant, int group = 0)
{
  constexpr double pi = 3.141592653589793238462643383279502884;
  const int length = warped_min_length + (variant * 37) % (warped_max_length - warped_min_length + 1);
  const double warp = -0.36 + 0.72 * static_cast<double>(variant % 20) / 19.0;
  const double phase = -0.05 + 0.10 * static_cast<double>(variant / 20) / 4.0;
  const double amplitude =
    0.92 + 0.16 * static_cast<double>((variant * 7) % 19) / 18.0;
  const double level =
    -0.08 + 0.16 * static_cast<double>((variant * 11) % 23) / 22.0;

  std::vector<data_t> series(static_cast<std::size_t>(length));
  for (int t = 0; t < length; ++t) {
    const double u = static_cast<double>(t) / static_cast<double>(length - 1);
    // A monotone nonlinear clock: its derivative is bounded below by
    // 1-|warp|-|phase| >= 0.59, so this is a time warp rather than a fold.
    const double tau = u + warp * u * (1.0 - u)
                       + phase * std::sin(2.0 * pi * u) / (2.0 * pi);
    const double shape = amplitude
                           * (1.8 * std::sin(2.0 * pi * tau)
                              + 0.65 * std::cos(4.0 * pi * tau)
                              + 0.30 * std::sin(10.0 * pi * tau))
                         + level;
    series[static_cast<std::size_t>(t)] =
      group * warped_group_separation + shape;
  }
  return series;
}

Problem make_warped_problem(int replicas)
{
  std::vector<std::vector<data_t>> series;
  std::vector<std::string> names;
  const int n = warped_groups * warped_variants * replicas;
  series.reserve(static_cast<std::size_t>(n));
  names.reserve(static_cast<std::size_t>(n));
  for (int group = 0; group < warped_groups; ++group) {
    for (int variant = 0; variant < warped_variants; ++variant) {
      const auto profile = warped_profile(variant, group);
      for (int replica = 0; replica < replicas; ++replica) {
        series.push_back(profile);
        names.push_back("g" + std::to_string(group) + "_v"
                        + std::to_string(variant) + "_r"
                        + std::to_string(replica));
      }
    }
  }
  Problem problem("one_batch_warped_scaling");
  problem.set_band(warped_band);
  problem.set_data(Data(std::move(series), std::move(names)));
  return problem;
}

struct WarpedOracle
{
  int best_variant = -1;
  int worst_variant = -1;
  double best_cost_per_replica = std::numeric_limits<double>::infinity();
  double worst_cost_per_replica = -1.0;
};

WarpedOracle warped_oracle()
{
  std::vector<std::vector<data_t>> profiles;
  profiles.reserve(warped_variants);
  for (int variant = 0; variant < warped_variants; ++variant)
    profiles.push_back(warped_profile(variant));

  WarpedOracle oracle;
  for (int candidate = 0; candidate < warped_variants; ++candidate) {
    double per_group = 0.0;
    for (int variant = 0; variant < warped_variants; ++variant) {
      per_group += dtwBanded<data_t>(profiles[static_cast<std::size_t>(candidate)],
                                     profiles[static_cast<std::size_t>(variant)],
                                     warped_band);
    }
    const double all_groups = warped_groups * per_group;
    if (all_groups < oracle.best_cost_per_replica) {
      oracle.best_cost_per_replica = all_groups;
      oracle.best_variant = candidate;
    }
    if (all_groups > oracle.worst_cost_per_replica) {
      oracle.worst_cost_per_replica = all_groups;
      oracle.worst_variant = candidate;
    }
  }
  return oracle;
}

constexpr std::uint64_t one_batch_max_evaluations(int n)
{
  // The N*m table omits its m self-pairs. If no selected medoid belongs to
  // the batch, exact labeling adds k*(N-1); any in-batch medoid removes a
  // complete labeling column. This is a tight implementation-independent
  // upper bound for the registered m and k.
  return static_cast<std::uint64_t>(n) * warped_batch - warped_batch
         + static_cast<std::uint64_t>(warped_groups) * (n - 1);
}

void require_warped_fixture_contract()
{
  std::set<int> lengths;
  for (int variant = 0; variant < warped_variants; ++variant) {
    const auto profile = warped_profile(variant);
    lengths.insert(static_cast<int>(profile.size()));
    for (double value : profile)
      REQUIRE(std::abs(value) <= warped_profile_abs_bound);
  }
  REQUIRE(*lengths.begin() == warped_min_length);
  REQUIRE(*lengths.rbegin() == warped_max_length);
  REQUIRE(lengths.size() == 65);
}

void require_warped_result(const core::ClusteringResult &result,
                           const algorithms::OneBatchPAMStats &stats,
                           int replicas, const WarpedOracle &oracle)
{
  const int group_size = warped_variants * replicas;
  const int n = warped_groups * group_size;
  require_valid(result, n, warped_groups);

  std::set<int> represented_groups;
  for (int medoid : result.medoid_indices)
    represented_groups.insert(medoid / group_size);
  REQUIRE(represented_groups.size() == warped_groups);

  const double exact_oracle = oracle.best_cost_per_replica * replicas;
  REQUIRE(result.total_cost <= exact_oracle * 1.05 + 1e-9);
  REQUIRE(stats.batch_size == warped_batch);
  REQUIRE(stats.distance_evaluations <= one_batch_max_evaluations(n));
  REQUIRE(stats.full_matrix_fraction
          <= static_cast<double>(one_batch_max_evaluations(n))
               / (static_cast<double>(n) * n));
  REQUIRE_FALSE(result.total_cost == 0.0);
}

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

TEST_CASE("OneBatchPAM warped scaling oracle is non-degenerate and discriminating",
          "[one_batch_pam][quality][warped_fixture]")
{
  require_warped_fixture_contract();
  const auto oracle = warped_oracle();

  REQUIRE(oracle.best_variant >= 0);
  REQUIRE(oracle.worst_variant >= 0);
  REQUIRE(oracle.best_variant != oracle.worst_variant);
  // Falsification check registered with the 5% quality band: deliberately
  // choosing the exhaustive oracle's worst profile in every group must fail
  // the very band used by the preflight and 50k simulations.
  REQUIRE(oracle.worst_cost_per_replica > oracle.best_cost_per_replica * 1.05);

  // The oracle is globally exact, not merely a feasible reference. Each
  // profile is bounded by +/-3.1 and adjacent groups are shifted by 1000.
  // Thus omitting a group costs at least 100*R*64*(1000-2*3.1), whereas an
  // band-admissible scaled-diagonal path to any one-per-group representative
  // costs at most
  // 500*R*(2*128-1)*(2*3.1). The former is strictly larger, so every global
  // k=5 optimum represents all five groups. Equal variant multiplicities and
  // translation invariance then reduce it exactly to the exhaustive search
  // over the 100 unique within-group candidates above.
  const double missing_group_lower = warped_variants * warped_min_length
                                     * (warped_group_separation - 2.0 * warped_profile_abs_bound);
  const double one_per_group_upper = warped_groups * warped_variants
                                     * (2 * warped_max_length - 1) * (2.0 * warped_profile_abs_bound);
  REQUIRE(missing_group_lower > one_per_group_upper);

  std::cout << std::setprecision(17)
            << "M6_ORACLE best_variant=" << oracle.best_variant
            << " worst_variant=" << oracle.worst_variant
            << " best_cost_per_replica=" << oracle.best_cost_per_replica
            << " worst_cost_per_replica=" << oracle.worst_cost_per_replica
            << " mutation_ratio="
            << oracle.worst_cost_per_replica / oracle.best_cost_per_replica
            << " missing_group_lower=" << missing_group_lower
            << " one_per_group_upper=" << one_per_group_upper
            << '\n';
}

TEST_CASE("OneBatchPAM warped scaling preflight",
          "[.][one_batch_pam][bench][preflight]")
{
  constexpr int replicas = 10;
  constexpr int n = warped_groups * warped_variants * replicas;
  const auto oracle = warped_oracle();
  auto problem = make_warped_problem(replicas);
  algorithms::OneBatchPAMOptions options;
  options.n_clusters = warped_groups;
  options.batch_size = warped_batch;
  options.random_seed = 42;
  algorithms::OneBatchPAMStats stats;

  const auto start = std::chrono::steady_clock::now();
  const auto result = algorithms::one_batch_pam(problem, options, &stats);
  const double seconds = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - start)
                           .count();
  require_warped_result(result, stats, replicas, oracle);

  std::cout << std::setprecision(17)
            << "M6_PREFLIGHT n=" << n
            << " variants=" << warped_variants
            << " replicas=" << replicas
            << " lengths=64..128 band=" << warped_band
            << " batch=" << stats.batch_size
            << " wall_s=" << seconds
            << " evaluations=" << stats.distance_evaluations
            << " max_evaluations=" << one_batch_max_evaluations(n)
            << " fraction=" << stats.full_matrix_fraction
            << " cost=" << result.total_cost
            << " exact_oracle=" << oracle.best_cost_per_replica * replicas
            << " ratio=" << result.total_cost / (oracle.best_cost_per_replica * replicas)
            << " accepted_swaps=" << stats.accepted_swaps << " medoids=";
  for (std::size_t i = 0; i < result.medoid_indices.size(); ++i)
    std::cout << (i == 0 ? "" : ",") << result.medoid_indices[i];
  std::cout << '\n';
}

TEST_CASE("OneBatchPAM 50k registered warped scaling and quality band",
          "[.][one_batch_pam][bench][50k]")
{
  constexpr int replicas = 100;
  constexpr int n = warped_groups * warped_variants * replicas;
  static_assert(n == 50000);
  static_assert(one_batch_max_evaluations(n) == 13049739);

  // Registered before execution:
  //   memory: N*m doubles = 102,400,000 B (97.65625 MiB) for the only table;
  //           series payload = 4,815,000 doubles (36.7355 MiB).
  //   work:   <=13,049,739 DTWs = 0.52198956% of N^2; with <=17 band
  //           cells per short-side row and lengths <=128, <=28,396,242,944
  //           scalar DP-cell updates (a conservative upper bound).
  //   quality: all five groups represented and cost <=1.05* the exact oracle.
  // Runtime is advisory on a shared host and is predicted from the separately
  // run, structurally identical replicas=10 preflight before this test starts.
  const auto oracle = warped_oracle();
  auto problem = make_warped_problem(replicas);
  algorithms::OneBatchPAMOptions options;
  options.n_clusters = warped_groups;
  options.batch_size = warped_batch;
  options.random_seed = 42;
  algorithms::OneBatchPAMStats stats;

  const auto start = std::chrono::steady_clock::now();
  const auto result = algorithms::one_batch_pam(problem, options, &stats);
  const double seconds = std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - start)
                           .count();
  require_warped_result(result, stats, replicas, oracle);

  std::cout << std::setprecision(17)
            << "M6_50K n=" << n
            << " variants=" << warped_variants
            << " replicas=" << replicas
            << " lengths=64..128 band=" << warped_band
            << " batch=" << stats.batch_size
            << " wall_s=" << seconds
            << " evaluations=" << stats.distance_evaluations
            << " max_evaluations=" << one_batch_max_evaluations(n)
            << " fraction=" << stats.full_matrix_fraction
            << " cost=" << result.total_cost
            << " exact_oracle=" << oracle.best_cost_per_replica * replicas
            << " ratio=" << result.total_cost / (oracle.best_cost_per_replica * replicas)
            << " accepted_swaps=" << stats.accepted_swaps << " medoids=";
  for (std::size_t i = 0; i < result.medoid_indices.size(); ++i)
    std::cout << (i == 0 ? "" : ",") << result.medoid_indices[i];
  std::cout << '\n';
}
