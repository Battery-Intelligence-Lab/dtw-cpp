/**
 * @file unit_test_one_batch_pam.cpp
 * @brief Correctness, work-bound, and registered quality bands for OneBatchPAM.
 *
 * Registered bands (before execution):
 *   HARD-WORK: actual distance calls / N^2 <= (m+k)/N and the Problem's full
 *              distance matrix remains unfilled.
 *   HARD-QUALITY: on separated synthetic data, objective <= 1.05 * FasterPAM.
 *   HARD-STATE: deterministic for a fixed seed; result is written to Problem.
 */

#include <dtwc.hpp>
#include <algorithms/one_batch_pam.hpp>
#include <core/medoid_assignment_policy.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace dtwc;

namespace {

void require_valid(const core::ClusteringResult &result, int n, int k);

constexpr int warped_groups = 5;
constexpr int warped_variants = 100;
constexpr int warped_min_length = 64;
constexpr int warped_max_length = 128;
constexpr int warped_band = warped_max_length - warped_min_length;
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

// A fixed, library-independent set of random walks (integer hash, no
// <random> distribution). Steps are multiples of 0.001, which binary64 cannot
// hold, so the DTW sums are inexact and a reduction whose order followed the
// thread count would change the last bits of the cost.
template <typename T>
Data walk_data(const std::vector<std::size_t> &lengths)
{
  std::uint64_t state = 0x9E3779B97F4A7C15ull;
  const auto next = [&state] {
    state += 0x9E3779B97F4A7C15ull;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
  };
  std::vector<std::vector<T>> series;
  std::vector<std::string> names;
  for (std::size_t i = 0; i < lengths.size(); ++i) {
    std::vector<T> walk(lengths[i]);
    double x = 0.0;
    for (T &value : walk) {
      x += (static_cast<double>(next() % 2001) - 1000.0) / 1000.0;
      value = static_cast<T>(x);
    }
    series.push_back(std::move(walk));
    names.push_back("w" + std::to_string(i));
  }
  return Data(std::move(series), std::move(names));
}

Problem make_walk_problem(int n, int length)
{
  Problem problem("one_batch_walks");
  problem.set_data(walk_data<data_t>(std::vector<std::size_t>(static_cast<std::size_t>(n),
                                                              static_cast<std::size_t>(length))));
  return problem;
}

/// Holds the OpenMP thread count at `threads` for the scope (a no-op without OpenMP).
struct ThreadCount
{
#ifdef _OPENMP
  int previous = omp_get_max_threads();
  explicit ThreadCount(int threads) { omp_set_num_threads(threads); }
  ~ThreadCount() { omp_set_num_threads(previous); }
#else
  explicit ThreadCount(int) {}
#endif
};

#ifdef _OPENMP
int workers_granted()
{
  int workers = 1;
#pragma omp parallel
  {
#pragma omp single
    workers = omp_get_num_threads();
  }
  return workers;
}
#endif

/// The scan the final assignment replaced: each point against the medoids in
/// slot order, the first strictly nearest wins, a medoid tied with another
/// serves itself, the objective is point-ordered.
template <typename Distance>
core::ClusteringResult serial_assignment(std::size_t n, const std::vector<index_t> &medoids,
                                         Distance &&distance)
{
  core::ClusteringResult result;
  result.medoid_indices = medoids;
  result.labels.assign(n, 0);
  std::vector<double> cost(n, 0.0);
  for (std::size_t point = 0; point < n; ++point) {
    double best = std::numeric_limits<double>::infinity();
    for (std::size_t slot = 0; slot < medoids.size(); ++slot) {
      const auto medoid = static_cast<std::size_t>(medoids[slot]);
      const double d = point == medoid ? 0.0 : distance(point, medoid);
      if (d < best) { best = d; result.labels[point] = static_cast<index_t>(slot); }
    }
    cost[point] = best;
  }
  for (std::size_t slot = 0; slot < medoids.size(); ++slot) {
    const auto medoid = static_cast<std::size_t>(medoids[slot]);
    if (cost[medoid] == 0.0) result.labels[medoid] = static_cast<index_t>(slot);
  }
  result.total_cost = core::detail::ordered_medoid_objective(cost, "serial_assignment");
  return result;
}

core::ClusteringResult serial_assignment(Problem &problem, const std::vector<index_t> &medoids)
{
  const auto &dtw = problem.dtw_function();
  return serial_assignment(problem.size(), medoids, [&](std::size_t point, std::size_t medoid) {
    return dtw(problem.series(point), problem.series(medoid));
  });
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

TEST_CASE("OneBatchPAM computes with the Problem's variant in every worker",
          "[one_batch_pam][dtw_function][m37]")
{
  constexpr int replicas = 33; // N=66 takes the OpenMP table-build path.
  std::vector<std::vector<data_t>> series;
  std::vector<std::string> names;
  series.reserve(2 * replicas);
  names.reserve(2 * replicas);
  for (int replica = 0; replica < replicas; ++replica) {
    series.push_back({0.0, 0.0});
    names.push_back("short_" + std::to_string(replica));
    series.push_back({0.0, 1.0, 2.0});
    names.push_back("long_" + std::to_string(replica));
  }

  Problem problem{"one_batch_raw_dispatch_mutation"};
  problem.set_data(Data{std::move(series), std::move(names)});
  core::DTWVariantParams adtw;
  adtw.variant = core::DTWVariant::ADTW;
  adtw.adtw_penalty = 1.0;
  problem.set_variant(adtw);

  algorithms::OneBatchPAMOptions options;
  options.n_clusters = 1;
  options.batch_size = 2 * replicas;
  options.max_iter = 1;
  options.random_seed = 17;

  // Every medoid has 33 opposite-shape replicas at ADTW distance 4. Standard
  // DTW would report 33*3=99, so 132 pins the variant's use by all workers
  // without relying on a scheduler-specific race manifestation.
  const auto result = algorithms::one_batch_pam(problem, options);
  REQUIRE(result.total_cost == 132.0);
  REQUIRE_FALSE(problem.is_distance_matrix_filled());
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

    // Portable-v1 seed 0 has literal first order {1,0,2}, so the fixed batch
    // contains exactly the two near points. The primitive order is pinned in
    // unit_test_portable_random rather than searched through a vendor shuffle.
    constexpr std::uint64_t counterexample_seed = 0;

    algorithms::OneBatchPAMOptions options;
    options.n_clusters = 1;
    options.batch_size = 2;
    options.random_seed = counterexample_seed;

    const auto result = algorithms::one_batch_pam(problem, options);

    // The experiment-code finite-max correction normalizes the fixed table by
    // Dmax=0.1, so the far nonsampled point cannot win merely because all
    // off-diagonal distances are numerically below 1.
    REQUIRE(result.medoid_indices == std::vector<index_t>{0});
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
    algorithms::OneBatchPAMStats stats;

    const auto result = algorithms::one_batch_pam(problem, options, &stats);

    require_valid(result, 3, 1);
    REQUIRE(result.total_cost == 0.0);
    REQUIRE(std::isfinite(stats.estimated_objective));
  }
}

TEST_CASE("OneBatchPAM relative tolerance scales with the estimated cost",
          "[one_batch_pam][tolerance][regression]")
{
  Problem problem("one_batch_relative_tolerance");
  problem.set_data(Data(std::vector<std::vector<data_t>>{
                          {0.0}, {0.01}, {0.02}, {1.0}},
                        std::vector<std::string>{"zero", "one", "two", "far"}));

  algorithms::OneBatchPAMOptions options;
  options.n_clusters = 2;
  options.batch_size = 4;
  options.max_iter = 1;
  options.random_seed = 9;

  // Pin the portable seeded initial state by rejecting every improving swap.
  options.relative_tolerance = 1.0;
  algorithms::OneBatchPAMStats initial_stats;
  const auto initial = algorithms::one_batch_pam(
    problem, options, &initial_stats);
  REQUIRE(initial.medoid_indices == std::vector<index_t>{3, 2});
  REQUIRE(std::abs(initial.total_cost - 0.03) <= 1e-12);
  REQUIRE(initial_stats.accepted_swaps == 0);

  // Derived by hand. The batch is all four points, the table maximum is 1, and
  // every point is its own nearest batch point, so each NNIW weight is 1 and a
  // medoid's own column costs 1 (the finite-max diagonal correction). Columns
  // 0..3 then cost 0.02, 0.01, 0.98, 0.98: the estimate is 1.99. Candidate 0
  // replacing medoid 3 gains 0.96 (columns 0..3 fall to 0.02, 0.01, 0.02, 0.98,
  // estimate 1.03); no later candidate gains more than 0.02.
  REQUIRE(std::abs(initial_stats.estimated_objective - 1.99) <= 1e-12);

  // 0.2 * 1.99 = 0.398 < 0.96: accepted.
  options.relative_tolerance = 0.2;
  algorithms::OneBatchPAMStats stats;
  const auto result = algorithms::one_batch_pam(problem, options, &stats);
  REQUIRE(result.medoid_indices == std::vector<index_t>{0, 2});
  REQUIRE(std::abs(result.total_cost - 0.99) <= 1e-12);
  REQUIRE(stats.accepted_swaps == 1);
  REQUIRE(std::abs(stats.estimated_objective - 1.03) <= 1e-12);

  // 0.5 * 1.99 = 0.995 > 0.96: rejected, although the absolute gain exceeds 0.5.
  options.relative_tolerance = 0.5;
  algorithms::OneBatchPAMStats rejected_stats;
  const auto rejected = algorithms::one_batch_pam(
    problem, options, &rejected_stats);
  REQUIRE(rejected.medoid_indices == initial.medoid_indices);
  REQUIRE(std::abs(rejected.total_cost - initial.total_cost) <= 1e-12);
  REQUIRE(rejected_stats.accepted_swaps == 0);
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

TEST_CASE("OneBatchPAM refuses an explicit batch smaller than the cluster count",
          "[one_batch_pam][loudness][options]")
{
  SECTION("an explicit undersized batch is InvalidInput") {
    auto problem = make_problem(12, 4);
    algorithms::OneBatchPAMOptions options;
    options.n_clusters = 4;
    options.batch_size = 2;
    options.max_iter = 1;
    REQUIRE_THROWS_MATCHES(
      algorithms::one_batch_pam(problem, options), InvalidInput,
      Catch::Matchers::Message("one_batch_pam: batch_size must be at least n_clusters. "
                               "Got batch_size=2, n_clusters=4."));
    REQUIRE(problem.clusters_ind.empty()); // refused before any write-back
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
  // canonical-band-admissible L-path to any one-per-group representative
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

TEST_CASE("OneBatchPAM's final assignment is loud about a non-finite distance",
          "[one_batch_pam][nonfinite]")
{
  // A6: `FixedBatchDistances::exact` was the only distance read in the algorithm
  // layer that skipped `require_finite_medoid_distance`, and the objective used
  // a plain std::accumulate instead of `ordered_medoid_objective`. `exact()` is
  // only reached for a medoid that is NOT in the fixed batch — the one path the
  // constructor's finiteness check cannot cover. A non-finite d there makes
  // `d < best` false for every slot, so the point keeps label 0 and the run
  // publishes a silently wrong partition with a non-finite total_cost, where
  // fast_pam and fast_clara throw.
  //
  // Poison pair: series {+DBL_MAX} and {-DBL_MAX}. Their length-1 L1 DTW is
  // 2*DBL_MAX = +inf, while every other pair stays finite (<= DBL_MAX). If
  // either extreme lands in the batch the constructor already throws; the gap is
  // the seeds where both are outside the batch but one is chosen as a medoid.
  const double huge = std::numeric_limits<double>::max();
  std::vector<std::vector<data_t>> series{
    { 0.0 }, { 1.0 }, { 2.0 }, { 3.0 }, { huge }, { -huge }
  };
  std::vector<std::string> names{ "a", "b", "c", "d", "pos", "neg" };

  int returned = 0, threw = 0;
  for (std::uint64_t seed = 0; seed < 64; ++seed) {
    Problem problem("obpam_nonfinite");
    problem.set_data(Data(std::vector<std::vector<data_t>>(series),
                          std::vector<std::string>(names)));
    algorithms::OneBatchPAMOptions options;
    options.n_clusters = 2;
    options.batch_size = 2;
    options.random_seed = seed;
    try {
      const auto result = algorithms::one_batch_pam(problem, options);
      ++returned;
      INFO("seed " << seed << " returned total_cost " << result.total_cost);
      REQUIRE(std::isfinite(result.total_cost));
    } catch (const InvalidInput&) {
      ++threw; // loud rejection: the required behaviour
    }
  }
  // Non-vacuity: the poison configuration really is reachable in this sweep.
  // (A `returned + threw == 64` check would be tautological — every iteration
  // increments exactly one counter and any other exception escapes the loop.)
  REQUIRE(threw > 0);
}

TEST_CASE("OneBatchPAM's final assignment is the serial scan at every thread count",
          "[one_batch_pam][parallel][determinism]")
{
  constexpr int n = 200; // above the 64-point threshold of the OpenMP path
  constexpr int k = 5;
  constexpr int m = 40;
  algorithms::OneBatchPAMOptions options;
  options.n_clusters = k;
  options.batch_size = m;
  options.random_seed = 3;

  const auto run = [&](int threads, algorithms::OneBatchPAMStats &stats) {
    const ThreadCount scope(threads);
#ifdef _OPENMP
    // The subject must run in parallel: without this the two runs could both be serial.
    if (threads > 1) REQUIRE(workers_granted() > 1);
#endif
    auto problem = make_walk_problem(n, 24);
    return algorithms::one_batch_pam(problem, options, &stats);
  };

  algorithms::OneBatchPAMStats one_stats, four_stats;
  const auto one = run(1, one_stats);
  const auto four = run(4, four_stats);

  REQUIRE(four.medoid_indices == one.medoid_indices);
  REQUIRE(four.labels == one.labels);
  REQUIRE(four.total_cost == one.total_cost);
  REQUIRE(four_stats.distance_evaluations == one_stats.distance_evaluations);

  // Both equal the serial scan over the published medoids.
  auto oracle_problem = make_walk_problem(n, 24);
  const auto oracle = serial_assignment(oracle_problem, one.medoid_indices);
  REQUIRE(one.labels == oracle.labels);
  REQUIRE(one.total_cost == oracle.total_cost);

  // The table costs m(N-1) calls and each medoid outside the batch (N-1) more:
  // at least one such medoid makes the exact-DTW branch the subject of this test.
  const std::uint64_t evaluations = one_stats.distance_evaluations;
  REQUIRE(evaluations % (n - 1) == 0);
  const std::uint64_t outside = evaluations / (n - 1) - m;
  REQUIRE(outside >= 1);
  REQUIRE(outside <= k);
}

TEST_CASE("OneBatchPAM's parallel final assignment reports the failure a serial scan meets first",
          "[one_batch_pam][parallel][nonfinite]")
{
  // The poison pair {+DBL_MAX} and {-DBL_MAX} has DTW +inf, every other pair is
  // finite. Rejecting every swap keeps the seeded initial medoids, so a seed
  // that draws a poison point as a medoid, with both poison points outside the
  // batch, fails in the assignment (the table check cannot see it). 70 points
  // take the OpenMP path.
  const double huge = std::numeric_limits<double>::max();
  constexpr int n = 70;
  std::vector<std::vector<data_t>> series;
  std::vector<std::string> names;
  for (int i = 0; i < n - 2; ++i) {
    series.push_back({ 0.001 * i });
    names.push_back("s" + std::to_string(i));
  }
  series.push_back({ huge });
  names.push_back("pos");
  series.push_back({ -huge });
  names.push_back("neg");

  const auto outcome = [&](int threads, std::uint64_t seed) {
    const ThreadCount scope(threads);
    Problem problem("obpam_parallel_nonfinite");
    problem.set_data(Data(std::vector<std::vector<data_t>>(series), std::vector<std::string>(names)));
    algorithms::OneBatchPAMOptions options;
    options.n_clusters = 2;
    options.batch_size = 4;
    options.relative_tolerance = 1e300;
    options.random_seed = seed;
    try {
      (void)algorithms::one_batch_pam(problem, options);
    } catch (const InvalidInput &error) {
      return std::string(error.what());
    }
    return std::string();
  };

  int failed_in_assignment = 0;
  for (std::uint64_t seed = 0; seed < 300; ++seed) {
    const std::string serial = outcome(1, seed);
    INFO("seed " << seed);
    REQUIRE(outcome(4, seed) == serial);
    if (serial.find("nearest-medoid distance at point") != std::string::npos) ++failed_in_assignment;
  }
  REQUIRE(failed_in_assignment > 0);
}

namespace {

bool same_bits(double a, double b)
{
  return std::memcmp(&a, &b, sizeof(double)) == 0;
}

} // namespace

TEST_CASE("OneBatchPAM's batch table through the lanes is bitwise the per-pair table",
          "[one_batch_pam][lanes]")
{
  // The table fill takes W columns of a row at a time through the lane function
  // (W = 8 for float64, 16 for float32) where they are as long as the row's
  // series, and every other column pair by pair. With the batch the whole data
  // set, every selected medoid is a table column and the final labels and cost
  // read N x k entries of the table: each must be the bits of the per-pair
  // function, whichever lane its column fell in. The batch order is the seed's,
  // so the seeds move the columns across the lanes and the blocks.
  struct FillCase
  {
    const char *name;
    bool f32;
    std::vector<std::size_t> lengths; // one per series
    int band;
    core::MetricType metric;
  };
  constexpr std::size_t n = 70; // above the 64-row threshold of the OpenMP path
  std::vector<std::size_t> alternating, mostly_one;
  for (std::size_t i = 0; i < n; ++i) alternating.push_back(i % 2 ? 53 : 50);
  for (std::size_t i = 0; i < n; ++i) mostly_one.push_back(i % 9 == 4 ? 53 : 50);
  const std::vector<FillCase> cases{
    { "float64, equal lengths, full", false, std::vector<std::size_t>(n, 50), -1, core::MetricType::L1 },
    { "float64, equal lengths, band 6, squared L2", false, std::vector<std::size_t>(n, 50), 6,
      core::MetricType::SquaredL2 },
    { "float32, equal lengths, full", true, std::vector<std::size_t>(n, 50), -1, core::MetricType::L1 },
    { "float32, equal lengths, band 4, squared L2", true, std::vector<std::size_t>(n, 50), 4,
      core::MetricType::SquaredL2 },
    { "float64, two lengths half and half", false, alternating, 3, core::MetricType::L1 },
    { "float64, two lengths, some blocks of one length", false, mostly_one, -1, core::MetricType::L1 },
    { "float32, two lengths, some blocks of one length", true, mostly_one, 5,
      core::MetricType::SquaredL2 },
  };

  for (const auto &c : cases) {
    for (std::uint64_t seed = 1; seed <= 4; ++seed) {
      CAPTURE(c.name, seed);
      Problem problem("one_batch_lanes");
      problem.set_data(c.f32 ? walk_data<float>(c.lengths) : walk_data<data_t>(c.lengths));
      problem.set_band(c.band);
      problem.set_metric(c.metric);
      algorithms::OneBatchPAMOptions options;
      options.n_clusters = 10;
      options.batch_size = static_cast<index_t>(n);
      options.random_seed = seed;
      algorithms::OneBatchPAMStats stats;
      const auto result = algorithms::one_batch_pam(problem, options, &stats);

      const auto oracle = serial_assignment(n, result.medoid_indices,
        [&](std::size_t point, std::size_t medoid) {
          return c.f32 ? problem.dtw_function_f32()(problem.data().series_f32(point),
                                                    problem.data().series_f32(medoid))
                       : problem.dtw_function()(problem.series(point), problem.series(medoid));
        });
      CHECK(result.labels == oracle.labels);
      CHECK(same_bits(result.total_cost, oracle.total_cost));
      // The N x N table minus its diagonal; every medoid is in the batch.
      CHECK(stats.distance_evaluations == n * (n - 1));
    }
  }
}
