/**
 * @file unit_test_mip.cpp
 * @brief Tests for MIP solver improvements: warm start, settings, correctness.
 *
 * @details Backend-neutral state tests run in every build. On machines with
 * HiGHS enabled, the tests also verify exact-solver result quality; no-solver
 * builds exercise the typed unavailable-backend failure contract.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#include <dtwc.hpp>
#include <mip/mip.hpp>
#include <mip/decode_assignment.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>
#include <vector>

/// Build a small Problem with N synthetic series of length L.
static dtwc::Problem make_small_problem(int N, int L)
{
  dtwc::Data data;
  data.p_vec.resize(static_cast<size_t>(N));
  data.p_names.resize(static_cast<size_t>(N));

  for (int i = 0; i < N; ++i) {
    data.p_vec[i].resize(static_cast<size_t>(L));
    for (int t = 0; t < L; ++t)
      data.p_vec[i][t] = std::sin(static_cast<double>(i) + static_cast<double>(t) / L * 6.283185307);
    data.p_names[i] = "s" + std::to_string(i);
  }

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  return prob;
}

static void require_highs_solver()
{
  if (!dtwc::highs_solver_available())
    SKIP("HiGHS is not compiled into this build.");
}

static dtwc::Problem make_seed_sensitive_problem()
{
  const std::vector<double> base{0.0, 0.01, -0.02, 0.03};
  std::vector<std::vector<double>> series;
  std::vector<std::string> names;
  for (int offset = 0; offset < 8; ++offset) {
    auto waveform = base;
    for (double &value : waveform) value += static_cast<double>(offset);
    series.push_back(std::move(waveform));
    names.push_back(std::to_string(offset));
  }
  dtwc::Problem prob("mip_seed_fixture");
  prob.set_data(dtwc::Data(std::move(series), std::move(names)));
  prob.set_n_clusters(3);
  return prob;
}

/// Facilities 1 and 3 open; points 0, 1 -> 1 and 2, 3 -> 3.
static std::vector<double> exact_assignment_fixture(bool point_major)
{
  constexpr std::size_t n_points = 4;
  std::vector<double> values(n_points * n_points, 0.0);
  auto index = [point_major](std::size_t facility, std::size_t point) {
    return point_major ? facility + point * n_points : facility * n_points + point;
  };
  values[index(1, 0)] = 1.0;
  values[index(1, 1)] = 1.0;
  values[index(3, 2)] = 1.0;
  values[index(3, 3)] = 1.0;
  return values;
}

static void require_valid_exact_clustering(
  const dtwc::Problem &problem, int n_clusters)
{
  REQUIRE(problem.centroids_ind.size() == static_cast<std::size_t>(n_clusters));
  auto sorted_medoids = problem.centroids_ind;
  std::sort(sorted_medoids.begin(), sorted_medoids.end());
  REQUIRE(std::adjacent_find(sorted_medoids.begin(), sorted_medoids.end())
          == sorted_medoids.end());
  for (dtwc::index_t medoid : problem.centroids_ind)
    REQUIRE((medoid >= 0 && static_cast<std::size_t>(medoid) < problem.size()));

  REQUIRE(problem.clusters_ind.size() == problem.size());
  for (dtwc::index_t label : problem.clusters_ind)
    REQUIRE((label >= 0 && label < n_clusters));
  for (std::size_t cluster = 0; cluster < problem.centroids_ind.size(); ++cluster) {
    const auto medoid = static_cast<std::size_t>(problem.centroids_ind[cluster]);
    REQUIRE(problem.clusters_ind[medoid] == static_cast<int>(cluster));
  }
}

TEST_CASE("MIPSettings: struct has correct defaults", "[mip]")
{
  dtwc::MIPSettings s;
  REQUIRE(s.mip_gap == 1e-5);
  REQUIRE(s.time_limit_sec == -1);
  REQUIRE(s.warm_start == true);
  REQUIRE(s.numeric_focus == 1);
  REQUIRE(s.mip_focus == 2);
  REQUIRE(s.verbose_solver == false);
}

TEST_CASE("The MIP warm start (seeded FastPAM) ignores the legacy RNG",
          "[mip][seed][warm-start]")
{
  const auto legacy_rng_original = dtwc::randGenerator;

  dtwc::randGenerator.seed(17);
  const auto legacy_rng_before_first = dtwc::randGenerator;
  auto first_problem = make_seed_sensitive_problem();
  const auto first = dtwc::fast_pam(
    first_problem, 3, dtwc::settings::DEFAULT_MAX_ITER, dtwc::settings::DEFAULT_RANDOM_SEED);
  CHECK(dtwc::randGenerator == legacy_rng_before_first);

  dtwc::randGenerator.seed(8675309);
  const auto legacy_rng_before_second = dtwc::randGenerator;
  auto second_problem = make_seed_sensitive_problem();
  const auto second = dtwc::fast_pam(
    second_problem, 3, dtwc::settings::DEFAULT_MAX_ITER, dtwc::settings::DEFAULT_RANDOM_SEED);
  CHECK(dtwc::randGenerator == legacy_rng_before_second);

  CHECK(first.medoid_indices == second.medoid_indices);
  CHECK(first.labels == second.labels);
  CHECK(first.total_cost == second.total_cost);
  CHECK(first.medoid_indices == std::vector<dtwc::index_t>{6, 2, 5});
  CHECK(first.total_cost == 24.0);

  auto override_problem = make_seed_sensitive_problem();
  const auto override_result = dtwc::fast_pam(
    override_problem, 3, dtwc::settings::DEFAULT_MAX_ITER, 43);
  CHECK(override_result.medoid_indices == std::vector<dtwc::index_t>{6, 4, 1});
  CHECK(override_result.total_cost == 20.0);

  dtwc::randGenerator = legacy_rng_original;
}

TEST_CASE("decode_assignment reads both solver matrix layouts", "[mip][decode]")
{
  for (const bool point_major : { false, true }) {
    INFO("point_major=" << point_major);
    const auto decoded = dtwc::mip::decode_assignment(
      exact_assignment_fixture(point_major), 4, 2, point_major, "test backend");
    CHECK(decoded.medoid_indices == std::vector<dtwc::index_t>{1, 3});
    CHECK(decoded.labels == std::vector<dtwc::index_t>{0, 0, 1, 1});

    auto problem = make_small_problem(4, 8);
    problem.set_result(decoded);
    require_valid_exact_clustering(problem, 2);
  }
}

TEST_CASE("decode_assignment refuses what is not a p-median solution", "[mip][decode]")
{
  auto twice = exact_assignment_fixture(false);
  twice[1 * 4 + 2] = 1.0; // point 2 served by facilities 1 and 3
  CHECK_THROWS_AS(dtwc::mip::decode_assignment(twice, 4, 2, false, "test"), dtwc::SolverError);

  auto closed = exact_assignment_fixture(false);
  closed[3 * 4 + 3] = 0.0; // facility 3 closed but still serving point 2
  closed[1 * 4 + 3] = 1.0;
  CHECK_THROWS_AS(dtwc::mip::decode_assignment(closed, 4, 1, false, "test"), dtwc::SolverError);

  CHECK_THROWS_AS(dtwc::mip::decode_assignment(exact_assignment_fixture(false), 4, 3, false, "test"),
                  dtwc::SolverError); // two medoids for k = 3

  auto nan = exact_assignment_fixture(true);
  nan[1 + 0 * 4] = std::nan("");
  CHECK_THROWS_AS(dtwc::mip::decode_assignment(nan, 4, 2, true, "test"), dtwc::SolverError);
}

TEST_CASE("Problem::set_result publishes only a well-formed clustering", "[mip][state]")
{
  auto problem = make_small_problem(4, 8);
  problem.set_n_clusters(2);
  problem.centroids_ind = {0, 2};
  problem.clusters_ind = {0, 0, 1, 1};
  const auto medoids_before = problem.centroids_ind;
  const auto labels_before = problem.clusters_ind;

  dtwc::core::ClusteringResult duplicate;
  duplicate.medoid_indices = {1, 1};
  duplicate.labels = {0, 0, 1, 1};
  CHECK_THROWS_AS(problem.set_result(duplicate), dtwc::InvalidInput);

  dtwc::core::ClusteringResult label_out_of_range;
  label_out_of_range.medoid_indices = {1, 3};
  label_out_of_range.labels = {0, 0, 2, 1};
  CHECK_THROWS_AS(problem.set_result(label_out_of_range), dtwc::InvalidInput);

  dtwc::core::ClusteringResult short_labels;
  short_labels.medoid_indices = {1, 3};
  short_labels.labels = {0, 1};
  CHECK_THROWS_AS(problem.set_result(short_labels), dtwc::InvalidInput);

  CHECK(problem.centroids_ind == medoids_before);
  CHECK(problem.clusters_ind == labels_before);

  dtwc::core::ClusteringResult valid;
  valid.medoid_indices = {3, 1, 0};
  valid.labels = {2, 1, 0, 0};
  problem.set_result(valid);
  CHECK(problem.n_clusters() == 3);
  CHECK(problem.centroids_ind == valid.medoid_indices);
  CHECK(problem.clusters_ind == valid.labels);
}

TEST_CASE("Unavailable direct HiGHS leaves caller clustering state unchanged",
          "[mip][state][m33][no-solver]")
{
  if (dtwc::highs_solver_available()) {
    SUCCEED("HiGHS is compiled in; the unavailable-backend branch is not active.");
    return;
  }

  auto problem = make_small_problem(4, 8);
  problem.set_n_clusters(2);
  problem.centroids_ind = {0, 2};
  problem.clusters_ind = {0, 0, 1, 1};
  REQUIRE(problem.set_solver(dtwc::Solver::HiGHS));
  problem.set_method(dtwc::Method::MIP);
  const auto medoids_before = problem.centroids_ind;
  const auto labels_before = problem.clusters_ind;

  REQUIRE_THROWS_AS(problem.cluster(), dtwc::SolverError);
  CHECK(problem.centroids_ind == medoids_before);
  CHECK(problem.clusters_ind == labels_before);
}

TEST_CASE("MIP HiGHS: warm start produces valid result", "[mip][highs]")
{
  require_highs_solver();
  const auto legacy_rng_original = dtwc::randGenerator;
  dtwc::randGenerator.seed(314159);
  const auto legacy_rng_before = dtwc::randGenerator;
  auto prob = make_small_problem(8, 20);
  prob.set_n_clusters(2);
  prob.mip_settings.warm_start = true;
  prob.mip_settings.time_limit_sec = 30; // the time_limit option is passed to HiGHS only when positive
  prob.mip_settings.verbose_solver = false;
  REQUIRE(prob.set_solver(dtwc::Solver::HiGHS));
  prob.set_method(dtwc::Method::MIP);
  prob.centroids_ind = {0, 7};
  prob.clusters_ind = {0, 0, 0, 0, 1, 1, 1, 1};
  prob.cluster();
  CHECK(dtwc::randGenerator == legacy_rng_before);
  dtwc::randGenerator = legacy_rng_original;

  require_valid_exact_clustering(prob, 2);
}

TEST_CASE("MIP HiGHS: cold start matches warm start cost", "[mip][highs]")
{
  require_highs_solver();
  auto prob1 = make_small_problem(8, 20);
  prob1.set_n_clusters(2);
  prob1.mip_settings.warm_start = false;
  prob1.mip_settings.verbose_solver = false;
  REQUIRE(prob1.set_solver(dtwc::Solver::HiGHS));
  prob1.set_method(dtwc::Method::MIP);
  prob1.cluster();
  double cold_cost = prob1.find_total_cost();

  auto prob2 = make_small_problem(8, 20);
  prob2.set_n_clusters(2);
  prob2.mip_settings.warm_start = true;
  prob2.mip_settings.verbose_solver = false;
  REQUIRE(prob2.set_solver(dtwc::Solver::HiGHS));
  prob2.set_method(dtwc::Method::MIP);
  prob2.cluster();
  double warm_cost = prob2.find_total_cost();

  // Both should find the same global optimum (small instance)
  REQUIRE(warm_cost <= cold_cost + 1e-6);
}

TEST_CASE("MIP HiGHS: k=1 trivial case", "[mip][highs]")
{
  require_highs_solver();
  auto prob = make_small_problem(5, 10);
  prob.set_n_clusters(1);
  prob.mip_settings.warm_start = true;
  prob.mip_settings.verbose_solver = false;
  REQUIRE(prob.set_solver(dtwc::Solver::HiGHS));
  prob.set_method(dtwc::Method::MIP);
  prob.cluster();

  REQUIRE(prob.centroids_ind.size() == 1);
  for (auto c : prob.clusters_ind)
    REQUIRE(c == 0);
}

TEST_CASE("MIP HiGHS: non-optimal (infeasible) solve throws, not silent empty result", "[mip][highs]")
{
  require_highs_solver();
  // Infeasible instance: 4 series but 5 clusters requested. warm_start=false so
  // we bypass fast_pam (which independently rejects k>N) and drive the solver
  // status path directly.
  auto prob = make_small_problem(4, 12);
  prob.set_n_clusters(5);
  prob.mip_settings.warm_start = false;
  prob.mip_settings.verbose_solver = false;
  REQUIRE(prob.set_solver(dtwc::Solver::HiGHS));
  prob.set_method(dtwc::Method::MIP);

  REQUIRE_THROWS_AS(prob.cluster(), std::runtime_error);
}
