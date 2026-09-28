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
#include <mip/solution_transaction.hpp>
#include <mip/warm_start.hpp>

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

static std::vector<double> exact_assignment_fixture(
  dtwc::mip::AssignmentMatrixLayout layout)
{
  constexpr std::size_t n_points = 4;
  std::vector<double> values(n_points * n_points, 0.0);
  auto index = [layout](std::size_t facility, std::size_t point) {
    if (layout == dtwc::mip::AssignmentMatrixLayout::FacilityMajor)
      return facility * n_points + point;
    return facility + point * n_points;
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
  for (int medoid : problem.centroids_ind)
    REQUIRE((medoid >= 0 && static_cast<std::size_t>(medoid) < problem.size()));

  REQUIRE(problem.clusters_ind.size() == problem.size());
  for (int label : problem.clusters_ind)
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

TEST_CASE("MIPSettings: Problem member accessible", "[mip]")
{
  auto prob = make_small_problem(4, 10);
  prob.mip_settings.mip_gap = 0.01;
  prob.mip_settings.warm_start = false;
  prob.mip_settings.time_limit_sec = 60;
  REQUIRE(prob.mip_settings.mip_gap == 0.01);
  REQUIRE(prob.mip_settings.warm_start == false);
  REQUIRE(prob.mip_settings.time_limit_sec == 60);
}

TEST_CASE("MIP FastPAM warm-start medoids are invocation-local",
          "[mip][seed][warm-start]")
{
  const auto legacy_rng_original = dtwc::randGenerator;

  dtwc::randGenerator.seed(17);
  const auto legacy_rng_before_first = dtwc::randGenerator;
  auto first_problem = make_seed_sensitive_problem();
  first_problem.centroids_ind = {0, 3, 7};
  first_problem.clusters_ind = {0, 0, 0, 1, 1, 1, 2, 2};
  const auto first_medoids_before = first_problem.centroids_ind;
  const auto first_labels_before = first_problem.clusters_ind;
  const auto first = dtwc::mip::make_warm_start(
    first_problem, dtwc::settings::DEFAULT_RANDOM_SEED);
  CHECK(dtwc::randGenerator == legacy_rng_before_first);
  CHECK(first_problem.centroids_ind == first_medoids_before);
  CHECK(first_problem.clusters_ind == first_labels_before);

  dtwc::randGenerator.seed(8675309);
  const auto legacy_rng_before_second = dtwc::randGenerator;
  auto second_problem = make_seed_sensitive_problem();
  const auto second = dtwc::mip::make_warm_start(
    second_problem, dtwc::settings::DEFAULT_RANDOM_SEED);
  CHECK(dtwc::randGenerator == legacy_rng_before_second);

  CHECK(first.medoid_indices == second.medoid_indices);
  CHECK(first.labels == second.labels);
  CHECK(first.total_cost == second.total_cost);
  CHECK(first.medoid_indices == std::vector<int>{6, 2, 5});
  CHECK(first.total_cost == 24.0);

  auto override_problem = make_seed_sensitive_problem();
  const auto override_result = dtwc::mip::make_warm_start(override_problem, 43);
  CHECK(override_result.medoid_indices == std::vector<int>{6, 4, 1});
  CHECK(override_result.total_cost == 20.0);

  dtwc::randGenerator = legacy_rng_original;
}

TEST_CASE("Direct MIP exact publication replaces state for both matrix layouts",
          "[mip][state][m33]")
{
  auto publish_fixture = [](dtwc::mip::AssignmentMatrixLayout layout) {
    auto problem = make_small_problem(4, 8);
    problem.set_n_clusters(2);
    problem.centroids_ind = {0, 2};
    problem.clusters_ind = {0, 0, 1, 1};

    {
      dtwc::mip::ExactClusteringTransaction transaction(problem);
      auto exact = dtwc::mip::extract_exact_clustering(
        exact_assignment_fixture(layout), 4, 2, layout, "test backend");
      transaction.publish(std::move(exact), "test backend");
    }

    CHECK(problem.centroids_ind == std::vector<int>{1, 3});
    CHECK(problem.clusters_ind == std::vector<int>{0, 0, 1, 1});
    require_valid_exact_clustering(problem, 2);
  };

  SECTION("HiGHS facility-major layout")
  {
    publish_fixture(dtwc::mip::AssignmentMatrixLayout::FacilityMajor);
  }
  SECTION("Gurobi point-major layout")
  {
    publish_fixture(dtwc::mip::AssignmentMatrixLayout::PointMajor);
  }
}

TEST_CASE("Direct MIP transaction restores state on forced backend failures",
          "[mip][state][m33]")
{
  auto problem = make_small_problem(4, 8);
  problem.set_n_clusters(2);
  problem.centroids_ind = {0, 2};
  problem.clusters_ind = {0, 0, 1, 1};
  const auto medoids_before = problem.centroids_ind;
  const auto labels_before = problem.clusters_ind;

  auto expose_incumbent = [&problem] {
    problem.centroids_ind = {1, 3};
    problem.clusters_ind = {0, 0, 1, 1};
  };

  SECTION("solve failure after warm-start state")
  {
    auto force_failure = [&] {
      dtwc::mip::ExactClusteringTransaction transaction(problem);
      expose_incumbent();
      throw dtwc::SolverError("forced backend solve failure");
    };
    REQUIRE_THROWS_AS(force_failure(), dtwc::SolverError);
  }

  SECTION("extraction failure after warm-start state")
  {
    auto force_failure = [&] {
      dtwc::mip::ExactClusteringTransaction transaction(problem);
      expose_incumbent();
      auto invalid = exact_assignment_fixture(
        dtwc::mip::AssignmentMatrixLayout::FacilityMajor);
      invalid[1 * 4 + 2] = 1.0;
      (void)dtwc::mip::extract_exact_clustering(
        invalid,
        4,
        2,
        dtwc::mip::AssignmentMatrixLayout::FacilityMajor,
        "forced extraction");
    };
    REQUIRE_THROWS_AS(force_failure(), dtwc::SolverError);
  }

  SECTION("publication rejects duplicate medoids")
  {
    auto force_failure = [&] {
      dtwc::mip::ExactClusteringTransaction transaction(problem);
      expose_incumbent();
      dtwc::core::ClusteringResult invalid;
      invalid.medoid_indices = {1, 1};
      invalid.labels = {0, 0, 1, 1};
      transaction.publish(std::move(invalid), "forced publication");
    };
    REQUIRE_THROWS_AS(force_failure(), dtwc::SolverError);
  }

  CHECK(problem.centroids_ind == medoids_before);
  CHECK(problem.clusters_ind == labels_before);
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
  prob.mip_settings.verbose_solver = false;
  REQUIRE(prob.set_solver(dtwc::Solver::HiGHS));
  prob.set_method(dtwc::Method::MIP);
  prob.centroids_ind = {0, 7};
  prob.clusters_ind = {0, 0, 0, 0, 1, 1, 1, 1};
  prob.cluster();
  CHECK(dtwc::randGenerator == legacy_rng_before);
  dtwc::randGenerator = legacy_rng_original;

  // If HiGHS is not compiled in, cluster() prints a warning and returns
  // with empty centroids_ind. Only check if solver actually ran.
  if (!prob.centroids_ind.empty()) {
    require_valid_exact_clustering(prob, 2);
  }
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

  // Skip if HiGHS not available
  if (prob1.centroids_ind.empty()) return;

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

TEST_CASE("MIP HiGHS: settings propagate without crash", "[mip][highs]")
{
  require_highs_solver();
  auto prob = make_small_problem(6, 15);
  prob.set_n_clusters(2);
  prob.mip_settings.mip_gap = 0.01;
  prob.mip_settings.time_limit_sec = 30;
  prob.mip_settings.verbose_solver = false;
  REQUIRE(prob.set_solver(dtwc::Solver::HiGHS));
  prob.set_method(dtwc::Method::MIP);

  REQUIRE_NOTHROW(prob.cluster());
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

  if (!prob.centroids_ind.empty()) {
    REQUIRE(prob.centroids_ind.size() == 1);
    for (auto c : prob.clusters_ind)
      REQUIRE(c == 0);
  }
}

// ---------------------------------------------------------------------------
// Task 0.5 regression: MIP status handling — assert() → real error path.
//
// mip_Highs.cpp guarded the HiGHS model status with
//     assert(model_status == HighsModelStatus::kOptimal);
// which is a NO-OP under NDEBUG (release builds). A non-optimal solve therefore
// fell through to extract_mip_solution(), which reads an empty/invalid solution
// vector and returns garbage / empty centroids (and indexes an empty
// centroids_ind out of bounds -> UB). The fix replaces the assert with an
// explicit status check that throws std::runtime_error carrying the solver
// status text. This test drives an INFEASIBLE p-median instance (more clusters
// than series, so the cardinality constraint "sum of N diagonal binaries == k"
// with k > N cannot hold) and requires a throw.
//
// Why the UNFIXED code fails this test: pre-fix, MIP_clustering_byHiGHS returns
// normally (assert compiled out) instead of throwing, so REQUIRE_THROWS_AS
// fails. Post-fix it throws. (Gurobi's analogous status/catch path is fixed the
// same way but is not exercised here — Gurobi requires a licensed install that
// this build does not have; the HiGHS path is the one wired in CI.)
// ---------------------------------------------------------------------------
TEST_CASE("MIP HiGHS: non-optimal (infeasible) solve throws, not silent empty result", "[mip][highs]")
{
  require_highs_solver();
  // DTWC_ENABLE_HIGHS is defined PUBLIC on the mip-solvers object library, which
  // links PRIVATE into dtwc++, so the macro is NOT visible in this test TU.
  // Detect HiGHS availability at runtime: a feasible instance produces a
  // non-empty result only when HiGHS is actually compiled in (otherwise the
  // #else branch merely warns and returns empty).
  {
    auto probe = make_small_problem(6, 15);
    probe.set_n_clusters(2);
    probe.mip_settings.warm_start = false;
    probe.mip_settings.verbose_solver = false;
    REQUIRE(probe.set_solver(dtwc::Solver::HiGHS));
    probe.set_method(dtwc::Method::MIP);
    probe.cluster();
    if (probe.centroids_ind.empty())
      return; // HiGHS not compiled into this build — nothing to exercise.
    REQUIRE(probe.centroids_ind.size() == 2); // sanity: the solver really ran.
  }

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
