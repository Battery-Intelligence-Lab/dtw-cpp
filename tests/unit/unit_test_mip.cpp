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
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <typeindex>
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

class ScopedCoutCapture {
public:
  ScopedCoutCapture() : previous_(std::cout.rdbuf(output_.rdbuf())) {}
  ScopedCoutCapture(const ScopedCoutCapture &) = delete;
  ScopedCoutCapture &operator=(const ScopedCoutCapture &) = delete;
  ~ScopedCoutCapture() { std::cout.rdbuf(previous_); }

  std::string str() const { return output_.str(); }

private:
  std::ostringstream output_;
  std::streambuf *previous_;
};

using ProblemInitializerFunction = void (*)(dtwc::Problem &);

struct ProblemConfigurationSnapshot {
  dtwc::Method method;
  int max_iter;
  int n_repetitions;
  std::uint64_t random_seed;
  int last_iterations;
  int band;
  double tadpole_dc;
  dtwc::core::DTWVariantParams variant_params;
  dtwc::core::MissingStrategy missing_strategy;
  dtwc::DistanceMatrixStrategy distance_strategy;
  dtwc::LowerBoundStrategy lb_strategy;
  dtwc::core::StoragePolicy storage_policy;
  dtwc::CUDASettings cuda_settings;
  dtwc::MIPSettings mip_settings;
  bool verbose;
  std::type_index initializer_type;
  bool initializer_is_function_pointer;
  ProblemInitializerFunction initializer_function;
  std::filesystem::path output_folder;
  std::string name;
  int n_clusters;
  std::size_t data_size;
  std::size_t data_ndim;
  dtwc::core::Precision data_precision;
  bool data_is_view;
  bool data_is_metadata_only;
  std::vector<std::vector<double>> series;
  std::vector<std::vector<float>> series_f32;
  std::vector<std::string> series_names;
  std::size_t distance_matrix_index;
  bool distance_matrix_filled;
  std::vector<double> distances;
};

static ProblemConfigurationSnapshot snapshot_configuration(dtwc::Problem &prob)
{
  const auto *initializer = prob.init_fun.target<ProblemInitializerFunction>();
  std::vector<double> distances;
  distances.reserve(prob.size() * prob.size());
  for (std::size_t i = 0; i < prob.size(); ++i)
    for (std::size_t j = 0; j < prob.size(); ++j)
      distances.push_back(prob.dist_by_ind(static_cast<int>(i), static_cast<int>(j)));

  return {
    prob.method,
    prob.maxIter,
    prob.N_repetition,
    prob.random_seed,
    prob.last_iterations,
    prob.band,
    prob.tadpole_dc,
    prob.variant_params,
    prob.missing_strategy,
    prob.distance_strategy,
    prob.lb_strategy,
    prob.storage_policy,
    prob.cuda_settings,
    prob.mip_settings,
    prob.verbose,
    std::type_index(prob.init_fun.target_type()),
    initializer != nullptr,
    initializer == nullptr ? nullptr : *initializer,
    prob.output_folder,
    prob.name,
    prob.n_clusters(),
    prob.size(),
    prob.data.ndim,
    prob.data.precision,
    prob.data.is_view(),
    prob.data.is_metadata_only(),
    prob.data.p_vec,
    prob.data.p_vec_f32,
    prob.data.p_names,
    prob.distance_matrix().index(),
    prob.is_distance_matrix_filled(),
    std::move(distances)
  };
}

static void check_configuration_unchanged(
  dtwc::Problem &prob, const ProblemConfigurationSnapshot &before)
{
  CHECK(prob.method == before.method);
  CHECK(prob.maxIter == before.max_iter);
  CHECK(prob.N_repetition == before.n_repetitions);
  CHECK(prob.random_seed == before.random_seed);
  CHECK(prob.last_iterations == before.last_iterations);
  CHECK(prob.band == before.band);
  CHECK(prob.tadpole_dc == before.tadpole_dc);
  CHECK(prob.variant_params.variant == before.variant_params.variant);
  CHECK(prob.variant_params.wdtw_g == before.variant_params.wdtw_g);
  CHECK(prob.variant_params.adtw_penalty == before.variant_params.adtw_penalty);
  CHECK(prob.variant_params.sdtw_gamma == before.variant_params.sdtw_gamma);
  CHECK(prob.variant_params.msm_c == before.variant_params.msm_c);
  CHECK(prob.variant_params.twe_nu == before.variant_params.twe_nu);
  CHECK(prob.variant_params.twe_lambda == before.variant_params.twe_lambda);
  CHECK(prob.variant_params.mv_mode == before.variant_params.mv_mode);
  CHECK(prob.missing_strategy == before.missing_strategy);
  CHECK(prob.distance_strategy == before.distance_strategy);
  CHECK(prob.lb_strategy == before.lb_strategy);
  CHECK(prob.storage_policy == before.storage_policy);
  CHECK(prob.cuda_settings.device_id == before.cuda_settings.device_id);
  CHECK(prob.cuda_settings.precision == before.cuda_settings.precision);
  CHECK(prob.mip_settings.mip_gap == before.mip_settings.mip_gap);
  CHECK(prob.mip_settings.time_limit_sec == before.mip_settings.time_limit_sec);
  CHECK(prob.mip_settings.warm_start == before.mip_settings.warm_start);
  CHECK(prob.mip_settings.numeric_focus == before.mip_settings.numeric_focus);
  CHECK(prob.mip_settings.mip_focus == before.mip_settings.mip_focus);
  CHECK(prob.mip_settings.verbose_solver == before.mip_settings.verbose_solver);
  CHECK(prob.mip_settings.max_benders_iter == before.mip_settings.max_benders_iter);
  CHECK(prob.mip_settings.benders == before.mip_settings.benders);
  CHECK(prob.verbose == before.verbose);
  const auto *initializer = prob.init_fun.target<ProblemInitializerFunction>();
  CHECK((std::type_index(prob.init_fun.target_type()) == before.initializer_type
         && (initializer != nullptr) == before.initializer_is_function_pointer
         && (initializer == nullptr || *initializer == before.initializer_function)));
  CHECK(prob.output_folder == before.output_folder);
  CHECK(prob.name == before.name);
  CHECK(prob.n_clusters() == before.n_clusters);
  CHECK((prob.size() == before.data_size
         && prob.data.is_view() == before.data_is_view
         && prob.data.is_metadata_only() == before.data_is_metadata_only));
  CHECK(prob.data.ndim == before.data_ndim);
  CHECK(prob.data.precision == before.data_precision);
  CHECK((prob.data.p_vec == before.series && prob.data.p_vec_f32 == before.series_f32));
  CHECK(prob.data.p_names == before.series_names);

  std::vector<double> distances;
  distances.reserve(prob.size() * prob.size());
  for (std::size_t i = 0; i < prob.size(); ++i)
    for (std::size_t j = 0; j < prob.size(); ++j)
      distances.push_back(prob.dist_by_ind(static_cast<int>(i), static_cast<int>(j)));
  CHECK((distances == before.distances
         && prob.distance_matrix().index() == before.distance_matrix_index
         && prob.is_distance_matrix_filled() == before.distance_matrix_filled));
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
  problem.mip_settings.benders = "off";
  problem.set_solver(dtwc::Solver::HiGHS);
  problem.method = dtwc::Method::MIP;
  const auto medoids_before = problem.centroids_ind;
  const auto labels_before = problem.clusters_ind;

  REQUIRE_THROWS_AS(problem.cluster(), dtwc::SolverError);
  CHECK(problem.centroids_ind == medoids_before);
  CHECK(problem.clusters_ind == labels_before);
}

TEST_CASE("MIP HiGHS: warm start produces valid result", "[mip][highs]")
{
  const auto legacy_rng_original = dtwc::randGenerator;
  dtwc::randGenerator.seed(314159);
  const auto legacy_rng_before = dtwc::randGenerator;
  auto prob = make_small_problem(8, 20);
  prob.set_numberOfClusters(2);
  prob.mip_settings.warm_start = true;
  prob.mip_settings.verbose_solver = false;
  prob.set_solver(dtwc::Solver::HiGHS);
  prob.method = dtwc::Method::MIP;
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
  auto prob1 = make_small_problem(8, 20);
  prob1.set_numberOfClusters(2);
  prob1.mip_settings.warm_start = false;
  prob1.mip_settings.verbose_solver = false;
  prob1.set_solver(dtwc::Solver::HiGHS);
  prob1.method = dtwc::Method::MIP;
  prob1.cluster();

  // Skip if HiGHS not available
  if (prob1.centroids_ind.empty()) return;

  double cold_cost = prob1.findTotalCost();

  auto prob2 = make_small_problem(8, 20);
  prob2.set_numberOfClusters(2);
  prob2.mip_settings.warm_start = true;
  prob2.mip_settings.verbose_solver = false;
  prob2.set_solver(dtwc::Solver::HiGHS);
  prob2.method = dtwc::Method::MIP;
  prob2.cluster();
  double warm_cost = prob2.findTotalCost();

  // Both should find the same global optimum (small instance)
  REQUIRE(warm_cost <= cold_cost + 1e-6);
}

TEST_CASE("MIP HiGHS: settings propagate without crash", "[mip][highs]")
{
  auto prob = make_small_problem(6, 15);
  prob.set_numberOfClusters(2);
  prob.mip_settings.mip_gap = 0.01;
  prob.mip_settings.time_limit_sec = 30;
  prob.mip_settings.verbose_solver = false;
  prob.set_solver(dtwc::Solver::HiGHS);
  prob.method = dtwc::Method::MIP;

  REQUIRE_NOTHROW(prob.cluster());
}

TEST_CASE("MIP HiGHS: k=1 trivial case", "[mip][highs]")
{
  auto prob = make_small_problem(5, 10);
  prob.set_numberOfClusters(1);
  prob.mip_settings.warm_start = true;
  prob.mip_settings.verbose_solver = false;
  prob.set_solver(dtwc::Solver::HiGHS);
  prob.method = dtwc::Method::MIP;
  prob.cluster();

  if (!prob.centroids_ind.empty()) {
    REQUIRE(prob.centroids_ind.size() == 1);
    for (auto c : prob.clusters_ind)
      REQUIRE(c == 0);
  }
}

// ---------------------------------------------------------------------------
// Benders decomposition coverage.
//
// Auto-dispatch to Benders only triggers for N > 200 in Problem::cluster_by_MIP
// (so previous small-N tests never exercise it). These tests force Benders on
// via mip_settings.benders = "on" so the decomposition loop runs on a tractable
// instance, covering MIP_clustering_byBenders end-to-end.
// ---------------------------------------------------------------------------

TEST_CASE("MIP Benders: forced on produces valid clustering", "[mip][highs][benders]")
{
  auto prob = make_small_problem(10, 20);
  // Benders warm-starts via k-medoids Lloyd which writes medoids CSVs. Route
  // output to a temp dir so the test doesn't depend on CWD ./results/.
  prob.output_folder = std::filesystem::temp_directory_path() / "dtwc_mip_benders_test";
  std::filesystem::create_directories(prob.output_folder);
  prob.set_numberOfClusters(2);
  prob.mip_settings.benders = "on";
  prob.mip_settings.warm_start = true;
  prob.mip_settings.verbose_solver = false;
  prob.set_solver(dtwc::Solver::HiGHS); // Benders uses HiGHS as the master/subproblem solver
  prob.method = dtwc::Method::MIP;
  prob.cluster();

  if (!prob.centroids_ind.empty()) {
    REQUIRE(prob.centroids_ind.size() == 2);
    REQUIRE(prob.clusters_ind.size() == 10);
    for (auto c : prob.clusters_ind)
      REQUIRE((c >= 0 && c < 2));
  }
}

TEST_CASE("MIP Benders: cost matches direct HiGHS on small instance", "[mip][highs][benders]")
{
  // On a small instance both Benders and direct HiGHS must find the global
  // optimum of the p-median MIP — costs should agree to numerical precision.
  const auto tmp = std::filesystem::temp_directory_path() / "dtwc_mip_benders_cost_test";
  std::filesystem::create_directories(tmp);

  auto prob_direct = make_small_problem(12, 15);
  prob_direct.output_folder = tmp;
  prob_direct.set_numberOfClusters(3);
  prob_direct.mip_settings.benders = "off";
  prob_direct.mip_settings.verbose_solver = false;
  prob_direct.set_solver(dtwc::Solver::HiGHS);
  prob_direct.method = dtwc::Method::MIP;
  prob_direct.cluster();

  if (prob_direct.centroids_ind.empty()) return; // HiGHS not available in build
  const double cost_direct = prob_direct.findTotalCost();

  auto prob_benders = make_small_problem(12, 15);
  prob_benders.output_folder = tmp;
  prob_benders.set_numberOfClusters(3);
  prob_benders.mip_settings.benders = "on";
  prob_benders.mip_settings.verbose_solver = false;
  prob_benders.set_solver(dtwc::Solver::HiGHS);
  prob_benders.method = dtwc::Method::MIP;
  prob_benders.cluster();

  REQUIRE(prob_benders.centroids_ind.size() == 3);
  const double cost_benders = prob_benders.findTotalCost();
  REQUIRE(std::abs(cost_direct - cost_benders) <= 1e-6 * std::max(1.0, std::abs(cost_direct)));
}

TEST_CASE("MIP Benders warm start preserves caller configuration on success",
          "[mip][highs][benders][state]")
{
  const auto nonce = std::to_string(
    std::chrono::steady_clock::now().time_since_epoch().count())
    + "_" + std::to_string(std::random_device{}());
  const auto tmp = std::filesystem::temp_directory_path()
                 / ("dtwc_mip_benders_state_success_" + nonce);
  std::filesystem::create_directories(tmp);

  auto prob = make_small_problem(10, 16);
  prob.output_folder = tmp;
  prob.set_n_clusters(2);
  prob.method = dtwc::Method::MIP;
  prob.maxIter = 7;
  prob.N_repetition = 4;
  prob.random_seed = 1234;
  prob.last_iterations = 77;
  prob.tadpole_dc = 0.125;
  prob.cuda_settings.device_id = 3;
  prob.cuda_settings.precision = 2;
  prob.mip_settings.benders = "on";
  prob.mip_settings.warm_start = true;
  prob.mip_settings.max_benders_iter = 50;
  prob.mip_settings.numeric_focus = 3;
  prob.mip_settings.mip_focus = 1;
  prob.centroids_ind = {0, 1};
  prob.clusters_ind.assign(prob.size(), 0);
  prob.fill_distance_matrix();
  const auto before = snapshot_configuration(prob);

  prob.cluster();

  check_configuration_unchanged(prob, before);
  CHECK((prob.centroids_ind.size() == 2
         && std::all_of(prob.centroids_ind.begin(), prob.centroids_ind.end(),
                        [&prob](int medoid) {
                          return medoid >= 0
                                 && static_cast<std::size_t>(medoid) < prob.size();
                        })));
  CHECK((prob.clusters_ind.size() == prob.size()
         && std::all_of(prob.clusters_ind.begin(), prob.clusters_ind.end(),
                        [](int label) { return label >= 0 && label < 2; })));
  const double final_cost = prob.find_total_cost();
  CHECK((std::isfinite(final_cost) && final_cost >= 0.0));

  std::error_code ec;
  std::filesystem::remove_all(tmp, ec);
}

TEST_CASE("MIP Benders warm start restores caller state when Lloyd throws",
          "[mip][highs][benders][state]")
{
  auto prob = make_small_problem(8, 12);
  prob.set_n_clusters(2);
  prob.method = dtwc::Method::MIP;
  prob.N_repetition = 5;
  prob.random_seed = 4321;
  prob.last_iterations = 88;
  prob.mip_settings.benders = "on";
  prob.mip_settings.warm_start = true;
  prob.centroids_ind = {6, 7};
  prob.clusters_ind.assign(prob.size(), 1);
  prob.init_fun = [](dtwc::Problem &nested) {
    nested.method = dtwc::Method::TADPole;
    nested.N_repetition = 17;
    nested.last_iterations = 999;
    nested.centroids_ind = {0, 1};
    nested.clusters_ind.assign(nested.size(), 0);
    throw std::runtime_error("forced nested Lloyd failure after state mutation");
  };
  prob.fill_distance_matrix();
  const auto before = snapshot_configuration(prob);
  const auto labels_before = prob.clusters_ind;
  const auto medoids_before = prob.centroids_ind;

  bool threw = false;
  try {
    prob.cluster();
  } catch (const std::runtime_error &error) {
    threw = true;
    CHECK(std::string(error.what())
          == "forced nested Lloyd failure after state mutation");
  }
  REQUIRE(threw);

  check_configuration_unchanged(prob, before);
  CHECK(prob.clusters_ind == labels_before);
  CHECK(prob.centroids_ind == medoids_before);
}

TEST_CASE("MIP Benders warm start does not persist nested Lloyd artifacts",
          "[mip][highs][benders][io]")
{
  const auto nonce = std::to_string(
    std::chrono::steady_clock::now().time_since_epoch().count())
    + "_" + std::to_string(std::random_device{}());
  const auto tmp_root = std::filesystem::temp_directory_path()
                      / ("dtwc_mip_benders_io_" + nonce);
  const auto benders_output = tmp_root / "benders";
  const auto lloyd_output = tmp_root / "lloyd";
  std::filesystem::create_directories(benders_output);
  std::filesystem::create_directories(lloyd_output);

  auto benders = make_small_problem(10, 16);
  benders.output_folder = benders_output;
  benders.name = "nested_";
  benders.set_n_clusters(2);
  benders.method = dtwc::Method::MIP;
  benders.random_seed = 1234;
  benders.mip_settings.benders = "on";
  benders.mip_settings.warm_start = true;
  benders.mip_settings.max_benders_iter = 50;
  std::string benders_stdout;
  {
    ScopedCoutCapture capture;
    benders.cluster();
    benders_stdout = capture.str();
  }

  const auto nested_medoids = benders_output / "nested_medoids_rep_0.csv";
  const auto nested_best_rep = benders_output / "nested__bestRepetition_Nc_2.csv";
  CHECK_FALSE(std::filesystem::exists(nested_medoids));
  CHECK_FALSE(std::filesystem::exists(nested_best_rep));
  CHECK(benders_stdout.find("Best repetition: 0\n") != std::string::npos);
  CHECK(benders_stdout.find("Benders warm start: PAM cost = 22.498")
        != std::string::npos);
  CHECK(benders_stdout.find("Benders converged at iteration 4")
        != std::string::npos);
  const double benders_cost = benders.find_total_cost();

  auto lloyd = make_small_problem(10, 16);
  lloyd.output_folder = lloyd_output;
  lloyd.name = "direct_";
  lloyd.set_n_clusters(2);
  lloyd.N_repetition = 1;
  lloyd.random_seed = 1234;
  std::string lloyd_stdout;
  {
    ScopedCoutCapture capture;
    lloyd.cluster_by_kmedoids_lloyd();
    lloyd_stdout = capture.str();
  }

  const auto medoids_artifact = lloyd_output / "direct_medoids_rep_0.csv";
  const auto best_rep_artifact = lloyd_output / "direct__bestRepetition_Nc_2.csv";
  CHECK(std::filesystem::is_regular_file(medoids_artifact));
  CHECK(std::filesystem::is_regular_file(best_rep_artifact));
  CHECK(lloyd_stdout.find("Best repetition: 0\n") != std::string::npos);
  CHECK((std::isfinite(benders_cost)
         && benders_cost <= lloyd.find_total_cost() + 1e-9));

  std::error_code ec;
  std::filesystem::remove_all(tmp_root, ec);
}

// ---------------------------------------------------------------------------
// Task 0.5 regression: MIP status handling — assert() → real error path.
//
// Targets audit finding #7 (handoff-2026-06-01-adversarial-audit.md:17):
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
  // DTWC_ENABLE_HIGHS is defined PUBLIC on the mip-solvers object library, which
  // links PRIVATE into dtwc++, so the macro is NOT visible in this test TU.
  // Detect HiGHS availability at runtime: a feasible instance produces a
  // non-empty result only when HiGHS is actually compiled in (otherwise the
  // #else branch merely warns and returns empty).
  {
    auto probe = make_small_problem(6, 15);
    probe.set_numberOfClusters(2);
    probe.mip_settings.warm_start = false;
    probe.mip_settings.verbose_solver = false;
    probe.set_solver(dtwc::Solver::HiGHS);
    probe.method = dtwc::Method::MIP;
    probe.cluster();
    if (probe.centroids_ind.empty())
      return; // HiGHS not compiled into this build — nothing to exercise.
    REQUIRE(probe.centroids_ind.size() == 2); // sanity: the solver really ran.
  }

  // Infeasible instance: 4 series but 5 clusters requested. warm_start=false so
  // we bypass fast_pam (which independently rejects k>N) and drive the solver
  // status path directly.
  auto prob = make_small_problem(4, 12);
  prob.set_numberOfClusters(5);
  prob.mip_settings.warm_start = false;
  prob.mip_settings.verbose_solver = false;
  prob.set_solver(dtwc::Solver::HiGHS);
  prob.method = dtwc::Method::MIP;

  REQUIRE_THROWS_AS(prob.cluster(), std::runtime_error);
}

TEST_CASE("MIP Benders: auto dispatches based on N threshold", "[mip][highs][benders]")
{
  // Sanity check the dispatch logic: benders = "auto" + N <= 200 uses direct;
  // benders = "auto" + N > 200 would use Benders (not tested here to keep
  // runtime reasonable). We verify "auto" + small N completes successfully.
  auto prob = make_small_problem(6, 15);
  prob.output_folder = std::filesystem::temp_directory_path() / "dtwc_mip_benders_auto_test";
  std::filesystem::create_directories(prob.output_folder);
  prob.set_numberOfClusters(2);
  prob.mip_settings.benders = "auto";
  prob.mip_settings.verbose_solver = false;
  prob.set_solver(dtwc::Solver::HiGHS);
  prob.method = dtwc::Method::MIP;

  REQUIRE_NOTHROW(prob.cluster());
  if (!prob.centroids_ind.empty()) {
    REQUIRE(prob.centroids_ind.size() == 2);
  }
}
