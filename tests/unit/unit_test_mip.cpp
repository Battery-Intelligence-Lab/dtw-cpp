/**
 * @file unit_test_mip.cpp
 * @brief Tests for MIP solver improvements: warm start, settings, correctness.
 *
 * @details Tests run unconditionally. On machines without HiGHS or Gurobi,
 * the MIP solver prints a warning and returns without solving — tests verify
 * that the MIPSettings struct and warm start path compile and don't crash.
 * On machines with HiGHS enabled, the tests verify solution quality.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#include <dtwc.hpp>
#include <mip/warm_start.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <random>
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
  const auto first = dtwc::mip::make_warm_start(
    first_problem, dtwc::settings::DEFAULT_RANDOM_SEED);
  CHECK(dtwc::randGenerator == legacy_rng_before_first);

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
  prob.cluster();
  CHECK(dtwc::randGenerator == legacy_rng_before);
  dtwc::randGenerator = legacy_rng_original;

  // If HiGHS is not compiled in, cluster() prints a warning and returns
  // with empty centroids_ind. Only check if solver actually ran.
  if (!prob.centroids_ind.empty()) {
    REQUIRE(prob.centroids_ind.size() == 2);
    REQUIRE(prob.clusters_ind.size() == 8);
    for (auto c : prob.clusters_ind)
      REQUIRE((c >= 0 && c < 2));
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
  const auto nonce = std::to_string(
    std::chrono::steady_clock::now().time_since_epoch().count())
    + "_" + std::to_string(std::random_device{}());
  const auto tmp_root = std::filesystem::temp_directory_path()
                      / ("dtwc_mip_benders_state_throw_" + nonce);
  const auto missing_output = tmp_root / "missing";
  std::error_code ec;
  std::filesystem::remove_all(tmp_root, ec);

  auto prob = make_small_problem(8, 12);
  prob.output_folder = missing_output;
  prob.set_n_clusters(2);
  prob.method = dtwc::Method::MIP;
  prob.N_repetition = 5;
  prob.random_seed = 4321;
  prob.last_iterations = 88;
  prob.mip_settings.benders = "on";
  prob.mip_settings.warm_start = true;
  prob.centroids_ind = {6, 7};
  prob.clusters_ind.assign(prob.size(), 1);
  prob.fill_distance_matrix();
  const auto before = snapshot_configuration(prob);
  const auto labels_before = prob.clusters_ind;
  const auto medoids_before = prob.centroids_ind;

  bool threw = false;
  try {
    prob.cluster();
  } catch (const std::runtime_error &error) {
    threw = true;
    CHECK(std::string(error.what()).find("Failed to open medoids output file:")
          != std::string::npos);
  }
  REQUIRE(threw);

  check_configuration_unchanged(prob, before);
  CHECK(prob.clusters_ind == labels_before);
  CHECK(prob.centroids_ind == medoids_before);

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
