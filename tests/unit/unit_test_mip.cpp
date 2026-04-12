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

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <vector>
#include <cmath>

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

TEST_CASE("MIP HiGHS: warm start produces valid result", "[mip][highs]")
{
  auto prob = make_small_problem(8, 20);
  prob.set_numberOfClusters(2);
  prob.mip_settings.warm_start = true;
  prob.mip_settings.verbose_solver = false;
  prob.set_solver(dtwc::Solver::HiGHS);
  prob.method = dtwc::Method::MIP;
  prob.cluster();

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
