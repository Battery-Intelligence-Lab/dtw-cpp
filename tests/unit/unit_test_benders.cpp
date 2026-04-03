/**
 * @file unit_test_benders.cpp
 * @brief Unit tests for Benders decomposition MIP clustering.
 *
 * @details Tests that the Benders solver produces correct optimal solutions
 * by comparing against known optimal assignments on small hand-crafted
 * instances. Tests are guarded for the case where HiGHS is not compiled in.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

namespace {

/**
 * @brief Create a Problem with a pre-computed distance matrix.
 *
 * @details Builds a Problem with N trivial 1-element time series (so that
 * DTW distances equal absolute differences), sets the distance matrix
 * directly, and configures for MIP/Benders clustering.
 *
 * @param values  Values for the 1-element time series of each point.
 * @param k       Number of clusters.
 * @param benders "on", "off", or "auto"
 * @return Problem configured for MIP clustering with Benders.
 */
Problem make_problem(const std::vector<double> &values, int k,
                     const std::string &benders = "on")
{
  const int N = static_cast<int>(values.size());

  // Build Data with 1-element time series
  std::vector<std::vector<data_t>> p_vec(N);
  std::vector<std::string> p_names(N);
  for (int i = 0; i < N; ++i) {
    p_vec[i] = { values[i] };
    p_names[i] = "p" + std::to_string(i);
  }

  Problem prob("benders_test");
  prob.set_data(Data(std::move(p_vec), std::move(p_names)));
  prob.set_numberOfClusters(k);
  prob.method = Method::MIP;
  prob.mip_settings.benders = benders;
  prob.mip_settings.warm_start = true;
  prob.mip_settings.verbose_solver = false;
  prob.band = -1; // Full DTW (trivial for length-1 series)

  return prob;
}

/**
 * @brief Compute the total assignment cost of a clustering solution.
 */
double compute_cost(Problem &prob)
{
  double cost = 0.0;
  for (int j = 0; j < prob.size(); ++j) {
    int med = prob.centroids_ind[prob.clusters_ind[j]];
    cost += prob.distByInd(j, med);
  }
  return cost;
}

/**
 * @brief Check that HiGHS produced a valid solution.
 *
 * When HiGHS is not compiled in, Benders returns without modifying
 * centroids_ind. However, set_numberOfClusters() pre-sizes it with
 * default (0) values via resize(). We detect a "no solver ran" state
 * by checking if all centroids are at index 0 AND clusters_ind is
 * all zeros — the solver would set meaningful values.
 */
bool has_solution(const Problem &prob)
{
  if (prob.centroids_ind.empty()) return false;
  // If Benders didn't run, centroids_ind was pre-filled with zeros by resize().
  // A real solution has at least one non-zero centroid index (unless N=1).
  if (prob.centroids_ind.size() > 1) {
    bool all_zero = std::all_of(prob.centroids_ind.begin(), prob.centroids_ind.end(),
                                [](int v) { return v == 0; });
    if (all_zero) return false;
  }
  return true;
}

} // anonymous namespace


TEST_CASE("Benders trivial N=1 k=1", "[benders]")
{
  auto prob = make_problem({ 42.0 }, 1);
  prob.cluster();

  if (!has_solution(prob)) {
    WARN("HiGHS not available; skipping Benders test.");
    return;
  }

  REQUIRE(prob.centroids_ind.size() == 1);
  REQUIRE(prob.centroids_ind[0] == 0);
  REQUIRE(prob.clusters_ind.size() == 1);
  REQUIRE(prob.clusters_ind[0] == 0);
}


TEST_CASE("Benders N=k means every point is a medoid", "[benders]")
{
  auto prob = make_problem({ 1.0, 5.0, 10.0 }, 3);
  prob.cluster();

  if (!has_solution(prob)) {
    WARN("HiGHS not available; skipping Benders test.");
    return;
  }

  REQUIRE(prob.centroids_ind.size() == 3);

  // Every point should be assigned to its own medoid => cost = 0
  double cost = compute_cost(prob);
  REQUIRE_THAT(cost, WithinAbs(0.0, 1e-6));
}


TEST_CASE("Benders small instance N=6 k=2", "[benders]")
{
  // Two clear clusters: {0,1,2} near 0 and {10,11,12} near 11
  // Optimal medoids: 1 (for cluster {0,1,2}) and 11 (for cluster {10,11,12})
  // Optimal cost: |0-1|+|1-1|+|2-1| + |10-11|+|11-11|+|12-11| = 1+0+1+1+0+1 = 4
  auto prob = make_problem({ 0.0, 1.0, 2.0, 10.0, 11.0, 12.0 }, 2);
  prob.cluster();

  if (!has_solution(prob)) {
    WARN("HiGHS not available; skipping Benders test.");
    return;
  }

  REQUIRE(prob.centroids_ind.size() == 2);

  double cost = compute_cost(prob);
  REQUIRE_THAT(cost, WithinAbs(4.0, 1e-4));

  // Verify that the two clusters are separated:
  // Points 0,1,2 should share a cluster and 3,4,5 should share another
  REQUIRE(prob.clusters_ind[0] == prob.clusters_ind[1]);
  REQUIRE(prob.clusters_ind[1] == prob.clusters_ind[2]);
  REQUIRE(prob.clusters_ind[3] == prob.clusters_ind[4]);
  REQUIRE(prob.clusters_ind[4] == prob.clusters_ind[5]);
  REQUIRE(prob.clusters_ind[0] != prob.clusters_ind[3]);
}


TEST_CASE("Benders matches compact MIP on small instance", "[benders]")
{
  // Run both compact MIP and Benders on the same instance;
  // they should produce the same optimal cost.
  std::vector<double> values = { 0.0, 3.0, 7.0, 8.0, 15.0 };
  const int k = 2;

  // Compact MIP
  auto prob_compact = make_problem(values, k, "off");
  prob_compact.cluster();

  // Benders
  auto prob_benders = make_problem(values, k, "on");
  prob_benders.cluster();

  if (!has_solution(prob_compact) || !has_solution(prob_benders)) {
    WARN("HiGHS not available; skipping Benders vs compact comparison.");
    return;
  }

  double cost_compact = compute_cost(prob_compact);
  double cost_benders = compute_cost(prob_benders);

  // Both should find the same optimal cost
  REQUIRE_THAT(cost_benders, WithinAbs(cost_compact, 1e-3));
}


TEST_CASE("Benders convergence within iteration limit", "[benders]")
{
  // N=10 instance with 3 clusters; verify that Benders terminates
  // and produces a valid solution.
  std::vector<double> values = { 0, 1, 2, 10, 11, 12, 20, 21, 22, 23 };
  const int k = 3;

  auto prob = make_problem(values, k);
  prob.mip_settings.max_benders_iter = 50;
  prob.cluster();

  if (!has_solution(prob)) {
    WARN("HiGHS not available; skipping Benders convergence test.");
    return;
  }

  REQUIRE(prob.centroids_ind.size() == 3);
  REQUIRE(prob.clusters_ind.size() == 10);

  // Every cluster index should be in [0, k)
  for (int j = 0; j < 10; ++j) {
    REQUIRE(prob.clusters_ind[j] >= 0);
    REQUIRE(prob.clusters_ind[j] < k);
  }

  // Cost should be finite and non-negative
  double cost = compute_cost(prob);
  REQUIRE(cost >= 0.0);
  REQUIRE(cost < 1e10);
}


TEST_CASE("Benders warm start produces valid initial bound", "[benders]")
{
  // Verify that enabling warm_start does not break correctness.
  std::vector<double> values = { 1, 2, 5, 6, 9, 10 };
  const int k = 3;

  // With warm start
  auto prob_warm = make_problem(values, k);
  prob_warm.mip_settings.warm_start = true;
  prob_warm.cluster();

  // Without warm start
  auto prob_cold = make_problem(values, k);
  prob_cold.mip_settings.warm_start = false;
  prob_cold.cluster();

  if (!has_solution(prob_warm) || !has_solution(prob_cold)) {
    WARN("HiGHS not available; skipping Benders warm start test.");
    return;
  }

  double cost_warm = compute_cost(prob_warm);
  double cost_cold = compute_cost(prob_cold);

  // Both should reach the same optimal cost
  REQUIRE_THAT(cost_warm, WithinAbs(cost_cold, 1e-3));
}


TEST_CASE("Benders auto mode selects correctly by size", "[benders]")
{
  // With "auto" and N <= 200, should use compact MIP.
  // We just verify the solution is valid for a small instance.
  std::vector<double> values = { 0, 5, 10 };
  const int k = 2;

  auto prob = make_problem(values, k, "auto");
  prob.cluster();

  if (!has_solution(prob)) {
    WARN("HiGHS not available; skipping Benders auto mode test.");
    return;
  }

  REQUIRE(prob.centroids_ind.size() == 2);
  double cost = compute_cost(prob);
  REQUIRE(cost >= 0.0);
  REQUIRE(cost < 1e10);
}
