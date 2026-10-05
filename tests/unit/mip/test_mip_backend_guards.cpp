/**
 * @file test_mip_backend_guards.cpp
 * @brief Regression gate for the exact-MIP honesty guards.
 *
 * Each case pins one guard that did not exist before:
 *
 *   LR-core publishes through Problem::set_result, so its result satisfies
 *       the same invariants as the HiGHS/Gurobi backends.
 *   Method::MIP runs the selected solver at every N (a removed large-N route
 *       used to hand a Gurobi request to HiGHS).
 *
 * @author Volkan Kumtepeli
 * @date 02 Sep 2026
 */

#include <dtwc.hpp>
#include <mip/mip.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

using namespace dtwc;

namespace {

/// Problem over 1-element series, so DTW distance == |value difference|.
Problem make_problem(const std::vector<double> &values, int k)
{
  const int N = static_cast<int>(values.size());
  std::vector<std::vector<data_t>> p_vec(static_cast<std::size_t>(N));
  std::vector<std::string> p_names(static_cast<std::size_t>(N));
  for (int i = 0; i < N; ++i) {
    p_vec[static_cast<std::size_t>(i)] = { values[static_cast<std::size_t>(i)] };
    p_names[static_cast<std::size_t>(i)] = "p" + std::to_string(i);
  }

  Problem prob("mip_guard_test");
  prob.set_data(Data(std::move(p_vec), std::move(p_names)));
  prob.set_n_clusters(k);
  prob.set_method(Method::MIP);
  prob.mip_settings.verbose_solver = false;
  prob.band = -1;
  return prob;
}

void require_highs()
{
  if (!highs_solver_available())
    SKIP("HiGHS is not compiled into this build.");
}

const std::vector<double> kTwoGroups = { 0.0, 1.0, 2.0, 10.0, 11.0, 12.0 };

} // namespace


// ---------------------------------------------------------------------------
// LR-core publishes a validated exact result.
// ---------------------------------------------------------------------------
TEST_CASE("LR-core publishes a transaction-validated clustering", "[lrcore][mip][guards]")
{
  const std::vector<double> values = { 0, 1, 2, 30, 31, 32, 60, 61, 62 };
  const int k = 3;
  auto prob = make_problem(values, k);
  prob.set_method(Method::LRCore);
  prob.cluster();

  // The invariants of an exact p-median clustering.
  REQUIRE(prob.centroids_ind.size() == static_cast<std::size_t>(k));
  auto sorted = prob.centroids_ind;
  std::sort(sorted.begin(), sorted.end());
  REQUIRE(std::adjacent_find(sorted.begin(), sorted.end()) == sorted.end());
  REQUIRE(prob.clusters_ind.size() == static_cast<std::size_t>(values.size()));
  for (int cluster = 0; cluster < k; ++cluster) {
    const auto medoid = static_cast<std::size_t>(prob.centroids_ind[static_cast<std::size_t>(cluster)]);
    REQUIRE(prob.clusters_ind[medoid] == cluster); // medoid in its own cluster.
  }
}

// ---------------------------------------------------------------------------
// Method::MIP above N = 200 runs the selected solver. The removed Benders route
// took every N > 200 request to HiGHS whatever solver was selected, so a
// Gurobi request was answered by HiGHS (or, on a Gurobi-only build, refused).
// ---------------------------------------------------------------------------
TEST_CASE("Method::MIP above N = 200 runs the selected solver", "[mip][guards]")
{
  require_highs();
  // 5 separated groups of 50 (gap 100, within-group spread 4.9), k = 5.
  const int k = 5;
  std::vector<double> values;
  for (int g = 0; g < k; ++g)
    for (int i = 0; i < 50; ++i)
      values.push_back(100.0 * g + 0.1 * i);

  auto prob = make_problem(values, k);
  REQUIRE(prob.size() > 200);
  if (!prob.set_solver(Solver::Gurobi)) {
    SUCCEED("Gurobi is not built here; the route needs a second solver to tell apart.");
    return;
  }

  // A Gurobi-only knob out of Gurobi's range: Gurobi rejects it, HiGHS never
  // reads it, so a reroute to HiGHS would solve instead of failing.
  auto rejected = make_problem(values, k);
  REQUIRE(rejected.set_solver(Solver::Gurobi));
  rejected.mip_settings.numeric_focus = 99;
  REQUIRE_THROWS_MATCHES(rejected.cluster(), SolverError,
                         Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring("Gurobi")));

  prob.cluster();
  REQUIRE(prob.centroids_ind.size() == static_cast<std::size_t>(k));
  // The separable optimum opens exactly one medoid inside each group of 50.
  auto medoids = prob.centroids_ind;
  std::sort(medoids.begin(), medoids.end());
  for (int g = 0; g < k; ++g)
    REQUIRE(medoids[static_cast<std::size_t>(g)] / 50 == g);
}

// ---------------------------------------------------------------------------
// The compact model's sizes pass through `int` in HiGHS (~3N² nonzeros) and in
// Gurobi's addVars (N² variables). Past INT_MAX a cast would build a wrong
// model, so the backend refuses before it fills the matrix or allocates.
// ---------------------------------------------------------------------------
TEST_CASE("The compact MIP refuses a model whose size does not fit int", "[mip][guards]")
{
  require_highs();
  const auto check = [](int N, Solver solver, const char *what) {
    auto prob = make_problem(std::vector<double>(static_cast<std::size_t>(N), 1.0), 2);
    if (!prob.set_solver(solver)) return; // Gurobi is not built here
    prob.mip_settings.warm_start = false;
    REQUIRE_THROWS_MATCHES(prob.cluster(), SolverError,
                           Catch::Matchers::MessageMatches(Catch::Matchers::ContainsSubstring(what)));
    REQUIRE_FALSE(prob.is_distance_matrix_filled());
  };
  check(26800, Solver::HiGHS, "more than INT_MAX nonzeros");   // 3N² − N = 2,154,694,400
  check(46341, Solver::Gurobi, "more than INT_MAX variables"); // N² = 2,147,488,281
}

// ---------------------------------------------------------------------------
// Method::LRCore's "raise the node cap" error is only actionable because
//       MIPSettings::lr_max_nodes exists and is forwarded. Cap the tree at one
//       node on an instance whose root does NOT certify: the uncertified
//       incumbent must be refused, and the same instance must certify under the
//       shipped default cap.
// ---------------------------------------------------------------------------
namespace {

/// Uniform, non-metric symmetric distances: the regime with a nonzero
/// integrality gap where the root dual does not close and the B&B engages
/// (the same generator idea as test_lagrangian_root.cpp's `uniform_D`).
Problem make_uniform_lr_problem(int N, int k, unsigned seed)
{
  std::vector<double> values(static_cast<std::size_t>(N), 0.0);
  for (int i = 0; i < N; ++i) values[static_cast<std::size_t>(i)] = static_cast<double>(i);
  Problem prob = make_problem(values, k);
  prob.set_method(Method::LRCore);
  prob.fill_distance_matrix();

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(1.0, 100.0);
  auto &dm = prob.writable_distance_matrix(); // packed lower-triangular => symmetric
  for (int i = 0; i < N; ++i) {
    dm.set(static_cast<std::size_t>(i), static_cast<std::size_t>(i), 0.0);
    for (int j = i + 1; j < N; ++j)
      dm.set(static_cast<std::size_t>(i), static_cast<std::size_t>(j), u(rng));
  }
  prob.fill_distance_matrix(); // every pair is set: marks the edit complete, computes nothing
  REQUIRE(prob.is_distance_matrix_filled());
  return prob;
}

} // namespace

TEST_CASE("LR-core honours MIPSettings::lr_max_nodes", "[lrcore][mip][guards]")
{
  const int N = 12, k = 3;

  // Locate a seed whose root dual does not certify, so the node cap is reachable
  // at all (on separable data the root certifies and the tree is a single node).
  int chosen = -1;
  for (unsigned seed = 1; seed <= 40 && chosen < 0; ++seed) {
    auto probe = make_uniform_lr_problem(N, k, 9000 + seed);
    probe.mip_settings.lr_max_nodes = 1;
    try {
      probe.cluster();
    } catch (const SolverError &) {
      chosen = static_cast<int>(seed);
    }
  }
  REQUIRE(chosen > 0); // the adversarial regime must occur within 40 seeds.

  auto capped = make_uniform_lr_problem(N, k, 9000 + static_cast<unsigned>(chosen));
  capped.mip_settings.lr_max_nodes = 1;
  REQUIRE_THROWS_AS(capped.cluster(), SolverError);

  auto full = make_uniform_lr_problem(N, k, 9000 + static_cast<unsigned>(chosen));
  REQUIRE(full.mip_settings.lr_max_nodes == 2000000); // the shipped default
  full.cluster();                                     // certifies => publishes
  REQUIRE(full.centroids_ind.size() == static_cast<std::size_t>(k));

  // A cap below one node is an input error, not a solver failure.
  auto invalid = make_uniform_lr_problem(N, k, 9000 + static_cast<unsigned>(chosen));
  invalid.mip_settings.lr_max_nodes = 0;
  REQUIRE_THROWS_AS(invalid.cluster(), InvalidInput);
}

// ---------------------------------------------------------------------------
// A negative mip_gap is bad INPUT, not a HiGHS failure. It used to reach
//       Highs::setOptionValue, be rejected as out-of-domain, and surface as
//       "HiGHS rejected option 'mip_rel_gap'".
// ---------------------------------------------------------------------------
TEST_CASE("A negative mip_gap is rejected as InvalidInput", "[mip][guards]")
{
  auto prob = make_problem(kTwoGroups, 2);
  prob.mip_settings.mip_gap = -1e-3;
  REQUIRE_THROWS_AS(prob.cluster(), InvalidInput);
}
