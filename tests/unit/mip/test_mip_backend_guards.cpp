/**
 * @file test_mip_backend_guards.cpp
 * @brief Regression gate for the exact-MIP honesty guards added on 2026-09-02
 *        (.claude/reports/2026-09-02-review-backends-bindings.md, findings
 *        A1-A5, A9, B1).
 *
 * Each case pins one guard that did not exist before:
 *
 *   A1  Benders exhausting `max_benders_iter` without closing the bound gap
 *       used to print "Benders decomposition complete" and publish a
 *       PAM-quality answer through the EXACT entry point. It must now throw
 *       SolverError, the same contract as mip_Highs.cpp / mip_Gurobi.cpp.
 *   A2  The Benders lower bound is the master's DUAL bound, not its incumbent
 *       objective; the published answer must equal the compact MIP optimum.
 *   A3  The cut tolerance scales with the distance magnitude, so a run on
 *       distances scaled by 1e6 must return the same partition (cost x 1e6).
 *   A4  LR-core publishes through mip::ExactClusteringTransaction, so its
 *       result satisfies the same invariants as the HiGHS/Gurobi backends.
 *   A9  A HiGHS option that the linked build rejects must throw instead of
 *       silently leaving the solve on HiGHS defaults (pdlp_lp_bound's `solver`
 *       string is caller-supplied and was completely unvalidated).
 *   B1  mip::lagrangian_root(Problem&) is BOUND-ONLY: it must not clobber the
 *       caller's centroids_ind / clusters_ind with its heuristic seed.
 *
 * @author Volkan Kumtepeli
 * @date 02 Sep 2026
 */

#include <dtwc.hpp>
#include <mip/benders.hpp>
#include <mip/lagrangian_root.hpp>
#include <mip/mip.hpp>
#include <mip/pdlp_lp.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <random>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

namespace {

/// Problem over 1-element series, so DTW distance == |value difference|.
Problem make_problem(const std::vector<double> &values, int k,
                     const std::string &benders_mode)
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
  prob.mip_settings.benders = benders_mode;
  prob.mip_settings.verbose_solver = false;
  prob.band = -1;
  return prob;
}

double solution_cost(Problem &prob)
{
  double cost = 0.0;
  for (int j = 0; j < prob.size(); ++j)
    cost += prob.dist_by_ind(j, prob.centroids_ind[static_cast<std::size_t>(prob.clusters_ind[static_cast<std::size_t>(j)])]);
  return cost;
}

void require_highs()
{
  if (!highs_solver_available())
    SKIP("HiGHS is not compiled into this build.");
}

const std::vector<double> kTwoGroups = { 0.0, 1.0, 2.0, 10.0, 11.0, 12.0 };

} // namespace


// ---------------------------------------------------------------------------
// A1 — an exhausted cut loop is a failure, not a result.
// ---------------------------------------------------------------------------
TEST_CASE("Benders throws when the iteration cap is hit before convergence", "[benders][mip][guards]")
{
  require_highs();
  // Iteration 0 solves a master with only the cardinality row, so its dual bound
  // is 0 while the true optimum is 4: one iteration cannot close the gap.
  auto prob = make_problem(kTwoGroups, 2, "on");
  prob.mip_settings.warm_start = true;
  prob.mip_settings.max_benders_iter = 1;

  REQUIRE_THROWS_AS(prob.cluster(), SolverError);
}

TEST_CASE("Benders leaves the caller's clustering untouched when it fails", "[benders][mip][guards]")
{
  require_highs();
  auto prob = make_problem(kTwoGroups, 2, "off");
  prob.cluster(); // compact MIP: a valid reference solution.
  const auto reference_medoids = prob.centroids_ind;
  const auto reference_labels = prob.clusters_ind;

  prob.mip_settings.benders = "on";
  prob.mip_settings.max_benders_iter = 1;
  REQUIRE_THROWS_AS(prob.cluster(), SolverError);

  REQUIRE(prob.centroids_ind == reference_medoids);
  REQUIRE(prob.clusters_ind == reference_labels);
}

// ---------------------------------------------------------------------------
// A2 — the published answer is the compact MIP optimum, not an early incumbent.
// ---------------------------------------------------------------------------
TEST_CASE("Benders matches the compact MIP on a 12-point instance", "[benders][mip][guards]")
{
  require_highs();
  const std::vector<double> values = { 0, 1, 3, 9, 10, 12, 20, 22, 23, 31, 32, 34 };
  const int k = 4;

  auto compact = make_problem(values, k, "off");
  compact.cluster();

  auto benders = make_problem(values, k, "on");
  benders.cluster();

  REQUIRE(benders.centroids_ind.size() == static_cast<std::size_t>(k));
  REQUIRE_THAT(solution_cost(benders), WithinAbs(solution_cost(compact), 1e-6));
}

// ---------------------------------------------------------------------------
// A3 — the cut tolerance tracks the distance scale.
// ---------------------------------------------------------------------------
TEST_CASE("Benders is invariant under a 1e6 distance rescaling", "[benders][mip][guards]")
{
  require_highs();
  std::vector<double> scaled(kTwoGroups.size());
  std::transform(kTwoGroups.begin(), kTwoGroups.end(), scaled.begin(),
                 [](double v) { return v * 1e6; });

  auto unit = make_problem(kTwoGroups, 2, "on");
  unit.cluster();

  auto big = make_problem(scaled, 2, "on");
  big.cluster();

  REQUIRE(big.centroids_ind == unit.centroids_ind);
  REQUIRE(big.clusters_ind == unit.clusters_ind);
  REQUIRE_THAT(solution_cost(big), WithinAbs(1e6 * solution_cost(unit), 1e-3));
}

// ---------------------------------------------------------------------------
// A4 — LR-core publishes a validated exact result.
// ---------------------------------------------------------------------------
TEST_CASE("LR-core publishes a transaction-validated clustering", "[lrcore][mip][guards]")
{
  const std::vector<double> values = { 0, 1, 2, 30, 31, 32, 60, 61, 62 };
  const int k = 3;
  auto prob = make_problem(values, k, "off");
  prob.set_method(Method::LRCore);
  prob.cluster();

  // The invariants ExactClusteringTransaction enforces for HiGHS/Gurobi.
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
// B1 — the bound-only entry point has no side effect on the caller.
// ---------------------------------------------------------------------------
TEST_CASE("lagrangian_root(Problem&) does not publish its heuristic seed", "[lagrangian][mip][guards]")
{
  auto prob = make_problem(kTwoGroups, 2, "off");
  prob.fill_distance_matrix();
  prob.centroids_ind = { 5, 0 };            // a caller-chosen, deliberately odd state
  prob.clusters_ind = { 1, 1, 1, 0, 0, 0 };

  const auto bound = mip::lagrangian_root(prob);
  REQUIRE(bound.lower_bound <= bound.upper_bound + 1e-9);

  REQUIRE(prob.centroids_ind == std::vector<int>{ 5, 0 });
  REQUIRE(prob.clusters_ind == std::vector<int>{ 1, 1, 1, 0, 0, 0 });
}

// ---------------------------------------------------------------------------
// A9 — a rejected HiGHS option must not be ignored.
// ---------------------------------------------------------------------------
TEST_CASE("pdlp_lp_bound rejects an unknown solver variant", "[pdlp][mip][guards]")
{
  require_highs();
  const int N = 4;
  const std::vector<double> D = {
    0, 1, 8, 9,
    1, 0, 7, 8,
    8, 7, 0, 1,
    9, 8, 1, 0
  };

  mip::PdlpParams params;
  params.variant = "not-a-solver"; // silently ran dual simplex before the guard.
  REQUIRE_THROWS_AS(mip::pdlp_lp_bound(D.data(), N, 2, params), SolverError);

  params.variant = "pdlp"; // positive control: the real variant still solves.
  const auto ok = mip::pdlp_lp_bound(D.data(), N, 2, params);
  REQUIRE(ok.solved);
}

// ---------------------------------------------------------------------------
// A3' — the two Benders tolerances are different quantities and must not share
//       a value. `abs_eps` compares whole assignment COSTS, so it scales with
//       the distance magnitude; the cut-coefficient floor filters the
//       non-negative coefficients of ONE cut, where dropping a positive value
//       shrinks that cut's left-hand side and can remove the true optimum.
//       Before the fix both sites read `abs_eps`.
// ---------------------------------------------------------------------------
TEST_CASE("Benders scales its cost tolerance but not its cut-coefficient floor", "[benders][mip][guards]")
{
  // Distances up to 2e6 => conditioning factor max(2e6/2, 1) = 1e6.
  REQUIRE_THAT(mip::benders_abs_eps(2e6), WithinAbs(1e-6 * 1e6, 0.0));
  REQUIRE_THAT(mip::benders_abs_eps(2.0), WithinAbs(1e-6, 0.0));
  REQUIRE_THAT(mip::benders_abs_eps(0.5), WithinAbs(1e-6, 0.0)); // floored at 1.

  // The coefficient floor is relative to that cut's own right-hand side only.
  REQUIRE_THAT(mip::benders_cut_coefficient_threshold(1e6), WithinAbs(1e-6, 0.0));
  REQUIRE_THAT(mip::benders_cut_coefficient_threshold(0.0), WithinAbs(1e-12, 0.0));

  // On any instance whose distances are not tiny the floor must be orders of
  // magnitude below the cost tolerance: every c_i <= d_nearest, so a floor equal
  // to abs_eps (the pre-fix code) could discard EVERY coefficient of a cut.
  REQUIRE(mip::benders_cut_coefficient_threshold(2e6)
          < 1e-5 * mip::benders_abs_eps(2e6));
}

// ---------------------------------------------------------------------------
// A2' — the master lower bound is the DUAL bound, not the master's incumbent
//       objective (sum_j theta_j), which is only an UPPER bound on the master
//       optimum. HiGHS is a PRIVATE dependency of mip-solvers and is not on the
//       tests' include path, so the read is a template and this pins it against
//       a stand-in master that reports both numbers.
// ---------------------------------------------------------------------------
namespace {

struct StubMasterInfo {
  double mip_dual_bound = 0.0;
  double objective_function_value = 0.0; // = sum_j theta_j for the Benders master
};

struct StubMaster {
  StubMasterInfo info;
  [[nodiscard]] const StubMasterInfo &getInfo() const { return info; }
};

} // namespace

TEST_CASE("Benders reads the master's dual bound, not its incumbent objective", "[benders][mip][guards]")
{
  // A master deliberately stopped early (a large mip_rel_gap) reports a dual
  // bound strictly below its incumbent sum(theta); reading the incumbent as the
  // LB is what let the cut loop certify a suboptimal medoid set.
  const StubMaster stopped_early{ { 7.0, 12.0 } };
  const double lb = mip::benders_master_lower_bound(stopped_early);
  REQUIRE_THAT(lb, WithinAbs(7.0, 0.0));
  REQUIRE(lb <= stopped_early.getInfo().objective_function_value); // LB <= sum(theta)

  const StubMaster closed{ { 9.5, 9.5 } }; // a fully closed master: the two agree.
  REQUIRE_THAT(mip::benders_master_lower_bound(closed), WithinAbs(9.5, 0.0));
}

// ---------------------------------------------------------------------------
// A1' — DEFAULT settings on the size range where "auto" engages Benders.
//       Method::MIP is the exact route and now throws instead of returning a
//       heuristic, so the shipped default cap must actually close this gap;
//       otherwise the documented default route aborts on every N > 200 call.
// ---------------------------------------------------------------------------
TEST_CASE("Benders converges on a default-settings N=250 instance", "[benders][mip][guards]")
{
  require_highs();
  // 5 separated groups of 50 (gap 100, within-group spread 4.9), k = 5.
  const int k = 5;
  std::vector<double> values;
  values.reserve(250);
  for (int g = 0; g < k; ++g)
    for (int i = 0; i < 50; ++i)
      values.push_back(100.0 * g + 0.1 * i);

  auto prob = make_problem(values, k, "auto"); // "auto" + N = 250 > 200 => Benders.
  REQUIRE(prob.size() == 250);
  REQUIRE(prob.mip_settings.max_benders_iter == 200); // the shipped default
  REQUIRE(prob.mip_settings.mip_gap == 1e-5);         // the shipped default

  // Must not throw. MEASURED on build/highs-1151 (clang, Release, HiGHS 1.15.1):
  // "Benders converged at iteration 4", i.e. 5 master solves, cost 312.5 against a
  // PAM warm start of 5156.2 — well inside the shipped cap of 200.
  prob.cluster();

  REQUIRE(prob.centroids_ind.size() == static_cast<std::size_t>(k));
  // The separable optimum opens exactly one medoid inside each group of 50.
  auto medoids = prob.centroids_ind;
  std::sort(medoids.begin(), medoids.end());
  for (int g = 0; g < k; ++g)
    REQUIRE(medoids[static_cast<std::size_t>(g)] / 50 == g);
}

// ---------------------------------------------------------------------------
// A4' — Method::LRCore's "raise the node cap" error is only actionable because
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
  Problem prob = make_problem(values, k, "off");
  prob.set_method(Method::LRCore);
  prob.fill_distance_matrix();

  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(1.0, 100.0);
  auto &dm = prob.dense_distance_matrix(); // packed lower-triangular => symmetric
  for (int i = 0; i < N; ++i) {
    dm.set(static_cast<std::size_t>(i), static_cast<std::size_t>(i), 0.0);
    for (int j = i + 1; j < N; ++j)
      dm.set(static_cast<std::size_t>(i), static_cast<std::size_t>(j), u(rng));
  }
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
// A6' — a negative mip_gap is bad INPUT, not a HiGHS failure. It used to reach
//       Highs::setOptionValue, be rejected as out-of-domain, and surface as
//       "HiGHS rejected option 'mip_rel_gap'".
// ---------------------------------------------------------------------------
TEST_CASE("A negative mip_gap is rejected as InvalidInput", "[mip][guards]")
{
  auto prob = make_problem(kTwoGroups, 2, "off");
  prob.mip_settings.mip_gap = -1e-3;
  REQUIRE_THROWS_AS(prob.cluster(), InvalidInput);
}
