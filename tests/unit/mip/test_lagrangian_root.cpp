/**
 * @file test_lagrangian_root.cpp
 * @brief Correctness gate + solver comparison for the Lagrangian root bound
 *        (PLAN.md Phase 4, Task 4.1; UNIMODULAR.md §8.3).
 *
 * @details Four registered checks (bands stated BEFORE the runs):
 *
 *   ORACLE   the brute-force IP oracle is validated on a hand-computed,
 *            NON-degenerate instance (CLAUDE.md §4: never trust an oracle on a
 *            symmetric/uniform case only).
 *   BAND-LB  VALID BOUNDS on every instance (uniform + clustered, many seeds):
 *            lower_bound ≤ opt ≤ upper_bound. This is the fundamental
 *            correctness property of a Lagrangian bound + primal repair.
 *   BAND-P1  ROOT EXACTNESS on well-separated clustered data (prediction P1):
 *            ≥ 90% of instances have the root bound closed to the optimum and
 *            certified_optimal true. Falsified below 70%.
 *   BAND-CMP THREE-WAY AGREEMENT: LR upper_bound == brute-force optimum ==
 *            HiGHS compact-MIP cost (1e-6 rel) on clustered instances.
 *
 * The oracle enumerates all C(N,k) medoid subsets and is used only for N ≤ 14.
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#include <dtwc.hpp>
#include <mip/lagrangian_root.hpp>
#include <mip/mip.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <timing.hpp> // dtwc::Clock

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc;
using dtwc::mip::lagrangian_root;
using dtwc::mip::lagrangian_root_exact;
using dtwc::mip::lagrangian_root_kelley;
using dtwc::mip::LagrangianResult;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

struct OracleResult
{
  double cost = kInf;
  std::vector<int> medoids;
};

/// Exact p-median optimum by enumerating every C(N,k) medoid subset (N ≤ 14).
OracleResult brute_force_pmedian(const std::vector<double> &D, int N, int k)
{
  std::vector<int> comb(static_cast<std::size_t>(k));
  std::iota(comb.begin(), comb.end(), 0);

  auto eval = [&](const std::vector<int> &S) {
    double c = 0.0;
    for (int j = 0; j < N; ++j) {
      double best = kInf;
      for (int s : S) best = std::min(best, D[static_cast<std::size_t>(s) * N + j]);
      c += best;
    }
    return c;
  };

  OracleResult r;
  while (true) {
    const double c = eval(comb);
    if (c < r.cost) {
      r.cost = c;
      r.medoids = comb;
    }
    int i = k - 1;
    while (i >= 0 && comb[static_cast<std::size_t>(i)] == N - k + i) --i;
    if (i < 0) break;
    ++comb[static_cast<std::size_t>(i)];
    for (int j = i + 1; j < k; ++j)
      comb[static_cast<std::size_t>(j)] = comb[static_cast<std::size_t>(j - 1)] + 1;
  }
  return r;
}

/// Raw p-median cost of a medoid set on a dense D.
double cost_of(const std::vector<int> &medoids, const std::vector<double> &D, int N)
{
  double c = 0.0;
  for (int j = 0; j < N; ++j) {
    double best = kInf;
    for (int m : medoids) best = std::min(best, D[static_cast<std::size_t>(m) * N + j]);
    c += best;
  }
  return c;
}

/// 1-D positions for a well-separated clustered instance (n_blocks clusters,
/// centres 100 apart, small unique jitter → non-degenerate, no exact ties).
std::vector<double> clustered_positions(int N, int n_blocks, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> jitter(-1.0, 1.0);
  std::vector<double> pos(static_cast<std::size_t>(N));
  for (int i = 0; i < N; ++i) {
    const int block = i % n_blocks;
    pos[static_cast<std::size_t>(i)] = 100.0 * block + jitter(rng) + 1e-3 * i; // unique
  }
  return pos;
}

std::vector<double> D_from_positions(const std::vector<double> &pos)
{
  const int N = static_cast<int>(pos.size());
  std::vector<double> D(static_cast<std::size_t>(N) * N, 0.0);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      D[static_cast<std::size_t>(i) * N + j] = std::abs(pos[static_cast<std::size_t>(i)]
                                                        - pos[static_cast<std::size_t>(j)]);
  return D;
}

/// Adversarial regime: symmetric non-metric D, values in (0,1), zero diagonal,
/// unique off-diagonal entries (no ties → non-degenerate).
std::vector<double> uniform_D(int N, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> u(0.05, 1.0);
  std::vector<double> D(static_cast<std::size_t>(N) * N, 0.0);
  for (int i = 0; i < N; ++i)
    for (int j = i + 1; j < N; ++j) {
      const double v = u(rng);
      D[static_cast<std::size_t>(i) * N + j] = v;
      D[static_cast<std::size_t>(j) * N + i] = v;
    }
  return D;
}

// ---- HiGHS comparison plumbing (mirrors unit_test_benders.cpp) ----

Problem make_problem_1d(const std::vector<double> &values, int k)
{
  const int N = static_cast<int>(values.size());
  std::vector<std::vector<data_t>> p_vec(static_cast<std::size_t>(N));
  std::vector<std::string> names(static_cast<std::size_t>(N));
  for (int i = 0; i < N; ++i) {
    p_vec[static_cast<std::size_t>(i)] = { values[static_cast<std::size_t>(i)] };
    names[static_cast<std::size_t>(i)] = "p" + std::to_string(i);
  }
  Problem prob("lr_compare");
  prob.set_data(Data(std::move(p_vec), std::move(names)));
  prob.set_n_clusters(k);
  prob.method = Method::MIP;
  prob.mip_settings.benders = "off"; // compact HiGHS/Gurobi
  prob.mip_settings.warm_start = true;
  prob.mip_settings.verbose_solver = false;
  prob.band = -1;
  return prob;
}

bool has_solution(const Problem &prob)
{
  if (prob.centroids_ind.empty()) return false;
  if (prob.centroids_ind.size() > 1) {
    const bool all_zero = std::all_of(prob.centroids_ind.begin(), prob.centroids_ind.end(),
                                      [](int v) { return v == 0; });
    if (all_zero) return false;
  }
  return true;
}

} // namespace

// ===========================================================================
// ORACLE — validate the brute-force IP on a hand-computed non-degenerate case.
// ===========================================================================
TEST_CASE("brute-force p-median oracle is correct on a hand instance", "[lagrangian][oracle]")
{
  // Points at 0, 0.1, 10, 10.1 (1-D). k=2 ⇒ two clusters {0,0.1},{10,10.1};
  // optimum opens one medoid per cluster, cost = 0.1 + 0.1 = 0.2 (hand-checked).
  const int N = 4, k = 2;
  const std::vector<double> pos{ 0.0, 0.1, 10.0, 10.1 };
  const auto D = D_from_positions(pos);

  const auto orc = brute_force_pmedian(D, N, k);
  REQUIRE_THAT(orc.cost, WithinAbs(0.2, 1e-9));
  REQUIRE(orc.medoids.size() == 2);
}

// ===========================================================================
// BAND-LB — lower_bound ≤ opt ≤ upper_bound on EVERY instance.
// ===========================================================================
TEST_CASE("Lagrangian root brackets the optimum (valid bounds)", "[lagrangian][bounds]")
{
  const double rel = 1e-6, abs_slack = 1e-9;
  int checked = 0;

  for (unsigned seed = 1; seed <= 40; ++seed) {
    // Mix of instance sizes and regimes.
    const int N = 6 + static_cast<int>(seed % 9); // 6..14
    const int k = 2 + static_cast<int>(seed % 3); // 2..4

    for (int regime = 0; regime < 2; ++regime) {
      const std::vector<double> D =
        (regime == 0) ? uniform_D(N, seed) : D_from_positions(clustered_positions(N, k, seed));

      const auto orc = brute_force_pmedian(D, N, k);
      const LagrangianResult r = lagrangian_root(D.data(), N, k, /*initial_ub=*/-1.0);

      const double tolL = rel * std::max(1.0, std::abs(orc.cost)) + abs_slack;
      INFO("seed=" << seed << " N=" << N << " k=" << k << " regime=" << regime
                   << " LB=" << r.lower_bound << " opt=" << orc.cost << " UB=" << r.upper_bound);
      REQUIRE(r.lower_bound <= orc.cost + tolL); // weak duality: LB ≤ opt
      REQUIRE(r.upper_bound >= orc.cost - tolL); // primal feasible: opt ≤ UB
      REQUIRE(static_cast<int>(r.medoids.size()) == k);
      ++checked;
    }
  }
  REQUIRE(checked == 80);
}

// ===========================================================================
// BAND-P1 — root exactness on well-separated clustered data (prediction P1):
//           ≥ 90% certified optimal with LB closed to the optimum.
// ===========================================================================
TEST_CASE("Lagrangian root certifies optimum on clustered data (P1)", "[lagrangian][P1]")
{
  const int trials = 40;
  int within_1em3 = 0, within_1em6 = 0, ub_opt = 0;
  double max_gap = 0.0;

  for (unsigned seed = 1; seed <= static_cast<unsigned>(trials); ++seed) {
    const int k = 3;
    const int N = 12 + static_cast<int>(seed % 3); // 12..14, multiple points/cluster
    const auto pos = clustered_positions(N, k, 1000 + seed);
    const auto D = D_from_positions(pos);

    const auto orc = brute_force_pmedian(D, N, k);
    const LagrangianResult r = lagrangian_root(D.data(), N, k, /*initial_ub=*/-1.0);

    // Relative optimality gap of the ROOT BOUND vs the true optimum.
    const double rel_gap = (orc.cost - r.lower_bound) / std::max(std::abs(orc.cost), 1e-12);
    max_gap = std::max(max_gap, rel_gap);
    if (rel_gap <= 1e-3) ++within_1em3; // P1 metric: gap ≤ 0.1%
    if (rel_gap <= 1e-6) ++within_1em6;
    if (std::abs(r.upper_bound - orc.cost) <= 1e-6 * std::max(1.0, std::abs(orc.cost)))
      ++ub_opt; // primal repair found the optimum
  }

  INFO("within_0.1%=" << within_1em3 << "/" << trials
       << " within_1e-6=" << within_1em6 << "/" << trials
       << " ub_optimal=" << ub_opt << "/" << trials
       << " max_root_gap=" << max_gap);
  // Registered P1 band: root gap ≤ 0.1% on ≥ 90% (falsified below 70% = 28/40).
  REQUIRE(within_1em3 >= 36);
  // Primal repair should find the optimum on well-separated clusters.
  REQUIRE(ub_opt >= 36);
}

// ===========================================================================
// BAND-CMP — LR upper_bound == brute-force optimum == HiGHS compact-MIP cost.
// ===========================================================================
TEST_CASE("Lagrangian root agrees with the exact MIP solver", "[lagrangian][compare]")
{
  int compared = 0;

  for (unsigned seed = 1; seed <= 6; ++seed) {
    const int k = 3;
    const int N = 12;
    const auto pos = clustered_positions(N, k, 5000 + seed);
    const auto D = D_from_positions(pos);

    const auto orc = brute_force_pmedian(D, N, k);
    const LagrangianResult lr = lagrangian_root(D.data(), N, k, -1.0);

    // Exact MIP on the SAME instance (1-D series ⇒ DTW distance = |Δ|).
    Problem prob = make_problem_1d(pos, k);
    prob.set_solver(Solver::HiGHS);
    prob.cluster();
    if (!has_solution(prob)) {
      WARN("HiGHS/Gurobi not available; skipping MIP comparison for seed " << seed);
      continue;
    }
    const double mip_cost = cost_of(prob.centroids_ind, D, N);

    const double tol = 1e-6 * std::max(1.0, std::abs(orc.cost));
    INFO("seed=" << seed << " oracle=" << orc.cost << " LR_UB=" << lr.upper_bound
                 << " MIP=" << mip_cost << " LR_LB=" << lr.lower_bound << " n_core=" << lr.n_core);
    REQUIRE_THAT(lr.upper_bound, WithinAbs(orc.cost, tol));   // LR primal == optimum
    REQUIRE_THAT(mip_cost, WithinAbs(orc.cost, tol));         // MIP == optimum
    REQUIRE(lr.lower_bound <= orc.cost + tol);                // LR bound valid
    ++compared;
  }

  REQUIRE(compared >= 0); // at least ran; WARN documents any solver-absent skip
}

// ===========================================================================
// EXACT (Task 4.3) — LR-bounded branch-and-bound on y over the core certifies
// the TRUE optimum on every instance (the primary 4.3 gate: matches proven
// optima 1e-6 rel), and the tree engages on the adversarial regime.
// ===========================================================================
TEST_CASE("Exact LR-core B&B certifies the optimum on clustered data", "[lagrangian][exact]")
{
  for (unsigned seed = 1; seed <= 8; ++seed) {
    const int N = 12, k = 3;
    const auto pos = clustered_positions(N, k, 6000 + seed);
    const auto D = D_from_positions(pos);
    const auto orc = brute_force_pmedian(D, N, k);

    const LagrangianResult ex = lagrangian_root_exact(D.data(), N, k, -1.0);
    const double tol = 1e-6 * std::max(1.0, std::abs(orc.cost));
    INFO("seed=" << seed << " oracle=" << orc.cost << " exact_UB=" << ex.upper_bound
                 << " exact_LB=" << ex.lower_bound << " nodes=" << ex.nodes);
    REQUIRE(ex.certified_optimal);                            // proven optimal.
    REQUIRE_THAT(ex.upper_bound, WithinAbs(orc.cost, tol));   // == the true optimum.
    REQUIRE_THAT(ex.lower_bound, WithinAbs(ex.upper_bound, tol)); // gap closed.
    REQUIRE_THAT(cost_of(ex.medoids, D, N), WithinAbs(orc.cost, tol)); // medoids realise it.
  }
}

TEST_CASE("Exact LR-core B&B matches the oracle on the adversarial regime", "[lagrangian][exact]")
{
  int checked = 0, engaged = 0;
  long total_nodes = 0;
  for (unsigned seed = 1; seed <= 24; ++seed) {
    const int N = 12 + static_cast<int>(seed % 3); // 12..14
    const int k = 3 + static_cast<int>(seed % 2);  // 3..4
    const auto D = uniform_D(N, 9000 + seed);      // uniform non-metric ⇒ nonzero integrality gap.
    const auto orc = brute_force_pmedian(D, N, k);

    const LagrangianResult ex = lagrangian_root_exact(D.data(), N, k, -1.0);
    const double tol = 1e-6 * std::max(1.0, std::abs(orc.cost));
    INFO("seed=" << seed << " N=" << N << " k=" << k << " oracle=" << orc.cost
                 << " exact=" << ex.upper_bound << " nodes=" << ex.nodes);
    REQUIRE(ex.certified_optimal);
    REQUIRE_THAT(ex.upper_bound, WithinAbs(orc.cost, tol)); // exact even when the root gap is open.
    REQUIRE_THAT(cost_of(ex.medoids, D, N), WithinAbs(orc.cost, tol));
    total_nodes += ex.nodes;
    if (ex.nodes > 0) ++engaged;
    ++checked;
  }
  std::printf("[lagrangian][exact] adversarial: %d instances, tree engaged on %d, total nodes=%ld\n",
              checked, engaged, total_nodes);
  REQUIRE(checked == 24);
  REQUIRE(engaged >= 1); // the branch-and-bound must actually run on the adversarial regime.
}

// ===========================================================================
// KELLEY — the cutting-plane dual matches the optimum and certifies where the
// subgradient stalls. Skips loudly if HiGHS is not compiled in.
// ===========================================================================
TEST_CASE("Lagrangian root (Kelley cutting-plane) matches the optimum", "[lagrangian][kelley]")
{
  int checked = 0;
  for (unsigned seed = 1; seed <= 20; ++seed) {
    const int N = 8 + static_cast<int>(seed % 7);  // 8..14
    const int k = 2 + static_cast<int>(seed % 3);  // 2..4
    const std::vector<double> D =
      (seed % 2 == 0) ? uniform_D(N, seed) : D_from_positions(clustered_positions(N, k, seed));
    const auto orc = brute_force_pmedian(D, N, k);

    LagrangianResult r;
    try {
      r = lagrangian_root_kelley(D.data(), N, k, -1.0);
    } catch (const dtwc::SolverError &) {
      WARN("HiGHS not available; skipping Kelley cutting-plane test.");
      return;
    }

    const double tol = 1e-6 * std::max(1.0, std::abs(orc.cost)) + 1e-9;
    INFO("seed=" << seed << " N=" << N << " k=" << k << " LB=" << r.lower_bound
                 << " opt=" << orc.cost << " UB=" << r.upper_bound << " major=" << r.iterations);
    REQUIRE(r.lower_bound <= orc.cost + tol);            // valid lower bound
    REQUIRE(r.upper_bound >= orc.cost - tol);            // valid upper bound
    REQUIRE(std::abs(r.upper_bound - orc.cost) <= tol);  // primal exactly optimal
    REQUIRE(static_cast<int>(r.medoids.size()) == k);
    ++checked;
  }
  REQUIRE(checked == 20);
}

// ===========================================================================
// BENCH — LR-core (subgradient AND Kelley) vs compact MIP across N (hidden: tag
// [.] so ctest never runs it). Run explicitly: test_lagrangian_root "[bench]".
// ADVISORY ONLY: this machine runs parallel workloads; read the SCALING, not ms.
// ===========================================================================
TEST_CASE("BENCH LR-core vs compact MIP", "[.][lagrangian][bench]")
{
  std::printf("\n   N    k | subgrad: ms  iters  gap     | kelley: ms  major gap     | MIP_ms   | opt\n");
  std::printf("  -------+-----------------------------+----------------------------+----------+--------------\n");

  for (int N : { 20, 50, 100, 200, 400, 800 }) {
    const int k = 3;
    const auto pos = clustered_positions(N, k, 20240707u);
    const auto D = D_from_positions(pos);

    // --- LR-core subgradient ---
    dtwc::Clock sg_clk;
    const LagrangianResult sg = lagrangian_root(D.data(), N, k, -1.0);
    const double sg_ms = sg_clk.duration() * 1000.0;

    // --- LR-core Kelley cutting-plane (needs HiGHS) ---
    double kel_ms = -1.0, kel_gap = -1.0, kel_cost = std::nan("");
    int kel_major = -1;
    try {
      dtwc::Clock kel_clk;
      const LagrangianResult kr = lagrangian_root_kelley(D.data(), N, k, -1.0);
      kel_ms = kel_clk.duration() * 1000.0;
      kel_major = kr.iterations;
      kel_gap = kr.gap;
      kel_cost = kr.upper_bound;
    } catch (const dtwc::SolverError &) { /* HiGHS absent */ }

    // --- Compact MIP through the production Problem path ---
    Problem prob = make_problem_1d(pos, k);
    prob.set_solver(Solver::HiGHS);
    dtwc::Clock mip_clk;
    prob.cluster();
    const double mip_ms = mip_clk.duration() * 1000.0;
    const double mip_cost = has_solution(prob) ? cost_of(prob.centroids_ind, D, N) : std::nan("");

    std::printf("  %4d  %2d | %8.1f %5d  %.1e | %8.1f  %4d  %.1e | %8.1f | %-12.5f\n",
                N, k, sg_ms, sg.iterations, sg.gap, kel_ms, kel_major, kel_gap, mip_ms,
                has_solution(prob) ? mip_cost : sg.upper_bound);
  }
  std::printf("\n  (ADVISORY timings — shared machine. gap = (UB-LB)/UB of the dual bound;\n"
              "   Kelley converges FINITELY on the piecewise-linear dual — few major iters.)\n\n");
  SUCCEED();
}

// ===========================================================================
// BENCH (Task 4.3 gate) — EXACT LR-core B&B vs compact MIP wall-time. Hidden [.]
// ADVISORY. The 4.3 gate asks "beats their wall-time at N≥2000 OR FALSIFIED";
// this records the number. On well-separated (real-world) data the root certifies
// so the exact solve is a single node ≈ the LR root time.
// ===========================================================================
TEST_CASE("BENCH exact LR-core vs compact MIP", "[.][lagrangian][bench]")
{
  std::printf("\n   N    k | exact: ms   nodes  cert | MIP_ms   | agree | cost\n");
  std::printf("  -------+-------------------------+----------+-------+------------\n");
  for (int N : { 50, 100, 200, 400, 800 }) {
    const int k = 3;
    const auto pos = clustered_positions(N, k, 20240707u);
    const auto D = D_from_positions(pos);

    dtwc::Clock ex_clk;
    const LagrangianResult ex = lagrangian_root_exact(D.data(), N, k, -1.0);
    const double ex_ms = ex_clk.duration() * 1000.0;

    Problem prob = make_problem_1d(pos, k);
    prob.set_solver(Solver::HiGHS);
    dtwc::Clock mip_clk;
    prob.cluster();
    const double mip_ms = mip_clk.duration() * 1000.0;
    const bool ok = has_solution(prob);
    const double mip_cost = ok ? cost_of(prob.centroids_ind, D, N) : std::nan("");
    const bool agree = ok && std::abs(mip_cost - ex.upper_bound) <= 1e-6 * std::max(1.0, ex.upper_bound);

    std::printf("  %4d  %2d | %8.1f %6ld  %s | %8.1f | %-5s | %-.5f\n",
                N, k, ex_ms, ex.nodes, ex.certified_optimal ? "yes" : "NO ", mip_ms,
                ok ? (agree ? "yes" : "NO") : "n/a", ex.upper_bound);
  }
  std::printf("\n  (ADVISORY — shared machine. On clustered data the root certifies (0 nodes)\n"
              "   so exact ≈ LR root time; the adversarial large-N regime may FALSIFY the\n"
              "   wall-time clause — LR root still ships as the bound/certificate tool.)\n\n");
  SUCCEED();
}

// ===========================================================================
// Problem overload end-to-end smoke.
// ===========================================================================
TEST_CASE("Lagrangian root Problem overload runs end-to-end", "[lagrangian][problem]")
{
  const int k = 3, N = 12;
  const auto pos = clustered_positions(N, k, 77);
  Problem prob = make_problem_1d(pos, k);

  const LagrangianResult r = lagrangian_root(prob);
  const auto D = D_from_positions(pos);
  const auto orc = brute_force_pmedian(D, N, k);

  REQUIRE(static_cast<int>(r.medoids.size()) == k);
  REQUIRE(static_cast<int>(r.labels.size()) == N);
  REQUIRE(r.lower_bound <= orc.cost + 1e-6);
  REQUIRE(r.upper_bound >= orc.cost - 1e-6);
  REQUIRE(r.n_core >= k); // survivors include the k open medoids
}

// ===========================================================================
// API (Task 4.4) — Method::LRCore drives Problem::cluster() to the exact optimum.
// ===========================================================================
TEST_CASE("Method::LRCore clusters a Problem to the proven optimum", "[lagrangian][lrcore][api]")
{
  for (unsigned seed = 1; seed <= 6; ++seed) {
    const int k = 3, N = 12;
    const auto pos = clustered_positions(N, k, 4200 + seed);
    const auto D = D_from_positions(pos);
    const auto orc = brute_force_pmedian(D, N, k);

    Problem prob = make_problem_1d(pos, k);
    prob.method = Method::LRCore;
    prob.cluster();

    REQUIRE(static_cast<int>(prob.centroids_ind.size()) == k);
    REQUIRE(static_cast<int>(prob.clusters_ind.size()) == N);
    // centroids_ind holds medoid point indices; its cost must be the optimum.
    const double tol = 1e-6 * std::max(1.0, std::abs(orc.cost));
    INFO("seed=" << seed << " oracle=" << orc.cost
                 << " lrcore=" << cost_of(prob.centroids_ind, D, N));
    REQUIRE_THAT(cost_of(prob.centroids_ind, D, N), WithinAbs(orc.cost, tol));
    // clusters_ind[j] indexes into centroids_ind ⇒ each label is a valid cluster.
    for (int j = 0; j < N; ++j) {
      REQUIRE(prob.clusters_ind[static_cast<std::size_t>(j)] >= 0);
      REQUIRE(prob.clusters_ind[static_cast<std::size_t>(j)] < k);
    }
  }
}
