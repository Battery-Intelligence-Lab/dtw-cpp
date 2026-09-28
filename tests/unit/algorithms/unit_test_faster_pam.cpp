/**
 * @file unit_test_faster_pam.cpp
 * @brief FasterPAM (eager SWAP) correctness against a brute-force arbiter.
 *
 * @details FasterPAM (Schubert & Rousseeuw 2021, Alg. 4) reaches a local optimum
 * of the k-medoids SWAP neighbourhood in O(N²) per sweep by decomposing the swap
 * gain ΔTD(m, x_c) = acc + ploss[m] in ONE O(N) pass per candidate. The
 * load-bearing claim is that this decomposition equals the TRUE cost change. We
 * test that with an INDEPENDENT arbiter: a brute-force best-swap ΔTD computed by
 * fully reassigning every point. If the decomposition had a wrong sign/term,
 * FasterPAM would stop at a NON-locally-optimal point and the brute-force scan
 * would still find an improving swap.
 *
 * Registered band (stated BEFORE the runs):
 *   BAND-LOCALOPT [HARD] — at FasterPAM convergence the brute-force best-swap
 *       ΔTD ≥ −1e-6·max(1,cost): no improving (medoid_out, point_in) swap exists.
 *
 * @author Volkan Kumtepeli
 * @date 08 Jul 2026
 */

#include <dtwc.hpp>
#include <algorithms/fast_pam.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// Synthetic time series with 3-group structure (as in unit_test_fast_pam.cpp),
/// scalable to any N. Deterministic — no RNG, so BUILD/SWAP compare cleanly.
Problem make_synthetic_problem(int N, const std::string& name = "faster_pam")
{
  std::vector<std::vector<data_t>> vecs;
  std::vector<std::string> names;
  vecs.reserve(N);
  names.reserve(N);
  for (int i = 0; i < N; ++i) {
    const int group = (i * 3) / N;
    const double baseline = group * 50.0;
    const double slope = (group == 2) ? -2.0 : (group + 1) * 1.5;
    const double off = i * 0.1;
    std::vector<data_t> ts;
    const int len = 20 + (i % 5);
    for (int j = 0; j < len; ++j) ts.push_back(baseline + slope * j + off);
    vecs.push_back(std::move(ts));
    names.push_back("ts_" + std::to_string(i));
  }
  Data data(std::move(vecs), std::move(names));
  Problem prob(name);
  prob.set_data(std::move(data));
  prob.fill_distance_matrix();
  return prob;
}

Problem make_unfilled_problem()
{
  std::vector<std::vector<data_t>> vecs{{0.0, 1.0}, {2.0, 3.0}};
  std::vector<std::string> names{"left", "right"};
  Problem prob("unfilled_fast_pam");
  prob.set_data(Data(std::move(vecs), std::move(names)));
  return prob;
}

/// Synthetic data with `n_groups` well-separated CONTIGUOUS groups (baselines 100
/// apart), ~N/n_groups points each, plus a unique monotone within-group spread so
/// each group has a distinct interior medoid. Contiguous blocks make even_medoids
/// spread one medoid per group (a good, non-degenerate BUILD). k = n_groups makes
/// ALL medoids meaningful — the FasterPAM-favourable / paper regime.
Problem make_kgroup_problem(int N, int n_groups, const std::string& name = "kgroup")
{
  std::vector<std::vector<data_t>> vecs;
  std::vector<std::string> names;
  vecs.reserve(N);
  names.reserve(N);
  for (int i = 0; i < N; ++i) {
    const int group = (i * n_groups) / N;          // contiguous blocks
    const double baseline = group * 100.0;
    const double off = i * 0.001;                   // unique monotone within-group spread
    std::vector<data_t> ts;
    const int len = 20;
    for (int j = 0; j < len; ++j) ts.push_back(baseline + off + 0.1 * j);
    vecs.push_back(std::move(ts));
    names.push_back("g" + std::to_string(i));
  }
  Data data(std::move(vecs), std::move(names));
  Problem prob(name);
  prob.set_data(std::move(data));
  prob.fill_distance_matrix();
  return prob;
}

/// Total assignment cost for a medoid set (each point → nearest medoid). O(N·k).
double assign_cost(Problem& prob, const std::vector<int>& medoids, int N)
{
  double total = 0.0;
  for (int p = 0; p < N; ++p) {
    double best = kInf;
    for (int m : medoids) best = std::min(best, prob.dist_by_ind(p, m));
    total += best;
  }
  return total;
}

/// INDEPENDENT arbiter: most-negative TRUE ΔTD over ALL (medoid slot, non-medoid
/// candidate) swaps, each evaluated by full point reassignment — no shared math
/// with FasterPAM's decomposition. Returns 0 if no swap lowers cost (local opt).
double brute_force_best_delta(Problem& prob, const std::vector<int>& medoids, int N)
{
  const double base = assign_cost(prob, medoids, N);
  std::set<int> med_set(medoids.begin(), medoids.end());
  const int k = static_cast<int>(medoids.size());
  double best = 0.0;
  std::vector<int> trial = medoids;
  for (int slot = 0; slot < k; ++slot) {
    const int keep = trial[slot];
    for (int x = 0; x < N; ++x) {
      if (med_set.count(x)) continue;
      trial[slot] = x;
      best = std::min(best, assign_cost(prob, trial, N) - base);
      trial[slot] = keep;
    }
  }
  return best;
}

} // namespace

TEST_CASE("Every public FastPAM entry resolves dimensions before effects",
          "[faster_pam][dimensions][effects]")
{
  SECTION("empty problems are refused")
  {
    Problem empty("empty_fast_pam");
    CHECK_THROWS_WITH(
      (void)fast_pam(empty, 1),
      "fast_pam: Problem has no data points.");
    CHECK_THROWS_WITH(
      (void)fast_pam_seeded(empty, 1, 29),
      "fast_pam_seeded: Problem has no data points.");
  }

  SECTION("invalid cluster counts do not materialise the distance matrix")
  {
    auto unseeded = make_unfilled_problem();
    REQUIRE_FALSE(unseeded.is_distance_matrix_filled());
    CHECK_THROWS_AS((void)fast_pam(unseeded, 0), InvalidInput);
    CHECK_FALSE(unseeded.is_distance_matrix_filled());

    auto seeded = make_unfilled_problem();
    REQUIRE_FALSE(seeded.is_distance_matrix_filled());
    CHECK_THROWS_AS((void)fast_pam_seeded(seeded, 3, 29), InvalidInput);
    CHECK_FALSE(seeded.is_distance_matrix_filled());
  }
}

// ===========================================================================
// BAND-LOCALOPT — both BUILDs end in a genuine local optimum (the ΔTD computation
// is correct). The brute-force arbiter is tie-INDEPENDENT: a wrong sign/term would
// stop at a non-optimum regardless of how ties break. Small N so the O(N²·k²)
// brute-force scan is cheap.
// ===========================================================================
TEST_CASE("FasterPAM converges to a brute-force-verified local optimum", "[faster_pam][arbiter]")
{
  for (int N : { 20, 30, 45 }) {
    for (int k : { 2, 3, 4 }) {
      for (const bool seeded : { false, true }) {
        Problem prob = make_synthetic_problem(N);
        const auto res = seeded ? fast_pam_seeded(prob, k, 29) : fast_pam(prob, k);

        // Result self-consistency: reported cost == recomputed from labels/medoids.
        double recomputed = 0.0;
        for (int p = 0; p < N; ++p)
          recomputed += prob.dist_by_ind(p, res.medoid_indices[res.labels[p]]);
        INFO("seeded=" << seeded << " N=" << N << " k=" << k);
        REQUIRE(res.converged);
        REQUIRE_THAT(res.total_cost, WithinAbs(recomputed, 1e-9));

        // Arbiter: no improving swap exists (local optimum) — proves the ΔTD math.
        const double best_delta = brute_force_best_delta(prob, res.medoid_indices, N);
        REQUIRE(best_delta >= -1e-6 * std::max(1.0, res.total_cost));
      }
    }
  }
}

// ===========================================================================
// k > max_iter — FastPAM1 made one swap per iteration and stopped unconverged at
// k = 200 (baselines/2026-07-08-faster-pam-bench.md); the eager sweep converges.
// ===========================================================================
TEST_CASE("FasterPAM converges at k = 200 within the default iteration cap", "[faster_pam][large_k]")
{
  Problem prob = make_kgroup_problem(1000, 200);
  const auto res = fast_pam(prob, 200);
  REQUIRE(res.converged);
  REQUIRE(res.iterations < 100);
  std::set<int> med(res.medoid_indices.begin(), res.medoid_indices.end());
  REQUIRE(med.size() == 200);
}

// ===========================================================================
// Determinism — same input ⇒ identical FasterPAM result.
// ===========================================================================
TEST_CASE("FasterPAM is deterministic", "[faster_pam][determinism]")
{
  Problem p1 = make_synthetic_problem(50);
  Problem p2 = make_synthetic_problem(50);
  const auto r1 = fast_pam_seeded(p1, 4, 7);
  const auto r2 = fast_pam_seeded(p2, 4, 7);
  REQUIRE(r1.medoid_indices == r2.medoid_indices);
  REQUIRE(r1.labels == r2.labels);
  REQUIRE_THAT(r1.total_cost, WithinAbs(r2.total_cost, 1e-12));
}

// ===========================================================================
// k=N edge — every point its own medoid, zero cost.
// ===========================================================================
TEST_CASE("FasterPAM k=N gives zero cost", "[faster_pam][kN]")
{
  const int N = 6;
  Problem prob = make_synthetic_problem(N);
  const auto res = fast_pam(prob, N);
  REQUIRE_THAT(res.total_cost, WithinAbs(0.0, 1e-10));
  std::set<int> med(res.medoid_indices.begin(), res.medoid_indices.end());
  REQUIRE(static_cast<int>(med.size()) == N);
}

// ===========================================================================
// k=1 — the single-medoid optimum is argmin_x Σ_o d(x,o); both BUILDs must find
// it exactly (regression guard: the removal-loss decomposition has no second-
// nearest at k=1, so it is special-cased — a NaN there would return the BUILD
// medoid instead of the optimum). Brute-force the true argmin as the oracle.
// ===========================================================================
TEST_CASE("FasterPAM finds the true 1-medoid at k=1", "[faster_pam][k1]")
{
  const int N = 40;
  Problem prob = make_synthetic_problem(N);

  int oracle = 0;
  double best = kInf;
  for (int x = 0; x < N; ++x) {
    double c = 0.0;
    for (int o = 0; o < N; ++o) c += prob.dist_by_ind(x, o);
    if (c < best) { best = c; oracle = x; }
  }

  for (const bool seeded : { false, true }) {
    const auto res = seeded ? fast_pam_seeded(prob, 1, 3) : fast_pam(prob, 1);
    INFO("seeded=" << seeded << " got=" << res.medoid_indices[0] << " oracle=" << oracle);
    REQUIRE(res.medoid_indices.size() == 1);
    REQUIRE(res.medoid_indices[0] == oracle);
    for (int p = 0; p < N; ++p) REQUIRE(res.labels[p] == 0);
  }
}
