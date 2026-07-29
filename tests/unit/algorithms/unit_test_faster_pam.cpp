/**
 * @file unit_test_faster_pam.cpp
 * @brief FasterPAM (eager SWAP) correctness + FastPAM1 cross-check + speed bench.
 *
 * @details FasterPAM (Schubert & Rousseeuw 2021, Alg. 4) reaches a local optimum
 * of the k-medoids SWAP neighbourhood in O(N²) per sweep (vs FastPAM1's O(N²·k)),
 * by decomposing the swap gain ΔTD(m, x_c) = acc + ploss[m] in ONE O(N) pass per
 * candidate. The load-bearing claim is that this decomposition equals the TRUE
 * cost change. We test that with an INDEPENDENT arbiter (CLAUDE.md §4): a
 * brute-force best-swap ΔTD computed by fully reassigning every point. If the
 * decomposition had a wrong sign/term, FasterPAM would stop at a NON-locally-
 * optimal point and the brute-force scan would still find an improving swap.
 *
 * Registered bands (stated BEFORE the runs):
 *   BAND-LOCALOPT [HARD] — at FasterPAM convergence the brute-force best-swap
 *       ΔTD ≥ −1e-6·max(1,cost): no improving (medoid_out, point_in) swap exists.
 *   BAND-NOWORSE  [HARD] — from an IDENTICAL BUILD, FasterPAM's final objective ≤
 *       FastPAM1's + 1e-9·max(1,obj) on every tested (data, init). Both are local
 *       optima of the same neighbourhood; FasterPAM does ≥ as many swaps (eager),
 *       so it must not end up worse. A worse result FALSIFIES the port.
 *   BAND-SPEED    [ADVISORY, [.] bench] — FasterPAM swap-phase wall-time ≥ 10×
 *       faster than FastPAM1 at N=1000, k=20 from the same BUILD. Shared machine:
 *       record the ratio; correctness (identical local optima) is the hard half.
 *
 * @author Volkan Kumtepeli
 * @date 08 Jul 2026
 */

#include <dtwc.hpp>
#include <algorithms/detail/fast_pam_plan.hpp>
#include <algorithms/fast_pam.hpp>
#include <timing.hpp> // dtwc::Clock

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <numeric>
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

/// Deterministic BUILD: k medoids evenly spaced over [0, N). Distinct by design.
std::vector<int> even_medoids(int N, int k)
{
  std::vector<int> m(k);
  for (int c = 0; c < k; ++c) m[c] = static_cast<int>((static_cast<long long>(c) * N) / k);
  return m;
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

TEST_CASE("FastPAM point counts are checked at the int-indexed boundary",
          "[faster_pam][dimensions]")
{
  using algorithms::detail::checked_fast_pam_point_count;
  const auto int_max = std::numeric_limits<int>::max();
  const auto size_int_max = static_cast<std::size_t>(int_max);

  CHECK(checked_fast_pam_point_count(1, "fast_pam") == 1);
  CHECK(checked_fast_pam_point_count(size_int_max, "fast_pam") == int_max);
  CHECK_THROWS_AS(
    (void)checked_fast_pam_point_count(0, "fast_pam"), InvalidInput);
  CHECK_THROWS_WITH(
    (void)checked_fast_pam_point_count(0, "fast_pam"),
    "fast_pam: Problem has no data points.");
  CHECK_THROWS_AS(
    (void)checked_fast_pam_point_count(size_int_max + 1, "fast_pam"),
    InvalidInput);
  CHECK_THROWS_WITH(
    (void)checked_fast_pam_point_count(size_int_max + 1, "fast_pam"),
    "fast_pam: N exceeds the int-indexed clustering result limit.");
}

TEST_CASE("Every public FastPAM entry resolves dimensions before effects",
          "[faster_pam][dimensions][effects]")
{
  SECTION("empty problems reach the shared checked boundary")
  {
    Problem empty("empty_fast_pam");
    CHECK_THROWS_WITH(
      (void)fast_pam(empty, 1),
      "fast_pam: Problem has no data points.");
    CHECK_THROWS_WITH(
      (void)fast_pam_seeded(empty, 1, 29),
      "fast_pam_seeded: Problem has no data points.");
    CHECK_THROWS_WITH(
      (void)fast_pam_swap(empty, {0}),
      "fast_pam_swap: Problem has no data points.");
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

  SECTION("invalid medoids do not materialise the distance matrix")
  {
    auto problem = make_unfilled_problem();
    REQUIRE_FALSE(problem.is_distance_matrix_filled());
    CHECK_THROWS(
      (void)fast_pam_swap(problem, {0, 0}, 100, PAMVariant::FasterPAM));
    CHECK_FALSE(problem.is_distance_matrix_filled());
  }
}

// ===========================================================================
// BAND-LOCALOPT — every SWAP variant converges to a genuine local optimum (its
// ΔTD computation is correct). The brute-force arbiter is tie-INDEPENDENT: a wrong
// sign/term would stop at a non-optimum regardless of how ties break. Small N so
// the O(N²·k) brute-force scan is cheap.
// ===========================================================================
TEST_CASE("Every PAM variant converges to a brute-force-verified local optimum", "[faster_pam][arbiter]")
{
  for (const auto variant : { PAMVariant::FastPAM1Naive, PAMVariant::FastPAM1, PAMVariant::FasterPAM }) {
    for (int N : { 20, 30, 45 }) {
      for (int k : { 2, 3, 4 }) {
        Problem prob = make_synthetic_problem(N);
        const auto init = even_medoids(N, k);
        const auto res = fast_pam_swap(prob, init, 100, variant);

        // Result self-consistency: reported cost == recomputed from labels/medoids.
        double recomputed = 0.0;
        for (int p = 0; p < N; ++p)
          recomputed += prob.dist_by_ind(p, res.medoid_indices[res.labels[p]]);
        INFO("variant=" << static_cast<int>(variant) << " N=" << N << " k=" << k);
        REQUIRE_THAT(res.total_cost, WithinAbs(recomputed, 1e-9));

        // Arbiter: no improving swap exists (local optimum) — proves the ΔTD math.
        const double best_delta = brute_force_best_delta(prob, res.medoid_indices, N);
        REQUIRE(best_delta >= -1e-6 * std::max(1.0, res.total_cost));
      }
    }
  }
}

// ===========================================================================
// BAND-NOWORSE — from an IDENTICAL BUILD, FasterPAM ends no worse than FastPAM1,
// and BOTH are local optima. (Cross-checks the two independent SWAP codepaths.)
// ===========================================================================
TEST_CASE("FasterPAM final objective is no worse than FastPAM1", "[faster_pam][quality]")
{
  for (int N : { 24, 40, 60, 90 }) {
    for (int k : { 2, 3, 5 }) {
      Problem prob = make_synthetic_problem(N);
      const auto init = even_medoids(N, k);

      const auto fp1 = fast_pam_swap(prob, init, 100, PAMVariant::FastPAM1);
      const auto fpr = fast_pam_swap(prob, init, 100, PAMVariant::FasterPAM);

      INFO("N=" << N << " k=" << k << " fastpam1=" << fp1.total_cost
                << " faster=" << fpr.total_cost);
      // FasterPAM must not be worse than FastPAM1 from the same init.
      REQUIRE(fpr.total_cost <= fp1.total_cost + 1e-9 * std::max(1.0, fp1.total_cost));
      // Both must be genuine local optima.
      REQUIRE(brute_force_best_delta(prob, fp1.medoid_indices, N) >= -1e-6 * std::max(1.0, fp1.total_cost));
      REQUIRE(brute_force_best_delta(prob, fpr.medoid_indices, N) >= -1e-6 * std::max(1.0, fpr.total_cost));
    }
  }
}

// ===========================================================================
// DECOMPOSITION ≡ NAIVE (objective) — the O(N)-decomposition FastPAM1 computes the
// SAME ΔTD as the naive O(N²·k) nested sum, so from an identical BUILD it reaches
// the SAME local-optimum objective. (Exact medoid identity is NOT required: acc +
// ploss[m] and the naive sum group the same terms in different order, so on an
// EXACT tie — e.g. two points symmetric about a group median, equal cost — FP
// rounding can tip the tie to a different but equal-cost medoid. The objective is
// the invariant; the arbiter test above independently proves each is a local opt.)
// ===========================================================================
TEST_CASE("FastPAM1 decomposition matches the naive baseline in objective", "[faster_pam][decomposition]")
{
  for (int N : { 24, 40, 60, 90 }) {
    for (int k : { 2, 3, 5, 8 }) {
      Problem prob = make_synthetic_problem(N);
      const auto init = even_medoids(N, k);
      const auto naive = fast_pam_swap(prob, init, 100, PAMVariant::FastPAM1Naive);
      const auto decomp = fast_pam_swap(prob, init, 100, PAMVariant::FastPAM1);
      INFO("N=" << N << " k=" << k << " naive=" << naive.total_cost << " decomp=" << decomp.total_cost);
      REQUIRE_THAT(decomp.total_cost, WithinAbs(naive.total_cost, 1e-9));
    }
  }
}

// ===========================================================================
// Determinism — same BUILD ⇒ identical FasterPAM result (no RNG in the swap).
// ===========================================================================
TEST_CASE("FasterPAM is deterministic for a fixed BUILD", "[faster_pam][determinism]")
{
  Problem p1 = make_synthetic_problem(50);
  Problem p2 = make_synthetic_problem(50);
  const auto init = even_medoids(50, 4);
  const auto r1 = fast_pam_swap(p1, init, 100, PAMVariant::FasterPAM);
  const auto r2 = fast_pam_swap(p2, init, 100, PAMVariant::FasterPAM);
  REQUIRE(r1.medoid_indices == r2.medoid_indices);
  REQUIRE(r1.labels == r2.labels);
  REQUIRE_THAT(r1.total_cost, WithinAbs(r2.total_cost, 1e-12));
}

// ===========================================================================
// k=N edge — every point its own medoid, zero cost (both variants).
// ===========================================================================
TEST_CASE("FasterPAM k=N gives zero cost", "[faster_pam][kN]")
{
  const int N = 6;
  Problem prob = make_synthetic_problem(N);
  std::vector<int> init(N);
  std::iota(init.begin(), init.end(), 0);
  const auto res = fast_pam_swap(prob, init, 100, PAMVariant::FasterPAM);
  REQUIRE_THAT(res.total_cost, WithinAbs(0.0, 1e-10));
  std::set<int> med(res.medoid_indices.begin(), res.medoid_indices.end());
  REQUIRE(static_cast<int>(med.size()) == N);
}

// ===========================================================================
// k=1 — the single-medoid optimum is argmin_x Σ_o d(x,o); every variant must find
// it exactly (regression guard: the removal-loss decomposition has no second-
// nearest at k=1, so it is special-cased — a NaN there would return the BUILD
// medoid instead of the optimum). Brute-force the true argmin as the oracle.
// ===========================================================================
TEST_CASE("Every PAM variant finds the true 1-medoid at k=1", "[faster_pam][k1]")
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

  for (const auto variant : { PAMVariant::FastPAM1Naive, PAMVariant::FastPAM1, PAMVariant::FasterPAM }) {
    const auto res = fast_pam_swap(prob, { 0 }, 100, variant); // deliberately bad BUILD (index 0)
    INFO("variant=" << static_cast<int>(variant) << " got=" << res.medoid_indices[0] << " oracle=" << oracle);
    REQUIRE(res.medoid_indices.size() == 1);
    REQUIRE(res.medoid_indices[0] == oracle);
    for (int p = 0; p < N; ++p) REQUIRE(res.labels[p] == 0);
  }
}

// ===========================================================================
// Invalid initial medoids throw (delegated to validate_medoids).
// ===========================================================================
TEST_CASE("FasterPAM rejects invalid initial medoids", "[faster_pam][errors]")
{
  Problem prob = make_synthetic_problem(10);
  REQUIRE_THROWS(fast_pam_swap(prob, {}, 100, PAMVariant::FasterPAM));           // empty
  REQUIRE_THROWS(fast_pam_swap(prob, { 0, 0 }, 100, PAMVariant::FasterPAM));     // duplicate
  REQUIRE_THROWS(fast_pam_swap(prob, { 0, 99 }, 100, PAMVariant::FasterPAM));    // out of range
}

// ===========================================================================
// BENCH — three SWAP variants on k-group data, SWEEPING k. Hidden [.] ADVISORY
// (shared machine): read the RATIOs and their TREND in k, not the milliseconds.
//   naive   = FastPAM1Naive : O(N²·k)/iter, parallel best-swap (the pre-5.1 code)
//   fp1     = FastPAM1       : O(N²)/iter, parallel best-swap (decomposition)   <- default
//   faster  = FasterPAM      : O(N²)/sweep, eager, sequential
//
// Registered bands (stated BEFORE the run):
//   B1 [decomposition win] — fp1 is FASTER than naive at every k, and the ratio
//        naive/fp1 grows with k (the O(N²·k)→O(N²) win). ≥3× by k=100.
//   B2 [eager convergence] — FasterPAM uses far fewer sweeps than fp1's iterations
//        (≈1 vs ≈k); at large k where fp1 would exceed a fixed iteration budget,
//        FasterPAM still converges. Both never worse in objective than naive.
//   B3 [correctness] — |fp1 − naive| == 0 (digit-identical); FasterPAM obj ≤ naive
//        + 1e-9·obj at every k (never worse). HARD even in the bench.
// ===========================================================================
TEST_CASE("BENCH FastPAM1-decomposition vs naive vs FasterPAM, sweeping k", "[.][faster_pam][bench]")
{
  std::printf("\n  k-group data, N=1000, sweep k  (ms; it/sweeps in parens):\n");
  std::printf("     N    k |   naive ms (it) |    fp1 ms (it) | faster ms (sw) | naive/fp1 | naive/faster\n");
  std::printf("  --------+-----------------+----------------+----------------+-----------+-------------\n");

  const int N = 1000;
  double ratio_fp1_at_100 = 0.0;
  bool fp1_beats_naive = true, faster_never_worse = true, fp1_identical = true;
  for (int k : { 10, 20, 50, 100, 200 }) {
    Problem prob = make_kgroup_problem(N, k);
    const auto init = even_medoids(N, k);

    dtwc::Clock cn;
    const auto nv = fast_pam_swap(prob, init, 100, PAMVariant::FastPAM1Naive);
    const double tn = cn.duration() * 1000.0;

    dtwc::Clock c1;
    const auto fp1 = fast_pam_swap(prob, init, 100, PAMVariant::FastPAM1);
    const double t1 = c1.duration() * 1000.0;

    dtwc::Clock cr;
    const auto fpr = fast_pam_swap(prob, init, 100, PAMVariant::FasterPAM);
    const double tr = cr.duration() * 1000.0;

    const double r_fp1 = tn / std::max(t1, 1e-9);
    const double r_fast = tn / std::max(tr, 1e-9);
    std::printf("  %6d %3d | %8.1f (%3d) | %7.1f (%3d) | %7.1f (%3d) | %8.2fx | %10.2fx\n",
                N, k, tn, nv.iterations, t1, fp1.iterations, tr, fpr.iterations, r_fp1, r_fast);

    if (t1 > tn) fp1_beats_naive = false;
    if (fpr.total_cost > nv.total_cost + 1e-9 * std::max(1.0, nv.total_cost)) faster_never_worse = false;
    // Objective identity, not medoid identity: the decomposition and naive sum ΔTD
    // in different orders, so an EXACT tie can pick a different but equal-cost medoid.
    if (std::abs(fp1.total_cost - nv.total_cost) > 1e-9 * std::max(1.0, nv.total_cost)) fp1_identical = false;
    if (k == 100) ratio_fp1_at_100 = r_fp1;
  }

  // Large-N scaling of the default (fp1) and FasterPAM; naive omitted (O(N²·k)).
  std::printf("\n  large-N (naive omitted):\n");
  for (int N2 : { 2000, 5000 }) {
    const int k = 50;
    Problem prob = make_kgroup_problem(N2, k, "big");
    const auto init = even_medoids(N2, k);
    dtwc::Clock c1; const auto fp1 = fast_pam_swap(prob, init, 100, PAMVariant::FastPAM1);
    const double t1 = c1.duration() * 1000.0;
    dtwc::Clock cr; const auto fpr = fast_pam_swap(prob, init, 100, PAMVariant::FasterPAM);
    const double tr = cr.duration() * 1000.0;
    std::printf("  %6d %3d | fp1 %.1f ms (%d it) | faster %.1f ms (%d sw) | obj fp1=%.1f faster=%.1f\n",
                N2, k, t1, fp1.iterations, tr, fpr.iterations, fp1.total_cost, fpr.total_cost);
  }

  std::printf("\n  (ADVISORY — shared machine. B1 fp1<naive all k: %s, naive/fp1@k=100 = %.1fx;\n"
              "   B2/B3: fp1 digit-identical to naive: %s; FasterPAM never worse: %s.)\n\n",
              fp1_beats_naive ? "yes" : "no", ratio_fp1_at_100,
              fp1_identical ? "yes" : "no", faster_never_worse ? "yes" : "no");
  REQUIRE(fp1_identical);       // HARD: decomposition == naive
  REQUIRE(faster_never_worse);  // HARD: eager never worse than the baseline optimum
  SUCCEED();
}
