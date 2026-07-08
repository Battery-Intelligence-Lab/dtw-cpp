/**
 * @file test_pdlp_lp.cpp
 * @brief Independent arbiter: HiGHS PDLP (first-order LP) vs the LR-core
 *        Lagrangian bound on the p-median LP relaxation (PLAN.md Phase 4, Task 4.5).
 *
 * @details Two computations from DIFFERENT mathematics must converge on the same
 * number (CLAUDE.md §4). The LR-core reaches the p-median LP-relaxation optimum by
 * MAXIMIZING the Lagrangian dual (matrix-free); PDLP reaches it by solving the
 * explicit LP with a first-order primal-dual method. Geoffrion's theorem makes the
 * two values equal, so their agreement validates the clever matrix-free bound at N
 * beyond the brute-force IP oracle's reach.
 *
 * Registered bands (stated BEFORE the runs):
 *
 *   BAND-ARB  |pdlp.lp_bound − kelley.lower_bound| / max(1,|kelley|) ≤ 1e-4 on
 *             EVERY instance (≥ 24 across clustered + uniform regimes). 1e-4 is a
 *             comfortable floor over PDLP's first-order accuracy (kkt_tol 1e-8)
 *             that still fails loudly if the two solve DIFFERENT LPs. Any instance
 *             above 1e-4, or a PDLP that does not solve, FALSIFIES the arbiter.
 *   BAND-LB   pdlp.lp_bound ≤ (integer optimum) + 1e-6 on every instance — the LP
 *             relaxation is a valid LOWER bound on the integer p-median cost.
 *   BAND-TIGHT On well-separated clustered data the LP is integral, so
 *             pdlp.lp_bound == integer optimum to 1e-4 (the same regime where the
 *             Lagrangian root certifies, prediction P1).
 *
 * Skips loudly (WARN + return) if HiGHS is not compiled in — PDLP needs it.
 *
 * @author Volkan Kumtepeli
 * @date 08 Jul 2026
 */

#include <dtwc.hpp>
#include <mip/pdlp_lp.hpp>
#include <mip/lagrangian_root.hpp>

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
using dtwc::mip::lagrangian_root_kelley;
using dtwc::mip::LagrangianResult;
using dtwc::mip::pdlp_lp_bound;
using dtwc::mip::PdlpParams;
using dtwc::mip::PdlpResult;

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// Exact p-median optimum by enumerating every C(N,k) medoid subset (N ≤ 14).
double brute_force_opt(const std::vector<double> &D, int N, int k)
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
  double best_cost = kInf;
  while (true) {
    best_cost = std::min(best_cost, eval(comb));
    int i = k - 1;
    while (i >= 0 && comb[static_cast<std::size_t>(i)] == N - k + i) --i;
    if (i < 0) break;
    ++comb[static_cast<std::size_t>(i)];
    for (int j = i + 1; j < k; ++j)
      comb[static_cast<std::size_t>(j)] = comb[static_cast<std::size_t>(j - 1)] + 1;
  }
  return best_cost;
}

/// Well-separated 1-D clusters (centres 100 apart, unique jitter → no exact ties).
std::vector<double> clustered_D(int N, int n_blocks, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> jitter(-1.0, 1.0);
  std::vector<double> pos(static_cast<std::size_t>(N));
  for (int i = 0; i < N; ++i)
    pos[static_cast<std::size_t>(i)] = 100.0 * (i % n_blocks) + jitter(rng) + 1e-3 * i;
  std::vector<double> D(static_cast<std::size_t>(N) * N, 0.0);
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      D[static_cast<std::size_t>(i) * N + j] =
        std::abs(pos[static_cast<std::size_t>(i)] - pos[static_cast<std::size_t>(j)]);
  return D;
}

/// Adversarial: symmetric non-metric D, unique entries in (0.05,1), zero diagonal.
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

/// One-time probe: is HiGHS compiled in? Distinguishes "HiGHS absent" (skip) from
/// a genuine solve failure (which must surface as a test failure, not a skip).
bool highs_present()
{
  const std::vector<double> tiny = uniform_D(3, 1);
  try {
    (void)pdlp_lp_bound(tiny.data(), 3, 1);
  } catch (const dtwc::SolverError &e) {
    if (std::string(e.what()).find("requires HiGHS") != std::string::npos) return false;
    // A different SolverError on a trivial instance is a real bug — let it re-throw.
    throw;
  }
  return true;
}

} // namespace

// ===========================================================================
// BAND-ARB + BAND-LB — the arbiter: PDLP LP optimum == Kelley Lagrangian bound,
// and both are a valid lower bound on the integer optimum.
// ===========================================================================
TEST_CASE("PDLP LP relaxation agrees with the LR-core Lagrangian bound", "[pdlp][arbiter]")
{
  if (!highs_present()) {
    WARN("HiGHS not compiled in; skipping PDLP arbiter.");
    return;
  }

  int checked = 0;
  double max_rel = 0.0;
  for (unsigned seed = 1; seed <= 24; ++seed) {
    const int N = 8 + static_cast<int>(seed % 7); // 8..14
    const int k = 2 + static_cast<int>(seed % 3); // 2..4
    const bool clustered = (seed % 2 == 0);
    const std::vector<double> D = clustered ? clustered_D(N, k, seed) : uniform_D(N, 700 + seed);

    const double opt = brute_force_opt(D, N, k);
    const LagrangianResult kel = lagrangian_root_kelley(D.data(), N, k, -1.0);
    const PdlpResult pd = pdlp_lp_bound(D.data(), N, k);

    REQUIRE(pd.solved); // PDLP must reach optimality on this LP.

    // BAND-ARB: PDLP LP optimum == Lagrangian LP bound (different mathematics).
    const double rel = std::abs(pd.lp_bound - kel.lower_bound) / std::max(1.0, std::abs(kel.lower_bound));
    max_rel = std::max(max_rel, rel);
    INFO("seed=" << seed << " N=" << N << " k=" << k << " clustered=" << clustered
                 << " pdlp=" << pd.lp_bound << " kelley_LB=" << kel.lower_bound
                 << " opt=" << opt << " rel=" << rel << " pdlp_iters=" << pd.iterations);
    REQUIRE(rel <= 1e-4);

    // BAND-LB: the LP relaxation is a valid lower bound on the integer optimum.
    REQUIRE(pd.lp_bound <= opt + 1e-6 * std::max(1.0, std::abs(opt)));
    ++checked;
  }
  std::printf("[pdlp][arbiter] %d instances, max |pdlp-kelley| rel = %.2e (band 1e-4)\n", checked, max_rel);
  REQUIRE(checked == 24);
}

// ===========================================================================
// BAND-TIGHT — on well-separated clustered data the LP is integral, so the PDLP
// relaxation optimum equals the integer optimum (no gap to close).
// ===========================================================================
TEST_CASE("PDLP LP relaxation is tight on clustered data", "[pdlp][tight]")
{
  if (!highs_present()) {
    WARN("HiGHS not compiled in; skipping PDLP tightness test.");
    return;
  }

  int tight = 0;
  const int trials = 12;
  for (unsigned seed = 1; seed <= static_cast<unsigned>(trials); ++seed) {
    const int N = 12, k = 3;
    const std::vector<double> D = clustered_D(N, k, 3000 + seed);
    const double opt = brute_force_opt(D, N, k);
    const PdlpResult pd = pdlp_lp_bound(D.data(), N, k);
    REQUIRE(pd.solved);
    const double rel = std::abs(pd.lp_bound - opt) / std::max(1.0, std::abs(opt));
    INFO("seed=" << seed << " pdlp=" << pd.lp_bound << " opt=" << opt << " rel=" << rel);
    if (rel <= 1e-4) ++tight;
  }
  // Registered: the LP is integral on well-separated clusters (P1 regime).
  REQUIRE(tight == trials);
}

// ===========================================================================
// GPU no-silent-fallback — requesting the GPU on a CPU-only HiGHS build must
// still return the correct bound and report gpu_used=false (it warns to stderr).
// On a CUPDLP_GPU build (DTWC_HIGHS_GPU) it would report gpu_used=true.
// ===========================================================================
TEST_CASE("PDLP GPU request is honoured or warned, never silently wrong", "[pdlp][gpu]")
{
  if (!highs_present()) {
    WARN("HiGHS not compiled in; skipping PDLP GPU test.");
    return;
  }

  const int N = 12, k = 3;
  const std::vector<double> D = clustered_D(N, k, 424242);
  const double opt = brute_force_opt(D, N, k);

  PdlpParams p;
  p.use_gpu = true; // on a CPU build this warns and runs CPU (see stderr).
  const PdlpResult pd = pdlp_lp_bound(D.data(), N, k, p);

  REQUIRE(pd.solved);
  REQUIRE(pd.lp_bound <= opt + 1e-6 * std::max(1.0, std::abs(opt)));
  REQUIRE_THAT(pd.lp_bound, WithinAbs(opt, 1e-4 * std::max(1.0, std::abs(opt)))); // tight (clustered)
  // Expectation is driven by the library's OWN build, queried at runtime — the
  // DTWC_HIGHS_GPU compile define lives in mip-solvers and does not reach this TU.
  if (dtwc::mip::pdlp_gpu_available())
    REQUIRE(pd.gpu_used);       // GPU build: the GPU backend actually ran.
  else
    REQUIRE_FALSE(pd.gpu_used); // CPU build: honest report, no silent GPU claim.
}

// ===========================================================================
// BENCH — PDLP (CPU) vs Kelley Lagrangian vs compact MIP across N. Hidden [.]
// ADVISORY: shared machine — read the SCALING, not the milliseconds. This is the
// solver-comparison deliverable (Task 4.5): the matrix-free Lagrangian is expected
// to dominate the general first-order LP on the structured p-median.
// ===========================================================================
TEST_CASE("BENCH PDLP vs Kelley vs MIP on the p-median LP", "[.][pdlp][bench]")
{
  if (!highs_present()) {
    WARN("HiGHS not compiled in; skipping PDLP bench.");
    return;
  }
  std::printf("\n   N    k | pdlp: ms   iters  bound        | kelley: ms  LB           | rel\n");
  std::printf("  -------+----------------------------------+--------------------------+--------\n");
  for (int N : { 20, 50, 100, 200, 400 }) {
    const int k = 3;
    const std::vector<double> D = clustered_D(N, k, 20240708u);

    dtwc::Clock pd_clk;
    const PdlpResult pd = pdlp_lp_bound(D.data(), N, k);
    const double pd_ms = pd_clk.duration() * 1000.0;

    dtwc::Clock kel_clk;
    const LagrangianResult kel = lagrangian_root_kelley(D.data(), N, k, -1.0);
    const double kel_ms = kel_clk.duration() * 1000.0;

    const double rel = std::abs(pd.lp_bound - kel.lower_bound) / std::max(1.0, std::abs(kel.lower_bound));
    std::printf("  %4d  %2d | %8.1f %6ld  %-11.4f | %8.1f  %-11.4f | %.1e\n",
                N, k, pd_ms, pd.iterations, pd.lp_bound, kel_ms, kel.lower_bound, rel);
  }
  std::printf("\n  (ADVISORY — shared machine. The matrix-free Lagrangian streams D once per\n"
              "   iteration and never forms the ~3N²-nonzero LP that PDLP must; expect Kelley\n"
              "   to dominate on this TU-structured p-median. A GPU PDLP build changes the\n"
              "   PDLP column but not this structural conclusion.)\n\n");
  SUCCEED();
}
