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
// GPU device is a COMPILE-TIME property, honestly reported — the CUPDLP_GPU
// build decides the device, NOT the per-call use_gpu flag (HiGHS has no per-solve
// CPU path on a GPU build). So gpu_used must equal pdlp_gpu_available() for the
// default "pdlp" variant, whether or not the caller requested the GPU; and a GPU
// request on a CPU-only build still returns the correct bound (warns to stderr).
// ===========================================================================
TEST_CASE("PDLP gpu_used reflects the build, never silently wrong", "[pdlp][gpu]")
{
  if (!highs_present()) {
    WARN("HiGHS not compiled in; skipping PDLP GPU test.");
    return;
  }

  const int N = 12, k = 3;
  const std::vector<double> D = clustered_D(N, k, 424242);
  const double opt = brute_force_opt(D, N, k);
  const bool built_gpu = dtwc::mip::pdlp_gpu_available(); // this build's OWN capability

  // Requesting the GPU: correct bound on either build; gpu_used == build capability.
  PdlpParams req;
  req.use_gpu = true; // on a CPU build this warns and runs CPU (see stderr).
  const PdlpResult on = pdlp_lp_bound(D.data(), N, k, req);
  REQUIRE(on.solved);
  REQUIRE(on.lp_bound <= opt + 1e-6 * std::max(1.0, std::abs(opt)));
  REQUIRE_THAT(on.lp_bound, WithinAbs(opt, 1e-4 * std::max(1.0, std::abs(opt)))); // tight (clustered)
  REQUIRE(on.gpu_used == built_gpu);

  // NOT requesting the GPU: on a GPU build solver="pdlp" STILL runs on the GPU
  // (compile-time switch), so gpu_used must NOT depend on the request flag.
  PdlpParams noreq; // use_gpu = false (default)
  const PdlpResult off = pdlp_lp_bound(D.data(), N, k, noreq);
  REQUIRE(off.solved);
  REQUIRE(off.gpu_used == built_gpu); // device is the build's, not the caller's, choice
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
  // Device is a COMPILE-TIME property of the HiGHS build (CUPDLP_GPU) — one PDLP
  // column, self-labelled by the build. Cross-build (CPU vs GPU) comparison is
  // assembled in the run-log by running this same bench on both builds.
  const bool gpu = dtwc::mip::pdlp_gpu_available();
  std::printf("\n  PDLP build: %s  →  device = %s\n",
              gpu ? "CUPDLP_GPU=ON" : "CPU-only", gpu ? "GPU (cuPDLP-C)" : "CPU");
  std::printf("\n   N    k | pdlp ms    iters  gpu? | kelley ms   LB          | rel      | pdlp/kel\n");
  std::printf("  -------+-------------------------+-------------------------+----------+---------\n");
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
    // Arbiter stays live inside the bench: the two bounds are the same LP optimum.
    REQUIRE(rel <= 1e-4);
    const double ratio = pd_ms / std::max(kel_ms, 1e-9);
    std::printf("  %4d  %2d | %9.1f %6ld %4s | %9.1f  %-10.4f | %.1e | %7.1f\n",
                N, k, pd_ms, pd.iterations, pd.gpu_used ? "yes" : "no",
                kel_ms, kel.lower_bound, rel, ratio);
  }
  std::printf("\n  (ADVISORY — shared machine; read the SCALING, not the milliseconds. The\n"
              "   matrix-free Lagrangian streams D once per iteration and never forms the\n"
              "   ~3N²-nonzero LP that PDLP must, so it dominates on this TU-structured\n"
              "   p-median. A GPU build only shifts the PDLP column (fixed launch floor,\n"
              "   gentler large-N slope) — the structural verdict (Kelley wins) holds.)\n\n");
  SUCCEED();
}
