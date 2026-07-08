/**
 * @file test_eap_dtw.cpp
 * @brief EAPruned (exact) DTW: digit-identity, UB validity, cell-pruning bench.
 *
 * @details Task 5.4 — Herrmann & Webb, "Early abandoning and pruning for
 *          elastic distances including dynamic time warping", DMKD 35(6), 2021
 *          (arXiv:2102.05221). `dtwFull_eap` computes the EXACT standard-DTW
 *          distance (no band) while pruning every cell the diagonal upper
 *          bound already beats. The load-bearing contract is therefore
 *          EXACTNESS: it must return the same value as the reference full DP
 *          (`dtwFull_L`) for ALL inputs — pruning that changes a distance is a
 *          silent correctness bug in every k-medoids / MIP consumer.
 *
 *          REGISTERED BANDS (fixed before running):
 *            - BAND-EXACT [HARD]:  |dtwFull_eap - dtwFull_L| <= 1e-9*max(1,ref)
 *              over random + adversarial + edge pairs (equal/unequal length,
 *              n=1, identical, monotone, anti-correlated, constant), L1 and
 *              SquaredL2. (Bit-identity expected; 1e-9 guards tie-ordering.)
 *            - BAND-UB [HARD]:  the diagonal L-path UB >= DTW for every pair
 *              (verified via the instrumented counter, which also re-derives
 *              the same distance — so the counter faithfully mirrors the
 *              production window).
 *            - BAND-SPEEDUP [ADVISORY, [.] bench, floor 1.5x]: kernel wall-time
 *              dtwFull_L / dtwFull_eap on UCR-representative lengths, well-
 *              aligned + warped synthetic pairs. Reported, not asserted hard.
 *
 *          All randomised tests use std::mt19937 with fixed seeds.
 *
 * @author Volkan Kumtepeli
 * @author Claude Opus 4.8
 * @date 2026-07-08
 */

#include <core/dtw_options.hpp>   // core::MetricType
#include <warping.hpp>            // dtwFull_eap, dtwFull_L

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using data_t = double;
using dtwc::core::MetricType;

// =========================================================================
//  Helpers
// =========================================================================

static std::vector<data_t> random_series(std::mt19937 &rng, std::size_t len,
                                          data_t lo = -10.0, data_t hi = 10.0)
{
  std::uniform_real_distribution<data_t> dist(lo, hi);
  std::vector<data_t> s(len);
  for (auto &v : s) v = dist(rng);
  return s;
}

/// A slightly warped copy of `base`: local time distortion + small jitter.
/// Produces well-aligned pairs (DTW << full matrix), EAP's favourable regime.
static std::vector<data_t> warped_copy(std::mt19937 &rng,
                                       const std::vector<data_t> &base,
                                       double jitter = 0.05)
{
  std::uniform_real_distribution<data_t> j(-jitter, jitter);
  std::uniform_int_distribution<int> dup(0, 4); // occasional stutter
  std::vector<data_t> out;
  out.reserve(base.size());
  for (std::size_t i = 0; i < base.size() && out.size() < base.size(); ++i) {
    out.push_back(base[i] + j(rng));
    if (dup(rng) == 0 && out.size() < base.size())
      out.push_back(base[i] + j(rng)); // repeat a sample -> warp
  }
  while (out.size() < base.size()) out.push_back(base.back() + j(rng));
  return out;
}

/// Test-side instrumented re-derivation of the EAP window, for the BAND-UB and
/// mechanism numbers. Returns the SAME distance as the production kernel by
/// construction (cross-checked in the tests) plus (cells computed, UB). Kept
/// separate so the production hot path carries no counter branch.
struct EapProbe {
  data_t distance;
  data_t ub;
  std::size_t cells_computed;
  std::size_t total_cells; // n_short * n_long
};

template <typename Metric>
static EapProbe eap_probe(const std::vector<data_t> &x,
                          const std::vector<data_t> &y, Metric cost_of)
{
  constexpr data_t INF = std::numeric_limits<data_t>::max();
  const std::size_t nx = x.size(), ny = y.size();
  const bool swap = nx > ny;
  const auto &X = swap ? y : x; // short axis
  const auto &Y = swap ? x : y; // long axis
  const std::size_t ns = X.size(), nl = Y.size();
  auto cost = [&](std::size_t s, std::size_t L) { return cost_of(X[s], Y[L]); };

  data_t ub = cost(0, 0);
  std::size_t sd = 1, ld = 1;
  while (sd < ns && ld < nl) { ub += cost(sd, ld); ++sd; ++ld; }
  while (ld < nl) { ub += cost(ns - 1, ld); ++ld; }
  // Same relaxed prune threshold as the production kernel (fast-math slack).
  const data_t thr = ub + std::abs(ub)
                   * (data_t(nl) * data_t(16) * std::numeric_limits<data_t>::epsilon());

  std::vector<data_t> prev_buf(ns, INF), curr_buf(ns, INF);
  data_t *prev = prev_buf.data(), *curr = curr_buf.data();
  std::size_t prev_lo = 0, prev_hi = 0, comp_start = 0, cells = 0;

  { // row 0
    std::size_t last_live = 0;
    for (std::size_t s = 0; s < ns; ++s) {
      const data_t d = (s == 0) ? cost(0, 0)
                                 : (curr[s - 1] == INF ? INF : curr[s - 1] + cost(s, 0));
      ++cells;
      if (d <= thr) { curr[s] = d; last_live = s; }
      else { curr[s] = INF; break; }
    }
    prev_lo = 0; prev_hi = last_live + 1; comp_start = 0;
  }
  for (std::size_t L = 1; L < nl; ++L) {
    std::swap(prev, curr);
    std::size_t first_live = ns, last_live = comp_start;
    for (std::size_t s = comp_start; s < ns; ++s) {
      const data_t up   = (s >= prev_lo && s < prev_hi) ? prev[s] : INF;
      const data_t diag = (s >= 1 && (s - 1) >= prev_lo && (s - 1) < prev_hi) ? prev[s - 1] : INF;
      const data_t left = (s > comp_start) ? curr[s - 1] : INF;
      data_t m = up;
      if (left < m) m = left;
      if (diag < m) m = diag;
      const data_t d = (m == INF) ? INF : m + cost(s, L);
      ++cells;
      if (d <= thr) { curr[s] = d; last_live = s; if (first_live == ns) first_live = s; }
      else curr[s] = INF;
      if (s < prev_hi) continue;
      if (d > thr) break;
    }
    comp_start = (first_live == ns) ? last_live : first_live;
    prev_lo = comp_start;
    prev_hi = last_live + 1;
  }
  return { curr[ns - 1], ub, cells, ns * nl };
}

static data_t l1(data_t a, data_t b) { return std::abs(a - b); }
static data_t sql2(data_t a, data_t b) { data_t d = a - b; return d * d; }

// =========================================================================
//  BAND-EXACT [HARD]: dtwFull_eap == dtwFull_L
// =========================================================================

TEST_CASE("EAP exact: matches dtwFull_L on random pairs (L1 + SquaredL2)", "[eap][exact]")
{
  std::mt19937 rng(20260708u);
  const MetricType metrics[] = { MetricType::L1, MetricType::SquaredL2 };
  int checked = 0;
  for (int rep = 0; rep < 400; ++rep) {
    std::uniform_int_distribution<std::size_t> len(1, 60);
    auto a = random_series(rng, len(rng));
    auto b = random_series(rng, len(rng));
    for (auto mt : metrics) {
      const data_t ref = dtwc::dtwFull_L<data_t>(a, b, data_t(-1), mt);
      const data_t got = dtwc::dtwFull_eap<data_t>(a, b, mt);
      REQUIRE_THAT(got, WithinRel(ref, 1e-12) || WithinAbs(ref, 1e-9));
      ++checked;
    }
  }
  REQUIRE(checked == 800);
}

TEST_CASE("EAP exact: well-aligned warped pairs (the pruning regime)", "[eap][exact]")
{
  std::mt19937 rng(11u);
  for (int rep = 0; rep < 100; ++rep) {
    std::uniform_int_distribution<std::size_t> len(20, 200);
    auto base = random_series(rng, len(rng));
    auto w = warped_copy(rng, base);
    for (auto mt : { MetricType::L1, MetricType::SquaredL2 }) {
      const data_t ref = dtwc::dtwFull_L<data_t>(base, w, data_t(-1), mt);
      const data_t got = dtwc::dtwFull_eap<data_t>(base, w, mt);
      REQUIRE_THAT(got, WithinRel(ref, 1e-12) || WithinAbs(ref, 1e-9));
    }
  }
}

TEST_CASE("EAP exact: edge cases", "[eap][exact][edge]")
{
  auto same = [](const std::vector<data_t> &a, const std::vector<data_t> &b) {
    const data_t ref = dtwc::dtwFull_L<data_t>(a, b);
    const data_t got = dtwc::dtwFull_eap<data_t>(a, b);
    REQUIRE_THAT(got, WithinRel(ref, 1e-12) || WithinAbs(ref, 1e-9));
  };

  SECTION("identical series -> 0") {
    std::vector<data_t> a{ 1, 2, 3, 4, 5, 6, 7 };
    REQUIRE(dtwc::dtwFull_eap<data_t>(a, a) == 0.0); // fast identity path
    std::vector<data_t> b = a;                        // distinct storage
    REQUIRE_THAT(dtwc::dtwFull_eap<data_t>(a, b), WithinAbs(0.0, 1e-12));
  }
  SECTION("length-1 short axis") {
    same({ 3.0 }, { 1.0, 4.0, 1.0, 5.0, 9.0 });
    same({ 1.0, 4.0, 1.0, 5.0, 9.0 }, { 3.0 });
  }
  SECTION("length 1 vs 1")            { same({ 2.0 }, { 7.0 }); }
  SECTION("monotone increasing")      { same({ 1, 2, 3, 4, 5, 6 }, { 1, 1, 2, 3, 5, 8, 13 }); }
  SECTION("anti-correlated")          { same({ 1, 2, 3, 4, 5 }, { 5, 4, 3, 2, 1 }); }
  SECTION("constant series")          { same({ 3, 3, 3, 3 }, { 3, 3, 3, 3, 3, 3 }); }
  SECTION("one constant one varying") { same({ 0, 0, 0, 0, 0 }, { -2, 5, -3, 8, 1 }); }
  SECTION("very unequal lengths")     {
    std::mt19937 rng(7u);
    same(random_series(rng, 3), random_series(rng, 97));
  }
}

// =========================================================================
//  BAND-UB [HARD]: diagonal L-path UB >= DTW; counter mirrors production
// =========================================================================

TEST_CASE("EAP: diagonal UB >= DTW and probe re-derives the exact distance", "[eap][ub]")
{
  std::mt19937 rng(999u);
  for (int rep = 0; rep < 300; ++rep) {
    std::uniform_int_distribution<std::size_t> len(1, 80);
    auto a = random_series(rng, len(rng));
    auto b = random_series(rng, len(rng));

    const data_t dtw_l1 = dtwc::dtwFull_eap<data_t>(a, b, MetricType::L1);
    const auto probe = eap_probe(a, b, l1);
    // UB validity.
    REQUIRE(probe.ub >= dtw_l1 - 1e-9);
    // The instrumented window returns the SAME distance as production -> the
    // counter faithfully mirrors the real kernel (so its cell count is honest).
    REQUIRE_THAT(probe.distance, WithinRel(dtw_l1, 1e-12) || WithinAbs(dtw_l1, 1e-9));
    REQUIRE(probe.cells_computed <= probe.total_cells);

    const data_t dtw_sq = dtwc::dtwFull_eap<data_t>(a, b, MetricType::SquaredL2);
    const auto probe_sq = eap_probe(a, b, sql2);
    REQUIRE(probe_sq.ub >= dtw_sq - 1e-9);
    REQUIRE_THAT(probe_sq.distance, WithinRel(dtw_sq, 1e-12) || WithinAbs(dtw_sq, 1e-9));
  }
}

// =========================================================================
//  BAND-SPEEDUP [ADVISORY, hidden bench]: kernel wall-time ratio + pruning
// =========================================================================

// Smooth random-walk base (consecutive samples correlated — realistic for DTW,
// unlike white noise). step ~ N(0,1).
static std::vector<data_t> random_walk(std::mt19937 &rng, std::size_t n)
{
  std::normal_distribution<data_t> step(0.0, 1.0);
  std::vector<data_t> s(n);
  data_t x = 0;
  for (auto &v : s) { x += step(rng); v = x; }
  return s;
}

// Regime generators — all EQUAL length (EAP is unbanded/equal-len here).
enum class Regime { NearDiag, MildWarp, Unrelated };

static std::pair<std::vector<data_t>, std::vector<data_t>>
make_pair(std::mt19937 &rng, std::size_t n, Regime r)
{
  auto a = random_walk(rng, n);
  if (r == Regime::Unrelated) return { a, random_walk(rng, n) };
  // Amplitude noise (near-diagonal alignment: diagonal UB is tight).
  std::normal_distribution<data_t> noise(0.0, 0.3);
  std::vector<data_t> b = a;
  for (auto &v : b) v += noise(rng);
  if (r == Regime::MildWarp) {
    // A few local time shifts: resample b with occasional stutter/skip, keeping
    // length n. Optimal path leaves the diagonal a little -> UB slightly looser.
    std::uniform_int_distribution<int> where(0, static_cast<int>(n) - 2);
    for (int k = 0; k < static_cast<int>(n) / 20; ++k) {
      int i = where(rng);
      std::swap(b[i], b[i + 1]); // local reordering = small warp
    }
  }
  return { a, b };
}

TEST_CASE("EAP speedup bench (regime sweep, warm timing)", "[.][eap][bench]")
{
  const std::size_t lens[] = { 128, 256, 512, 1024 };
  constexpr int PAIRS = 40;
  constexpr int ROUNDS = 3; // take min over rounds -> both fns warm, order-bias free

  const char *rname[] = { "near-diagonal", "mild-warp    ", "unrelated    " };
  const Regime regimes[] = { Regime::NearDiag, Regime::MildWarp, Regime::Unrelated };

  auto time_fn = [&](auto fn, const std::vector<std::vector<data_t>> &A,
                     const std::vector<std::vector<data_t>> &B) {
    double best = 1e18;
    for (int r = 0; r < ROUNDS; ++r) {
      volatile data_t sink = 0;
      auto t0 = std::chrono::steady_clock::now();
      for (std::size_t p = 0; p < A.size(); ++p) sink += fn(A[p], B[p]);
      auto t1 = std::chrono::steady_clock::now();
      best = std::min(best, std::chrono::duration<double, std::milli>(t1 - t0).count());
      (void)sink;
    }
    return best;
  };

  std::printf("\n  regime          len   plainDP(ms)   eap(ms)   speedup   cells%%\n");
  for (std::size_t ri = 0; ri < 3; ++ri) {
    std::mt19937 rng(2026u + static_cast<unsigned>(ri));
    for (auto n : lens) {
      std::vector<std::vector<data_t>> A, B;
      std::size_t cells_sum = 0, total_sum = 0;
      for (int p = 0; p < PAIRS; ++p) {
        auto [a, b] = make_pair(rng, n, regimes[ri]);
        const auto pr = eap_probe(a, b, l1);
        cells_sum += pr.cells_computed; total_sum += pr.total_cells;
        A.push_back(std::move(a)); B.push_back(std::move(b));
      }
      // Warm both once (fill caches, page-in) before the timed rounds.
      { volatile data_t s = 0;
        for (std::size_t p = 0; p < A.size(); ++p) {
          s += dtwc::dtwFull_L<data_t>(A[p], B[p], data_t(-1), MetricType::L1);
          s += dtwc::dtwFull_eap<data_t>(A[p], B[p], MetricType::L1);
        } (void)s; }
      const double plain_ms = time_fn(
        [](const auto &x, const auto &y) { return dtwc::dtwFull_L<data_t>(x, y, data_t(-1), MetricType::L1); }, A, B);
      const double eap_ms = time_fn(
        [](const auto &x, const auto &y) { return dtwc::dtwFull_eap<data_t>(x, y, MetricType::L1); }, A, B);
      std::printf("  %s  %4zu   %9.3f   %8.3f   %6.2fx   %5.1f\n",
                  rname[ri], n, plain_ms, eap_ms, plain_ms / eap_ms,
                  100.0 * double(cells_sum) / double(total_sum));
    }
  }
  std::printf("  REGISTERED BAND-SPEEDUP floor: >=1.5x on the near-diagonal regime\n"
              "  (the diagonal UB's design regime). Advisory: verdict recorded in run-log.\n\n");
  SUCCEED();
}
