/**
 * @file test_lb_enhanced_webb.cpp
 * @brief Adversarial validity and tightness tests for LB_Enhanced and the
 *        local LB_Webb_NoLR-plus-tail-cap implementation.
 *
 * @details The load-bearing contract of any DTW lower bound is
 *              LB(A, B) <= DTW_w(A, B)
 *          in the finite, equal-length scalar L1/squared-L2 domain with the
 *          same window and point cost.
 *          If it ever fails, pruning built on the bound silently returns wrong
 *          results. These are the primitives added in Task 5.2; the exact-matrix
 *          build gains no DTW-call reduction from them (see the run-log), so the
 *          value asserted here is: (1) VALIDITY [HARD], (2) the local
 *          NoLR-plus-tail-cap result >= matching-direction LB_Keogh [HARD,
 *          provable], (3) envelope correctness. Tightness magnitude is an
 *          ADVISORY bench, not asserted here.
 *
 *          REGISTERED BANDS (fixed before running):
 *            - LB_Enhanced <= DTW_w + 1e-9 and LB_Webb <= DTW_w + 1e-9
 *              (L1 and SquaredL2), random + adversarial, bands {0,1,2,5,10,20,
 *              10% of n}, lengths incl. edge {2,3,4,5,6,7} and n slightly > 2V.
 *            - Local Webb(A,B) >= LB_Keogh(A, env B) - 1e-9 (per instance).
 *            - Local Webb_sym >= LB_Keogh_sym - 1e-9 (per instance).
 *            - LB >= 0 ; LB(x, x) == 0.
 *          Effective V=1 Enhanced dominates matching-direction Keogh. For
 *          effective V>=2, the D3 exact oracle—not SDM 2019—contains strict
 *          witnesses in both order directions.
 *
 *          All randomised tests use std::mt19937 seed=12345 for reproducibility.
 *
 * @author Volkan Kumtepeli
 * @author Claude Opus 4.8
 * @date 2026-07-08
 */

#include <core/lower_bound_impl.hpp>
#include <core/distance_metric.hpp>
#include <warping.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;
using data_t = double;
using dtwc::core::L1Metric;
using dtwc::core::SquaredL2Metric;
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

static std::size_t random_length(std::mt19937 &rng, std::size_t min_len, std::size_t max_len)
{
  std::uniform_int_distribution<std::size_t> dist(min_len, max_len);
  return dist(rng);
}

/// LB_Enhanced (L1) via precomputed candidate envelope.
static data_t enhanced_l1(const std::vector<data_t> &A, const std::vector<data_t> &B,
                          int band, int V = 5)
{
  auto envB = dtwc::core::compute_envelope(B, band);
  return dtwc::core::lb_enhanced<data_t, L1Metric>(A.data(), B.data(), A.size(),
                                                   envB.upper.data(), envB.lower.data(),
                                                   band, V);
}

/// LB_Enhanced (SquaredL2).
static data_t enhanced_sq(const std::vector<data_t> &A, const std::vector<data_t> &B,
                          int band, int V = 5)
{
  auto envB = dtwc::core::compute_envelope(B, band);
  return dtwc::core::lb_enhanced<data_t, SquaredL2Metric>(A.data(), B.data(), A.size(),
                                                          envB.upper.data(), envB.lower.data(),
                                                          band, V);
}

/// LB_Webb (L1, one-directional A query vs B candidate).
static data_t webb_l1(const std::vector<data_t> &A, const std::vector<data_t> &B, int band)
{
  auto wa = dtwc::core::compute_webb_envelope(A, band);
  auto wb = dtwc::core::compute_webb_envelope(B, band);
  return dtwc::core::lb_webb<L1Metric>(A, wa, B, wb, band);
}

/// LB_Webb (SquaredL2, one-directional).
static data_t webb_sq(const std::vector<data_t> &A, const std::vector<data_t> &B, int band)
{
  auto wa = dtwc::core::compute_webb_envelope(A, band);
  auto wb = dtwc::core::compute_webb_envelope(B, band);
  return dtwc::core::lb_webb<SquaredL2Metric>(A, wa, B, wb, band);
}

static data_t webb_sym_l1(const std::vector<data_t> &A, const std::vector<data_t> &B, int band)
{
  auto wa = dtwc::core::compute_webb_envelope(A, band);
  auto wb = dtwc::core::compute_webb_envelope(B, band);
  return dtwc::core::lb_webb_symmetric<L1Metric>(A, wa, B, wb, band);
}

// =========================================================================
//  AREA 1: VALIDITY — LB <= DTW_w  (the fundamental contract)
// =========================================================================

TEST_CASE("LB_Enhanced <= banded DTW (L1), random pairs, several bands",
          "[adversarial][lb_enhanced][contract]")
{
  std::mt19937 rng(12345);
  for (int band : { 0, 1, 2, 5, 10, 20 }) {
    for (int p = 0; p < 40; ++p) {
      const auto n = random_length(rng, 30, 200);
      auto x = random_series(rng, n);
      auto y = random_series(rng, n);
      const data_t lb = enhanced_l1(x, y, band);
      const data_t dtw = dtwc::dtwBanded<data_t>(x, y, band);
      INFO("band=" << band << " n=" << n << " LB_Enhanced=" << lb << " DTW=" << dtw);
      REQUIRE(lb <= dtw + 1e-9);
      REQUIRE(lb >= 0.0);
    }
  }
}

TEST_CASE("LB_Enhanced <= banded DTW (SquaredL2), random pairs",
          "[adversarial][lb_enhanced][contract][squared]")
{
  std::mt19937 rng(12345);
  for (int band : { 1, 5, 10, 20 }) {
    for (int p = 0; p < 40; ++p) {
      const auto n = random_length(rng, 30, 200);
      auto x = random_series(rng, n);
      auto y = random_series(rng, n);
      const data_t lb = enhanced_sq(x, y, band);
      const data_t dtw = dtwc::dtwBanded<data_t>(x, y, band, -1.0, MetricType::SquaredL2);
      INFO("band=" << band << " n=" << n << " LB_Enhanced_sq=" << lb << " DTW_sq=" << dtw);
      REQUIRE(lb <= dtw + 1e-6);   // squared magnitudes: looser abs tol
      REQUIRE(lb >= 0.0);
    }
  }
}

TEST_CASE("LB_Webb <= banded DTW (L1), random pairs, several bands",
          "[adversarial][lb_webb][contract]")
{
  std::mt19937 rng(12345);
  for (int band : { 0, 1, 2, 5, 10, 20 }) {
    for (int p = 0; p < 40; ++p) {
      const auto n = random_length(rng, 30, 200);
      auto x = random_series(rng, n);
      auto y = random_series(rng, n);
      const data_t lb = webb_l1(x, y, band);
      const data_t dtw = dtwc::dtwBanded<data_t>(x, y, band);
      INFO("band=" << band << " n=" << n << " LB_Webb=" << lb << " DTW=" << dtw);
      REQUIRE(lb <= dtw + 1e-9);
      REQUIRE(lb >= 0.0);
    }
  }
}

TEST_CASE("LB_Webb <= banded DTW (SquaredL2), random pairs",
          "[adversarial][lb_webb][contract][squared]")
{
  std::mt19937 rng(12345);
  for (int band : { 1, 5, 10, 20 }) {
    for (int p = 0; p < 40; ++p) {
      const auto n = random_length(rng, 30, 200);
      auto x = random_series(rng, n);
      auto y = random_series(rng, n);
      const data_t lb = webb_sq(x, y, band);
      const data_t dtw = dtwc::dtwBanded<data_t>(x, y, band, -1.0, MetricType::SquaredL2);
      INFO("band=" << band << " n=" << n << " LB_Webb_sq=" << lb << " DTW_sq=" << dtw);
      REQUIRE(lb <= dtw + 1e-6);
      REQUIRE(lb >= 0.0);
    }
  }
}

TEST_CASE("LB_Webb symmetric <= banded DTW (L1)",
          "[adversarial][lb_webb][symmetric][contract]")
{
  std::mt19937 rng(12345);
  constexpr int band = 8;
  for (int p = 0; p < 40; ++p) {
    const auto n = random_length(rng, 40, 150);
    auto x = random_series(rng, n);
    auto y = random_series(rng, n);
    const data_t lb = webb_sym_l1(x, y, band);
    const data_t dtw = dtwc::dtwBanded<data_t>(x, y, band);
    INFO("n=" << n << " LB_Webb_sym=" << lb << " DTW=" << dtw);
    REQUIRE(lb <= dtw + 1e-9);
  }
}

TEST_CASE("LB_Enhanced <= banded DTW at 10% band (the plan's band setting)",
          "[adversarial][lb_enhanced][lb_webb][contract][band10pct]")
{
  std::mt19937 rng(2024);
  for (int p = 0; p < 40; ++p) {
    const auto n = random_length(rng, 50, 300);
    const int band = std::max(1, static_cast<int>(n) / 10);
    auto x = random_series(rng, n);
    auto y = random_series(rng, n);
    const data_t dtw = dtwc::dtwBanded<data_t>(x, y, band);
    const data_t lbe = enhanced_l1(x, y, band);
    const data_t lbw = webb_l1(x, y, band);
    INFO("n=" << n << " band=" << band << " enh=" << lbe << " webb=" << lbw << " dtw=" << dtw);
    REQUIRE(lbe <= dtw + 1e-9);
    REQUIRE(lbw <= dtw + 1e-9);
  }
}

// =========================================================================
//  AREA 2: Edge lengths — the disjointness/clamp regime (n small vs 2V)
// =========================================================================

TEST_CASE("LB_Enhanced/LB_Webb <= DTW for tiny lengths and n ~ 2V",
          "[adversarial][lb_enhanced][lb_webb][contract][edge]")
{
  std::mt19937 rng(777);
  const std::vector<std::size_t> lengths = { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 };
  for (std::size_t n : lengths) {
    for (int band : { 0, 1, 2, 5 }) {
      for (int p = 0; p < 30; ++p) {
        auto x = random_series(rng, n);
        auto y = random_series(rng, n);
        const data_t dtw_l1 = dtwc::dtwBanded<data_t>(x, y, band);
        const data_t lbe = enhanced_l1(x, y, band);     // V=5 clamped to n/2
        const data_t lbw = webb_l1(x, y, band);
        INFO("n=" << n << " band=" << band << " enh=" << lbe << " webb=" << lbw
             << " dtw=" << dtw_l1);
        REQUIRE(lbe <= dtw_l1 + 1e-9);
        REQUIRE(lbw <= dtw_l1 + 1e-9);
        REQUIRE(lbe >= 0.0);
        REQUIRE(lbw >= 0.0);
      }
    }
  }
}

// =========================================================================
//  AREA 3: local NoLR-plus-tail-cap >= Keogh (bridge + nonnegative corrections)
// =========================================================================

TEST_CASE("LB_Webb(A,B) >= LB_Keogh(A, env B) per instance (L1)",
          "[adversarial][lb_webb][tightness]")
{
  std::mt19937 rng(12345);
  for (int band : { 1, 3, 8, 20 }) {
    for (int p = 0; p < 40; ++p) {
      const auto n = random_length(rng, 30, 150);
      auto x = random_series(rng, n);
      auto y = random_series(rng, n);
      auto wy = dtwc::core::compute_webb_envelope(y, band);
      const data_t keogh1 = dtwc::core::lb_keogh(x.data(), n, wy.upper.data(), wy.lower.data());
      const data_t webb = webb_l1(x, y, band);
      INFO("band=" << band << " n=" << n << " keogh1=" << keogh1 << " webb=" << webb);
      REQUIRE(webb >= keogh1 - 1e-9);
    }
  }
}

TEST_CASE("LB_Webb symmetric >= LB_Keogh symmetric per instance (L1)",
          "[adversarial][lb_webb][symmetric][tightness]")
{
  std::mt19937 rng(54321);
  for (int band : { 1, 5, 15 }) {
    for (int p = 0; p < 40; ++p) {
      const auto n = random_length(rng, 30, 150);
      auto x = random_series(rng, n);
      auto y = random_series(rng, n);
      auto ex = dtwc::core::compute_envelope(x, band);
      auto ey = dtwc::core::compute_envelope(y, band);
      const data_t keogh_sym = dtwc::core::lb_keogh_symmetric(x, ex, y, ey);
      const data_t webb_sym = webb_sym_l1(x, y, band);
      INFO("band=" << band << " n=" << n << " keogh_sym=" << keogh_sym
           << " webb_sym=" << webb_sym);
      REQUIRE(webb_sym >= keogh_sym - 1e-9);
    }
  }
}

// =========================================================================
//  AREA 4: Identity, non-negativity
// =========================================================================

TEST_CASE("LB_Enhanced(x,x) == 0 and LB_Webb(x,x) == 0",
          "[adversarial][lb_enhanced][lb_webb][identity]")
{
  std::mt19937 rng(12345);
  for (int band : { 0, 1, 5, 10, 50 }) {
    for (int p = 0; p < 15; ++p) {
      const auto n = random_length(rng, 5, 120);
      auto x = random_series(rng, n);
      INFO("band=" << band << " n=" << n);
      REQUIRE_THAT(enhanced_l1(x, x, band), WithinAbs(0.0, 1e-12));
      REQUIRE_THAT(webb_l1(x, x, band), WithinAbs(0.0, 1e-12));
      REQUIRE_THAT(webb_sym_l1(x, x, band), WithinAbs(0.0, 1e-12));
    }
  }
}

// =========================================================================
//  AREA 5: Adversarial validity (shared endpoints, just-outside-envelope,
//          constant, extreme values)
// =========================================================================

TEST_CASE("LB_Enhanced/LB_Webb <= DTW adversarial: shared endpoints",
          "[adversarial][lb_enhanced][lb_webb][contract][endpoints]")
{
  std::mt19937 rng(12345);
  constexpr int band = 6;
  for (int p = 0; p < 30; ++p) {
    const auto n = random_length(rng, 30, 100);
    auto x = random_series(rng, n, -100.0, 100.0);
    auto y = random_series(rng, n, -100.0, 100.0);
    x.front() = y.front() = 0.0;
    x.back() = y.back() = 5.0;
    const data_t dtw = dtwc::dtwBanded<data_t>(x, y, band);
    REQUIRE(enhanced_l1(x, y, band) <= dtw + 1e-9);
    REQUIRE(webb_l1(x, y, band) <= dtw + 1e-9);
  }
}

TEST_CASE("LB_Webb <= DTW adversarial: query exactly epsilon above envelope",
          "[adversarial][lb_webb][contract][envelope_edge]")
{
  std::mt19937 rng(12345);
  constexpr int band = 5;
  for (int p = 0; p < 20; ++p) {
    const auto n = random_length(rng, 30, 80);
    auto cand = random_series(rng, n, -5.0, 5.0);
    auto wc = dtwc::core::compute_webb_envelope(cand, band);
    const data_t epsilon = 0.01;
    std::vector<data_t> query(n);
    for (std::size_t i = 0; i < n; ++i) query[i] = wc.upper[i] + epsilon;
    auto wq = dtwc::core::compute_webb_envelope(query, band);
    const data_t lb = dtwc::core::lb_webb<L1Metric>(query, wq, cand, wc, band);
    const data_t dtw = dtwc::dtwBanded<data_t>(query, cand, band);
    INFO("n=" << n << " lb=" << lb << " dtw=" << dtw);
    REQUIRE(lb <= dtw + 1e-9);
    REQUIRE(lb >= 0.0);
  }
}

TEST_CASE("LB_Enhanced/LB_Webb handle constant and extreme series",
          "[adversarial][lb_enhanced][lb_webb][extreme]")
{
  const std::vector<data_t> const_a(40, 42.0);
  const std::vector<data_t> const_b(40, 100.0);
  for (int band : { 1, 5, 20 }) {
    const data_t dtw = dtwc::dtwBanded<data_t>(const_a, const_b, band);
    REQUIRE(enhanced_l1(const_a, const_b, band) <= dtw + 1e-9);
    REQUIRE(webb_l1(const_a, const_b, band) <= dtw + 1e-9);
  }
  // Identical constant series -> 0.
  REQUIRE_THAT(webb_l1(const_a, const_a, 5), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(enhanced_l1(const_a, const_a, 5), WithinAbs(0.0, 1e-12));

  // Extreme magnitudes.
  const std::vector<data_t> big = { 1e12, -1e12, 1e12, -1e12, 1e12, -1e12, 1e12, -1e12 };
  const std::vector<data_t> smallv(8, 1e-9);
  for (int band : { 1, 2, 4 }) {
    const data_t dtw = dtwc::dtwBanded<data_t>(big, smallv, band);
    REQUIRE(enhanced_l1(big, smallv, band) <= dtw + 1e-3);
    REQUIRE(webb_l1(big, smallv, band) <= dtw + 1e-3);
    REQUIRE(enhanced_l1(big, smallv, band) >= 0.0);
    REQUIRE(webb_l1(big, smallv, band) >= 0.0);
  }
}

// =========================================================================
//  AREA 6: WebbEnvelope secondary-envelope correctness vs naive
// =========================================================================

/// Naive sliding-window min/max over [i-w, i+w].
static void naive_minmax(const std::vector<data_t> &s, int band,
                         std::vector<data_t> &mx, std::vector<data_t> &mn)
{
  const std::size_t n = s.size();
  const std::size_t w = static_cast<std::size_t>(std::max(band, 0));
  mx.resize(n); mn.resize(n);
  for (std::size_t p = 0; p < n; ++p) {
    const std::size_t lo = (p > w) ? p - w : 0;
    const std::size_t hi = std::min(p + w + 1, n);
    mx[p] = *std::max_element(s.begin() + lo, s.begin() + hi);
    mn[p] = *std::min_element(s.begin() + lo, s.begin() + hi);
  }
}

TEST_CASE("WebbEnvelope: lu = L(U), ul = U(L) match naive double-window",
          "[adversarial][lb_webb][envelope]")
{
  std::mt19937 rng(31415);
  for (int band : { 1, 3, 7, 20 }) {
    for (int p = 0; p < 15; ++p) {
      const auto n = random_length(rng, 20, 120);
      auto s = random_series(rng, n);
      auto we = dtwc::core::compute_webb_envelope(s, band);

      std::vector<data_t> U, L, luRef, tmp, ulRef;
      naive_minmax(s, band, U, L);              // primary
      naive_minmax(U, band, tmp, luRef);        // lu = min-window of U
      naive_minmax(L, band, ulRef, tmp);        // ul = max-window of L

      for (std::size_t i = 0; i < n; ++i) {
        INFO("band=" << band << " i=" << i);
        REQUIRE_THAT(we.upper[i], WithinAbs(U[i], 1e-12));
        REQUIRE_THAT(we.lower[i], WithinAbs(L[i], 1e-12));
        REQUIRE_THAT(we.lu[i], WithinAbs(luRef[i], 1e-12));
        REQUIRE_THAT(we.ul[i], WithinAbs(ulRef[i], 1e-12));
      }
      // Sanity: L <= LU <= U and L <= UL <= U (secondary envelopes nest inside).
      for (std::size_t i = 0; i < n; ++i) {
        REQUIRE(we.lower[i] <= we.lu[i] + 1e-12);
        REQUIRE(we.lu[i] <= we.upper[i] + 1e-12);
        REQUIRE(we.lower[i] <= we.ul[i] + 1e-12);
        REQUIRE(we.ul[i] <= we.upper[i] + 1e-12);
      }
    }
  }
}

// =========================================================================
//  AREA 7: Tightness bench (hidden, ADVISORY) — mean LB/DTW ratio at band=10%
//
//  REGISTERED BANDS (before the run):
//    - mean(webb_sym) >= mean(keogh_sym) >= mean(kim)     [expected]
//    - mean(enhanced_sym): reported, with no ordering asserted at effective
//      V>=2 (later D3 exact witnesses run in both directions). If
//      mean(webb_sym) < mean(keogh_sym) on clustered data the
//      "Webb tightens the cascade" claim is FALSIFIED (a deliverable).
//  These are TIGHTNESS ratios (bound / true DTW; higher = tighter, max 1.0),
//  NOT a matrix-build speedup: on an exact matrix no LB reduces DTW calls.
// =========================================================================

/// One "clustered" synthetic set: K random-walk prototypes, each cloned with
/// small additive noise (equal length, so envelope bounds apply).
static std::vector<std::vector<data_t>> make_clustered_set(std::mt19937 &rng,
                                                           int K, int per, std::size_t n)
{
  std::normal_distribution<data_t> step(0.0, 1.0), noise(0.0, 0.25);
  std::vector<std::vector<data_t>> protos(K, std::vector<data_t>(n));
  for (int k = 0; k < K; ++k) {
    data_t v = 0;
    for (std::size_t t = 0; t < n; ++t) { v += step(rng); protos[k][t] = v; }
  }
  std::vector<std::vector<data_t>> out;
  for (int k = 0; k < K; ++k)
    for (int c = 0; c < per; ++c) {
      std::vector<data_t> s(n);
      for (std::size_t t = 0; t < n; ++t) s[t] = protos[k][t] + noise(rng);
      out.push_back(std::move(s));
    }
  return out;
}

/// Run the tightness table for one band; returns {mean_keogh, mean_enhanced, mean_webb}.
static void tightness_table(std::mt19937 &rng, std::size_t n, int band, const char *label)
{
  std::printf("\n LB tightness = mean(LB / DTW), band=%d (%s of %zu)  [higher = tighter, max 1.0]\n",
              band, label, n);
  std::printf(" set |   pairs |     kim |  keogh_sym | enhanced_sym | webb_sym\n");
  double gk = 0, gke = 0, ge = 0, gw = 0; long gpairs = 0;
  for (int set = 0; set < 5; ++set) {
    auto data = make_clustered_set(rng, /*K=*/4, /*per=*/8, n);   // 32 series
    const int M = static_cast<int>(data.size());
    std::vector<dtwc::core::Envelope> env(M);
    std::vector<dtwc::core::WebbEnvelope> wenv(M);
    std::vector<dtwc::core::SeriesSummary> summ(M);
    for (int i = 0; i < M; ++i) {
      env[i] = dtwc::core::compute_envelope(data[i], band);
      wenv[i] = dtwc::core::compute_webb_envelope(data[i], band);
      summ[i] = dtwc::core::compute_summary(data[i]);
    }
    double sk = 0, ske = 0, se = 0, sw = 0; long pairs = 0;
    std::vector<char> scratch;
    for (int i = 0; i < M; ++i)
      for (int j = i + 1; j < M; ++j) {
        const double dtw = dtwc::dtwBanded<data_t>(data[i], data[j], band);
        if (dtw <= 1e-12) continue;
        sk  += dtwc::core::lb_kim(summ[i], summ[j]) / dtw;
        ske += dtwc::core::lb_keogh_symmetric(data[i], env[i], data[j], env[j]) / dtw;
        se  += dtwc::core::lb_enhanced_symmetric(std::span<const double>(data[i]), env[i],
                 std::span<const double>(data[j]), env[j], band) / dtw;
        sw  += dtwc::core::lb_webb_symmetric(std::span<const double>(data[i]), wenv[i],
                 std::span<const double>(data[j]), wenv[j], band, &scratch) / dtw;
        ++pairs;
      }
    std::printf(" %3d | %7ld | %7.4f | %10.4f | %12.4f | %8.4f\n",
                set, pairs, sk / pairs, ske / pairs, se / pairs, sw / pairs);
    gk += sk; gke += ske; ge += se; gw += sw; gpairs += pairs;
  }
  std::printf(" ALL | %7ld | %7.4f | %10.4f | %12.4f | %8.4f\n",
              gpairs, gk / gpairs, gke / gpairs, ge / gpairs, gw / gpairs);
  std::printf(" verdict: webb_sym %s keogh_sym (rel gain %.1f%%); enhanced_sym rel gain %.1f%%\n",
              (gw >= gke - 1e-12) ? ">=" : "< FALSIFIED",
              100.0 * (gw - gke) / gke, 100.0 * (ge - gke) / gke);
}

TEST_CASE("LB tightness bench: kim vs keogh vs enhanced vs webb, narrow + wide band",
          "[.][lb_tightness][bench]")
{
  const std::size_t n = 128;
  // Two regimes: the plan's 10% band, and a wide 40% band where LB_Enhanced's
  // elastic bands are designed to pay off (Tan et al. SDM 2019: optimal V grows
  // with the window). Fresh RNG per table so each starts from the same stream.
  { std::mt19937 rng(20260708); tightness_table(rng, n, static_cast<int>(n) / 10, "10%"); }
  { std::mt19937 rng(20260708); tightness_table(rng, n, static_cast<int>(2 * n) / 5, "40%"); }
}
