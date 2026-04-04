/**
 * @file test_lower_bounds_adversarial.cpp
 * @brief Adversarial tests for LB_Kim and LB_Keogh lower bound functions.
 *
 * @details These tests verify MATHEMATICAL PROPERTIES, not implementation details.
 *          The fundamental contract of any DTW lower bound is:
 *              LB(x, y) <= DTW(x, y)   for ALL x, y
 *          If this property fails, any pruning based on the lower bound will
 *          silently produce WRONG RESULTS (missed nearest neighbours).
 *
 *          Tests are grouped into:
 *            Area 1: LB <= DTW (the fundamental contract)
 *            Area 2: Envelope correctness
 *
 *          All randomised tests use std::mt19937 with seed=12345 for reproducibility.
 *
 * @author Volkan Kumtepeli
 * @author Claude 4.6
 * @date 28 Mar 2026
 */

#include <core/lower_bound_impl.hpp>
#include <warping.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;
using data_t = double;

// =========================================================================
//  Helpers
// =========================================================================

/// Generate a random series of given length using the provided RNG.
static std::vector<data_t> random_series(std::mt19937 &rng, std::size_t len,
                                         data_t lo = -10.0, data_t hi = 10.0)
{
  std::uniform_real_distribution<data_t> dist(lo, hi);
  std::vector<data_t> s(len);
  for (auto &v : s) v = dist(rng);
  return s;
}

/// Generate a random length in [min_len, max_len].
static std::size_t random_length(std::mt19937 &rng, std::size_t min_len, std::size_t max_len)
{
  std::uniform_int_distribution<std::size_t> dist(min_len, max_len);
  return dist(rng);
}

/// Compute envelopes and return lb_keogh in one call (convenience wrapper).
static data_t compute_lb_keogh(const std::vector<data_t> &query,
                               const std::vector<data_t> &candidate, int band)
{
  std::vector<data_t> upper, lower;
  dtwc::core::compute_envelopes(candidate, band, upper, lower);
  return dtwc::core::lb_keogh(query, upper, lower);
}

/// Symmetric LB_Keogh: max(lb_keogh(x, env_y), lb_keogh(y, env_x)).
static data_t compute_lb_keogh_symmetric(const std::vector<data_t> &x,
                                         const std::vector<data_t> &y, int band)
{
  std::vector<data_t> upper_x, lower_x, upper_y, lower_y;
  dtwc::core::compute_envelopes(x, band, upper_x, lower_x);
  dtwc::core::compute_envelopes(y, band, upper_y, lower_y);
  const data_t lb1 = dtwc::core::lb_keogh(x, upper_y, lower_y);
  const data_t lb2 = dtwc::core::lb_keogh(y, upper_x, lower_x);
  return std::max(lb1, lb2);
}

/// Simple series summary for LB_Kim consistency tests.
struct SeriesSummary {
  data_t first = 0, last = 0, min_val = 0, max_val = 0;
};

static SeriesSummary compute_summary(const std::vector<data_t> &series)
{
  if (series.empty()) return {};
  SeriesSummary s;
  s.first = series.front();
  s.last = series.back();
  s.min_val = *std::min_element(series.begin(), series.end());
  s.max_val = *std::max_element(series.begin(), series.end());
  return s;
}

/// LB_Kim from precomputed summaries (must match the direct version).
static data_t lb_kim_from_summary(const SeriesSummary &a, const SeriesSummary &b)
{
  data_t d = 0;
  d = std::max(d, std::abs(a.first - b.first));
  d = std::max(d, std::abs(a.last - b.last));
  d = std::max(d, std::abs(a.min_val - b.min_val));
  d = std::max(d, std::abs(a.max_val - b.max_val));
  return d;
}


// =========================================================================
//  AREA 1: LB <= DTW  (The Fundamental Contract)
// =========================================================================

// -------------------------------------------------------------------------
// 1a. LB_Kim <= DTW for 50 random pairs (lengths 50-200)
// -------------------------------------------------------------------------
TEST_CASE("LB_Kim <= DTW for 50 random pairs", "[adversarial][lb_kim][contract]")
{
  std::mt19937 rng(12345);

  for (int p = 0; p < 50; ++p) {
    const auto nx = random_length(rng, 50, 200);
    const auto ny = random_length(rng, 50, 200);
    auto x = random_series(rng, nx);
    auto y = random_series(rng, ny);

    const data_t lb = dtwc::core::lb_kim(x, y);
    const data_t dtw = dtwc::dtwFull_L<data_t>(x, y);

    INFO("Pair " << p << " (nx=" << nx << ", ny=" << ny
         << "): LB_Kim=" << lb << " DTW=" << dtw);
    REQUIRE(lb <= dtw + 1e-10);
  }
}

// -------------------------------------------------------------------------
// 1b. LB_Keogh <= DTW for 50 random pairs, band=10
// -------------------------------------------------------------------------
TEST_CASE("LB_Keogh <= DTW for 50 random equal-length pairs, band=10",
          "[adversarial][lb_keogh][contract]")
{
  std::mt19937 rng(12345);
  constexpr int band = 10;

  for (int p = 0; p < 50; ++p) {
    const auto n = random_length(rng, 50, 200);
    auto x = random_series(rng, n);
    auto y = random_series(rng, n);

    const data_t lb = compute_lb_keogh(x, y, band);
    const data_t dtw = dtwc::dtwBanded<data_t>(x, y, band);

    INFO("Pair " << p << " (n=" << n << ", band=10): LB_Keogh=" << lb
         << " DTW_banded=" << dtw);
    REQUIRE(lb <= dtw + 1e-10);
  }
}

// -------------------------------------------------------------------------
// 1c. LB_Keogh <= DTW for 50 random pairs, band=50
// -------------------------------------------------------------------------
TEST_CASE("LB_Keogh <= DTW for 50 random equal-length pairs, band=50",
          "[adversarial][lb_keogh][contract]")
{
  std::mt19937 rng(12345);
  constexpr int band = 50;

  for (int p = 0; p < 50; ++p) {
    const auto n = random_length(rng, 60, 200);
    auto x = random_series(rng, n);
    auto y = random_series(rng, n);

    const data_t lb = compute_lb_keogh(x, y, band);
    const data_t dtw = dtwc::dtwBanded<data_t>(x, y, band);

    INFO("Pair " << p << " (n=" << n << ", band=50): LB_Keogh=" << lb
         << " DTW_banded=" << dtw);
    REQUIRE(lb <= dtw + 1e-10);
  }
}

// -------------------------------------------------------------------------
// 1d. LB_Kim <= DTW for adversarial: identical endpoints, different interiors
//     This is designed to stress the min/max features of LB_Kim.
// -------------------------------------------------------------------------
TEST_CASE("LB_Kim <= DTW adversarial: same endpoints, different interiors",
          "[adversarial][lb_kim][contract]")
{
  std::mt19937 rng(12345);

  for (int p = 0; p < 20; ++p) {
    const auto n = random_length(rng, 30, 100);

    // Both series share the same first and last element
    const data_t shared_first = 0.0;
    const data_t shared_last = 5.0;

    auto x = random_series(rng, n, -100.0, 100.0);
    auto y = random_series(rng, n, -100.0, 100.0);

    x.front() = shared_first;  x.back() = shared_last;
    y.front() = shared_first;  y.back() = shared_last;

    const data_t lb = dtwc::core::lb_kim(x, y);
    const data_t dtw = dtwc::dtwFull_L<data_t>(x, y);

    INFO("Pair " << p << ": LB_Kim=" << lb << " DTW=" << dtw);
    REQUIRE(lb <= dtw + 1e-10);

    // Since endpoints match, LB_Kim's first/last features are 0.
    // The bound comes entirely from min/max features.
    REQUIRE(lb >= 0.0);
  }
}

// -------------------------------------------------------------------------
// 1e. LB_Keogh <= DTW adversarial: query just outside the envelope band
//     Construct query that is exactly epsilon above the upper envelope
//     at every point. LB should be small but positive, and <= DTW.
// -------------------------------------------------------------------------
TEST_CASE("LB_Keogh <= DTW adversarial: query just outside envelope",
          "[adversarial][lb_keogh][contract]")
{
  std::mt19937 rng(12345);
  constexpr int band = 5;

  for (int p = 0; p < 20; ++p) {
    const auto n = random_length(rng, 30, 80);
    auto candidate = random_series(rng, n, -5.0, 5.0);

    std::vector<data_t> upper(n), lower(n);
    dtwc::core::compute_envelopes(candidate, band, upper, lower);

    // Construct query that sits epsilon above the upper envelope
    const data_t epsilon = 0.01;
    std::vector<data_t> query(n);
    for (std::size_t i = 0; i < n; ++i) {
      query[i] = upper[i] + epsilon;
    }

    const data_t lb = dtwc::core::lb_keogh(query, upper, lower);
    const data_t dtw = dtwc::dtwBanded<data_t>(query, candidate, band);

    INFO("Pair " << p << " (n=" << n << "): LB_Keogh=" << lb
         << " DTW_banded=" << dtw);
    REQUIRE(lb <= dtw + 1e-10);

    // LB should be approximately n * epsilon (each point contributes epsilon)
    const data_t expected_lb = static_cast<data_t>(n) * epsilon;
    REQUIRE_THAT(lb, WithinAbs(expected_lb, 1e-10));
  }
}

// -------------------------------------------------------------------------
// 1f. Symmetric LB_Keogh <= DTW
// -------------------------------------------------------------------------
TEST_CASE("Symmetric LB_Keogh <= DTW for 30 random pairs",
          "[adversarial][lb_keogh][symmetric][contract]")
{
  std::mt19937 rng(12345);
  constexpr int band = 8;

  for (int p = 0; p < 30; ++p) {
    const auto n = random_length(rng, 40, 150);
    auto x = random_series(rng, n);
    auto y = random_series(rng, n);

    const data_t lb_sym = compute_lb_keogh_symmetric(x, y, band);
    const data_t dtw = dtwc::dtwBanded<data_t>(x, y, band);

    INFO("Pair " << p << " (n=" << n << "): LB_Keogh_sym=" << lb_sym
         << " DTW_banded=" << dtw);
    REQUIRE(lb_sym <= dtw + 1e-10);
  }
}

// -------------------------------------------------------------------------
// 1g. Symmetric LB_Keogh >= max(lb_keogh(x, env_y), lb_keogh(y, env_x))
//     The symmetric version should be AT LEAST as tight as either direction.
// -------------------------------------------------------------------------
TEST_CASE("Symmetric LB_Keogh >= each single-direction LB_Keogh",
          "[adversarial][lb_keogh][symmetric][tightness]")
{
  std::mt19937 rng(12345);
  constexpr int band = 5;

  for (int p = 0; p < 30; ++p) {
    const auto n = random_length(rng, 30, 100);
    auto x = random_series(rng, n);
    auto y = random_series(rng, n);

    std::vector<data_t> upper_x, lower_x, upper_y, lower_y;
    dtwc::core::compute_envelopes(x, band, upper_x, lower_x);
    dtwc::core::compute_envelopes(y, band, upper_y, lower_y);

    const data_t lb_xy = dtwc::core::lb_keogh(x, upper_y, lower_y);
    const data_t lb_yx = dtwc::core::lb_keogh(y, upper_x, lower_x);
    const data_t lb_sym = std::max(lb_xy, lb_yx);

    INFO("Pair " << p << ": lb(x,env_y)=" << lb_xy << " lb(y,env_x)=" << lb_yx
         << " lb_sym=" << lb_sym);
    REQUIRE(lb_sym >= lb_xy - 1e-10);
    REQUIRE(lb_sym >= lb_yx - 1e-10);
  }
}

// -------------------------------------------------------------------------
// 1h. LB_Kim(x, x) == 0: identical series
// -------------------------------------------------------------------------
TEST_CASE("LB_Kim(x, x) == 0 for various series", "[adversarial][lb_kim][identity]")
{
  std::mt19937 rng(12345);

  for (int p = 0; p < 20; ++p) {
    const auto n = random_length(rng, 10, 100);
    auto x = random_series(rng, n, -50.0, 50.0);

    const data_t lb = dtwc::core::lb_kim(x, x);
    REQUIRE_THAT(lb, WithinAbs(0.0, 1e-15));
  }
}

// -------------------------------------------------------------------------
// 1i. LB_Keogh(x, x) == 0: identical series (query == candidate)
// -------------------------------------------------------------------------
TEST_CASE("LB_Keogh(x, x) == 0 for various series and bands",
          "[adversarial][lb_keogh][identity]")
{
  std::mt19937 rng(12345);
  const std::vector<int> bands = { 0, 1, 5, 10, 50 };

  for (int p = 0; p < 10; ++p) {
    const auto n = random_length(rng, 20, 100);
    auto x = random_series(rng, n);

    for (int band : bands) {
      std::vector<data_t> upper, lower;
      dtwc::core::compute_envelopes(x, band, upper, lower);
      const data_t lb = dtwc::core::lb_keogh(x, upper, lower);

      INFO("Series " << p << ", n=" << n << ", band=" << band << ": lb=" << lb);
      REQUIRE_THAT(lb, WithinAbs(0.0, 1e-15));
    }
  }
}

// -------------------------------------------------------------------------
// 1j. LB_Kim >= 0: non-negativity
// -------------------------------------------------------------------------
TEST_CASE("LB_Kim >= 0 for all random pairs", "[adversarial][lb_kim][non_negative]")
{
  std::mt19937 rng(12345);

  for (int p = 0; p < 50; ++p) {
    const auto nx = random_length(rng, 10, 200);
    const auto ny = random_length(rng, 10, 200);
    auto x = random_series(rng, nx, -100.0, 100.0);
    auto y = random_series(rng, ny, -100.0, 100.0);

    const data_t lb = dtwc::core::lb_kim(x, y);
    INFO("Pair " << p << ": lb_kim=" << lb);
    REQUIRE(lb >= 0.0);
  }
}

// -------------------------------------------------------------------------
// 1k. LB_Keogh >= 0: non-negativity
// -------------------------------------------------------------------------
TEST_CASE("LB_Keogh >= 0 for all random pairs", "[adversarial][lb_keogh][non_negative]")
{
  std::mt19937 rng(12345);
  constexpr int band = 10;

  for (int p = 0; p < 50; ++p) {
    const auto n = random_length(rng, 20, 200);
    auto x = random_series(rng, n);
    auto y = random_series(rng, n);

    const data_t lb = compute_lb_keogh(x, y, band);

    INFO("Pair " << p << ": lb_keogh=" << lb);
    REQUIRE(lb >= 0.0);
  }
}


// =========================================================================
//  AREA 2: Envelope Correctness
// =========================================================================

// -------------------------------------------------------------------------
// 2a. Envelope contains the series: lower[i] <= series[i] <= upper[i]
// -------------------------------------------------------------------------
TEST_CASE("Envelope always contains the original series",
          "[adversarial][envelope][containment]")
{
  std::mt19937 rng(12345);
  const std::vector<int> bands = { 0, 1, 3, 10, 50 };

  for (int p = 0; p < 20; ++p) {
    const auto n = random_length(rng, 10, 200);
    auto series = random_series(rng, n, -100.0, 100.0);

    for (int band : bands) {
      std::vector<data_t> upper(n), lower(n);
      dtwc::core::compute_envelopes(series, band, upper, lower);

      for (std::size_t i = 0; i < n; ++i) {
        INFO("Series " << p << ", band=" << band << ", i=" << i
             << ": series=" << series[i] << " lower=" << lower[i]
             << " upper=" << upper[i]);
        REQUIRE(lower[i] <= series[i] + 1e-15);
        REQUIRE(series[i] <= upper[i] + 1e-15);
      }
    }
  }
}

// -------------------------------------------------------------------------
// 2b. Envelope width: upper[i] - lower[i] matches sliding window min-max
// -------------------------------------------------------------------------
TEST_CASE("Envelope width matches sliding window min-max",
          "[adversarial][envelope][width]")
{
  std::mt19937 rng(12345);

  for (int p = 0; p < 15; ++p) {
    const auto n = random_length(rng, 20, 100);
    auto series = random_series(rng, n);
    constexpr int band = 4;

    std::vector<data_t> upper(n), lower(n);
    dtwc::core::compute_envelopes(series, band, upper, lower);

    for (std::size_t i = 0; i < n; ++i) {
      // Compute the expected window min/max by brute force
      const std::size_t lo = (i > static_cast<std::size_t>(band)) ? i - band : 0;
      const std::size_t hi = std::min(i + static_cast<std::size_t>(band) + 1, n);

      data_t expected_min = series[lo];
      data_t expected_max = series[lo];
      for (std::size_t j = lo + 1; j < hi; ++j) {
        expected_min = std::min(expected_min, series[j]);
        expected_max = std::max(expected_max, series[j]);
      }

      INFO("Series " << p << ", i=" << i << ": expected [" << expected_min
           << ", " << expected_max << "], got [" << lower[i] << ", "
           << upper[i] << "]");
      REQUIRE_THAT(lower[i], WithinAbs(expected_min, 1e-15));
      REQUIRE_THAT(upper[i], WithinAbs(expected_max, 1e-15));
    }
  }
}

// -------------------------------------------------------------------------
// 2c. Band=0 envelope: upper[i] == lower[i] == series[i]
// -------------------------------------------------------------------------
TEST_CASE("Band=0 envelope is identity (no wiggle room)",
          "[adversarial][envelope][band_zero]")
{
  std::mt19937 rng(12345);

  for (int p = 0; p < 10; ++p) {
    const auto n = random_length(rng, 10, 100);
    auto series = random_series(rng, n);

    std::vector<data_t> upper(n), lower(n);
    dtwc::core::compute_envelopes(series, 0, upper, lower);

    for (std::size_t i = 0; i < n; ++i) {
      REQUIRE_THAT(upper[i], WithinAbs(series[i], 1e-15));
      REQUIRE_THAT(lower[i], WithinAbs(series[i], 1e-15));
    }
  }
}

// -------------------------------------------------------------------------
// 2d. Band >= length: envelope is global min/max everywhere
// -------------------------------------------------------------------------
TEST_CASE("Band >= length: envelope is global min/max",
          "[adversarial][envelope][full_band]")
{
  std::mt19937 rng(12345);

  for (int p = 0; p < 10; ++p) {
    const auto n = random_length(rng, 10, 100);
    auto series = random_series(rng, n, -50.0, 50.0);

    const data_t global_min = *std::min_element(series.begin(), series.end());
    const data_t global_max = *std::max_element(series.begin(), series.end());

    // Use band much larger than series length
    std::vector<data_t> upper(n), lower(n);
    dtwc::core::compute_envelopes(series, static_cast<int>(n + 100), upper, lower);

    for (std::size_t i = 0; i < n; ++i) {
      INFO("Series " << p << ", i=" << i);
      REQUIRE_THAT(upper[i], WithinAbs(global_max, 1e-15));
      REQUIRE_THAT(lower[i], WithinAbs(global_min, 1e-15));
    }
  }
}

// -------------------------------------------------------------------------
// 2e. SeriesSummary correctness: first, last, min, max
// -------------------------------------------------------------------------
TEST_CASE("compute_summary returns correct first, last, min, max",
          "[adversarial][summary][correctness]")
{
  std::mt19937 rng(12345);

  for (int p = 0; p < 20; ++p) {
    const auto n = random_length(rng, 5, 200);
    auto series = random_series(rng, n, -1000.0, 1000.0);

    SeriesSummary s = compute_summary(series);

    REQUIRE_THAT(s.first, WithinAbs(series.front(), 1e-15));
    REQUIRE_THAT(s.last, WithinAbs(series.back(), 1e-15));
    REQUIRE_THAT(s.min_val, WithinAbs(*std::min_element(series.begin(), series.end()), 1e-15));
    REQUIRE_THAT(s.max_val, WithinAbs(*std::max_element(series.begin(), series.end()), 1e-15));
  }
}

// -------------------------------------------------------------------------
// 2e-bis. compute_summary on empty series
// -------------------------------------------------------------------------
TEST_CASE("compute_summary on empty series returns zeros",
          "[adversarial][summary][edge]")
{
  const std::vector<data_t> empty;
  SeriesSummary s = compute_summary(empty);

  REQUIRE_THAT(s.first, WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(s.last, WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(s.min_val, WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(s.max_val, WithinAbs(0.0, 1e-15));
}

// -------------------------------------------------------------------------
// 2f. lb_kim(summary_a, summary_b) == lb_kim(vector_a, vector_b)
//     The precomputed-summary version MUST match the direct version.
// -------------------------------------------------------------------------
TEST_CASE("lb_kim with precomputed summaries matches direct computation",
          "[adversarial][lb_kim][summary][consistency]")
{
  std::mt19937 rng(12345);

  for (int p = 0; p < 50; ++p) {
    const auto nx = random_length(rng, 5, 200);
    const auto ny = random_length(rng, 5, 200);
    auto x = random_series(rng, nx, -100.0, 100.0);
    auto y = random_series(rng, ny, -100.0, 100.0);

    const data_t lb_direct = dtwc::core::lb_kim(x, y);

    SeriesSummary sx = compute_summary(x);
    SeriesSummary sy = compute_summary(y);
    const data_t lb_summary = lb_kim_from_summary(sx, sy);

    INFO("Pair " << p << ": lb_direct=" << lb_direct << " lb_summary=" << lb_summary);
    REQUIRE_THAT(lb_summary, WithinAbs(lb_direct, 1e-12));
  }
}

// -------------------------------------------------------------------------
// 2g. Monotonicity: wider band => wider (or equal) envelope
// -------------------------------------------------------------------------
TEST_CASE("Wider band produces wider or equal envelope",
          "[adversarial][envelope][monotonicity]")
{
  std::mt19937 rng(12345);

  for (int p = 0; p < 15; ++p) {
    const auto n = random_length(rng, 20, 100);
    auto series = random_series(rng, n);

    std::vector<data_t> upper_narrow(n), lower_narrow(n);
    std::vector<data_t> upper_wide(n), lower_wide(n);
    dtwc::core::compute_envelopes(series, 2, upper_narrow, lower_narrow);
    dtwc::core::compute_envelopes(series, 10, upper_wide, lower_wide);

    for (std::size_t i = 0; i < n; ++i) {
      INFO("Series " << p << ", i=" << i);
      // Wider band => lower envelope is lower or equal
      REQUIRE(lower_wide[i] <= lower_narrow[i] + 1e-15);
      // Wider band => upper envelope is higher or equal
      REQUIRE(upper_wide[i] >= upper_narrow[i] - 1e-15);
    }
  }
}

// -------------------------------------------------------------------------
// 2h. LB_Keogh with wider band produces SMALLER or equal lower bound
// -------------------------------------------------------------------------
TEST_CASE("LB_Keogh with wider band <= LB_Keogh with narrow band",
          "[adversarial][lb_keogh][monotonicity]")
{
  std::mt19937 rng(12345);

  for (int p = 0; p < 20; ++p) {
    const auto n = random_length(rng, 30, 100);
    auto x = random_series(rng, n);
    auto y = random_series(rng, n);

    const data_t lb_narrow = compute_lb_keogh(x, y, 3);
    const data_t lb_wide = compute_lb_keogh(x, y, 15);

    INFO("Pair " << p << ": lb_narrow=" << lb_narrow << " lb_wide=" << lb_wide);
    REQUIRE(lb_wide <= lb_narrow + 1e-10);
  }
}

// -------------------------------------------------------------------------
// 2i. LB_Keogh with band=0 and distinct series > 0
// -------------------------------------------------------------------------
TEST_CASE("LB_Keogh with band=0 for distinct series is positive",
          "[adversarial][lb_keogh][band_zero]")
{
  std::mt19937 rng(12345);

  for (int p = 0; p < 20; ++p) {
    const auto n = random_length(rng, 20, 80);
    auto x = random_series(rng, n);
    auto y = random_series(rng, n);

    // Guarantee at least one difference
    x[0] = y[0] + 1.0;

    const data_t lb = compute_lb_keogh(x, y, 0);

    INFO("Pair " << p << ": lb=" << lb);
    REQUIRE(lb > 0.0);
  }
}

// -------------------------------------------------------------------------
// 2j. Stress test: extreme values (very large, very small, near zero)
// -------------------------------------------------------------------------
TEST_CASE("LB_Kim and LB_Keogh handle extreme values correctly",
          "[adversarial][extreme][contract]")
{
  // Series with very large values
  const std::vector<data_t> big = { 1e15, -1e15, 1e15, -1e15, 1e15 };
  const std::vector<data_t> small_vals = { 1e-15, -1e-15, 1e-15, -1e-15, 1e-15 };

  // LB_Kim: must be non-negative and <= DTW
  {
    const data_t lb = dtwc::core::lb_kim(big, small_vals);
    const data_t dtw = dtwc::dtwFull_L<data_t>(big, small_vals);
    REQUIRE(lb >= 0.0);
    REQUIRE(lb <= dtw + 1e-5);
  }

  // LB_Keogh: must be non-negative and <= DTW
  {
    const data_t lb = compute_lb_keogh(big, small_vals, 2);
    const data_t dtw = dtwc::dtwBanded<data_t>(big, small_vals, 2);
    REQUIRE(lb >= 0.0);
    REQUIRE(lb <= dtw + 1e-5);
  }

  // All-zero series
  const std::vector<data_t> zeros(50, 0.0);
  {
    const data_t lb_k = dtwc::core::lb_kim(zeros, zeros);
    REQUIRE_THAT(lb_k, WithinAbs(0.0, 1e-15));

    const data_t lb_keogh = compute_lb_keogh(zeros, zeros, 5);
    REQUIRE_THAT(lb_keogh, WithinAbs(0.0, 1e-15));
  }

  // Constant non-zero series vs. different constant series
  const std::vector<data_t> const_a(30, 42.0);
  const std::vector<data_t> const_b(30, 100.0);
  {
    const data_t lb = dtwc::core::lb_kim(const_a, const_b);
    const data_t dtw = dtwc::dtwFull_L<data_t>(const_a, const_b);
    REQUIRE(lb >= 0.0);
    REQUIRE(lb <= dtw + 1e-10);
    // For constant series: lb_kim = |42-100| = 58
    REQUIRE_THAT(lb, WithinAbs(58.0, 1e-10));
  }
}

// -------------------------------------------------------------------------
// 2k. LB_Keogh with banded DTW comparison (the correct guarantee)
//     LB_Keogh(band=b) <= DTW_banded(band=b) is the formal guarantee.
// -------------------------------------------------------------------------
TEST_CASE("LB_Keogh <= banded DTW with matching band",
          "[adversarial][lb_keogh][banded_dtw][contract]")
{
  std::mt19937 rng(12345);
  constexpr int band = 5;

  for (int p = 0; p < 30; ++p) {
    const auto n = random_length(rng, 30, 100);
    auto x = random_series(rng, n);
    auto y = random_series(rng, n);

    const data_t lb = compute_lb_keogh(x, y, band);
    const data_t dtw_banded = dtwc::dtwBanded<data_t>(x, y, band);

    INFO("Pair " << p << ": LB_Keogh(band=5)=" << lb
         << " DTW_banded(band=5)=" << dtw_banded);
    REQUIRE(lb <= dtw_banded + 1e-10);
  }
}

// =========================================================================
//  AREA 3: Envelope O(n) vs naive O(n*w) cross-validation
//  Cross-checks the Lemire sliding-window implementation against a reference
//  brute-force implementation for multiple (n, w) combinations including
//  critical edge cases: n=1, n=2, w=0, w>=n.
// =========================================================================

/// Naive O(n*w) reference implementation of compute_envelopes.
static void naive_envelopes(const std::vector<data_t> &series, int band,
                            std::vector<data_t> &upper_out,
                            std::vector<data_t> &lower_out)
{
  const std::size_t n = series.size();
  upper_out.resize(n);
  lower_out.resize(n);
  const std::size_t w = static_cast<std::size_t>(std::max(band, 0));
  for (std::size_t p = 0; p < n; ++p) {
    const std::size_t lo = (p > w) ? p - w : 0;
    const std::size_t hi = std::min(p + w + 1, n);  // exclusive
    upper_out[p] = *std::max_element(series.begin() + lo, series.begin() + hi);
    lower_out[p] = *std::min_element(series.begin() + lo, series.begin() + hi);
  }
}

// -------------------------------------------------------------------------
// 3a. Cross-validate O(n) vs naive for edge-case lengths {1, 2, 10, 100}
//     and band sizes {0, 1, 5, 50} with deterministic random series.
// -------------------------------------------------------------------------
TEST_CASE("compute_envelopes O(n) matches naive O(n*w) reference — edge-case sizes",
          "[adversarial][envelope][cross_validate][edge_cases]")
{
  std::mt19937 rng(99991);
  const std::vector<std::size_t> lengths = { 1, 2, 10, 100 };
  const std::vector<int> bands = { 0, 1, 5, 50 };

  for (std::size_t n : lengths) {
    auto series = random_series(rng, n, -20.0, 20.0);

    for (int band : bands) {
      std::vector<data_t> upper_naive, lower_naive;
      naive_envelopes(series, band, upper_naive, lower_naive);

      std::vector<data_t> upper_fast(n), lower_fast(n);
      dtwc::core::compute_envelopes(series, band, upper_fast, lower_fast);

      for (std::size_t i = 0; i < n; ++i) {
        INFO("n=" << n << " band=" << band << " i=" << i
             << " naive_upper=" << upper_naive[i] << " fast_upper=" << upper_fast[i]
             << " naive_lower=" << lower_naive[i] << " fast_lower=" << lower_fast[i]);
        REQUIRE_THAT(upper_fast[i], WithinAbs(upper_naive[i], 1e-12));
        REQUIRE_THAT(lower_fast[i], WithinAbs(lower_naive[i], 1e-12));
      }
    }
  }
}

// -------------------------------------------------------------------------
// 3b. w=0: upper[i] == lower[i] == series[i] for all lengths
// -------------------------------------------------------------------------
TEST_CASE("compute_envelopes w=0 is identity for n=1,2,10,100",
          "[adversarial][envelope][cross_validate][w_zero]")
{
  std::mt19937 rng(77777);
  for (std::size_t n : { 1u, 2u, 10u, 100u }) {
    auto series = random_series(rng, n);
    std::vector<data_t> upper(n), lower(n);
    dtwc::core::compute_envelopes(series, 0, upper, lower);

    for (std::size_t i = 0; i < n; ++i) {
      INFO("n=" << n << " i=" << i << " series=" << series[i]
           << " upper=" << upper[i] << " lower=" << lower[i]);
      REQUIRE_THAT(upper[i], WithinAbs(series[i], 1e-15));
      REQUIRE_THAT(lower[i], WithinAbs(series[i], 1e-15));
    }
  }
}

// -------------------------------------------------------------------------
// 3c. w >= n (fast-path): envelope is global min/max for all positions
// -------------------------------------------------------------------------
TEST_CASE("compute_envelopes w>=n fast-path gives global min/max for n=1,2,10",
          "[adversarial][envelope][cross_validate][fast_path]")
{
  std::mt19937 rng(11111);
  for (std::size_t n : { 1u, 2u, 10u }) {
    auto series = random_series(rng, n, -50.0, 50.0);
    const data_t gmax = *std::max_element(series.begin(), series.end());
    const data_t gmin = *std::min_element(series.begin(), series.end());

    // band = n (exactly equal) and band = n+5 (strictly greater)
    for (int band : { static_cast<int>(n), static_cast<int>(n + 5) }) {
      std::vector<data_t> upper(n), lower(n);
      dtwc::core::compute_envelopes(series, band, upper, lower);

      for (std::size_t i = 0; i < n; ++i) {
        INFO("n=" << n << " band=" << band << " i=" << i);
        REQUIRE_THAT(upper[i], WithinAbs(gmax, 1e-15));
        REQUIRE_THAT(lower[i], WithinAbs(gmin, 1e-15));
      }
    }
  }
}

// -------------------------------------------------------------------------
// 3d. Ring-buffer capacity invariant: with cap=w+2, at most w+1 elements
//     should be live in the deque simultaneously (no aliasing).
//     This is verified indirectly: if the ring buffer overflows it would
//     corrupt an index that is still live, producing wrong envelope values.
//     Cross-validation against the naive reference catches such corruption.
// -------------------------------------------------------------------------
TEST_CASE("compute_envelopes ring-buffer no-aliasing: 50 random series, bands 0-50",
          "[adversarial][envelope][cross_validate][ring_buffer]")
{
  std::mt19937 rng(31415);

  for (int trial = 0; trial < 50; ++trial) {
    const std::size_t n = 20 + (rng() % 80);  // lengths 20..99
    const int band = static_cast<int>(rng() % 51);  // bands 0..50
    auto series = random_series(rng, n);

    std::vector<data_t> upper_naive, lower_naive;
    naive_envelopes(series, band, upper_naive, lower_naive);

    std::vector<data_t> upper_fast(n), lower_fast(n);
    dtwc::core::compute_envelopes(series, band, upper_fast, lower_fast);

    for (std::size_t i = 0; i < n; ++i) {
      INFO("trial=" << trial << " n=" << n << " band=" << band << " i=" << i);
      REQUIRE_THAT(upper_fast[i], WithinAbs(upper_naive[i], 1e-12));
      REQUIRE_THAT(lower_fast[i], WithinAbs(lower_naive[i], 1e-12));
    }
  }
}

// -------------------------------------------------------------------------
// 3e. n=2, various bands: minimal non-trivial case
// -------------------------------------------------------------------------
TEST_CASE("compute_envelopes n=2 all band values are correct",
          "[adversarial][envelope][cross_validate][n2]")
{
  // For n=2: with band=0, upper={s[0],s[1]}, lower={s[0],s[1]}
  //          with band>=1, upper={max,max},  lower={min,min}
  const std::vector<data_t> series = { 3.0, 7.0 };
  std::vector<data_t> upper(2), lower(2);

  // band=0
  dtwc::core::compute_envelopes(series, 0, upper, lower);
  REQUIRE_THAT(upper[0], WithinAbs(3.0, 1e-15));
  REQUIRE_THAT(upper[1], WithinAbs(7.0, 1e-15));
  REQUIRE_THAT(lower[0], WithinAbs(3.0, 1e-15));
  REQUIRE_THAT(lower[1], WithinAbs(7.0, 1e-15));

  // band=1 (w=1 >= n-1=1, but n=2 and w=1, w < n=2 so NOT fast-path)
  // window for p=0: [0,1] -> max=7, min=3
  // window for p=1: [0,1] -> max=7, min=3
  dtwc::core::compute_envelopes(series, 1, upper, lower);
  REQUIRE_THAT(upper[0], WithinAbs(7.0, 1e-15));
  REQUIRE_THAT(upper[1], WithinAbs(7.0, 1e-15));
  REQUIRE_THAT(lower[0], WithinAbs(3.0, 1e-15));
  REQUIRE_THAT(lower[1], WithinAbs(3.0, 1e-15));

  // band=2 (w=2 >= n=2: fast-path)
  dtwc::core::compute_envelopes(series, 2, upper, lower);
  REQUIRE_THAT(upper[0], WithinAbs(7.0, 1e-15));
  REQUIRE_THAT(upper[1], WithinAbs(7.0, 1e-15));
  REQUIRE_THAT(lower[0], WithinAbs(3.0, 1e-15));
  REQUIRE_THAT(lower[1], WithinAbs(3.0, 1e-15));
}

// -------------------------------------------------------------------------
// 3f. n=1: trivial case — upper[0] == lower[0] == series[0] for all bands
// -------------------------------------------------------------------------
TEST_CASE("compute_envelopes n=1 gives series value regardless of band",
          "[adversarial][envelope][cross_validate][n1]")
{
  const std::vector<data_t> series = { 42.5 };
  std::vector<data_t> upper(1), lower(1);

  for (int band : { 0, 1, 5, 100 }) {
    dtwc::core::compute_envelopes(series, band, upper, lower);
    INFO("band=" << band);
    REQUIRE_THAT(upper[0], WithinAbs(42.5, 1e-15));
    REQUIRE_THAT(lower[0], WithinAbs(42.5, 1e-15));
  }
}
