/**
 * @file test_arow_adversarial.cpp
 * @brief Adversarial stress tests for the DTW-AROW implementation.
 *
 * @details Tests are derived from the mathematical specification of DTW-AROW,
 *          NOT from reading the implementation. They attempt to find bugs by
 *          exercising pathological NaN patterns, random bulk comparisons,
 *          and cross-variant consistency checks.
 *
 * Variants tested:
 *   dtwAROW_L   — linear-space rolling buffer (O(min(m,n)) memory)
 *   dtwAROW     — full cost matrix (O(m*n) memory, same result)
 *   dtwAROW_banded — Sakoe-Chiba band restriction
 *
 * Reference: Yurtman, A., Soenen, J., Meert, W. & Blockeel, H. (2023).
 *            "Estimating DTW Distance Between Time Series with Missing Data."
 *            ECML-PKDD 2023, LNCS 14173.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#include <dtwc.hpp>
#include <warping_missing_arow.hpp>
#include <warping_missing.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <sstream>

using namespace dtwc;
using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// Constants and helpers
// ---------------------------------------------------------------------------

static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
static constexpr double INF = std::numeric_limits<double>::max();
static constexpr double EPS = 1e-10;

/// Generate a random series of given length, values in [lo, hi].
static std::vector<double> random_series(std::mt19937 &rng, size_t len,
                                         double lo = -5.0, double hi = 5.0)
{
  std::uniform_real_distribution<double> dist(lo, hi);
  std::vector<double> v(len);
  for (auto &x : v) x = dist(rng);
  return v;
}

/// Inject NaN values into a series at positions chosen by a Bernoulli trial.
/// p_nan = probability each element is NaN.
static std::vector<double> inject_nan(std::mt19937 &rng,
                                      const std::vector<double> &v,
                                      double p_nan)
{
  std::bernoulli_distribution coin(p_nan);
  std::vector<double> out = v;
  for (auto &x : out)
    if (coin(rng)) x = NaN;
  return out;
}

/// True iff a value is finite and non-negative (i.e., a valid distance).
static bool is_valid_distance(double d)
{
  return std::isfinite(d) && d >= 0.0;
}

/// Format a vector for test diagnostics (truncated to first 10 elements).
static std::string fmt(const std::vector<double> &v)
{
  std::ostringstream ss;
  ss << "[";
  for (size_t i = 0; i < v.size() && i < 10; ++i) {
    if (i) ss << ", ";
    if (std::isnan(v[i])) ss << "NaN";
    else ss << v[i];
  }
  if (v.size() > 10) ss << ", ...(" << v.size() - 10 << " more)";
  ss << "]";
  return ss.str();
}

// ===========================================================================
// 1. AROW >= ZeroCost mathematical invariant (bulk random test)
// ===========================================================================

TEST_CASE("AROW >= ZeroCost: 200 random pairs with random NaN patterns",
          "[arow_adversarial][invariant]")
{
  // AROW restricts missing cells to diagonal-only; ZeroCost allows any path direction.
  // Therefore AROW must always produce distance >= ZeroCost distance.
  std::mt19937 rng(0xDEADBEEF);
  std::uniform_int_distribution<size_t> len_dist(2, 20);
  std::uniform_real_distribution<double> nan_rate_dist(0.05, 0.70);

  int violations = 0;
  for (int trial = 0; trial < 200; ++trial) {
    const size_t nx = len_dist(rng);
    const size_t ny = len_dist(rng);
    const double p_nan = nan_rate_dist(rng);

    auto x_raw = random_series(rng, nx);
    auto y_raw = random_series(rng, ny);
    auto x = inject_nan(rng, x_raw, p_nan);
    auto y = inject_nan(rng, y_raw, p_nan);

    const double arow      = dtwAROW_L<double>(x, y);
    const double zero_cost = dtwMissing_L<double>(x, y);

    // Both must be valid distances
    INFO("Trial " << trial << ": x=" << fmt(x) << " y=" << fmt(y));
    REQUIRE(is_valid_distance(arow));
    REQUIRE(is_valid_distance(zero_cost));

    // AROW >= ZeroCost (allow tiny floating-point slack)
    if (arow < zero_cost - 1e-9) {
      ++violations;
      FAIL_CHECK("AROW < ZeroCost: arow=" << arow << " zero_cost=" << zero_cost
                 << " diff=" << (zero_cost - arow)
                 << "\n  x=" << fmt(x) << "\n  y=" << fmt(y));
    }
  }
  REQUIRE(violations == 0);
}

// ===========================================================================
// 2. Consistency: dtwAROW_L == dtwAROW (linear-space == full-matrix)
//    Tested on pathological NaN patterns
// ===========================================================================

TEST_CASE("Linear-space == full-matrix: all NaN except first and last",
          "[arow_adversarial][consistency]")
{
  // The interior is completely missing — only the endpoints carry information.
  const int N = 8;
  std::vector<double> x(N, NaN), y(N, NaN);
  x[0] = 1.0; x[N - 1] = 10.0;
  y[0] = 2.0; y[N - 1] =  8.0;

  const double linear = dtwAROW_L<double>(x, y);
  const double full   = dtwAROW<double>(x, y);

  INFO("x=" << fmt(x) << " y=" << fmt(y));
  REQUIRE(is_valid_distance(linear));
  REQUIRE(is_valid_distance(full));
  REQUIRE_THAT(linear, WithinAbs(full, EPS));
}

TEST_CASE("Linear-space == full-matrix: alternating NaN/value (equal length)",
          "[arow_adversarial][consistency]")
{
  // Even indices observed, odd indices missing.
  std::vector<double> x = {1.0, NaN, 3.0, NaN, 5.0, NaN, 7.0, NaN};
  std::vector<double> y = {NaN, 2.0, NaN, 4.0, NaN, 6.0, NaN, 8.0};

  const double linear = dtwAROW_L<double>(x, y);
  const double full   = dtwAROW<double>(x, y);

  INFO("x=" << fmt(x) << " y=" << fmt(y));
  REQUIRE(is_valid_distance(linear));
  REQUIRE(is_valid_distance(full));
  REQUIRE_THAT(linear, WithinAbs(full, EPS));
}

TEST_CASE("Linear-space == full-matrix: alternating NaN/value (unequal length)",
          "[arow_adversarial][consistency]")
{
  std::vector<double> x = {1.0, NaN, 3.0, NaN, 5.0};
  std::vector<double> y = {NaN, 2.0, NaN, 4.0, NaN, 6.0, NaN, 8.0};

  const double linear = dtwAROW_L<double>(x, y);
  const double full   = dtwAROW<double>(x, y);

  INFO("x=" << fmt(x) << " y=" << fmt(y));
  REQUIRE(is_valid_distance(linear));
  REQUIRE(is_valid_distance(full));
  REQUIRE_THAT(linear, WithinAbs(full, EPS));
}

TEST_CASE("Linear-space == full-matrix: NaN block at start",
          "[arow_adversarial][consistency]")
{
  std::vector<double> x = {NaN, NaN, NaN, 4.0, 5.0, 6.0};
  std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  const double linear = dtwAROW_L<double>(x, y);
  const double full   = dtwAROW<double>(x, y);

  INFO("x=" << fmt(x) << " y=" << fmt(y));
  REQUIRE(is_valid_distance(linear));
  REQUIRE(is_valid_distance(full));
  REQUIRE_THAT(linear, WithinAbs(full, EPS));
}

TEST_CASE("Linear-space == full-matrix: NaN block in middle",
          "[arow_adversarial][consistency]")
{
  std::vector<double> x = {1.0, 2.0, NaN, NaN, NaN, 6.0, 7.0};
  std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

  const double linear = dtwAROW_L<double>(x, y);
  const double full   = dtwAROW<double>(x, y);

  INFO("x=" << fmt(x) << " y=" << fmt(y));
  REQUIRE(is_valid_distance(linear));
  REQUIRE(is_valid_distance(full));
  REQUIRE_THAT(linear, WithinAbs(full, EPS));
}

TEST_CASE("Linear-space == full-matrix: NaN block at end",
          "[arow_adversarial][consistency]")
{
  std::vector<double> x = {1.0, 2.0, 3.0, NaN, NaN, NaN};
  std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};

  const double linear = dtwAROW_L<double>(x, y);
  const double full   = dtwAROW<double>(x, y);

  INFO("x=" << fmt(x) << " y=" << fmt(y));
  REQUIRE(is_valid_distance(linear));
  REQUIRE(is_valid_distance(full));
  REQUIRE_THAT(linear, WithinAbs(full, EPS));
}

TEST_CASE("Linear-space == full-matrix: NaN blocks in both series",
          "[arow_adversarial][consistency]")
{
  // NaN at start of x, NaN at end of y — opposite missing regions.
  std::vector<double> x = {NaN, NaN, 3.0, 4.0, 5.0, 6.0};
  std::vector<double> y = {1.0, 2.0, 3.0, 4.0, NaN, NaN};

  const double linear = dtwAROW_L<double>(x, y);
  const double full   = dtwAROW<double>(x, y);

  INFO("x=" << fmt(x) << " y=" << fmt(y));
  REQUIRE(is_valid_distance(linear));
  REQUIRE(is_valid_distance(full));
  REQUIRE_THAT(linear, WithinAbs(full, EPS));
}

TEST_CASE("Linear-space == full-matrix: one series all NaN, other all observed",
          "[arow_adversarial][consistency]")
{
  std::vector<double> x = {NaN, NaN, NaN, NaN, NaN};
  std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0};

  const double linear = dtwAROW_L<double>(x, y);
  const double full   = dtwAROW<double>(x, y);

  INFO("x=" << fmt(x) << " y=" << fmt(y));
  REQUIRE(is_valid_distance(linear));
  REQUIRE(is_valid_distance(full));
  REQUIRE_THAT(linear, WithinAbs(full, EPS));
  // When one series is all NaN, every cell is diagonal-only with zero cost -> result = 0
  REQUIRE_THAT(linear, WithinAbs(0.0, EPS));
}

TEST_CASE("Linear-space == full-matrix: 100 random pairs, bulk",
          "[arow_adversarial][consistency]")
{
  std::mt19937 rng(0xCAFEBABE);
  std::uniform_int_distribution<size_t> len_dist(2, 30);
  std::uniform_real_distribution<double> nan_rate_dist(0.0, 0.8);

  int mismatches = 0;
  for (int trial = 0; trial < 100; ++trial) {
    const size_t nx = len_dist(rng);
    const size_t ny = len_dist(rng);
    const double p  = nan_rate_dist(rng);

    auto xraw = random_series(rng, nx);
    auto yraw = random_series(rng, ny);
    auto x = inject_nan(rng, xraw, p);
    auto y = inject_nan(rng, yraw, p);

    const double linear = dtwAROW_L<double>(x, y);
    const double full   = dtwAROW<double>(x, y);

    INFO("Trial " << trial << ": nx=" << nx << " ny=" << ny << " p=" << p);
    REQUIRE(is_valid_distance(linear));
    REQUIRE(is_valid_distance(full));

    if (std::abs(linear - full) > EPS) {
      ++mismatches;
      FAIL_CHECK("Linear/full mismatch: linear=" << linear << " full=" << full
                 << " diff=" << std::abs(linear - full)
                 << "\n  x=" << fmt(x) << "\n  y=" << fmt(y));
    }
  }
  REQUIRE(mismatches == 0);
}

// ===========================================================================
// 3. Banded consistency: banded with large band matches unbanded
// ===========================================================================

TEST_CASE("Banded == unbanded for short series (band >= length)",
          "[arow_adversarial][banded]")
{
  // For series of length <= 8, band=100 is effectively unbounded.
  std::mt19937 rng(0xBADF00D);
  std::uniform_int_distribution<size_t> len_dist(2, 8);

  int mismatches = 0;
  for (int trial = 0; trial < 50; ++trial) {
    const size_t nx = len_dist(rng);
    const size_t ny = len_dist(rng);

    auto xraw = random_series(rng, nx);
    auto yraw = random_series(rng, ny);
    // Inject about 30% NaN
    auto x = inject_nan(rng, xraw, 0.30);
    auto y = inject_nan(rng, yraw, 0.30);

    const double unbanded = dtwAROW_L<double>(x, y);
    const double banded   = dtwAROW_banded<double>(x, y, 100);

    INFO("Trial " << trial << ": nx=" << nx << " ny=" << ny);
    REQUIRE(is_valid_distance(unbanded));
    REQUIRE(is_valid_distance(banded));

    if (std::abs(banded - unbanded) > EPS) {
      ++mismatches;
      FAIL_CHECK("Banded/unbanded mismatch: banded=" << banded
                 << " unbanded=" << unbanded
                 << "\n  x=" << fmt(x) << "\n  y=" << fmt(y));
    }
  }
  REQUIRE(mismatches == 0);
}

TEST_CASE("Banded == unbanded: NaN at both ends, band >> length",
          "[arow_adversarial][banded]")
{
  std::vector<double> x = {NaN, 2.0, 3.0, NaN};
  std::vector<double> y = {NaN, 4.0, 5.0, NaN};

  const double unbanded = dtwAROW_L<double>(x, y);
  const double banded   = dtwAROW_banded<double>(x, y, 100);

  INFO("x=" << fmt(x) << " y=" << fmt(y));
  REQUIRE(is_valid_distance(unbanded));
  REQUIRE(is_valid_distance(banded));
  REQUIRE_THAT(banded, WithinAbs(unbanded, EPS));
}

// ===========================================================================
// 4. Boundary torture tests
// ===========================================================================

TEST_CASE("Boundary: both x[0] and y[0] are NaN",
          "[arow_adversarial][boundary]")
{
  // C(0,0) must be 0 (not +inf), subsequent cells must remain finite.
  std::vector<double> x = {NaN, 2.0, 3.0};
  std::vector<double> y = {NaN, 5.0, 6.0};

  const double linear = dtwAROW_L<double>(x, y);
  const double full   = dtwAROW<double>(x, y);

  INFO("x=" << fmt(x) << " y=" << fmt(y));
  REQUIRE(is_valid_distance(linear));
  REQUIRE(is_valid_distance(full));
  REQUIRE_THAT(linear, WithinAbs(full, EPS));
}

TEST_CASE("Boundary: entire first half of x is NaN, entire second half of y is NaN",
          "[arow_adversarial][boundary]")
{
  // Observation windows do not overlap: x observed only in second half,
  // y observed only in first half. The AROW diagonal constraint means
  // costs must propagate across the missing region.
  std::vector<double> x = {NaN, NaN, NaN, NaN, 3.0, 4.0, 5.0, 6.0};
  std::vector<double> y = {1.0, 2.0, 3.0, 4.0, NaN, NaN, NaN, NaN};

  const double linear = dtwAROW_L<double>(x, y);
  const double full   = dtwAROW<double>(x, y);

  INFO("x=" << fmt(x) << " y=" << fmt(y));
  // Result must be finite (not +inf) and non-negative
  REQUIRE(is_valid_distance(linear));
  REQUIRE(is_valid_distance(full));
  REQUIRE_THAT(linear, WithinAbs(full, EPS));
}

TEST_CASE("Boundary: series of length 1 where the single value is NaN",
          "[arow_adversarial][boundary]")
{
  std::vector<double> x = {NaN};
  std::vector<double> y = {5.0};

  const double linear = dtwAROW_L<double>(x, y);
  const double full   = dtwAROW<double>(x, y);

  INFO("x={NaN} y={5}");
  REQUIRE(is_valid_distance(linear));
  REQUIRE(is_valid_distance(full));
  // NaN at C(0,0) -> cost = 0
  REQUIRE_THAT(linear, WithinAbs(0.0, EPS));
  REQUIRE_THAT(full, WithinAbs(0.0, EPS));
}

TEST_CASE("Boundary: both length-1 series are NaN",
          "[arow_adversarial][boundary]")
{
  std::vector<double> x = {NaN};
  std::vector<double> y = {NaN};

  const double linear = dtwAROW_L<double>(x, y);
  const double full   = dtwAROW<double>(x, y);

  INFO("x={NaN} y={NaN}");
  REQUIRE(is_valid_distance(linear));
  REQUIRE(is_valid_distance(full));
  REQUIRE_THAT(linear, WithinAbs(0.0, EPS));
  REQUIRE_THAT(full, WithinAbs(0.0, EPS));
}

TEST_CASE("Boundary: series of length 2 where both values are NaN",
          "[arow_adversarial][boundary]")
{
  std::vector<double> x = {NaN, NaN};
  std::vector<double> y = {NaN, NaN};

  const double linear = dtwAROW_L<double>(x, y);
  const double full   = dtwAROW<double>(x, y);

  INFO("x={NaN,NaN} y={NaN,NaN}");
  REQUIRE(is_valid_distance(linear));
  REQUIRE(is_valid_distance(full));
  REQUIRE_THAT(linear, WithinAbs(0.0, EPS));
  REQUIRE_THAT(full, WithinAbs(0.0, EPS));
}

TEST_CASE("Boundary: length-2 series, one all-NaN, one observed",
          "[arow_adversarial][boundary]")
{
  std::vector<double> x = {NaN, NaN};
  std::vector<double> y = {3.0, 7.0};

  const double linear = dtwAROW_L<double>(x, y);
  const double full   = dtwAROW<double>(x, y);

  INFO("x={NaN,NaN} y={3,7}");
  REQUIRE(is_valid_distance(linear));
  REQUIRE(is_valid_distance(full));
  REQUIRE_THAT(linear, WithinAbs(full, EPS));
  // x is all NaN => result should be 0
  REQUIRE_THAT(linear, WithinAbs(0.0, EPS));
}

TEST_CASE("Boundary: NaN at C(0,0) cascades — first column and row stay finite",
          "[arow_adversarial][boundary]")
{
  // If the implementation naively set C(0,0) = +inf for missing,
  // the first column and row would cascade to +inf and the final result would be +inf.
  std::vector<double> x = {NaN, 1.0, 2.0, 3.0};
  std::vector<double> y = {NaN, 4.0, 5.0, 6.0};

  const double linear = dtwAROW_L<double>(x, y);
  const double full   = dtwAROW<double>(x, y);

  INFO("x=" << fmt(x) << " y=" << fmt(y));
  REQUIRE(is_valid_distance(linear));
  REQUIRE(is_valid_distance(full));
  REQUIRE_THAT(linear, WithinAbs(full, EPS));
}

// ===========================================================================
// 5. Asymmetric lengths with NaN
// ===========================================================================

TEST_CASE("Asymmetric: x length 3, y length 7, NaN only in x",
          "[arow_adversarial][asymmetric]")
{
  std::vector<double> x = {NaN, 2.0, NaN};
  std::vector<double> y = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

  const double linear = dtwAROW_L<double>(x, y);
  const double full   = dtwAROW<double>(x, y);

  INFO("x=" << fmt(x) << " y=" << fmt(y));
  REQUIRE(is_valid_distance(linear));
  REQUIRE(is_valid_distance(full));
  REQUIRE_THAT(linear, WithinAbs(full, EPS));
}

TEST_CASE("Asymmetric: x length 3, y length 7, NaN only in y",
          "[arow_adversarial][asymmetric]")
{
  std::vector<double> x = {1.0, 2.0, 3.0};
  std::vector<double> y = {NaN, 2.0, NaN, 4.0, NaN, 6.0, NaN};

  const double linear = dtwAROW_L<double>(x, y);
  const double full   = dtwAROW<double>(x, y);

  INFO("x=" << fmt(x) << " y=" << fmt(y));
  REQUIRE(is_valid_distance(linear));
  REQUIRE(is_valid_distance(full));
  REQUIRE_THAT(linear, WithinAbs(full, EPS));
}

TEST_CASE("Asymmetric: x length 3, y length 7, NaN in both",
          "[arow_adversarial][asymmetric]")
{
  std::vector<double> x = {1.0, NaN, 3.0};
  std::vector<double> y = {NaN, 2.0, NaN, 4.0, 5.0, NaN, 7.0};

  const double linear = dtwAROW_L<double>(x, y);
  const double full   = dtwAROW<double>(x, y);

  INFO("x=" << fmt(x) << " y=" << fmt(y));
  REQUIRE(is_valid_distance(linear));
  REQUIRE(is_valid_distance(full));
  REQUIRE_THAT(linear, WithinAbs(full, EPS));
}

TEST_CASE("Asymmetric: symmetry — dtwAROW(x,y) == dtwAROW(y,x) when lengths differ",
          "[arow_adversarial][asymmetric]")
{
  // The linear-space version swaps x/y when nx > ny to reduce buffer size.
  // The result must be the same regardless of argument order.
  std::mt19937 rng(0xFEEDFACE);
  std::uniform_int_distribution<size_t> len_dist(2, 15);

  int asymmetries = 0;
  for (int trial = 0; trial < 50; ++trial) {
    const size_t nx = len_dist(rng);
    const size_t ny = len_dist(rng);

    auto xraw = random_series(rng, nx);
    auto yraw = random_series(rng, ny);
    auto x = inject_nan(rng, xraw, 0.25);
    auto y = inject_nan(rng, yraw, 0.25);

    const double d_xy = dtwAROW_L<double>(x, y);
    const double d_yx = dtwAROW_L<double>(y, x);

    INFO("Trial " << trial << ": nx=" << nx << " ny=" << ny);
    REQUIRE(is_valid_distance(d_xy));
    REQUIRE(is_valid_distance(d_yx));

    if (std::abs(d_xy - d_yx) > EPS) {
      ++asymmetries;
      FAIL_CHECK("Symmetry violation: d(x,y)=" << d_xy << " d(y,x)=" << d_yx
                 << "\n  x=" << fmt(x) << "\n  y=" << fmt(y));
    }
  }
  REQUIRE(asymmetries == 0);
}

// ===========================================================================
// 6. Very long series: no crash, no NaN output
// ===========================================================================

TEST_CASE("Long series: length 500, 50% random NaN — does not crash or produce NaN",
          "[arow_adversarial][robustness]")
{
  std::mt19937 rng(0x12345678);
  auto xraw = random_series(rng, 500);
  auto yraw = random_series(rng, 500);
  auto x = inject_nan(rng, xraw, 0.50);
  auto y = inject_nan(rng, yraw, 0.50);

  const double linear = dtwAROW_L<double>(x, y);
  const double full   = dtwAROW<double>(x, y);

  REQUIRE(is_valid_distance(linear));
  REQUIRE(is_valid_distance(full));
  REQUIRE_THAT(linear, WithinAbs(full, EPS));
}

TEST_CASE("Long series: length 1000, 70% NaN — no crash or NaN",
          "[arow_adversarial][robustness]")
{
  std::mt19937 rng(0xABCDEF01);
  auto xraw = random_series(rng, 1000);
  auto yraw = random_series(rng, 1000);
  auto x = inject_nan(rng, xraw, 0.70);
  auto y = inject_nan(rng, yraw, 0.70);

  const double linear = dtwAROW_L<double>(x, y);

  REQUIRE(is_valid_distance(linear));
}

TEST_CASE("Long series: length 700 vs 300, mixed NaN — no crash",
          "[arow_adversarial][robustness]")
{
  std::mt19937 rng(0x55AA55AA);
  auto xraw = random_series(rng, 700);
  auto yraw = random_series(rng, 300);
  auto x = inject_nan(rng, xraw, 0.40);
  auto y = inject_nan(rng, yraw, 0.40);

  const double linear = dtwAROW_L<double>(x, y);
  const double full   = dtwAROW<double>(x, y);

  REQUIRE(is_valid_distance(linear));
  REQUIRE(is_valid_distance(full));
  REQUIRE_THAT(linear, WithinAbs(full, EPS));
}

// ===========================================================================
// 7. Triangle inequality: DTW is NOT a metric — AROW also shows violations.
//    This is EXPECTED behavior. We just document and verify the counterexample exists.
// ===========================================================================

TEST_CASE("Triangle inequality: AROW (like standard DTW) can violate d(a,c) <= d(a,b) + d(b,c)",
          "[arow_adversarial][triangle_inequality]")
{
  // Classic DTW triangle inequality counterexample adapted for AROW.
  // We find a triplet where d(a,c) > d(a,b) + d(b,c).
  // This is NOT a bug — DTW is not a metric.
  // We use a brute-force search over a small parameter space.

  std::mt19937 rng(0x99887766);
  std::uniform_int_distribution<size_t> len_dist(3, 10);

  bool found_counterexample = false;

  for (int trial = 0; trial < 500 && !found_counterexample; ++trial) {
    const size_t na = len_dist(rng);
    const size_t nb = len_dist(rng);
    const size_t nc = len_dist(rng);

    auto a = random_series(rng, na, -10.0, 10.0);
    auto b = random_series(rng, nb, -10.0, 10.0);
    auto c = random_series(rng, nc, -10.0, 10.0);

    const double dab = dtwAROW_L<double>(a, b);
    const double dbc = dtwAROW_L<double>(b, c);
    const double dac = dtwAROW_L<double>(a, c);

    if (!is_valid_distance(dab) || !is_valid_distance(dbc) || !is_valid_distance(dac))
      continue;

    if (dac > dab + dbc + 1e-9) {
      found_counterexample = true;
      INFO("Triangle inequality violation found (EXPECTED — DTW is not a metric):");
      INFO("  d(a,c)=" << dac << " d(a,b)=" << dab << " d(b,c)=" << dbc);
      INFO("  violation by: " << (dac - dab - dbc));
      // This is documented expected behavior, NOT a test failure.
      // We just record that the counterexample was found.
    }
  }

  // We expect to find at least one violation — if not, it's suspicious.
  // Mark as a warning (not a failure) if no violation found in 500 trials.
  if (!found_counterexample) {
    WARN("No triangle inequality violation found in 500 trials — "
         "this is unusual but may happen with this RNG seed.");
  }
  // No REQUIRE here — this test always passes; it just documents behavior.
}

// ===========================================================================
// 8. NaN result invariant: output is NEVER NaN — always finite and non-negative
// ===========================================================================

TEST_CASE("Output is never NaN: systematic NaN pattern sweep",
          "[arow_adversarial][nan_output]")
{
  // Sweep all possible NaN subsets for a small length-4 series.
  // 2^4 = 16 patterns per series, so 16*16 = 256 pairs total.
  const int N = 4;
  const double vals[4] = {1.0, 2.0, 3.0, 4.0};

  int nan_outputs = 0;

  for (int px = 0; px < (1 << N); ++px) {
    std::vector<double> x(N);
    for (int i = 0; i < N; ++i)
      x[i] = (px >> i) & 1 ? NaN : vals[i];

    for (int py = 0; py < (1 << N); ++py) {
      std::vector<double> y(N);
      for (int i = 0; i < N; ++i)
        y[i] = (py >> i) & 1 ? NaN : vals[i];

      const double linear = dtwAROW_L<double>(x, y);
      const double full   = dtwAROW<double>(x, y);
      const double banded = dtwAROW_banded<double>(x, y, 100);

      if (!is_valid_distance(linear)) {
        ++nan_outputs;
        FAIL_CHECK("dtwAROW_L returned invalid distance " << linear
                   << " for x=" << fmt(x) << " y=" << fmt(y));
      }
      if (!is_valid_distance(full)) {
        ++nan_outputs;
        FAIL_CHECK("dtwAROW returned invalid distance " << full
                   << " for x=" << fmt(x) << " y=" << fmt(y));
      }
      if (!is_valid_distance(banded)) {
        ++nan_outputs;
        FAIL_CHECK("dtwAROW_banded returned invalid distance " << banded
                   << " for x=" << fmt(x) << " y=" << fmt(y));
      }
    }
  }
  REQUIRE(nan_outputs == 0);
}

TEST_CASE("Output is never NaN: random 200 pairs, all variants",
          "[arow_adversarial][nan_output]")
{
  std::mt19937 rng(0x11223344);
  std::uniform_int_distribution<size_t> len_dist(1, 25);
  std::uniform_real_distribution<double> nan_rate_dist(0.0, 1.0);

  int nan_outputs = 0;

  for (int trial = 0; trial < 200; ++trial) {
    const size_t nx = len_dist(rng);
    const size_t ny = len_dist(rng);
    const double p  = nan_rate_dist(rng);

    auto xraw = random_series(rng, nx);
    auto yraw = random_series(rng, ny);
    auto x = inject_nan(rng, xraw, p);
    auto y = inject_nan(rng, yraw, p);

    const double linear = dtwAROW_L<double>(x, y);
    const double full   = dtwAROW<double>(x, y);
    const double banded = dtwAROW_banded<double>(x, y, 50);

    INFO("Trial " << trial << ": nx=" << nx << " ny=" << ny << " p_nan=" << p);

    if (!is_valid_distance(linear)) {
      ++nan_outputs;
      FAIL_CHECK("dtwAROW_L: " << linear << " x=" << fmt(x) << " y=" << fmt(y));
    }
    if (!is_valid_distance(full)) {
      ++nan_outputs;
      FAIL_CHECK("dtwAROW: " << full << " x=" << fmt(x) << " y=" << fmt(y));
    }
    if (!is_valid_distance(banded)) {
      ++nan_outputs;
      FAIL_CHECK("dtwAROW_banded: " << banded << " x=" << fmt(x) << " y=" << fmt(y));
    }
  }
  REQUIRE(nan_outputs == 0);
}

// ===========================================================================
// 9. Extra: cross-metric consistency (SquaredL2 variant)
// ===========================================================================

TEST_CASE("Linear-space == full-matrix: SquaredL2 metric with NaN",
          "[arow_adversarial][metrics]")
{
  std::mt19937 rng(0xAABBCCDD);
  std::uniform_int_distribution<size_t> len_dist(2, 15);

  int mismatches = 0;
  for (int trial = 0; trial < 50; ++trial) {
    const size_t nx = len_dist(rng);
    const size_t ny = len_dist(rng);

    auto xraw = random_series(rng, nx);
    auto yraw = random_series(rng, ny);
    auto x = inject_nan(rng, xraw, 0.30);
    auto y = inject_nan(rng, yraw, 0.30);

    const double linear = dtwAROW_L<double>(x, y, core::MetricType::SquaredL2);
    const double full   = dtwAROW<double>(x, y, core::MetricType::SquaredL2);

    INFO("Trial " << trial);
    REQUIRE(is_valid_distance(linear));
    REQUIRE(is_valid_distance(full));

    if (std::abs(linear - full) > EPS) {
      ++mismatches;
      FAIL_CHECK("SquaredL2 mismatch: linear=" << linear << " full=" << full
                 << "\n  x=" << fmt(x) << "\n  y=" << fmt(y));
    }
  }
  REQUIRE(mismatches == 0);
}

TEST_CASE("AROW L1 >= ZeroCost L1: SquaredL2 variant also satisfies invariant",
          "[arow_adversarial][metrics]")
{
  std::mt19937 rng(0xEEFF0011);
  std::uniform_int_distribution<size_t> len_dist(2, 15);

  int violations = 0;
  for (int trial = 0; trial < 100; ++trial) {
    const size_t nx = len_dist(rng);
    const size_t ny = len_dist(rng);

    auto xraw = random_series(rng, nx);
    auto yraw = random_series(rng, ny);
    auto x = inject_nan(rng, xraw, 0.30);
    auto y = inject_nan(rng, yraw, 0.30);

    const double arow      = dtwAROW_L<double>(x, y, core::MetricType::SquaredL2);
    const double zero_cost = dtwMissing_L<double>(x, y, -1, core::MetricType::SquaredL2);

    REQUIRE(is_valid_distance(arow));
    REQUIRE(is_valid_distance(zero_cost));

    if (arow < zero_cost - 1e-9) {
      ++violations;
      FAIL_CHECK("SquaredL2 AROW < ZeroCost: arow=" << arow
                 << " zero_cost=" << zero_cost
                 << "\n  x=" << fmt(x) << "\n  y=" << fmt(y));
    }
  }
  REQUIRE(violations == 0);
}

// ===========================================================================
// 10. Pointer API: pointer+length overloads agree with vector overloads
// ===========================================================================

TEST_CASE("Pointer API: dtwAROW_L pointer == vector overload",
          "[arow_adversarial][api]")
{
  std::mt19937 rng(0x13579BDF);
  std::uniform_int_distribution<size_t> len_dist(2, 20);

  for (int trial = 0; trial < 30; ++trial) {
    const size_t nx = len_dist(rng);
    const size_t ny = len_dist(rng);

    auto xraw = random_series(rng, nx);
    auto yraw = random_series(rng, ny);
    auto x = inject_nan(rng, xraw, 0.25);
    auto y = inject_nan(rng, yraw, 0.25);

    const double vec_result = dtwAROW_L<double>(x, y);
    const double ptr_result = dtwAROW_L<double>(x.data(), x.size(), y.data(), y.size());

    INFO("Trial " << trial);
    REQUIRE_THAT(ptr_result, WithinAbs(vec_result, EPS));
  }
}

TEST_CASE("Pointer API: dtwAROW pointer == vector overload",
          "[arow_adversarial][api]")
{
  std::mt19937 rng(0x2468ACE0);
  std::uniform_int_distribution<size_t> len_dist(2, 20);

  for (int trial = 0; trial < 30; ++trial) {
    const size_t nx = len_dist(rng);
    const size_t ny = len_dist(rng);

    auto xraw = random_series(rng, nx);
    auto yraw = random_series(rng, ny);
    auto x = inject_nan(rng, xraw, 0.25);
    auto y = inject_nan(rng, yraw, 0.25);

    const double vec_result = dtwAROW<double>(x, y);
    const double ptr_result = dtwAROW<double>(x.data(), x.size(), y.data(), y.size());

    INFO("Trial " << trial);
    REQUIRE_THAT(ptr_result, WithinAbs(vec_result, EPS));
  }
}
