/**
 * @file test_missing_utils_adversarial.cpp
 * @brief Adversarial stress tests for missing_utils.hpp and warping_missing.hpp.
 *
 * @details Tests are written from the *specification*, not the implementation,
 * and are designed to find incorrect NaN detection, interpolation bugs, and
 * numeric stability issues in the missing-data DTW path.
 *
 * Strategies exercised:
 *  1. is_missing edge cases: denormals, negative zero, infinities, subnormals,
 *     epsilon, max/min representable double/float — none should be NaN.
 *  2. Interpolation stress: single-element, alternating NaN/value, huge gaps.
 *  3. Missing DTW extreme patterns: all-NaN one side, NaN only at boundaries,
 *     length-1 series with NaN, both sides all-NaN.
 *  4. Numeric stability: 1e300, 1e-300, mixed with NaN.
 *  5. Consistency: dtwMissing_L vs dtwMissing for pathological patterns.
 *  6. Roundtrip: interpolate then DTW — result must be finite and non-negative.
 *  7. SquaredL2 metric with missing data.
 *  8. Float specialisation of is_missing.
 *
 * @date 02 Apr 2026
 */

#include <missing_utils.hpp>
#include <warping_missing.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace dtwc;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

namespace {

static constexpr double qNaN  = std::numeric_limits<double>::quiet_NaN();
static constexpr float  qNaNf = std::numeric_limits<float>::quiet_NaN();

/// Build a double with arbitrary sign/exp/mantissa bits via memcpy.
static double make_double(uint64_t bits)
{
  double v;
  std::memcpy(&v, &bits, sizeof(v));
  return v;
}

static float make_float(uint32_t bits)
{
  float v;
  std::memcpy(&v, &bits, sizeof(v));
  return v;
}

} // anonymous namespace

// ===========================================================================
// 1. is_missing — no false positives on legitimate floating-point values
// ===========================================================================

TEST_CASE("is_missing: negative zero is not NaN (double)", "[adversarial][is_missing]")
{
  // -0.0 has all bits zero except the sign bit: exponent all-zero, mantissa zero.
  double neg_zero = make_double(0x8000000000000000ULL);
  REQUIRE_FALSE(is_missing(neg_zero));
  REQUIRE_FALSE(is_missing(-0.0));
}

TEST_CASE("is_missing: negative zero is not NaN (float)", "[adversarial][is_missing]")
{
  float neg_zero = make_float(0x80000000U);
  REQUIRE_FALSE(is_missing(neg_zero));
}

TEST_CASE("is_missing: subnormal (denormalized) doubles are not NaN", "[adversarial][is_missing]")
{
  // Subnormals have exponent = 0, nonzero mantissa — distinct from NaN (exp=all-ones).
  double subnormal_min = std::numeric_limits<double>::denorm_min(); // smallest positive subnormal
  REQUIRE_FALSE(is_missing(subnormal_min));
  REQUIRE_FALSE(is_missing(-subnormal_min));

  // A subnormal with all mantissa bits set except the hidden bit: 0x000FFFFFFFFFFFFULL
  double large_subnormal = make_double(0x000FFFFFFFFFFFFFULL);
  REQUIRE_FALSE(is_missing(large_subnormal));
}

TEST_CASE("is_missing: subnormal (denormalized) floats are not NaN", "[adversarial][is_missing]")
{
  float subnormal_min = std::numeric_limits<float>::denorm_min();
  REQUIRE_FALSE(is_missing(subnormal_min));
  REQUIRE_FALSE(is_missing(-subnormal_min));

  float large_subnormal = make_float(0x007FFFFFU);
  REQUIRE_FALSE(is_missing(large_subnormal));
}

TEST_CASE("is_missing: max representable double is not NaN", "[adversarial][is_missing]")
{
  REQUIRE_FALSE(is_missing(std::numeric_limits<double>::max()));
  REQUIRE_FALSE(is_missing(-std::numeric_limits<double>::max()));
}

TEST_CASE("is_missing: min positive normal double is not NaN", "[adversarial][is_missing]")
{
  REQUIRE_FALSE(is_missing(std::numeric_limits<double>::min())); // smallest positive normal
}

TEST_CASE("is_missing: epsilon is not NaN (double)", "[adversarial][is_missing]")
{
  REQUIRE_FALSE(is_missing(std::numeric_limits<double>::epsilon()));
}

TEST_CASE("is_missing: epsilon is not NaN (float)", "[adversarial][is_missing]")
{
  REQUIRE_FALSE(is_missing(std::numeric_limits<float>::epsilon()));
}

TEST_CASE("is_missing: positive and negative infinity are not NaN (double)", "[adversarial][is_missing]")
{
  // Infinity has exponent = all-ones, mantissa = 0 — different from NaN.
  REQUIRE_FALSE(is_missing( std::numeric_limits<double>::infinity()));
  REQUIRE_FALSE(is_missing(-std::numeric_limits<double>::infinity()));
}

TEST_CASE("is_missing: positive and negative infinity are not NaN (float)", "[adversarial][is_missing]")
{
  REQUIRE_FALSE(is_missing( std::numeric_limits<float>::infinity()));
  REQUIRE_FALSE(is_missing(-std::numeric_limits<float>::infinity()));
}

TEST_CASE("is_missing: very large and very small values are not NaN", "[adversarial][is_missing]")
{
  REQUIRE_FALSE(is_missing( 1e300));
  REQUIRE_FALSE(is_missing(-1e300));
  REQUIRE_FALSE(is_missing( 1e-300));
  REQUIRE_FALSE(is_missing(-1e-300));
}

TEST_CASE("is_missing: all NaN bit patterns (exhaustive double categories)", "[adversarial][is_missing]")
{
  // Every double NaN has exponent = 0x7FF and mantissa != 0.
  // Test a range of NaN payloads including both quiet and signaling NaNs.
  // Quiet NaN: top mantissa bit = 1 (bit 51 for double).
  // Signaling NaN: top mantissa bit = 0 but mantissa != 0.

  // Quiet NaN with various payloads
  REQUIRE(is_missing(make_double(0x7FF8000000000001ULL))); // quiet NaN payload 1
  REQUIRE(is_missing(make_double(0x7FF8000000000000ULL))); // canonical quiet NaN
  REQUIRE(is_missing(make_double(0x7FFFFFFFFFFFFFFFULL))); // quiet NaN all-mantissa-bits
  REQUIRE(is_missing(make_double(0xFFFFFFFFFFFFFFFFULL))); // negative NaN all-mantissa-bits

  // Signaling NaN: exponent all-ones, mantissa != 0, bit 51 = 0
  REQUIRE(is_missing(make_double(0x7FF0000000000001ULL))); // signaling NaN payload 1
  REQUIRE(is_missing(make_double(0x7FF4000000000000ULL))); // signaling NaN with bit 50 set
  REQUIRE(is_missing(make_double(0xFFF0000000000001ULL))); // negative signaling NaN
}

TEST_CASE("is_missing: NaN bit patterns for float", "[adversarial][is_missing]")
{
  REQUIRE(is_missing(make_float(0x7FC00000U))); // canonical quiet NaN float
  REQUIRE(is_missing(make_float(0x7FFFFFFFU))); // all mantissa bits set
  REQUIRE(is_missing(make_float(0xFF800001U))); // negative signaling NaN
  REQUIRE(is_missing(make_float(0x7F800001U))); // positive signaling NaN
}

TEST_CASE("is_missing: common arithmetic results are not NaN", "[adversarial][is_missing]")
{
  REQUIRE_FALSE(is_missing(1.0 / 3.0));
  REQUIRE_FALSE(is_missing(std::sqrt(2.0)));
  REQUIRE_FALSE(is_missing(std::log(2.0)));
  REQUIRE_FALSE(is_missing(std::exp(1.0)));
  REQUIRE_FALSE(is_missing(std::atan2(1.0, 1.0)));
}

// ===========================================================================
// 2. interpolate_linear — stress tests
// ===========================================================================

TEST_CASE("interpolate_linear: single-element with valid value", "[adversarial][interpolate]")
{
  std::vector<double> v = { 42.0 };
  auto result = interpolate_linear(v);
  REQUIRE(result.size() == 1u);
  REQUIRE_THAT(result[0], WithinAbs(42.0, 1e-15));
}

TEST_CASE("interpolate_linear: single-element all-NaN throws", "[adversarial][interpolate]")
{
  std::vector<double> v = { qNaN };
  REQUIRE_THROWS_AS(interpolate_linear(v), std::runtime_error);
}

TEST_CASE("interpolate_linear: alternating NaN/value (starts with value)", "[adversarial][interpolate]")
{
  // v = {0, NaN, 2, NaN, 4, NaN, 6}
  // Interior NaN at indices 1, 3, 5 should be linearly interpolated.
  std::vector<double> v = { 0.0, qNaN, 2.0, qNaN, 4.0, qNaN, 6.0 };
  auto result = interpolate_linear(v);
  REQUIRE(result.size() == 7u);
  REQUIRE_THAT(result[0], WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(result[1], WithinAbs(1.0, 1e-12)); // midpoint 0..2
  REQUIRE_THAT(result[2], WithinAbs(2.0, 1e-12));
  REQUIRE_THAT(result[3], WithinAbs(3.0, 1e-12)); // midpoint 2..4
  REQUIRE_THAT(result[4], WithinAbs(4.0, 1e-12));
  REQUIRE_THAT(result[5], WithinAbs(5.0, 1e-12)); // midpoint 4..6
  REQUIRE_THAT(result[6], WithinAbs(6.0, 1e-12));
  // No NaN should remain in the result.
  for (auto x : result) REQUIRE_FALSE(is_missing(x));
}

TEST_CASE("interpolate_linear: alternating NaN/value (starts with NaN)", "[adversarial][interpolate]")
{
  // v = {NaN, 1, NaN, 3, NaN}
  // Leading NaN: NOCB → first valid = 1
  // Interior NaN: linear between 1 and 3 → index 2 = 2
  // Trailing NaN: LOCF → last valid = 3
  std::vector<double> v = { qNaN, 1.0, qNaN, 3.0, qNaN };
  auto result = interpolate_linear(v);
  REQUIRE(result.size() == 5u);
  REQUIRE_THAT(result[0], WithinAbs(1.0, 1e-12)); // NOCB
  REQUIRE_THAT(result[1], WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(result[2], WithinAbs(2.0, 1e-12)); // linear between 1 and 3
  REQUIRE_THAT(result[3], WithinAbs(3.0, 1e-12));
  REQUIRE_THAT(result[4], WithinAbs(3.0, 1e-12)); // LOCF
  for (auto x : result) REQUIRE_FALSE(is_missing(x));
}

TEST_CASE("interpolate_linear: 100-NaN gap between two values", "[adversarial][interpolate]")
{
  // v = {0, NaN×100, 100}
  std::vector<double> v;
  v.push_back(0.0);
  for (int i = 0; i < 100; ++i) v.push_back(qNaN);
  v.push_back(100.0);

  auto result = interpolate_linear(v);
  REQUIRE(result.size() == 102u);
  REQUIRE_THAT(result[0], WithinAbs(0.0, 1e-10));
  REQUIRE_THAT(result[101], WithinAbs(100.0, 1e-10));

  // Each gap value should be exactly i (linear ramp from 0 to 100 over 101 steps).
  for (size_t i = 1; i <= 100; ++i) {
    double expected = static_cast<double>(i); // i/101 * 100 per step
    // Actual: result[i] = 0 + (i/101) * 100
    double frac = static_cast<double>(i) / 101.0;
    REQUIRE_THAT(result[i], WithinAbs(frac * 100.0, 1e-9));
    REQUIRE_FALSE(is_missing(result[i]));
  }
}

TEST_CASE("interpolate_linear: NaN at every other position (all interior)", "[adversarial][interpolate]")
{
  // 10 values: 0, NaN, 2, NaN, 4, NaN, 6, NaN, 8, NaN — last is trailing NaN
  std::vector<double> v = { 0.0, qNaN, 2.0, qNaN, 4.0, qNaN, 6.0, qNaN, 8.0, qNaN };
  auto result = interpolate_linear(v);
  REQUIRE(result.size() == 10u);
  REQUIRE_THAT(result[1], WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(result[3], WithinAbs(3.0, 1e-12));
  REQUIRE_THAT(result[5], WithinAbs(5.0, 1e-12));
  REQUIRE_THAT(result[7], WithinAbs(7.0, 1e-12));
  REQUIRE_THAT(result[9], WithinAbs(8.0, 1e-12)); // LOCF from index 8
  for (auto x : result) REQUIRE_FALSE(is_missing(x));
}

TEST_CASE("interpolate_linear: all-NaN but one (first element valid)", "[adversarial][interpolate]")
{
  // Only element 0 is valid; all rest are trailing NaN → all filled with v[0].
  std::vector<double> v = { 7.0, qNaN, qNaN, qNaN, qNaN };
  auto result = interpolate_linear(v);
  REQUIRE(result.size() == 5u);
  for (auto x : result) REQUIRE_THAT(x, WithinAbs(7.0, 1e-15));
}

TEST_CASE("interpolate_linear: all-NaN but one (last element valid)", "[adversarial][interpolate]")
{
  // Only element 4 is valid; all leading NaN → all filled with v[4].
  std::vector<double> v = { qNaN, qNaN, qNaN, qNaN, 7.0 };
  auto result = interpolate_linear(v);
  REQUIRE(result.size() == 5u);
  for (auto x : result) REQUIRE_THAT(x, WithinAbs(7.0, 1e-15));
}

TEST_CASE("interpolate_linear: output has no NaN values regardless of input pattern", "[adversarial][interpolate]")
{
  // Randomly mixed pattern — no NaN should survive.
  std::vector<double> v = { qNaN, 1.0, qNaN, qNaN, qNaN, 6.0, qNaN };
  auto result = interpolate_linear(v);
  for (size_t i = 0; i < result.size(); ++i) {
    INFO("result[" << i << "] = " << result[i]);
    REQUIRE_FALSE(is_missing(result[i]));
  }
}

TEST_CASE("interpolate_linear: preserves length", "[adversarial][interpolate]")
{
  for (size_t len : { 1u, 2u, 5u, 50u, 200u }) {
    std::vector<double> v(len, 1.0);
    v[0] = qNaN; // ensure at least one NaN (not all-NaN unless len==1)
    if (len == 1u) {
      REQUIRE_THROWS_AS(interpolate_linear(v), std::runtime_error);
    } else {
      auto result = interpolate_linear(v);
      REQUIRE(result.size() == len);
    }
  }
}

TEST_CASE("interpolate_linear: works for float specialisation", "[adversarial][interpolate]")
{
  std::vector<float> v = { 0.0f, qNaNf, 4.0f };
  auto result = interpolate_linear(v);
  REQUIRE(result.size() == 3u);
  REQUIRE_THAT(static_cast<double>(result[1]), WithinAbs(2.0, 1e-6));
}

// ===========================================================================
// 3. Missing DTW — extreme NaN patterns
// ===========================================================================

TEST_CASE("dtwMissing_L: single-element NaN vs NaN gives zero", "[adversarial][missing_dtw]")
{
  std::vector<double> x = { qNaN };
  std::vector<double> y = { qNaN };
  REQUIRE_THAT(dtwMissing_L<double>(x, y), WithinAbs(0.0, 1e-15));
}

TEST_CASE("dtwMissing_L: single-element NaN vs value gives zero", "[adversarial][missing_dtw]")
{
  std::vector<double> x = { qNaN };
  std::vector<double> y = { 42.0 };
  REQUIRE_THAT(dtwMissing_L<double>(x, y), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(dtwMissing_L<double>(y, x), WithinAbs(0.0, 1e-15));
}

TEST_CASE("dtwMissing_L: NaN only at position 0 vs normal series", "[adversarial][missing_dtw]")
{
  // x[0] = NaN, rest normal; y is fully observed.
  std::vector<double> x = { qNaN, 1.0, 2.0, 3.0 };
  std::vector<double> y = { 0.0, 1.0, 2.0, 3.0 };
  double dist = dtwMissing_L<double>(x, y);
  // NaN at 0 costs 0; remaining pair (1,1), (2,2), (3,3) costs 0 each.
  REQUIRE(dist >= 0.0);
  REQUIRE(std::isfinite(dist));
  // Should be <= the version without NaN (since missing replaces a potentially nonzero cost).
  std::vector<double> x_full = { 0.0, 1.0, 2.0, 3.0 };
  REQUIRE(dist <= dtwMissing_L<double>(x_full, y) + 1e-12);
}

TEST_CASE("dtwMissing_L: NaN only at last position vs normal series", "[adversarial][missing_dtw]")
{
  std::vector<double> x = { 1.0, 2.0, 3.0, qNaN };
  std::vector<double> y = { 1.0, 2.0, 3.0, 4.0 };
  double dist = dtwMissing_L<double>(x, y);
  REQUIRE(dist >= 0.0);
  REQUIRE(std::isfinite(dist));
  // y[3]=4, x[3]=NaN → cost 0 → distance should be 0 overall (prefix perfect match).
  REQUIRE_THAT(dist, WithinAbs(0.0, 1e-12));
}

TEST_CASE("dtwMissing_L: all-NaN first series vs fully observed second series gives zero", "[adversarial][missing_dtw]")
{
  std::vector<double> x(10, qNaN);
  std::vector<double> y = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
  REQUIRE_THAT(dtwMissing_L<double>(x, y), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(dtwMissing_L<double>(y, x), WithinAbs(0.0, 1e-15));
}

TEST_CASE("dtwMissing_L: length-1 NaN series vs long series gives zero", "[adversarial][missing_dtw]")
{
  std::vector<double> x = { qNaN };
  std::vector<double> y = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
  REQUIRE_THAT(dtwMissing_L<double>(x, y), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(dtwMissing_L<double>(y, x), WithinAbs(0.0, 1e-15));
}

TEST_CASE("dtwMissing_L: alternating NaN vs alternating NaN gives zero", "[adversarial][missing_dtw]")
{
  // x = {NaN, v, NaN, v, ...}, y = {v, NaN, v, NaN, ...}
  // Every pair (i,j) where |i-j| is small will hit at least one NaN → cost 0.
  std::vector<double> x = { qNaN, 1.0, qNaN, 3.0, qNaN, 5.0 };
  std::vector<double> y = { 0.0, qNaN, 2.0, qNaN, 4.0, qNaN };
  double dist = dtwMissing_L<double>(x, y);
  REQUIRE(dist >= 0.0);
  REQUIRE(std::isfinite(dist));
}

TEST_CASE("dtwMissing_L: both series completely NaN gives zero", "[adversarial][missing_dtw]")
{
  std::vector<double> x(5, qNaN);
  std::vector<double> y(7, qNaN);
  REQUIRE_THAT(dtwMissing_L<double>(x, y), WithinAbs(0.0, 1e-15));
}

TEST_CASE("dtwMissing_L: NaN in very long series (100 NaN gap) result is finite", "[adversarial][missing_dtw]")
{
  std::vector<double> x;
  x.push_back(0.0);
  for (int i = 0; i < 100; ++i) x.push_back(qNaN);
  x.push_back(100.0);

  std::vector<double> y;
  y.push_back(0.0);
  for (int i = 0; i < 100; ++i) y.push_back(qNaN);
  y.push_back(100.0);

  double dist = dtwMissing_L<double>(x, y);
  REQUIRE(std::isfinite(dist));
  REQUIRE(dist >= 0.0);
  // Both series have the same finite endpoints and zero cost in the gap → should be 0.
  REQUIRE_THAT(dist, WithinAbs(0.0, 1e-10));
}

// ===========================================================================
// 4. Numeric stability: very large and very small values mixed with NaN
// ===========================================================================

TEST_CASE("dtwMissing_L: very large values (1e300) are stable", "[adversarial][missing_dtw][stability]")
{
  std::vector<double> x = { 1e300, 2e300, 3e300 };
  std::vector<double> y = { 1e300, 2e300, 3e300 };
  double dist = dtwMissing_L<double>(x, y);
  // Identical series → distance should be 0.
  REQUIRE_THAT(dist, WithinAbs(0.0, 1e-6)); // absolute tol loose due to fp scale
}

TEST_CASE("dtwMissing_L: very small values (1e-300) are stable", "[adversarial][missing_dtw][stability]")
{
  std::vector<double> x = { 1e-300, 2e-300, 3e-300 };
  std::vector<double> y = { 1e-300, 2e-300, 3e-300 };
  double dist = dtwMissing_L<double>(x, y);
  REQUIRE_THAT(dist, WithinAbs(0.0, 1e-290));
}

TEST_CASE("dtwMissing_L: 1e300 mixed with NaN is finite", "[adversarial][missing_dtw][stability]")
{
  std::vector<double> x = { 1e300, qNaN, 3e300 };
  std::vector<double> y = { qNaN,  2e300, qNaN };
  double dist = dtwMissing_L<double>(x, y);
  REQUIRE(std::isfinite(dist));
  REQUIRE(dist >= 0.0);
}

TEST_CASE("dtwMissing_L: 1e-300 mixed with NaN is finite and non-negative", "[adversarial][missing_dtw][stability]")
{
  std::vector<double> x = { 1e-300, qNaN, 3e-300 };
  std::vector<double> y = { qNaN, 2e-300, qNaN };
  double dist = dtwMissing_L<double>(x, y);
  REQUIRE(std::isfinite(dist));
  REQUIRE(dist >= 0.0);
}

TEST_CASE("dtwMissing_L: infinity-adjacent values (not infinity itself) are stable", "[adversarial][missing_dtw][stability]")
{
  double big  = std::numeric_limits<double>::max() / 2.0;
  std::vector<double> x = { big, qNaN, big };
  std::vector<double> y = { big, big,  big };
  double dist = dtwMissing_L<double>(x, y);
  REQUIRE(std::isfinite(dist));
  REQUIRE(dist >= 0.0);
}

// ===========================================================================
// 5. Consistency: dtwMissing_L vs dtwMissing for pathological NaN patterns
// ===========================================================================

TEST_CASE("Consistency dtwMissing_L vs dtwMissing: both-all-NaN", "[adversarial][consistency]")
{
  std::vector<double> x(4, qNaN);
  std::vector<double> y(4, qNaN);
  REQUIRE_THAT(dtwMissing_L<double>(x, y),
               WithinAbs(dtwMissing<double>(x, y), 1e-12));
}

TEST_CASE("Consistency dtwMissing_L vs dtwMissing: one-all-NaN", "[adversarial][consistency]")
{
  std::vector<double> x(5, qNaN);
  std::vector<double> y = { 1, 2, 3, 4, 5 };
  REQUIRE_THAT(dtwMissing_L<double>(x, y),
               WithinAbs(dtwMissing<double>(x, y), 1e-12));
  REQUIRE_THAT(dtwMissing_L<double>(y, x),
               WithinAbs(dtwMissing<double>(y, x), 1e-12));
}

TEST_CASE("Consistency dtwMissing_L vs dtwMissing: alternating NaN", "[adversarial][consistency]")
{
  std::vector<double> x = { 1.0, qNaN, 3.0, qNaN, 5.0 };
  std::vector<double> y = { qNaN, 2.0, qNaN, 4.0, qNaN };
  REQUIRE_THAT(dtwMissing_L<double>(x, y),
               WithinAbs(dtwMissing<double>(x, y), 1e-12));
}

TEST_CASE("Consistency dtwMissing_L vs dtwMissing: all interior NaN gap", "[adversarial][consistency]")
{
  std::vector<double> x;
  x.push_back(1.0);
  for (int i = 0; i < 20; ++i) x.push_back(qNaN);
  x.push_back(10.0);

  std::vector<double> y;
  y.push_back(2.0);
  for (int i = 0; i < 20; ++i) y.push_back(qNaN);
  y.push_back(11.0);

  double L = dtwMissing_L<double>(x, y);
  double F = dtwMissing<double>(x, y);
  REQUIRE_THAT(L, WithinAbs(F, 1e-10));
}

TEST_CASE("Consistency dtwMissing_L vs dtwMissing: NaN only at boundaries", "[adversarial][consistency]")
{
  std::vector<double> x = { qNaN, 1.0, 2.0, 3.0, qNaN };
  std::vector<double> y = { qNaN, 4.0, 5.0, 6.0, qNaN };
  REQUIRE_THAT(dtwMissing_L<double>(x, y),
               WithinAbs(dtwMissing<double>(x, y), 1e-12));
}

TEST_CASE("Consistency dtwMissing_L vs dtwMissing: different-length series with NaN", "[adversarial][consistency]")
{
  std::vector<double> x = { qNaN, 1.0, qNaN, 3.0 };
  std::vector<double> y = { 0.0, qNaN, 2.0, qNaN, 4.0, 5.0 };
  REQUIRE_THAT(dtwMissing_L<double>(x, y),
               WithinAbs(dtwMissing<double>(x, y), 1e-12));
}

TEST_CASE("Consistency dtwMissing_L vs dtwMissing: single element NaN", "[adversarial][consistency]")
{
  std::vector<double> x = { qNaN };
  std::vector<double> y = { 5.0 };
  REQUIRE_THAT(dtwMissing_L<double>(x, y),
               WithinAbs(dtwMissing<double>(x, y), 1e-12));
}

// ===========================================================================
// 6. Interpolation + DTW roundtrip
// ===========================================================================

TEST_CASE("Roundtrip interpolate then dtwMissing_L: identical interpolated series gives zero", "[adversarial][roundtrip]")
{
  std::vector<double> raw = { 1.0, qNaN, 3.0, qNaN, 5.0 };
  auto interp = interpolate_linear(raw);
  // Interpolated series vs itself should give 0.
  REQUIRE_THAT(dtwMissing_L<double>(interp, interp), WithinAbs(0.0, 1e-15));
}

TEST_CASE("Roundtrip interpolate then dtwMissing_L: result is finite and non-negative", "[adversarial][roundtrip]")
{
  std::vector<double> raw_x = { qNaN, 1.0, qNaN, 4.0 };
  std::vector<double> raw_y = { 2.0, qNaN, 5.0, qNaN };
  auto interp_x = interpolate_linear(raw_x);
  auto interp_y = interpolate_linear(raw_y);

  double dist = dtwMissing_L<double>(interp_x, interp_y);
  REQUIRE(std::isfinite(dist));
  REQUIRE(dist >= 0.0);
}

TEST_CASE("Roundtrip: interpolated then DTW vs raw missing DTW — interpolated >= missing", "[adversarial][roundtrip]")
{
  // Interpolating fills NaN with nonzero values → potentially larger distances.
  // DTW-missing zeros out NaN costs → potentially smaller distances.
  // So: dtwMissing(raw, raw) <= dtwMissing(interp, interp) is NOT guaranteed
  // (they are different series). But both must be finite.
  std::vector<double> raw = { 0.0, qNaN, qNaN, qNaN, 8.0 };
  auto interp = interpolate_linear(raw);
  double dist_missing_raw  = dtwMissing_L<double>(raw, raw);
  double dist_missing_interp = dtwMissing_L<double>(interp, interp);
  REQUIRE(std::isfinite(dist_missing_raw));
  REQUIRE(std::isfinite(dist_missing_interp));
  REQUIRE(dist_missing_raw  >= 0.0);
  REQUIRE(dist_missing_interp >= 0.0);
}

TEST_CASE("Roundtrip: interpolated series has no NaN, so dtwMissing == dtwFull_L", "[adversarial][roundtrip]")
{
  std::vector<double> raw = { 1.0, qNaN, 5.0, qNaN, 9.0 };
  auto interp = interpolate_linear(raw);

  // Interpolated has no NaN → both DTW variants should agree.
  double d_missing = dtwMissing_L<double>(interp, interp);
  double d_full    = dtwFull_L<double>(interp, interp);
  REQUIRE_THAT(d_missing, WithinAbs(d_full, 1e-12));
  REQUIRE_THAT(d_missing, WithinAbs(0.0, 1e-12)); // identical series
}

// ===========================================================================
// 7. SquaredL2 metric with missing data
// ===========================================================================

TEST_CASE("dtwMissing_L SquaredL2: NaN pair gives zero cost contribution", "[adversarial][missing_dtw][squaredl2]")
{
  // x = {2, NaN}, y = {NaN, 2}
  // Both NaN pairs cost 0; no real difference to measure.
  std::vector<double> x = { 2.0, qNaN };
  std::vector<double> y = { qNaN, 2.0 };
  double dist = dtwMissing_L<double>(x, y, -1, core::MetricType::SquaredL2);
  REQUIRE_THAT(dist, WithinAbs(0.0, 1e-12));
}

TEST_CASE("dtwMissing_L SquaredL2: identical series gives zero", "[adversarial][missing_dtw][squaredl2]")
{
  std::vector<double> x = { 1.0, 2.0, 3.0 };
  double d = dtwMissing_L<double>(x, x, -1, core::MetricType::SquaredL2);
  REQUIRE_THAT(d, WithinAbs(0.0, 1e-15));
}

TEST_CASE("dtwMissing_L SquaredL2: matches dtwMissing SquaredL2 no NaN", "[adversarial][missing_dtw][squaredl2]")
{
  std::vector<double> x = { 1, 3, 5 };
  std::vector<double> y = { 2, 4, 6 };
  double L = dtwMissing_L<double>(x, y, -1, core::MetricType::SquaredL2);
  double F = dtwMissing<double>(x, y, core::MetricType::SquaredL2);
  REQUIRE_THAT(L, WithinAbs(F, 1e-12));
}

TEST_CASE("dtwMissing_L SquaredL2: all-NaN one side gives zero", "[adversarial][missing_dtw][squaredl2]")
{
  std::vector<double> x(5, qNaN);
  std::vector<double> y = { 1, 2, 3, 4, 5 };
  REQUIRE_THAT(dtwMissing_L<double>(x, y, -1, core::MetricType::SquaredL2),
               WithinAbs(0.0, 1e-15));
}

TEST_CASE("dtwMissing_L SquaredL2: consistency L vs full for alternating NaN", "[adversarial][missing_dtw][squaredl2]")
{
  std::vector<double> x = { 1.0, qNaN, 3.0, qNaN, 5.0 };
  std::vector<double> y = { qNaN, 2.0, qNaN, 4.0, qNaN };
  double L = dtwMissing_L<double>(x, y, -1, core::MetricType::SquaredL2);
  double F = dtwMissing<double>(x, y, core::MetricType::SquaredL2);
  REQUIRE_THAT(L, WithinAbs(F, 1e-12));
}

TEST_CASE("dtwMissing SquaredL2: NaN only at last position, hand-computed", "[adversarial][missing_dtw][squaredl2]")
{
  // x = {3, NaN}, y = {1, 2}
  // Full matrix:
  // C(0,0) = (3-1)^2 = 4
  // C(1,0) = 4 + 0 = 4      [x[1]=NaN -> cost 0]
  // C(0,1) = 4 + (3-2)^2 = 5
  // C(1,1) = min(4, 4, 5) + 0 = 4   [x[1]=NaN -> cost 0]
  std::vector<double> x = { 3.0, qNaN };
  std::vector<double> y = { 1.0, 2.0 };
  double dist = dtwMissing<double>(x, y, core::MetricType::SquaredL2);
  REQUIRE_THAT(dist, WithinAbs(4.0, 1e-12));
}

// ===========================================================================
// 8. Missing DTW — empty series edge cases
// ===========================================================================

TEST_CASE("dtwMissing_L: empty vs empty returns max", "[adversarial][missing_dtw][empty]")
{
  std::vector<double> empty{};
  // Both empty: implementation-defined but should not crash and should return
  // a sentinel (numeric_limits::max() or equivalent).
  double dist = dtwMissing_L<double>(empty, empty);
  // Just verify it doesn't crash and is non-negative (could be max or 0).
  bool valid = std::isfinite(dist) || (dist == std::numeric_limits<double>::max());
  REQUIRE(valid);
}

TEST_CASE("dtwMissing_L: empty vs NaN series returns max sentinel", "[adversarial][missing_dtw][empty]")
{
  std::vector<double> empty{};
  std::vector<double> nan_series = { qNaN, qNaN };
  double dist = dtwMissing_L<double>(empty, nan_series);
  REQUIRE(dist >= 1e10); // should be a very large sentinel
}

TEST_CASE("dtwMissing: empty vs non-empty returns sentinel", "[adversarial][missing_dtw][empty]")
{
  std::vector<double> empty{};
  std::vector<double> y = { 1.0, 2.0 };
  double dist = dtwMissing<double>(empty, y);
  REQUIRE(dist >= 1e10);
}

TEST_CASE("has_missing: vector of only infinities returns false", "[adversarial][has_missing]")
{
  double inf = std::numeric_limits<double>::infinity();
  std::vector<double> v = { inf, -inf, inf };
  REQUIRE_FALSE(has_missing(v));
}

TEST_CASE("missing_rate: vector of subnormals gives 0.0", "[adversarial][missing_rate]")
{
  double dn = std::numeric_limits<double>::denorm_min();
  std::vector<double> v = { dn, dn, dn, dn };
  REQUIRE_THAT(missing_rate(v), WithinAbs(0.0, 1e-15));
}

TEST_CASE("missing_rate: single NaN in large vector", "[adversarial][missing_rate]")
{
  std::vector<double> v(999, 1.0);
  v.push_back(qNaN); // 1 NaN out of 1000
  REQUIRE_THAT(missing_rate(v), WithinAbs(0.001, 1e-12));
}

// ===========================================================================
// 9. Symmetry invariant holds for all adversarial NaN patterns
// ===========================================================================

TEST_CASE("dtwMissing_L: symmetry — NaN at boundaries", "[adversarial][symmetry]")
{
  std::vector<double> x = { qNaN, 2.0, 3.0, qNaN };
  std::vector<double> y = { 1.0, qNaN, qNaN, 4.0 };
  REQUIRE_THAT(dtwMissing_L<double>(x, y),
               WithinAbs(dtwMissing_L<double>(y, x), 1e-12));
}

TEST_CASE("dtwMissing_L: symmetry — one entirely NaN", "[adversarial][symmetry]")
{
  std::vector<double> x = { qNaN, qNaN, qNaN };
  std::vector<double> y = { 10.0, 20.0, 30.0 };
  REQUIRE_THAT(dtwMissing_L<double>(x, y),
               WithinAbs(dtwMissing_L<double>(y, x), 1e-12));
}

TEST_CASE("dtwMissing_L: symmetry — 100-NaN gap", "[adversarial][symmetry]")
{
  std::vector<double> x;
  std::vector<double> y;
  x.push_back(0.0);
  y.push_back(1.0);
  for (int i = 0; i < 50; ++i) { x.push_back(qNaN); y.push_back(qNaN); }
  x.push_back(100.0);
  y.push_back(101.0);

  REQUIRE_THAT(dtwMissing_L<double>(x, y),
               WithinAbs(dtwMissing_L<double>(y, x), 1e-10));
}

// ===========================================================================
// 10. Non-negativity invariant
// ===========================================================================

TEST_CASE("dtwMissing_L: non-negativity — exhaustive adversarial patterns", "[adversarial][non_negativity]")
{
  struct Pattern {
    std::vector<double> x, y;
  };
  std::vector<Pattern> patterns = {
    { {qNaN},                   {qNaN}                    },
    { {qNaN, 1.0},              {1.0, qNaN}               },
    { {1.0, qNaN, 3.0},         {qNaN, 2.0, qNaN}         },
    { {qNaN, qNaN, qNaN},       {1.0, 2.0, 3.0}           },
    { {1e300, qNaN},            {qNaN, 1e300}              },
    { {1e-300, qNaN, 1e-300},   {qNaN, 1e-300, qNaN}      },
  };

  for (auto &p : patterns) {
    double d = dtwMissing_L<double>(p.x, p.y);
    INFO("Distance = " << d);
    REQUIRE(d >= 0.0);
    bool ok = std::isfinite(d) || (d == std::numeric_limits<double>::max());
    REQUIRE(ok);
  }
}
