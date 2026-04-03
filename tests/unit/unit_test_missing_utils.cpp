/**
 * @file unit_test_missing_utils.cpp
 * @brief Unit tests for missing_utils.hpp (bitwise NaN check, interpolation).
 *
 * @details Tests is_missing<T>(), has_missing(), missing_rate(), and
 * interpolate_linear() — all of which use bitwise NaN detection safe
 * under -ffast-math / /fp:fast.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#include <missing_utils.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>
#include <vector>

using Catch::Matchers::WithinAbs;

// ===========================================================================
// is_missing — double
// ===========================================================================

TEST_CASE("is_missing: detects quiet NaN (double)", "[missing_utils]")
{
  double val = std::numeric_limits<double>::quiet_NaN();
  REQUIRE(dtwc::is_missing(val));
}

TEST_CASE("is_missing: detects signaling NaN (double)", "[missing_utils]")
{
  double val = std::numeric_limits<double>::signaling_NaN();
  REQUIRE(dtwc::is_missing(val));
}

TEST_CASE("is_missing: rejects normal values (double)", "[missing_utils]")
{
  REQUIRE_FALSE(dtwc::is_missing(0.0));
  REQUIRE_FALSE(dtwc::is_missing(1.0));
  REQUIRE_FALSE(dtwc::is_missing(-1.0));
  REQUIRE_FALSE(dtwc::is_missing(1e300));
  REQUIRE_FALSE(dtwc::is_missing(-1e-300));
}

TEST_CASE("is_missing: rejects infinity (double)", "[missing_utils]")
{
  REQUIRE_FALSE(dtwc::is_missing(std::numeric_limits<double>::infinity()));
  REQUIRE_FALSE(dtwc::is_missing(-std::numeric_limits<double>::infinity()));
}

// ===========================================================================
// is_missing — float
// ===========================================================================

TEST_CASE("is_missing: detects quiet NaN (float)", "[missing_utils]")
{
  float nan_f = std::numeric_limits<float>::quiet_NaN();
  REQUIRE(dtwc::is_missing(nan_f));
}

TEST_CASE("is_missing: rejects normal values (float)", "[missing_utils]")
{
  REQUIRE_FALSE(dtwc::is_missing(0.0f));
  REQUIRE_FALSE(dtwc::is_missing(1.0f));
  REQUIRE_FALSE(dtwc::is_missing(-1.0f));
}

TEST_CASE("is_missing: rejects infinity (float)", "[missing_utils]")
{
  REQUIRE_FALSE(dtwc::is_missing(std::numeric_limits<float>::infinity()));
  REQUIRE_FALSE(dtwc::is_missing(-std::numeric_limits<float>::infinity()));
}

// ===========================================================================
// has_missing
// ===========================================================================

TEST_CASE("has_missing: empty vector returns false", "[missing_utils]")
{
  std::vector<double> v;
  REQUIRE_FALSE(dtwc::has_missing(v));
}

TEST_CASE("has_missing: no NaN returns false", "[missing_utils]")
{
  std::vector<double> v = { 1.0, 2.0, 3.0, 4.0 };
  REQUIRE_FALSE(dtwc::has_missing(v));
}

TEST_CASE("has_missing: vector with one NaN returns true", "[missing_utils]")
{
  double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> v = { 1.0, nan, 3.0 };
  REQUIRE(dtwc::has_missing(v));
}

TEST_CASE("has_missing: all NaN returns true", "[missing_utils]")
{
  double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> v = { nan, nan, nan };
  REQUIRE(dtwc::has_missing(v));
}

// ===========================================================================
// missing_rate
// ===========================================================================

TEST_CASE("missing_rate: no NaN gives 0.0", "[missing_utils]")
{
  std::vector<double> v = { 1.0, 2.0, 3.0, 4.0 };
  REQUIRE_THAT(dtwc::missing_rate(v), WithinAbs(0.0, 1e-15));
}

TEST_CASE("missing_rate: half NaN gives 0.5", "[missing_utils]")
{
  double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> v = { 1.0, nan, 3.0, nan };
  REQUIRE_THAT(dtwc::missing_rate(v), WithinAbs(0.5, 1e-15));
}

TEST_CASE("missing_rate: empty vector gives 0.0", "[missing_utils]")
{
  std::vector<double> v;
  REQUIRE_THAT(dtwc::missing_rate(v), WithinAbs(0.0, 1e-15));
}

// ===========================================================================
// interpolate_linear
// ===========================================================================

TEST_CASE("interpolate_linear: no NaN returns unchanged values", "[missing_utils]")
{
  std::vector<double> v = { 1.0, 2.0, 3.0 };
  auto result = dtwc::interpolate_linear(v);
  REQUIRE(result.size() == 3u);
  REQUIRE_THAT(result[0], WithinAbs(1.0, 1e-15));
  REQUIRE_THAT(result[1], WithinAbs(2.0, 1e-15));
  REQUIRE_THAT(result[2], WithinAbs(3.0, 1e-15));
}

TEST_CASE("interpolate_linear: single interior NaN is linearly interpolated", "[missing_utils]")
{
  double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> v = { 1.0, nan, 3.0 };
  auto result = dtwc::interpolate_linear(v);
  REQUIRE_THAT(result[0], WithinAbs(1.0, 1e-15));
  REQUIRE_THAT(result[1], WithinAbs(2.0, 1e-15));
  REQUIRE_THAT(result[2], WithinAbs(3.0, 1e-15));
}

TEST_CASE("interpolate_linear: multi-gap interior NaN is linearly interpolated", "[missing_utils]")
{
  double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> v = { 0.0, nan, nan, 6.0 };
  auto result = dtwc::interpolate_linear(v);
  REQUIRE_THAT(result[0], WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(result[1], WithinAbs(2.0, 1e-15));
  REQUIRE_THAT(result[2], WithinAbs(4.0, 1e-15));
  REQUIRE_THAT(result[3], WithinAbs(6.0, 1e-15));
}

TEST_CASE("interpolate_linear: leading NaN filled with NOCB (first valid value)", "[missing_utils]")
{
  double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> v = { nan, nan, 3.0, 4.0 };
  auto result = dtwc::interpolate_linear(v);
  REQUIRE_THAT(result[0], WithinAbs(3.0, 1e-15));
  REQUIRE_THAT(result[1], WithinAbs(3.0, 1e-15));
  REQUIRE_THAT(result[2], WithinAbs(3.0, 1e-15));
  REQUIRE_THAT(result[3], WithinAbs(4.0, 1e-15));
}

TEST_CASE("interpolate_linear: trailing NaN filled with LOCF (last valid value)", "[missing_utils]")
{
  double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> v = { 1.0, 2.0, nan, nan };
  auto result = dtwc::interpolate_linear(v);
  REQUIRE_THAT(result[0], WithinAbs(1.0, 1e-15));
  REQUIRE_THAT(result[1], WithinAbs(2.0, 1e-15));
  REQUIRE_THAT(result[2], WithinAbs(2.0, 1e-15));
  REQUIRE_THAT(result[3], WithinAbs(2.0, 1e-15));
}

TEST_CASE("interpolate_linear: all NaN throws runtime_error", "[missing_utils]")
{
  double nan = std::numeric_limits<double>::quiet_NaN();
  std::vector<double> v = { nan, nan, nan };
  REQUIRE_THROWS_AS(dtwc::interpolate_linear(v), std::runtime_error);
}

TEST_CASE("interpolate_linear: empty vector returns empty", "[missing_utils]")
{
  std::vector<double> v;
  auto result = dtwc::interpolate_linear(v);
  REQUIRE(result.empty());
}
