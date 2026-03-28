/**
 * @file unit_test_z_normalize.cpp
 * @brief Unit tests for z-normalization of time series.
 *
 * These tests exercise the production dtwc::core::z_normalize() and
 * dtwc::core::z_normalized() functions from dtwc/core/z_normalize.hpp.
 *
 * @date 28 Mar 2026
 */

#include <core/z_normalize.hpp>
#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>
#include <cmath>
#include <numeric>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

// ---------------------------------------------------------------------------
// Local helpers for post-condition verification only.
// ---------------------------------------------------------------------------
namespace verify {

inline double mean(const std::vector<data_t> &v)
{
  if (v.empty()) return 0.0;
  return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

// Uses population stddev (divides by N, not N-1).
inline double stddev(const std::vector<data_t> &v)
{
  if (v.size() <= 1) return 0.0;
  double m = mean(v);
  double sq_sum = 0.0;
  for (auto x : v)
    sq_sum += (x - m) * (x - m);
  return std::sqrt(sq_sum / static_cast<double>(v.size()));
}

} // namespace verify


// ---------------------------------------------------------------------------
// 1. Known values
// ---------------------------------------------------------------------------
TEST_CASE("z_normalize known values: {2,4,4,4,5,5,7,9}", "[Phase1][z_normalize]")
{
  std::vector<data_t> v{ 2, 4, 4, 4, 5, 5, 7, 9 };
  // mean = 5, population stddev = 2
  dtwc::core::z_normalize(v.data(), v.size());

  std::vector<double> expected{ -1.5, -0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 2.0 };
  REQUIRE(v.size() == expected.size());
  for (size_t i = 0; i < v.size(); ++i) {
    REQUIRE_THAT(v[i], WithinAbs(expected[i], 1e-10));
  }
}

// ---------------------------------------------------------------------------
// 2. After z_normalize, mean should be ~0
// ---------------------------------------------------------------------------
TEST_CASE("z_normalized series has mean ~0", "[Phase1][z_normalize]")
{
  std::vector<data_t> v{ 10, 20, 30, 40, 50 };
  dtwc::core::z_normalize(v.data(), v.size());

  double m = verify::mean(v);
  REQUIRE_THAT(m, WithinAbs(0.0, 1e-10));
}

// ---------------------------------------------------------------------------
// 3. After z_normalize, stddev should be ~1
// ---------------------------------------------------------------------------
TEST_CASE("z_normalized series has stddev ~1", "[Phase1][z_normalize]")
{
  std::vector<data_t> v{ 10, 20, 30, 40, 50 };
  dtwc::core::z_normalize(v.data(), v.size());

  double s = verify::stddev(v);
  REQUIRE_THAT(s, WithinAbs(1.0, 1e-10));
}

// ---------------------------------------------------------------------------
// 4. Constant series (stddev==0) should become all zeros
// ---------------------------------------------------------------------------
TEST_CASE("z_normalize of constant series produces all zeros", "[Phase1][z_normalize]")
{
  std::vector<data_t> v{ 7.0, 7.0, 7.0, 7.0 };
  dtwc::core::z_normalize(v.data(), v.size());

  for (auto x : v) {
    REQUIRE_THAT(x, WithinAbs(0.0, 1e-15));
  }
}

// ---------------------------------------------------------------------------
// 5. Single-element series should become zero
// ---------------------------------------------------------------------------
TEST_CASE("z_normalize of single-element series produces zero", "[Phase1][z_normalize]")
{
  std::vector<data_t> v{ 42.0 };
  dtwc::core::z_normalize(v.data(), v.size());

  REQUIRE_THAT(v[0], WithinAbs(0.0, 1e-15));
}

// ---------------------------------------------------------------------------
// 6. z_normalized() returns a copy, does not modify original
// ---------------------------------------------------------------------------
TEST_CASE("z_normalized returns copy without modifying original", "[Phase1][z_normalize]")
{
  std::vector<data_t> original{ 2, 4, 4, 4, 5, 5, 7, 9 };
  std::vector<data_t> saved = original;

  auto normed = dtwc::core::z_normalized(original.data(), original.size());

  // Original should be unchanged.
  REQUIRE(original.size() == saved.size());
  for (size_t i = 0; i < original.size(); ++i) {
    REQUIRE_THAT(original[i], WithinAbs(saved[i], 1e-15));
  }

  // Normed should be different from original (since original is not already normalized).
  bool any_different = false;
  for (size_t i = 0; i < normed.size(); ++i) {
    if (std::abs(normed[i] - original[i]) > 1e-10) {
      any_different = true;
      break;
    }
  }
  REQUIRE(any_different);
}

// ---------------------------------------------------------------------------
// 7. Empty series: z_normalize is a no-op
// ---------------------------------------------------------------------------
TEST_CASE("z_normalize of empty series is a no-op", "[Phase1][z_normalize]")
{
  std::vector<data_t> v;
  REQUIRE_NOTHROW(dtwc::core::z_normalize(v.data(), v.size()));
  REQUIRE(v.empty());
}

// ---------------------------------------------------------------------------
// 8. Two-element series
// ---------------------------------------------------------------------------
TEST_CASE("z_normalize of two-element series", "[Phase1][z_normalize]")
{
  std::vector<data_t> v{ 3.0, 5.0 };
  // mean = 4, population stddev = 1
  dtwc::core::z_normalize(v.data(), v.size());

  REQUIRE_THAT(v[0], WithinAbs(-1.0, 1e-10));
  REQUIRE_THAT(v[1], WithinAbs(1.0, 1e-10));
}

// ---------------------------------------------------------------------------
// 9. Large-range series: verify numeric stability
// ---------------------------------------------------------------------------
TEST_CASE("z_normalize handles large-range values", "[Phase1][z_normalize]")
{
  std::vector<data_t> v{ 1e6, 2e6, 3e6, 4e6, 5e6 };
  dtwc::core::z_normalize(v.data(), v.size());

  double m = verify::mean(v);
  double s = verify::stddev(v);

  REQUIRE_THAT(m, WithinAbs(0.0, 1e-6));
  REQUIRE_THAT(s, WithinAbs(1.0, 1e-6));
}
