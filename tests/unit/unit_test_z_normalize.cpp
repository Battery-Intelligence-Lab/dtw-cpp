/**
 * @file unit_test_z_normalize.cpp
 * @brief Unit tests for z-normalization of time series.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include <core/z_normalize.hpp>
#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

namespace verify {

inline double mean(const std::vector<data_t> &v)
{
  if (v.empty()) return 0.0;
  return std::accumulate(v.begin(), v.end(), 0.0) / static_cast<double>(v.size());
}

// Population stddev (divides by N, not N-1).
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

TEST_CASE("z_normalize known values: {2,4,4,4,5,5,7,9}", "[z_normalize]")
{
  std::vector<data_t> v{ 2, 4, 4, 4, 5, 5, 7, 9 };
  // mean = 5, population stddev = 2
  dtwc::core::z_normalize(v.data(), v.size());

  std::vector<double> expected{ -1.5, -0.5, -0.5, -0.5, 0.0, 0.0, 1.0, 2.0 };
  REQUIRE(v.size() == expected.size());
  for (size_t i = 0; i < v.size(); ++i)
    REQUIRE_THAT(v[i], WithinAbs(expected[i], 1e-10));
}

TEST_CASE("z_normalize of constant series produces all zeros", "[z_normalize]")
{
  std::vector<data_t> v{ 7.0, 7.0, 7.0, 7.0 };
  dtwc::core::z_normalize(v.data(), v.size());

  for (auto x : v)
    REQUIRE_THAT(x, WithinAbs(0.0, 1e-15));
}

TEST_CASE("z_normalize of single-element series produces zero", "[z_normalize]")
{
  std::vector<data_t> v{ 42.0 };
  dtwc::core::z_normalize(v.data(), v.size());

  REQUIRE_THAT(v[0], WithinAbs(0.0, 1e-15));
}

TEST_CASE("z_normalized returns copy without modifying original", "[z_normalize]")
{
  const std::vector<data_t> expected_original{ 2, 4, 4, 4, 5, 5, 7, 9 };
  std::vector<data_t> original = expected_original;

  auto normed = dtwc::core::z_normalized(original.data(), original.size());

  REQUIRE(original == expected_original);
  REQUIRE(normed.size() == original.size());
  REQUIRE_THAT(normed[0], WithinAbs(-1.5, 1e-10));
}

TEST_CASE("z_normalize of empty series is a no-op", "[z_normalize]")
{
  std::vector<data_t> v;
  REQUIRE_NOTHROW(dtwc::core::z_normalize(v.data(), v.size()));
  REQUIRE(v.empty());
}

// Large-range series: numeric stability.
TEST_CASE("z_normalize handles large-range values", "[z_normalize]")
{
  std::vector<data_t> v{ 1e6, 2e6, 3e6, 4e6, 5e6 };
  dtwc::core::z_normalize(v.data(), v.size());

  REQUIRE_THAT(verify::mean(v), WithinAbs(0.0, 1e-6));
  REQUIRE_THAT(verify::stddev(v), WithinAbs(1.0, 1e-6));
}

// A spread far below the 1e-10 stddev floor is treated as constant.
TEST_CASE("z_normalize treats near-zero variance as constant", "[z_normalize]")
{
  std::vector<data_t> v(64);
  for (std::size_t i = 0; i < v.size(); ++i)
    v[i] = 1.0 + static_cast<double>(i) * 1e-16;
  dtwc::core::z_normalize(v.data(), v.size());

  for (const auto x : v)
    REQUIRE_THAT(x, WithinAbs(0.0, 1e-10));
}

// Negative values: mean 0, population stddev sqrt(50).
TEST_CASE("z_normalize of negative and positive values", "[z_normalize]")
{
  std::vector<data_t> v{ -10, -5, 0, 5, 10 };
  dtwc::core::z_normalize(v.data(), v.size());

  const double s = std::sqrt(50.0);
  const std::vector<double> expected{ -10 / s, -5 / s, 0.0, 5 / s, 10 / s };
  for (std::size_t i = 0; i < v.size(); ++i)
    REQUIRE_THAT(v[i], WithinAbs(expected[i], 1e-12));
}

// A series holding NaN or inf is refused, not set to zeros.
TEST_CASE("z_normalize refuses a series holding NaN or inf", "[z_normalize]")
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();
  for (std::vector<data_t> v : { std::vector<data_t>{ 1.0, nan, 3.0 }, std::vector<data_t>{ 1.0, inf, 3.0 },
                                 std::vector<data_t>{ nan } })
    CHECK_THROWS_AS(dtwc::core::z_normalize(v.data(), v.size()), dtwc::InvalidInput);
}
