/**
 * @file unit_test_z_normalize.cpp
 * @brief Unit tests for z-normalization utility.
 *
 * @date 28 Mar 2026
 */

#include <core/z_normalize.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <numeric>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc::core;

TEST_CASE("z_normalize known values", "[z_normalize]")
{
  // Series: {2, 4, 4, 4, 5, 5, 7, 9}
  // Mean = 5.0, population stddev = 2.0
  std::vector<double> series = { 2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0 };
  z_normalize(series.data(), series.size());

  REQUIRE_THAT(series[0], WithinAbs(-1.5, 1e-10));   // (2 - 5) / 2
  REQUIRE_THAT(series[1], WithinAbs(-0.5, 1e-10));   // (4 - 5) / 2
  REQUIRE_THAT(series[4], WithinAbs(0.0, 1e-10));    // (5 - 5) / 2
  REQUIRE_THAT(series[7], WithinAbs(2.0, 1e-10));    // (9 - 5) / 2
}

TEST_CASE("z_normalize results in zero mean and unit stddev", "[z_normalize]")
{
  std::vector<double> series = { 1.0, 3.0, 5.0, 7.0, 9.0, 11.0 };
  z_normalize(series.data(), series.size());

  // Compute mean
  double mean = 0.0;
  for (auto v : series) mean += v;
  mean /= static_cast<double>(series.size());
  REQUIRE_THAT(mean, WithinAbs(0.0, 1e-10));

  // Compute population standard deviation
  double sq_sum = 0.0;
  for (auto v : series) sq_sum += v * v;
  double stddev = std::sqrt(sq_sum / static_cast<double>(series.size()));
  REQUIRE_THAT(stddev, WithinAbs(1.0, 1e-10));
}

TEST_CASE("z_normalize constant series (zero stddev)", "[z_normalize]")
{
  std::vector<double> series = { 5.0, 5.0, 5.0, 5.0 };
  z_normalize(series.data(), series.size());

  for (auto v : series)
    REQUIRE(v == 0.0);
}

TEST_CASE("z_normalize single element series", "[z_normalize]")
{
  std::vector<double> series = { 42.0 };
  z_normalize(series.data(), series.size());

  // Single element: n <= 1, should be unchanged
  REQUIRE(series[0] == 42.0);
}

TEST_CASE("z_normalize empty series", "[z_normalize]")
{
  std::vector<double> series;
  // Should not crash
  z_normalize(series.data(), series.size());
  REQUIRE(series.empty());
}

TEST_CASE("z_normalized returns copy", "[z_normalize]")
{
  std::vector<double> original = { 2.0, 4.0, 6.0, 8.0 };
  auto normalized = z_normalized(original.data(), original.size());

  // Original should be unchanged
  REQUIRE(original[0] == 2.0);
  REQUIRE(original[3] == 8.0);

  // Normalized copy should have zero mean
  double mean = 0.0;
  for (auto v : normalized) mean += v;
  mean /= static_cast<double>(normalized.size());
  REQUIRE_THAT(mean, WithinAbs(0.0, 1e-10));
}

TEST_CASE("z_normalize with float type", "[z_normalize]")
{
  std::vector<float> series = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
  z_normalize(series.data(), series.size());

  float mean = 0.0f;
  for (auto v : series) mean += v;
  mean /= static_cast<float>(series.size());
  REQUIRE_THAT(static_cast<double>(mean), WithinAbs(0.0, 1e-5));
}

TEST_CASE("z_normalize near-zero stddev", "[z_normalize]")
{
  // Very tiny variation, below the 1e-10 threshold
  std::vector<double> series = { 1.0, 1.0 + 1e-15, 1.0 - 1e-15 };
  z_normalize(series.data(), series.size());

  // Should be treated as constant series
  for (auto v : series)
    REQUIRE(v == 0.0);
}
