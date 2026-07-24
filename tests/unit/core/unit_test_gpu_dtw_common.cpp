/**
 * @file unit_test_gpu_dtw_common.cpp
 * @brief Host-only contract tests for shared GPU result normalization.
 */

#include <core/gpu_dtw_common.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>

TEST_CASE("GPU public distance normalization preserves values and exact sentinel",
          "[gpu][host][F12]")
{
  using dtwc::gpu::detail::normalize_public_distance;
  constexpr auto float_max = std::numeric_limits<float>::max();
  constexpr auto double_max = std::numeric_limits<double>::max();

  STATIC_REQUIRE(normalize_public_distance(float_max) == double_max);
  STATIC_REQUIRE(normalize_public_distance(double_max) == double_max);
  STATIC_REQUIRE(normalize_public_distance(1.25f) == 1.25);
  STATIC_REQUIRE(normalize_public_distance(-3.5) == -3.5);

  const float below_float_max = std::nextafter(float_max, 0.0f);
  REQUIRE(normalize_public_distance(below_float_max)
          == static_cast<double>(below_float_max));
  REQUIRE(normalize_public_distance(below_float_max) != double_max);

  REQUIRE(std::isinf(normalize_public_distance(
      std::numeric_limits<float>::infinity())));
  REQUIRE(std::isnan(normalize_public_distance(
      std::numeric_limits<float>::quiet_NaN())));
}
