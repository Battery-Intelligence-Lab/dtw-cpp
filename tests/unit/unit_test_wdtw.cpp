/**
 * @file unit_test_wdtw.cpp
 * @brief Unit tests for Weighted DTW (WDTW) wrappers and weight-span checks.
 *
 * Reference: Jeong, Y.-S., Jeong, M. K., & Omitaomu, O. A. (2011).
 *   Weighted dynamic time warping for time series classification.
 *   Pattern Recognition, 44(9), 2231-2240.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>
#include <span>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

TEST_CASE("wdtwBanded, band -1: g=0 gives weights all 0.5, hand-computed example", "[wdtw]")
{
  // With g=0, logistic w(k) = 1/(1+exp(0)) = 0.5 for all k. The full DP:
  //   C(0,0) = 0.5 * |1-2| = 0.5
  //   C(1,0) = C(0,0) + 0.5 * |2-2| = 0.5
  //   C(2,0) = C(1,0) + 0.5 * |3-2| = 1.0
  //   C(0,1) = C(0,0) + 0.5 * |1-3| = 1.5
  //   C(0,2) = C(0,1) + 0.5 * |1-4| = 3.0
  //   C(1,1) = min(1.5, 0.5, 0.5) + 0.5*|2-3| = 0.5 + 0.5 = 1.0
  //   C(2,1) = min(1.0, 1.0, 0.5) + 0.5*|3-3| = 0.5 + 0.0 = 0.5
  //   C(1,2) = min(3.0, 1.0, 1.5) + 0.5*|2-4| = 1.0 + 1.0 = 2.0
  //   C(2,2) = min(2.0, 0.5, 1.0) + 0.5*|3-4| = 0.5 + 0.5 = 1.0
  std::vector<double> x{ 1, 2, 3 }, y{ 2, 3, 4 };
  REQUIRE_THAT(wdtwBanded(x, y, -1, 0.0), WithinAbs(1.0, 1e-12));
}

TEST_CASE("wdtwBanded, band -1: empty series returns maxValue", "[wdtw]")
{
  constexpr double max_value = std::numeric_limits<double>::max();
  std::vector<double> x{ 1, 2, 3 }, empty{};

  REQUIRE(wdtwBanded(x, empty, -1, 0.05) == max_value);
  REQUIRE(wdtwBanded(empty, x, -1, 0.05) == max_value);
  REQUIRE(wdtwBanded(empty, empty, -1, 0.05) == max_value);
}

// SpanWeightedL1Cost indexes weights[|row - col|], whose maximum is
// max(nx, ny) - 1. A shorter array would be read out of bounds and return a
// plausible-looking number, so every weights-taking overload must refuse it.
TEST_CASE("wdtw: a weight array shorter than max(|x|, |y|) is rejected", "[wdtw][precondition]")
{
  const std::vector<double> x = { 1, 2, 3, 4, 5 };
  const std::vector<double> y = { 5, 4, 3, 2, 1 };
  const std::vector<double> short_w = { 1.0, 1.0 };            // needs 5
  const auto full_w = wdtw_weights<double>(4, 0.05);           // max_dev = 4 -> size 5

  const std::span<const double> sw{ short_w };
  const std::span<const double> fw{ full_w };

  CHECK_THROWS_AS(wdtwBanded(x, y, sw), InvalidInput);
  CHECK_THROWS_AS(wdtwBanded(x, y, sw, 2), InvalidInput);
  CHECK_NOTHROW(wdtwBanded(x, y, fw));
  CHECK_NOTHROW(wdtwBanded(x, y, fw, 2));

  // Multivariate: the bound is in timesteps, not flat elements.
  const std::vector<double> xm = { 1, 10, 2, 20, 3, 30 };  // 3 steps x 2 channels
  const std::vector<double> ym = { 3, 30, 2, 20, 1, 10 };
  const auto short_mv = wdtw_weights<double>(1, 0.05);     // size 2, needs 3
  const auto full_mv = wdtw_weights<double>(2, 0.05);      // size 3
  const std::span<const double> smv{ short_mv };
  const std::span<const double> fmv{ full_mv };

  CHECK_THROWS_AS(wdtwBanded_mv<double>(xm.data(), 3, ym.data(), 3, 2, smv), InvalidInput);
  CHECK_THROWS_AS(wdtwBanded_mv<double>(xm.data(), 3, ym.data(), 3, 2, smv, 1), InvalidInput);
  CHECK_NOTHROW(wdtwBanded_mv<double>(xm.data(), 3, ym.data(), 3, 2, fmv));
  CHECK_NOTHROW(wdtwBanded_mv<double>(xm.data(), 3, ym.data(), 3, 2, fmv, 1));
}
