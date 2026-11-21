/**
 * @file unit_test_missing_dtw.cpp
 * @brief Unit tests for DTW with missing data (ZeroCost and Interpolate).
 *
 * @details Tests for dtwMissing_L and dtwMissing_banded
 * which handle NaN values in time series by treating missing pairs as zero cost.
 *
 * Reference: Yurtman, Soenen, Meert & Blockeel (2023), "Estimating DTW Distance
 *            Between Time Series with Missing Data", ECML-PKDD 2023, LNCS 14173.
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 */

#include <dtwc.hpp>

#include "../support/missing_dtw_oracle.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <limits>
#include <utility>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

TEST_CASE("dtwMissing_L: identical series gives zero", "[missing_dtw]")
{
  std::vector<double> x{ 1, 2, 3, 4, 5 };
  REQUIRE_THAT(dtwMissing_L<double>(x, x), WithinAbs(0.0, 1e-15));
}

TEST_CASE("dtwMissing_L: single-element series", "[missing_dtw]")
{
  std::vector<double> x{ 5.0 };
  std::vector<double> y{ 3.0 };
  std::vector<double> y_nan{ NaN };

  REQUIRE_THAT(dtwMissing_L<double>(x, y), WithinAbs(2.0, 1e-12));
  REQUIRE_THAT(dtwMissing_L<double>(x, y_nan), WithinAbs(0.0, 1e-15));
}

TEST_CASE("dtwMissing_L: early abandon triggers correctly", "[missing_dtw]")
{
  std::vector<double> x{ 1, 2, 3, 4, 5 };
  std::vector<double> y{ 100, 200, 300, 400, 500 };
  constexpr double maxValue = std::numeric_limits<double>::max();

  // With a small threshold, early abandon should kick in
  const auto result = dtwMissing_L<double>(x, y, 10.0);
  REQUIRE(result == maxValue);

  // With no threshold, should return the actual distance
  const auto full = dtwMissing_L<double>(x, y, -1.0);
  REQUIRE(full < maxValue);
}

TEST_CASE("dtwMissing_L: hand-computed nonzero result with NaN", "[missing_dtw]")
{
  // x = {1, 2}, y = {NaN, 5}
  // C(0,0) = 0 (y[0]=NaN)
  // C(1,0) = 0 + 0 = 0 (y[0]=NaN)
  // C(0,1) = 0 + |1-5| = 4
  // C(1,1) = min(0, 0, 4) + |2-5| = 0 + 3 = 3
  std::vector<double> x{ 1, 2 };
  std::vector<double> y{ NaN, 5 };

  REQUIRE_THAT(dtwMissing_L<double>(x, y), WithinAbs(3.0, 1e-12));
}

// An all-NaN series under MissingStrategy::Interpolate must be rejected by
// the SERIAL pre-scan, with a diagnostic naming the offending series — not by
// interpolate_linear_into() throwing from inside the parallel per-pair lambda.
TEST_CASE("Interpolate: all-NaN series is rejected by the serial pre-scan",
          "[missing_dtw][interpolate][prescan]")
{
  Data data;
  data.ndim = 1;
  data.p_vec = { { 1, 2, 3, 4 }, { NaN, NaN, NaN, NaN }, { 2, 3, 4, 5 } };
  data.p_names = { "clean", "all_nan", "other" };

  Problem prob;
  prob.set_data(std::move(data));
  prob.set_verbose(false);
  prob.set_missing_strategy(core::MissingStrategy::Interpolate);

  // The message must identify the series, which only the serial pre-scan can do
  // (the per-pair lambda sees two anonymous spans).
  REQUIRE_THROWS_WITH(
    prob.fill_distance_matrix(),
    Catch::Matchers::ContainsSubstring("all_nan")
      && Catch::Matchers::ContainsSubstring("index 1"));
}

TEST_CASE("ZeroCost kernels equal the recurrence for every NaN pattern, in both argument orders",
          "[missing_dtw][oracle]")
{
  namespace ts = dtwc::test_support;
  std::size_t pairs = 0;
  ts::for_each_masked_pair([&](const std::vector<double>& x, const std::vector<double>& y) {
    ++pairs;
    CAPTURE(x, y);
    for (const bool squared : { false, true }) {
      const auto metric = squared ? core::MetricType::SquaredL2 : core::MetricType::L1;
      const auto want = [&](int band) {
        return ts::missing_dtw_oracle(ts::MissingRule::ZeroCost, x, y, squared, band);
      };
      for (const bool swapped : { false, true }) {
        const auto& a = swapped ? y : x;
        const auto& b = swapped ? x : y;
        CHECK(dtwMissing_L<double>(a, b, -1, metric) == want(-1));
        // The masked pairs' best paths stay within one step of the diagonal: only band 0 binds.
        for (const int band : { -1, 0, 2, 100 })
          CHECK(ts::same_distance(dtwMissing_banded<double>(a, b, band, -1, metric), want(band)));
      }
    }
  });
  CHECK(pairs > 500);
}

TEST_CASE("dtwMissing_L SquaredL2: a missing last step costs nothing, hand-computed", "[missing_dtw]")
{
  // x = {3, NaN}, y = {1, 2}
  // C(0,0) = (3-1)^2 = 4
  // C(1,0) = 4 + 0 = 4      [x[1] is NaN]
  // C(0,1) = 4 + (3-2)^2 = 5
  // C(1,1) = min(4, 4, 5) + 0 = 4   [x[1] is NaN]
  std::vector<double> x = { 3.0, NaN };
  std::vector<double> y = { 1.0, 2.0 };
  REQUIRE_THAT(dtwMissing_L<double>(x, y, -1, core::MetricType::SquaredL2), WithinAbs(4.0, 1e-12));
}

TEST_CASE("dtwMissing_L: an empty series gives the no-path sentinel", "[missing_dtw]")
{
  std::vector<double> x{ 1.0, 2.0 };
  std::vector<double> empty{};
  constexpr double max_value = std::numeric_limits<double>::max();

  REQUIRE(dtwMissing_L<double>(empty, x) == max_value);
  REQUIRE(dtwMissing_L<double>(x, empty) == max_value);
  REQUIRE(dtwMissing_banded<double>(x, empty, 2) == max_value);
  REQUIRE(dtwMissing_banded<double>(empty, x, 2) == max_value);
}
