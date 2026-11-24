/**
 * @file unit_test_arow_dtw.cpp
 * @brief Unit tests for DTW-AROW (diagonal-only alignment for missing values).
 *
 * Reference: Yurtman, A., Soenen, J., Meert, W. & Blockeel, H. (2023).
 *            "Estimating DTW Distance Between Time Series with Missing Data."
 *            ECML-PKDD 2023, LNCS 14173.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#include <dtwc.hpp>

#include "../support/missing_dtw_oracle.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>
#include <utility>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

TEST_CASE("dtwAROW_banded, band -1: hand-computed, nonzero with NaN", "[arow_dtw]")
{
  // x = {1, 2}, y = {NaN, 5}
  // C(0,0): y[0]=NaN -> 0
  // C(1,0): y[0]=NaN -> C(0,0)=0 (boundary: propagate)
  // C(0,1): y[1]=5, x[0]=1 -> C(0,0)+|1-5|=4
  // C(1,1): both present: min(C(0,0),C(0,1),C(1,0))+|2-5|=min(0,4,0)+3=3
  // Result = 3
  std::vector<double> x{ 1, 2 };
  std::vector<double> y{ NaN, 5 };

  REQUIRE_THAT(dtwAROW_banded(x, y), WithinAbs(3.0, 1e-12));
}

TEST_CASE("dtwAROW_banded, band -1: empty vectors return maxValue", "[arow_dtw][edge]")
{
  std::vector<double> x{ 1, 2, 3 };
  std::vector<double> empty{};
  constexpr double maxValue = std::numeric_limits<double>::max();

  REQUIRE(dtwAROW_banded(x, empty) == maxValue);
  REQUIRE(dtwAROW_banded(empty, x) == maxValue);
}

TEST_CASE("dtwAROW_banded, band -1: identical series gives zero", "[arow_dtw][edge]")
{
  std::vector<double> x{ 1, 2, 3, 4, 5 };
  REQUIRE_THAT(dtwAROW_banded(x, x), WithinAbs(0.0, 1e-15));
}

TEST_CASE("dtwAROW_banded, band -1: single-element series", "[arow_dtw][edge]")
{
  std::vector<double> x{ 5.0 };
  std::vector<double> y{ 3.0 };
  std::vector<double> y_nan{ NaN };

  REQUIRE_THAT(dtwAROW_banded(x, y), WithinAbs(2.0, 1e-12));
  REQUIRE_THAT(dtwAROW_banded(x, y_nan), WithinAbs(0.0, 1e-15));
}

TEST_CASE("MV AROW: per-channel skip when one channel is NaN", "[arow_dtw][mv]")
{
  // x and y match on channel 1; channel 0 of x has a NaN at step 2.
  // Per-channel-skip: the NaN channel contributes 0, channel 1 pair is
  // comparable so AROW is NOT triggered at that step. Distance should
  // equal Standard DTW distance computed on channel 1 only (which is 0,
  // since channel 1 matches between x and y).
  dtwc::Data data;
  data.ndim = 2;
  data.p_vec = {
    //  t0    t1    t2       t3
    // ch0 ch1 ch0 ch1 ch0 ch1 ch0 ch1
    {  1,  5,  2,  6, NaN, 7,   4,  8 },   // x
    {  1,  5,  2,  6,  3,  7,   4,  8 },   // y
  };
  data.p_names = { "a", "b" };
  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.set_missing_strategy(dtwc::core::MissingStrategy::AROW);
  prob.fill_distance_matrix();
  // Sum of per-channel L1 = 0 at every step (channel 1 matches everywhere;
  // channel 0 matches except at step 2 where one side is NaN, per-channel
  // skip -> 0). Final distance = 0.
  REQUIRE_THAT(prob.dist_by_ind(0, 1), WithinAbs(0.0, 1e-10));
}

TEST_CASE("MV AROW: all channels NaN at a step triggers diagonal carry", "[arow_dtw][mv]")
{
  // At step 2, BOTH channels of x are NaN. Per-channel-skip on the pair
  // yields 0 comparable channels -> cost returns NaN -> AROWCell triggers
  // diagonal carry. Distance should therefore NOT accumulate |x-y| at step
  // 2, but propagate from step 1.
  const double nan = NaN;
  dtwc::Data data;
  data.ndim = 2;
  data.p_vec = {
    //  t0    t1      t2       t3
    { 1, 5,  2, 6, nan, nan, 100, 800 },  // x — step 2 fully missing, step 3 very different
    { 1, 5,  2, 6,  3,   7,    3,   7 },  // y — step 2 present, step 3 matches step 2
  };
  data.p_names = { "a", "b" };
  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.set_missing_strategy(dtwc::core::MissingStrategy::AROW);
  prob.fill_distance_matrix();
  const double d_with_missing = prob.dist_by_ind(0, 1);
  REQUIRE(d_with_missing >= 0.0);
  REQUIRE(d_with_missing < std::numeric_limits<double>::max() / 2.0);
  // Sanity: identical series yields 0; this series has a drastic mismatch at
  // step 3 even after AROW skips step 2, so the distance must be strictly > 0.
  REQUIRE(d_with_missing > 0.0);
}

TEST_CASE("Problem::set_missing_strategy(AROW) selects the AROW kernel for the fill",
          "[arow_dtw][problem][setter]")
{
  const std::vector<double> x{ 1.0, 2.0, 3.0 };
  const std::vector<double> y_nan{ 1.0, NaN, 3.0 };

  dtwc::Data data;
  data.p_vec   = { x, y_nan };
  data.p_names = { "x", "y_nan" };

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.set_verbose(false);
  REQUIRE_NOTHROW(prob.set_missing_strategy(dtwc::core::MissingStrategy::AROW));
  CHECK(prob.missing_strategy() == dtwc::core::MissingStrategy::AROW);
  prob.fill_distance_matrix();

  // The Problem's bound kernel and the free function are separate code paths.
  REQUIRE_THAT(prob.dist_by_ind(0, 1), WithinAbs(dtwAROW_banded(x, y_nan), 1e-12));
}

TEST_CASE("DTW-AROW kernels equal the recurrence for every NaN pattern, in both argument orders",
          "[arow_dtw][oracle]")
{
  namespace ts = dtwc::test_support;
  std::size_t pairs = 0;
  ts::for_each_masked_pair([&](const std::vector<double>& x, const std::vector<double>& y) {
    ++pairs;
    CAPTURE(x, y);
    for (const bool squared : { false, true }) {
      const auto metric = squared ? core::MetricType::SquaredL2 : core::MetricType::L1;
      const auto want = [&](int band) {
        return ts::missing_dtw_oracle(ts::MissingRule::AROW, x, y, squared, band);
      };
      // AROW admits fewer paths than ZeroCost, so it is never below it.
      const double zero_cost = ts::missing_dtw_oracle(ts::MissingRule::ZeroCost, x, y, squared);
      for (const bool swapped : { false, true }) {
        const auto& a = swapped ? y : x;
        const auto& b = swapped ? x : y;
        const double arow = dtwAROW_banded(a, b, -1, metric);
        CHECK(arow == want(-1));
        CHECK(arow >= zero_cost);
        // The masked pairs' best paths stay within one step of the diagonal: only band 0 binds.
        for (const int band : { -1, 0, 2, 100 })
          CHECK(ts::same_distance(dtwAROW_banded(a, b, band, metric), want(band)));
      }
    }
  });
  CHECK(pairs > 500);
}
