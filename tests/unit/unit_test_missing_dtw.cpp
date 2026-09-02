/**
 * @file unit_test_missing_dtw.cpp
 * @brief Unit tests for DTW with missing data (NaN-aware, DTW-AROW).
 *
 * @details Tests for dtwMissing, dtwMissing_L, and dtwMissing_banded functions
 * which handle NaN values in time series by treating missing pairs as zero cost.
 *
 * Reference: Yurtman, Soenen, Meert & Blockeel (2023), "Estimating DTW Distance
 *            Between Time Series with Missing Data", ECML-PKDD 2023, LNCS 14173.
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <limits>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

// ===========================================================================
// dtwMissing_L — no missing data (should match standard DTW)
// ===========================================================================

TEST_CASE("dtwMissing_L: no NaN matches standard DTW", "[missing_dtw]")
{
  std::vector<double> x{ 1, 2, 3, 4, 5 };
  std::vector<double> y{ 2, 3, 4 };

  const auto missing_result = dtwMissing_L<double>(x, y);
  const auto standard_result = dtwFull_L<double>(x, y);

  REQUIRE_THAT(missing_result, WithinAbs(standard_result, 1e-12));
}

TEST_CASE("dtwMissing_L: no NaN matches standard DTW (equal length)", "[missing_dtw]")
{
  std::vector<double> x{ 1, 3, 5 };
  std::vector<double> y{ 2, 4, 6 };

  const auto missing_result = dtwMissing_L<double>(x, y);
  const auto standard_result = dtwFull_L<double>(x, y);

  REQUIRE_THAT(missing_result, WithinAbs(standard_result, 1e-12));
}

TEST_CASE("dtwMissing_L: no NaN matches standard DTW (SquaredL2)", "[missing_dtw]")
{
  std::vector<double> x{ 1, 2, 3 };
  std::vector<double> y{ 2, 4, 5 };

  const auto missing_result = dtwMissing_L<double>(x, y, -1, core::MetricType::SquaredL2);
  const auto standard_result = dtwFull_L<double>(x, y, -1, core::MetricType::SquaredL2);

  REQUIRE_THAT(missing_result, WithinAbs(standard_result, 1e-12));
}

// ===========================================================================
// dtwMissing_L — all missing data
// ===========================================================================

TEST_CASE("dtwMissing_L: all NaN in both series gives zero", "[missing_dtw]")
{
  std::vector<double> x{ NaN, NaN, NaN };
  std::vector<double> y{ NaN, NaN };

  REQUIRE_THAT(dtwMissing_L<double>(x, y), WithinAbs(0.0, 1e-15));
}

TEST_CASE("dtwMissing_L: one series entirely NaN gives zero", "[missing_dtw]")
{
  std::vector<double> x{ 1, 2, 3, 4, 5 };
  std::vector<double> y_nan{ NaN, NaN, NaN };

  REQUIRE_THAT(dtwMissing_L<double>(x, y_nan), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(dtwMissing_L<double>(y_nan, x), WithinAbs(0.0, 1e-15));
}

// ===========================================================================
// dtwMissing_L — partial missing data
// ===========================================================================

TEST_CASE("dtwMissing_L: single missing at start of x", "[missing_dtw]")
{
  // x = {NaN, 2, 3}, y = {1, 2, 3}
  // cost(NaN, 1) = 0
  // Remaining path through present values should give less than or equal to
  // standard DTW on the original
  std::vector<double> x{ NaN, 2, 3 };
  std::vector<double> y{ 1, 2, 3 };

  const auto result = dtwMissing_L<double>(x, y);
  REQUIRE(result >= 0.0);
  // With NaN at x[0], cost(x[0], y[j]) = 0 for any j.
  // Hand-compute: C(0,0)=0, C(1,0)=0+|2-1|=1, C(2,0)=1+|3-1|=3
  // C(0,1)=0+0=0, C(0,2)=0+0=0
  // C(1,1)=min(0,1,0)+|2-2|=0, C(1,2)=min(0,0,0)+|2-3|=1
  // C(2,1)=min(0,1,0)+|3-2|=1, C(2,2)=min(0,1,1)+|3-3|=0
  REQUIRE_THAT(result, WithinAbs(0.0, 1e-12));
}

TEST_CASE("dtwMissing_L: single missing in the middle", "[missing_dtw]")
{
  // x = {1, NaN, 3}, y = {1, 2, 3}
  std::vector<double> x{ 1, NaN, 3 };
  std::vector<double> y{ 1, 2, 3 };

  const auto result = dtwMissing_L<double>(x, y);
  REQUIRE(result >= 0.0);
  // Hand-compute:
  // C(0,0)=|1-1|=0, C(1,0)=0+0=0, C(2,0)=0+|3-1|=2
  // C(0,1)=0+|1-2|=1, C(0,2)=1+|1-3|=3
  // C(1,1)=min(0,0,1)+0=0, C(1,2)=min(1,0,0)+0=0
  // C(2,1)=min(0,0,0)+|3-2|=1, C(2,2)=min(0,0,1)+|3-3|=0
  REQUIRE_THAT(result, WithinAbs(0.0, 1e-12));
}

TEST_CASE("dtwMissing_L: single missing at end", "[missing_dtw]")
{
  // x = {1, 2, NaN}, y = {1, 2, 3}
  std::vector<double> x{ 1, 2, NaN };
  std::vector<double> y{ 1, 2, 3 };

  const auto result = dtwMissing_L<double>(x, y);
  REQUIRE(result >= 0.0);
  // C(0,0)=0, C(1,0)=0+|2-1|=1, C(2,0)=1+0=1
  // C(0,1)=0+|1-2|=1, C(0,2)=1+|1-3|=3
  // C(1,1)=min(0,1,1)+|2-2|=0, C(1,2)=min(1,0,0)+|2-3|=1
  // C(2,1)=min(0,1,0)+0=0, C(2,2)=min(0,0,1)+0=0
  REQUIRE_THAT(result, WithinAbs(0.0, 1e-12));
}

TEST_CASE("dtwMissing_L: both series have NaN at different positions", "[missing_dtw]")
{
  // x = {1, NaN, 3, 4}, y = {NaN, 2, 3, 4}
  std::vector<double> x{ 1, NaN, 3, 4 };
  std::vector<double> y{ NaN, 2, 3, 4 };

  const auto result = dtwMissing_L<double>(x, y);
  REQUIRE(result >= 0.0);
  // Symmetry holds since cost function is symmetric
  const auto reverse = dtwMissing_L<double>(y, x);
  REQUIRE_THAT(result, WithinAbs(reverse, 1e-12));
}

// ===========================================================================
// dtwMissing_L — symmetry
// ===========================================================================

TEST_CASE("dtwMissing_L: symmetry with no NaN", "[missing_dtw]")
{
  std::vector<double> x{ 1, 3, 5, 2 };
  std::vector<double> y{ 2, 4, 6 };

  REQUIRE_THAT(dtwMissing_L<double>(x, y), WithinAbs(dtwMissing_L<double>(y, x), 1e-12));
}

TEST_CASE("dtwMissing_L: symmetry with NaN", "[missing_dtw]")
{
  std::vector<double> x{ 1, NaN, 5, 2 };
  std::vector<double> y{ NaN, 4, 6 };

  REQUIRE_THAT(dtwMissing_L<double>(x, y), WithinAbs(dtwMissing_L<double>(y, x), 1e-12));
}

// ===========================================================================
// dtwMissing_L — edge cases
// ===========================================================================

TEST_CASE("dtwMissing_L: empty vectors return maxValue", "[missing_dtw]")
{
  std::vector<double> x{ 1, 2, 3 };
  std::vector<double> empty{};

  REQUIRE(dtwMissing_L<double>(x, empty) > 1e10);
  REQUIRE(dtwMissing_L<double>(empty, x) > 1e10);
}

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

TEST_CASE("dtwMissing_L: non-negativity", "[missing_dtw]")
{
  std::vector<double> x{ 1, NaN, 3 };
  std::vector<double> y{ 5, 6, NaN, 8 };

  REQUIRE(dtwMissing_L<double>(x, y) >= 0.0);
}

// ===========================================================================
// dtwMissing_L — early abandon
// ===========================================================================

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

// ===========================================================================
// dtwMissing — full matrix version
// ===========================================================================

TEST_CASE("dtwMissing: matches dtwMissing_L for no NaN", "[missing_dtw]")
{
  std::vector<double> x{ 1, 2, 3, 4 };
  std::vector<double> y{ 2, 4, 5 };

  const auto full = dtwMissing<double>(x, y);
  const auto light = dtwMissing_L<double>(x, y);

  REQUIRE_THAT(full, WithinAbs(light, 1e-12));
}

TEST_CASE("dtwMissing: matches dtwMissing_L with NaN", "[missing_dtw]")
{
  std::vector<double> x{ 1, NaN, 3, 4 };
  std::vector<double> y{ NaN, 2, 3 };

  const auto full = dtwMissing<double>(x, y);
  const auto light = dtwMissing_L<double>(x, y);

  REQUIRE_THAT(full, WithinAbs(light, 1e-12));
}

TEST_CASE("dtwMissing: all NaN gives zero", "[missing_dtw]")
{
  std::vector<double> x{ NaN, NaN };
  std::vector<double> y{ NaN, NaN, NaN };

  REQUIRE_THAT(dtwMissing<double>(x, y), WithinAbs(0.0, 1e-15));
}

// ===========================================================================
// dtwMissing_banded — banded version
// ===========================================================================

TEST_CASE("dtwMissing_banded: negative band falls back to dtwMissing_L", "[missing_dtw]")
{
  std::vector<double> x{ 1, NaN, 3 };
  std::vector<double> y{ 2, 4, NaN };

  const auto full = dtwMissing_L<double>(x, y);
  const auto banded = dtwMissing_banded<double>(x, y, -1);

  REQUIRE_THAT(banded, WithinAbs(full, 1e-12));
}

TEST_CASE("dtwMissing_banded: large band matches unbanded", "[missing_dtw]")
{
  std::vector<double> x{ 1, NaN, 3, 4, 5 };
  std::vector<double> y{ 2, 4, NaN, 6, 7, 8 };

  const auto full = dtwMissing_L<double>(x, y);
  const auto banded = dtwMissing_banded<double>(x, y, 100);

  REQUIRE_THAT(banded, WithinAbs(full, 1e-12));
}

TEST_CASE("dtwMissing_banded: no NaN matches standard dtwBanded", "[missing_dtw]")
{
  std::vector<double> x{ 1, 2, 3, 4, 5 };
  std::vector<double> y{ 2, 3, 4, 5, 6, 7, 8 };
  int band = 2;

  const auto missing_result = dtwMissing_banded<double>(x, y, band);
  const auto standard_result = dtwBanded<double>(x, y, band);

  REQUIRE_THAT(missing_result, WithinAbs(standard_result, 1e-12));
}

TEST_CASE("dtwMissing_banded: symmetry", "[missing_dtw]")
{
  std::vector<double> x{ 1, NaN, 3, 4, 5 };
  std::vector<double> y{ 2, 4, NaN, 6, 7 };
  int band = 2;

  REQUIRE_THAT(dtwMissing_banded<double>(x, y, band),
               WithinAbs(dtwMissing_banded<double>(y, x, band), 1e-12));
}

TEST_CASE("dtwMissing_banded: all NaN gives zero", "[missing_dtw]")
{
  std::vector<double> x{ NaN, NaN, NaN, NaN, NaN };
  std::vector<double> y{ NaN, NaN, NaN, NaN, NaN };

  REQUIRE_THAT(dtwMissing_banded<double>(x, y, 2), WithinAbs(0.0, 1e-15));
}

TEST_CASE("dtwMissing_banded: empty vectors", "[missing_dtw]")
{
  std::vector<double> x{ 1, 2, 3 };
  std::vector<double> empty{};

  REQUIRE(dtwMissing_banded<double>(x, empty, 2) > 1e10);
  REQUIRE(dtwMissing_banded<double>(empty, x, 2) > 1e10);
}

TEST_CASE("dtwMissing_banded: singleton path requires endpoint offset", "[missing_dtw]")
{
  std::vector<double> x{ 5.0 };
  std::vector<double> y{ 1, 2, 3, 4, 5, 6, 7, 8 };
  constexpr auto max_value = std::numeric_limits<double>::max();

  const auto full = dtwMissing_L<double>(x, y);

  REQUIRE(dtwMissing_banded<double>(x, y, 2) == max_value);
  REQUIRE(dtwMissing_banded<double>(y, x, 2) == max_value);
  REQUIRE_THAT(dtwMissing_banded<double>(x, y, 7), WithinAbs(full, 1e-12));
}

// ===========================================================================
// dtwMissing_L — hand-computed values with specific missing patterns
// ===========================================================================

TEST_CASE("dtwMissing_L: hand-computed with NaN in y", "[missing_dtw]")
{
  // x = {1, 2, 3}, y = {1, NaN, 3}
  // C(0,0) = |1-1| = 0
  // C(1,0) = 0 + |2-1| = 1
  // C(2,0) = 1 + |3-1| = 3
  // C(0,1) = 0 + 0 = 0   (y[1] is NaN)
  // C(0,2) = 0 + |1-3| = 2
  // C(1,1) = min(0, 1, 0) + 0 = 0   (y[1] is NaN)
  // C(1,2) = min(0, 0, 0) + |2-3| = 1
  // C(2,1) = min(1, 0, 0) + 0 = 0   (y[1] is NaN)
  // C(2,2) = min(0, 0, 1) + |3-3| = 0
  std::vector<double> x{ 1, 2, 3 };
  std::vector<double> y{ 1, NaN, 3 };

  REQUIRE_THAT(dtwMissing_L<double>(x, y), WithinAbs(0.0, 1e-12));
}

TEST_CASE("dtwMissing_L: hand-computed with multiple NaN", "[missing_dtw]")
{
  // x = {10, NaN}, y = {NaN, 10}
  // C(0,0) = 0 (x[0]=10, y[0]=NaN -> 0)
  // C(1,0) = 0 + 0 = 0 (x[1]=NaN)
  // C(0,1) = 0 + |10-10| = 0
  // C(1,1) = min(0, 0, 0) + 0 = 0 (x[1]=NaN)
  std::vector<double> x{ 10, NaN };
  std::vector<double> y{ NaN, 10 };

  REQUIRE_THAT(dtwMissing_L<double>(x, y), WithinAbs(0.0, 1e-12));
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

// ===========================================================================
// dtwMissing_L — missing less than or equal to standard DTW
// ===========================================================================

TEST_CASE("dtwMissing_L: adding NaN does not increase distance", "[missing_dtw]")
{
  // Replace a value with NaN: the distance should decrease or stay the same
  // because NaN pairs cost 0 instead of the actual distance.
  std::vector<double> x_full{ 1, 5, 3 };
  std::vector<double> x_nan{ 1, NaN, 3 };
  std::vector<double> y{ 1, 2, 3 };

  const auto dist_full = dtwMissing_L<double>(x_full, y);
  const auto dist_nan = dtwMissing_L<double>(x_nan, y);

  // The NaN version should be <= the full version because we zero out a
  // cost that was previously > 0.
  REQUIRE(dist_nan <= dist_full + 1e-12);
}

// ===========================================================================
// dtwMissing_banded with SquaredL2 metric
// ===========================================================================

TEST_CASE("dtwMissing_banded: SquaredL2 no NaN matches standard", "[missing_dtw]")
{
  std::vector<double> x{ 1, 2, 3, 4, 5 };
  std::vector<double> y{ 2, 3, 4, 5, 6, 7, 8 };
  int band = 3;
  auto metric = core::MetricType::SquaredL2;

  const auto missing_result = dtwMissing_banded<double>(x, y, band, -1, metric);
  const auto standard_result = dtwBanded<double>(x, y, band, -1, metric);

  REQUIRE_THAT(missing_result, WithinAbs(standard_result, 1e-12));
}

// ===========================================================================
// A3: an all-NaN series under MissingStrategy::Interpolate must be rejected by
// the SERIAL pre-scan, with a diagnostic naming the offending series — not by
// interpolate_linear() throwing from inside the parallel per-pair lambda.
// ===========================================================================

TEST_CASE("Interpolate: all-NaN series is rejected by the serial pre-scan",
          "[missing_dtw][interpolate][prescan][regression]")
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

TEST_CASE("Interpolate: a partially-missing series still fills normally",
          "[missing_dtw][interpolate][prescan]")
{
  Data data;
  data.ndim = 1;
  data.p_vec = { { 1, 2, 3, 4 }, { NaN, 2, NaN, 4 } };
  data.p_names = { "clean", "gappy" };

  Problem prob;
  prob.set_data(std::move(data));
  prob.set_verbose(false);
  prob.set_missing_strategy(core::MissingStrategy::Interpolate);
  REQUIRE_NOTHROW(prob.fill_distance_matrix());
  REQUIRE(!is_missing(prob.dist_by_ind(0, 1)));
}

// ===========================================================================
// A4: MissingStrategy::Error means "throw on NaN". It was implemented only in
// Problem::fill_distance_matrix; the pairwise entry points ran the recurrence
// on NaN and returned NaN, which is ALSO the "uncomputed" sentinel of
// DenseDistanceMatrix — an unfillable matrix with no diagnostic.
// ===========================================================================

TEST_CASE("distance::dtw honours MissingStrategy::Error",
          "[missing_dtw][error_strategy][regression]")
{
  const std::vector<double> x{ 1, 2, NaN, 4 };
  const std::vector<double> y{ 1, 2, 3, 4 };
  const core::DTWVariantParams params{}; // Standard

  REQUIRE_THROWS_AS(
    distance::dtw<double>(x, y, params, -1, core::MetricType::L1,
                          core::MissingStrategy::Error),
    InvalidInput);
  REQUIRE_THROWS_AS(
    distance::dtw<double>(y, x, params, -1, core::MetricType::L1,
                          core::MissingStrategy::Error),
    InvalidInput);

  // Clean input on the same path is untouched.
  REQUIRE_NOTHROW(distance::dtw<double>(y, y, params, -1, core::MetricType::L1,
                                        core::MissingStrategy::Error));
}

TEST_CASE("core::dtw_runtime honours MissingStrategy::Error",
          "[missing_dtw][error_strategy][regression]")
{
  const std::vector<double> x{ 1, 2, NaN, 4 };
  const std::vector<double> y{ 1, 2, 3, 4 };

  core::DTWOptions opts;
  opts.missing_strategy = core::MissingStrategy::Error;

  REQUIRE_THROWS_AS(
    core::dtw_runtime(x.data(), x.size(), y.data(), y.size(), opts),
    InvalidInput);
  REQUIRE_THROWS_AS(
    core::dtw_runtime(y.data(), y.size(), x.data(), x.size(), opts),
    InvalidInput);

  // A NaN-free call returns the ordinary distance, unchanged.
  REQUIRE_THAT(core::dtw_runtime(y.data(), y.size(), y.data(), y.size(), opts),
               WithinAbs(0.0, 1e-12));
}
