/**
 * @file unit_test_lower_bounds.cpp
 * @brief Unit tests for LB_Keogh and LB_Kim lower bound functions.
 *
 * @details Verifies correctness of lower-bound computations:
 *   - Envelope construction for known series
 *   - LB_Keogh and LB_Kim are always <= actual DTW distance
 *   - Identical series yield LB == 0
 *   - Edge cases (single element, empty series)
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include <core/lower_bound_impl.hpp>
#include <warping.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>
#include <random>

using Catch::Matchers::WithinAbs;
using data_t = double;


// ---------------------------------------------------------------------------
// Test 1: Envelope correctness for a known series with band=1
// ---------------------------------------------------------------------------
TEST_CASE("compute_envelopes known series band=1", "[lower_bounds][envelopes]")
{
  // Series:  {1, 3, 2, 4, 1}
  // band = 1 means window [i-1, i+1]
  //
  // i=0: window [0,1] -> values {1,3} -> upper=3, lower=1
  // i=1: window [0,2] -> values {1,3,2} -> upper=3, lower=1
  // i=2: window [1,3] -> values {3,2,4} -> upper=4, lower=2
  // i=3: window [2,4] -> values {2,4,1} -> upper=4, lower=1
  // i=4: window [3,4] -> values {4,1} -> upper=4, lower=1

  const std::vector<data_t> series = { 1.0, 3.0, 2.0, 4.0, 1.0 };
  std::vector<data_t> upper(5), lower(5);

  dtwc::core::compute_envelopes(series, 1, upper, lower);

  // Upper envelope
  REQUIRE_THAT(upper[0], WithinAbs(3.0, 1e-15));
  REQUIRE_THAT(upper[1], WithinAbs(3.0, 1e-15));
  REQUIRE_THAT(upper[2], WithinAbs(4.0, 1e-15));
  REQUIRE_THAT(upper[3], WithinAbs(4.0, 1e-15));
  REQUIRE_THAT(upper[4], WithinAbs(4.0, 1e-15));

  // Lower envelope
  REQUIRE_THAT(lower[0], WithinAbs(1.0, 1e-15));
  REQUIRE_THAT(lower[1], WithinAbs(1.0, 1e-15));
  REQUIRE_THAT(lower[2], WithinAbs(2.0, 1e-15));
  REQUIRE_THAT(lower[3], WithinAbs(1.0, 1e-15));
  REQUIRE_THAT(lower[4], WithinAbs(1.0, 1e-15));
}

// ---------------------------------------------------------------------------
// Test 2: Envelope with band=0 (each position is its own window)
// ---------------------------------------------------------------------------
TEST_CASE("compute_envelopes band=0 identity", "[lower_bounds][envelopes]")
{
  const std::vector<data_t> series = { 5.0, 2.0, 8.0, 1.0 };
  std::vector<data_t> upper(4), lower(4);

  dtwc::core::compute_envelopes(series, 0, upper, lower);

  for (std::size_t i = 0; i < series.size(); ++i) {
    REQUIRE_THAT(upper[i], WithinAbs(series[i], 1e-15));
    REQUIRE_THAT(lower[i], WithinAbs(series[i], 1e-15));
  }
}

// ---------------------------------------------------------------------------
// Test 3: LB_Keogh == 0 for identical series
// ---------------------------------------------------------------------------
TEST_CASE("lb_keogh identical series gives zero", "[lower_bounds][lb_keogh]")
{
  const std::vector<data_t> series = { 1.0, 3.0, 2.0, 4.0, 1.0 };
  std::vector<data_t> upper, lower;

  dtwc::core::compute_envelopes(series, 1, upper, lower);
  const data_t lb = dtwc::core::lb_keogh(series, upper, lower);

  REQUIRE_THAT(lb, WithinAbs(0.0, 1e-15));
}

// ---------------------------------------------------------------------------
// Test 4: LB_Kim == 0 for identical series
// ---------------------------------------------------------------------------
TEST_CASE("lb_kim identical series gives zero", "[lower_bounds][lb_kim]")
{
  const std::vector<data_t> series = { 1.0, 3.0, 2.0, 4.0, 1.0 };
  const data_t lb = dtwc::core::lb_kim(series, series);

  REQUIRE_THAT(lb, WithinAbs(0.0, 1e-15));
}

// ---------------------------------------------------------------------------
// Test 5: LB_Keogh <= DTW for random series (property test, 10 pairs)
// ---------------------------------------------------------------------------
TEST_CASE("lb_keogh <= DTW for random series", "[lower_bounds][lb_keogh][property]")
{
  std::mt19937 rng(42); // deterministic seed
  std::uniform_real_distribution<data_t> dist(-10.0, 10.0);

  constexpr int N_pairs = 10;
  constexpr int series_len = 20;
  constexpr int band = 3;

  for (int p = 0; p < N_pairs; ++p) {
    std::vector<data_t> x(series_len), y(series_len);
    for (int i = 0; i < series_len; ++i) {
      x[i] = dist(rng);
      y[i] = dist(rng);
    }

    // Compute envelopes of y, then LB_Keogh(x, envelope_of_y)
    std::vector<data_t> upper, lower;
    dtwc::core::compute_envelopes(y, band, upper, lower);
    const data_t lb = dtwc::core::lb_keogh(x, upper, lower);

    // Compute banded DTW for comparison
    const data_t dtw_dist = dtwc::dtwBanded<data_t>(x, y, band);

    INFO("Pair " << p << ": LB_Keogh=" << lb << " DTW=" << dtw_dist);
    REQUIRE(lb <= dtw_dist + 1e-10); // allow tiny floating-point tolerance
  }
}

// ---------------------------------------------------------------------------
// Test 6: LB_Kim <= DTW for random series (property test, 10 pairs)
// ---------------------------------------------------------------------------
TEST_CASE("lb_kim <= DTW for random series", "[lower_bounds][lb_kim][property]")
{
  std::mt19937 rng(123); // deterministic seed
  std::uniform_real_distribution<data_t> dist(-10.0, 10.0);

  constexpr int N_pairs = 10;
  constexpr int series_len = 20;

  for (int p = 0; p < N_pairs; ++p) {
    std::vector<data_t> x(series_len), y(series_len);
    for (int i = 0; i < series_len; ++i) {
      x[i] = dist(rng);
      y[i] = dist(rng);
    }

    const data_t lb = dtwc::core::lb_kim(x, y);
    const data_t dtw_dist = dtwc::dtwFull<data_t>(x, y);

    INFO("Pair " << p << ": LB_Kim=" << lb << " DTW=" << dtw_dist);
    REQUIRE(lb <= dtw_dist + 1e-10);
  }
}

// ---------------------------------------------------------------------------
// Test 7: LB_Keogh with query inside envelope gives 0
// ---------------------------------------------------------------------------
TEST_CASE("lb_keogh query within envelope gives zero", "[lower_bounds][lb_keogh]")
{
  // Candidate series
  const std::vector<data_t> candidate = { 1.0, 5.0, 3.0, 7.0, 2.0 };
  std::vector<data_t> upper, lower;
  dtwc::core::compute_envelopes(candidate, 2, upper, lower);

  // Construct a query that is strictly within the envelope at every point
  std::vector<data_t> query(candidate.size());
  for (std::size_t i = 0; i < candidate.size(); ++i) {
    query[i] = (upper[i] + lower[i]) / 2.0; // midpoint is always within
  }

  const data_t lb = dtwc::core::lb_keogh(query, upper, lower);
  REQUIRE_THAT(lb, WithinAbs(0.0, 1e-15));
}

// ---------------------------------------------------------------------------
// Test 8: LB_Kim with single-element series
// ---------------------------------------------------------------------------
TEST_CASE("lb_kim single element series", "[lower_bounds][lb_kim]")
{
  const std::vector<data_t> x = { 5.0 };
  const std::vector<data_t> y = { 3.0 };

  const data_t lb = dtwc::core::lb_kim(x, y);
  // For single-element series, LB_Kim = |5 - 3| = 2
  // (first and last are same element; min/max branch skipped for length < 2)
  REQUIRE_THAT(lb, WithinAbs(2.0, 1e-15));

  // Also verify <= DTW
  const data_t dtw_dist = dtwc::dtwFull<data_t>(x, y);
  REQUIRE(lb <= dtw_dist + 1e-10);
}

// ---------------------------------------------------------------------------
// Test 9: LB_Kim with empty series returns 0
// ---------------------------------------------------------------------------
TEST_CASE("lb_kim empty series returns zero", "[lower_bounds][lb_kim]")
{
  const std::vector<data_t> empty;
  const std::vector<data_t> x = { 1.0, 2.0, 3.0 };

  REQUIRE_THAT(dtwc::core::lb_kim(empty, x), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(dtwc::core::lb_kim(x, empty), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(dtwc::core::lb_kim(empty, empty), WithinAbs(0.0, 1e-15));
}

// ---------------------------------------------------------------------------
// Test 10: LB_Kim <= DTW for known simple case with exact expected values
// ---------------------------------------------------------------------------
TEST_CASE("lb_kim <= dtw for known simple case", "[lower_bounds][lb_kim]")
{
  // x = {1, 2, 3}, y = {3, 4, 5, 6, 7}
  // DTW(x,y) = 13 (from existing warping tests)
  const std::vector<data_t> x = { 1.0, 2.0, 3.0 };
  const std::vector<data_t> y = { 3.0, 4.0, 5.0, 6.0, 7.0 };

  const data_t lb = dtwc::core::lb_kim(x, y);
  const data_t dtw_dist = dtwc::dtwFull<data_t>(x, y);

  REQUIRE_THAT(dtw_dist, WithinAbs(13.0, 1e-15));
  REQUIRE(lb <= dtw_dist + 1e-10);
  // LB_Kim: max(|1-3|, |3-7|, |1-3|, |3-7|) = max(2, 4, 2, 4) = 4
  REQUIRE_THAT(lb, WithinAbs(4.0, 1e-15));
}

// ---------------------------------------------------------------------------
// Test 11: Envelope with large band covers entire series
// ---------------------------------------------------------------------------
TEST_CASE("compute_envelopes large band covers full series", "[lower_bounds][envelopes]")
{
  const std::vector<data_t> series = { 3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0 };
  std::vector<data_t> upper, lower;

  // band = 100 >> series length, so every position sees all values
  dtwc::core::compute_envelopes(series, 100, upper, lower);

  const data_t series_min = *std::min_element(series.begin(), series.end());
  const data_t series_max = *std::max_element(series.begin(), series.end());

  for (std::size_t i = 0; i < series.size(); ++i) {
    REQUIRE_THAT(upper[i], WithinAbs(series_max, 1e-15));
    REQUIRE_THAT(lower[i], WithinAbs(series_min, 1e-15));
  }
}
