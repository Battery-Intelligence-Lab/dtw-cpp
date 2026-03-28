/**
 * @file unit_test_adtw.cpp
 * @brief Unit tests for Amerced DTW (ADTW)
 *
 * @details Tests for adtwFull_L and adtwBanded functions which add a penalty
 * for non-diagonal warping steps.
 *
 * Reference: Herrmann & Shifaz (2023), "Amercing: An intuitive and effective
 *            constraint for DTW"
 *
 * @date 28 Mar 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

// ---------------------------------------------------------------------------
// adtwFull_L tests
// ---------------------------------------------------------------------------

TEST_CASE("adtwFull_L: zero penalty equals standard DTW", "[adtw]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 };

  const auto dtw_result = dtwFull_L<data_t>(x, y);
  const auto adtw_result = adtwFull_L<data_t>(x, y, 0.0);

  REQUIRE_THAT(adtw_result, WithinAbs(dtw_result, 1e-12));
}

TEST_CASE("adtwFull_L: zero penalty equals standard DTW (reversed)", "[adtw]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 };

  const auto dtw_result = dtwFull_L<data_t>(y, x);
  const auto adtw_result = adtwFull_L<data_t>(y, x, 0.0);

  REQUIRE_THAT(adtw_result, WithinAbs(dtw_result, 1e-12));
}

TEST_CASE("adtwFull_L: identical series gives zero for any penalty", "[adtw]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3, 4, 5 };

  REQUIRE_THAT(adtwFull_L<data_t>(x, x, 0.0), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(adtwFull_L<data_t>(x, x, 1.0), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(adtwFull_L<data_t>(x, x, 100.0), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(adtwFull_L<data_t>(x, x, 1e18), WithinAbs(0.0, 1e-15));
}

TEST_CASE("adtwFull_L: huge penalty forces diagonal — equal-length", "[adtw]")
{
  using data_t = double;
  // With penalty=1e18 on equal-length series, only diagonal path is viable.
  // Cost = sum |x[i] - y[i]| = |1-2| + |2-4| + |3-5| = 1 + 2 + 2 = 5
  std::vector<data_t> x{ 1, 2, 3 }, y{ 2, 4, 5 };

  const auto result = adtwFull_L<data_t>(x, y, 1e18);
  const data_t expected = 1.0 + 2.0 + 2.0; // 5.0
  REQUIRE_THAT(result, WithinAbs(expected, 1e-6));
}

TEST_CASE("adtwFull_L: symmetry", "[adtw]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 3, 5, 2 }, y{ 2, 4, 6 };

  for (double p : { 0.0, 0.5, 1.0, 5.0, 100.0 }) {
    const auto fwd = adtwFull_L<data_t>(x, y, p);
    const auto bwd = adtwFull_L<data_t>(y, x, p);
    REQUIRE_THAT(fwd, WithinAbs(bwd, 1e-12));
  }
}

TEST_CASE("adtwFull_L: non-negativity", "[adtw]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 5, 6, 7, 8 };

  for (double p : { 0.0, 1.0, 10.0 }) {
    REQUIRE(adtwFull_L<data_t>(x, y, p) >= 0.0);
  }
}

TEST_CASE("adtwFull_L: monotonicity — higher penalty gives >= cost", "[adtw]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 };

  const auto cost_p0 = adtwFull_L<data_t>(x, y, 0.0);
  const auto cost_p1 = adtwFull_L<data_t>(x, y, 1.0);
  const auto cost_p5 = adtwFull_L<data_t>(x, y, 5.0);
  const auto cost_p50 = adtwFull_L<data_t>(x, y, 50.0);

  REQUIRE(cost_p1 >= cost_p0 - 1e-12);
  REQUIRE(cost_p5 >= cost_p1 - 1e-12);
  REQUIRE(cost_p50 >= cost_p5 - 1e-12);
}

TEST_CASE("adtwFull_L: ADTW >= standard DTW always", "[adtw]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 };

  const auto dtw_val = dtwFull_L<data_t>(x, y);
  for (double p : { 0.0, 0.1, 1.0, 10.0, 100.0 }) {
    const auto adtw_val = adtwFull_L<data_t>(x, y, p);
    REQUIRE(adtw_val >= dtw_val - 1e-12);
  }
}

TEST_CASE("adtwFull_L: hand-computed 3-element example", "[adtw]")
{
  using data_t = double;
  // x = {1, 2, 3}, y = {2, 4, 5}, penalty = 1.0
  // Hand-computed: C(2,2) = 5.0  (see derivation in commit message / implementation notes)
  std::vector<data_t> x{ 1, 2, 3 }, y{ 2, 4, 5 };

  const auto result = adtwFull_L<data_t>(x, y, 1.0);
  REQUIRE_THAT(result, WithinAbs(5.0, 1e-12));
}

TEST_CASE("adtwFull_L: empty and single-element edge cases", "[adtw]")
{
  using data_t = double;
  std::vector<data_t> empty{};
  std::vector<data_t> single{ 42.0 };
  std::vector<data_t> x{ 1, 2, 3 };

  // Empty -> maxValue
  REQUIRE(adtwFull_L<data_t>(x, empty, 1.0) > 1e10);
  REQUIRE(adtwFull_L<data_t>(empty, x, 1.0) > 1e10);
  std::vector<data_t> empty2{};
  REQUIRE(adtwFull_L<data_t>(empty, empty2, 1.0) > 1e10);

  // Single element
  // adtw({42}, {42}, p) == 0
  REQUIRE_THAT(adtwFull_L<data_t>(single, single, 5.0), WithinAbs(0.0, 1e-15));

  // adtw({1,2,3}, {42}, p) — only path is vertical steps with penalty
  // cost = |1-42| + (|2-42| + p) + (|3-42| + p) = 41 + 41 + 40 + 2p
  // With p=1: 41 + 41 + 40 + 2 = 124  (but we need to verify direction)
  // Actually: short_side is {42}, long_vec is {1,2,3}
  // C(0,0) = |42-1| = 41. Then j=1: C(0,1) = C(0,0) + p + |42-2| = 41+1+40 = 82
  // j=2: C(0,2) = C(0,1) + p + |42-3| = 82+1+39 = 122
  // Wait — when one side is length 1, short_side has 1 element.
  // short_side[0] = |42-1| = 41
  // j=1: diag=41, short_side[0] = 41 + p + |42-2| = 41+1+40 = 82
  //   but no inner loop since m_short=1
  // j=2: diag=82, short_side[0] = 82 + p + |42-3| = 82+1+39 = 122
  const auto result = adtwFull_L<data_t>(x, single, 1.0);
  REQUIRE_THAT(result, WithinAbs(122.0, 1e-12));
}

TEST_CASE("adtwFull_L: different lengths work", "[adtw]")
{
  using data_t = double;
  std::vector<data_t> a{ 1, 2, 3, 4 };
  std::vector<data_t> b{ 1, 2, 3, 4, 5, 6 };

  // Should not crash and should be symmetric
  const auto fwd = adtwFull_L<data_t>(a, b, 2.0);
  const auto bwd = adtwFull_L<data_t>(b, a, 2.0);
  REQUIRE_THAT(fwd, WithinAbs(bwd, 1e-12));
  REQUIRE(fwd >= 0.0);
}

// ---------------------------------------------------------------------------
// adtwBanded tests
// ---------------------------------------------------------------------------

TEST_CASE("adtwBanded: zero penalty equals standard dtwBanded", "[adtw]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 };
  int band = 2;

  const auto dtw_result = dtwBanded<data_t>(x, y, band);
  const auto adtw_result = adtwBanded<data_t>(x, y, band, 0.0);

  REQUIRE_THAT(adtw_result, WithinAbs(dtw_result, 1e-12));
}

TEST_CASE("adtwBanded: negative band falls back to adtwFull_L", "[adtw]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 2, 4, 5 };
  double penalty = 1.0;

  const auto full_result = adtwFull_L<data_t>(x, y, penalty);
  const auto banded_result = adtwBanded<data_t>(x, y, -1, penalty);

  REQUIRE_THAT(banded_result, WithinAbs(full_result, 1e-12));
}

TEST_CASE("adtwBanded: large band equals adtwFull_L", "[adtw]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3, 4 }, y{ 2, 4, 5, 6, 7, 8 };
  double penalty = 2.0;

  const auto full_result = adtwFull_L<data_t>(x, y, penalty);
  const auto banded_result = adtwBanded<data_t>(x, y, 100, penalty);

  REQUIRE_THAT(banded_result, WithinAbs(full_result, 1e-12));
}

TEST_CASE("adtwBanded: identical series gives zero", "[adtw]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3, 4, 5 };

  REQUIRE_THAT(adtwBanded<data_t>(x, x, 2, 5.0), WithinAbs(0.0, 1e-15));
}

TEST_CASE("adtwBanded: symmetry", "[adtw]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 3, 5, 2, 7 }, y{ 2, 4, 6, 1, 3 };
  int band = 2;

  for (double p : { 0.0, 1.0, 5.0 }) {
    const auto fwd = adtwBanded<data_t>(x, y, band, p);
    const auto bwd = adtwBanded<data_t>(y, x, band, p);
    REQUIRE_THAT(fwd, WithinAbs(bwd, 1e-12));
  }
}

TEST_CASE("adtwBanded: monotonicity in penalty", "[adtw]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3, 4, 5 }, y{ 2, 4, 5, 6, 7, 8 };
  int band = 3;

  const auto cost_p0 = adtwBanded<data_t>(x, y, band, 0.0);
  const auto cost_p1 = adtwBanded<data_t>(x, y, band, 1.0);
  const auto cost_p10 = adtwBanded<data_t>(x, y, band, 10.0);

  REQUIRE(cost_p1 >= cost_p0 - 1e-12);
  REQUIRE(cost_p10 >= cost_p1 - 1e-12);
}

TEST_CASE("adtwBanded: ADTW >= standard banded DTW", "[adtw]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3, 4, 5 }, y{ 2, 4, 5, 6, 7, 8 };
  int band = 3;

  const auto dtw_val = dtwBanded<data_t>(x, y, band);
  for (double p : { 0.0, 1.0, 10.0 }) {
    const auto adtw_val = adtwBanded<data_t>(x, y, band, p);
    REQUIRE(adtw_val >= dtw_val - 1e-12);
  }
}

TEST_CASE("adtwBanded: empty vectors", "[adtw]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3 }, empty{};

  REQUIRE(adtwBanded<data_t>(x, empty, 2, 1.0) > 1e10);
  REQUIRE(adtwBanded<data_t>(empty, x, 2, 1.0) > 1e10);
}
