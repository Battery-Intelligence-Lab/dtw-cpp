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
 * @author Volkan Kumtepeli
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

TEST_CASE("adtwBanded: early_abandon disabled (-1) gives exact result", "[adtw]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3, 4, 5 }, y{ 2, 4, 5, 6, 7, 8 };
  int band = 3;
  double penalty = 1.0;

  const auto no_abandon   = adtwBanded<data_t>(x, y, band, penalty, -1.0);
  const auto default_call = adtwBanded<data_t>(x, y, band, penalty);  // default = -1

  REQUIRE_THAT(no_abandon, WithinAbs(default_call, 1e-12));
}

TEST_CASE("adtwBanded: early_abandon above true cost gives exact result", "[adtw]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3, 4, 5 }, y{ 2, 4, 5, 6, 7, 8 };
  int band = 3;
  double penalty = 1.0;

  const auto exact        = adtwBanded<data_t>(x, y, band, penalty);
  const auto high_abandon = adtwBanded<data_t>(x, y, band, penalty, exact + 1e6);

  REQUIRE_THAT(high_abandon, WithinAbs(exact, 1e-12));
}

TEST_CASE("adtwBanded: early_abandon below true cost triggers abandon", "[adtw]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3, 4, 5 }, y{ 2, 4, 5, 6, 7, 8 };
  int band = 3;
  double penalty = 1.0;

  const auto exact         = adtwBanded<data_t>(x, y, band, penalty);
  constexpr double maxValue = std::numeric_limits<double>::max();

  // Threshold strictly below true cost triggers abandon
  const auto abandoned = adtwBanded<data_t>(x, y, band, penalty, exact * 0.1);
  REQUIRE(abandoned >= maxValue * 0.5);  // returned maxValue sentinel
}

// ---------------------------------------------------------------------------
// Reference implementation for cross-checking (O(n*m) full matrix, no rolling
// buffer, no early abandon). Implements the exact ADTW recurrence:
//   C(i,j) = d(x[i],y[j]) + min(C(i-1,j-1),
//                                C(i-1,j) + penalty,
//                                C(i,j-1) + penalty)
// with C(-1,*) = C(*,-1) = maxValue  (open boundaries).
// ---------------------------------------------------------------------------
namespace {
template <typename data_t>
data_t adtwBanded_reference(const std::vector<data_t> &x, const std::vector<data_t> &y,
                            int band, data_t penalty)
{
  const int nx = static_cast<int>(x.size());
  const int ny = static_cast<int>(y.size());
  if (nx == 0 || ny == 0) return std::numeric_limits<data_t>::max();

  // For band < 0 treat as full matrix.
  const int effective_band = (band < 0) ? std::max(nx, ny) : band;

  // Sakoe-Chiba window using the same slope/window formula as adtwBanded.
  // short = min-length side, long = max-length side.
  // We map short index 'si' -> long index 'li' with center = slope * si.
  const bool x_is_short = (nx <= ny);
  const int m_short = x_is_short ? nx : ny;
  const int m_long  = x_is_short ? ny : nx;
  auto short_val = [&](int i) -> data_t { return x_is_short ? x[i] : y[i]; };
  auto long_val  = [&](int i) -> data_t { return x_is_short ? y[i] : x[i]; };

  const double slope  = (m_short == 1) ? 0.0
                                       : static_cast<double>(m_long - 1) / (m_short - 1);
  const double window = std::max(static_cast<double>(effective_band), slope / 2.0);

  constexpr data_t maxVal = std::numeric_limits<data_t>::max();

  // Full O(m_short * m_long) matrix.
  std::vector<std::vector<data_t>> C(m_short, std::vector<data_t>(m_long, maxVal));

  for (int si = 0; si < m_short; ++si) {
    const double center = slope * si;
    const int lo = static_cast<int>(std::ceil(std::round(100.0 * (center - window)) / 100.0));
    const int hi_excl = static_cast<int>(std::floor(std::round(100.0 * (center + window)) / 100.0)) + 1;
    for (int li = std::max(lo, 0); li < std::min(hi_excl, m_long); ++li) {
      const data_t d = std::abs(short_val(si) - long_val(li));
      data_t prev_diag = (si > 0 && li > 0) ? C[si - 1][li - 1] : maxVal;
      data_t prev_left  = maxVal; // C[si-1][li] + penalty  (step in short dim)
      data_t prev_above = maxVal; // C[si][li-1] + penalty  (step in long dim)
      if (si > 0 && C[si - 1][li] < maxVal)  prev_left  = C[si - 1][li]  + penalty;
      if (li > 0 && C[si][li - 1] < maxVal)  prev_above = C[si][li - 1]  + penalty;
      if (si == 0 && li == 0) {
        C[si][li] = d;
      } else {
        const data_t best = std::min({ prev_diag, prev_left, prev_above });
        C[si][li] = (best < maxVal) ? best + d : maxVal;
      }
    }
  }
  return C[m_short - 1][m_long - 1];
}

// Same for full (no band restriction).
template <typename data_t>
data_t adtwFull_reference(const std::vector<data_t> &x, const std::vector<data_t> &y,
                          data_t penalty)
{
  return adtwBanded_reference(x, y, -1, penalty);
}
} // namespace

// ---------------------------------------------------------------------------
// Adversarial cross-check tests
// ---------------------------------------------------------------------------

TEST_CASE("adtwBanded: reference cross-check — equal length, various penalties", "[adtw][adversarial]")
{
  using data_t = double;
  // Hand-chosen sequences with non-trivial structure.
  std::vector<data_t> x{ 1.0, 3.0, 2.0, 5.0, 4.0 };
  std::vector<data_t> y{ 2.0, 2.0, 4.0, 3.0, 6.0 };

  for (int band : { 1, 2, 3, 10 }) {
    for (double penalty : { 0.0, 0.5, 1.0, 5.0, 100.0 }) {
      const auto ref    = adtwBanded_reference(x, y, band, penalty);
      const auto actual = adtwBanded<data_t>(x, y, band, penalty);
      INFO("band=" << band << " penalty=" << penalty);
      REQUIRE_THAT(actual, WithinAbs(ref, 1e-9));
    }
  }
}

TEST_CASE("adtwBanded: reference cross-check — asymmetric nx=3 ny=7", "[adtw][adversarial]")
{
  using data_t = double;
  std::vector<data_t> x{ 1.0, 4.0, 2.0 };
  std::vector<data_t> y{ 0.5, 1.5, 3.0, 4.5, 3.5, 2.5, 1.0 };

  for (int band : { 1, 2, 3, 5, 10 }) {
    for (double penalty : { 0.0, 0.5, 1.0, 2.0, 10.0 }) {
      const auto ref    = adtwBanded_reference(x, y, band, penalty);
      const auto actual = adtwBanded<data_t>(x, y, band, penalty);
      INFO("nx=3 ny=7 band=" << band << " penalty=" << penalty);
      REQUIRE_THAT(actual, WithinAbs(ref, 1e-9));
    }
  }
}

TEST_CASE("adtwBanded: reference cross-check — asymmetric nx=7 ny=3 (transposed)", "[adtw][adversarial]")
{
  using data_t = double;
  std::vector<data_t> x{ 0.5, 1.5, 3.0, 4.5, 3.5, 2.5, 1.0 };
  std::vector<data_t> y{ 1.0, 4.0, 2.0 };

  for (int band : { 1, 2, 3, 5, 10 }) {
    for (double penalty : { 0.0, 0.5, 1.0, 2.0, 10.0 }) {
      const auto ref    = adtwBanded_reference(x, y, band, penalty);
      const auto actual = adtwBanded<data_t>(x, y, band, penalty);
      INFO("nx=7 ny=3 band=" << band << " penalty=" << penalty);
      REQUIRE_THAT(actual, WithinAbs(ref, 1e-9));
    }
  }
}

TEST_CASE("adtwBanded: asymmetric (nx=3,ny=7) equals (nx=7,ny=3) — symmetry", "[adtw][adversarial]")
{
  using data_t = double;
  // adtwBanded is symmetric by construction (swapping x,y should give same result
  // because the short/long split is on the inner dimension, not on which is x vs y).
  std::vector<data_t> x{ 1.0, 4.0, 2.0 };
  std::vector<data_t> y{ 0.5, 1.5, 3.0, 4.5, 3.5, 2.5, 1.0 };

  for (int band : { 1, 2, 4, 7 }) {
    for (double penalty : { 0.0, 1.0, 5.0 }) {
      const auto fwd = adtwBanded<data_t>(x, y, band, penalty);
      const auto bwd = adtwBanded<data_t>(y, x, band, penalty);
      INFO("band=" << band << " penalty=" << penalty);
      REQUIRE_THAT(fwd, WithinAbs(bwd, 1e-12));
    }
  }
}

TEST_CASE("adtwBanded: reference cross-check — asymmetric nx=2 ny=6", "[adtw][adversarial]")
{
  using data_t = double;
  // Extreme ratio: short=2, long=6. slope=1.0, window=max(band, 0.5).
  std::vector<data_t> x{ 1.0, 5.0 };
  std::vector<data_t> y{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };

  for (int band : { 0, 1, 2, 3, 10 }) {
    for (double penalty : { 0.0, 1.0, 3.0 }) {
      const auto ref    = adtwBanded_reference(x, y, band, penalty);
      const auto actual = adtwBanded<data_t>(x, y, band, penalty);
      INFO("nx=2 ny=6 band=" << band << " penalty=" << penalty);
      REQUIRE_THAT(actual, WithinAbs(ref, 1e-9));
    }
  }
}

TEST_CASE("adtwBanded: band=0 equal-length — only diagonal path", "[adtw][adversarial]")
{
  using data_t = double;
  // With band=0 on equal-length sequences, slope=1, window=max(0,0.5)=0.5.
  // Bounds for row i: lo=ceil(i-0.5)=i, hi=floor(i+0.5)+1=i+1.
  // So only cell (i,i) is reachable => cost = sum |x[i]-y[i]|.
  std::vector<data_t> x{ 1.0, 3.0, 5.0, 2.0 };
  std::vector<data_t> y{ 2.0, 1.0, 4.0, 3.0 };

  const double expected = std::abs(1-2) + std::abs(3-1) + std::abs(5-4) + std::abs(2-3); // 1+2+1+1=5
  const auto actual = adtwBanded<data_t>(x, y, 0, 0.0);
  const auto ref    = adtwBanded_reference(x, y, 0, 0.0);
  REQUIRE_THAT(actual, WithinAbs(expected, 1e-12));
  REQUIRE_THAT(ref,    WithinAbs(expected, 1e-12));
}

TEST_CASE("adtwBanded: band=0 equal-length with penalty — only diagonal", "[adtw][adversarial]")
{
  using data_t = double;
  std::vector<data_t> x{ 1.0, 3.0, 5.0, 2.0 };
  std::vector<data_t> y{ 2.0, 1.0, 4.0, 3.0 };

  // Pure diagonal path incurs zero penalty (all steps are diagonal).
  const double expected = std::abs(1-2) + std::abs(3-1) + std::abs(5-4) + std::abs(2-3);
  for (double p : { 0.0, 1.0, 100.0 }) {
    const auto actual = adtwBanded<data_t>(x, y, 0, p);
    const auto ref    = adtwBanded_reference(x, y, 0, p);
    INFO("penalty=" << p);
    REQUIRE_THAT(actual, WithinAbs(expected, 1e-12));
    REQUIRE_THAT(ref,    WithinAbs(expected, 1e-12));
  }
}

TEST_CASE("adtwFull_L: reference cross-check — various lengths and penalties", "[adtw][adversarial]")
{
  using data_t = double;

  struct Case {
    std::vector<data_t> x, y;
  };
  std::vector<Case> cases = {
    { { 1.0, 2.0, 3.0 }, { 1.0, 2.0, 3.0, 4.0, 5.0 } },
    { { 5.0, 3.0, 1.0 }, { 1.0, 2.0, 3.0 } },
    { { 1.0 },           { 2.0, 3.0, 4.0 } },
    { { 0.0, 10.0 },     { 1.0, 2.0, 3.0, 8.0, 10.0 } },
    { { 1.0, 1.0, 1.0, 1.0 }, { 2.0, 2.0 } },
  };

  for (auto &c : cases) {
    for (double penalty : { 0.0, 0.5, 1.0, 5.0 }) {
      const auto ref    = adtwFull_reference(c.x, c.y, penalty);
      const auto actual = adtwFull_L<data_t>(c.x, c.y, penalty);
      INFO("nx=" << c.x.size() << " ny=" << c.y.size() << " penalty=" << penalty);
      REQUIRE_THAT(actual, WithinAbs(ref, 1e-9));
    }
  }
}

TEST_CASE("adtwBanded vs adtwFull_L: large band gives identical result", "[adtw][adversarial]")
{
  // When band covers the entire matrix both should agree exactly.
  using data_t = double;
  std::vector<data_t> x{ 2.0, 1.0, 3.0, 5.0, 4.0 };
  std::vector<data_t> y{ 1.0, 3.0, 2.0 };

  for (double p : { 0.0, 0.5, 2.0, 10.0 }) {
    const auto full   = adtwFull_L<data_t>(x, y, p);
    const auto banded = adtwBanded<data_t>(x, y, 1000, p);
    INFO("penalty=" << p);
    REQUIRE_THAT(banded, WithinAbs(full, 1e-9));
  }
}

TEST_CASE("adtwBanded: stale thread_local buffer — sequential calls give correct results", "[adtw][adversarial]")
{
  // The rolling `col` buffer is thread_local and is reset with col.assign(m_long, maxValue)
  // at the start of each call. This test verifies that sequential calls with DIFFERENT
  // sequence lengths do not pollute each other.
  using data_t = double;

  std::vector<data_t> x_short{ 1.0, 2.0, 3.0 };
  std::vector<data_t> y_short{ 2.0, 3.0, 4.0 };

  std::vector<data_t> x_long{ 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0 };
  std::vector<data_t> y_long{ 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };

  const int band    = 2;
  const double p    = 1.0;

  // Compute "ground truth" for each pair in isolation.
  const auto truth_long  = adtwBanded<data_t>(x_long,  y_long,  band, p);
  const auto truth_short = adtwBanded<data_t>(x_short, y_short, band, p);

  // Now interleave calls: long first, then short, then long again.
  const auto r1 = adtwBanded<data_t>(x_long,  y_long,  band, p);
  const auto r2 = adtwBanded<data_t>(x_short, y_short, band, p);
  const auto r3 = adtwBanded<data_t>(x_long,  y_long,  band, p);

  REQUIRE_THAT(r1, WithinAbs(truth_long,  1e-12));
  REQUIRE_THAT(r2, WithinAbs(truth_short, 1e-12));
  REQUIRE_THAT(r3, WithinAbs(truth_long,  1e-12));
}

TEST_CASE("adtwFull_L: stale thread_local buffer — sequential calls with different lengths", "[adtw][adversarial]")
{
  using data_t = double;

  std::vector<data_t> a{ 1.0, 2.0 };
  std::vector<data_t> b{ 3.0, 4.0, 5.0, 6.0, 7.0 };
  std::vector<data_t> c{ 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0 };
  const double p = 1.0;

  const auto t1 = adtwFull_L<data_t>(a, b, p);
  const auto t2 = adtwFull_L<data_t>(a, c, p);
  const auto t3 = adtwFull_L<data_t>(b, c, p);

  // Re-run in different order — results must match.
  const auto r3 = adtwFull_L<data_t>(b, c, p);
  const auto r1 = adtwFull_L<data_t>(a, b, p);
  const auto r2 = adtwFull_L<data_t>(a, c, p);

  REQUIRE_THAT(r1, WithinAbs(t1, 1e-12));
  REQUIRE_THAT(r2, WithinAbs(t2, 1e-12));
  REQUIRE_THAT(r3, WithinAbs(t3, 1e-12));
}

TEST_CASE("adtwBanded: early_abandon at exactly true cost does NOT abandon", "[adtw][adversarial]")
{
  using data_t = double;
  std::vector<data_t> x{ 1.0, 2.0, 3.0, 4.0 };
  std::vector<data_t> y{ 2.0, 3.0, 4.0, 5.0, 6.0 };
  int band = 3;
  double penalty = 1.0;

  const auto exact = adtwBanded<data_t>(x, y, band, penalty);
  // Threshold exactly at the true cost: should return the true cost, not maxValue.
  const auto at_exact = adtwBanded<data_t>(x, y, band, penalty, exact);
  REQUIRE_THAT(at_exact, WithinAbs(exact, 1e-9));
}

TEST_CASE("adtwFull_L: early_abandon at exactly true cost does NOT abandon", "[adtw][adversarial]")
{
  using data_t = double;
  std::vector<data_t> x{ 1.0, 2.0, 3.0 };
  std::vector<data_t> y{ 2.0, 3.0, 4.0, 5.0 };
  double penalty = 1.0;

  const auto exact = adtwFull_L<data_t>(x, y, penalty);
  const auto at_exact = adtwFull_L<data_t>(x, y, penalty, exact);
  REQUIRE_THAT(at_exact, WithinAbs(exact, 1e-9));
}

TEST_CASE("adtwFull_L: early_abandon just below true cost triggers maxValue", "[adtw][adversarial]")
{
  using data_t = double;
  std::vector<data_t> x{ 1.0, 2.0, 3.0 };
  std::vector<data_t> y{ 5.0, 6.0, 7.0, 8.0, 9.0 };
  double penalty = 1.0;

  const auto exact = adtwFull_L<data_t>(x, y, penalty);
  constexpr double maxVal = std::numeric_limits<double>::max();

  // Threshold strictly below true cost (only slightly, but definitely below).
  const double threshold = exact - 1e-6;
  if (threshold > 0) {
    const auto abandoned = adtwFull_L<data_t>(x, y, penalty, threshold);
    REQUIRE(abandoned >= maxVal * 0.5);
  }
}

TEST_CASE("adtwBanded: early_abandon just above true cost gives exact result", "[adtw][adversarial]")
{
  using data_t = double;
  std::vector<data_t> x{ 1.0, 3.0, 2.0, 5.0 };
  std::vector<data_t> y{ 2.0, 2.0, 4.0, 3.0, 6.0 };
  int band = 2;
  double penalty = 1.5;

  const auto exact = adtwBanded<data_t>(x, y, band, penalty);
  // Threshold just barely above → should NOT abandon.
  const auto result = adtwBanded<data_t>(x, y, band, penalty, exact + 1e-9);
  REQUIRE_THAT(result, WithinAbs(exact, 1e-9));
}

TEST_CASE("adtwBanded: m_short==m_long equal-length reference cross-check", "[adtw][adversarial]")
{
  // slope=1, window=max(band, 0.5). Verify for several band values.
  using data_t = double;
  std::vector<data_t> x{ 1.0, 4.0, 2.0, 7.0, 3.0, 6.0 };
  std::vector<data_t> y{ 2.0, 3.0, 5.0, 6.0, 2.0, 4.0 };

  for (int band : { 0, 1, 2, 3, 5, 100 }) {
    for (double p : { 0.0, 1.0, 5.0 }) {
      const auto ref    = adtwBanded_reference(x, y, band, p);
      const auto actual = adtwBanded<data_t>(x, y, band, p);
      INFO("equal-length band=" << band << " penalty=" << p);
      REQUIRE_THAT(actual, WithinAbs(ref, 1e-9));
    }
  }
}

TEST_CASE("adtwBanded: nx=1 or ny=1 falls back to adtwFull_L", "[adtw][adversarial]")
{
  using data_t = double;
  std::vector<data_t> single{ 5.0 };
  std::vector<data_t> many{ 1.0, 2.0, 3.0, 4.0, 5.0 };

  for (double p : { 0.0, 1.0, 3.0 }) {
    const auto full_ab  = adtwFull_L<data_t>(single, many, p);
    const auto banded_ab = adtwBanded<data_t>(single, many, 2, p);
    const auto full_ba  = adtwFull_L<data_t>(many, single, p);
    const auto banded_ba = adtwBanded<data_t>(many, single, 2, p);
    INFO("penalty=" << p);
    REQUIRE_THAT(banded_ab, WithinAbs(full_ab, 1e-12));
    REQUIRE_THAT(banded_ba, WithinAbs(full_ba, 1e-12));
  }
}

TEST_CASE("adtwBanded: reference cross-check — random-ish sequences, many band+penalty combos", "[adtw][adversarial]")
{
  // Deterministic pseudo-random sequences to catch any systematic recurrence errors.
  using data_t = double;

  // Sequence generated by x[i] = fmod(i*i*2.71828 + i*3.14159, 10.0)
  auto gen = [](int n, double seed1, double seed2) {
    std::vector<double> v(n);
    for (int i = 0; i < n; ++i)
      v[i] = std::fmod(static_cast<double>(i) * seed1 + static_cast<double>(i * i) * seed2, 10.0);
    return v;
  };

  const auto x5  = gen(5,  2.71828, 0.31415);
  const auto y7  = gen(7,  1.41421, 0.27182);
  const auto x8  = gen(8,  3.14159, 0.14142);
  const auto y5  = gen(5,  1.73205, 0.57721);
  const auto x10 = gen(10, 2.23606, 0.69314);
  const auto y6  = gen(6,  1.61803, 0.42424);

  struct Pair { const std::vector<double>* a; const std::vector<double>* b; };
  std::vector<Pair> pairs = { {&x5, &y7}, {&y7, &x5}, {&x8, &y5}, {&x10, &y6}, {&x5, &y5} };

  for (auto &pr : pairs) {
    for (int band : { 1, 2, 3, 5 }) {
      for (double p : { 0.0, 0.5, 1.0, 3.0 }) {
        const auto ref    = adtwBanded_reference(*pr.a, *pr.b, band, p);
        const auto actual = adtwBanded<data_t>(*pr.a, *pr.b, band, p);
        INFO("nx=" << pr.a->size() << " ny=" << pr.b->size()
             << " band=" << band << " penalty=" << p);
        REQUIRE_THAT(actual, WithinAbs(ref, 1e-9));
      }
    }
  }
}

TEST_CASE("adtwFull_L: early_abandon disabled is consistent with enabled above threshold", "[adtw][adversarial]")
{
  // adtwFull_L(x, y, p) and adtwFull_L(x, y, p, exact+epsilon) must agree.
  using data_t = double;
  std::vector<data_t> x{ 2.0, 1.0, 4.0, 3.0 };
  std::vector<data_t> y{ 1.0, 3.0, 2.0, 5.0, 4.0 };

  for (double p : { 0.0, 1.0, 5.0 }) {
    const auto exact  = adtwFull_L<data_t>(x, y, p);
    const auto result = adtwFull_L<data_t>(x, y, p, exact + 1e6);
    INFO("penalty=" << p);
    REQUIRE_THAT(result, WithinAbs(exact, 1e-12));
  }
}

TEST_CASE("adtwBanded: length-2 sequences cross-check", "[adtw][adversarial]")
{
  // Minimal non-trivial case: 2 vs 4 elements.
  using data_t = double;
  std::vector<data_t> x{ 1.0, 5.0 };
  std::vector<data_t> y{ 1.0, 3.0, 4.0, 5.0 };

  for (int band : { 0, 1, 2, 3 }) {
    for (double p : { 0.0, 1.0, 2.0 }) {
      const auto ref    = adtwBanded_reference(x, y, band, p);
      const auto actual = adtwBanded<data_t>(x, y, band, p);
      INFO("band=" << band << " penalty=" << p);
      REQUIRE_THAT(actual, WithinAbs(ref, 1e-9));
    }
  }
}

// ===========================================================================
//  ADTW + Pruned distance matrix integration tests
//
//  These tests verify that fill_distance_matrix_pruned produces IDENTICAL
//  results to BruteForce when the ADTW variant is selected. This covers the
//  code path added in pruned_distance_matrix.cpp (dtw_with_abandon lambda
//  branching on is_adtw) and in Problem.cpp (pruned_strategy_applicable
//  now returns true for DTWVariant::ADTW).
//
//  The key correctness invariant: LB_Keogh is a valid lower bound for ADTW
//  because ADTW(penalty>=0) >= DTW >= LB_Keogh, so early-abandon pruning
//  cannot produce wrong results (only skip work safely).
// ===========================================================================

#include <core/pruned_distance_matrix.hpp>
#include <vector>
#include <string>
#include <cmath>
#include <random>

namespace {

// Helper: build a Problem with ADTW variant from inline vectors.
static dtwc::Problem make_adtw_problem(
  std::vector<std::vector<double>> vecs,
  std::vector<std::string> names,
  double penalty,
  int band_val = -1)
{
  dtwc::Problem prob("test_adtw_pruned");
  prob.band = band_val;

  dtwc::core::DTWVariantParams vp;
  vp.variant = dtwc::core::DTWVariant::ADTW;
  vp.adtw_penalty = penalty;
  prob.set_variant(vp);

  dtwc::Data d(std::move(vecs), std::move(names));
  prob.set_data(std::move(d));
  return prob;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// INT-1: ADTW + banded pruned matches BruteForce — small synthetic data
// ---------------------------------------------------------------------------
TEST_CASE("ADTW pruned strategy matches BruteForce — synthetic, banded",
          "[adtw][pruned_matrix][integration]")
{
  // 5 series of length 8, band=2, penalty=1.0
  std::vector<std::vector<double>> vecs = {
    { 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0, 0.0 },
    { 0.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0, 1.0 },
    { 5.0, 4.0, 3.0, 2.0, 1.0, 2.0, 3.0, 4.0 },
    { 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0 },
    { 0.5, 1.5, 2.5, 3.5, 2.5, 1.5, 0.5, -0.5 },
  };
  std::vector<std::string> names = { "a", "b", "c", "d", "e" };
  const int band = 2;
  const double penalty = 1.0;
  const int N = static_cast<int>(vecs.size());

  // BruteForce reference
  auto prob_brute = make_adtw_problem(vecs, names, penalty, band);
  prob_brute.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
  prob_brute.fillDistanceMatrix();

  // Pruned (direct call to fill_distance_matrix_pruned)
  auto prob_pruned = make_adtw_problem(vecs, names, penalty, band);
  dtwc::core::fill_distance_matrix_pruned(prob_pruned, band);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(prob_pruned.distByInd(i, j),
                   WithinAbs(prob_brute.distByInd(i, j), 1e-10));
    }
  }
}

// ---------------------------------------------------------------------------
// INT-2: ADTW + full (no band) pruned matches BruteForce
// ---------------------------------------------------------------------------
TEST_CASE("ADTW pruned strategy matches BruteForce — no band (full DTW)",
          "[adtw][pruned_matrix][integration]")
{
  std::vector<std::vector<double>> vecs = {
    { 1.0, 3.0, 5.0, 2.0, 4.0 },
    { 2.0, 4.0, 1.0, 3.0, 5.0 },
    { 5.0, 5.0, 5.0, 5.0, 5.0 },
    { 0.0, 1.0, 0.0, 1.0, 0.0 },
    { 3.0, 2.0, 1.0, 2.0, 3.0 },
  };
  std::vector<std::string> names = { "a", "b", "c", "d", "e" };
  const double penalty = 2.0;
  const int N = static_cast<int>(vecs.size());

  // BruteForce (band=-1: full)
  auto prob_brute = make_adtw_problem(vecs, names, penalty, -1);
  prob_brute.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
  prob_brute.fillDistanceMatrix();

  // Pruned with band=-1 (no LB_Keogh, only LB_Kim)
  auto prob_pruned = make_adtw_problem(vecs, names, penalty, -1);
  dtwc::core::fill_distance_matrix_pruned(prob_pruned, -1);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      INFO("i=" << i << " j=" << j);
      REQUIRE_THAT(prob_pruned.distByInd(i, j),
                   WithinAbs(prob_brute.distByInd(i, j), 1e-10));
    }
  }
}

// ---------------------------------------------------------------------------
// INT-3: LB_Keogh is a lower bound for ADTW — verify analytically
//   LB_Keogh(x, env_y) <= DTW_banded(x,y) <= ADTW_banded(x,y,penalty)
//   for penalty >= 0.
// ---------------------------------------------------------------------------
TEST_CASE("LB_Keogh <= ADTW for 30 random pairs, band=5, penalty=2.0",
          "[adtw][pruned_matrix][lb_keogh][integration]")
{
  std::mt19937 rng(555);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);
  constexpr int band = 5;
  constexpr double penalty = 2.0;

  for (int trial = 0; trial < 30; ++trial) {
    const int n = 20 + static_cast<int>(rng() % 80);
    std::vector<double> x(n), y(n);
    for (auto &v : x) v = dist(rng);
    for (auto &v : y) v = dist(rng);

    std::vector<double> upper(n), lower(n);
    dtwc::core::compute_envelopes(y, band, upper, lower);
    const double lb = dtwc::core::lb_keogh(x, upper, lower);

    const double adtw_val = dtwc::adtwBanded<double>(x, y, band, penalty);
    const double dtw_val  = dtwc::dtwBanded<double>(x, y, band);

    INFO("trial=" << trial << " n=" << n
         << " LB=" << lb << " DTW=" << dtw_val << " ADTW=" << adtw_val);
    // Chain: LB <= DTW <= ADTW
    REQUIRE(lb <= dtw_val + 1e-10);
    REQUIRE(dtw_val <= adtw_val + 1e-10);
    REQUIRE(lb <= adtw_val + 1e-10);
  }
}

// ---------------------------------------------------------------------------
// INT-4: ADTW + Auto strategy dispatches to Pruned for band >= 0
//   (Problem.cpp: pruned_strategy_applicable returns true for ADTW+band>=0+size>=64)
//   For small problems (size < 64), Auto falls back to BruteForce, but
//   directly forcing Pruned must still give the correct answer.
// ---------------------------------------------------------------------------
TEST_CASE("ADTW fill_distance_matrix_pruned with various penalties matches BruteForce",
          "[adtw][pruned_matrix][integration][penalty_sweep]")
{
  // Use 6 series length 10, band=3
  std::vector<std::vector<double>> vecs = {
    { 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0 },
    { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0 },
    { 5.0, 4.0, 3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0, 4.0 },
    { 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5, 2.5 },
    { 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 2.0, 1.5, 1.0, 0.5 },
    { 4.0, 3.0, 5.0, 2.0, 4.0, 1.0, 3.0, 0.0, 2.0, 1.0 },
  };
  std::vector<std::string> names = { "a", "b", "c", "d", "e", "f" };
  const int band = 3;
  const int N = static_cast<int>(vecs.size());

  for (double penalty : { 0.0, 0.5, 1.0, 5.0, 100.0 }) {
    auto prob_brute = make_adtw_problem(vecs, names, penalty, band);
    prob_brute.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
    prob_brute.fillDistanceMatrix();

    auto prob_pruned = make_adtw_problem(vecs, names, penalty, band);
    dtwc::core::fill_distance_matrix_pruned(prob_pruned, band);

    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        INFO("penalty=" << penalty << " i=" << i << " j=" << j);
        REQUIRE_THAT(prob_pruned.distByInd(i, j),
                     WithinAbs(prob_brute.distByInd(i, j), 1e-10));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// INT-5: Early-abandon sentinel correctness — maxValue returned by ADTW when
//   early_abandon fires is detected by the >= inf*0.5 check, triggering a
//   re-run.  Verify that no pair in the distance matrix gets a sentinel value.
// ---------------------------------------------------------------------------
TEST_CASE("ADTW pruned matrix has no maxValue sentinel entries",
          "[adtw][pruned_matrix][integration][sentinel]")
{
  std::vector<std::vector<double>> vecs = {
    { 10.0,  0.0, 10.0,  0.0, 10.0 },
    {  0.0, 10.0,  0.0, 10.0,  0.0 },
    {  5.0,  5.0,  5.0,  5.0,  5.0 },
    {  1.0,  9.0,  2.0,  8.0,  3.0 },
    {  9.0,  1.0,  8.0,  2.0,  7.0 },
  };
  std::vector<std::string> names = { "a", "b", "c", "d", "e" };
  const int band = 2;
  const double penalty = 1.0;
  const int N = static_cast<int>(vecs.size());
  constexpr double half_max = std::numeric_limits<double>::max() * 0.5;

  auto prob = make_adtw_problem(vecs, names, penalty, band);
  dtwc::core::fill_distance_matrix_pruned(prob, band);

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      const double d = prob.distByInd(i, j);
      INFO("i=" << i << " j=" << j << " d=" << d);
      // No entry should be the early-abandon sentinel
      REQUIRE(d < half_max);
      // Self-distance must be zero
      if (i == j) REQUIRE_THAT(d, WithinAbs(0.0, 1e-15));
      // Non-negative
      REQUIRE(d >= 0.0);
    }
  }
}
