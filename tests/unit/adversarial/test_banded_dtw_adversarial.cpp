/**
 * @file test_banded_dtw_adversarial.cpp
 * @brief Adversarial tests for banded DTW boundary conditions and rolling buffer correctness.
 *
 * Tests are derived from the Sakoe-Chiba band specification:
 *   H. Sakoe and S. Chiba, "Dynamic programming algorithm optimization for spoken
 *   word recognition". IEEE Trans. Acoustics, Speech, Signal Processing, 26(1), 43-49 (1978).
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <numeric>
#include <utility>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using data_t = double;

namespace {

// Generate a random series of given length using the provided RNG.
std::vector<data_t> random_series(std::mt19937 &rng, int len, double lo = -10.0, double hi = 10.0)
{
  std::uniform_real_distribution<data_t> dist(lo, hi);
  std::vector<data_t> v(static_cast<size_t>(len));
  for (auto &val : v)
    val = dist(rng);
  return v;
}

// Independent full-matrix oracle for the canonical Sakoe-Chiba adjustment
// window |i-j| <= band. This intentionally shares no production row-bound or
// rolling-buffer helper with dtwBanded.
data_t canonical_banded_oracle(const std::vector<data_t> &x,
                               const std::vector<data_t> &y,
                               int band,
                               bool squared)
{
  const auto n = x.size();
  const auto m = y.size();
  const auto stride = m + 1;
  const auto inf = std::numeric_limits<data_t>::infinity();
  std::vector<data_t> dp((n + 1) * (m + 1), inf);
  dp[0] = 0.0;

  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t j = 0; j < m; ++j) {
      const auto diagonal_offset = (i > j) ? (i - j) : (j - i);
      if (diagonal_offset > static_cast<std::size_t>(band)) continue;

      auto local = std::abs(x[i] - y[j]);
      if (squared) local *= local;
      const auto diagonal = dp[i * stride + j];
      const auto up = dp[i * stride + (j + 1)];
      const auto left = dp[(i + 1) * stride + j];
      dp[(i + 1) * stride + (j + 1)] =
          local + std::min(diagonal, std::min(up, left));
    }
  }

  return dp[n * stride + m];
}

struct ExhaustivePathResult {
  std::size_t path_count{0};
  data_t min_l1{std::numeric_limits<data_t>::infinity()};
  data_t min_squared_l2{std::numeric_limits<data_t>::infinity()};
};

// Third arbiter: enumerate monotone paths explicitly. This uses neither
// dynamic programming nor a production bound/buffer helper.
ExhaustivePathResult exhaustive_path_oracle(const std::vector<data_t> &x,
                                            const std::vector<data_t> &y,
                                            int band)
{
  ExhaustivePathResult result;
  if (x.empty() || y.empty() || band < 0) return result;

  const auto visit = [&](auto &&self, std::size_t i, std::size_t j,
                         data_t l1, data_t squared_l2) -> void {
    const auto offset = (i > j) ? (i - j) : (j - i);
    if (offset > static_cast<std::size_t>(band)) return;

    const auto delta = std::abs(x[i] - y[j]);
    l1 += delta;
    squared_l2 += delta * delta;

    if (i + 1 == x.size() && j + 1 == y.size()) {
      ++result.path_count;
      result.min_l1 = std::min(result.min_l1, l1);
      result.min_squared_l2 = std::min(result.min_squared_l2, squared_l2);
      return;
    }

    if (i + 1 < x.size()) self(self, i + 1, j, l1, squared_l2);
    if (j + 1 < y.size()) self(self, i, j + 1, l1, squared_l2);
    if (i + 1 < x.size() && j + 1 < y.size())
      self(self, i + 1, j + 1, l1, squared_l2);
  };

  visit(visit, 0, 0, 0.0, 0.0);
  return result;
}

} // anonymous namespace

// ===========================================================================
// Area 1: Banded DTW Boundary Conditions
// ===========================================================================

TEST_CASE("Banded DTW: band < 0 falls back to full DTW", "[dtwBanded][boundary]")
{
  std::mt19937 rng(42);

  SECTION("Short equal-length series") {
    auto x = random_series(rng, 20);
    auto y = random_series(rng, 20);
    const auto banded = dtwc::dtwBanded<data_t>(x, y, -1);
    const auto full = dtwc::dtwFull_L<data_t>(x, y);
    REQUIRE_THAT(banded, WithinAbs(full, 1e-12));
  }

  SECTION("Unequal-length series") {
    auto x = random_series(rng, 15);
    auto y = random_series(rng, 30);
    const auto banded = dtwc::dtwBanded<data_t>(x, y, -1);
    const auto full = dtwc::dtwFull_L<data_t>(x, y);
    REQUIRE_THAT(banded, WithinAbs(full, 1e-12));
  }

  SECTION("band = -100 also falls back") {
    auto x = random_series(rng, 10);
    auto y = random_series(rng, 10);
    const auto banded = dtwc::dtwBanded<data_t>(x, y, -100);
    const auto full = dtwc::dtwFull_L<data_t>(x, y);
    REQUIRE_THAT(banded, WithinAbs(full, 1e-12));
  }
}

TEST_CASE("Banded DTW: canonical Sakoe-Chiba unequal-length oracle",
          "[dtwBanded][boundary][D1][oracle]")
{
  // Non-degenerate fixture registered before execution in
  // .claude/baselines/2026-07-23-r2-d1-dtw.md.
  const std::vector<data_t> x{0.0, 1.0, 0.0, 2.0, 0.0};
  const std::vector<data_t> y{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0};
  constexpr auto max_value = std::numeric_limits<data_t>::max();

  SECTION("Independent exact ledger") {
    REQUIRE(std::isinf(canonical_banded_oracle(x, y, 0, false)));
    REQUIRE(std::isinf(canonical_banded_oracle(x, y, 1, false)));
    REQUIRE(canonical_banded_oracle(x, y, 2, false) == 5.0);
    REQUIRE(canonical_banded_oracle(x, y, 3, false) == 3.0);
    REQUIRE(canonical_banded_oracle(x, y, 7, false) == 3.0);

    REQUIRE(std::isinf(canonical_banded_oracle(x, y, 0, true)));
    REQUIRE(std::isinf(canonical_banded_oracle(x, y, 1, true)));
    REQUIRE(canonical_banded_oracle(x, y, 2, true) == 9.0);
    REQUIRE(canonical_banded_oracle(x, y, 3, true) == 5.0);
    REQUIRE(canonical_banded_oracle(x, y, 7, true) == 5.0);
  }

  SECTION("Exhaustive monotone-path enumeration is an independent arbiter") {
    struct Expected {
      int band;
      std::size_t paths;
      data_t l1;
      data_t squared_l2;
    };
    constexpr Expected ledger[] = {
        {2, 696, 5.0, 9.0},
        {3, 1143, 3.0, 5.0},
        {7, 1289, 3.0, 5.0}
    };

    for (const auto &expected : ledger) {
      INFO("band=" << expected.band);
      const auto actual = exhaustive_path_oracle(x, y, expected.band);
      REQUIRE(actual.path_count == expected.paths);
      REQUIRE(actual.min_l1 == expected.l1);
      REQUIRE(actual.min_squared_l2 == expected.squared_l2);
    }
  }

  SECTION("A band narrower than the endpoint offset has no path") {
    for (const int band : {0, 1}) {
      INFO("band=" << band);
      REQUIRE(dtwc::dtwBanded<data_t>(x, y, band) == max_value);
      REQUIRE(dtwc::dtwBanded<data_t>(y, x, band) == max_value);
      REQUIRE(dtwc::dtwBanded<data_t>(
                  x, y, band, -1.0, dtwc::core::MetricType::SquaredL2)
              == max_value);
      REQUIRE(dtwc::dtwBanded<data_t>(
                  y, x, band, -1.0, dtwc::core::MetricType::SquaredL2)
              == max_value);
    }
  }

  SECTION("Public banded routes equal the independent oracle") {
    for (const int band : {2, 3, 7}) {
      INFO("band=" << band);
      const auto expected_l1 = canonical_banded_oracle(x, y, band, false);
      const auto expected_sq = canonical_banded_oracle(x, y, band, true);

      REQUIRE(dtwc::dtwBanded<data_t>(x, y, band) == expected_l1);
      REQUIRE(dtwc::dtwBanded<data_t>(y, x, band) == expected_l1);
      REQUIRE(dtwc::dtwBanded<data_t>(
                  x, y, band, -1.0, dtwc::core::MetricType::SquaredL2)
              == expected_sq);
      REQUIRE(dtwc::dtwBanded<data_t>(
                  y, x, band, -1.0, dtwc::core::MetricType::SquaredL2)
              == expected_sq);
    }
  }

  SECTION("Unbanded public routes match the registered full-DTW values") {
    REQUIRE(dtwc::dtwFull<data_t>(x, y) == 3.0);
    REQUIRE(dtwc::dtwFull_L<data_t>(x, y) == 3.0);
    REQUIRE(dtwc::dtwBanded<data_t>(x, y, -1) == 3.0);
    REQUIRE(dtwc::dtwFull<data_t>(
                x, y, dtwc::core::MetricType::SquaredL2)
            == 5.0);
    REQUIRE(dtwc::dtwFull_L<data_t>(
                x, y, -1.0, dtwc::core::MetricType::SquaredL2)
            == 5.0);
    REQUIRE(dtwc::dtwBanded<data_t>(
                x, y, -1, -1.0, dtwc::core::MetricType::SquaredL2)
            == 5.0);
  }

  SECTION("Singleton unequal lengths obey endpoint feasibility") {
    const std::vector<data_t> singleton{0.0};
    const std::vector<data_t> longer{1.0, 2.0, 3.0};

    REQUIRE(dtwc::dtwBanded<data_t>(singleton, longer, 1) == max_value);
    REQUIRE(dtwc::dtwBanded<data_t>(longer, singleton, 1) == max_value);
    REQUIRE(dtwc::dtwBanded<data_t>(
                singleton, longer, 1, -1.0,
                dtwc::core::MetricType::SquaredL2)
            == max_value);
    REQUIRE(dtwc::dtwBanded<data_t>(singleton, longer, 2) == 6.0);
    REQUIRE(dtwc::dtwBanded<data_t>(longer, singleton, 2) == 6.0);
    REQUIRE(dtwc::dtwBanded<data_t>(
                singleton, longer, 2, -1.0,
                dtwc::core::MetricType::SquaredL2)
            == 14.0);
  }

  SECTION("Dependent multivariate wrapper cannot bypass feasibility") {
    constexpr std::size_t ndim = 2;
    constexpr auto widest_band = std::numeric_limits<int>::max();
    const std::vector<data_t> singleton{0.0, 10.0};
    const std::vector<data_t> longer{
        1.0, 11.0,
        2.0, 12.0,
        3.0, 13.0
    };

    REQUIRE(dtwc::dtwBanded_mv<data_t>(
                singleton.data(), 1, longer.data(), 3, ndim, 1)
            == max_value);
    REQUIRE(dtwc::dtwBanded_mv<data_t>(
                longer.data(), 3, singleton.data(), 1, ndim, 1)
            == max_value);
    REQUIRE(dtwc::dtwBanded_mv<data_t>(
                singleton.data(), 1, longer.data(), 3, ndim, 1, -1.0,
                dtwc::core::MetricType::SquaredL2)
            == max_value);
    REQUIRE(dtwc::dtwBanded_mv<data_t>(
                singleton.data(), 1, longer.data(), 3, ndim, 2)
            == 12.0);
    REQUIRE(dtwc::dtwBanded_mv<data_t>(
                longer.data(), 3, singleton.data(), 1, ndim, 2)
            == 12.0);
    REQUIRE(dtwc::dtwBanded_mv<data_t>(
                singleton.data(), 1, longer.data(), 3, ndim, 2, -1.0,
                dtwc::core::MetricType::SquaredL2)
            == 28.0);
    REQUIRE(dtwc::dtwBanded_mv<data_t>(
                singleton.data(), 1, longer.data(), 3, ndim, widest_band)
            == 12.0);
    REQUIRE(dtwc::dtwBanded_mv<data_t>(
                singleton.data(), 1, longer.data(), 3, ndim, widest_band, -1.0,
                dtwc::core::MetricType::SquaredL2)
            == 28.0);
  }

  SECTION("Maximum int band covers the matrix without signed overflow") {
    constexpr auto widest_band = std::numeric_limits<int>::max();
    REQUIRE(dtwc::dtwBanded<data_t>(x, y, widest_band) == 3.0);
    REQUIRE(dtwc::dtwBanded<data_t>(
                x, y, widest_band, -1.0,
                dtwc::core::MetricType::SquaredL2)
            == 5.0);
  }

  SECTION("Band bounds retain size_t indices beyond INT_MAX") {
    constexpr auto widest_band = std::numeric_limits<int>::max();
    constexpr auto int_max = static_cast<std::size_t>(widest_band);
    constexpr auto row = int_max + 3;
    constexpr auto columns = int_max + 10;

    REQUIRE((dtwc::core::dtw_band_bounds(widest_band, row, columns)
             == std::pair<std::size_t, std::size_t>{3, columns}));
    REQUIRE((dtwc::core::dtw_band_bounds(0, row, columns)
             == std::pair<std::size_t, std::size_t>{row, row + 1}));
  }

  SECTION("DTW-AROW public wrapper shares canonical path feasibility") {
    constexpr auto widest_band = std::numeric_limits<int>::max();
    const std::vector<data_t> singleton{0.0};
    const std::vector<data_t> longer{1.0, 2.0, 3.0};

    REQUIRE(dtwc::dtwAROW_banded<data_t>(singleton, longer, 1) == max_value);
    REQUIRE(dtwc::dtwAROW_banded<data_t>(longer, singleton, 1) == max_value);
    REQUIRE(dtwc::dtwAROW_banded<data_t>(
                singleton, longer, 1, dtwc::core::MetricType::SquaredL2)
            == max_value);
    REQUIRE(dtwc::dtwAROW_banded<data_t>(singleton, longer, 2) == 6.0);
    REQUIRE(dtwc::dtwAROW_banded<data_t>(
                singleton, longer, 2, dtwc::core::MetricType::SquaredL2)
            == 14.0);
    REQUIRE(dtwc::dtwAROW_banded<data_t>(
                singleton, longer, widest_band)
            == 6.0);
    REQUIRE(dtwc::dtwAROW_banded<data_t>(
                singleton, longer, widest_band,
                dtwc::core::MetricType::SquaredL2)
            == 14.0);
  }
}

TEST_CASE("Banded DTW: band = 0 forces diagonal alignment for equal-length series", "[dtwBanded][boundary]")
{
  // With band=0 and equal lengths, the only valid path is the diagonal.
  // Cost should equal sum of |x[i] - y[i]|.

  SECTION("Equal-length series, length 2") {
    // Diagonal path cost = |x[0]-y[0]| + |x[1]-y[1]|.
    std::vector<data_t> x{1.0, 5.0};
    std::vector<data_t> y{3.0, 2.0};
    const auto result = dtwc::dtwBanded<data_t>(x, y, 0);
    const data_t diagonal_cost = std::abs(1.0 - 3.0) + std::abs(5.0 - 2.0); // 2 + 3 = 5
    REQUIRE_THAT(result, WithinAbs(diagonal_cost, 1e-12));
  }

  SECTION("Equal-length series, length 5") {
    std::vector<data_t> x{1.0, 2.0, 3.0, 4.0, 5.0};
    std::vector<data_t> y{5.0, 4.0, 3.0, 2.0, 1.0};
    const auto result = dtwc::dtwBanded<data_t>(x, y, 0);
    // Diagonal cost: |1-5| + |2-4| + |3-3| + |4-2| + |5-1| = 4+2+0+2+4 = 12
    const data_t diagonal_cost = 12.0;
    REQUIRE_THAT(result, WithinAbs(diagonal_cost, 1e-12));
  }
}

TEST_CASE("Banded DTW: band >= max(len) equals full DTW", "[dtwBanded][boundary]")
{
  std::mt19937 rng(42);

  for (int trial = 0; trial < 10; ++trial) {
    const int len_x = 5 + (trial * 3);
    const int len_y = 8 + (trial * 2);
    auto x = random_series(rng, len_x);
    auto y = random_series(rng, len_y);

    const auto banded = dtwc::dtwBanded<data_t>(x, y, 10000);
    const auto full = dtwc::dtwFull_L<data_t>(x, y);
    REQUIRE_THAT(banded, WithinAbs(full, 1e-10));
  }
}

TEST_CASE("Banded DTW: larger band gives <= cost (monotonicity)", "[dtwBanded][boundary]")
{
  // More paths available with larger band => cost can only decrease or stay the same.
  std::mt19937 rng(42);

  SECTION("Systematic band sweep") {
    auto x = random_series(rng, 50);
    auto y = random_series(rng, 50);

    data_t prev_cost = dtwc::dtwBanded<data_t>(x, y, 0);
    for (int band = 1; band <= 50; ++band) {
      const auto cost = dtwc::dtwBanded<data_t>(x, y, band);
      REQUIRE(cost <= prev_cost + 1e-12); // allow tiny floating-point noise
      prev_cost = cost;
    }
  }

  SECTION("Unequal lengths") {
    auto x = random_series(rng, 30);
    auto y = random_series(rng, 60);

    data_t prev_cost = dtwc::dtwBanded<data_t>(x, y, 0);
    for (int band = 1; band <= 60; ++band) {
      const auto cost = dtwc::dtwBanded<data_t>(x, y, band);
      REQUIRE(cost <= prev_cost + 1e-12);
      prev_cost = cost;
    }
  }
}

TEST_CASE("Banded DTW: band=1 allows +/-1 diagonal deviation", "[dtwBanded][boundary]")
{
  // Construct series where optimal full DTW path needs exactly +/-1 deviation from diagonal.
  // x = [0, 0, 1, 1, 0]
  // y = [0, 1, 1, 0, 0]
  // The optimal alignment shifts y[1]=1 to match x[2]=1, which is a +1 deviation.
  // Band=1 should allow this. Band=0 (diagonal only) should give higher cost.
  std::vector<data_t> x{0.0, 0.0, 1.0, 1.0, 0.0};
  std::vector<data_t> y{0.0, 1.0, 1.0, 0.0, 0.0};

  const auto cost_band0 = dtwc::dtwBanded<data_t>(x, y, 0);
  const auto cost_band1 = dtwc::dtwBanded<data_t>(x, y, 1);
  const auto cost_full = dtwc::dtwFull_L<data_t>(x, y);

  // Band=1 should be at least as good as band=0
  REQUIRE(cost_band1 <= cost_band0 + 1e-12);

  // Band=1 should match full DTW for this gentle shift
  // (full DTW path only needs +/-1 deviation)
  REQUIRE_THAT(cost_band1, WithinAbs(cost_full, 1e-12));
}

TEST_CASE("Banded DTW: unequal lengths respect endpoint feasibility", "[dtwBanded][boundary]")
{
  constexpr auto max_value = std::numeric_limits<data_t>::max();

  SECTION("Short vs long") {
    std::vector<data_t> x{1.0, 2.0, 3.0};
    std::vector<data_t> y{1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

    REQUIRE(dtwc::dtwBanded<data_t>(x, y, 2) == max_value);
    REQUIRE(dtwc::dtwBanded<data_t>(x, y, 4) < max_value);
  }

  SECTION("Length ratio 1:10") {
    std::mt19937 rng(42);
    auto x = random_series(rng, 10);
    auto y = random_series(rng, 100);

    REQUIRE(dtwc::dtwBanded<data_t>(x, y, 5) == max_value);
    REQUIRE(dtwc::dtwBanded<data_t>(x, y, 90) < max_value);
  }
}

TEST_CASE("Banded DTW: very short series with band", "[dtwBanded][boundary]")
{
  // Single-element series
  std::vector<data_t> x{1.0};
  std::vector<data_t> y{2.0};
  const auto result = dtwc::dtwBanded<data_t>(x, y, 10);
  REQUIRE_THAT(result, WithinAbs(1.0, 1e-12));

  // Single-element, same value
  std::vector<data_t> a{5.0};
  std::vector<data_t> b{5.0};
  REQUIRE_THAT(dtwc::dtwBanded<data_t>(a, b, 10), WithinAbs(0.0, 1e-12));

  // Single-element vs multi-element
  std::vector<data_t> c{0.0};
  std::vector<data_t> d{1.0, 2.0, 3.0};
  const auto result2 = dtwc::dtwBanded<data_t>(c, d, 10);
  // Full DTW of {0} vs {1,2,3}: cost = |0-1| + |0-2| + |0-3| = 1+2+3 = 6
  REQUIRE_THAT(result2, WithinAbs(6.0, 1e-12));
}

// ===========================================================================
// Area 2: Rolling Buffer Memory Correctness
// ===========================================================================

TEST_CASE("Rolling buffer vs full matrix: random pairs with large band", "[dtwBanded][buffer]")
{
  // When band covers the full matrix, dtwBanded must equal dtwFull_L.
  std::mt19937 rng(42);

  for (int trial = 0; trial < 50; ++trial) {
    std::uniform_int_distribution<int> len_dist(2, 100);
    const int len_x = len_dist(rng);
    const int len_y = len_dist(rng);
    auto x = random_series(rng, len_x);
    auto y = random_series(rng, len_y);

    const int big_band = std::max(len_x, len_y) + 10;
    const auto banded = dtwc::dtwBanded<data_t>(x, y, big_band);
    const auto full = dtwc::dtwFull_L<data_t>(x, y);

    REQUIRE_THAT(banded, WithinAbs(full, 1e-10));
  }
}

TEST_CASE("Large series with small band: no crash, reasonable memory", "[dtwBanded][buffer]")
{
  // N=5000, band=5. The rolling buffer should handle this without excessive memory.
  std::mt19937 rng(42);
  auto x = random_series(rng, 5000);
  auto y = random_series(rng, 5000);

  const auto result = dtwc::dtwBanded<data_t>(x, y, 5);
  REQUIRE(std::isfinite(result));
  REQUIRE(result >= 0.0);
}

TEST_CASE("Repeated calls with different sizes: thread-local buffer reuse", "[dtwBanded][buffer]")
{
  // Call dtwBanded with varying sizes to exercise thread-local buffer resizing.
  std::mt19937 rng(42);

  // Call 1: 100x100, band=5
  auto x1 = random_series(rng, 100);
  auto y1 = random_series(rng, 100);
  const auto r1 = dtwc::dtwBanded<data_t>(x1, y1, 5);
  REQUIRE(std::isfinite(r1));
  REQUIRE(r1 >= 0.0);

  // Call 2: 50x50, band=10
  auto x2 = random_series(rng, 50);
  auto y2 = random_series(rng, 50);
  const auto r2 = dtwc::dtwBanded<data_t>(x2, y2, 10);
  REQUIRE(std::isfinite(r2));
  REQUIRE(r2 >= 0.0);

  // Call 3: 200x200, band=3
  auto x3 = random_series(rng, 200);
  auto y3 = random_series(rng, 200);
  const auto r3 = dtwc::dtwBanded<data_t>(x3, y3, 3);
  REQUIRE(std::isfinite(r3));
  REQUIRE(r3 >= 0.0);

  // Verify correctness by comparing with full DTW (large band)
  const auto r1_full = dtwc::dtwBanded<data_t>(x1, y1, 200);
  const auto r1_ref = dtwc::dtwFull_L<data_t>(x1, y1);
  REQUIRE_THAT(r1_full, WithinAbs(r1_ref, 1e-10));

  const auto r2_full = dtwc::dtwBanded<data_t>(x2, y2, 200);
  const auto r2_ref = dtwc::dtwFull_L<data_t>(x2, y2);
  REQUIRE_THAT(r2_full, WithinAbs(r2_ref, 1e-10));

  const auto r3_full = dtwc::dtwBanded<data_t>(x3, y3, 200);
  const auto r3_ref = dtwc::dtwFull_L<data_t>(x3, y3);
  REQUIRE_THAT(r3_full, WithinAbs(r3_ref, 1e-10));
}

TEST_CASE("Symmetry with band: dtwBanded(x,y,band) == dtwBanded(y,x,band)", "[dtwBanded][buffer]")
{
  std::mt19937 rng(42);

  SECTION("Equal-length series, various bands") {
    auto x = random_series(rng, 40);
    auto y = random_series(rng, 40);

    for (int band : {0, 1, 2, 5, 10, 20, 100}) {
      const auto xy = dtwc::dtwBanded<data_t>(x, y, band);
      const auto yx = dtwc::dtwBanded<data_t>(y, x, band);
      REQUIRE_THAT(xy, WithinAbs(yx, 1e-12));
    }
  }

  SECTION("Unequal-length series, various bands") {
    auto x = random_series(rng, 25);
    auto y = random_series(rng, 50);

    for (int band : {0, 1, 3, 10, 30, 100}) {
      const auto xy = dtwc::dtwBanded<data_t>(x, y, band);
      const auto yx = dtwc::dtwBanded<data_t>(y, x, band);
      REQUIRE_THAT(xy, WithinAbs(yx, 1e-12));
    }
  }
}

TEST_CASE("Non-negativity with band: dtwBanded(x,y,band) >= 0", "[dtwBanded][buffer]")
{
  std::mt19937 rng(42);

  for (int trial = 0; trial < 30; ++trial) {
    std::uniform_int_distribution<int> len_dist(1, 100);
    std::uniform_int_distribution<int> band_dist(0, 50);
    const int len_x = len_dist(rng);
    const int len_y = len_dist(rng);
    const int band = band_dist(rng);

    auto x = random_series(rng, len_x);
    auto y = random_series(rng, len_y);

    const auto result = dtwc::dtwBanded<data_t>(x, y, band);
    REQUIRE(result >= 0.0);
  }

  // Identity: distance to self is zero
  auto x = random_series(rng, 50);
  REQUIRE_THAT(dtwc::dtwBanded<data_t>(x, x, 5), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(dtwc::dtwBanded<data_t>(x, x, 0), WithinAbs(0.0, 1e-12));
}
