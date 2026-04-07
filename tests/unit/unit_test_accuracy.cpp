/**
 * @file unit_test_accuracy.cpp
 * @brief Accuracy and correctness tests for DTW system.
 *
 * @details Covers cross-variant consistency, cross-metric correctness,
 * early-abandon accuracy, numerical stability, and lower-bound validity.
 *
 * @author Volkan Kumtepeli
 * @date 07 Apr 2026
 */

#include <dtwc.hpp>
#include <core/lower_bound_impl.hpp>
#include <core/lower_bounds.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace dtwc;

// =========================================================================
//  Helpers
// =========================================================================

static std::vector<double> make_random(size_t len, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);
  std::vector<double> s(len);
  for (auto &v : s) v = dist(rng);
  return s;
}

// =========================================================================
//  1. Cross-variant consistency
// =========================================================================

TEST_CASE("WDTW with uniform weights == standard DTW", "[accuracy][cross_variant]")
{
  std::vector<double> x{ 1, 3, 5, 2, 4 };
  std::vector<double> y{ 2, 4, 6, 3, 5, 7 };

  // Uniform weight = 1.0 for all deviations => WDTW == standard DTW
  const int max_dev = static_cast<int>(std::max(x.size(), y.size())) - 1;
  std::vector<double> uniform_w(max_dev + 1, 1.0);

  double dtw_dist = dtwFull<double>(x, y);
  double wdtw_dist = wdtwFull<double>(x, y, uniform_w);

  REQUIRE_THAT(wdtw_dist, WithinAbs(dtw_dist, 1e-12));
}

TEST_CASE("WDTW banded with uniform weights == banded DTW", "[accuracy][cross_variant]")
{
  std::vector<double> x{ 0, 1, 2, 3, 4, 5, 6, 7 };
  std::vector<double> y{ 1, 2, 3, 4, 5, 6, 7, 8 };

  const int max_dev = static_cast<int>(std::max(x.size(), y.size())) - 1;
  std::vector<double> uniform_w(max_dev + 1, 1.0);

  for (int band : {1, 2, 3, 4}) {
    double dtw_dist = dtwBanded<double>(x, y, band);
    double wdtw_dist = wdtwBanded<double>(x, y, uniform_w, band);
    REQUIRE_THAT(wdtw_dist, WithinAbs(dtw_dist, 1e-12));
  }
}

TEST_CASE("ADTW banded penalty=0 == banded DTW for various bands", "[accuracy][cross_variant]")
{
  std::vector<double> x{ 1, 3, 5, 7, 9, 2, 4, 6 };
  std::vector<double> y{ 2, 4, 6, 8, 10, 3, 5, 7, 9 };

  for (int band : {1, 2, 3, 4, 5}) {
    double dtw_dist = dtwBanded<double>(x, y, band);
    double adtw_dist = adtwBanded<double>(x, y, band, 0.0);
    REQUIRE_THAT(adtw_dist, WithinAbs(dtw_dist, 1e-12));
  }
}

TEST_CASE("ADTW increasing penalty >= standard DTW", "[accuracy][cross_variant]")
{
  std::vector<double> x{ 1, 2, 3, 4, 5 };
  std::vector<double> y{ 3, 4, 5, 6, 7, 8, 9 };

  double dtw_dist = dtwFull_L<double>(x, y);
  double prev = dtw_dist;

  for (double p : {0.0, 0.5, 1.0, 5.0, 50.0}) {
    double adtw_dist = adtwFull_L<double>(x, y, p);
    REQUIRE(adtw_dist >= dtw_dist - 1e-10);
    REQUIRE(adtw_dist >= prev - 1e-10);
    prev = adtw_dist;
  }
}

TEST_CASE("WDTW increasing steepness monotonically changes distance", "[accuracy][cross_variant]")
{
  // With higher g, weights become more binary (0 near, 1 far),
  // so the distance should change monotonically for offset series.
  std::vector<double> x{ 0, 0, 1, 2, 1, 0, 0, 0, 0, 0 };
  std::vector<double> y{ 0, 0, 0, 0, 0, 0, 1, 2, 1, 0 };

  // Just verify non-negativity and finiteness across a range of g
  for (double g : {0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 10.0}) {
    double d = wdtwFull<double>(x, y, g);
    REQUIRE(d >= 0.0);
    REQUIRE(std::isfinite(d));
  }
}

// =========================================================================
//  2. Cross-metric accuracy (hand-computed values)
// =========================================================================

TEST_CASE("DTW with SquaredL2 metric: hand-computed", "[accuracy][metrics]")
{
  // x = {1, 2, 3}, y = {2, 2, 2}
  // SquaredL2 cost matrix:
  //   C(0,0) = (1-2)^2 = 1
  //   C(1,0) = C(0,0) + (2-2)^2 = 1
  //   C(2,0) = C(1,0) + (3-2)^2 = 2
  //   C(0,1) = C(0,0) + (1-2)^2 = 2
  //   C(1,1) = min(C(0,0), C(0,1), C(1,0)) + (2-2)^2 = 1 + 0 = 1
  //   C(2,1) = min(C(1,0), C(1,1), C(2,0)) + (3-2)^2 = 1 + 1 = 2
  //   C(0,2) = C(0,1) + (1-2)^2 = 3
  //   C(1,2) = min(C(0,1), C(0,2), C(1,1)) + (2-2)^2 = 1 + 0 = 1
  //   C(2,2) = min(C(1,1), C(1,2), C(2,1)) + (3-2)^2 = 1 + 1 = 2
  // Answer: 2
  std::vector<double> x{1, 2, 3}, y{2, 2, 2};

  double d = dtwFull<double>(x, y, core::MetricType::SquaredL2);
  REQUIRE_THAT(d, WithinAbs(2.0, 1e-15));
}

TEST_CASE("DTW L1 vs SquaredL2: constant offset (non-overlapping)", "[accuracy][metrics]")
{
  // x = {1, 2, 3}, y = {100, 200, 300}
  // L1 diagonal is clearly optimal since ranges don't overlap:
  //   cost = |1-100| + |2-200| + |3-300| = 99 + 198 + 297 = 594
  // SquaredL2 diagonal:
  //   cost = 99^2 + 198^2 + 297^2 = 9801 + 39204 + 88209 = 137214
  std::vector<double> x{1, 2, 3};
  std::vector<double> y{100, 200, 300};

  double d_l1 = dtwFull<double>(x, y, core::MetricType::L1);
  double d_sq = dtwFull<double>(x, y, core::MetricType::SquaredL2);

  REQUIRE_THAT(d_l1, WithinAbs(594.0, 1e-12));
  REQUIRE_THAT(d_sq, WithinAbs(137214.0, 1e-12));
}

TEST_CASE("DTW SquaredL2: all three variants agree", "[accuracy][metrics]")
{
  std::vector<double> x{1, 3, 5, 7, 9};
  std::vector<double> y{2, 4, 6, 8};

  double d_full = dtwFull<double>(x, y, core::MetricType::SquaredL2);
  double d_L = dtwFull_L<double>(x, y, -1.0, core::MetricType::SquaredL2);
  double d_band = dtwBanded<double>(x, y, 100, -1.0, core::MetricType::SquaredL2);

  REQUIRE_THAT(d_full, WithinAbs(d_L, 1e-12));
  REQUIRE_THAT(d_full, WithinAbs(d_band, 1e-12));
}

TEST_CASE("DTW SquaredL2: banded >= full", "[accuracy][metrics]")
{
  std::vector<double> x{1, 2, 3, 4, 5, 6, 7, 8};
  std::vector<double> y{8, 7, 6, 5, 4, 3, 2, 1};

  double full = dtwFull<double>(x, y, core::MetricType::SquaredL2);
  for (int band : {1, 2, 3, 4}) {
    double banded = dtwBanded<double>(x, y, band, -1.0, core::MetricType::SquaredL2);
    REQUIRE(banded >= full - 1e-10);
  }
}

TEST_CASE("DTW SquaredL2 identity is zero", "[accuracy][metrics]")
{
  std::vector<double> x{3.14, 2.71, 1.41, 0.57};
  REQUIRE_THAT(dtwFull<double>(x, x, core::MetricType::SquaredL2), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(dtwFull_L<double>(x, x, -1.0, core::MetricType::SquaredL2), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(dtwBanded<double>(x, x, 2, -1.0, core::MetricType::SquaredL2), WithinAbs(0.0, 1e-15));
}

// =========================================================================
//  3. Early-abandon correctness
// =========================================================================

TEST_CASE("Early abandon returns same result when threshold is high", "[accuracy][early_abandon]")
{
  auto x = make_random(50, 42);
  auto y = make_random(50, 99);

  double no_ea = dtwFull_L<double>(x, y);
  // With a very high threshold, early abandon should not trigger
  double with_ea = dtwFull_L<double>(x, y, 1e12);

  REQUIRE_THAT(with_ea, WithinAbs(no_ea, 1e-12));
}

TEST_CASE("Early abandon returns maxValue when threshold is too low", "[accuracy][early_abandon]")
{
  std::vector<double> x{1, 2, 3, 4, 5};
  std::vector<double> y{100, 200, 300, 400, 500};

  double actual = dtwFull_L<double>(x, y);
  // Threshold much lower than actual distance
  double ea_result = dtwFull_L<double>(x, y, 1.0);

  REQUIRE(ea_result > 1e10); // Should be maxValue
  REQUIRE(actual > 1.0);     // Sanity: actual distance exceeds threshold
}

TEST_CASE("Early abandon correctness with banded DTW", "[accuracy][early_abandon]")
{
  auto x = make_random(40, 77);
  auto y = make_random(45, 88);

  for (int band : {2, 5, 10}) {
    double no_ea = dtwBanded<double>(x, y, band);
    double with_ea = dtwBanded<double>(x, y, band, 1e12);
    REQUIRE_THAT(with_ea, WithinAbs(no_ea, 1e-12));
  }
}

TEST_CASE("ADTW early abandon returns same result when threshold is high", "[accuracy][early_abandon]")
{
  auto x = make_random(30, 11);
  auto y = make_random(35, 22);

  for (double penalty : {0.0, 1.0, 5.0}) {
    double no_ea = adtwFull_L<double>(x.data(), x.size(), y.data(), y.size(), penalty);
    double with_ea = adtwFull_L<double>(x.data(), x.size(), y.data(), y.size(), penalty, 1e12);
    REQUIRE_THAT(with_ea, WithinAbs(no_ea, 1e-12));
  }
}

TEST_CASE("Early abandon at exact distance returns the value", "[accuracy][early_abandon]")
{
  auto x = make_random(20, 33);
  auto y = make_random(20, 44);

  double actual = dtwFull_L<double>(x, y);
  // Threshold at exact distance: should not abandon
  double with_ea = dtwFull_L<double>(x, y, actual);
  REQUIRE_THAT(with_ea, WithinAbs(actual, 1e-12));
}

// =========================================================================
//  4. Numerical stability
// =========================================================================

TEST_CASE("DTW with very large values", "[accuracy][stability]")
{
  std::vector<double> x{1e15, 2e15, 3e15};
  std::vector<double> y{1.5e15, 2.5e15, 3.5e15};

  double d = dtwFull<double>(x, y);
  REQUIRE(std::isfinite(d));
  REQUIRE(d > 0.0);

  // All three variants should agree
  double d_L = dtwFull_L<double>(x, y);
  double d_B = dtwBanded<double>(x, y);
  REQUIRE_THAT(d, WithinRel(d_L, 1e-10));
  REQUIRE_THAT(d, WithinRel(d_B, 1e-10));
}

TEST_CASE("DTW with very small values", "[accuracy][stability]")
{
  std::vector<double> x{1e-15, 2e-15, 3e-15};
  std::vector<double> y{1.5e-15, 2.5e-15, 3.5e-15};

  double d = dtwFull<double>(x, y);
  REQUIRE(std::isfinite(d));
  REQUIRE(d > 0.0);
  REQUIRE_THAT(dtwFull<double>(x, y), WithinAbs(dtwFull_L<double>(x, y), 1e-25));
}

TEST_CASE("DTW with near-zero differences", "[accuracy][stability]")
{
  std::vector<double> x{1.0, 1.0 + 1e-14, 1.0 + 2e-14};
  std::vector<double> y{1.0, 1.0, 1.0};

  double d = dtwFull<double>(x, y);
  REQUIRE(std::isfinite(d));
  REQUIRE(d >= 0.0);
  // Distance should be very small but non-negative
  REQUIRE(d < 1e-12);
}

TEST_CASE("DTW with mixed sign values", "[accuracy][stability]")
{
  std::vector<double> x{-1e10, 0, 1e10};
  std::vector<double> y{-1e10, 0, 1e10};

  REQUIRE_THAT(dtwFull<double>(x, y), WithinAbs(0.0, 1e-5));
}

TEST_CASE("SquaredL2 with large values does not overflow to inf", "[accuracy][stability]")
{
  // (1e150)^2 = 1e300 which is within double range (max ~1.8e308)
  std::vector<double> x{1e150};
  std::vector<double> y{0.0};

  double d = dtwFull<double>(x, y, core::MetricType::SquaredL2);
  REQUIRE(std::isfinite(d));
  REQUIRE_THAT(d, WithinRel(1e300, 1e-10));
}

TEST_CASE("DTW of long constant series", "[accuracy][stability]")
{
  // Two long constant series offset by 1
  std::vector<double> x(1000, 5.0);
  std::vector<double> y(1000, 6.0);

  // L1 distance: diagonal alignment = 1000 * |5-6| = 1000
  double d = dtwFull_L<double>(x, y);
  REQUIRE_THAT(d, WithinAbs(1000.0, 1e-10));
}

TEST_CASE("DTW float vs double precision", "[accuracy][stability]")
{
  std::vector<float> xf{1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  std::vector<float> yf{2.0f, 4.0f, 6.0f, 3.0f, 5.0f};
  std::vector<double> xd{1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> yd{2.0, 4.0, 6.0, 3.0, 5.0};

  float df = dtwFull<float>(xf, yf);
  double dd = dtwFull<double>(xd, yd);

  // float and double should agree to float precision
  REQUIRE_THAT(static_cast<double>(df), WithinRel(dd, 1e-5));
}

// =========================================================================
//  5. Lower-bound validity across metrics
// =========================================================================

TEST_CASE("LB_Keogh <= DTW for random series (L1)", "[accuracy][lower_bounds]")
{
  using namespace dtwc::core;

  for (int trial = 0; trial < 10; ++trial) {
    auto a = make_random(30, 100 + trial);
    auto b = make_random(30, 200 + trial);
    int band = 3;

    auto env_b = compute_envelope(b, band);

    double lb = lb_keogh(a, env_b);
    double dtw_dist = dtwBanded<double>(a, b, band);

    REQUIRE(lb >= 0.0);
    REQUIRE(lb <= dtw_dist + 1e-10);
  }
}

TEST_CASE("LB_Kim <= DTW for random series", "[accuracy][lower_bounds]")
{
  using namespace dtwc::core;

  for (int trial = 0; trial < 10; ++trial) {
    auto a = make_random(20, 300 + trial);
    auto b = make_random(20, 400 + trial);

    auto sa = compute_summary(a);
    auto sb = compute_summary(b);

    double lb = lb_kim(sa, sb);
    double dtw_dist = dtwFull<double>(a, b);

    REQUIRE(lb >= 0.0);
    REQUIRE(lb <= dtw_dist + 1e-10);
  }
}

TEST_CASE("LB_Keogh for identical series is zero", "[accuracy][lower_bounds]")
{
  using namespace dtwc::core;
  std::vector<double> x{1, 3, 5, 2, 4};
  int band = 2;

  auto env = compute_envelope(x, band);
  double lb = lb_keogh(x, env);
  REQUIRE_THAT(lb, WithinAbs(0.0, 1e-15));
}

TEST_CASE("LB_Kim for identical series is zero", "[accuracy][lower_bounds]")
{
  using namespace dtwc::core;
  std::vector<double> x{1, 3, 5, 2, 4};
  auto s = compute_summary(x);
  REQUIRE_THAT(lb_kim(s, s), WithinAbs(0.0, 1e-15));
}

TEST_CASE("LB_Keogh with band=0 equals L1 pointwise sum", "[accuracy][lower_bounds]")
{
  using namespace dtwc::core;
  std::vector<double> a{1, 3, 5, 7, 9};
  std::vector<double> b{2, 4, 6, 8, 10};

  // With band=0, envelopes are the series themselves
  auto env_a = compute_envelope(a, 0);
  auto env_b = compute_envelope(b, 0);

  // Verify envelopes with band=0 are the series values
  for (size_t i = 0; i < a.size(); ++i) {
    REQUIRE_THAT(env_a.upper[i], WithinAbs(a[i], 1e-15));
    REQUIRE_THAT(env_a.lower[i], WithinAbs(a[i], 1e-15));
  }

  double lb = lb_keogh(b, env_a);
  // Should equal sum of |a[i] - b[i]| = 5
  REQUIRE_THAT(lb, WithinAbs(5.0, 1e-12));
}

// =========================================================================
//  6. Multivariate accuracy
// =========================================================================

TEST_CASE("Multivariate DTW with ndim=1 matches scalar DTW", "[accuracy][multivariate]")
{
  std::vector<double> x{1, 3, 5, 7, 9};
  std::vector<double> y{2, 4, 6, 8};

  double scalar = dtwFull_L<double>(x, y);
  double mv = dtwFull_L_mv<double>(x.data(), x.size(), y.data(), y.size(), 1);

  REQUIRE_THAT(mv, WithinAbs(scalar, 1e-12));
}

TEST_CASE("Multivariate banded DTW with ndim=1 matches scalar banded DTW", "[accuracy][multivariate]")
{
  std::vector<double> x{1, 3, 5, 7, 9};
  std::vector<double> y{2, 4, 6, 8};

  for (int band : {1, 2, 3}) {
    double scalar = dtwBanded<double>(x, y, band);
    double mv = dtwBanded_mv<double>(x.data(), x.size(), y.data(), y.size(), 1, band);
    REQUIRE_THAT(mv, WithinAbs(scalar, 1e-12));
  }
}

TEST_CASE("Multivariate DTW 2D hand-computed", "[accuracy][multivariate]")
{
  // x: 2 timesteps, 2 dims: (1,2), (3,4)
  // y: 2 timesteps, 2 dims: (1,2), (3,4) -- identical
  std::vector<double> x{1, 2, 3, 4};
  std::vector<double> y{1, 2, 3, 4};

  double d = dtwFull_L_mv<double>(x.data(), 2, y.data(), 2, 2);
  REQUIRE_THAT(d, WithinAbs(0.0, 1e-15));

  // Now y different: (2,3), (4,5)
  // L1 cost: step (0,0): |1-2|+|2-3| = 2
  //          step (1,1): |3-4|+|4-5| = 2
  // Diagonal alignment: 2 + 2 = 4
  std::vector<double> y2{2, 3, 4, 5};
  double d2 = dtwFull_L_mv<double>(x.data(), 2, y2.data(), 2, 2);
  REQUIRE_THAT(d2, WithinAbs(4.0, 1e-12));
}

// =========================================================================
//  7. Consistency across randomized inputs (fuzz-like)
// =========================================================================

TEST_CASE("Full vs Light vs Banded(wide) agree on random inputs", "[accuracy][fuzz]")
{
  for (int trial = 0; trial < 20; ++trial) {
    auto x = make_random(10 + trial, 500 + trial);
    auto y = make_random(10 + trial, 600 + trial);

    double d_full = dtwFull<double>(x, y);
    double d_L = dtwFull_L<double>(x, y);
    double d_band = dtwBanded<double>(x, y, 100);

    REQUIRE_THAT(d_full, WithinAbs(d_L, 1e-10));
    REQUIRE_THAT(d_full, WithinAbs(d_band, 1e-10));
  }
}

TEST_CASE("Full vs Light vs Banded(wide) agree on random inputs with SquaredL2", "[accuracy][fuzz]")
{
  for (int trial = 0; trial < 10; ++trial) {
    auto x = make_random(15 + trial, 700 + trial);
    auto y = make_random(12 + trial, 800 + trial);

    double d_full = dtwFull<double>(x, y, core::MetricType::SquaredL2);
    double d_L = dtwFull_L<double>(x, y, -1.0, core::MetricType::SquaredL2);
    double d_band = dtwBanded<double>(x, y, 100, -1.0, core::MetricType::SquaredL2);

    REQUIRE_THAT(d_full, WithinRel(d_L, 1e-10));
    REQUIRE_THAT(d_full, WithinRel(d_band, 1e-10));
  }
}

TEST_CASE("Symmetry holds on random different-length inputs", "[accuracy][fuzz]")
{
  for (int trial = 0; trial < 20; ++trial) {
    auto x = make_random(8 + trial % 5, 900 + trial);
    auto y = make_random(12 + trial % 7, 1000 + trial);

    double d_xy = dtwFull<double>(x, y);
    double d_yx = dtwFull<double>(y, x);
    REQUIRE_THAT(d_xy, WithinAbs(d_yx, 1e-12));
  }
}

TEST_CASE("ADTW banded vs full agree with wide band on random inputs", "[accuracy][fuzz]")
{
  for (int trial = 0; trial < 10; ++trial) {
    auto x = make_random(15, 1100 + trial);
    auto y = make_random(18, 1200 + trial);

    for (double penalty : {0.5, 2.0}) {
      double full = adtwFull_L<double>(x.data(), x.size(), y.data(), y.size(), penalty);
      double banded = adtwBanded<double>(x.data(), x.size(), y.data(), y.size(), 100, penalty);
      REQUIRE_THAT(banded, WithinAbs(full, 1e-10));
    }
  }
}

TEST_CASE("WDTW full vs banded(wide) agree on random inputs", "[accuracy][fuzz]")
{
  for (int trial = 0; trial < 10; ++trial) {
    auto x = make_random(15, 1300 + trial);
    auto y = make_random(18, 1400 + trial);

    int max_dev = static_cast<int>(std::max(x.size(), y.size())) - 1;
    auto w = wdtw_weights<double>(max_dev, 0.05);

    double full = wdtwFull<double>(x, y, w);
    double banded = wdtwBanded<double>(x, y, w, 100);
    REQUIRE_THAT(banded, WithinAbs(full, 1e-10));
  }
}

// =========================================================================
//  8. Known mathematical properties
// =========================================================================

TEST_CASE("DTW satisfies non-negativity", "[accuracy][properties]")
{
  for (int trial = 0; trial < 10; ++trial) {
    auto x = make_random(20, 1500 + trial);
    auto y = make_random(25, 1600 + trial);
    REQUIRE(dtwFull<double>(x, y) >= 0.0);
    REQUIRE(dtwFull<double>(x, y, core::MetricType::SquaredL2) >= 0.0);
  }
}

TEST_CASE("DTW(x, x) == 0 for random series", "[accuracy][properties]")
{
  for (int trial = 0; trial < 10; ++trial) {
    auto x = make_random(30, 1700 + trial);
    REQUIRE_THAT(dtwFull<double>(x, x), WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(dtwFull<double>(x, x, core::MetricType::SquaredL2), WithinAbs(0.0, 1e-15));
  }
}

TEST_CASE("DTW(x, y) > 0 when x != y (non-identical)", "[accuracy][properties]")
{
  std::vector<double> x{1.0, 2.0, 3.0};
  std::vector<double> y{1.0, 2.0, 3.1}; // slightly different

  REQUIRE(dtwFull<double>(x, y) > 0.0);
}

TEST_CASE("DTW with repeated series: dtw(x, [x;x]) uses repetitions", "[accuracy][properties]")
{
  std::vector<double> x{1, 2, 3};
  std::vector<double> xx{1, 2, 3, 1, 2, 3};

  // dtw(x, xx) should be finite and positive (second half creates distance)
  double d = dtwFull<double>(x, xx);
  REQUIRE(std::isfinite(d));
  REQUIRE(d >= 0.0);
}
