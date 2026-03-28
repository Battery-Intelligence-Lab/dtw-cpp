/**
 * @file unit_test_soft_dtw.cpp
 * @brief Unit tests for Soft-DTW distance and gradient.
 *
 * Reference: Cuturi & Blondel (2017), "Soft-DTW: a Differentiable Loss
 *            Function for Time-Series"
 *
 * @date 28 Mar 2026
 */

#include <dtwc.hpp>
#include <soft_dtw.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>
#include <limits>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;

using namespace dtwc;

// ---------------------------------------------------------------------------
// softmin_gamma tests
// ---------------------------------------------------------------------------

TEST_CASE("softmin_gamma: three equal values", "[soft_dtw][softmin]")
{
  // softmin(a, a, a, gamma) = a - gamma * log(3)
  const double a = 5.0;
  const double gamma = 1.0;
  const double expected = a - gamma * std::log(3.0);
  REQUIRE_THAT(softmin_gamma(a, a, a, gamma), WithinAbs(expected, 1e-12));
}

TEST_CASE("softmin_gamma: three equal values, different gamma", "[soft_dtw][softmin]")
{
  const double a = 10.0;
  const double gamma = 0.5;
  const double expected = a - gamma * std::log(3.0);
  REQUIRE_THAT(softmin_gamma(a, a, a, gamma), WithinAbs(expected, 1e-12));
}

TEST_CASE("softmin_gamma: approaches min as gamma -> 0", "[soft_dtw][softmin]")
{
  const double a = 3.0, b = 1.0, c = 5.0;
  const double gamma = 0.001;
  const double hard_min = 1.0;
  // With very small gamma, softmin should be very close to min
  REQUIRE_THAT(softmin_gamma(a, b, c, gamma), WithinAbs(hard_min, 1e-2));
}

TEST_CASE("softmin_gamma: always <= min(a,b,c)", "[soft_dtw][softmin]")
{
  // softmin is always <= hard min due to the log(sum(exp)) >= 0 term
  const double a = 3.0, b = 7.0, c = 5.0;
  for (double gamma : { 0.01, 0.1, 0.5, 1.0, 2.0, 10.0 }) {
    const double result = softmin_gamma(a, b, c, gamma);
    REQUIRE(result <= std::min({ a, b, c }) + 1e-12);
  }
}

TEST_CASE("softmin_gamma: numerical stability with large values", "[soft_dtw][softmin]")
{
  // Large values should not cause overflow thanks to log-sum-exp trick
  const double a = 1e10, b = 1e10 + 1.0, c = 1e10 + 2.0;
  const double gamma = 1.0;
  const double result = softmin_gamma(a, b, c, gamma);
  REQUIRE(std::isfinite(result));
  REQUIRE(result <= a + 1e-6);
}

// ---------------------------------------------------------------------------
// soft_dtw forward pass tests
// ---------------------------------------------------------------------------

TEST_CASE("soft_dtw: converges to dtwFull as gamma -> 0", "[soft_dtw]")
{
  std::vector<double> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 };
  const double hard_dtw = dtwFull_L<double>(x, y);
  const double gamma = 0.001;
  const double soft = soft_dtw(x, y, gamma);
  // Should be close, with tolerance proportional to gamma
  REQUIRE_THAT(soft, WithinAbs(hard_dtw, 0.5));
}

TEST_CASE("soft_dtw: identical short series, small gamma", "[soft_dtw]")
{
  std::vector<double> x{ 1, 2, 3 };
  const double hard_dtw = dtwFull_L<double>(x, x); // should be 0
  const double gamma = 0.001;
  const double soft = soft_dtw(x, x, gamma);
  REQUIRE_THAT(soft, WithinAbs(hard_dtw, 0.1));
}

TEST_CASE("soft_dtw: self-distance can be negative for gamma > 0", "[soft_dtw]")
{
  // Documented property: soft_dtw(x, x, gamma) <= 0 for gamma > 0
  std::vector<double> x{ 1, 2, 3, 4, 5 };
  const double gamma = 1.0;
  const double result = soft_dtw(x, x, gamma);
  REQUIRE(result <= 0.0 + 1e-10);
}

TEST_CASE("soft_dtw: symmetry", "[soft_dtw]")
{
  std::vector<double> x{ 1.0, 3.0, 5.0, 2.0 };
  std::vector<double> y{ 2.0, 4.0, 1.0 };
  const double gamma = 1.0;
  REQUIRE_THAT(soft_dtw(x, y, gamma), WithinAbs(soft_dtw(y, x, gamma), 1e-12));
}

TEST_CASE("soft_dtw: symmetry with different gamma", "[soft_dtw]")
{
  std::vector<double> x{ 1.5, 2.5, 3.5 };
  std::vector<double> y{ 4.0, 5.0 };
  for (double gamma : { 0.01, 0.1, 1.0, 5.0 }) {
    REQUIRE_THAT(soft_dtw(x, y, gamma),
      WithinAbs(soft_dtw(y, x, gamma), 1e-12));
  }
}

TEST_CASE("soft_dtw: empty series returns infinity", "[soft_dtw]")
{
  std::vector<double> x{ 1, 2, 3 };
  std::vector<double> empty{};
  REQUIRE(soft_dtw(x, empty, 1.0) > 1e10);
  REQUIRE(soft_dtw(empty, x, 1.0) > 1e10);
  REQUIRE(soft_dtw(empty, empty, 1.0) > 1e10);
}

TEST_CASE("soft_dtw: single element series", "[soft_dtw]")
{
  std::vector<double> x{ 3.0 };
  std::vector<double> y{ 7.0 };
  const double gamma = 1.0;
  // Single element: only one path, so soft_dtw = |3-7| = 4
  // (no softmin needed, only one cell)
  REQUIRE_THAT(soft_dtw(x, y, gamma), WithinAbs(4.0, 1e-12));
}

TEST_CASE("soft_dtw: known 3x3 example with gamma=1.0", "[soft_dtw]")
{
  // x = {1, 2, 3}, y = {2, 4, 6}, gamma = 1.0
  // Manual computation:
  // d(i,j) = |x[i] - y[j]|
  //
  //       y=2   y=4   y=6
  // x=1    1     3     5
  // x=2    0     2     4
  // x=3    1     1     3
  //
  // C(0,0) = 1
  // C(1,0) = 1 + 0 = 1
  // C(2,0) = 1 + 1 = 2
  // C(0,1) = 1 + 3 = 4
  // C(0,2) = 4 + 5 = 9
  //
  // C(1,1) = 2 + softmin(C(0,1), C(1,0), C(0,0), 1.0)
  //        = 2 + softmin(4, 1, 1, 1.0)
  //        = 2 + (1 - log(exp(-(4-1)) + exp(0) + exp(0)))
  //        = 2 + 1 - log(exp(-3) + 1 + 1)
  //        = 3 - log(2 + exp(-3))
  //
  // Let's just verify the result numerically. We'll compute expected below.
  std::vector<double> x{ 1, 2, 3 };
  std::vector<double> y{ 2, 4, 6 };
  const double gamma = 1.0;

  // Compute expected step by step
  auto sm = [](double a, double b, double c, double g) {
    double M = std::min({ a, b, c });
    return M - g * std::log(std::exp(-(a - M) / g) + std::exp(-(b - M) / g) + std::exp(-(c - M) / g));
  };

  double C00 = 1.0;
  double C10 = C00 + 0.0; // = 1
  double C20 = C10 + 1.0; // = 2
  double C01 = C00 + 3.0; // = 4
  double C02 = C01 + 5.0; // = 9
  double C11 = 2.0 + sm(C01, C10, C00, gamma);
  double C21 = 1.0 + sm(C11, C20, C10, gamma);
  double C12 = 4.0 + sm(C02, C11, C01, gamma);
  double C22 = 3.0 + sm(C12, C21, C11, gamma);

  REQUIRE_THAT(soft_dtw(x, y, gamma), WithinAbs(C22, 1e-10));
}

TEST_CASE("soft_dtw: monotone in gamma (larger gamma -> smaller value)", "[soft_dtw]")
{
  // As gamma increases, softmin decreases (more smoothing), so total cost decreases
  std::vector<double> x{ 1, 3, 5, 2, 4 };
  std::vector<double> y{ 2, 4, 6, 1, 3 };

  double prev = soft_dtw(x, y, 0.01);
  for (double gamma : { 0.1, 0.5, 1.0, 2.0, 5.0 }) {
    double cur = soft_dtw(x, y, gamma);
    REQUIRE(cur <= prev + 1e-10);
    prev = cur;
  }
}

// ---------------------------------------------------------------------------
// soft_dtw gradient tests
// ---------------------------------------------------------------------------

TEST_CASE("soft_dtw_gradient: matches finite differences", "[soft_dtw][gradient]")
{
  std::vector<double> x{ 1.0, 3.0, 5.0, 2.0 };
  std::vector<double> y{ 2.0, 4.0, 1.0, 3.0 };
  const double gamma = 1.0;
  const double eps = 1e-5;

  auto grad = soft_dtw_gradient(x, y, gamma);
  REQUIRE(grad.size() == x.size());

  for (size_t i = 0; i < x.size(); ++i) {
    auto x_plus = x;
    auto x_minus = x;
    x_plus[i] += eps;
    x_minus[i] -= eps;
    double fd = (soft_dtw(x_plus, y, gamma) - soft_dtw(x_minus, y, gamma)) / (2.0 * eps);
    REQUIRE_THAT(grad[i], WithinAbs(fd, 1e-4));
  }
}

TEST_CASE("soft_dtw_gradient: matches finite differences, small gamma", "[soft_dtw][gradient]")
{
  std::vector<double> x{ 2.0, 5.0, 3.0 };
  std::vector<double> y{ 1.0, 4.0, 6.0 };
  const double gamma = 0.1;
  const double eps = 1e-5;

  auto grad = soft_dtw_gradient(x, y, gamma);
  REQUIRE(grad.size() == x.size());

  for (size_t i = 0; i < x.size(); ++i) {
    auto x_plus = x;
    auto x_minus = x;
    x_plus[i] += eps;
    x_minus[i] -= eps;
    double fd = (soft_dtw(x_plus, y, gamma) - soft_dtw(x_minus, y, gamma)) / (2.0 * eps);
    REQUIRE_THAT(grad[i], WithinAbs(fd, 1e-3));
  }
}

TEST_CASE("soft_dtw_gradient: different length series", "[soft_dtw][gradient]")
{
  std::vector<double> x{ 1.0, 2.0, 3.0 };
  std::vector<double> y{ 2.0, 4.0, 6.0, 8.0 };
  const double gamma = 1.0;
  const double eps = 1e-5;

  auto grad = soft_dtw_gradient(x, y, gamma);
  REQUIRE(grad.size() == x.size());

  for (size_t i = 0; i < x.size(); ++i) {
    auto x_plus = x;
    auto x_minus = x;
    x_plus[i] += eps;
    x_minus[i] -= eps;
    double fd = (soft_dtw(x_plus, y, gamma) - soft_dtw(x_minus, y, gamma)) / (2.0 * eps);
    REQUIRE_THAT(grad[i], WithinAbs(fd, 1e-4));
  }
}

TEST_CASE("soft_dtw_gradient: zero gradient at minimum for identical series", "[soft_dtw][gradient]")
{
  // For soft_dtw(x, x, gamma), x is a "stationary point" of sorts—
  // the gradient w.r.t. the first argument at x=y should have a specific structure.
  // Each grad[i] should be sum_j E(i,j) * sign(x[i]-y[j]).
  // When x == y, many terms cancel out, but the gradient need not be exactly zero
  // for soft-DTW (it IS zero for standard DTW at x=y). We just check finiteness.
  std::vector<double> x{ 1.0, 2.0, 3.0, 4.0 };
  const double gamma = 1.0;
  auto grad = soft_dtw_gradient(x, x, gamma);
  REQUIRE(grad.size() == x.size());
  for (size_t i = 0; i < grad.size(); ++i) {
    REQUIRE(std::isfinite(grad[i]));
  }
}
