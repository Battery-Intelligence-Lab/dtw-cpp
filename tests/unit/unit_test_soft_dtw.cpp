/**
 * @file unit_test_soft_dtw.cpp
 * @brief Unit tests for Soft-DTW distance and gradient.
 *
 * Reference: Cuturi & Blondel (2017), "Soft-DTW: a Differentiable Loss
 *            Function for Time-Series"
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include <dtwc.hpp>
#include <soft_dtw.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

TEST_CASE("minimum-positive gamma remains numerically defined across Soft-DTW",
          "[soft_dtw][softmin][gradient][denorm]")
{
  const auto check_precision = []<typename T> {
    const T gamma = std::numeric_limits<T>::denorm_min();
    REQUIRE(gamma > T(0));

    // Every cost is 0 and the first row and column keep their one predecessor,
    // so the value is the softmin of three zeros: -gamma ln 3.
    const std::vector<T> x{T(0), T(0)};
    const std::vector<T> y{T(0), T(0)};
    const T value = soft_dtw<T>(x, y, gamma);
    CHECK(std::isfinite(value));
    CHECK(value == -gamma * std::log(T(3)));

    const auto gradient = soft_dtw_gradient<T>(x, y, gamma);
    REQUIRE(gradient.size() == x.size());
    for (const T component : gradient) {
      CHECK(std::isfinite(component));
      CHECK(component == T(0));
    }
  };

  SECTION("float64") { check_precision.template operator()<double>(); }
  SECTION("float32") { check_precision.template operator()<float>(); }
}

TEST_CASE("soft_dtw: converges to dtwFull as gamma -> 0", "[soft_dtw]")
{
  std::vector<double> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 };
  const double hard_dtw = dtwFull_L<double>(x, y);
  REQUIRE_THAT(soft_dtw(x, y, 0.001), WithinAbs(hard_dtw, 0.5));
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
  std::vector<double> x{ 1, 2, 3 };
  std::vector<double> y{ 2, 4, 6 };
  const double gamma = 1.0;

  auto sm = [](double a, double b, double c, double g) {
    double M = std::min({ a, b, c });
    return M - g * std::log(std::exp(-(a - M) / g) + std::exp(-(b - M) / g) + std::exp(-(c - M) / g));
  };

  double C00 = 1.0;
  double C10 = C00 + 0.0;
  double C20 = C10 + 1.0;
  double C01 = C00 + 3.0;
  double C02 = C01 + 5.0;
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

TEST_CASE("soft_dtw_gradient: matches finite differences", "[soft_dtw][gradient]")
{
  struct Case
  {
    std::vector<double> x, y;
    double gamma, tol;
  };
  const Case cases[] = {
    { { 1.0, 3.0, 5.0, 2.0 }, { 2.0, 4.0, 1.0, 3.0 }, 1.0, 1e-4 },
    { { 2.0, 5.0, 3.0 }, { 1.0, 4.0, 6.0 }, 0.1, 1e-3 },
    { { 1.0, 2.0, 3.0 }, { 2.0, 4.0, 6.0, 8.0 }, 1.0, 1e-4 },
  };
  constexpr double eps = 1e-5;
  for (const auto &c : cases) {
    const auto grad = soft_dtw_gradient(c.x, c.y, c.gamma);
    REQUIRE(grad.size() == c.x.size());
    for (std::size_t i = 0; i < c.x.size(); ++i) {
      auto x_plus = c.x, x_minus = c.x;
      x_plus[i] += eps;
      x_minus[i] -= eps;
      const double fd = (soft_dtw(x_plus, c.y, c.gamma) - soft_dtw(x_minus, c.y, c.gamma)) / (2.0 * eps);
      REQUIRE_THAT(grad[i], WithinAbs(fd, c.tol));
    }
  }
}

TEST_CASE("soft_dtw_gradient: a sample far above the other series leaves the rest exact", "[soft_dtw][gradient]")
{
  // x's last sample lies 1e13 above y: every path pays that cost once, at the last cell, whose neighbour in that row
  // weighs exp(-1e13 / gamma) = 0. The gradient for the other samples is then that of the soft-min of the two cells
  // entering the last one from the row before, whose finite difference needs x without that sample (the full value's
  // 1e13 would drown it). Rebuilding a soft-min as a cumulative cost minus the 1e13 local cost loses it.
  const double gamma = 0.1;
  std::vector<double> x(51), y(51);
  for (std::size_t i = 0; i < x.size(); ++i) {
    x[i] = 0.1 * std::sin(1.7 * double(i));
    y[i] = 0.1 * std::cos(1.3 * double(i));
  }
  x.back() = 1e13;
  const auto grad = soft_dtw_gradient(x, y, gamma);
  const std::vector<double> head(x.begin(), x.end() - 1), y_head(y.begin(), y.end() - 1);
  const auto entering = [&](const std::vector<double> &h) {
    const double a = soft_dtw(h, y, gamma), b = soft_dtw(h, y_head, gamma), m = std::min(a, b);
    return m - gamma * std::log(std::exp((m - a) / gamma) + std::exp((m - b) / gamma));
  };
  const double eps = 1e-6;
  for (std::size_t i = 0; i < head.size(); ++i) {
    auto plus = head, minus = head;
    plus[i] += eps;
    minus[i] -= eps;
    REQUIRE_THAT(grad[i], WithinAbs((entering(plus) - entering(minus)) / (2.0 * eps), 1e-6));
  }
  REQUIRE(grad.back() == 1.0); // sign(x - y) at the last cell; no other cell of that row carries weight
}
