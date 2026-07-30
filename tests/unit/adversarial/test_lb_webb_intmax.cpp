/**
 * @file test_lb_webb_intmax.cpp
 * @brief F57 regression gate for saturated CPU lower-bound window arithmetic.
 *
 * The acceptance band and exact marker were registered in
 * .claude/baselines/2026-07-30-d3-lb-enhanced-webb.md before this test was
 * executed.  For a two-sample series, radii n-1, n, and INT_MAX all describe
 * the same global window.  Keeping all three calls independent catches signed
 * radius arithmetic before a coincidentally correct shared result can hide it.
 */

#include <core/lower_bound_impl.hpp>

#include <catch2/catch_test_macros.hpp>

#include <climits>
#include <iostream>
#include <vector>

namespace {

using Series = std::vector<double>;

struct BoundValues
{
  double webb_l1;
  double webb_squared;
  double enhanced_l1;
  double enhanced_squared;
};

BoundValues evaluate_bounds(const Series &a, const Series &b, int radius)
{
  const auto envelope_a = dtwc::core::compute_webb_envelope(a, radius);
  const auto envelope_b = dtwc::core::compute_webb_envelope(b, radius);

  return {
    dtwc::core::lb_webb<dtwc::core::L1Metric>(
      a, envelope_a, b, envelope_b, radius),
    dtwc::core::lb_webb<dtwc::core::SquaredL2Metric>(
      a, envelope_a, b, envelope_b, radius),
    dtwc::core::lb_enhanced<double, dtwc::core::L1Metric>(
      a.data(), b.data(), a.size(), envelope_b.upper.data(), envelope_b.lower.data(), radius),
    dtwc::core::lb_enhanced<double, dtwc::core::SquaredL2Metric>(
      a.data(), b.data(), a.size(), envelope_b.upper.data(), envelope_b.lower.data(), radius)
  };
}

} // namespace

TEST_CASE("F57 LB_Webb saturates an INT_MAX radius",
          "[adversarial][lb_webb][intmax][F57]")
{
  const Series a{ -2.0, -2.0 };
  const Series b{ 0.0, 0.0 };
  constexpr double exact_global_l1 = 4.0;
  constexpr double exact_global_squared = 8.0;
  const int radius_n_minus_one = static_cast<int>(a.size() - 1);
  const int radius_n = static_cast<int>(a.size());

  const auto at_n_minus_one = evaluate_bounds(a, b, radius_n_minus_one);
  const auto at_n = evaluate_bounds(a, b, radius_n);
  const auto at_intmax = evaluate_bounds(a, b, INT_MAX);

  REQUIRE(at_n_minus_one.webb_l1 == exact_global_l1);
  REQUIRE(at_n.webb_l1 == exact_global_l1);
  REQUIRE(at_intmax.webb_l1 == exact_global_l1);
  REQUIRE(at_n_minus_one.webb_squared == exact_global_squared);
  REQUIRE(at_n.webb_squared == exact_global_squared);
  REQUIRE(at_intmax.webb_squared == exact_global_squared);

  REQUIRE(at_n.webb_l1 == at_n_minus_one.webb_l1);
  REQUIRE(at_intmax.webb_l1 == at_n_minus_one.webb_l1);
  REQUIRE(at_n.webb_squared == at_n_minus_one.webb_squared);
  REQUIRE(at_intmax.webb_squared == at_n_minus_one.webb_squared);

  REQUIRE(at_n_minus_one.enhanced_l1 == exact_global_l1);
  REQUIRE(at_n.enhanced_l1 == at_n_minus_one.enhanced_l1);
  REQUIRE(at_intmax.enhanced_l1 == at_n_minus_one.enhanced_l1);
  REQUIRE(at_n_minus_one.enhanced_squared == exact_global_squared);
  REQUIRE(at_n.enhanced_squared == at_n_minus_one.enhanced_squared);
  REQUIRE(at_intmax.enhanced_squared == at_n_minus_one.enhanced_squared);

  REQUIRE(at_intmax.webb_l1 <= exact_global_l1);
  REQUIRE(at_intmax.webb_squared <= exact_global_squared);

  std::cout
    << "F57_LB_WEBB_INTMAX l1=4/4 squared=8/8 global_parity=2/2 "
       "admissible=2/2 skips=0 verdict=PASS\n";
}
