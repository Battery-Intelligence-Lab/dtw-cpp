/**
 * @file unit_test_variant_domains.cpp
 * @brief Phase 8 M34: exact runtime domains for DTW-variant parameters.
 *
 * Registered before the production repair.  Every public route must reject an
 * invalid parameter before an empty/self-distance shortcut or distance work,
 * and must use dtwc::InvalidInput with the exact shared diagnostic.  The two
 * mathematically valid zero boundaries (WDTW g and ADTW penalty) and positive
 * values immediately above zero remain executable.
 */

#include <Problem.hpp>
#include <core/dtw.hpp>
#include <core/msm.hpp>
#include <core/twe.hpp>
#include <distance.hpp>
#include <error.hpp>
#include <soft_dtw.hpp>
#include <warping_adtw.hpp>
#include <warping_wdtw.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

using Catch::Matchers::WithinAbs;

namespace {

template <typename Fn>
void require_invalid_input(Fn &&fn, const char *expected)
{
  bool caught = false;
  try {
    std::forward<Fn>(fn)();
  } catch (const dtwc::InvalidInput &error) {
    caught = true;
    REQUIRE(std::string(error.what()) == expected);
  } catch (const std::exception &error) {
    FAIL("wrong exception type: " << error.what());
  }
  REQUIRE(caught);
}

const std::vector<double> non_finite_values{
  std::numeric_limits<double>::quiet_NaN(),
  std::numeric_limits<double>::infinity(),
  -std::numeric_limits<double>::infinity(),
};

} // namespace

TEST_CASE("M34 free-function variant domains are typed and exact",
          "[m34][variant-domain][free-function]")
{
  const std::vector<double> x{0.0};
  const std::vector<double> y{0.0, 0.0};
  const std::span<const double> xs{x};
  const std::span<const double> ys{y};

  SECTION("WDTW g is finite and non-negative") {
    REQUIRE_THAT(dtwc::wdtwFull<double>(xs, ys, 0.0), WithinAbs(0.0, 0.0));
    for (const double bad : non_finite_values) {
      require_invalid_input(
        [&] { (void)dtwc::wdtwFull<double>(xs, ys, bad); },
        "WDTW g must be finite and non-negative.");
    }
    require_invalid_input(
      [&] { (void)dtwc::wdtwBanded<double>(xs, ys, 1, -1.0); },
      "WDTW g must be finite and non-negative.");
  }

  SECTION("ADTW penalty is finite and non-negative") {
    REQUIRE_THAT(dtwc::adtwFull_L<double>(xs, ys, 0.0), WithinAbs(0.0, 0.0));
    for (const double bad : non_finite_values) {
      require_invalid_input(
        [&] { (void)dtwc::adtwFull_L<double>(xs, ys, bad); },
        "ADTW penalty must be finite and non-negative.");
    }
    require_invalid_input(
      [&] { (void)dtwc::adtwBanded<double>(xs, ys, 1, -1.0); },
      "ADTW penalty must be finite and non-negative.");

    // Parameter validation precedes both short-circuit routes.
    const std::span<const double> empty{};
    require_invalid_input(
      [&] { (void)dtwc::adtwFull_L<double>(empty, ys, -1.0); },
      "ADTW penalty must be finite and non-negative.");
    require_invalid_input(
      [&] { (void)dtwc::adtwFull_L<double>(xs, xs, -1.0); },
      "ADTW penalty must be finite and non-negative.");
  }

  SECTION("Soft-DTW gamma is finite and positive") {
    const double near_zero = std::numeric_limits<double>::min();
    REQUIRE(std::isfinite(dtwc::soft_dtw<double>(xs, ys, near_zero)));
    REQUIRE_THAT(dtwc::soft_dtw<double>(xs, ys, near_zero), WithinAbs(0.0, 0.0));

    std::vector<double> bad_values{0.0, -1.0};
    bad_values.insert(bad_values.end(), non_finite_values.begin(), non_finite_values.end());
    for (const double bad : bad_values) {
      require_invalid_input(
        [&] { (void)dtwc::soft_dtw<double>(xs, ys, bad); },
        "Soft-DTW gamma must be finite and positive.");
      require_invalid_input(
        [&] { (void)dtwc::soft_dtw_gradient<double>(xs, ys, bad); },
        "Soft-DTW gamma must be finite and positive.");
    }
  }

  SECTION("MSM c is finite and positive") {
    const double near_zero = std::numeric_limits<double>::min();
    REQUIRE_THAT(dtwc::core::msm_distance<double>(xs, ys, near_zero),
                 WithinAbs(near_zero, 0.0));

    std::vector<double> bad_values{0.0, -1.0};
    bad_values.insert(bad_values.end(), non_finite_values.begin(), non_finite_values.end());
    for (const double bad : bad_values) {
      require_invalid_input(
        [&] { (void)dtwc::core::msm_distance<double>(xs, ys, bad); },
        "MSM c must be finite and positive.");
    }
  }

  SECTION("TWE nu and lambda are finite and positive") {
    const double near_zero = std::numeric_limits<double>::min();
    REQUIRE(std::isfinite(dtwc::core::twe_distance<double>(xs, ys, near_zero, 0.8)));
    REQUIRE(std::isfinite(dtwc::core::twe_distance<double>(xs, ys, 0.1, near_zero)));

    std::vector<double> bad_values{0.0, -1.0};
    bad_values.insert(bad_values.end(), non_finite_values.begin(), non_finite_values.end());
    for (const double bad : bad_values) {
      require_invalid_input(
        [&] { (void)dtwc::core::twe_distance<double>(xs, ys, bad, 0.8); },
        "TWE nu must be finite and positive.");
      require_invalid_input(
        [&] { (void)dtwc::core::twe_distance<double>(xs, ys, 0.1, bad); },
        "TWE lambda must be finite and positive.");
    }
  }
}

TEST_CASE("M34 aggregate dispatch validates every stored variant parameter",
          "[m34][variant-domain][dispatch]")
{
  const std::vector<double> x{0.0};
  const std::vector<double> y{0.0, 0.0};
  dtwc::core::DTWOptions options;
  options.variant_params.variant = dtwc::core::DTWVariant::Standard;
  options.variant_params.adtw_penalty = -1.0; // inactive must not poison stored state

  require_invalid_input(
    [&] { (void)dtwc::core::dtw_runtime(x.data(), x.size(), y.data(), y.size(), options); },
    "ADTW penalty must be finite and non-negative.");
  require_invalid_input(
    [&] { (void)dtwc::distance::dtw<double>(x, y, options.variant_params); },
    "ADTW penalty must be finite and non-negative.");
}

TEST_CASE("M34 Problem rejects invalid variant state transactionally",
          "[m34][variant-domain][problem]")
{
  dtwc::Problem problem("m34");
  const auto original = problem.variant_params;

  auto invalid = original;
  invalid.variant = dtwc::core::DTWVariant::ADTW;
  invalid.adtw_penalty = -1.0;
  require_invalid_input(
    [&] { problem.set_variant(invalid); },
    "ADTW penalty must be finite and non-negative.");

  REQUIRE(problem.variant_params.variant == original.variant);
  REQUIRE(problem.variant_params.adtw_penalty == original.adtw_penalty);
}
