/**
 * @file unit_test_distance_semantics.cpp
 * @brief Phase 8 M36: metric-token and variant/missing semantic guards.
 *
 * Registered before production edits. Invalid cross-products must fail with
 * one typed diagnostic before a dispatcher/kernel is selected. Accepted
 * Standard/missing and non-Standard/Error routes remain exact.
 */

#include <Problem.hpp>
#include <core/dtw.hpp>
#include <core/dtw_dispatch.hpp>
#include <distance.hpp>
#include <error.hpp>
#include <warping_adtw.hpp>
#include <warping_missing.hpp>
#include <warping_missing_arow.hpp>

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

constexpr const char *cross_product_error =
  "Non-Standard DTW variants require MissingStrategy::Error.";

template <typename Fn>
void require_semantic_error(Fn &&fn)
{
  bool caught = false;
  try {
    std::forward<Fn>(fn)();
  } catch (const dtwc::InvalidInput &error) {
    caught = true;
    REQUIRE(std::string(error.what()) == cross_product_error);
  } catch (const std::exception &error) {
    FAIL("wrong exception type: " << error.what());
  }
  REQUIRE(caught);
}

} // namespace

TEST_CASE("M36 free facade rejects every variant/missing cross-product",
          "[m36][distance-semantics][free-function]")
{
  const std::span<const double> empty{};
  const std::vector<dtwc::core::DTWVariant> variants{
    dtwc::core::DTWVariant::DDTW,
    dtwc::core::DTWVariant::WDTW,
    dtwc::core::DTWVariant::ADTW,
    dtwc::core::DTWVariant::SoftDTW,
    dtwc::core::DTWVariant::MSM,
    dtwc::core::DTWVariant::TWE,
  };
  const std::vector<dtwc::core::MissingStrategy> strategies{
    dtwc::core::MissingStrategy::ZeroCost,
    dtwc::core::MissingStrategy::AROW,
    dtwc::core::MissingStrategy::Interpolate,
  };

  for (const auto variant : variants) {
    for (const auto strategy : strategies) {
      dtwc::core::DTWVariantParams params;
      params.variant = variant;
      require_semantic_error([&] {
        (void)dtwc::distance::dtw<double>(
          empty, empty, params, -1, dtwc::core::MetricType::L1, strategy);
      });
    }
  }
}

TEST_CASE("M36 runtime and Problem resolver reject before selecting a kernel",
          "[m36][distance-semantics][dispatch]")
{
  const std::vector<double> x{0.0};
  const std::vector<double> y{0.0, 0.0};

  dtwc::core::DTWOptions options;
  options.variant_params.variant = dtwc::core::DTWVariant::ADTW;
  options.variant_params.adtw_penalty = 2.0;
  options.missing_strategy = dtwc::core::MissingStrategy::ZeroCost;
  require_semantic_error([&] {
    (void)dtwc::core::dtw_runtime(
      x.data(), x.size(), y.data(), y.size(), options);
  });

  dtwc::Problem problem("m36");
  problem.variant_params.variant = dtwc::core::DTWVariant::ADTW;
  problem.missing_strategy = dtwc::core::MissingStrategy::ZeroCost;
  require_semantic_error([&] {
    (void)dtwc::core::resolve_dtw_fn<double>(problem);
  });
  require_semantic_error([&] {
    (void)dtwc::core::resolve_dtw_fn<float>(problem);
  });
}

TEST_CASE("M36 accepted semantic fingerprints remain exact",
          "[m36][distance-semantics][fingerprint]")
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> x{0.0, nan, 2.0};
  const std::vector<double> y{0.0, 1.0, 2.0};
  dtwc::core::DTWVariantParams standard;

  const double zero_cost = dtwc::distance::dtw<double>(
    x, y, standard, -1, dtwc::core::MetricType::L1,
    dtwc::core::MissingStrategy::ZeroCost);
  REQUIRE_THAT(zero_cost,
               WithinAbs(dtwc::dtwMissing_banded<double>(x, y, -1), 0.0));

  const double arow = dtwc::distance::dtw<double>(
    x, y, standard, -1, dtwc::core::MetricType::L1,
    dtwc::core::MissingStrategy::AROW);
  REQUIRE_THAT(arow, WithinAbs(dtwc::dtwAROW_banded<double>(x, y, -1), 0.0));

  // The runtime API must execute—not merely accept—the same Standard/missing
  // semantics. Before M36 it ignored this option and entered ordinary DTW.
  for (const auto strategy : {
         dtwc::core::MissingStrategy::ZeroCost,
         dtwc::core::MissingStrategy::AROW,
         dtwc::core::MissingStrategy::Interpolate}) {
    dtwc::core::DTWOptions options;
    options.missing_strategy = strategy;
    const double expected = dtwc::distance::dtw<double>(
      x, y, standard, -1, dtwc::core::MetricType::L1, strategy);
    REQUIRE_THAT(
      dtwc::core::dtw_runtime(x.data(), x.size(), y.data(), y.size(), options),
      WithinAbs(expected, 0.0));
  }

  dtwc::core::DTWVariantParams adtw;
  adtw.variant = dtwc::core::DTWVariant::ADTW;
  adtw.adtw_penalty = 0.75;
  const std::vector<double> finite_x{0.0};
  const std::vector<double> finite_y{0.0, 0.0};
  REQUIRE_THAT(
    dtwc::distance::dtw<double>(
      finite_x, finite_y, adtw, -1, dtwc::core::MetricType::L1,
      dtwc::core::MissingStrategy::Error),
    WithinAbs(dtwc::adtwFull_L<double>(finite_x, finite_y, 0.75), 0.0));
}

TEST_CASE("CPU float32 DTW normalizes its finite no-path sentinel",
          "[F13][distance-semantics][float32][sentinel]")
{
  dtwc::Problem problem("f13_f32_public_distance");
  problem.band = 0;

  const auto f32_distance = dtwc::core::resolve_dtw_fn<float>(problem);
  const auto f64_distance = dtwc::core::resolve_dtw_fn<double>(problem);
  const std::vector<float> short_f32{0.0f};
  const std::vector<float> long_f32{0.0f, 0.0f, 0.0f};
  const std::vector<double> short_f64{0.0};
  const std::vector<double> long_f64{0.0, 0.0, 0.0};

  REQUIRE(f32_distance(short_f32, long_f32)
          == std::numeric_limits<double>::max());
  REQUIRE(f64_distance(short_f64, long_f64)
          == std::numeric_limits<double>::max());

  const float adjacent =
    std::nextafter(std::numeric_limits<float>::max(), 0.0f);
  REQUIRE(f32_distance(std::vector<float>{0.0f},
                       std::vector<float>{adjacent})
          == static_cast<double>(adjacent));
  REQUIRE(std::isinf(f32_distance(
    std::vector<float>{0.0f},
    std::vector<float>{std::numeric_limits<float>::infinity()})));
  REQUIRE(std::isnan(f32_distance(
    std::vector<float>{0.0f},
    std::vector<float>{std::numeric_limits<float>::quiet_NaN()})));
}
