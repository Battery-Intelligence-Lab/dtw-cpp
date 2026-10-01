/**
 * @file unit_test_distance_semantics.cpp
 * @brief Phase 8 M36: metric-token and variant/missing semantic guards;
 *        FX-19: variant/metric guards of the distance::dtw dispatcher.
 *
 * Registered before production edits. Invalid cross-products must fail with
 * one typed diagnostic before a dispatcher/kernel is selected. Accepted
 * Standard/missing and non-Standard/Error routes remain exact.
 */

#include <Problem.hpp>
#include <core/dtw_dispatch.hpp>
#include <distance.hpp>
#include <base/error.hpp>
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
void require_semantic_error(Fn &&fn,
                            const std::string &expected = cross_product_error)
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

/// FX-19: the refusal, by `where`, of a metric the variant's kernel lacks.
std::string metric_refusal(const char *where, const char *metric,
                           const char *variant)
{
  return std::string(where) + ": metric " + metric
       + " is implemented for Standard DTW and DDTW only, but variant = "
       + variant + " was requested. Use metric L1 for this configuration.";
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

TEST_CASE("M36 Problem resolver rejects before selecting a kernel",
          "[m36][distance-semantics][dispatch]")
{
  // A Problem refuses the combination at the setter, so no kernel is ever
  // resolved for it.
  dtwc::Problem problem("m36");
  problem.set_variant(dtwc::core::DTWVariant::ADTW);
  require_semantic_error([&] {
    problem.set_missing_strategy(dtwc::core::MissingStrategy::ZeroCost);
  });
  REQUIRE(problem.missing_strategy() == dtwc::core::MissingStrategy::Error);
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

TEST_CASE("FX-19 dispatcher refuses a metric the variant's kernel does not take",
          "[FX-19][distance-semantics][metric]")
{
  using dtwc::core::DTWVariant;
  using dtwc::core::MetricType;
  const std::vector<double> x{0.0, 3.0};
  const std::vector<double> y{0.0, 1.0};
  const std::vector<float> xf{0.0f, 3.0f};
  const std::vector<float> yf{0.0f, 1.0f};
  const std::vector<double> nan_x{0.0, std::numeric_limits<double>::quiet_NaN()};
  const dtwc::core::DTWVariantParams defaults;

  // WDTW, ADTW, Soft-DTW, MSM and TWE compute an L1 cost; `l1` is the
  // variant's own function, which the dispatcher must still return for L1.
  const struct {
    DTWVariant variant;
    const char *name;
    double l1;
  } cases[]{
    {DTWVariant::WDTW, "WDTW", dtwc::distance::wdtw<double>(x, y, -1, defaults.wdtw_g)},
    {DTWVariant::ADTW, "ADTW", dtwc::distance::adtw<double>(x, y, -1, defaults.adtw_penalty)},
    {DTWVariant::SoftDTW, "SoftDTW", dtwc::distance::soft_dtw<double>(x, y, defaults.sdtw_gamma)},
    {DTWVariant::MSM, "MSM", dtwc::distance::msm<double>(x, y, defaults.msm_c)},
    {DTWVariant::TWE, "TWE",
     dtwc::distance::twe<double>(x, y, defaults.twe_nu, defaults.twe_lambda)},
  };
  const std::pair<MetricType, const char *> metrics[]{
    {MetricType::SquaredL2, "SquaredL2"}, {MetricType::L2, "L2"}};

  for (const auto &c : cases) {
    CAPTURE(c.name);
    dtwc::core::DTWVariantParams params;
    params.variant = c.variant;
    for (const auto &[metric, metric_name] : metrics) {
      CAPTURE(metric_name);
      const std::string expected = metric_refusal("distance::dtw", metric_name, c.name);
      require_semantic_error([&] {
        (void)dtwc::distance::dtw<double>(x, y, params, -1, metric);
      }, expected);
      require_semantic_error([&] {
        (void)dtwc::distance::dtw<double>(
          std::span<const double>{x}, std::span<const double>{y}, params, 1, metric);
      }, expected);
      require_semantic_error([&] {
        (void)dtwc::distance::dtw<float>(xf, yf, params, -1, metric);
      }, expected);
      // A configuration error is reported before the input is scanned.
      require_semantic_error([&] {
        (void)dtwc::distance::dtw<double>(nan_x, y, params, -1, metric);
      }, expected);
    }
    REQUIRE(dtwc::distance::dtw<double>(x, y, params, -1, MetricType::L1) == c.l1);
  }
}

TEST_CASE("FX-19 dispatcher keeps the metric where the kernel takes one",
          "[FX-19][distance-semantics][metric]")
{
  using dtwc::core::DTWVariant;
  using dtwc::core::MetricType;
  using dtwc::core::MissingStrategy;
  // Hand oracle on x = {0, 3}, y = {0, 1}, where the diagonal path is optimal.
  // Standard: |0-0| + |3-1| = 2; squared 0 + 4 = 4. DDTW differentiates each
  // two-point series to a constant, (3, 3) and (1, 1): 2 + 2 = 4; squared 8.
  const std::vector<double> x{0.0, 3.0};
  const std::vector<double> y{0.0, 1.0};
  dtwc::core::DTWVariantParams standard;
  dtwc::core::DTWVariantParams ddtw;
  ddtw.variant = DTWVariant::DDTW;

  REQUIRE(dtwc::distance::dtw<double>(x, y, standard, -1, MetricType::L1) == 2.0);
  REQUIRE(dtwc::distance::dtw<double>(x, y, standard, -1, MetricType::SquaredL2) == 4.0);
  REQUIRE(dtwc::distance::dtw<double>(x, y, standard, 1, MetricType::SquaredL2) == 4.0);
  REQUIRE(dtwc::distance::dtw<double>(x, y, ddtw, -1, MetricType::L1) == 4.0);
  REQUIRE(dtwc::distance::dtw<double>(x, y, ddtw, -1, MetricType::SquaredL2) == 8.0);
  REQUIRE(dtwc::distance::dtw<float>(std::vector<float>{0.0f, 3.0f},
                                     std::vector<float>{0.0f, 1.0f}, ddtw, -1,
                                     MetricType::SquaredL2) == 8.0f);

  // Without a NaN every missing-data strategy is Standard DTW.
  for (const auto strategy : {MissingStrategy::ZeroCost, MissingStrategy::AROW,
                              MissingStrategy::Interpolate}) {
    CAPTURE(static_cast<int>(strategy));
    REQUIRE(dtwc::distance::dtw<double>(
              x, y, standard, -1, MetricType::SquaredL2, strategy) == 4.0);
  }
}

TEST_CASE("CPU float32 DTW normalizes its finite no-path sentinel",
          "[F13][distance-semantics][float32][sentinel]")
{
  dtwc::Problem problem("f13_f32_public_distance");
  problem.band = 0;

  const auto f32_distance = dtwc::core::resolve_dtw_fn<float>(problem.distance(), problem.data());
  const auto f64_distance = dtwc::core::resolve_dtw_fn<double>(problem.distance(), problem.data());
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

TEST_CASE("parse_metric_token maps each accepted spelling to the metric it names",
          "[distance-semantics][metric][token]")
{
  using dtwc::core::MetricType;
  CHECK(dtwc::core::parse_metric_token("l1") == MetricType::L1);
  CHECK(dtwc::core::parse_metric_token("squared_euclidean") == MetricType::SquaredL2);
  CHECK(dtwc::core::parse_metric_token("sqeuclidean") == MetricType::SquaredL2);
}
