/**
 * @file test_distance_config.cpp
 * @brief One description of a distance (core::DistanceConfig) and one rule for it,
 *        core::validate: the checked dtwc::distance::dtw and a Problem accept the same
 *        configurations, compute the same distances, and refuse the rest in one message.
 *
 * @details Every variant x metric x missing-data strategy on univariate series (the
 *          facade takes no channels), in double and float. Registered before the run:
 *          20 configurations are accepted (Standard with every metric and strategy: 12;
 *          DDTW with every metric: 3; WDTW, ADTW, Soft-DTW, MSM and TWE with L1: 5) and
 *          both entries refuse the other 64. Where accepted, the fill and the facade agree
 *          on every pair within dtw_routes_agree (tests/support/dtw_route_bound.hpp).
 *          The table of refusals has one row per rule of core::validate.
 */

#include <dtwc.hpp>

#include "../../support/deterministic_series.hpp"
#include "../../support/dtw_route_bound.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace ts = dtwc::test_support;
using dtwc::core::DistanceConfig;
using dtwc::core::DTWVariant;
using dtwc::core::DTWVariantParams;
using dtwc::core::MetricType;
using dtwc::core::MissingStrategy;

namespace {

constexpr DTWVariant kVariants[] = { DTWVariant::Standard, DTWVariant::DDTW, DTWVariant::WDTW, DTWVariant::ADTW,
                                     DTWVariant::SoftDTW,  DTWVariant::MSM,  DTWVariant::TWE };
constexpr MetricType kMetrics[] = { MetricType::L1, MetricType::L2, MetricType::SquaredL2 };
constexpr MissingStrategy kStrategies[] = { MissingStrategy::Error, MissingStrategy::ZeroCost, MissingStrategy::AROW,
                                            MissingStrategy::Interpolate };

/// Parameters away from their defaults, so a route that dropped one would show.
DTWVariantParams params_for(DTWVariant variant)
{
  return { .variant = variant, .wdtw_g = 0.3, .adtw_penalty = 0.5, .sdtw_gamma = 0.7, .msm_c = 0.8,
           .twe_nu = 0.05, .twe_lambda = 0.6 };
}

/// Five series of 7 to 11 steps. Under a missing-data strategy series 1 and 3 hold NaN
/// (inside, first and last); series 0 and 2 never do.
template <typename T>
std::vector<std::vector<T>> five_series(bool with_nan)
{
  std::vector<std::vector<T>> out;
  for (unsigned i = 0; i < 5; ++i) {
    std::vector<T> s;
    for (const double v : ts::benchmark_series(7 + i, 700 + i)) s.push_back(static_cast<T>(v));
    out.push_back(std::move(s));
  }
  if (with_nan) {
    const T nan = std::numeric_limits<T>::quiet_NaN();
    out[1][3] = nan;
    out[3].front() = nan;
    out[3].back() = nan;
  }
  return out;
}

/// Whether `call` throws dtwc::InvalidInput (any other exception fails the test).
template <typename Call>
bool refuses(Call &&call)
{
  try {
    call();
  } catch (const dtwc::InvalidInput &) {
    return true;
  }
  return false;
}

} // namespace

TEMPLATE_TEST_CASE("the facade and a Problem's fill accept the same configurations and agree",
                   "[distance_config][equivalence]", double, float)
{
  using T = TestType;
  std::size_t accepted = 0, pairs = 0;
  for (const auto variant : kVariants)
    for (const auto metric : kMetrics)
      for (const auto missing : kStrategies) {
        const auto params = params_for(variant);
        const auto series = five_series<T>(missing != MissingStrategy::Error);
        const std::span<const T> clean_x{ series[0] }, clean_y{ series[2] };
        INFO("variant " << static_cast<int>(variant) << ", metric " << static_cast<int>(metric) << ", missing "
                        << static_cast<int>(missing));

        dtwc::Problem problem("distance_config");
        problem.set_data(dtwc::Data{ std::vector<std::vector<T>>(series), { "s0", "s1", "s2", "s3", "s4" } });
        const bool problem_refuses =
          refuses([&] { problem.set_distance(DistanceConfig{ params, metric, missing, -1 }); });
        const bool facade_refuses =
          refuses([&] { (void)dtwc::distance::dtw<T>(clean_x, clean_y, params, -1, metric, missing); });
        CHECK(problem_refuses == facade_refuses);
        if (problem_refuses || facade_refuses) continue;
        ++accepted;

        for (const int band : { -1, 4 }) { // 4: the widest length difference
          problem.set_band(band);
          problem.fill_distance_matrix();
          for (std::size_t i = 0; i < series.size(); ++i)
            for (std::size_t j = i + 1; j < series.size(); ++j) {
              const double fill = problem.dist_by_ind(static_cast<dtwc::index_t>(i), static_cast<dtwc::index_t>(j));
              const double facade = dtwc::distance::dtw<T>(std::span<const T>{ series[i] },
                                                           std::span<const T>{ series[j] }, params, band, metric, missing);
              ++pairs;
              INFO("band " << band << ", pair " << i << "," << j << ": fill " << fill << ", facade " << facade);
              CHECK(ts::dtw_routes_agree<T>(fill, facade, series[i].size(), series[j].size()));
            }
        }
      }
  CHECK(accepted == 20);
  CHECK(pairs == 20 * 2 * 10);
}

namespace {

/// The message of the dtwc::InvalidInput `call` throws, or "" when it throws none.
template <typename Call>
std::string message_of(Call &&call)
{
  try {
    call();
  } catch (const dtwc::InvalidInput &error) {
    return error.what();
  }
  return {};
}

/// Two series of two steps of `ndim` channels each.
dtwc::Data two_series(bool f32, std::size_t ndim)
{
  if (f32)
    return dtwc::Data{ std::vector<std::vector<float>>{ std::vector<float>(2 * ndim, 0.0f),
                                                        std::vector<float>(2 * ndim, 1.0f) },
                       { "x", "y" }, ndim };
  return dtwc::Data{ std::vector<std::vector<double>>{ std::vector<double>(2 * ndim, 0.0),
                                                       std::vector<double>(2 * ndim, 1.0) },
                     { "x", "y" }, ndim };
}

/// The facade's message for `c` on x = {0, NaN}: the configuration is checked before x is read.
template <typename T>
std::string facade_message(const DistanceConfig &c)
{
  const std::vector<T> x{ T(0), std::numeric_limits<T>::quiet_NaN() }, y{ T(0), T(1) };
  return message_of([&] { (void)dtwc::distance::dtw<T>(x, y, c.variant, c.band, c.metric, c.missing); });
}

/// A configuration no kernel implements, with the rule it breaks and the one message for it.
struct Refusal
{
  const char *rule;
  DistanceConfig config;
  bool f32;
  std::string message;
};

std::vector<Refusal> refusals()
{
  constexpr double inf = std::numeric_limits<double>::infinity();
  constexpr double nan = std::numeric_limits<double>::quiet_NaN();
  const std::string f32 = " cannot be represented in float32 without becoming zero or non-finite.";
  const std::string univariate = " is univariate in this release (ndim must be 1)";
  using V = DTWVariant;
  using M = MissingStrategy;
  constexpr auto independent = dtwc::core::MVMode::Independent;
  return {
    { "an enum value outside its set", { .metric = static_cast<MetricType>(7) }, false,
      "7 is not a MetricType value." },
    { "WDTW g >= 0", { .variant = { .variant = V::WDTW, .wdtw_g = -1 } }, false,
      "WDTW g must be finite and non-negative." },
    { "ADTW penalty >= 0", { .variant = { .variant = V::ADTW, .adtw_penalty = nan } }, false,
      "ADTW penalty must be finite and non-negative." },
    { "Soft-DTW gamma > 0", { .variant = { .variant = V::SoftDTW, .sdtw_gamma = 0 } }, false,
      "Soft-DTW gamma must be finite and positive." },
    { "MSM c > 0", { .variant = { .variant = V::MSM, .msm_c = -inf } }, false, "MSM c must be finite and positive." },
    { "TWE nu > 0", { .variant = { .variant = V::TWE, .twe_nu = 0 } }, false, "TWE nu must be finite and positive." },
    { "TWE lambda > 0", { .variant = { .variant = V::TWE, .twe_lambda = inf } }, false,
      "TWE lambda must be finite and positive." },
    { "an inactive parameter is checked too", { .variant = { .adtw_penalty = -1 } }, false,
      "ADTW penalty must be finite and non-negative." },
    { "a missing-data strategy takes Standard DTW", { .variant = { .variant = V::WDTW }, .missing = M::ZeroCost },
      false, "Non-Standard DTW variants require MissingStrategy::Error." },
    { "WDTW, ADTW, Soft-DTW, MSM and TWE take L1", { .variant = { .variant = V::ADTW }, .metric = MetricType::SquaredL2 },
      false,
      "metric SquaredL2 is implemented for Standard DTW and DDTW only, but variant = adtw was requested. Use "
      "metric L1 for this configuration." },
    { "float32 WDTW g", { .variant = { .variant = V::WDTW, .wdtw_g = 1e-300 } }, true, "WDTW g" + f32 },
    { "float32 ADTW penalty", { .variant = { .variant = V::ADTW, .adtw_penalty = 1e300 } }, true,
      "ADTW penalty" + f32 },
    { "float32 Soft-DTW gamma", { .variant = { .variant = V::SoftDTW, .sdtw_gamma = 1e-50 } }, true,
      "Soft-DTW gamma" + f32 },
    { "float32 MSM c", { .variant = { .variant = V::MSM, .msm_c = 1e39 } }, true, "MSM c" + f32 },
    { "float32 TWE nu", { .variant = { .variant = V::TWE, .twe_nu = 1e-46 } }, true, "TWE nu" + f32 },
    { "float32 TWE lambda", { .variant = { .variant = V::TWE, .twe_lambda = 4e38 } }, true, "TWE lambda" + f32 },
    { "independent channels take Standard DTW", { .variant = { .variant = V::ADTW, .mv_mode = independent }, .ndim = 2 },
      false, "Independent multivariate mode is implemented for the Standard DTW variant only in this release" },
    { "independent channels take no missing-data strategy",
      { .variant = { .mv_mode = independent }, .missing = M::ZeroCost, .ndim = 2 }, false,
      "Independent multivariate mode does not support a missing-data strategy in this release (set "
      "missing_strategy = Error)" },
    { "MSM is univariate", { .variant = { .variant = V::MSM }, .ndim = 2 }, false, "MSM distance" + univariate },
    { "TWE is univariate", { .variant = { .variant = V::TWE }, .ndim = 2 }, false, "TWE distance" + univariate },
    { "Soft-DTW is univariate", { .variant = { .variant = V::SoftDTW }, .ndim = 2 }, false, "Soft-DTW" + univariate },
    { "Interpolate is univariate", { .missing = M::Interpolate, .ndim = 2 }, false,
      "MissingStrategy::Interpolate" + univariate },
    { "AROW has no multivariate L2 cost", { .metric = MetricType::L2, .missing = M::AROW, .ndim = 2 }, false,
      "metric L2 with MissingStrategy::AROW" + univariate },
  };
}

} // namespace

TEST_CASE("core::validate refuses each rule's configuration, and so does every entry", "[distance_config][validate]")
{
  for (const auto &row : refusals()) {
    INFO(row.rule);
    CHECK(message_of([&] { dtwc::core::validate(row.config, row.f32); }) == row.message);

    // A Problem refuses it at the setter and keeps the settings it had.
    dtwc::Problem problem("refusal");
    problem.set_data(two_series(row.f32, row.config.ndim));
    const auto before = problem.distance();
    CHECK(message_of([&] { problem.set_distance(row.config); }) == row.message);
    CHECK(problem.distance() == before);

    // Float64 series take the float32-only refusals; the float32 function refuses them.
    if (row.f32) {
      dtwc::Problem f64("refusal_f64");
      f64.set_data(two_series(false, row.config.ndim));
      f64.set_distance(row.config);
      CHECK(message_of([&] { (void)f64.dtw_function_f32(); }) == row.message);
    }

    // The facade computes univariate series only.
    if (row.config.ndim == 1)
      CHECK((row.f32 ? facade_message<float>(row.config) : facade_message<double>(row.config)) == row.message);
  }
}
