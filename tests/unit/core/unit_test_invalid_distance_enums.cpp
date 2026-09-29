/**
 * @file unit_test_invalid_distance_enums.cpp
 * @brief Every declared selector value, and every spelling of one, still works.
 *
 * An enum-class selector reaches the library from a user only as a name
 * (bindings, CLI, config files), and each name table rejects an unknown
 * spelling (tests/python/test_invalid_distance_enums.py pins the binding side);
 * every switch over an enum is exhaustive at compile time (-Werror=switch,
 * /we4062). What is left to prove here is the positive side: the legitimate
 * values are accepted and keep their meaning.
 */

#include <dtwc.hpp>

#include <core/distance_semantics.hpp>
#include <core/dtw_dispatch.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

using namespace dtwc;

namespace {

Data basic_f64_data(std::size_t ndim = 1)
{
  if (ndim == 1) {
    return Data(
      std::vector<std::vector<double>>{{0.0, 0.0}, {0.0, 1.0, 2.0}},
      std::vector<std::string>{"x", "y"});
  }
  return Data(
    std::vector<std::vector<double>>{
      {0.0, 10.0, 1.0, 11.0},
      {0.0, 10.0, 2.0, 12.0}},
    std::vector<std::string>{"x", "y"}, ndim);
}

Data basic_f32_data(std::size_t ndim = 1)
{
  if (ndim == 1) {
    return Data(
      std::vector<std::vector<float>>{{0.0f, 0.0f}, {0.0f, 1.0f, 2.0f}},
      std::vector<std::string>{"x", "y"});
  }
  return Data(
    std::vector<std::vector<float>>{
      {0.0f, 10.0f, 1.0f, 11.0f},
      {0.0f, 10.0f, 2.0f, 12.0f}},
    std::vector<std::string>{"x", "y"}, ndim);
}

} // namespace

TEST_CASE("M47 legitimate selectors and aliases retain registered fingerprints",
          "[m47][enum][control]")
{
  const std::vector<double> x{0.0, 0.0, 10.0};
  const std::vector<double> y{0.0, 10.0, 10.0};

  CHECK(core::parse_metric_token("l1") == core::MetricType::L1);
  CHECK(core::parse_metric_token("squared_euclidean")
        == core::MetricType::SquaredL2);
  CHECK(core::parse_metric_token("sqeuclidean")
        == core::MetricType::SquaredL2);
  CHECK(dtwFull<double>(x.data(), x.size(), y.data(), y.size(),
                        core::MetricType::L1)
        == dtwFull<double>(x.data(), x.size(), y.data(), y.size(),
                           core::MetricType::L2));
  for (const auto metric : {
         core::MetricType::L1,
         core::MetricType::L2,
         core::MetricType::SquaredL2}) {
    CHECK_NOTHROW((void)dtwFull<double>(
      x.data(), x.size(), y.data(), y.size(), metric));
  }

  core::DTWOptions options;
  options.band = 0;
  options.constraint = core::ConstraintType::None;
  CHECK(core::dtw_runtime(x.data(), x.size(), y.data(), y.size(), options) == 0.0);
  options.constraint = core::ConstraintType::SakoeChibaBand;
  CHECK(core::dtw_runtime(x.data(), x.size(), y.data(), y.size(), options) == 10.0);

  for (const auto variant : {
         core::DTWVariant::Standard,
         core::DTWVariant::DDTW,
         core::DTWVariant::WDTW,
         core::DTWVariant::ADTW,
         core::DTWVariant::SoftDTW,
         core::DTWVariant::MSM,
         core::DTWVariant::TWE}) {
    core::DTWVariantParams params;
    params.variant = variant;
    Problem problem{"m47_valid_variant"};
    problem.set_data(basic_f64_data());
    CHECK_NOTHROW(problem.set_variant(params));
    CHECK_NOTHROW((void)core::resolve_dtw_fn<double>(problem));
  }

  for (const auto missing : {
         core::MissingStrategy::Error,
         core::MissingStrategy::ZeroCost,
         core::MissingStrategy::AROW,
         core::MissingStrategy::Interpolate}) {
    Problem problem{"m47_valid_missing"};
    problem.set_data(basic_f64_data());
    CHECK_NOTHROW(problem.set_missing_strategy(missing));
  }

  Problem dependent{"m47_valid_dependent"};
  dependent.set_data(basic_f64_data(2));
  CHECK_NOTHROW((void)core::resolve_dtw_fn<double>(dependent));
  core::DTWVariantParams independent_params;
  independent_params.mv_mode = core::MVMode::Independent;
  dependent.set_variant(independent_params);
  CHECK_NOTHROW((void)core::resolve_dtw_fn<double>(dependent));

  for (const auto strategy : {
         DistanceMatrixStrategy::Auto,
         DistanceMatrixStrategy::BruteForce,
         DistanceMatrixStrategy::CUDA,
         DistanceMatrixStrategy::Metal}) {
    Problem problem{"m47_valid_matrix_strategy"};
    CHECK_NOTHROW(problem.set_distance_strategy(strategy));
  }

  Problem f32_problem{"m47_valid_f32_precision"};
  CHECK_NOTHROW(f32_problem.set_data(basic_f32_data()));
  CHECK(f32_problem.data().precision == core::Precision::Float32);
  Problem f64_problem{"m47_valid_f64_precision"};
  CHECK_NOTHROW(f64_problem.set_data(basic_f64_data()));
  CHECK(f64_problem.data().precision == core::Precision::Float64);

  for (const auto precision : {GpuPrecision::Auto, GpuPrecision::FP32, GpuPrecision::FP64}) {
    Problem problem{"m47_valid_cuda_settings_precision"};
    CUDASettings settings;
    settings.precision = precision;
    CHECK_NOTHROW(problem.set_cuda_settings(settings));
  }

#if defined(DTWC_HAS_CUDA)
  constexpr std::array<cuda::CUDAPrecision, 3> cuda_precisions{
    cuda::CUDAPrecision::Auto,
    cuda::CUDAPrecision::FP32,
    cuda::CUDAPrecision::FP64
  };
  for (std::size_t i = 0; i < cuda_precisions.size(); ++i)
    CHECK(static_cast<int>(cuda_precisions[i]) == static_cast<int>(i));
#endif

#if defined(DTWC_HAS_METAL)
  constexpr std::array<metal::MetalPrecision, 3> metal_precisions{
    metal::MetalPrecision::Auto,
    metal::MetalPrecision::FP32,
    metal::MetalPrecision::FP64
  };
  for (std::size_t i = 0; i < metal_precisions.size(); ++i)
    CHECK(static_cast<int>(metal_precisions[i]) == static_cast<int>(i));
#endif
}
