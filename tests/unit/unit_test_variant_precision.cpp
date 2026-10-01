/**
 * @file unit_test_variant_precision.cpp
 * @brief Float32 representability at a Problem's data boundary (M45); the rule
 *        itself, per parameter, is core/test_distance_config.cpp's table.
 */

#include <dtwc.hpp>
#include <base/error.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <functional>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace dtwc;

namespace {

Data make_f64_data(double offset = 0.0)
{
  return Data(
    std::vector<std::vector<double>>{{offset, 1.0}, {offset, 2.0}},
    std::vector<std::string>{"x", "y"});
}

Data make_f32_data(float offset = 0.0f)
{
  return Data(
    std::vector<std::vector<float>>{{offset, 1.0f}, {offset, 2.0f}},
    std::vector<std::string>{"x", "y"});
}

struct DenseSnapshot
{
  const double *address;
  size_t size;
  size_t computed;
  double pair;
};

DenseSnapshot snapshot_dense(const Problem &problem)
{
  const auto &matrix = problem.distance_matrix();
  return {matrix.raw(), matrix.size(), matrix.count_computed(), matrix.get(0, 1)};
}

void require_dense_unchanged(const Problem &problem, const DenseSnapshot &before)
{
  const auto after = snapshot_dense(problem);
  CHECK(after.address == before.address);
  CHECK(after.size == before.size);
  CHECK(after.computed == before.computed);
  CHECK(after.pair == before.pair);
}

bool catches_exact(const std::function<void()> &operation,
                   std::string_view expected)
{
  try {
    operation();
  } catch (const InvalidInput &error) {
    CHECK(std::string_view(error.what()) == expected);
    return true;
  } catch (const std::exception &error) {
    FAIL("wrong exception type: " << error.what());
  }
  return false;
}

} // namespace

TEST_CASE("float32 heap and view data replacement validate before mutation",
          "[problem][variant][f32][data][transaction][m45]")
{
  auto configured_f64 = [] {
    Problem problem{"m45_data_transaction"};
    problem.set_data(make_f64_data());
    auto params = problem.variant_params();
    params.variant = core::DTWVariant::SoftDTW;
    params.sdtw_gamma = std::numeric_limits<double>::min();
    problem.set_variant(params);
    problem.fill_distance_matrix();
    problem.clusters_ind = {0, 1};
    problem.centroids_ind = {0, 1};
    return problem;
  };

  SECTION("owning set_data")
  {
    auto problem = configured_f64();
    const auto original_cache = snapshot_dense(problem);
    const bool caught = catches_exact(
      [&] { problem.set_data(make_f32_data(10.0f)); },
      "Soft-DTW gamma cannot be represented in float32 without becoming zero or non-finite.");
    CHECK(caught);
    const bool preserved_f64 = !problem.data().is_f32();
    CHECK(preserved_f64);
    if (preserved_f64) CHECK(problem.series(0)[0] == 0.0);
    CHECK(problem.labels() == std::vector<index_t>{0, 1});
    CHECK(problem.medoids() == std::vector<index_t>{0, 1});
    if (caught) require_dense_unchanged(problem, original_cache);
  }

  SECTION("non-owning set_view_data")
  {
    auto problem = configured_f64();
    const auto original_cache = snapshot_dense(problem);
    std::vector<float> x{10.0f, 11.0f};
    std::vector<float> y{12.0f, 13.0f};
    std::vector<std::span<const float>> spans{std::span<const float>{x},
                                              std::span<const float>{y}};
    std::vector<std::string_view> names{"vx", "vy"};
    Data view(std::move(spans), std::move(names), 1);

    const bool caught = catches_exact(
      [&] { problem.set_view_data(std::move(view)); },
      "Soft-DTW gamma cannot be represented in float32 without becoming zero or non-finite.");
    CHECK(caught);
    const bool preserved_f64 = !problem.data().is_f32();
    CHECK(preserved_f64);
    if (preserved_f64) {
      CHECK_FALSE(problem.data().is_view());
      CHECK(problem.series(0)[0] == 0.0);
    }
    CHECK(problem.labels() == std::vector<index_t>{0, 1});
    CHECK(problem.medoids() == std::vector<index_t>{0, 1});
    if (caught) require_dense_unchanged(problem, original_cache);
  }
}

TEST_CASE("float32 representable boundaries and inactive parameters remain valid",
          "[problem][variant][f32][boundary][m45]")
{
  const double smallest_f32 = static_cast<double>(std::numeric_limits<float>::denorm_min());

  SECTION("inactive unrepresentable parameter does not restrict Standard")
  {
    Problem problem{"m45_inactive"};
    problem.set_data(make_f32_data());
    auto params = problem.variant_params();
    params.sdtw_gamma = std::numeric_limits<double>::min();
    REQUIRE_NOTHROW(problem.set_variant(params));
    REQUIRE_NOTHROW((void)problem.dtw_function_f32());
    REQUIRE(problem.variant_params().variant == core::DTWVariant::Standard);
  }

  SECTION("exact zero limits")
  {
    for (const auto variant : {core::DTWVariant::WDTW, core::DTWVariant::ADTW}) {
      Problem problem{"m45_zero"};
      problem.set_data(make_f32_data());
      auto params = problem.variant_params();
      params.variant = variant;
      if (variant == core::DTWVariant::WDTW) params.wdtw_g = 0.0;
      else params.adtw_penalty = 0.0;
      REQUIRE_NOTHROW(problem.set_variant(params));
      const auto &function = problem.dtw_function_f32();
      REQUIRE(std::isfinite(function(problem.data().series_f32(0),
                                     problem.data().series_f32(1))));
    }
  }

  SECTION("minimum positive float32 values")
  {
    for (const auto variant : {core::DTWVariant::SoftDTW,
                               core::DTWVariant::MSM,
                               core::DTWVariant::TWE}) {
      Problem problem{"m45_denorm"};
      problem.set_data(make_f32_data());
      auto params = problem.variant_params();
      params.variant = variant;
      if (variant == core::DTWVariant::SoftDTW) params.sdtw_gamma = smallest_f32;
      if (variant == core::DTWVariant::MSM) params.msm_c = smallest_f32;
      if (variant == core::DTWVariant::TWE) {
        params.twe_nu = smallest_f32;
        params.twe_lambda = smallest_f32;
      }
      REQUIRE_NOTHROW(problem.set_variant(params));
      const auto &function = problem.dtw_function_f32();
      REQUIRE(std::isfinite(function(problem.data().series_f32(0),
                                     problem.data().series_f32(1))));
    }
  }
}
