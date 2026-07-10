/**
 * @file unit_test_invalid_distance_enums.cpp
 * @brief M47 preregistration: no invalid public distance selector may fall back.
 *
 * Every enum is probed at four distinct out-of-range underlying values:
 * -1, first value above the declared domain, INT_MIN, and INT_MAX.  The test
 * intentionally spans free/facade, runtime, Problem, resolver, storage, cache,
 * and checkpoint boundaries.  Production validation is added only after this
 * Release-mode red is committed.
 */

#include <dtwc.hpp>

#include <core/distance_semantics.hpp>
#include <core/dtw_dispatch.hpp>
#include <core/pruned_distance_matrix.hpp>
#include <enums/KernelOverride.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <climits>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using namespace dtwc;

namespace {

constexpr std::string_view variant_error = "Invalid DTWVariant value.";
constexpr std::string_view missing_error = "Invalid MissingStrategy value.";
constexpr std::string_view metric_error = "Invalid MetricType value.";
constexpr std::string_view mv_mode_error = "Invalid MVMode value.";
constexpr std::string_view constraint_error = "Invalid ConstraintType value.";
constexpr std::string_view matrix_strategy_error =
  "Invalid DistanceMatrixStrategy value.";
constexpr std::string_view lower_bound_error =
  "Invalid LowerBoundStrategy value.";
constexpr std::string_view storage_policy_error =
  "Invalid StoragePolicy value.";
constexpr std::string_view precision_error = "Invalid Precision value.";
constexpr std::string_view cuda_settings_precision_error =
  "Invalid CUDA precision value.";
constexpr std::string_view kernel_override_error =
  "Invalid KernelOverride value.";
constexpr std::string_view cuda_precision_error =
  "Invalid CUDAPrecision value.";
constexpr std::string_view metal_precision_error =
  "Invalid MetalPrecision value.";

template <typename Enum, Enum Last, typename Function>
void for_each_invalid_enum(Function &&function)
{
  static_assert(std::is_same_v<std::underlying_type_t<Enum>, int>);
  constexpr int first_above = static_cast<int>(Last) + 1;
  constexpr std::array<int, 4> invalid_values{
    -1, first_above, INT_MIN, INT_MAX
  };
  for (const int raw : invalid_values) {
    CAPTURE(raw);
    std::forward<Function>(function)(static_cast<Enum>(raw));
  }
}

template <typename Function>
void check_invalid_input(
  std::string_view boundary, std::string_view expected, Function &&function)
{
  INFO("boundary=" << boundary);
  bool caught = false;
  try {
    std::forward<Function>(function)();
  } catch (const InvalidInput &error) {
    caught = true;
    CHECK(std::string_view(error.what()) == expected);
  } catch (const std::exception &error) {
    FAIL_CHECK("wrong exception type: " << error.what());
  } catch (...) {
    FAIL_CHECK("wrong non-standard exception type");
  }
  CHECK(caught);
}

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

void seed_dense_sentinel(Problem &problem, double sentinel = 123.0)
{
  problem.set_data(basic_f64_data());
  auto &matrix = problem.dense_distance_matrix();
  matrix.resize(2);
  matrix.set(0, 0, 0.0);
  matrix.set(0, 1, sentinel);
  matrix.set(1, 1, 0.0);
  REQUIRE(problem.is_distance_matrix_filled());
}

void check_dense_sentinel(const Problem &problem, double sentinel = 123.0)
{
  try {
    const auto &matrix = problem.dense_distance_matrix();
    CHECK(matrix.size() == 2);
    if (matrix.size() == 2) {
      CHECK(matrix.all_computed());
      CHECK(matrix.get(0, 1) == sentinel);
    }
  } catch (const std::exception &error) {
    FAIL_CHECK("sentinel inspection threw: " << error.what());
  }
}

void check_dense_unallocated(const Problem &problem)
{
  try {
    CHECK(problem.dense_distance_matrix().size() == 0);
  } catch (const std::exception &error) {
    FAIL_CHECK("matrix inspection threw: " << error.what());
  }
}

struct ScratchDirectory {
  fs::path root;

  explicit ScratchDirectory(std::string_view stem)
    : root(fs::temp_directory_path()
           / (std::string(stem) + "_"
              + std::to_string(reinterpret_cast<std::uintptr_t>(this))))
  {
    fs::create_directories(root);
  }

  ~ScratchDirectory()
  {
    std::error_code error;
    fs::remove_all(root, error);
  }
};

} // namespace

TEST_CASE("M47 rejects every invalid MetricType at public distance boundaries",
          "[m47][enum][metric]")
{
  const std::vector<double> x{0.0, 1.0};
  const std::vector<double> y{0.0, 2.0};
  const std::vector<float> xf{0.0f, 1.0f};
  const std::vector<float> yf{0.0f, 2.0f};
  const std::vector<double> xmv{0.0, 10.0, 1.0, 11.0};
  const std::vector<double> ymv{0.0, 10.0, 2.0, 12.0};
  const std::vector<float> xmvf{0.0f, 10.0f, 1.0f, 11.0f};
  const std::vector<float> ymvf{0.0f, 10.0f, 2.0f, 12.0f};

  for_each_invalid_enum<core::MetricType, core::MetricType::SquaredL2>(
    [&](core::MetricType invalid) {
      check_invalid_input("dtwFull f64", metric_error, [&] {
        (void)dtwFull<double>(x.data(), x.size(), y.data(), y.size(), invalid);
      });
      check_invalid_input("dtwFull f32", metric_error, [&] {
        (void)dtwFull<float>(xf.data(), xf.size(), yf.data(), yf.size(), invalid);
      });
      check_invalid_input("dtwFull_L f64", metric_error, [&] {
        (void)dtwFull_L<double>(
          x.data(), x.size(), y.data(), y.size(), -1.0, invalid);
      });
      check_invalid_input("dtwFull_L f32", metric_error, [&] {
        (void)dtwFull_L<float>(
          xf.data(), xf.size(), yf.data(), yf.size(), -1.0f, invalid);
      });
      check_invalid_input("dtwFull_eap f64", metric_error, [&] {
        (void)dtwFull_eap<double>(
          x.data(), x.size(), y.data(), y.size(), invalid);
      });
      check_invalid_input("dtwFull_eap f32", metric_error, [&] {
        (void)dtwFull_eap<float>(
          xf.data(), xf.size(), yf.data(), yf.size(), invalid);
      });
      check_invalid_input("dtwBanded f64", metric_error, [&] {
        (void)dtwBanded<double>(
          x.data(), x.size(), y.data(), y.size(), 0, -1.0, invalid);
      });
      check_invalid_input("dtwBanded f32", metric_error, [&] {
        (void)dtwBanded<float>(
          xf.data(), xf.size(), yf.data(), yf.size(), 0, -1.0f, invalid);
      });
      check_invalid_input("dtwFull_L_mv f64", metric_error, [&] {
        (void)dtwFull_L_mv<double>(
          xmv.data(), 2, ymv.data(), 2, 2, -1.0, invalid);
      });
      check_invalid_input("dtwFull_L_mv f32", metric_error, [&] {
        (void)dtwFull_L_mv<float>(
          xmvf.data(), 2, ymvf.data(), 2, 2, -1.0f, invalid);
      });
      check_invalid_input("dtwBanded_mv f64", metric_error, [&] {
        (void)dtwBanded_mv<double>(
          xmv.data(), 2, ymv.data(), 2, 2, 0, -1.0, invalid);
      });
      check_invalid_input("dtwBanded_mv f32", metric_error, [&] {
        (void)dtwBanded_mv<float>(
          xmvf.data(), 2, ymvf.data(), 2, 2, 0, -1.0f, invalid);
      });
      check_invalid_input("dtw_independent_mv f64", metric_error, [&] {
        (void)dtw_independent_mv<double>(
          xmv.data(), 2, ymv.data(), 2, 2, 0, invalid);
      });
      check_invalid_input("dtw_independent_mv f32", metric_error, [&] {
        (void)dtw_independent_mv<float>(
          xmvf.data(), 2, ymvf.data(), 2, 2, 0, invalid);
      });
      check_invalid_input("dtwMissing_L f64", metric_error, [&] {
        (void)dtwMissing_L<double>(
          x.data(), x.size(), y.data(), y.size(), -1.0, invalid);
      });
      check_invalid_input("dtwMissing_L f32", metric_error, [&] {
        (void)dtwMissing_L<float>(
          xf.data(), xf.size(), yf.data(), yf.size(), -1.0f, invalid);
      });
      check_invalid_input("dtwMissing full f64", metric_error, [&] {
        (void)dtwMissing<double>(
          x.data(), x.size(), y.data(), y.size(), invalid);
      });
      check_invalid_input("dtwMissing full f32", metric_error, [&] {
        (void)dtwMissing<float>(
          xf.data(), xf.size(), yf.data(), yf.size(), invalid);
      });
      check_invalid_input("dtwMissing_banded f64", metric_error, [&] {
        (void)dtwMissing_banded<double>(
          x.data(), x.size(), y.data(), y.size(), 0, -1.0, invalid);
      });
      check_invalid_input("dtwMissing_banded f32", metric_error, [&] {
        (void)dtwMissing_banded<float>(
          xf.data(), xf.size(), yf.data(), yf.size(), 0, -1.0f, invalid);
      });
      check_invalid_input("dtwMissing_L_mv f64", metric_error, [&] {
        (void)dtwMissing_L_mv<double>(
          xmv.data(), 2, ymv.data(), 2, 2, -1.0, invalid);
      });
      check_invalid_input("dtwMissing_L_mv f32", metric_error, [&] {
        (void)dtwMissing_L_mv<float>(
          xmvf.data(), 2, ymvf.data(), 2, 2, -1.0f, invalid);
      });
      check_invalid_input("dtwMissing_banded_mv f64", metric_error, [&] {
        (void)dtwMissing_banded_mv<double>(
          xmv.data(), 2, ymv.data(), 2, 2, 0, -1.0, invalid);
      });
      check_invalid_input("dtwMissing_banded_mv f32", metric_error, [&] {
        (void)dtwMissing_banded_mv<float>(
          xmvf.data(), 2, ymvf.data(), 2, 2, 0, -1.0f, invalid);
      });
      check_invalid_input("dtwAROW_L f64", metric_error, [&] {
        (void)dtwAROW_L<double>(
          x.data(), x.size(), y.data(), y.size(), invalid);
      });
      check_invalid_input("dtwAROW_L f32", metric_error, [&] {
        (void)dtwAROW_L<float>(
          xf.data(), xf.size(), yf.data(), yf.size(), invalid);
      });
      check_invalid_input("dtwAROW full f64", metric_error, [&] {
        (void)dtwAROW<double>(
          x.data(), x.size(), y.data(), y.size(), invalid);
      });
      check_invalid_input("dtwAROW full f32", metric_error, [&] {
        (void)dtwAROW<float>(
          xf.data(), xf.size(), yf.data(), yf.size(), invalid);
      });
      check_invalid_input("dtwAROW_banded f64", metric_error, [&] {
        (void)dtwAROW_banded<double>(
          x.data(), x.size(), y.data(), y.size(), 0, invalid);
      });
      check_invalid_input("dtwAROW_banded f32", metric_error, [&] {
        (void)dtwAROW_banded<float>(
          xf.data(), xf.size(), yf.data(), yf.size(), 0, invalid);
      });
      check_invalid_input("DDTW full f64", metric_error, [&] {
        (void)ddtwFull_L<double>(
          x.data(), x.size(), y.data(), y.size(), invalid);
      });
      check_invalid_input("DDTW full f32", metric_error, [&] {
        (void)ddtwFull_L<float>(
          xf.data(), xf.size(), yf.data(), yf.size(), invalid);
      });
      check_invalid_input("DDTW banded f64", metric_error, [&] {
        (void)ddtwBanded<double>(
          x.data(), x.size(), y.data(), y.size(), 0, invalid);
      });
      check_invalid_input("DDTW banded f32", metric_error, [&] {
        (void)ddtwBanded<float>(
          xf.data(), xf.size(), yf.data(), yf.size(), 0, invalid);
      });
      check_invalid_input("DDTW full span f64", metric_error, [&] {
        (void)ddtwFull_L<double>(
          std::span<const double>{x}, std::span<const double>{y}, invalid);
      });
      check_invalid_input("DDTW full span f32", metric_error, [&] {
        (void)ddtwFull_L<float>(
          std::span<const float>{xf}, std::span<const float>{yf}, invalid);
      });
      check_invalid_input("DDTW banded span f64", metric_error, [&] {
        (void)ddtwBanded<double>(
          std::span<const double>{x}, std::span<const double>{y}, 0, invalid);
      });
      check_invalid_input("DDTW banded span f32", metric_error, [&] {
        (void)ddtwBanded<float>(
          std::span<const float>{xf}, std::span<const float>{yf}, 0, invalid);
      });

      check_invalid_input("core::dtw_distance vector f64", metric_error, [&] {
        (void)core::dtw_distance<double>(x, y, -1, invalid);
      });
      check_invalid_input("core::dtw_distance vector f32", metric_error, [&] {
        (void)core::dtw_distance<float>(xf, yf, -1, invalid);
      });
      check_invalid_input("core::dtw_distance pointer f64", metric_error, [&] {
        (void)core::dtw_distance<double>(
          x.data(), x.size(), y.data(), y.size(), -1, invalid);
      });
      check_invalid_input("core::dtw_distance pointer f32", metric_error, [&] {
        (void)core::dtw_distance<float>(
          xf.data(), xf.size(), yf.data(), yf.size(), -1, invalid);
      });

      const std::vector<double> empty;
      check_invalid_input("dtwMissing empty shortcut", metric_error, [&] {
        (void)dtwMissing_L<double>(
          empty.data(), 0, y.data(), y.size(), -1.0, invalid);
      });
      check_invalid_input("dtwMissing self shortcut", metric_error, [&] {
        (void)dtwMissing_L<double>(
          x.data(), x.size(), x.data(), x.size(), -1.0, invalid);
      });
      check_invalid_input("dtwMissing MV empty shortcut", metric_error, [&] {
        (void)dtwMissing_L_mv<double>(
          empty.data(), 0, ymv.data(), 2, 2, -1.0, invalid);
      });
      check_invalid_input("dtwMissing MV self shortcut", metric_error, [&] {
        (void)dtwMissing_L_mv<double>(
          xmv.data(), 2, xmv.data(), 2, 2, -1.0, invalid);
      });
      check_invalid_input("AROW empty shortcut", metric_error, [&] {
        (void)dtwAROW_L<double>(
          empty.data(), 0, y.data(), y.size(), invalid);
      });
      check_invalid_input("AROW self shortcut", metric_error, [&] {
        (void)dtwAROW_L<double>(
          x.data(), x.size(), x.data(), x.size(), invalid);
      });
      check_invalid_input("DDTW empty pre-allocation", metric_error, [&] {
        (void)ddtwFull_L<double>(
          empty.data(), 0, y.data(), y.size(), invalid);
      });
      check_invalid_input("DDTW self pre-allocation", metric_error, [&] {
        (void)ddtwFull_L<double>(
          x.data(), x.size(), x.data(), x.size(), invalid);
      });
      check_invalid_input("dtwBanded empty shortcut", metric_error, [&] {
        (void)dtwBanded<double>(
          empty.data(), 0, y.data(), y.size(), 0, -1.0, invalid);
      });
      check_invalid_input("dtwBanded_mv empty shortcut", metric_error, [&] {
        (void)dtwBanded_mv<double>(
          empty.data(), 0, ymv.data(), 2, 2, 0, -1.0, invalid);
      });
      check_invalid_input("dtw_independent_mv empty shortcut", metric_error, [&] {
        (void)dtw_independent_mv<double>(
          empty.data(), 0, ymv.data(), 2, 2, 0, invalid);
      });
      check_invalid_input("distance facade f64", metric_error, [&] {
        (void)distance::dtw<double>(x, y, -1, invalid);
      });
      check_invalid_input("distance facade f32", metric_error, [&] {
        (void)distance::dtw<float>(xf, yf, -1, invalid);
      });
      check_invalid_input("missing facade", metric_error, [&] {
        (void)distance::missing<double>(x, y, -1, invalid);
      });
      check_invalid_input("AROW facade", metric_error, [&] {
        (void)distance::arow<double>(x, y, -1, invalid);
      });
      check_invalid_input("DDTW facade f64", metric_error, [&] {
        (void)distance::ddtw<double>(x, y, -1, invalid);
      });
      check_invalid_input("DDTW facade f32", metric_error, [&] {
        (void)distance::ddtw<float>(xf, yf, -1, invalid);
      });

      core::DTWVariantParams interpolate_params;
      check_invalid_input("Interpolate facade f64", metric_error, [&] {
        (void)distance::dtw<double>(
          x, y, interpolate_params, -1, invalid,
          core::MissingStrategy::Interpolate);
      });
      check_invalid_input("Interpolate facade f32", metric_error, [&] {
        (void)distance::dtw<float>(
          xf, yf, interpolate_params, -1, invalid,
          core::MissingStrategy::Interpolate);
      });

      for (const auto ignored_metric_variant : {
             core::DTWVariant::WDTW,
             core::DTWVariant::ADTW,
             core::DTWVariant::SoftDTW,
             core::DTWVariant::MSM,
             core::DTWVariant::TWE}) {
        CAPTURE(static_cast<int>(ignored_metric_variant));
        core::DTWVariantParams ignored_params;
        ignored_params.variant = ignored_metric_variant;
        check_invalid_input("ignored-metric facade f64", metric_error, [&] {
          (void)distance::dtw<double>(x, y, ignored_params, -1, invalid);
        });
        check_invalid_input("ignored-metric facade f32", metric_error, [&] {
          (void)distance::dtw<float>(xf, yf, ignored_params, -1, invalid);
        });

        core::DTWOptions ignored_options;
        ignored_options.metric = invalid;
        ignored_options.variant_params = ignored_params;
        check_invalid_input("ignored-metric dtw_runtime", metric_error, [&] {
          (void)core::dtw_runtime(
            x.data(), x.size(), y.data(), y.size(), ignored_options);
        });
      }

      std::array<double, 4> pruned_output{11.0, 12.0, 13.0, 14.0};
      const std::vector<std::vector<double>> pruned_series{x, y};
      check_invalid_input("compute_distance_matrix_pruned", metric_error, [&] {
        (void)core::compute_distance_matrix_pruned(
          pruned_series, pruned_output.data(), 0, invalid);
      });
      const std::array<double, 4> expected_pruned_output{11.0, 12.0, 13.0, 14.0};
      CHECK(pruned_output == expected_pruned_output);

      core::DTWOptions options;
      options.metric = invalid;
      check_invalid_input("dtw_runtime", metric_error, [&] {
        (void)core::dtw_runtime(
          x.data(), x.size(), y.data(), y.size(), options);
      });

      core::DTWOptions interpolate_options;
      interpolate_options.metric = invalid;
      interpolate_options.missing_strategy =
        core::MissingStrategy::Interpolate;
      check_invalid_input("Interpolate dtw_runtime", metric_error, [&] {
        (void)core::dtw_runtime(
          x.data(), x.size(), y.data(), y.size(), interpolate_options);
      });

      ScratchDirectory scratch{"m47_metric_cache"};
      const fs::path cache = scratch.root / "invalid.dtwcache";
      Problem problem{"m47_metric_cache"};
      problem.set_data(basic_f64_data());
      check_invalid_input("mmap cache identity", metric_error, [&] {
        problem.use_mmap_distance_matrix(cache, invalid);
      });
      CHECK_FALSE(fs::exists(cache));
    });
}

TEST_CASE("M47 rejects every invalid DTWVariant before dispatch or mutation",
          "[m47][enum][variant]")
{
  const std::vector<double> x{0.0, 0.0};
  const std::vector<double> y{0.0, 1.0, 2.0};
  const std::vector<float> xf{0.0f, 0.0f};
  const std::vector<float> yf{0.0f, 1.0f, 2.0f};

  for_each_invalid_enum<core::DTWVariant, core::DTWVariant::TWE>(
    [&](core::DTWVariant invalid) {
      core::DTWVariantParams params;
      params.variant = invalid;

      check_invalid_input("validate_variant_params", variant_error, [&] {
        core::validate_variant_params(params);
      });
      const char *f32_error = core::active_variant_params_f32_error(params);
      CHECK(f32_error != nullptr);
      if (f32_error != nullptr)
        CHECK(std::string_view(f32_error) == variant_error);
      CHECK_FALSE(core::active_variant_params_representable_f32(params));
      check_invalid_input("validate_active_variant_params_f32",
                          variant_error, [&] {
        core::validate_active_variant_params_f32(params);
      });
      check_invalid_input("validate_variant_missing_semantics", variant_error, [&] {
        core::validate_variant_missing_semantics(
          invalid, core::MissingStrategy::Error);
      });
      check_invalid_input("distance facade f64", variant_error, [&] {
        (void)distance::dtw<double>(x, y, params);
      });
      check_invalid_input("distance facade f32", variant_error, [&] {
        (void)distance::dtw<float>(xf, yf, params);
      });

      core::DTWOptions options;
      options.variant_params = params;
      check_invalid_input("dtw_runtime", variant_error, [&] {
        (void)core::dtw_runtime(
          x.data(), x.size(), y.data(), y.size(), options);
      });

      Problem resolver64{"m47_variant_resolver64"};
      resolver64.set_data(basic_f64_data());
      resolver64.variant_params = params;
      check_invalid_input("resolve_dtw_fn f64", variant_error, [&] {
        (void)core::resolve_dtw_fn<double>(resolver64);
      });

      Problem resolver32{"m47_variant_resolver32"};
      resolver32.set_data(basic_f32_data());
      resolver32.variant_params = params;
      check_invalid_input("resolve_dtw_fn f32", variant_error, [&] {
        (void)core::resolve_dtw_fn<float>(resolver32);
      });

      Problem getter64{"m47_variant_getter64"};
      getter64.set_data(basic_f64_data());
      getter64.variant_params = params;
      check_invalid_input("Problem::dtw_function", variant_error, [&] {
        (void)getter64.dtw_function();
      });

      Problem getter32{"m47_variant_getter32"};
      getter32.set_data(basic_f32_data());
      getter32.variant_params = params;
      check_invalid_input("Problem::dtw_function_f32", variant_error, [&] {
        (void)getter32.dtw_function_f32();
      });

      Problem enum_setter{"m47_variant_enum_setter"};
      seed_dense_sentinel(enum_setter);
      check_invalid_input("Problem::set_variant(enum)", variant_error, [&] {
        enum_setter.set_variant(invalid);
      });
      CHECK(enum_setter.variant_params.variant == core::DTWVariant::Standard);
      check_dense_sentinel(enum_setter);

      Problem params_setter{"m47_variant_params_setter"};
      seed_dense_sentinel(params_setter);
      check_invalid_input("Problem::set_variant(params)", variant_error, [&] {
        params_setter.set_variant(params);
      });
      CHECK(params_setter.variant_params.variant == core::DTWVariant::Standard);
      check_dense_sentinel(params_setter);

      Problem fill{"m47_variant_fill"};
      fill.set_data(basic_f64_data());
      fill.variant_params = params;
      check_invalid_input("Problem::fill_distance_matrix", variant_error, [&] {
        fill.fill_distance_matrix();
      });
      fill.variant_params = {};
      check_dense_unallocated(fill);

      ScratchDirectory cache_scratch{"m47_variant_cache"};
      const fs::path cache = cache_scratch.root / "invalid.dtwcache";
      Problem cache_problem{"m47_variant_cache"};
      cache_problem.set_data(basic_f64_data());
      cache_problem.variant_params = params;
      check_invalid_input("mmap cache identity", variant_error, [&] {
        cache_problem.use_mmap_distance_matrix(cache);
      });
      CHECK_FALSE(fs::exists(cache));
      cache_problem.variant_params = {};
      check_dense_unallocated(cache_problem);

      ScratchDirectory scratch{"m47_variant_checkpoint"};
      const fs::path checkpoint = scratch.root / "checkpoint";
      Problem save{"m47_variant_checkpoint"};
      seed_dense_sentinel(save);
      save.variant_params = params;
      check_invalid_input("save_checkpoint", variant_error, [&] {
        save_checkpoint(save, checkpoint.string());
      });
      CHECK_FALSE(fs::exists(checkpoint));
    });
}

TEST_CASE("M47 rejects every invalid MissingStrategy before dispatch or mutation",
          "[m47][enum][missing]")
{
  const std::vector<double> x{0.0, 0.0};
  const std::vector<double> y{0.0, 1.0};
  const std::vector<float> xf{0.0f, 0.0f};
  const std::vector<float> yf{0.0f, 1.0f};

  for_each_invalid_enum<core::MissingStrategy, core::MissingStrategy::Interpolate>(
    [&](core::MissingStrategy invalid) {
      const core::DTWVariantParams params;
      check_invalid_input("validate_variant_missing_semantics", missing_error, [&] {
        core::validate_variant_missing_semantics(
          core::DTWVariant::Standard, invalid);
      });
      check_invalid_input("distance facade f64", missing_error, [&] {
        (void)distance::dtw<double>(x, y, params, -1, core::MetricType::L1, invalid);
      });
      check_invalid_input("distance facade f32", missing_error, [&] {
        (void)distance::dtw<float>(xf, yf, params, -1, core::MetricType::L1, invalid);
      });

      core::DTWOptions options;
      options.missing_strategy = invalid;
      check_invalid_input("dtw_runtime", missing_error, [&] {
        (void)core::dtw_runtime(
          x.data(), x.size(), y.data(), y.size(), options);
      });

      Problem resolver64{"m47_missing_resolver64"};
      resolver64.set_data(basic_f64_data());
      resolver64.missing_strategy = invalid;
      check_invalid_input("resolve_dtw_fn f64", missing_error, [&] {
        (void)core::resolve_dtw_fn<double>(resolver64);
      });

      Problem resolver32{"m47_missing_resolver32"};
      resolver32.set_data(basic_f32_data());
      resolver32.missing_strategy = invalid;
      check_invalid_input("resolve_dtw_fn f32", missing_error, [&] {
        (void)core::resolve_dtw_fn<float>(resolver32);
      });

      Problem getter64{"m47_missing_getter64"};
      getter64.set_data(basic_f64_data());
      getter64.missing_strategy = invalid;
      check_invalid_input("Problem::dtw_function", missing_error, [&] {
        (void)getter64.dtw_function();
      });

      Problem getter32{"m47_missing_getter32"};
      getter32.set_data(basic_f32_data());
      getter32.missing_strategy = invalid;
      check_invalid_input("Problem::dtw_function_f32", missing_error, [&] {
        (void)getter32.dtw_function_f32();
      });

      Problem setter{"m47_missing_setter"};
      seed_dense_sentinel(setter);
      check_invalid_input("Problem::set_missing_strategy", missing_error, [&] {
        setter.set_missing_strategy(invalid);
      });
      CHECK(setter.missing_strategy == core::MissingStrategy::Error);
      check_dense_sentinel(setter);

      Problem fill{"m47_missing_fill"};
      fill.set_data(basic_f64_data());
      fill.missing_strategy = invalid;
      check_invalid_input("Problem::fill_distance_matrix", missing_error, [&] {
        fill.fill_distance_matrix();
      });
      fill.missing_strategy = core::MissingStrategy::Error;
      check_dense_unallocated(fill);

      ScratchDirectory scratch{"m47_missing_cache"};
      const fs::path cache = scratch.root / "invalid.dtwcache";
      Problem cache_problem{"m47_missing_cache"};
      cache_problem.set_data(basic_f64_data());
      cache_problem.missing_strategy = invalid;
      check_invalid_input("mmap cache identity", missing_error, [&] {
        cache_problem.use_mmap_distance_matrix(cache);
      });
      CHECK_FALSE(fs::exists(cache));
      cache_problem.missing_strategy = core::MissingStrategy::Error;
      check_dense_unallocated(cache_problem);
    });
}

TEST_CASE("M47 rejects every invalid MVMode in both resolver precisions",
          "[m47][enum][mv_mode]")
{
  const std::vector<double> x{0.0, 10.0, 1.0, 11.0};
  const std::vector<double> y{0.0, 10.0, 2.0, 12.0};
  const std::vector<float> xf{0.0f, 10.0f, 1.0f, 11.0f};
  const std::vector<float> yf{0.0f, 10.0f, 2.0f, 12.0f};

  for_each_invalid_enum<core::MVMode, core::MVMode::Independent>(
    [&](core::MVMode invalid) {
      core::DTWVariantParams params;
      params.mv_mode = invalid;

      check_invalid_input("validate_variant_params", mv_mode_error, [&] {
        core::validate_variant_params(params);
      });
      check_invalid_input("distance facade f64", mv_mode_error, [&] {
        (void)distance::dtw<double>(x, y, params);
      });
      check_invalid_input("distance facade f32", mv_mode_error, [&] {
        (void)distance::dtw<float>(xf, yf, params);
      });

      core::DTWOptions options;
      options.variant_params = params;
      check_invalid_input("dtw_runtime", mv_mode_error, [&] {
        (void)core::dtw_runtime(
          x.data(), x.size(), y.data(), y.size(), options);
      });

      Problem resolver64{"m47_mv_resolver64"};
      resolver64.set_data(basic_f64_data(2));
      resolver64.variant_params = params;
      check_invalid_input("resolve_dtw_fn f64", mv_mode_error, [&] {
        (void)core::resolve_dtw_fn<double>(resolver64);
      });

      Problem resolver32{"m47_mv_resolver32"};
      resolver32.set_data(basic_f32_data(2));
      resolver32.variant_params = params;
      check_invalid_input("resolve_dtw_fn f32", mv_mode_error, [&] {
        (void)core::resolve_dtw_fn<float>(resolver32);
      });

      Problem getter64{"m47_mv_getter64"};
      getter64.set_data(basic_f64_data(2));
      getter64.variant_params = params;
      check_invalid_input("Problem::dtw_function", mv_mode_error, [&] {
        (void)getter64.dtw_function();
      });

      Problem getter32{"m47_mv_getter32"};
      getter32.set_data(basic_f32_data(2));
      getter32.variant_params = params;
      check_invalid_input("Problem::dtw_function_f32", mv_mode_error, [&] {
        (void)getter32.dtw_function_f32();
      });

      Problem setter{"m47_mv_setter"};
      seed_dense_sentinel(setter);
      check_invalid_input("Problem::set_variant(params)", mv_mode_error, [&] {
        setter.set_variant(params);
      });
      CHECK(setter.variant_params.mv_mode == core::MVMode::Dependent);
      check_dense_sentinel(setter);

      Problem fill{"m47_mv_fill"};
      fill.set_data(basic_f64_data(2));
      fill.variant_params = params;
      check_invalid_input("Problem::fill_distance_matrix", mv_mode_error, [&] {
        fill.fill_distance_matrix();
      });
      fill.variant_params = {};
      check_dense_unallocated(fill);

      ScratchDirectory scratch{"m47_mv_cache"};
      const fs::path cache = scratch.root / "invalid.dtwcache";
      Problem cache_problem{"m47_mv_cache"};
      cache_problem.set_data(basic_f64_data(2));
      cache_problem.variant_params = params;
      check_invalid_input("mmap cache identity", mv_mode_error, [&] {
        cache_problem.use_mmap_distance_matrix(cache);
      });
      CHECK_FALSE(fs::exists(cache));
      cache_problem.variant_params = {};
      check_dense_unallocated(cache_problem);
    });
}

TEST_CASE("M47 rejects every invalid ConstraintType before runtime dispatch",
          "[m47][enum][constraint]")
{
  const std::vector<double> x{0.0, 0.0, 10.0};
  const std::vector<double> y{0.0, 10.0, 10.0};

  for_each_invalid_enum<core::ConstraintType, core::ConstraintType::SakoeChibaBand>(
    [&](core::ConstraintType invalid) {
      core::DTWOptions options;
      options.constraint = invalid;
      options.band = 0;
      check_invalid_input("dtw_runtime", constraint_error, [&] {
        (void)core::dtw_runtime(
          x.data(), x.size(), y.data(), y.size(), options);
      });
    });
}

TEST_CASE("M47 rejects every invalid distance-matrix and lower-bound strategy",
          "[m47][enum][strategy]")
{
  SECTION("DistanceMatrixStrategy")
  {
    for_each_invalid_enum<DistanceMatrixStrategy, DistanceMatrixStrategy::Metal>(
      [&](DistanceMatrixStrategy invalid) {
        Problem setter{"m47_matrix_strategy_setter"};
        seed_dense_sentinel(setter);
        check_invalid_input("Problem::set_distance_strategy", matrix_strategy_error, [&] {
          setter.set_distance_strategy(invalid);
        });
        CHECK(setter.distance_strategy == DistanceMatrixStrategy::Auto);
        check_dense_sentinel(setter);

        Problem fill{"m47_matrix_strategy_fill"};
        fill.set_data(basic_f64_data());
        fill.distance_strategy = invalid;
        check_invalid_input("Problem::fill_distance_matrix", matrix_strategy_error, [&] {
          fill.fill_distance_matrix();
        });
        fill.distance_strategy = DistanceMatrixStrategy::Auto;
        check_dense_unallocated(fill);

        ScratchDirectory scratch{"m47_matrix_strategy_cache"};
        const fs::path cache = scratch.root / "invalid.dtwcache";
        Problem cache_problem{"m47_matrix_strategy_cache"};
        seed_dense_sentinel(cache_problem);
        cache_problem.distance_strategy = invalid;
        check_invalid_input("mmap cache identity", matrix_strategy_error, [&] {
          cache_problem.use_mmap_distance_matrix(cache);
        });
        CHECK_FALSE(fs::exists(cache));
        cache_problem.distance_strategy = DistanceMatrixStrategy::Auto;
        check_dense_sentinel(cache_problem);
      });
  }

  SECTION("LowerBoundStrategy")
  {
    for_each_invalid_enum<LowerBoundStrategy, LowerBoundStrategy::Webb>(
      [&](LowerBoundStrategy invalid) {
        Problem direct{"m47_lower_bound_direct"};
        direct.set_data(basic_f64_data());
        check_invalid_input("fill_distance_matrix_pruned", lower_bound_error, [&] {
          (void)core::fill_distance_matrix_pruned(direct, 0, invalid);
        });
        check_dense_unallocated(direct);

        Problem fill{"m47_lower_bound_fill"};
        fill.set_data(basic_f64_data());
        fill.set_distance_strategy(DistanceMatrixStrategy::Pruned);
        fill.lb_strategy = invalid;
        check_invalid_input("Problem::fill_distance_matrix", lower_bound_error, [&] {
          fill.fill_distance_matrix();
        });
        fill.lb_strategy = LowerBoundStrategy::Auto;
        check_dense_unallocated(fill);
      });
  }

  SECTION("CUDASettings precision selector")
  {
    for (const int invalid : {-1, 3, INT_MIN, INT_MAX}) {
      CAPTURE(invalid);
      Problem setter{"m47_cuda_precision_setter"};
      seed_dense_sentinel(setter);
      CUDASettings settings = setter.cuda_settings;
      settings.precision = invalid;
      check_invalid_input("Problem::set_cuda_settings",
                          cuda_settings_precision_error, [&] {
        setter.set_cuda_settings(settings);
      });
      CHECK(setter.cuda_settings.precision == 0);
      check_dense_sentinel(setter);

      Problem capability_order{"m47_cuda_precision_capability_order"};
      capability_order.set_data(basic_f64_data());
      capability_order.distance_strategy = DistanceMatrixStrategy::CUDA;
      capability_order.cuda_settings.precision = invalid;
      check_invalid_input("CUDA precision before backend capability",
                          cuda_settings_precision_error, [&] {
        capability_order.fill_distance_matrix();
      });
      capability_order.distance_strategy = DistanceMatrixStrategy::Auto;
      capability_order.cuda_settings.precision = 0;
      check_dense_unallocated(capability_order);

      ScratchDirectory scratch{"m47_cuda_precision_cache"};
      const fs::path cache = scratch.root / "invalid.dtwcache";
      Problem cache_problem{"m47_cuda_precision_cache"};
      seed_dense_sentinel(cache_problem);
      cache_problem.cuda_settings.precision = invalid;
      check_invalid_input("mmap cache identity",
                          cuda_settings_precision_error, [&] {
        cache_problem.use_mmap_distance_matrix(cache);
      });
      CHECK_FALSE(fs::exists(cache));
      cache_problem.cuda_settings.precision = 0;
      check_dense_sentinel(cache_problem);
    }
  }
}

TEST_CASE("M47 rejects every invalid storage policy and active precision",
          "[m47][enum][storage][precision]")
{
  SECTION("StoragePolicy")
  {
    for_each_invalid_enum<core::StoragePolicy, core::StoragePolicy::Mmap>(
      [&](core::StoragePolicy invalid) {
        DataLoader loader;
        check_invalid_input("DataLoader::storage_policy", storage_policy_error, [&] {
          (void)loader.storage_policy(invalid);
        });
      });
  }

  SECTION("Precision")
  {
    for_each_invalid_enum<core::Precision, core::Precision::Float64>(
      [&](core::Precision invalid) {
        Data candidate = basic_f64_data();
        candidate.precision = invalid;
        Problem setter{"m47_precision_set_data"};
        seed_dense_sentinel(setter);
        check_invalid_input("Problem::set_data", precision_error, [&] {
          setter.set_data(std::move(candidate));
        });
        CHECK(setter.data.precision == core::Precision::Float64);
        CHECK(setter.series(1).size() == 3);
        check_dense_sentinel(setter);

        std::vector<double> view_x{0.0, 0.0};
        std::vector<double> view_y{0.0, 1.0, 2.0};
        std::vector<std::string> view_names{"x", "y"};
        Data view_candidate(
          std::vector<std::span<const double>>{
            std::span<const double>{view_x},
            std::span<const double>{view_y}},
          std::vector<std::string_view>{view_names[0], view_names[1]}, 1);
        view_candidate.precision = invalid;
        Problem view_setter{"m47_precision_set_view_data"};
        seed_dense_sentinel(view_setter);
        check_invalid_input("Problem::set_view_data", precision_error, [&] {
          view_setter.set_view_data(std::move(view_candidate));
        });
        CHECK(view_setter.data.precision == core::Precision::Float64);
        CHECK_FALSE(view_setter.data.is_view());
        CHECK(view_setter.series(1).size() == 3);
        check_dense_sentinel(view_setter);

        Problem refresh{"m47_precision_refresh"};
        seed_dense_sentinel(refresh);
        refresh.data.precision = invalid;
        check_invalid_input("Problem::refresh_distance_matrix",
                            precision_error, [&] {
          refresh.refresh_distance_matrix();
        });
        refresh.data.precision = core::Precision::Float64;
        check_dense_sentinel(refresh);

        Problem getter{"m47_precision_getter"};
        getter.set_data(basic_f64_data());
        getter.data.precision = invalid;
        check_invalid_input("Problem::dtw_function", precision_error, [&] {
          (void)getter.dtw_function();
        });
        getter.data.precision = core::Precision::Float64;
        check_dense_unallocated(getter);

        Problem lazy{"m47_precision_lazy"};
        lazy.set_data(basic_f64_data());
        lazy.data.precision = invalid;
        check_invalid_input("Problem::dist_by_ind", precision_error, [&] {
          (void)lazy.dist_by_ind(0, 1);
        });
        lazy.data.precision = core::Precision::Float64;
        check_dense_unallocated(lazy);

        Problem fill{"m47_precision_fill"};
        fill.set_data(basic_f64_data());
        fill.data.precision = invalid;
        check_invalid_input("Problem::fill_distance_matrix", precision_error, [&] {
          fill.fill_distance_matrix();
        });
        fill.data.precision = core::Precision::Float64;
        check_dense_unallocated(fill);

        ScratchDirectory scratch{"m47_precision_cache"};
        const fs::path cache = scratch.root / "invalid.dtwcache";
        Problem cache_problem{"m47_precision_cache"};
        cache_problem.set_data(basic_f64_data());
        cache_problem.data.precision = invalid;
        check_invalid_input("mmap cache identity", precision_error, [&] {
          cache_problem.use_mmap_distance_matrix(cache);
        });
        CHECK_FALSE(fs::exists(cache));
        cache_problem.data.precision = core::Precision::Float64;
        check_dense_unallocated(cache_problem);
      });
  }
}

TEST_CASE("M47 conditionally rejects invalid GPU kernel and precision selectors",
          "[m47][enum][gpu]")
{
#if defined(DTWC_HAS_CUDA)
  SECTION("CUDA selectors")
  {
    const std::vector<std::vector<double>> series{{0.0}, {1.0}};
    for_each_invalid_enum<KernelOverride, KernelOverride::RegTile>(
      [&](KernelOverride invalid) {
        cuda::CUDADistMatOptions options;
        options.kernel_override = invalid;
        check_invalid_input("CUDA KernelOverride", kernel_override_error, [&] {
          (void)cuda::compute_distance_matrix_cuda(series, options);
        });
      });
    for_each_invalid_enum<cuda::CUDAPrecision, cuda::CUDAPrecision::FP64>(
      [&](cuda::CUDAPrecision invalid) {
        cuda::CUDADistMatOptions options;
        options.precision = invalid;
        check_invalid_input("CUDAPrecision", cuda_precision_error, [&] {
          (void)cuda::compute_distance_matrix_cuda(series, options);
        });
      });
  }
#endif

#if defined(DTWC_HAS_METAL)
  SECTION("Metal selectors")
  {
    const std::vector<std::vector<double>> series{{0.0}, {1.0}};
    for_each_invalid_enum<KernelOverride, KernelOverride::RegTile>(
      [&](KernelOverride invalid) {
        metal::MetalDistMatOptions options;
        options.kernel_override = invalid;
        check_invalid_input("Metal KernelOverride", kernel_override_error, [&] {
          (void)metal::compute_distance_matrix_metal(series, options);
        });
      });
    for_each_invalid_enum<metal::MetalPrecision, metal::MetalPrecision::FP64>(
      [&](metal::MetalPrecision invalid) {
        metal::MetalDistMatOptions options;
        options.precision = invalid;
        check_invalid_input("MetalPrecision", metal_precision_error, [&] {
          (void)metal::compute_distance_matrix_metal(series, options);
        });
      });
  }
#endif

#if !defined(DTWC_HAS_CUDA) && !defined(DTWC_HAS_METAL)
  SUCCEED("GPU selector entry points are not compiled in this configuration");
#endif
}

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
         DistanceMatrixStrategy::Pruned,
         DistanceMatrixStrategy::CUDA,
         DistanceMatrixStrategy::Metal}) {
    Problem problem{"m47_valid_matrix_strategy"};
    CHECK_NOTHROW(problem.set_distance_strategy(strategy));
  }
  for (const auto policy : {
         core::StoragePolicy::Auto,
         core::StoragePolicy::Heap,
         core::StoragePolicy::Mmap}) {
    DataLoader loader;
    CHECK_NOTHROW((void)loader.storage_policy(policy));
  }

  for (const auto lower_bound : {
         LowerBoundStrategy::Auto,
         LowerBoundStrategy::None,
         LowerBoundStrategy::Kim,
         LowerBoundStrategy::Keogh,
         LowerBoundStrategy::KimKeogh,
         LowerBoundStrategy::Enhanced,
         LowerBoundStrategy::Webb}) {
    Problem problem{"m47_valid_lower_bound"};
    problem.set_data(Data(
      std::vector<std::vector<double>>{
        {0.0, 0.0, 0.0}, {0.0, 1.0, 2.0}},
      std::vector<std::string>{"x", "y"}));
    CHECK_NOTHROW((void)core::fill_distance_matrix_pruned(
      problem, 1, lower_bound));
  }

  Problem f32_problem{"m47_valid_f32_precision"};
  CHECK_NOTHROW(f32_problem.set_data(basic_f32_data()));
  CHECK(f32_problem.data.precision == core::Precision::Float32);
  Problem f64_problem{"m47_valid_f64_precision"};
  CHECK_NOTHROW(f64_problem.set_data(basic_f64_data()));
  CHECK(f64_problem.data.precision == core::Precision::Float64);

  for (const int precision : {0, 1, 2}) {
    Problem problem{"m47_valid_cuda_settings_precision"};
    CUDASettings settings;
    settings.precision = precision;
    CHECK_NOTHROW(problem.set_cuda_settings(settings));
  }

  constexpr std::array<KernelOverride, 5> kernel_overrides{
    KernelOverride::Auto,
    KernelOverride::Wavefront,
    KernelOverride::WavefrontGlobal,
    KernelOverride::BandedRow,
    KernelOverride::RegTile
  };
  for (std::size_t i = 0; i < kernel_overrides.size(); ++i)
    CHECK(static_cast<int>(kernel_overrides[i]) == static_cast<int>(i));

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
