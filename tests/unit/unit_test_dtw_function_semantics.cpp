/**
 * @file unit_test_dtw_function_semantics.cpp
 * @brief Regression tests for Problem's matrix-free DTW dispatcher accessors.
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

namespace {

Problem make_bound_problem()
{
  std::vector<std::vector<data_t>> series{{0.0, 0.0}, {0.0, 1.0, 2.0}};
  std::vector<std::string> names{"x", "y"};
  Problem problem{"dtw_function_semantics"};
  problem.set_data(Data{std::move(series), std::move(names)});
  problem.set_band(-1);
  return problem;
}

template <typename T>
std::span<const T> as_span(const std::vector<T> &values)
{
  return {values.data(), values.size()};
}

struct ScratchCaches
{
  std::filesystem::path directory;
  std::filesystem::path first;
  std::filesystem::path second;

  explicit ScratchCaches(std::string_view stem)
    : directory(std::filesystem::temp_directory_path()
                / (std::string(stem) + "_"
                   + std::to_string(reinterpret_cast<std::uintptr_t>(this)))),
      first(directory / "first.dtwcache"),
      second(directory / "second.dtwcache")
  {
    std::filesystem::create_directories(directory);
  }

  ~ScratchCaches()
  {
    std::error_code error;
    std::filesystem::remove_all(directory, error);
  }
};

} // namespace

TEST_CASE("mutable DTW function getters refresh raw variant mutations",
          "[problem][dtw_function][semantic_mutation][m37]")
{
  Problem problem = make_bound_problem();
  core::DTWVariantParams params = problem.variant_params;
  params.variant = core::DTWVariant::ADTW;
  params.adtw_penalty = 1.0;
  problem.variant_params = params; // Legacy raw mutation bypasses set_variant().

  // Registered before the repair: Standard=3, ADTW(p=1)=4 for these series.
  SECTION("float64")
  {
    REQUIRE_THAT(
      problem.dtw_function()(problem.series(0), problem.series(1)),
      WithinAbs(4.0, 1e-12));
  }

  SECTION("float32")
  {
    const std::vector<float> x{0.0f, 0.0f};
    const std::vector<float> y{0.0f, 1.0f, 2.0f};
    REQUIRE_THAT(
      problem.dtw_function_f32()(as_span(x), as_span(y)),
      WithinAbs(4.0, 1e-6));
  }
}

TEST_CASE("mutable DTW function getters refresh raw missing-policy mutations",
          "[problem][dtw_function][semantic_mutation][missing][m37]")
{
  Problem problem = make_bound_problem();
  problem.missing_strategy = core::MissingStrategy::ZeroCost;

  SECTION("float64")
  {
    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> x{0.0, nan, 2.0};
    const std::vector<double> y{0.0, 2.0, 2.0};
    REQUIRE_THAT(
      problem.dtw_function()(as_span(x), as_span(y)),
      WithinAbs(0.0, 1e-12));
  }

  SECTION("float32")
  {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const std::vector<float> x{0.0f, nan, 2.0f};
    const std::vector<float> y{0.0f, 2.0f, 2.0f};
    REQUIRE_THAT(
      problem.dtw_function_f32()(as_span(x), as_span(y)),
      WithinAbs(0.0, 1e-6));
  }
}

TEST_CASE("const DTW function getters reject stale raw semantics",
          "[problem][dtw_function][const][semantic_mutation][m37]")
{
  SECTION("variant mutation rejects both precisions")
  {
    Problem problem = make_bound_problem();
    problem.variant_params.variant = core::DTWVariant::ADTW;
    const Problem &view = problem;

    CHECK_THROWS_AS(view.dtw_function(), std::runtime_error);
    CHECK_THROWS_AS(view.dtw_function_f32(), std::runtime_error);
  }

  SECTION("missing-policy mutation rejects both precisions")
  {
    Problem problem = make_bound_problem();
    problem.missing_strategy = core::MissingStrategy::ZeroCost;
    const Problem &view = problem;

    CHECK_THROWS_AS(view.dtw_function(), std::runtime_error);
    CHECK_THROWS_AS(view.dtw_function_f32(), std::runtime_error);
  }
}

TEST_CASE("unchanged DTW function getters keep stable bound callable storage",
          "[problem][dtw_function][allocation_free][m37]")
{
  Problem problem = make_bound_problem();

  const auto *f64 = &problem.dtw_function();
  const auto *f32 = &problem.dtw_function_f32();
  for (int repeat = 0; repeat < 1024; ++repeat) {
    REQUIRE(&problem.dtw_function() == f64);
    REQUIRE(&problem.dtw_function_f32() == f32);
  }

  const Problem &view = problem;
  REQUIRE(&view.dtw_function() == f64);
  REQUIRE(&view.dtw_function_f32() == f32);
}

TEST_CASE("DTW function semantic guards preserve mapped-cache invariants",
          "[problem][dtw_function][semantic_mutation][mmap][m37]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in (DTWC_ENABLE_LLFIO=OFF)");
#else
  SECTION("unchanged mutable and const getters retain mapped storage")
  {
    ScratchCaches cache{"m37_unchanged_mmap"};
    Problem problem = make_bound_problem();
    problem.use_mmap_distance_matrix(cache.first);

    const auto *function = &problem.dtw_function();
    REQUIRE(std::holds_alternative<core::MmapDistanceMatrix>(
      problem.distance_matrix()));

    const Problem &view = problem;
    REQUIRE(&view.dtw_function() == function);
    REQUIRE(std::holds_alternative<core::MmapDistanceMatrix>(
      view.distance_matrix()));
  }

  SECTION("mutable stale getter detaches mmap and rebinds without rewriting it")
  {
    ScratchCaches cache{"m37_mutable_mmap"};
    Problem problem = make_bound_problem();
    problem.use_mmap_distance_matrix(cache.first);
    problem.variant_params.variant = core::DTWVariant::ADTW;

    REQUIRE_THAT(
      problem.dtw_function()(problem.series(0), problem.series(1)),
      WithinAbs(4.0, 1e-12));
    REQUIRE(std::holds_alternative<core::DenseDistanceMatrix>(
      problem.distance_matrix()));

    Problem original_semantics = make_bound_problem();
    REQUIRE_NOTHROW(original_semantics.use_mmap_distance_matrix(cache.first));
  }

  SECTION("const stale getter rejects without detaching mmap")
  {
    ScratchCaches cache{"m37_const_mmap"};
    Problem problem = make_bound_problem();
    problem.use_mmap_distance_matrix(cache.first);
    problem.variant_params.variant = core::DTWVariant::ADTW;

    const Problem &stale_view = problem;
    REQUIRE_THROWS_AS(stale_view.dtw_function(), std::runtime_error);

    // Restoring the raw value makes the original mapping observable again;
    // the rejecting const accessor must not have detached or rewritten it.
    problem.variant_params.variant = core::DTWVariant::Standard;
    const Problem &restored_view = problem;
    REQUIRE(std::holds_alternative<core::MmapDistanceMatrix>(
      restored_view.distance_matrix()));
  }

  SECTION("mmap replacement reconciles dispatcher before publishing new identity")
  {
    ScratchCaches cache{"m37_replace_mmap"};
    Problem problem = make_bound_problem();
    problem.use_mmap_distance_matrix(cache.first);
    problem.variant_params.variant = core::DTWVariant::ADTW;

    problem.use_mmap_distance_matrix(cache.second);
    REQUIRE(std::holds_alternative<core::MmapDistanceMatrix>(
      problem.distance_matrix()));
    REQUIRE_THAT(problem.dist_by_ind(0, 1), WithinAbs(4.0, 1e-12));
  }
#endif
}
