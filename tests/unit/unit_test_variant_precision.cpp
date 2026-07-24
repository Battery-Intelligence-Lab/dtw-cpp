/**
 * @file unit_test_variant_precision.cpp
 * @brief Float32 representability and transactional Problem-boundary tests (M45).
 */

#include <dtwc.hpp>
#include <core/dtw_dispatch.hpp>
#include <error.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
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

bool params_equal(const core::DTWVariantParams &a,
                  const core::DTWVariantParams &b)
{
  return a.variant == b.variant
      && a.wdtw_g == b.wdtw_g
      && a.adtw_penalty == b.adtw_penalty
      && a.sdtw_gamma == b.sdtw_gamma
      && a.msm_c == b.msm_c
      && a.twe_nu == b.twe_nu
      && a.twe_lambda == b.twe_lambda
      && a.mv_mode == b.mv_mode;
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
  const auto &matrix = std::get<core::DenseDistanceMatrix>(
    problem.distance_matrix());
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

struct NarrowingCase
{
  core::DTWVariant variant;
  void (*poison)(core::DTWVariantParams &);
  std::string_view message;
};

constexpr std::array<NarrowingCase, 12> narrowing_cases{{
  {core::DTWVariant::WDTW,
   +[](core::DTWVariantParams &p) { p.wdtw_g = std::numeric_limits<double>::min(); },
   "WDTW g cannot be represented in float32 without becoming zero or non-finite."},
  {core::DTWVariant::WDTW,
   +[](core::DTWVariantParams &p) { p.wdtw_g = std::numeric_limits<double>::max(); },
   "WDTW g cannot be represented in float32 without becoming zero or non-finite."},
  {core::DTWVariant::ADTW,
   +[](core::DTWVariantParams &p) { p.adtw_penalty = std::numeric_limits<double>::min(); },
   "ADTW penalty cannot be represented in float32 without becoming zero or non-finite."},
  {core::DTWVariant::ADTW,
   +[](core::DTWVariantParams &p) { p.adtw_penalty = std::numeric_limits<double>::max(); },
   "ADTW penalty cannot be represented in float32 without becoming zero or non-finite."},
  {core::DTWVariant::SoftDTW,
   +[](core::DTWVariantParams &p) { p.sdtw_gamma = std::numeric_limits<double>::min(); },
   "Soft-DTW gamma cannot be represented in float32 without becoming zero or non-finite."},
  {core::DTWVariant::SoftDTW,
   +[](core::DTWVariantParams &p) { p.sdtw_gamma = std::numeric_limits<double>::max(); },
   "Soft-DTW gamma cannot be represented in float32 without becoming zero or non-finite."},
  {core::DTWVariant::MSM,
   +[](core::DTWVariantParams &p) { p.msm_c = std::numeric_limits<double>::min(); },
   "MSM c cannot be represented in float32 without becoming zero or non-finite."},
  {core::DTWVariant::MSM,
   +[](core::DTWVariantParams &p) { p.msm_c = std::numeric_limits<double>::max(); },
   "MSM c cannot be represented in float32 without becoming zero or non-finite."},
  {core::DTWVariant::TWE,
   +[](core::DTWVariantParams &p) { p.twe_nu = std::numeric_limits<double>::min(); },
   "TWE nu cannot be represented in float32 without becoming zero or non-finite."},
  {core::DTWVariant::TWE,
   +[](core::DTWVariantParams &p) { p.twe_nu = std::numeric_limits<double>::max(); },
   "TWE nu cannot be represented in float32 without becoming zero or non-finite."},
  {core::DTWVariant::TWE,
   +[](core::DTWVariantParams &p) { p.twe_lambda = std::numeric_limits<double>::min(); },
   "TWE lambda cannot be represented in float32 without becoming zero or non-finite."},
  {core::DTWVariant::TWE,
   +[](core::DTWVariantParams &p) { p.twe_lambda = std::numeric_limits<double>::max(); },
   "TWE lambda cannot be represented in float32 without becoming zero or non-finite."},
}};

struct ScratchCache
{
  std::filesystem::path directory;
  std::filesystem::path path;

  explicit ScratchCache(std::string_view stem)
    : directory(std::filesystem::temp_directory_path()
                / (std::string(stem) + "_"
                   + std::to_string(reinterpret_cast<std::uintptr_t>(this)))),
      path(directory / "distances.dtwcache")
  {
    std::filesystem::create_directories(directory);
  }

  ~ScratchCache()
  {
    std::error_code error;
    std::filesystem::remove_all(directory, error);
  }
};

struct RawOperation
{
  std::string_view name;
  void (*run)(Problem &);
};

constexpr std::array<RawOperation, 5> raw_operations{{
  {"float64 dispatcher", +[](Problem &p) { (void)p.dtw_function(); }},
  {"float32 dispatcher", +[](Problem &p) { (void)p.dtw_function_f32(); }},
  {"dense fill", +[](Problem &p) { p.fill_distance_matrix(); }},
  {"lazy distance", +[](Problem &p) { (void)p.dist_by_ind(0, 1); }},
  {"public refresh", +[](Problem &p) { p.refresh_distance_matrix(); }},
}};

std::string read_problem_source()
{
  const auto repo_root = std::filesystem::path{DTWC_TEST_DATA_DIR}.parent_path();
  std::ifstream source(repo_root / "dtwc" / "Problem.cpp", std::ios::binary);
  REQUIRE(source.is_open());
  return {std::istreambuf_iterator<char>{source},
          std::istreambuf_iterator<char>{}};
}

} // namespace

TEST_CASE("float32 set_variant rejects narrowing transactionally",
          "[problem][variant][f32][transaction][m45]")
{
  for (const auto &test : narrowing_cases) {
    CAPTURE(static_cast<int>(test.variant), test.message);
    Problem problem{"m45_set_variant"};
    problem.set_data(make_f32_data());
    problem.fill_distance_matrix();
    problem.clusters_ind = {1, 0};
    problem.centroids_ind = {1};

    const auto original_params = problem.variant_params;
    const auto original_cache = snapshot_dense(problem);
    auto candidate = original_params;
    candidate.variant = test.variant;
    test.poison(candidate);

    const bool caught = catches_exact(
      [&] { problem.set_variant(candidate); }, test.message);
    CHECK(caught);
    CHECK(params_equal(problem.variant_params, original_params));
    CHECK(problem.data().is_f32());
    CHECK(problem.data().series_f32(1)[1] == 2.0f);
    CHECK(problem.labels() == std::vector<int>{1, 0});
    CHECK(problem.medoids() == std::vector<int>{1});
    if (caught) require_dense_unchanged(problem, original_cache);
  }

  SECTION("selector-only overload validates the candidate before mutation")
  {
    Problem problem{"m45_selector"};
    problem.set_data(make_f32_data());
    problem.fill_distance_matrix();
    const auto original_cache = snapshot_dense(problem);
    problem.variant_params.sdtw_gamma = std::numeric_limits<double>::min();

    const bool caught = catches_exact(
      [&] { problem.set_variant(core::DTWVariant::SoftDTW); },
      "Soft-DTW gamma cannot be represented in float32 without becoming zero or non-finite.");
    CHECK(caught);
    CHECK(problem.variant_params.variant == core::DTWVariant::Standard);
    problem.variant_params.sdtw_gamma = 1.0;
    if (caught) require_dense_unchanged(problem, original_cache);
  }
}

TEST_CASE("float32 heap and view data replacement validate before mutation",
          "[problem][variant][f32][data][transaction][m45]")
{
  auto configured_f64 = [] {
    Problem problem{"m45_data_transaction"};
    problem.set_data(make_f64_data());
    auto params = problem.variant_params;
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
    CHECK(problem.labels() == std::vector<int>{0, 1});
    CHECK(problem.medoids() == std::vector<int>{0, 1});
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
    CHECK(problem.labels() == std::vector<int>{0, 1});
    CHECK(problem.medoids() == std::vector<int>{0, 1});
    if (caught) require_dense_unchanged(problem, original_cache);
  }
}

TEST_CASE("raw float32 reconciliation fails before cache mutation",
          "[problem][variant][f32][raw][cache][m45]")
{
  for (const auto &operation : raw_operations) {
    CAPTURE(operation.name);
    Problem problem{"m45_raw_reconcile"};
    problem.set_data(make_f32_data());
    problem.fill_distance_matrix();
    const auto original_params = problem.variant_params;
    const auto original_cache = snapshot_dense(problem);

    problem.variant_params.variant = core::DTWVariant::SoftDTW;
    problem.variant_params.sdtw_gamma = std::numeric_limits<double>::min();
    const bool caught = catches_exact(
      [&] { operation.run(problem); },
      "Soft-DTW gamma cannot be represented in float32 without becoming zero or non-finite.");
    CHECK(caught);

    // The caller-owned raw edit remains, but failed reconciliation must not
    // clear the cache/callable that preceded it. Restore it solely so the
    // semantic snapshot permits direct inspection.
    problem.variant_params = original_params;
    if (caught) {
      require_dense_unchanged(problem, original_cache);
      CHECK(problem.is_distance_matrix_filled());
    }
  }
}

TEST_CASE("float32 narrowing preflight preserves mmap and bind transactions",
          "[problem][variant][f32][raw][mmap][transaction][m45]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in (DTWC_ENABLE_LLFIO=OFF)");
#else
  SECTION("binding a new mmap does not clear dense state or create a file")
  {
    ScratchCache cache{"dtwc_m45_bind_preflight"};
    Problem problem{"m45_mmap_bind"};
    problem.set_data(make_f32_data());
    problem.fill_distance_matrix();
    const auto original_params = problem.variant_params;
    const auto original_cache = snapshot_dense(problem);

    problem.variant_params.variant = core::DTWVariant::SoftDTW;
    problem.variant_params.sdtw_gamma = std::numeric_limits<double>::min();
    const bool caught = catches_exact(
      [&] { problem.use_mmap_distance_matrix(cache.path); },
      "Soft-DTW gamma cannot be represented in float32 without becoming zero or non-finite.");
    CHECK(caught);
    CHECK_FALSE(std::filesystem::exists(cache.path));

    problem.variant_params = original_params;
    if (caught) require_dense_unchanged(problem, original_cache);
  }

  SECTION("public refresh does not detach an existing mmap")
  {
    ScratchCache cache{"dtwc_m45_refresh_preflight"};
    Problem problem{"m45_mmap_refresh"};
    problem.set_data(make_f32_data());
    problem.use_mmap_distance_matrix(cache.path);
    problem.fill_distance_matrix();
    const auto original_params = problem.variant_params;

    problem.variant_params.variant = core::DTWVariant::SoftDTW;
    problem.variant_params.sdtw_gamma = std::numeric_limits<double>::min();
    const bool caught = catches_exact(
      [&] { problem.refresh_distance_matrix(); },
      "Soft-DTW gamma cannot be represented in float32 without becoming zero or non-finite.");
    CHECK(caught);

    problem.variant_params = original_params;
    CHECK(std::holds_alternative<core::MmapDistanceMatrix>(
      std::as_const(problem).distance_matrix()));
    if (caught) CHECK(problem.is_distance_matrix_filled());
  }
#endif
}

TEST_CASE("f64 accepts its full domain and explicit f32 access stays transactional",
          "[problem][variant][f64][f32_getter][m45]")
{
  for (const auto &test : narrowing_cases) {
    CAPTURE(static_cast<int>(test.variant), test.message);
    Problem problem{"m45_f64_acceptance"};
    problem.set_data(make_f64_data());
    auto params = problem.variant_params;
    params.variant = test.variant;
    test.poison(params);
    REQUIRE_NOTHROW(problem.set_variant(params));
    REQUIRE_NOTHROW((void)problem.dtw_function());
    problem.fill_distance_matrix();
    const auto original_cache = snapshot_dense(problem);

    const bool caught = catches_exact(
      [&] { (void)problem.dtw_function_f32(); }, test.message);
    CHECK(caught);
    const Problem &const_problem = problem;
    CHECK(catches_exact(
      [&] { (void)const_problem.dtw_function_f32(); }, test.message));
    CHECK(catches_exact(
      [&] { (void)core::resolve_dtw_fn<float>(problem); }, test.message));
    CHECK(params_equal(problem.variant_params, params));
    CHECK_FALSE(problem.data().is_f32());
    if (caught) require_dense_unchanged(problem, original_cache);
  }
}

TEST_CASE("Problem float32 callable uses stay behind validated access",
          "[problem][variant][f32][source_guard][m45]")
{
  const std::string source = read_problem_source();

  CHECK(source.find("dtw_fn_f32_(") == std::string::npos);
  CHECK(source.find("validated_dtw_function_f32()") != std::string::npos);
}

TEST_CASE("float32 representable boundaries and inactive parameters remain valid",
          "[problem][variant][f32][boundary][m45]")
{
  const double smallest_f32 = static_cast<double>(std::numeric_limits<float>::denorm_min());

  SECTION("inactive unrepresentable parameter does not restrict Standard")
  {
    Problem problem{"m45_inactive"};
    problem.set_data(make_f32_data());
    auto params = problem.variant_params;
    params.sdtw_gamma = std::numeric_limits<double>::min();
    REQUIRE_NOTHROW(problem.set_variant(params));
    REQUIRE_NOTHROW((void)problem.dtw_function_f32());
    REQUIRE(problem.variant_params.variant == core::DTWVariant::Standard);
  }

  SECTION("exact zero limits")
  {
    for (const auto variant : {core::DTWVariant::WDTW, core::DTWVariant::ADTW}) {
      Problem problem{"m45_zero"};
      problem.set_data(make_f32_data());
      auto params = problem.variant_params;
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
      auto params = problem.variant_params;
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
