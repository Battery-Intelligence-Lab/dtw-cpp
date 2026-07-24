/**
 * @file unit_test_problem_semantic_transactions.cpp
 * @brief Phase 8 M48: semantic setters must provide a strong exception guarantee.
 *
 * Registered before production changes. A rejected variant/missing
 * cross-product must preserve the prior selectors, both active-precision
 * dispatchers, and every bit of an already-published dense or mapped cache.
 */

#include <dtwc.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

namespace {

namespace fs = std::filesystem;

template <typename T>
dtwc::Data two_series()
{
  return dtwc::Data(
    std::vector<std::vector<T>>{{T(0)}, {T(1)}},
    std::vector<std::string>{"x", "y"});
}

template <typename T>
void set_two_series(dtwc::Problem &problem)
{
  problem.set_data(two_series<T>());
  REQUIRE(problem.data().is_f32() == std::is_same_v<T, float>);
}

void set_two_series_mv(dtwc::Problem &problem)
{
  problem.set_data(dtwc::Data(
    std::vector<std::vector<dtwc::data_t>>{{0.0, 0.0}, {1.0, 1.0}},
    std::vector<std::string>{"x", "y"}, 2));
  REQUIRE(problem.data().ndim == 2);
}

dtwc::Data owning_mv_data(double offset = 10.0)
{
  return dtwc::Data(
    std::vector<std::vector<dtwc::data_t>>{
      {offset, offset + 1.0}, {offset + 2.0, offset + 3.0}},
    std::vector<std::string>{"mx", "my"}, 2);
}

void check_original_univariate_data(const dtwc::Problem &problem)
{
  CHECK_FALSE(problem.data().is_view());
  CHECK_FALSE(problem.data().is_f32());
  CHECK(problem.data().ndim == 1);
  CHECK(problem.data().size() == 2);
  CHECK(problem.series(0)[0] == 0.0);
  CHECK(problem.series(1)[0] == 1.0);
}

void inject_complete_dense_cache(dtwc::Problem &problem, double sentinel)
{
  auto &matrix = problem.dense_distance_matrix();
  matrix.resize(2);
  matrix.set(0, 0, 0.0);
  matrix.set(0, 1, sentinel);
  matrix.set(1, 1, 0.0);
}

void require_complete_dense_cache(dtwc::Problem &problem, double sentinel)
{
  REQUIRE(problem.is_distance_matrix_filled());
  REQUIRE(problem.dist_by_ind(0, 1) == sentinel);
}

void check_complete_dense_cache(dtwc::Problem &problem, double sentinel)
{
  CHECK(problem.is_distance_matrix_filled());
  double after = std::numeric_limits<double>::quiet_NaN();
  CHECK_NOTHROW(after = problem.dist_by_ind(0, 1));
  CHECK(after == sentinel);
}

void check_candidate_rejection_cache(dtwc::Problem &problem, double sentinel)
{
  // A post-move capability failure leaves current state internally invalid;
  // do not trigger a second reconciliation merely to inspect the red. Once
  // prevalidation is fixed, the still-visible cache must retain its exact bit.
  const bool filled = problem.is_distance_matrix_filled();
  CHECK(filled);
  if (filled) CHECK(problem.dist_by_ind(0, 1) == sentinel);
}

constexpr const char *cross_product_error =
  "Non-Standard DTW variants require MissingStrategy::Error.";

#ifdef DTWC_HAS_MMAP
struct ScratchCache
{
  fs::path directory;
  fs::path path;

  explicit ScratchCache(std::string_view stem)
    : directory(fs::temp_directory_path()
                / (std::string(stem) + "_"
                   + std::to_string(reinterpret_cast<std::uintptr_t>(this)))),
      path(directory / "distances.dtwcache")
  {
    std::error_code ec;
    fs::remove_all(directory, ec);
    fs::create_directories(directory);
  }

  ~ScratchCache()
  {
    std::error_code ec;
    fs::remove_all(directory, ec);
  }
};

void inject_complete_mmap_cache(dtwc::Problem &problem, double sentinel)
{
  auto &storage = problem.distance_matrix();
  auto &matrix = std::get<dtwc::core::MmapDistanceMatrix>(storage);
  matrix.set(0, 0, 0.0);
  matrix.set(0, 1, sentinel);
  matrix.set(1, 1, 0.0);
  matrix.sync();
}

bool mapped_cache_matches(const dtwc::Problem &problem, double sentinel)
{
  const auto &storage = problem.distance_matrix();
  if (!std::holds_alternative<dtwc::core::MmapDistanceMatrix>(storage))
    return false;
  const auto &matrix = std::get<dtwc::core::MmapDistanceMatrix>(storage);
  return matrix.size() == 2 && matrix.all_computed()
      && matrix.get(0, 0) == 0.0
      && matrix.get(0, 1) == sentinel
      && matrix.get(1, 1) == 0.0;
}
#endif

} // namespace

TEMPLATE_TEST_CASE(
  "M48 rejected semantic mutations preserve dense state and cache",
  "[m48][problem][transaction][dense]", double, float)
{
  SECTION("missing-strategy setter from ADTW/Error")
  {
    dtwc::Problem problem("m48-missing");
    set_two_series<TestType>(problem);
    problem.set_variant(dtwc::core::DTWVariant::ADTW);
    inject_complete_dense_cache(problem, 123.0);
    require_complete_dense_cache(problem, 123.0);

    CHECK_THROWS_WITH(
      problem.set_missing_strategy(dtwc::core::MissingStrategy::ZeroCost),
      Catch::Matchers::Equals(cross_product_error));

    CHECK(problem.variant_params.variant == dtwc::core::DTWVariant::ADTW);
    CHECK(problem.missing_strategy == dtwc::core::MissingStrategy::Error);
    check_complete_dense_cache(problem, 123.0);
  }

  SECTION("enum variant setter from Standard/ZeroCost")
  {
    dtwc::Problem problem("m48-variant-enum");
    set_two_series<TestType>(problem);
    problem.set_missing_strategy(dtwc::core::MissingStrategy::ZeroCost);
    inject_complete_dense_cache(problem, 456.0);
    require_complete_dense_cache(problem, 456.0);

    CHECK_THROWS_WITH(
      problem.set_variant(dtwc::core::DTWVariant::ADTW),
      Catch::Matchers::Equals(cross_product_error));

    CHECK(problem.variant_params.variant == dtwc::core::DTWVariant::Standard);
    CHECK(problem.missing_strategy == dtwc::core::MissingStrategy::ZeroCost);
    check_complete_dense_cache(problem, 456.0);
  }

  SECTION("aggregate variant setter from Standard/ZeroCost")
  {
    dtwc::Problem problem("m48-variant-params");
    set_two_series<TestType>(problem);
    problem.set_missing_strategy(dtwc::core::MissingStrategy::ZeroCost);
    inject_complete_dense_cache(problem, 789.0);
    require_complete_dense_cache(problem, 789.0);

    auto candidate = problem.variant_params;
    candidate.variant = dtwc::core::DTWVariant::ADTW;
    candidate.adtw_penalty = 7.5;
    CHECK_THROWS_WITH(
      problem.set_variant(candidate),
      Catch::Matchers::Equals(cross_product_error));

    CHECK(problem.variant_params.variant == dtwc::core::DTWVariant::Standard);
    CHECK(problem.variant_params.adtw_penalty == 1.0);
    CHECK(problem.missing_strategy == dtwc::core::MissingStrategy::ZeroCost);
    check_complete_dense_cache(problem, 789.0);
  }
}

TEMPLATE_TEST_CASE(
  "M48 accepted no-op and valid semantic setters retain existing behavior",
  "[m48][problem][transaction][control]", double, float)
{
  SECTION("no-op missing and both variant overloads preserve cache")
  {
    dtwc::Problem problem("m48-noop");
    set_two_series<TestType>(problem);
    problem.set_variant(dtwc::core::DTWVariant::ADTW);
    inject_complete_dense_cache(problem, 321.0);
    require_complete_dense_cache(problem, 321.0);

    CHECK_NOTHROW(
      problem.set_missing_strategy(dtwc::core::MissingStrategy::Error));
    check_complete_dense_cache(problem, 321.0);
    CHECK_NOTHROW(problem.set_variant(dtwc::core::DTWVariant::ADTW));
    check_complete_dense_cache(problem, 321.0);
    const auto same = problem.variant_params;
    CHECK_NOTHROW(problem.set_variant(same));
    check_complete_dense_cache(problem, 321.0);
  }

  SECTION("valid missing-strategy change publishes and invalidates")
  {
    dtwc::Problem problem("m48-valid-missing");
    set_two_series<TestType>(problem);
    inject_complete_dense_cache(problem, 654.0);
    require_complete_dense_cache(problem, 654.0);

    CHECK_NOTHROW(
      problem.set_missing_strategy(dtwc::core::MissingStrategy::ZeroCost));
    CHECK(problem.missing_strategy == dtwc::core::MissingStrategy::ZeroCost);
    CHECK_FALSE(problem.is_distance_matrix_filled());
  }

  SECTION("valid enum variant change publishes and invalidates")
  {
    dtwc::Problem problem("m48-valid-enum");
    set_two_series<TestType>(problem);
    inject_complete_dense_cache(problem, 987.0);
    require_complete_dense_cache(problem, 987.0);

    CHECK_NOTHROW(problem.set_variant(dtwc::core::DTWVariant::ADTW));
    CHECK(problem.variant_params.variant == dtwc::core::DTWVariant::ADTW);
    CHECK_FALSE(problem.is_distance_matrix_filled());
  }

  SECTION("valid aggregate variant change publishes and invalidates")
  {
    dtwc::Problem problem("m48-valid-params");
    set_two_series<TestType>(problem);
    inject_complete_dense_cache(problem, 246.0);
    require_complete_dense_cache(problem, 246.0);

    auto candidate = problem.variant_params;
    candidate.variant = dtwc::core::DTWVariant::WDTW;
    candidate.wdtw_g = 0.2;
    CHECK_NOTHROW(problem.set_variant(candidate));
    CHECK(problem.variant_params.variant == dtwc::core::DTWVariant::WDTW);
    CHECK(problem.variant_params.wdtw_g == 0.2);
    CHECK_FALSE(problem.is_distance_matrix_filled());
  }
}

TEST_CASE("M48 rejected capability mutations preserve multivariate state and cache",
          "[m48][problem][transaction][capability]")
{
  struct CapabilityCase
  {
    const char *name;
    dtwc::core::DTWVariantParams candidate;
    bool enum_overload;
    const char *message;
    double sentinel;
  };

  auto msm = dtwc::core::DTWVariantParams{};
  msm.variant = dtwc::core::DTWVariant::MSM;
  auto twe = dtwc::core::DTWVariantParams{};
  twe.variant = dtwc::core::DTWVariant::TWE;
  twe.twe_lambda = 0.8;
  auto independent_adtw = dtwc::core::DTWVariantParams{};
  independent_adtw.variant = dtwc::core::DTWVariant::ADTW;
  independent_adtw.adtw_penalty = 2.0;
  independent_adtw.mv_mode = dtwc::core::MVMode::Independent;

  const std::vector<CapabilityCase> cases{
    {"MSM enum", msm, true,
     "MSM distance is univariate in this release (ndim must be 1)", 111.0},
    {"TWE aggregate", twe, false,
     "TWE distance is univariate in this release (ndim must be 1)", 222.0},
    {"Independent ADTW aggregate", independent_adtw, false,
     "Independent multivariate mode is implemented for the Standard DTW variant only in this release",
     333.0},
  };

  for (const auto &test : cases) {
    DYNAMIC_SECTION(test.name)
    {
      dtwc::Problem problem("m48-capability");
      set_two_series_mv(problem);
      inject_complete_dense_cache(problem, test.sentinel);
      require_complete_dense_cache(problem, test.sentinel);

      if (test.enum_overload) {
        CHECK_THROWS_WITH(
          problem.set_variant(test.candidate.variant),
          Catch::Matchers::Equals(test.message));
      } else {
        CHECK_THROWS_WITH(
          problem.set_variant(test.candidate),
          Catch::Matchers::Equals(test.message));
      }

      CHECK(problem.variant_params.variant == dtwc::core::DTWVariant::Standard);
      CHECK(problem.variant_params.mv_mode == dtwc::core::MVMode::Dependent);
      CHECK(problem.variant_params.twe_lambda == 1.0);
      CHECK(problem.variant_params.adtw_penalty == 1.0);
      CHECK(problem.missing_strategy == dtwc::core::MissingStrategy::Error);
      check_complete_dense_cache(problem, test.sentinel);
    }
  }
}

TEST_CASE("M48 data candidates validate dimensional capabilities before publication",
          "[m48][problem][transaction][data]")
{
  SECTION("owning set_data cannot move an MSM Problem into multivariate state")
  {
    dtwc::Problem problem("m48-data-msm");
    set_two_series<double>(problem);
    problem.set_variant(dtwc::core::DTWVariant::MSM);
    inject_complete_dense_cache(problem, 444.0);
    require_complete_dense_cache(problem, 444.0);

    auto candidate = owning_mv_data();
    CHECK_THROWS_WITH(
      problem.set_data(std::move(candidate)),
      Catch::Matchers::Equals(
        "MSM distance is univariate in this release (ndim must be 1)"));
    CHECK(problem.variant_params.variant == dtwc::core::DTWVariant::MSM);
    check_original_univariate_data(problem);
    check_candidate_rejection_cache(problem, 444.0);
  }

  SECTION("set_view_data cannot move a TWE Problem into multivariate state")
  {
    dtwc::Problem problem("m48-view-twe");
    set_two_series<double>(problem);
    problem.set_variant(dtwc::core::DTWVariant::TWE);
    inject_complete_dense_cache(problem, 555.0);
    require_complete_dense_cache(problem, 555.0);

    std::vector<double> x{10.0, 11.0};
    std::vector<double> y{12.0, 13.0};
    std::vector<std::span<const double>> spans{
      std::span<const double>{x}, std::span<const double>{y}};
    std::vector<std::string_view> names{"vx", "vy"};
    dtwc::Data candidate(std::move(spans), std::move(names), 2);
    CHECK_THROWS_WITH(
      problem.set_view_data(std::move(candidate)),
      Catch::Matchers::Equals(
        "TWE distance is univariate in this release (ndim must be 1)"));
    CHECK(problem.variant_params.variant == dtwc::core::DTWVariant::TWE);
    check_original_univariate_data(problem);
    check_candidate_rejection_cache(problem, 555.0);
  }
}

TEST_CASE("M48 valid multivariate data replacements retain existing behavior",
          "[m48][problem][transaction][data][control]")
{
  SECTION("owning set_data publishes Standard multivariate data")
  {
    dtwc::Problem problem("m48-data-valid");
    set_two_series<double>(problem);
    inject_complete_dense_cache(problem, 666.0);
    CHECK_NOTHROW(problem.set_data(owning_mv_data(20.0)));
    CHECK(problem.data().ndim == 2);
    CHECK_FALSE(problem.data().is_view());
    CHECK(problem.series(0)[0] == 20.0);
    CHECK_FALSE(problem.is_distance_matrix_filled());
  }

  SECTION("set_view_data publishes Standard multivariate view data")
  {
    dtwc::Problem problem("m48-view-valid");
    set_two_series<double>(problem);
    inject_complete_dense_cache(problem, 777.0);
    std::vector<double> x{30.0, 31.0};
    std::vector<double> y{32.0, 33.0};
    std::vector<std::span<const double>> spans{
      std::span<const double>{x}, std::span<const double>{y}};
    std::vector<std::string_view> names{"vx", "vy"};
    dtwc::Data candidate(std::move(spans), std::move(names), 2);
    CHECK_NOTHROW(problem.set_view_data(std::move(candidate)));
    CHECK(problem.data().ndim == 2);
    CHECK(problem.data().is_view());
    CHECK(problem.series(1)[0] == 32.0);
    CHECK_FALSE(problem.is_distance_matrix_filled());
  }
}

TEMPLATE_TEST_CASE(
  "M48 rejected semantic mutations preserve mapped state and ready cache",
  "[m48][problem][transaction][mmap]", double, float)
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in (DTWC_ENABLE_LLFIO=OFF)");
#else
  SECTION("missing-strategy setter from ADTW/Error")
  {
    ScratchCache cache{"m48-missing-mmap"};
    {
      dtwc::Problem problem("m48-missing-mmap");
      set_two_series<TestType>(problem);
      problem.set_variant(dtwc::core::DTWVariant::ADTW);
      problem.use_mmap_distance_matrix(cache.path);
      inject_complete_mmap_cache(problem, 135.0);
      REQUIRE(mapped_cache_matches(problem, 135.0));

      CHECK_THROWS_WITH(
        problem.set_missing_strategy(dtwc::core::MissingStrategy::ZeroCost),
        Catch::Matchers::Equals(cross_product_error));
      CHECK(problem.variant_params.variant == dtwc::core::DTWVariant::ADTW);
      CHECK(problem.missing_strategy == dtwc::core::MissingStrategy::Error);
      CHECK(problem.is_distance_matrix_filled());
      bool preserved = false;
      CHECK_NOTHROW(preserved = mapped_cache_matches(problem, 135.0));
      CHECK(preserved);
    }

    dtwc::Problem reopened("m48-missing-mmap-reopen");
    set_two_series<TestType>(reopened);
    reopened.set_variant(dtwc::core::DTWVariant::ADTW);
    CHECK_NOTHROW(reopened.use_mmap_distance_matrix(cache.path));
    CHECK(mapped_cache_matches(reopened, 135.0));
  }

  SECTION("enum variant setter from Standard/ZeroCost")
  {
    ScratchCache cache{"m48-variant-mmap"};
    {
      dtwc::Problem problem("m48-variant-mmap");
      set_two_series<TestType>(problem);
      problem.set_missing_strategy(dtwc::core::MissingStrategy::ZeroCost);
      problem.use_mmap_distance_matrix(cache.path);
      inject_complete_mmap_cache(problem, 864.0);
      REQUIRE(mapped_cache_matches(problem, 864.0));

      CHECK_THROWS_WITH(
        problem.set_variant(dtwc::core::DTWVariant::ADTW),
        Catch::Matchers::Equals(cross_product_error));
      CHECK(problem.variant_params.variant == dtwc::core::DTWVariant::Standard);
      CHECK(problem.missing_strategy == dtwc::core::MissingStrategy::ZeroCost);
      CHECK(problem.is_distance_matrix_filled());
      bool preserved = false;
      CHECK_NOTHROW(preserved = mapped_cache_matches(problem, 864.0));
      CHECK(preserved);
    }

    dtwc::Problem reopened("m48-variant-mmap-reopen");
    set_two_series<TestType>(reopened);
    reopened.set_missing_strategy(dtwc::core::MissingStrategy::ZeroCost);
    CHECK_NOTHROW(reopened.use_mmap_distance_matrix(cache.path));
    CHECK(mapped_cache_matches(reopened, 864.0));
  }
#endif
}
