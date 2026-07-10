/**
 * @file unit_test_problem_semantic_transactions.cpp
 * @brief Phase 8 M48: semantic setters must provide a strong exception guarantee.
 *
 * Registered before production changes. A rejected variant/missing
 * cross-product must preserve both the prior selectors and every bit of an
 * already-published distance cache.
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <limits>
#include <string>
#include <vector>

namespace {

dtwc::Data two_series()
{
  return dtwc::Data(
    std::vector<std::vector<dtwc::data_t>>{{0.0}, {1.0}},
    std::vector<std::string>{"x", "y"});
}

void inject_complete_cache(dtwc::Problem &problem, double sentinel)
{
  auto &matrix = problem.dense_distance_matrix();
  matrix.resize(2);
  matrix.set(0, 0, 0.0);
  matrix.set(0, 1, sentinel);
  matrix.set(1, 1, 0.0);
}

constexpr const char *cross_product_error =
  "Non-Standard DTW variants require MissingStrategy::Error.";

} // namespace

TEST_CASE("M48 rejected missing-policy mutation preserves ADTW state and cache",
          "[m48][problem][transaction]")
{
  dtwc::Problem problem("m48-missing");
  problem.set_data(two_series());
  problem.set_variant(dtwc::core::DTWVariant::ADTW);
  inject_complete_cache(problem, 123.0);
  REQUIRE(problem.is_distance_matrix_filled());
  REQUIRE(problem.dist_by_ind(0, 1) == 123.0);

  CHECK_THROWS_WITH(
    problem.set_missing_strategy(dtwc::core::MissingStrategy::ZeroCost),
    Catch::Matchers::Equals(cross_product_error));

  CHECK(problem.variant_params.variant == dtwc::core::DTWVariant::ADTW);
  CHECK(problem.missing_strategy == dtwc::core::MissingStrategy::Error);
  CHECK(problem.is_distance_matrix_filled());
  double after = std::numeric_limits<double>::quiet_NaN();
  CHECK_NOTHROW(after = problem.dist_by_ind(0, 1));
  CHECK(after == 123.0);
}

TEST_CASE("M48 rejected variant mutation preserves ZeroCost state and cache",
          "[m48][problem][transaction]")
{
  dtwc::Problem problem("m48-variant");
  problem.set_data(two_series());
  problem.set_missing_strategy(dtwc::core::MissingStrategy::ZeroCost);
  inject_complete_cache(problem, 456.0);
  REQUIRE(problem.is_distance_matrix_filled());
  REQUIRE(problem.dist_by_ind(0, 1) == 456.0);

  CHECK_THROWS_WITH(
    problem.set_variant(dtwc::core::DTWVariant::ADTW),
    Catch::Matchers::Equals(cross_product_error));

  CHECK(problem.variant_params.variant == dtwc::core::DTWVariant::Standard);
  CHECK(problem.missing_strategy == dtwc::core::MissingStrategy::ZeroCost);
  CHECK(problem.is_distance_matrix_filled());
  double after = std::numeric_limits<double>::quiet_NaN();
  CHECK_NOTHROW(after = problem.dist_by_ind(0, 1));
  CHECK(after == 456.0);
}
