/**
 * @file unit_test_distance_sampling_weights.cpp
 * @brief Direct contract tests for the degenerate D-sampling seam.
 *
 * Public entry point exercised by every case in this file:
 * `dtwc::core::distance_sampling_weights` (dtwc/core/distance_sampling_weights.hpp).
 *
 * That function is the weight step of `dtwc::core::kmedoids_pp`, the one
 * D-sampling loop behind `dtwc::init::Kmeanspp`, `dtwc::fast_pam` and barycenter
 * k-means. Those routes are covered behaviourally in
 * unit_test_clustering_algorithms.cpp and unit_test_fast_pam.cpp, but they only
 * ever reach the seam through a distance matrix, so they cannot pin the
 * rejection paths. This file pins the contract.
 *
 * Equality is asserted exactly, not within a tolerance, because exactness IS the
 * contract: nonnegative inputs must pass through unchanged, selected entries must
 * be exactly 0.0, and kmedoids_pp branches on `total <= 0.0`. Every literal below
 * is chosen so that each running partial sum is exactly representable in binary64,
 * which keeps the totals reassociation-invariant under the project's
 * `-fassociative-math` build (`DTWC_FP_FLAGS` in cmake/StandardProjectSettings.cmake).
 *
 * Style note: inputs are named locals rather than inline braced-init-lists,
 * because a brace list does not protect its commas from preprocessor argument
 * splitting inside the two-parameter CHECK_THROWS_AS / CHECK_THROWS_WITH macros.
 * @author Volkan Kumtepeli
 */

#include <core/distance_sampling_weights.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using Catch::Matchers::ContainsSubstring;

namespace {

// Distinctive on purpose: the throwing cases assert this exact string reaches the
// message, which is what makes a failed initialisation attributable to its route.
const std::string kCaller = "unit::d_sampling_probe";

dtwc::core::DistanceSamplingWeights weigh(const std::vector<double> &distances,
                                          const std::vector<dtwc::index_t> &selected)
{
  return dtwc::core::distance_sampling_weights(distances, selected, kCaller);
}

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

} // namespace

// Entry point: dtwc::core::distance_sampling_weights.
TEST_CASE("distance_sampling_weights leaves nonnegative input unchanged",
          "[core][distance_sampling][f10]")
{
  const std::vector<double> distances{ 0.0, 1.5, 2.25, 4.0 };
  const auto weights = weigh(distances, {});

  REQUIRE(weights.values.size() == distances.size());
  for (std::size_t i = 0; i < distances.size(); ++i)
    CHECK(weights.values[i] == distances[i]); // exact: the weight is the value

  // Partial sums 0.0, 1.5, 3.75, 7.75 are each exact in binary64.
  CHECK(weights.total == 7.75);
}

// Entry point: dtwc::core::distance_sampling_weights.
TEST_CASE("distance_sampling_weights zeroes selected entries and drops them from the total",
          "[core][distance_sampling][f10]")
{
  const std::vector<double> distances{ 1.0, 2.0, 4.0, 8.0 };
  const std::vector<dtwc::index_t> selected{ 1, 3 };
  const auto weights = weigh(distances, selected);

  // Selected slots keep the value-initialised 0.0; their input value never leaks.
  CHECK(weights.values[1] == 0.0);
  CHECK(weights.values[3] == 0.0);
  CHECK(weights.values[0] == 1.0);
  CHECK(weights.values[2] == 4.0);
  CHECK(weights.total == 5.0); // 2.0 and 8.0 excluded
}

// Entry point: dtwc::core::distance_sampling_weights.
TEST_CASE("distance_sampling_weights reports an exactly zero total for degenerate input",
          "[core][distance_sampling][f10][degenerate]")
{
  // Both callers branch on `total <= 0.0` and fall back to a deterministic
  // first-unselected pick (dtwc/initialisation.cpp:180 and :202,
  // dtwc/algorithms/fast_pam.cpp:495), because std::discrete_distribution
  // requires a positive total weight.
  const std::vector<double> expected_zeros{ 0.0, 0.0, 0.0 };

  SECTION("identical series give literal zero weights")
  {
    const std::vector<double> distances{ 0.0, 0.0, 0.0 };
    const auto weights = weigh(distances, {});
    CHECK(weights.values == expected_zeros);
    CHECK(weights.total == 0.0);
  }

  SECTION("an empty distance vector yields an empty, zero-total result")
  {
    const auto weights = weigh({}, {});
    CHECK(weights.values.empty());
    CHECK(weights.total == 0.0);
  }
}

// Entry point: dtwc::core::distance_sampling_weights.
TEST_CASE("distance_sampling_weights rejects a non-finite unselected distance",
          "[core][distance_sampling][f10][validation]")
{
  // std::isfinite is meaningful here because the build deliberately omits
  // -ffinite-math-only (`DTWC_FP_FLAGS` in cmake/StandardProjectSettings.cmake).

  SECTION("NaN")
  {
    const std::vector<double> distances{ 1.0, kNaN, 2.0 };
    CHECK_THROWS_AS(weigh(distances, {}), std::runtime_error);
    CHECK_THROWS_WITH(weigh(distances, {}),
                      ContainsSubstring("initialization distance must be finite"));
    CHECK_THROWS_WITH(weigh(distances, {}), ContainsSubstring(kCaller));
  }

  SECTION("positive infinity")
  {
    const std::vector<double> distances{ 1.0, kInf };
    CHECK_THROWS_AS(weigh(distances, {}), std::runtime_error);
    CHECK_THROWS_WITH(weigh(distances, {}), ContainsSubstring(kCaller));
  }

  SECTION("negative infinity")
  {
    const std::vector<double> distances{ -kInf, 1.0 };
    CHECK_THROWS_AS(weigh(distances, {}), std::runtime_error);
    CHECK_THROWS_WITH(weigh(distances, {}), ContainsSubstring(kCaller));
  }
}

// Entry point: dtwc::core::distance_sampling_weights.
TEST_CASE("distance_sampling_weights rejects a non-finite selected distance",
          "[core][distance_sampling][f10][validation]")
{
  // Every non-finite input must fail closed. Selected slots
  // are excluded from the distribution, but accepting poison there would make
  // the seam's validity depend on caller-specific overwrites.
  const std::vector<dtwc::index_t> selected{ 0 };

  SECTION("NaN")
  {
    const std::vector<double> distances{ kNaN, 2.0, 3.0 };
    CHECK_THROWS_AS(weigh(distances, selected), std::runtime_error);
    CHECK_THROWS_WITH(weigh(distances, selected),
                      ContainsSubstring("initialization distance must be finite"));
  }

  SECTION("positive infinity")
  {
    const std::vector<double> distances{ kInf, 2.0, 3.0 };
    CHECK_THROWS_AS(weigh(distances, selected), std::runtime_error);
  }

  SECTION("negative infinity")
  {
    const std::vector<double> distances{ -kInf, 2.0, 3.0 };
    CHECK_THROWS_AS(weigh(distances, selected), std::runtime_error);
  }
}

// Entry point: dtwc::core::distance_sampling_weights.
TEST_CASE("distance_sampling_weights rejects an out-of-range selected index",
          "[core][distance_sampling][f10][validation]")
{
  const std::vector<double> distances{ 1.0, 2.0, 3.0 };

  SECTION("index at or past the end")
  {
    const std::vector<dtwc::index_t> selected{ 3 };
    CHECK_THROWS_AS(weigh(distances, selected), std::logic_error);
    CHECK_THROWS_WITH(weigh(distances, selected),
                      ContainsSubstring("selected index is out of range"));
    CHECK_THROWS_WITH(weigh(distances, selected), ContainsSubstring(kCaller));
  }

  SECTION("negative index")
  {
    const std::vector<dtwc::index_t> selected{ -1 };
    CHECK_THROWS_AS(weigh(distances, selected), std::logic_error);
    CHECK_THROWS_WITH(weigh(distances, selected), ContainsSubstring(kCaller));
  }

  SECTION("any selected index against an empty distance vector")
  {
    const std::vector<dtwc::index_t> selected{ 0 };
    CHECK_THROWS_AS(weigh({}, selected), std::logic_error);
  }
}
