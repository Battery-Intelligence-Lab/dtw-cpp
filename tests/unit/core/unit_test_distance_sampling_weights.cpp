/**
 * @file unit_test_distance_sampling_weights.cpp
 * @brief Direct contract tests for the signed/degenerate D-sampling seam (F10).
 *
 * Public entry point exercised by every case in this file:
 * `dtwc::core::distance_sampling_weights` (dtwc/core/distance_sampling_weights.hpp:31).
 *
 * That function is the shared seam behind all three D-sampling callers --
 * `dtwc::init::Kmeanspp` and `dtwc::init::Kmeanspp_seeded`
 * (dtwc/initialisation.cpp:113) and `dtwc::fast_pam_seeded`
 * (dtwc/algorithms/fast_pam.cpp:492). Those routes are covered behaviourally in
 * unit_test_clustering_algorithms.cpp and unit_test_fast_pam.cpp, but they only
 * ever reach the seam through a distance matrix, so they cannot pin the
 * translation arithmetic or the rejection paths. This file pins the contract.
 *
 * Equality is asserted exactly, not within a tolerance, because exactness IS the
 * contract: nonnegative inputs must pass through unchanged, selected entries must
 * be exactly 0.0, and both callers branch on `total <= 0.0`. Every literal below
 * is chosen so that each running partial sum is exactly representable in binary64,
 * which keeps the totals reassociation-invariant under the project's
 * `-fassociative-math` build (cmake/StandardProjectSettings.cmake:59-70).
 *
 * Style note: inputs are named locals rather than inline braced-init-lists,
 * because a brace list does not protect its commas from preprocessor argument
 * splitting inside the two-parameter CHECK_THROWS_AS / CHECK_THROWS_WITH macros.
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
                                          const std::vector<int> &selected)
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
    CHECK(weights.values[i] == distances[i]); // exact: shift is 0.0, so d - 0.0 == d

  // Partial sums 0.0, 1.5, 3.75, 7.75 are each exact in binary64.
  CHECK(weights.total == 7.75);
}

// Entry point: dtwc::core::distance_sampling_weights.
TEST_CASE("distance_sampling_weights zeroes selected entries and drops them from the total",
          "[core][distance_sampling][f10]")
{
  const std::vector<double> distances{ 1.0, 2.0, 4.0, 8.0 };
  const std::vector<int> selected{ 1, 3 };
  const auto weights = weigh(distances, selected);

  // Selected slots keep the value-initialised 0.0; their input value never leaks.
  CHECK(weights.values[1] == 0.0);
  CHECK(weights.values[3] == 0.0);
  CHECK(weights.values[0] == 1.0);
  CHECK(weights.values[2] == 4.0);
  CHECK(weights.total == 5.0); // 2.0 and 8.0 excluded
}

// Entry point: dtwc::core::distance_sampling_weights.
TEST_CASE("distance_sampling_weights translates negative input preserving differences",
          "[core][distance_sampling][f10][signed]")
{
  // Raw Soft-DTW is an admissible variant that yields negative off-diagonal
  // dissimilarities. A weighted sampler cannot consume those directly, so the
  // whole unselected vector is translated by the same -min(0, d_min).
  const std::vector<double> distances{ -4.0, -1.0, 0.0, 3.0 };
  const auto weights = weigh(distances, {});

  CHECK(weights.values[0] == 0.0); // the minimum lands exactly on zero
  CHECK(weights.values[1] == 3.0);
  CHECK(weights.values[2] == 4.0);
  CHECK(weights.values[3] == 7.0);

  // A common shift preserves order and every pairwise difference. It does NOT
  // preserve proportional sampling probabilities: smaller weights gain more
  // relative mass. R2-D13 owns the research decision on that tradeoff.
  CHECK(weights.values[3] - weights.values[1] == distances[3] - distances[1]);
  CHECK(weights.values[2] - weights.values[0] == distances[2] - distances[0]);

  // Partial sums 0.0, 3.0, 7.0, 14.0 are each exact in binary64.
  CHECK(weights.total == 14.0);
}

// Entry point: dtwc::core::distance_sampling_weights.
TEST_CASE("distance_sampling_weights computes the shift from unselected entries only",
          "[core][distance_sampling][f10][signed][regression]")
{
  // Pins the `if (is_selected[i]) continue;` guard in the minimum scan at
  // dtwc/core/distance_sampling_weights.hpp:46. Letting a selected entry into
  // that scan is a real bug class: selected slots hold stale or forced values
  // (dtwc/algorithms/fast_pam.cpp:486 forces them to 0.0) that are not part of
  // the sampling distribution, so they must not be able to move it.

  SECTION("a selected outlier far below zero must not translate a nonnegative vector")
  {
    const std::vector<double> distances{ -1000.0, 2.0, 3.0 };
    const std::vector<int> selected{ 0 };
    const auto weights = weigh(distances, selected);

    // min over unselected is 2.0, so shift = min(0.0, 2.0) = 0.0 and nothing moves.
    // Including index 0 in the scan would give shift = -1000.0, weights 1002/1003.
    CHECK(weights.values[0] == 0.0);
    CHECK(weights.values[1] == 2.0);
    CHECK(weights.values[2] == 3.0);
    CHECK(weights.total == 5.0);
  }

  SECTION("a selected outlier must not set the shift in the negative regime")
  {
    const std::vector<double> distances{ -100.0, -3.0, -1.0 };
    const std::vector<int> selected{ 0 };
    const auto weights = weigh(distances, selected);

    // shift = -3.0 (the min over unselected), not -100.0.
    CHECK(weights.values[0] == 0.0);
    CHECK(weights.values[1] == 0.0);
    CHECK(weights.values[2] == 2.0); // would be 99.0 if the selected entry counted
    CHECK(weights.total == 2.0);
  }
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

  SECTION("equal negative dissimilarities translate onto exactly zero")
  {
    const std::vector<double> distances{ -2.5, -2.5, -2.5 };
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
  // -ffinite-math-only (cmake/StandardProjectSettings.cmake:59-70).

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

  SECTION("a translation that overflows to infinity is rejected")
  {
    // shift = -DBL_MAX, so the largest entry translates to 2*DBL_MAX -> +inf.
    // This is the only route into the second guard, on line 60 of the header.
    const double big = std::numeric_limits<double>::max();
    const std::vector<double> distances{ -big, big };
    CHECK_THROWS_AS(weigh(distances, {}), std::runtime_error);
    CHECK_THROWS_WITH(weigh(distances, {}),
                      ContainsSubstring("translated initialization weight is invalid"));
  }
}

// Entry point: dtwc::core::distance_sampling_weights.
TEST_CASE("distance_sampling_weights rejects a non-finite selected distance",
          "[core][distance_sampling][f10][validation]")
{
  // PLAN R3-F10 requires every non-finite input to fail closed. Selected slots
  // are excluded from the distribution, but accepting poison there would make
  // the seam's validity depend on caller-specific overwrites.
  const std::vector<int> selected{ 0 };

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
    const std::vector<int> selected{ 3 };
    CHECK_THROWS_AS(weigh(distances, selected), std::logic_error);
    CHECK_THROWS_WITH(weigh(distances, selected),
                      ContainsSubstring("selected index is out of range"));
    CHECK_THROWS_WITH(weigh(distances, selected), ContainsSubstring(kCaller));
  }

  SECTION("negative index")
  {
    const std::vector<int> selected{ -1 };
    CHECK_THROWS_AS(weigh(distances, selected), std::logic_error);
    CHECK_THROWS_WITH(weigh(distances, selected), ContainsSubstring(kCaller));
  }

  SECTION("any selected index against an empty distance vector")
  {
    const std::vector<int> selected{ 0 };
    CHECK_THROWS_AS(weigh({}, selected), std::logic_error);
  }
}
