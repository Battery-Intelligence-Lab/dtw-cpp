/**
 * @file unit_test_medoid_utils.cpp
 * @brief Unit tests for dtwc::algorithms::detail medoid utilities.
 *
 * @details Tests verify correctness of validate_medoids using inputs that do not
 * require a Problem or distance matrix.
 *
 * D1: the assign_to_nearest / compute_nearest_and_second / find_cluster_medoid
 * cases were removed with the shared helpers themselves. They had no caller
 * and their semantics differed from every shipping assignment loop, so they were
 * false coverage for logic that lives in the algorithms.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#include <algorithms/detail/medoid_utils.hpp>

#include <catch2/catch_test_macros.hpp>

#include <stdexcept>
#include <vector>

using namespace dtwc::algorithms::detail;

// ===========================================================================
// validate_medoids
// ===========================================================================

TEST_CASE("validate_medoids: valid inputs do not throw", "[medoid_utils][validate]")
{
    REQUIRE_NOTHROW(validate_medoids({0, 3, 7}, 10));
    REQUIRE_NOTHROW(validate_medoids({0}, 1));
    REQUIRE_NOTHROW(validate_medoids({9}, 10));
}

TEST_CASE("validate_medoids: empty list throws", "[medoid_utils][validate]")
{
    REQUIRE_THROWS_AS(validate_medoids({}, 10), std::runtime_error);
}

TEST_CASE("validate_medoids: index equal to N throws (out of range)", "[medoid_utils][validate]")
{
    REQUIRE_THROWS_AS(validate_medoids({0, 10}, 10), std::runtime_error);
}

TEST_CASE("validate_medoids: negative index throws", "[medoid_utils][validate]")
{
    REQUIRE_THROWS_AS(validate_medoids({-1, 3}, 10), std::runtime_error);
}

TEST_CASE("validate_medoids: duplicate indices throw", "[medoid_utils][validate]")
{
    REQUIRE_THROWS_AS(validate_medoids({3, 3}, 10), std::runtime_error);
    REQUIRE_THROWS_AS(validate_medoids({1, 4, 4, 7}, 10), std::runtime_error);
}
