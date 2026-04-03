/**
 * @file unit_test_medoid_utils.cpp
 * @brief Unit tests for dtwc::algorithms::detail medoid utilities.
 *
 * @details Tests verify correctness of assign_to_nearest, compute_nearest_and_second,
 * find_cluster_medoid, and validate_medoids using simple synthetic distance functions
 * that do not require a Problem or distance matrix.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#include <algorithms/detail/medoid_utils.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc::algorithms::detail;

// ---------------------------------------------------------------------------
// Simple distance functions used across tests.
// ---------------------------------------------------------------------------

// |i - j| on the integer line.
auto abs_dist = [](int i, int j) -> double { return static_cast<double>(std::abs(i - j)); };

// Always zero — useful for tie-break tests.
auto zero_dist = [](int, int) -> double { return 0.0; };

// ===========================================================================
// assign_to_nearest
// ===========================================================================

TEST_CASE("assign_to_nearest: basic two medoids", "[medoid_utils][assign]")
{
    // Medoids at 0 and 9 on a 10-point integer line.
    std::vector<int> medoids = {0, 9};
    std::vector<int> labels;
    double cost = assign_to_nearest(abs_dist, medoids, 10, labels);

    REQUIRE(labels.size() == 10);

    // Points 0-4 are closer to medoid index 0 (point 0).
    // Point 4: dist to 0 = 4, dist to 9 = 5  -> label 0
    // Point 5: dist to 0 = 5, dist to 9 = 4  -> label 1
    for (int i = 0; i <= 4; ++i) REQUIRE(labels[i] == 0);
    for (int i = 5; i <= 9; ++i) REQUIRE(labels[i] == 1);

    REQUIRE(cost > 0.0);
}

TEST_CASE("assign_to_nearest: k=1 all points same cluster", "[medoid_utils][assign]")
{
    // Single medoid at 5; all 10 points go to label 0.
    std::vector<int> medoids = {5};
    std::vector<int> labels;
    double cost = assign_to_nearest(abs_dist, medoids, 10, labels);

    REQUIRE(labels.size() == 10);
    for (int i = 0; i < 10; ++i) REQUIRE(labels[i] == 0);

    // Cost = |0-5|+|1-5|+...+|9-5| = 5+4+3+2+1+0+1+2+3+4 = 25
    REQUIRE_THAT(cost, WithinAbs(25.0, 1e-10));
}

TEST_CASE("assign_to_nearest: medoid is its own nearest (zero distance)", "[medoid_utils][assign]")
{
    std::vector<int> medoids = {2, 7};
    std::vector<int> labels;
    assign_to_nearest(abs_dist, medoids, 10, labels);

    // Medoid 2 belongs to label 0 (dist 0 < any positive distance to medoid 7).
    REQUIRE(labels[2] == 0);
    // Medoid 7 belongs to label 1.
    REQUIRE(labels[7] == 1);
}

TEST_CASE("assign_to_nearest: labels vector is resized correctly", "[medoid_utils][assign]")
{
    std::vector<int> medoids = {3};
    std::vector<int> labels = {99, 99};  // pre-existing wrong size
    assign_to_nearest(abs_dist, medoids, 5, labels);
    REQUIRE(labels.size() == 5);
}

// ===========================================================================
// compute_nearest_and_second
// ===========================================================================

TEST_CASE("compute_nearest_and_second: three medoids", "[medoid_utils][nearest_second]")
{
    // Medoids at points 0, 5, 9.
    std::vector<int> medoids = {0, 5, 9};
    std::vector<int> nearest;
    std::vector<double> nearest_dist, second_dist;

    compute_nearest_and_second(abs_dist, medoids, 10, nearest, nearest_dist, second_dist);

    REQUIRE(nearest.size() == 10);
    REQUIRE(nearest_dist.size() == 10);
    REQUIRE(second_dist.size() == 10);

    // Point 0: nearest = medoid index 0 (dist 0), second = medoid index 1 (dist 5)
    REQUIRE(nearest[0] == 0);
    REQUIRE_THAT(nearest_dist[0], WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(second_dist[0], WithinAbs(5.0, 1e-10));

    // Point 9: nearest = medoid index 2 (dist 0), second = medoid index 1 (dist 4)
    REQUIRE(nearest[9] == 2);
    REQUIRE_THAT(nearest_dist[9], WithinAbs(0.0, 1e-10));
    REQUIRE_THAT(second_dist[9], WithinAbs(4.0, 1e-10));

    // Point 3: dist to medoid 0 = 3, dist to medoid 5 = 2, dist to medoid 9 = 6
    //          nearest = medoid index 1 (point 5, dist 2), second = medoid index 0 (dist 3)
    REQUIRE(nearest[3] == 1);
    REQUIRE_THAT(nearest_dist[3], WithinAbs(2.0, 1e-10));
    REQUIRE_THAT(second_dist[3], WithinAbs(3.0, 1e-10));
}

TEST_CASE("compute_nearest_and_second: k=1 second_dist is max", "[medoid_utils][nearest_second]")
{
    std::vector<int> medoids = {4};
    std::vector<int> nearest;
    std::vector<double> nearest_dist, second_dist;

    compute_nearest_and_second(abs_dist, medoids, 8, nearest, nearest_dist, second_dist);

    // With only one medoid there is no second-nearest.
    for (int i = 0; i < 8; ++i) {
        REQUIRE(nearest[i] == 0);
        REQUIRE_THAT(second_dist[i], WithinAbs(std::numeric_limits<double>::max(), 0.0));
    }
}

TEST_CASE("compute_nearest_and_second: output vectors resized correctly", "[medoid_utils][nearest_second]")
{
    std::vector<int> medoids = {1, 3};
    std::vector<int> nearest = {99};      // wrong size
    std::vector<double> nd, sd;

    compute_nearest_and_second(abs_dist, medoids, 5, nearest, nd, sd);

    REQUIRE(nearest.size() == 5);
    REQUIRE(nd.size() == 5);
    REQUIRE(sd.size() == 5);
}

// ===========================================================================
// find_cluster_medoid
// ===========================================================================

TEST_CASE("find_cluster_medoid: symmetric cluster picks centre", "[medoid_utils][find_medoid]")
{
    // Points 0-4 in cluster 0.  With |i-j|: point 2 minimises sum = 0+1+2+3+4 vs neighbours.
    // Sum for point 0: |0-0|+|0-1|+|0-2|+|0-3|+|0-4| = 0+1+2+3+4 = 10
    // Sum for point 2: |2-0|+|2-1|+|2-2|+|2-3|+|2-4| = 2+1+0+1+2 = 6  <- minimum
    std::vector<int> labels = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    int medoid = find_cluster_medoid(abs_dist, labels, 0, 10);
    REQUIRE(medoid == 2);
}

TEST_CASE("find_cluster_medoid: second cluster picks centre", "[medoid_utils][find_medoid]")
{
    // Points 5-9 in cluster 1.  Minimum-sum point is 7 (middle of {5,6,7,8,9}).
    std::vector<int> labels = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    int medoid = find_cluster_medoid(abs_dist, labels, 1, 10);
    REQUIRE(medoid == 7);
}

TEST_CASE("find_cluster_medoid: single point cluster returns that point", "[medoid_utils][find_medoid]")
{
    std::vector<int> labels = {0, 1, 1, 1};
    int medoid = find_cluster_medoid(abs_dist, labels, 0, 4);
    REQUIRE(medoid == 0);
}

TEST_CASE("find_cluster_medoid: tie breaks by smallest index", "[medoid_utils][find_medoid]")
{
    // Zero distance everywhere -> all costs equal -> smallest index wins.
    std::vector<int> labels = {0, 0, 0};
    int medoid = find_cluster_medoid(zero_dist, labels, 0, 3);
    REQUIRE(medoid == 0);
}

TEST_CASE("find_cluster_medoid: returns -1 for empty cluster", "[medoid_utils][find_medoid]")
{
    std::vector<int> labels = {1, 1, 1};
    int medoid = find_cluster_medoid(abs_dist, labels, 0, 3);
    REQUIRE(medoid == -1);
}

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
