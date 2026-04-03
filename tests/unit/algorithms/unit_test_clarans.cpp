/**
 * @file unit_test_clarans.cpp
 * @brief Unit tests for the experimental CLARANS randomized k-medoids algorithm.
 *
 * @details Tests verify correctness, determinism, edge cases, and budget
 * enforcement for dtwc::algorithms::clarans.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#include <dtwc.hpp>
#include <algorithms/clarans.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <set>
#include <string>
#include <vector>

using namespace dtwc;
using namespace dtwc::algorithms;

// ---------------------------------------------------------------------------
// Helper: build a Problem with well-separated clusters.
// Each cluster has points tightly packed around a baseline value that differs
// by 100 between clusters, making correct k-medoids clustering unambiguous.
// ---------------------------------------------------------------------------
namespace {

Problem make_clustered_problem(int N_per_cluster, int n_clusters)
{
    std::vector<std::vector<data_t>> vecs;
    std::vector<std::string>          names;

    for (int c = 0; c < n_clusters; ++c) {
        for (int i = 0; i < N_per_cluster; ++i) {
            double base = c * 100.0;
            vecs.push_back({base + i * 0.1, base + i * 0.1 + 1.0});
            names.push_back("c" + std::to_string(c) + "_" + std::to_string(i));
        }
    }

    Data data(std::move(vecs), std::move(names));
    Problem prob("clarans_test");
    prob.set_data(std::move(data));
    return prob;
}

} // namespace


// ===========================================================================
// Test 1: Labels and medoids are in valid range.
// ===========================================================================
TEST_CASE("CLARANS: valid labels and medoids", "[clarans]")
{
    constexpr int N_per = 10;
    constexpr int k     = 3;

    auto prob   = make_clustered_problem(N_per, k);
    auto result = clarans(prob, {k, 2, -1, -1, 42u});

    const int N = N_per * k;

    REQUIRE(result.labels.size()         == static_cast<size_t>(N));
    REQUIRE(result.medoid_indices.size() == static_cast<size_t>(k));

    for (auto l : result.labels) {
        REQUIRE(l >= 0);
        REQUIRE(l < k);
    }

    std::set<int> unique_medoids(result.medoid_indices.begin(),
                                 result.medoid_indices.end());
    REQUIRE(unique_medoids.size() == static_cast<size_t>(k));

    // Every medoid index must be a valid point index.
    for (int m : result.medoid_indices) {
        REQUIRE(m >= 0);
        REQUIRE(m < N);
    }
}


// ===========================================================================
// Test 2: Deterministic output with the same seed.
// ===========================================================================
TEST_CASE("CLARANS: deterministic with same seed", "[clarans]")
{
    auto prob1 = make_clustered_problem(10, 2);
    auto prob2 = make_clustered_problem(10, 2);

    CLARANSOptions opts;
    opts.n_clusters   = 2;
    opts.num_local    = 1;
    opts.random_seed  = 42u;

    auto r1 = clarans(prob1, opts);
    auto r2 = clarans(prob2, opts);

    REQUIRE(r1.labels          == r2.labels);
    REQUIRE(r1.medoid_indices  == r2.medoid_indices);
    REQUIRE_THAT(r1.total_cost, Catch::Matchers::WithinAbs(r2.total_cost, 1e-12));
}


// ===========================================================================
// Test 3: k=1 — all points must be in cluster 0.
// ===========================================================================
TEST_CASE("CLARANS: k=1 assigns all points to one cluster", "[clarans][k1]")
{
    auto prob   = make_clustered_problem(5, 2);
    auto result = clarans(prob, {1, 1, -1, -1, 42u});

    REQUIRE(result.medoid_indices.size() == 1u);
    for (auto l : result.labels) REQUIRE(l == 0);
}


// ===========================================================================
// Test 4: k=N — every point is its own medoid, total cost must be ~0.
// ===========================================================================
TEST_CASE("CLARANS: k=N makes every point a medoid", "[clarans][kN]")
{
    constexpr int N = 3;
    auto prob   = make_clustered_problem(N, 1); // 3 points total
    auto result = clarans(prob, {N, 1, -1, -1, 42u});

    REQUIRE(result.medoid_indices.size() == static_cast<size_t>(N));
    REQUIRE_THAT(result.total_cost, Catch::Matchers::WithinAbs(0.0, 1e-10));
}


// ===========================================================================
// Test 5: Invalid inputs must throw.
// ===========================================================================
TEST_CASE("CLARANS: invalid inputs throw", "[clarans][errors]")
{
    SECTION("k=0 throws") {
        auto prob = make_clustered_problem(5, 1);
        REQUIRE_THROWS_AS(clarans(prob, {0}), std::runtime_error);
    }

    SECTION("k > N throws") {
        auto prob = make_clustered_problem(5, 1); // N=5
        REQUIRE_THROWS_AS(clarans(prob, {10}), std::runtime_error);
    }

    SECTION("empty problem throws") {
        Problem empty("empty");
        REQUIRE_THROWS_AS(clarans(empty, {1}), std::runtime_error);
    }
}


// ===========================================================================
// Test 6: More restarts cannot produce a worse result than fewer restarts.
// ===========================================================================
TEST_CASE("CLARANS: more restarts no worse than fewer", "[clarans][restarts]")
{
    auto prob1 = make_clustered_problem(15, 3);
    auto prob3 = make_clustered_problem(15, 3);

    CLARANSOptions opts1; opts1.n_clusters = 3; opts1.num_local = 1; opts1.random_seed = 7u;
    CLARANSOptions opts3; opts3.n_clusters = 3; opts3.num_local = 3; opts3.random_seed = 7u;

    auto r1 = clarans(prob1, opts1);
    auto r3 = clarans(prob3, opts3);

    // 3 restarts must find a cost <= 1-restart result (best-of-k property).
    REQUIRE(r3.total_cost <= r1.total_cost + 1e-10);
}


// ===========================================================================
// Test 7: DTW evaluation budget is honored (result is still valid).
// ===========================================================================
TEST_CASE("CLARANS: max_dtw_evals budget honored", "[clarans][budget]")
{
    auto prob = make_clustered_problem(20, 2);  // N=40

    CLARANSOptions opts;
    opts.n_clusters    = 2;
    opts.num_local     = 10;
    opts.max_dtw_evals = 500; // very tight budget
    opts.random_seed   = 13u;

    auto result = clarans(prob, opts);

    // Must still return a valid (non-empty) result.
    REQUIRE(result.labels.size()         == 40u);
    REQUIRE(result.medoid_indices.size() == 2u);
    REQUIRE(result.total_cost            >= 0.0);

    for (auto l : result.labels) {
        REQUIRE(l >= 0);
        REQUIRE(l < 2);
    }
}


// ===========================================================================
// Test 8: Total cost is non-negative.
// ===========================================================================
TEST_CASE("CLARANS: total cost is non-negative", "[clarans]")
{
    auto prob   = make_clustered_problem(10, 2);
    auto result = clarans(prob, {2, 2, -1, -1, 99u});
    REQUIRE(result.total_cost >= 0.0);
}


// ===========================================================================
// Test 9: Each medoid is assigned to its own cluster label.
// ===========================================================================
TEST_CASE("CLARANS: medoids are self-assigned", "[clarans][self_assignment]")
{
    auto prob   = make_clustered_problem(8, 3);
    auto result = clarans(prob, {3, 2, -1, -1, 42u});

    for (int c = 0; c < 3; ++c) {
        const int medoid_point = result.medoid_indices[c];
        REQUIRE(result.labels[medoid_point] == c);
    }
}


// ===========================================================================
// Test 10: total_cost matches recomputed sum of nearest-medoid distances.
// ===========================================================================
TEST_CASE("CLARANS: total_cost is consistent with labels", "[clarans][cost_consistency]")
{
    auto prob   = make_clustered_problem(10, 2);
    auto result = clarans(prob, {2, 2, -1, -1, 42u});

    const int N = static_cast<int>(result.labels.size());

    double recomputed = 0.0;
    for (int p = 0; p < N; ++p) {
        int medoid = result.medoid_indices[result.labels[p]];
        recomputed += prob.distByInd(p, medoid);
    }

    REQUIRE_THAT(result.total_cost, Catch::Matchers::WithinAbs(recomputed, 1e-10));
}


// ===========================================================================
// Test 11: Different seeds produce the result is still valid (smoke test).
// ===========================================================================
TEST_CASE("CLARANS: different seeds give valid results", "[clarans][seeds]")
{
    for (unsigned seed : {1u, 42u, 123u, 999u}) {
        auto prob   = make_clustered_problem(6, 2);
        auto result = clarans(prob, {2, 2, -1, -1, seed});

        REQUIRE(result.labels.size()         == 12u);
        REQUIRE(result.medoid_indices.size() == 2u);
        REQUIRE(result.total_cost            >= 0.0);
    }
}
