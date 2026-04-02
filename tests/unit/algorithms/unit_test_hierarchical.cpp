/**
 * @file unit_test_hierarchical.cpp
 * @brief Unit tests for agglomerative hierarchical clustering.
 *
 * @details Tests use a hand-computed 4-point example with L1 DTW distances
 * (single-element series), verifying merge order and distances for all three
 * linkage criteria, as well as cut_dendrogram correctness.
 *
 * Distance matrix (symmetric):
 *       0    1    2    3
 *   0 [ 0    1    5    6 ]
 *   1 [ 1    0    4    5 ]
 *   2 [ 5    4    0    1 ]
 *   3 [ 6    5    1    0 ]
 *
 * @date 02 Apr 2026
 */

#include <dtwc.hpp>
#include <algorithms/hierarchical.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <set>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------
namespace {

/// Build a 4-point problem whose pairwise DTW distances match the table above.
/// Series are single scalars: {0}, {1}, {5}, {6}.
/// DTW with L1 norm on 1D series of length 1 gives |a - b|.
dtwc::Problem make_4point_problem()
{
  dtwc::Data data;
  data.p_vec  = { { 0.0 }, { 1.0 }, { 5.0 }, { 6.0 } };
  data.p_names = { "a", "b", "c", "d" };

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.verbose = false;
  prob.fillDistanceMatrix();
  return prob;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Linkage-specific dendrogram tests
// ---------------------------------------------------------------------------

TEST_CASE("Hierarchical: single linkage 4-point", "[hierarchical]")
{
  auto prob = make_4point_problem();
  auto dend = dtwc::algorithms::build_dendrogram(
    prob, { dtwc::algorithms::Linkage::Single });

  REQUIRE(dend.n_points == 4);
  REQUIRE(dend.merges.size() == 3);

  // The two tie-distance-1 pairs are {0,1} and {2,3}.
  // Tie-breaking: lexicographically smallest pair first → {0,1}, then {2,3}.
  REQUIRE_THAT(dend.merges[0].distance, WithinAbs(1.0, 1e-10));
  REQUIRE(dend.merges[0].cluster_a == 0);
  REQUIRE(dend.merges[0].cluster_b == 1);

  REQUIRE_THAT(dend.merges[1].distance, WithinAbs(1.0, 1e-10));
  REQUIRE(dend.merges[1].cluster_a == 2);
  REQUIRE(dend.merges[1].cluster_b == 3);

  // Single linkage between {0,1} and {2,3}: min(d(0,2), d(0,3), d(1,2), d(1,3))
  //   = min(5, 6, 4, 5) = 4.0
  REQUIRE_THAT(dend.merges[2].distance, WithinAbs(4.0, 1e-10));
  REQUIRE(dend.merges[2].new_size == 4);
}

TEST_CASE("Hierarchical: complete linkage 4-point", "[hierarchical]")
{
  auto prob = make_4point_problem();
  auto dend = dtwc::algorithms::build_dendrogram(
    prob, { dtwc::algorithms::Linkage::Complete });

  REQUIRE(dend.merges.size() == 3);

  REQUIRE_THAT(dend.merges[0].distance, WithinAbs(1.0, 1e-10));
  REQUIRE(dend.merges[0].cluster_a == 0);
  REQUIRE(dend.merges[0].cluster_b == 1);

  REQUIRE_THAT(dend.merges[1].distance, WithinAbs(1.0, 1e-10));
  REQUIRE(dend.merges[1].cluster_a == 2);
  REQUIRE(dend.merges[1].cluster_b == 3);

  // Complete linkage: max(d(0,2), d(0,3), d(1,2), d(1,3))
  //   After merges: d({0,1},{2})=max(5,4)=5, d({0,1},{3})=max(6,5)=6
  //   d({0,1},{2,3}) = max(5,6) = 6.0
  REQUIRE_THAT(dend.merges[2].distance, WithinAbs(6.0, 1e-10));
}

TEST_CASE("Hierarchical: average linkage 4-point", "[hierarchical]")
{
  auto prob = make_4point_problem();
  auto dend = dtwc::algorithms::build_dendrogram(
    prob, { dtwc::algorithms::Linkage::Average });

  REQUIRE(dend.merges.size() == 3);

  REQUIRE_THAT(dend.merges[0].distance, WithinAbs(1.0, 1e-10));
  REQUIRE(dend.merges[0].cluster_a == 0);
  REQUIRE(dend.merges[0].cluster_b == 1);

  REQUIRE_THAT(dend.merges[1].distance, WithinAbs(1.0, 1e-10));
  REQUIRE(dend.merges[1].cluster_a == 2);
  REQUIRE(dend.merges[1].cluster_b == 3);

  // Average (UPGMA) linkage:
  //   After step 1: d({0,1}, 2) = (1*5 + 1*4)/2 = 4.5
  //                 d({0,1}, 3) = (1*6 + 1*5)/2 = 5.5
  //   After step 2: d({0,1}, {2,3}) = (1*4.5 + 1*5.5)/2 = 5.0
  REQUIRE_THAT(dend.merges[2].distance, WithinAbs(5.0, 1e-10));
}

// ---------------------------------------------------------------------------
// cut_dendrogram tests
// ---------------------------------------------------------------------------

TEST_CASE("Hierarchical: cut(1) gives all same label", "[hierarchical]")
{
  auto prob = make_4point_problem();
  auto dend = dtwc::algorithms::build_dendrogram(prob);
  auto result = dtwc::algorithms::cut_dendrogram(dend, prob, 1);

  REQUIRE(result.labels.size() == 4);
  for (auto l : result.labels)
    REQUIRE(l == 0);
  REQUIRE(result.medoid_indices.size() == 1);
}

TEST_CASE("Hierarchical: cut(4) gives all different labels", "[hierarchical]")
{
  auto prob = make_4point_problem();
  auto dend = dtwc::algorithms::build_dendrogram(prob);
  auto result = dtwc::algorithms::cut_dendrogram(dend, prob, 4);

  REQUIRE(result.labels.size() == 4);

  std::set<int> unique_labels(result.labels.begin(), result.labels.end());
  REQUIRE(unique_labels.size() == 4);
  REQUIRE(result.medoid_indices.size() == 4);

  // Each point is its own medoid.
  for (int i = 0; i < 4; ++i)
    REQUIRE(result.medoid_indices[static_cast<size_t>(result.labels[i])] == i);
}

TEST_CASE("Hierarchical: cut(2) medoids correct", "[hierarchical]")
{
  auto prob = make_4point_problem();
  auto dend = dtwc::algorithms::build_dendrogram(prob);
  auto result = dtwc::algorithms::cut_dendrogram(dend, prob, 2);

  REQUIRE(result.labels.size() == 4);
  REQUIRE(result.medoid_indices.size() == 2);

  // Cluster {0,1}: both points have sum-of-distances = 1 → tie → medoid = 0
  // Cluster {2,3}: both points have sum-of-distances = 1 → tie → medoid = 2
  // Points 0 and 1 share a label; points 2 and 3 share a different label.
  REQUIRE(result.labels[0] == result.labels[1]);
  REQUIRE(result.labels[2] == result.labels[3]);
  REQUIRE(result.labels[0] != result.labels[2]);

  // Medoids must be the two tie-breaking winners (smallest index).
  const int label_01 = result.labels[0];
  const int label_23 = result.labels[2];
  REQUIRE(result.medoid_indices[static_cast<size_t>(label_01)] == 0);
  REQUIRE(result.medoid_indices[static_cast<size_t>(label_23)] == 2);

  REQUIRE(result.total_cost >= 0.0);
}

// ---------------------------------------------------------------------------
// Guard / error condition tests
// ---------------------------------------------------------------------------

TEST_CASE("Hierarchical: throws when N > max_points", "[hierarchical]")
{
  dtwc::Data data;
  for (int i = 0; i < 100; ++i) {
    data.p_vec.push_back({ static_cast<double>(i) });
    data.p_names.push_back("s" + std::to_string(i));
  }

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.verbose = false;
  prob.fillDistanceMatrix();

  dtwc::algorithms::HierarchicalOptions opts;
  opts.max_points = 50;
  REQUIRE_THROWS(dtwc::algorithms::build_dendrogram(prob, opts));
}

TEST_CASE("Hierarchical: throws when matrix not computed", "[hierarchical]")
{
  dtwc::Data data;
  data.p_vec   = { { 1.0 }, { 2.0 }, { 3.0 } };
  data.p_names = { "a", "b", "c" };

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.verbose = false;
  // Intentionally NOT calling fillDistanceMatrix().

  REQUIRE_THROWS(dtwc::algorithms::build_dendrogram(prob));
}

// ---------------------------------------------------------------------------
// Determinism test
// ---------------------------------------------------------------------------

TEST_CASE("Hierarchical: deterministic tie-breaking", "[hierarchical]")
{
  // All pairwise distances are 0 — every merge is a tie.
  auto equal_dist_prob = []() {
    dtwc::Data data;
    data.p_vec   = { { 0.0 }, { 0.0 }, { 0.0 } };
    data.p_names = { "a", "b", "c" };
    dtwc::Problem prob;
    prob.set_data(std::move(data));
    prob.verbose = false;
    prob.fillDistanceMatrix();
    return prob;
  };

  auto prob1 = equal_dist_prob();
  auto prob2 = equal_dist_prob();
  auto d1 = dtwc::algorithms::build_dendrogram(prob1);
  auto d2 = dtwc::algorithms::build_dendrogram(prob2);

  REQUIRE(d1.merges.size() == d2.merges.size());
  for (size_t i = 0; i < d1.merges.size(); ++i) {
    REQUIRE(d1.merges[i].cluster_a == d2.merges[i].cluster_a);
    REQUIRE(d1.merges[i].cluster_b == d2.merges[i].cluster_b);
    REQUIRE_THAT(d1.merges[i].distance, WithinAbs(d2.merges[i].distance, 1e-15));
  }
}

// ---------------------------------------------------------------------------
// Merge size bookkeeping test
// ---------------------------------------------------------------------------

TEST_CASE("Hierarchical: merge sizes are correct", "[hierarchical]")
{
  auto prob = make_4point_problem();
  auto dend = dtwc::algorithms::build_dendrogram(prob);

  // Step 0: {0}+{1} → size 2
  REQUIRE(dend.merges[0].new_size == 2);
  // Step 1: {2}+{3} → size 2
  REQUIRE(dend.merges[1].new_size == 2);
  // Step 2: {0,1}+{2,3} → size 4
  REQUIRE(dend.merges[2].new_size == 4);
}
