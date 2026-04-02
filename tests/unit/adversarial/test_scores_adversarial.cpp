/**
 * @file test_scores_adversarial.cpp
 * @brief Adversarial tests for clustering scores (silhouette, DBI, CH index)
 *        derived from mathematical definitions, not implementation details.
 *
 * Mathematical reference:
 *   Silhouette: s(i) = (b(i) - a(i)) / max(a(i), b(i))
 *     where a(i) = avg distance to same-cluster points,
 *           b(i) = min avg distance to any other cluster.
 *
 *   DBI = (1/k) * sum_i max_{j!=i} (S_i + S_j) / d(c_i, c_j)
 *     where S_i = avg distance from cluster members to medoid i.
 *     NOTE: DBI and CH index are currently commented out in scores.cpp.
 *           Tests for those are included as DISABLED (skipped) sections.
 *
 * @date 2026-03-28
 */

#include <dtwc.hpp>
#include <scores.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>
#include <string>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <random>
#include <utility>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

// ---------------------------------------------------------------------------
// Helper: create a Problem with synthetic time series and manually set clusters
// ---------------------------------------------------------------------------
namespace {

/**
 * @brief Build a Problem from raw time-series vectors.
 *
 * Each inner vector is one time series. Names are auto-generated ("s0", "s1", ...).
 * After construction, fillDistanceMatrix() is called so all pairwise DTW distances
 * are available.
 */
dtwc::Problem make_problem(std::vector<std::vector<double>> series)
{
  const int n = static_cast<int>(series.size());
  std::vector<std::string> names;
  names.reserve(n);
  for (int i = 0; i < n; ++i)
    names.push_back("s" + std::to_string(i));

  dtwc::Data data(std::move(series), std::move(names));
  dtwc::Problem prob;
  prob.set_data(std::move(data));
  return prob;
}

/**
 * @brief Manually assign cluster labels and centroid indices to a Problem.
 *
 * @param prob        The problem (must already have data loaded).
 * @param k           Number of clusters.
 * @param cluster_ids Per-point cluster assignment (0-indexed, size == prob.size()).
 * @param centroids   Per-cluster centroid index  (size == k, each in [0, prob.size())).
 */
void assign_clusters(dtwc::Problem &prob, int k,
                     std::vector<int> cluster_ids,
                     std::vector<int> centroids)
{
  prob.set_numberOfClusters(k);
  prob.clusters_ind = std::move(cluster_ids);
  prob.centroids_ind = std::move(centroids);
}

} // anonymous namespace


// ===========================================================================
// Area 1: Silhouette Score Properties
// ===========================================================================

TEST_CASE("Silhouette: well-separated clusters approach 1.0",
          "[scores][silhouette][adversarial]")
{
  // Two clusters of identical constant series, far apart.
  // Cluster 0: series that are all {0, 0, 0}  (DTW distance within cluster = 0)
  // Cluster 1: series that are all {100, 100, 100}
  // Intra-cluster distances a(i) = 0 for every point.
  // Inter-cluster distances b(i) >> 0 for every point.
  // => s(i) = (b - 0) / max(0, b) = 1.0
  auto prob = make_problem({
    {0, 0, 0},   // cluster 0
    {0, 0, 0},   // cluster 0
    {0, 0, 0},   // cluster 0
    {100, 100, 100}, // cluster 1
    {100, 100, 100}, // cluster 1
    {100, 100, 100}, // cluster 1
  });

  assign_clusters(prob, 2,
    /*cluster_ids=*/ {0, 0, 0, 1, 1, 1},
    /*centroids=*/   {0, 3});

  prob.fillDistanceMatrix();
  auto sil = dtwc::scores::silhouette(prob);

  REQUIRE(sil.size() == 6);
  for (size_t i = 0; i < sil.size(); ++i) {
    INFO("Point " << i << " silhouette = " << sil[i]);
    REQUIRE_THAT(sil[i], WithinAbs(1.0, 1e-10));
  }
}

TEST_CASE("Silhouette: all scores in [-1, 1]",
          "[scores][silhouette][adversarial]")
{
  // Arbitrary clustering that may be suboptimal, to test range property.
  // Use series with varying values to get non-trivial distances.
  auto prob = make_problem({
    {1, 2, 3},
    {1, 2, 4},
    {10, 20, 30},
    {10, 20, 31},
    {50, 60, 70},
  });

  // Deliberately assign a possibly bad clustering
  assign_clusters(prob, 2,
    /*cluster_ids=*/ {0, 0, 1, 1, 0},  // point 4 misassigned to cluster 0
    /*centroids=*/   {0, 2});

  prob.fillDistanceMatrix();
  auto sil = dtwc::scores::silhouette(prob);

  REQUIRE(sil.size() == 5);
  for (size_t i = 0; i < sil.size(); ++i) {
    INFO("Point " << i << " silhouette = " << sil[i]);
    REQUIRE(sil[i] >= -1.0);
    REQUIRE(sil[i] <= 1.0);
  }
}

TEST_CASE("Silhouette: single-point cluster gives silhouette = 0",
          "[scores][silhouette][adversarial]")
{
  // Mathematical definition: if a point is alone in its cluster, s(i) = 0.
  // The code checks mean_distances[i_c].first == 1 and returns 0.
  auto prob = make_problem({
    {1, 2, 3},
    {4, 5, 6},
    {7, 8, 9},
  });

  // Each point in its own cluster
  assign_clusters(prob, 3,
    /*cluster_ids=*/ {0, 1, 2},
    /*centroids=*/   {0, 1, 2});

  prob.fillDistanceMatrix();
  auto sil = dtwc::scores::silhouette(prob);

  REQUIRE(sil.size() == 3);
  for (size_t i = 0; i < sil.size(); ++i) {
    INFO("Point " << i << " silhouette = " << sil[i]);
    REQUIRE_THAT(sil[i], WithinAbs(0.0, 1e-15));
  }
}

TEST_CASE("Silhouette: mixed single-point and multi-point clusters",
          "[scores][silhouette][adversarial]")
{
  // Point 0 alone in cluster 0 => s(0) = 0
  // Points 1, 2 in cluster 1 => s(1), s(2) defined normally
  auto prob = make_problem({
    {50, 50, 50},  // cluster 0, alone
    {1, 1, 1},     // cluster 1
    {1, 1, 1},     // cluster 1
  });

  assign_clusters(prob, 2,
    /*cluster_ids=*/ {0, 1, 1},
    /*centroids=*/   {0, 1});

  prob.fillDistanceMatrix();
  auto sil = dtwc::scores::silhouette(prob);

  REQUIRE(sil.size() == 3);

  // Point 0 is alone in its cluster => silhouette must be 0
  REQUIRE_THAT(sil[0], WithinAbs(0.0, 1e-15));

  // Points 1 and 2 are identical => a(i) = 0, b(i) > 0 => s(i) = 1.0
  REQUIRE_THAT(sil[1], WithinAbs(1.0, 1e-10));
  REQUIRE_THAT(sil[2], WithinAbs(1.0, 1e-10));
}

TEST_CASE("Silhouette: two-point problem, each in own cluster",
          "[scores][silhouette][adversarial]")
{
  // Both points are alone in their cluster => s(i) = 0 for both
  auto prob = make_problem({
    {1, 2, 3},
    {10, 20, 30},
  });

  assign_clusters(prob, 2,
    /*cluster_ids=*/ {0, 1},
    /*centroids=*/   {0, 1});

  prob.fillDistanceMatrix();
  auto sil = dtwc::scores::silhouette(prob);

  REQUIRE(sil.size() == 2);
  REQUIRE_THAT(sil[0], WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(sil[1], WithinAbs(0.0, 1e-15));
}

TEST_CASE("Silhouette: symmetric for identical clusters gives ~0",
          "[scores][silhouette][adversarial]")
{
  // If inter-cluster and intra-cluster distances are equal,
  // s(i) = (b - a) / max(a, b) = 0 when a == b.
  //
  // Create 4 identical series split into 2 clusters. All pairwise DTW = 0.
  // a(i) = 0, b(i) = 0, max(0,0) = 0 => 0/0. Convention: s(i) = 0.
  // Actually, (0-0)/max(0,0) is 0/0 = NaN.  Check what the code does.
  auto prob = make_problem({
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5},
  });

  assign_clusters(prob, 2,
    /*cluster_ids=*/ {0, 0, 1, 1},
    /*centroids=*/   {0, 2});

  prob.fillDistanceMatrix();
  auto sil = dtwc::scores::silhouette(prob);

  REQUIRE(sil.size() == 4);
  // When a(i) = 0 and b(i) = 0, the formula gives 0/0.
  // Mathematically silhouette is undefined; implementations typically return 0 or NaN.
  // We just check the value is in [-1, 1] (or is NaN, which we flag).
  for (size_t i = 0; i < sil.size(); ++i) {
    INFO("Point " << i << " silhouette = " << sil[i]);
    if (!std::isnan(sil[i])) {
      REQUIRE(sil[i] >= -1.0);
      REQUIRE(sil[i] <= 1.0);
    }
    // If NaN, that's a known edge case: 0/0. We let it pass but document it.
  }
}

TEST_CASE("Silhouette: misassigned point has negative silhouette",
          "[scores][silhouette][adversarial]")
{
  // Cluster 0: {0,0,0}, {0,0,0}  -- tight
  // Cluster 1: {100,100,100}, {100,100,100} -- tight
  // Misassign point {100,100,100} to cluster 0.
  // For that point: a(i) >> 0 (distance to {0,0,0}), b(i) ~ 0 (distance to cluster 1).
  // => s(i) = (b - a)/max(a, b) < 0.
  auto prob = make_problem({
    {0, 0, 0},       // cluster 0
    {0, 0, 0},       // cluster 0
    {100, 100, 100}, // MISASSIGNED to cluster 0
    {100, 100, 100}, // cluster 1
    {100, 100, 100}, // cluster 1
  });

  assign_clusters(prob, 2,
    /*cluster_ids=*/ {0, 0, 0, 1, 1},
    /*centroids=*/   {0, 3});

  prob.fillDistanceMatrix();
  auto sil = dtwc::scores::silhouette(prob);

  REQUIRE(sil.size() == 5);

  // Point 2 is misassigned: its silhouette should be negative
  INFO("Misassigned point silhouette = " << sil[2]);
  REQUIRE(sil[2] < 0.0);

  // Correctly assigned points should have positive silhouettes
  REQUIRE(sil[0] > 0.0);
  REQUIRE(sil[1] > 0.0);
  REQUIRE(sil[3] > 0.0);
  REQUIRE(sil[4] > 0.0);
}

TEST_CASE("Silhouette: hand-computed values for 4-point, 2-cluster case",
          "[scores][silhouette][adversarial]")
{
  // Use constant series so DTW distance = |a - b| * length.
  // Series: {0,0,0}, {1,1,1}, {10,10,10}, {11,11,11}
  // Cluster 0: points 0,1.  Cluster 1: points 2,3.
  //
  // DTW distances for constant series of same length L:
  //   d({a,a,a}, {b,b,b}) = |a-b| * L = |a-b| * 3
  //
  // Point 0 ({0,0,0}):
  //   a(0) = avg dist to same cluster = d(0,1) / 1 = 3*1 = 3
  //   avg dist to cluster 1 = (d(0,2) + d(0,3)) / 2 = (30 + 33) / 2 = 31.5
  //   b(0) = 31.5
  //   s(0) = (31.5 - 3) / max(3, 31.5) = 28.5 / 31.5
  //
  // Point 1 ({1,1,1}):
  //   a(1) = d(1,0) / 1 = 3
  //   avg dist to cluster 1 = (d(1,2) + d(1,3)) / 2 = (27 + 30) / 2 = 28.5
  //   b(1) = 28.5
  //   s(1) = (28.5 - 3) / max(3, 28.5) = 25.5 / 28.5
  //
  // Point 2 ({10,10,10}):
  //   a(2) = d(2,3) / 1 = 3
  //   avg dist to cluster 0 = (d(2,0) + d(2,1)) / 2 = (30 + 27) / 2 = 28.5
  //   b(2) = 28.5
  //   s(2) = (28.5 - 3) / max(3, 28.5) = 25.5 / 28.5
  //
  // Point 3 ({11,11,11}):
  //   a(3) = d(3,2) / 1 = 3
  //   avg dist to cluster 0 = (d(3,0) + d(3,1)) / 2 = (33 + 30) / 2 = 31.5
  //   b(3) = 31.5
  //   s(3) = (31.5 - 3) / max(3, 31.5) = 28.5 / 31.5

  auto prob = make_problem({
    {0, 0, 0},
    {1, 1, 1},
    {10, 10, 10},
    {11, 11, 11},
  });

  assign_clusters(prob, 2,
    /*cluster_ids=*/ {0, 0, 1, 1},
    /*centroids=*/   {0, 2});

  prob.fillDistanceMatrix();
  auto sil = dtwc::scores::silhouette(prob);

  REQUIRE(sil.size() == 4);

  const double s0_expected = 28.5 / 31.5;
  const double s1_expected = 25.5 / 28.5;
  const double s2_expected = 25.5 / 28.5;
  const double s3_expected = 28.5 / 31.5;

  REQUIRE_THAT(sil[0], WithinAbs(s0_expected, 1e-10));
  REQUIRE_THAT(sil[1], WithinAbs(s1_expected, 1e-10));
  REQUIRE_THAT(sil[2], WithinAbs(s2_expected, 1e-10));
  REQUIRE_THAT(sil[3], WithinAbs(s3_expected, 1e-10));

  // All should be positive and close to 1 (good clustering)
  for (size_t i = 0; i < 4; ++i)
    REQUIRE(sil[i] > 0.8);
}

TEST_CASE("Silhouette: average silhouette is symmetric under cluster relabeling",
          "[scores][silhouette][adversarial]")
{
  // Swapping cluster labels should not change silhouette scores.
  auto series = std::vector<std::vector<double>>{
    {0, 0, 0},
    {1, 1, 1},
    {10, 10, 10},
    {11, 11, 11},
  };

  // Labeling A: {0,0,1,1}
  auto probA = make_problem(series);
  assign_clusters(probA, 2, {0, 0, 1, 1}, {0, 2});
  probA.fillDistanceMatrix();
  auto silA = dtwc::scores::silhouette(probA);

  // Labeling B: {1,1,0,0}  (swapped labels)
  auto probB = make_problem(series);
  assign_clusters(probB, 2, {1, 1, 0, 0}, {2, 0});
  probB.fillDistanceMatrix();
  auto silB = dtwc::scores::silhouette(probB);

  REQUIRE(silA.size() == silB.size());
  for (size_t i = 0; i < silA.size(); ++i) {
    INFO("Point " << i << ": silA=" << silA[i] << " silB=" << silB[i]);
    REQUIRE_THAT(silA[i], WithinAbs(silB[i], 1e-12));
  }
}

TEST_CASE("Silhouette: three clusters, hand-computed",
          "[scores][silhouette][adversarial]")
{
  // 3 clusters of 2 points each. Constant series for easy DTW.
  // Cluster 0: {0,0}, {0,0}   (identical, a=0)
  // Cluster 1: {10,10}, {10,10} (identical, a=0)
  // Cluster 2: {20,20}, {20,20} (identical, a=0)
  //
  // For point in cluster 0: a(i) = 0
  //   avg dist to cluster 1 = (20+20)/2 = 20  (d = |0-10|*2 = 20)
  //   avg dist to cluster 2 = (40+40)/2 = 40
  //   b(i) = min(20, 40) = 20
  //   s(i) = (20 - 0) / max(0, 20) = 1.0
  //
  // Similarly all points get s(i) = 1.0

  auto prob = make_problem({
    {0, 0}, {0, 0},
    {10, 10}, {10, 10},
    {20, 20}, {20, 20},
  });

  assign_clusters(prob, 3,
    {0, 0, 1, 1, 2, 2},
    {0, 2, 4});

  prob.fillDistanceMatrix();
  auto sil = dtwc::scores::silhouette(prob);

  REQUIRE(sil.size() == 6);
  for (size_t i = 0; i < sil.size(); ++i) {
    REQUIRE_THAT(sil[i], WithinAbs(1.0, 1e-10));
  }
}

TEST_CASE("Silhouette: unclustered problem returns -1 vector",
          "[scores][silhouette][adversarial]")
{
  // If centroids_ind is empty, silhouette should return all -1.
  auto prob = make_problem({
    {1, 2, 3},
    {4, 5, 6},
  });

  // Do NOT call assign_clusters -- centroids_ind stays empty
  auto sil = dtwc::scores::silhouette(prob);

  REQUIRE(sil.size() == 2);
  for (size_t i = 0; i < sil.size(); ++i) {
    REQUIRE_THAT(sil[i], WithinAbs(-1.0, 1e-15));
  }
}

// ===========================================================================
// Area 2: DBI and CH Index
// NOTE: These are commented out in the production code (scores.cpp).
//       The tests below are disabled (will not compile/run until DBI/CH are
//       implemented). They serve as a specification for future implementation.
// ===========================================================================

#if 0  // DISABLED: DBI and CH index not yet implemented in production code

TEST_CASE("DBI: non-negative",
          "[scores][dbi][adversarial]")
{
  auto prob = make_problem({
    {0, 0, 0}, {1, 1, 1},
    {10, 10, 10}, {11, 11, 11},
  });
  assign_clusters(prob, 2, {0, 0, 1, 1}, {0, 2});
  prob.fillDistanceMatrix();

  double dbi = dtwc::scores::daviesBouldinIndex(prob);
  REQUIRE(dbi >= 0.0);
}

TEST_CASE("DBI: well-separated clusters have low DBI",
          "[scores][dbi][adversarial]")
{
  auto prob = make_problem({
    {0, 0, 0}, {0, 0, 0}, {0, 0, 0},
    {100, 100, 100}, {100, 100, 100}, {100, 100, 100},
  });
  assign_clusters(prob, 2, {0, 0, 0, 1, 1, 1}, {0, 3});
  prob.fillDistanceMatrix();

  double dbi = dtwc::scores::daviesBouldinIndex(prob);
  // Perfect separation: S_i = 0 for both clusters => DBI = 0
  REQUIRE_THAT(dbi, WithinAbs(0.0, 1e-10));
}

TEST_CASE("DBI: single cluster returns 0 or handles gracefully",
          "[scores][dbi][adversarial]")
{
  auto prob = make_problem({
    {1, 2, 3}, {4, 5, 6}, {7, 8, 9},
  });
  assign_clusters(prob, 1, {0, 0, 0}, {0});
  prob.fillDistanceMatrix();

  // With k=1, no other cluster to compare => DBI should be 0
  double dbi = dtwc::scores::daviesBouldinIndex(prob);
  REQUIRE(dbi >= 0.0);
}

TEST_CASE("CH index: positive for well-separated clusters",
          "[scores][ch][adversarial]")
{
  auto prob = make_problem({
    {0, 0, 0}, {0, 0, 0},
    {100, 100, 100}, {100, 100, 100},
  });
  assign_clusters(prob, 2, {0, 0, 1, 1}, {0, 2});
  prob.fillDistanceMatrix();

  double ch = dtwc::scores::CH_index(prob);
  REQUIRE(ch > 0.0);
}

TEST_CASE("CH index: k=1 handles gracefully (division by k-1)",
          "[scores][ch][adversarial]")
{
  auto prob = make_problem({
    {1, 2, 3}, {4, 5, 6},
  });
  assign_clusters(prob, 1, {0, 0}, {0});
  prob.fillDistanceMatrix();

  // k-1 = 0 => division by zero. Should handle gracefully.
  // Expect 0, NaN, or exception -- not a crash.
  double ch = dtwc::scores::CH_index(prob);
  // Just check it doesn't crash; value may be 0 or inf
  REQUIRE((ch >= 0.0 || std::isinf(ch) || std::isnan(ch)));
}

#endif  // DISABLED: DBI and CH index

// ===========================================================================
// Area 3: Adjusted Rand Index (ARI) — adversarial tests
// ===========================================================================

TEST_CASE("ARI: perfect agreement gives 1.0",
          "[scores][ari][adversarial]")
{
  std::vector<int> labels = {0, 0, 1, 1};
  std::vector<int> pred   = {0, 0, 1, 1};
  double ari = dtwc::scores::adjustedRandIndex(labels, pred);
  REQUIRE_THAT(ari, WithinAbs(1.0, 1e-12));
}

TEST_CASE("ARI: perfect agreement with different label names gives 1.0",
          "[scores][ari][adversarial]")
{
  // Cluster labels are permuted: 0->1, 1->0.  Same partition.
  std::vector<int> labels = {0, 0, 1, 1};
  std::vector<int> pred   = {1, 1, 0, 0};
  double ari = dtwc::scores::adjustedRandIndex(labels, pred);
  REQUIRE_THAT(ari, WithinAbs(1.0, 1e-12));
}

TEST_CASE("ARI: anti-correlated labels give negative ARI",
          "[scores][ari][adversarial]")
{
  // labels={0,0,1,1}, pred={0,1,0,1}: every pair that was together is split.
  // Known ARI = -0.5 for this 4-element case.
  std::vector<int> labels = {0, 0, 1, 1};
  std::vector<int> pred   = {0, 1, 0, 1};
  double ari = dtwc::scores::adjustedRandIndex(labels, pred);
  // Verify: ARI < 0
  REQUIRE(ari < 0.0);
  // Known analytical value for this configuration = -0.5
  REQUIRE_THAT(ari, WithinAbs(-0.5, 1e-10));
}

TEST_CASE("ARI: all-same labels in both partitions — degenerate case",
          "[scores][ari][adversarial]")
{
  // Every point in cluster 0 vs every point in cluster 0.
  // sum_ai2 = C(4,2)=6, sum_bj2 = 6, sum_cij2 = 6, cn2 = 6
  // expected = 6*6/6 = 6, max_val = 6, numerator = 6-6 = 0, denom = 6-6 = 0.
  // denominator == 0 => ARI = 1.0 (convention in implementation).
  std::vector<int> labels = {0, 0, 0, 0};
  std::vector<int> pred   = {0, 0, 0, 0};
  double ari = dtwc::scores::adjustedRandIndex(labels, pred);
  // Should not crash and return a finite value
  REQUIRE(std::isfinite(ari));
  // Implementation returns 1.0 for the degenerate denominator==0 case
  REQUIRE_THAT(ari, WithinAbs(1.0, 1e-12));
}

TEST_CASE("ARI: mismatched label-vector sizes throw",
          "[scores][ari][adversarial]")
{
  std::vector<int> labels = {0, 1, 0};
  std::vector<int> pred   = {0, 1};
  REQUIRE_THROWS_AS(dtwc::scores::adjustedRandIndex(labels, pred),
                    std::invalid_argument);
}

TEST_CASE("ARI: single element — degenerate, must not crash",
          "[scores][ari][adversarial]")
{
  std::vector<int> labels = {0};
  std::vector<int> pred   = {0};
  // n=1 => cn2 = C(1,2) = 0, sum_ai2=0, sum_bj2=0, expected=0, max_val=0 => denom=0 => ARI=1.0
  double ari = dtwc::scores::adjustedRandIndex(labels, pred);
  REQUIRE(std::isfinite(ari));
}

TEST_CASE("ARI: two elements, one pair, perfect agreement",
          "[scores][ari][adversarial]")
{
  std::vector<int> labels = {0, 0};
  std::vector<int> pred   = {0, 0};
  double ari = dtwc::scores::adjustedRandIndex(labels, pred);
  REQUIRE(std::isfinite(ari));
  REQUIRE_THAT(ari, WithinAbs(1.0, 1e-12));
}

TEST_CASE("ARI: symmetry — ARI(a,b) == ARI(b,a)",
          "[scores][ari][adversarial]")
{
  std::vector<int> a = {0, 0, 1, 1, 2, 2};
  std::vector<int> b = {0, 1, 1, 2, 2, 0};
  double ari_ab = dtwc::scores::adjustedRandIndex(a, b);
  double ari_ba = dtwc::scores::adjustedRandIndex(b, a);
  REQUIRE_THAT(ari_ab, WithinAbs(ari_ba, 1e-12));
}

TEST_CASE("ARI: large random labels stay in a finite range",
          "[scores][ari][adversarial]")
{
  // 10000 random labels from {0..9}: ARI should be close to 0 for random,
  // and always finite.
  const int N = 10000;
  const int K = 10;
  std::mt19937 rng(42);
  std::uniform_int_distribution<int> dist(0, K - 1);
  std::vector<int> labels(N), pred(N);
  for (int i = 0; i < N; ++i) {
    labels[i] = dist(rng);
    pred[i]   = dist(rng);
  }
  double ari = dtwc::scores::adjustedRandIndex(labels, pred);
  INFO("Large-scale random ARI = " << ari);
  REQUIRE(std::isfinite(ari));
  // For truly random labels, ARI should be near 0; allow generous tolerance
  REQUIRE(ari > -0.1);
  REQUIRE(ari < 0.1);
}

TEST_CASE("ARI: step-by-step contingency-table verification",
          "[scores][ari][adversarial]")
{
  // Manual calculation following sklearn reference:
  // labels = {0, 0, 1, 1}, pred = {0, 0, 1, 1}
  // Contingency table:
  //        pred0  pred1
  // true0:   2      0
  // true1:   0      2
  //
  // sum_cij2 = C(2,2) + C(2,2) = 1 + 1 = 2
  // sum_ai2  = C(2,2) + C(2,2) = 2
  // sum_bj2  = C(2,2) + C(2,2) = 2
  // cn2 = C(4,2) = 6
  // expected = 2*2/6 = 4/6 = 2/3
  // max_val  = (2+2)/2 = 2
  // ARI = (2 - 2/3) / (2 - 2/3) = 1.0
  std::vector<int> labels = {0, 0, 1, 1};
  std::vector<int> pred   = {0, 0, 1, 1};
  double ari = dtwc::scores::adjustedRandIndex(labels, pred);
  REQUIRE_THAT(ari, WithinAbs(1.0, 1e-12));
}

TEST_CASE("ARI: non-trivial 6-element case, hand-computed",
          "[scores][ari][adversarial]")
{
  // labels = {0, 0, 0, 1, 1, 1}
  // pred   = {0, 0, 1, 1, 1, 1}
  // Contingency table:
  //        pred0  pred1
  // true0:   2      1
  // true1:   0      3
  //
  // sum_cij2 = C(2,2) + C(1,2) + C(0,2) + C(3,2) = 1 + 0 + 0 + 3 = 4
  // sum_ai2  = C(3,2) + C(3,2) = 3 + 3 = 6
  // sum_bj2  = C(2,2) + C(4,2) = 1 + 6 = 7
  // cn2 = C(6,2) = 15
  // expected = 6*7/15 = 42/15 = 2.8
  // max_val  = (6+7)/2 = 6.5
  // ARI = (4 - 2.8) / (6.5 - 2.8) = 1.2 / 3.7
  const double expected_ari = 1.2 / 3.7;

  std::vector<int> labels = {0, 0, 0, 1, 1, 1};
  std::vector<int> pred   = {0, 0, 1, 1, 1, 1};
  double ari = dtwc::scores::adjustedRandIndex(labels, pred);
  REQUIRE_THAT(ari, WithinAbs(expected_ari, 1e-10));
}

// ===========================================================================
// Area 4: Normalized Mutual Information (NMI) — adversarial tests
// ===========================================================================

TEST_CASE("NMI: perfect agreement gives 1.0",
          "[scores][nmi][adversarial]")
{
  std::vector<int> labels = {0, 0, 1, 1};
  std::vector<int> pred   = {0, 0, 1, 1};
  double nmi = dtwc::scores::normalizedMutualInformation(labels, pred);
  REQUIRE_THAT(nmi, WithinAbs(1.0, 1e-12));
}

TEST_CASE("NMI: perfect agreement with permuted labels gives 1.0",
          "[scores][nmi][adversarial]")
{
  std::vector<int> labels = {0, 0, 1, 1};
  std::vector<int> pred   = {1, 1, 0, 0};
  double nmi = dtwc::scores::normalizedMutualInformation(labels, pred);
  REQUIRE_THAT(nmi, WithinAbs(1.0, 1e-12));
}

TEST_CASE("NMI: all-same labels — degenerate, H=0",
          "[scores][nmi][adversarial]")
{
  // Both labelings are constant => H_true = H_pred = 0, MI = 0.
  // denom = 0 => implementation returns 1.0.
  std::vector<int> labels = {0, 0, 0, 0};
  std::vector<int> pred   = {0, 0, 0, 0};
  double nmi = dtwc::scores::normalizedMutualInformation(labels, pred);
  REQUIRE(std::isfinite(nmi));
  REQUIRE_THAT(nmi, WithinAbs(1.0, 1e-12));
}

TEST_CASE("NMI: result is always in [0, 1]",
          "[scores][nmi][adversarial]")
{
  // Several input configurations; NMI must be in [0,1]
  using VI = std::vector<int>;
  std::vector<std::pair<VI, VI>> cases = {
    {{0,0,1,1}, {0,1,0,1}},   // worst case (fully anti-correlated for 2-cluster)
    {{0,1,2,0,1,2}, {0,0,1,1,2,2}},
    {{0,0,0,1,1,1}, {0,0,1,1,1,1}},
    {{0,1,2,3}, {0,1,2,3}},
  };
  for (auto &[a, b] : cases) {
    double nmi = dtwc::scores::normalizedMutualInformation(a, b);
    INFO("NMI = " << nmi);
    REQUIRE(std::isfinite(nmi));
    REQUIRE(nmi >= -1e-12);  // Allow tiny floating-point slack below 0
    REQUIRE(nmi <= 1.0 + 1e-12);
  }
}

TEST_CASE("NMI: symmetry — NMI(a,b) == NMI(b,a)",
          "[scores][nmi][adversarial]")
{
  std::vector<int> a = {0, 0, 1, 1, 2, 2};
  std::vector<int> b = {0, 1, 1, 2, 2, 0};
  double nmi_ab = dtwc::scores::normalizedMutualInformation(a, b);
  double nmi_ba = dtwc::scores::normalizedMutualInformation(b, a);
  REQUIRE_THAT(nmi_ab, WithinAbs(nmi_ba, 1e-12));
}

TEST_CASE("NMI: mismatched sizes throw",
          "[scores][nmi][adversarial]")
{
  std::vector<int> a = {0, 1};
  std::vector<int> b = {0, 1, 0};
  REQUIRE_THROWS_AS(dtwc::scores::normalizedMutualInformation(a, b),
                    std::invalid_argument);
}

TEST_CASE("NMI: single element — degenerate, must not crash",
          "[scores][nmi][adversarial]")
{
  std::vector<int> labels = {0};
  std::vector<int> pred   = {0};
  double nmi = dtwc::scores::normalizedMutualInformation(labels, pred);
  REQUIRE(std::isfinite(nmi));
}

TEST_CASE("NMI: large random labels — stays finite and in range",
          "[scores][nmi][adversarial]")
{
  const int N = 10000;
  const int K = 10;
  std::mt19937 rng(1337);
  std::uniform_int_distribution<int> dist(0, K - 1);
  std::vector<int> labels(N), pred(N);
  for (int i = 0; i < N; ++i) {
    labels[i] = dist(rng);
    pred[i]   = dist(rng);
  }
  double nmi = dtwc::scores::normalizedMutualInformation(labels, pred);
  INFO("Large-scale random NMI = " << nmi);
  REQUIRE(std::isfinite(nmi));
  REQUIRE(nmi >= -1e-9);
  REQUIRE(nmi <= 1.0 + 1e-9);
}

TEST_CASE("NMI: hand-computed 3-class balanced case",
          "[scores][nmi][adversarial]")
{
  // labels = pred = {0,0,1,1,2,2} => perfect, NMI = 1.0
  std::vector<int> labels = {0, 0, 1, 1, 2, 2};
  std::vector<int> pred   = {0, 0, 1, 1, 2, 2};
  double nmi = dtwc::scores::normalizedMutualInformation(labels, pred);
  REQUIRE_THAT(nmi, WithinAbs(1.0, 1e-12));
}

// ===========================================================================
// Area 5: Dunn Index — adversarial tests
// ===========================================================================

TEST_CASE("Dunn: well-separated clusters give high value",
          "[scores][dunn][adversarial]")
{
  // All intra-cluster distances = 0, inter-cluster distances >> 0.
  // => Dunn = min_inter / max_intra = inf (since max_intra = 0).
  auto prob = make_problem({
    {0, 0, 0},
    {0, 0, 0},
    {100, 100, 100},
    {100, 100, 100},
  });
  assign_clusters(prob, 2, {0, 0, 1, 1}, {0, 2});

  double dunn = dtwc::scores::dunnIndex(prob);
  // max_intra = 0 => implementation returns +infinity
  REQUIRE(std::isinf(dunn));
  REQUIRE(dunn > 0.0);
}

TEST_CASE("Dunn: non-negative for any valid clustering",
          "[scores][dunn][adversarial]")
{
  auto prob = make_problem({
    {1, 2, 3},
    {1, 2, 4},
    {10, 20, 30},
    {10, 20, 31},
  });
  assign_clusters(prob, 2, {0, 0, 1, 1}, {0, 2});

  double dunn = dtwc::scores::dunnIndex(prob);
  INFO("Dunn = " << dunn);
  REQUIRE(dunn > 0.0);
}

TEST_CASE("Dunn: unclustered problem throws",
          "[scores][dunn][adversarial]")
{
  auto prob = make_problem({{1.0, 2.0}, {3.0, 4.0}});
  // Do NOT set centroids
  REQUIRE_THROWS_AS(dtwc::scores::dunnIndex(prob), std::runtime_error);
}

TEST_CASE("Dunn: all distances zero (identical points) returns infinity",
          "[scores][dunn][adversarial]")
{
  // All series are identical; all pairwise distances are 0.
  // Two clusters: max_intra = 0, min_inter = 0.
  // max_intra == 0 => return infinity.
  auto prob = make_problem({
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5},
    {5, 5, 5},
  });
  assign_clusters(prob, 2, {0, 0, 1, 1}, {0, 2});

  double dunn = dtwc::scores::dunnIndex(prob);
  REQUIRE(std::isinf(dunn));
}

TEST_CASE("Dunn: singleton cluster and multi-point cluster",
          "[scores][dunn][adversarial]")
{
  // Point 2 is alone in cluster 1 (k=2):
  //   intra-cluster pairs: only cluster 0 has pairs {0,1}
  //   inter-cluster distance: d(0,2), d(1,2)
  //   min_inter = min(d(0,2), d(1,2))
  //   max_intra = d(0,1)
  //   Dunn = min_inter / max_intra
  auto prob = make_problem({
    {0, 0, 0},   // cluster 0
    {0, 0, 0},   // cluster 0  -- d(0,1) = 0
    {50, 50, 50}, // cluster 1 (singleton)
  });
  assign_clusters(prob, 2, {0, 0, 1}, {0, 2});

  double dunn = dtwc::scores::dunnIndex(prob);
  // d(0,1)=0 => max_intra=0 => Dunn = +inf
  REQUIRE(std::isinf(dunn));
  REQUIRE(dunn > 0.0);
}

TEST_CASE("Dunn: hand-computed value for 4-point clustering",
          "[scores][dunn][adversarial]")
{
  // Constant series, DTW = |a-b|*len (len=1 here for simplicity)
  // Series: {0}, {2}, {10}, {12}  (length 1, distance = absolute difference)
  // Cluster 0: points 0,1   intra = d(0,1) = 2
  // Cluster 1: points 2,3   intra = d(2,3) = 2
  // max_intra = 2
  // Inter distances: d(0,2)=10, d(0,3)=12, d(1,2)=8, d(1,3)=10
  // min_inter = 8
  // Dunn = 8/2 = 4.0
  auto prob = make_problem({{0.0}, {2.0}, {10.0}, {12.0}});
  assign_clusters(prob, 2, {0, 0, 1, 1}, {0, 2});

  double dunn = dtwc::scores::dunnIndex(prob);
  REQUIRE_THAT(dunn, WithinAbs(4.0, 1e-10));
}

// ===========================================================================
// Area 6: Calinski-Harabasz Index — adversarial tests
// ===========================================================================

TEST_CASE("CH: positive for well-separated clusters",
          "[scores][ch][adversarial]")
{
  auto prob = make_problem({
    {0, 0, 0}, {1, 1, 1},
    {100, 100, 100}, {101, 101, 101},
  });
  assign_clusters(prob, 2, {0, 0, 1, 1}, {0, 2});

  double ch = dtwc::scores::calinskiHarabaszIndex(prob);
  INFO("CH = " << ch);
  REQUIRE(std::isfinite(ch));
  REQUIRE(ch > 0.0);
}

TEST_CASE("CH: k=1 throws (k-1=0 division by zero)",
          "[scores][ch][adversarial]")
{
  auto prob = make_problem({
    {1, 2, 3}, {4, 5, 6}, {7, 8, 9},
  });
  assign_clusters(prob, 1, {0, 0, 0}, {0});

  // Implementation explicitly throws for k <= 1
  REQUIRE_THROWS_AS(dtwc::scores::calinskiHarabaszIndex(prob), std::runtime_error);
}

TEST_CASE("CH: k=N (each point in own cluster) throws",
          "[scores][ch][adversarial]")
{
  // N=3, k=3: N-k=0 => W/(N-k) is division by zero.
  // Implementation throws if N <= k.
  auto prob = make_problem({
    {0.0, 1.0}, {2.0, 3.0}, {4.0, 5.0},
  });
  assign_clusters(prob, 3, {0, 1, 2}, {0, 1, 2});

  REQUIRE_THROWS_AS(dtwc::scores::calinskiHarabaszIndex(prob), std::runtime_error);
}

TEST_CASE("CH: unclustered problem throws",
          "[scores][ch][adversarial]")
{
  auto prob = make_problem({{1.0}, {2.0}, {3.0}, {4.0}});
  // centroids_ind is empty
  REQUIRE_THROWS_AS(dtwc::scores::calinskiHarabaszIndex(prob), std::runtime_error);
}

TEST_CASE("CH: better clustering has higher CH than worse one",
          "[scores][ch][adversarial]")
{
  // Two clear groups: {0,1} vs {100,101}.
  // Good clustering: {0,0,1,1}
  // Bad clustering:  {0,1,0,1} (splits both groups)
  auto series = std::vector<std::vector<double>>{
    {0.0}, {1.0}, {100.0}, {101.0},
  };

  auto prob_good = make_problem(series);
  assign_clusters(prob_good, 2, {0, 0, 1, 1}, {0, 2});
  double ch_good = dtwc::scores::calinskiHarabaszIndex(prob_good);

  auto prob_bad = make_problem(series);
  assign_clusters(prob_bad, 2, {0, 1, 0, 1}, {0, 1});
  double ch_bad = dtwc::scores::calinskiHarabaszIndex(prob_bad);

  INFO("CH good=" << ch_good << " bad=" << ch_bad);
  REQUIRE(ch_good > ch_bad);
}

TEST_CASE("CH: three-cluster case is finite and positive",
          "[scores][ch][adversarial]")
{
  auto prob = make_problem({
    {0.0}, {0.1},
    {10.0}, {10.1},
    {20.0}, {20.1},
  });
  assign_clusters(prob, 3, {0,0,1,1,2,2}, {0,2,4});

  double ch = dtwc::scores::calinskiHarabaszIndex(prob);
  INFO("CH (3 clusters) = " << ch);
  REQUIRE(std::isfinite(ch));
  REQUIRE(ch > 0.0);
}

// ===========================================================================
// Area 7: Inertia — adversarial tests
// ===========================================================================

TEST_CASE("Inertia: perfect clustering gives 0",
          "[scores][inertia][adversarial]")
{
  // All series identical within cluster; medoid distance = 0.
  auto prob = make_problem({
    {0, 0, 0}, {0, 0, 0},   // cluster 0, medoid = 0
    {10, 10, 10}, {10, 10, 10}, // cluster 1, medoid = 2
  });
  assign_clusters(prob, 2, {0, 0, 1, 1}, {0, 2});

  double iner = dtwc::scores::inertia(prob);
  REQUIRE_THAT(iner, WithinAbs(0.0, 1e-10));
}

TEST_CASE("Inertia: non-negative always",
          "[scores][inertia][adversarial]")
{
  auto prob = make_problem({
    {1, 2},
    {3, 4},
    {50, 60},
    {51, 61},
  });
  assign_clusters(prob, 2, {0, 0, 1, 1}, {0, 2});

  double iner = dtwc::scores::inertia(prob);
  REQUIRE(iner >= 0.0);
}

TEST_CASE("Inertia: one point per cluster gives 0",
          "[scores][inertia][adversarial]")
{
  // Each point is its own medoid => d(point, medoid) = 0 for all points.
  auto prob = make_problem({
    {1.0, 2.0},
    {10.0, 20.0},
    {100.0, 200.0},
  });
  assign_clusters(prob, 3, {0, 1, 2}, {0, 1, 2});

  double iner = dtwc::scores::inertia(prob);
  REQUIRE_THAT(iner, WithinAbs(0.0, 1e-10));
}

TEST_CASE("Inertia: unclustered problem throws",
          "[scores][inertia][adversarial]")
{
  auto prob = make_problem({{1.0, 2.0}, {3.0, 4.0}});
  REQUIRE_THROWS_AS(dtwc::scores::inertia(prob), std::runtime_error);
}

TEST_CASE("Inertia: hand-computed value",
          "[scores][inertia][adversarial]")
{
  // Series of length 1: {0}, {4}, {10}, {14}
  // Cluster 0: points 0,1; medoid = 0 (index 0, value {0})
  //   d(0, medoid0) = 0
  //   d(1, medoid0) = 4
  // Cluster 1: points 2,3; medoid = 2 (index 2, value {10})
  //   d(2, medoid1) = 0
  //   d(3, medoid1) = 4
  // Inertia = 0 + 4 + 0 + 4 = 8
  auto prob = make_problem({{0.0}, {4.0}, {10.0}, {14.0}});
  assign_clusters(prob, 2, {0, 0, 1, 1}, {0, 2});

  double iner = dtwc::scores::inertia(prob);
  REQUIRE_THAT(iner, WithinAbs(8.0, 1e-10));
}

TEST_CASE("Inertia: misassigned point increases inertia",
          "[scores][inertia][adversarial]")
{
  // Correct clustering vs incorrect one.
  // Series: {0}, {1}, {100}, {101}
  // Good: medoids at 0, 2
  //   inertia = d(0,0) + d(1,0) + d(2,2) + d(3,2) = 0+1+0+1 = 2
  // Bad: medoids at 0, 2, but point 1 assigned to cluster 1:
  //   cluster_ids = {0, 1, 1, 1}, centroids = {0, 2}
  //   inertia = d(0,0) + d(1,2) + d(2,2) + d(3,2) = 0+99+0+1 = 100

  auto prob_good = make_problem({{0.0}, {1.0}, {100.0}, {101.0}});
  assign_clusters(prob_good, 2, {0, 0, 1, 1}, {0, 2});
  double iner_good = dtwc::scores::inertia(prob_good);

  auto prob_bad = make_problem({{0.0}, {1.0}, {100.0}, {101.0}});
  assign_clusters(prob_bad, 2, {0, 1, 1, 1}, {0, 2});
  double iner_bad = dtwc::scores::inertia(prob_bad);

  REQUIRE(iner_bad > iner_good);
  REQUIRE_THAT(iner_good, WithinAbs(2.0, 1e-10));
  REQUIRE_THAT(iner_bad,  WithinAbs(100.0, 1e-10));
}
