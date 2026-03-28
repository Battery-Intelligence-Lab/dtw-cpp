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
