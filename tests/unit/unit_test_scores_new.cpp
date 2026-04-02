/**
 * @file unit_test_scores_new.cpp
 * @brief Unit tests for 5 new cluster quality metrics:
 *   Dunn Index, Inertia, Calinski-Harabasz Index (internal),
 *   Adjusted Rand Index, Normalized Mutual Information (external).
 *
 * @date 02 Apr 2026
 * @author Claude (coding agent)
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace dtwc;

// ---------------------------------------------------------------------------
// Helper: 4-point, 2-cluster problem with known distance matrix
//
//   Points: {0.0}, {1.0}, {5.0}, {6.0}  (length-1 series)
//   DTW on length-1 series = |a - b|
//   Distance matrix:
//       0   1   5   6
//       1   0   4   5
//       5   4   0   1
//       6   5   1   0
//
//   Clusters: {0,1} -> cluster 0,  {2,3} -> cluster 1
//   Medoids:  point 0 for cluster 0, point 2 for cluster 1
// ---------------------------------------------------------------------------
static Problem make_4point_problem()
{
  std::vector<std::vector<data_t>> vecs = {
    { 0.0 }, // point 0
    { 1.0 }, // point 1
    { 5.0 }, // point 2
    { 6.0 }, // point 3
  };
  std::vector<std::string> names = { "p0", "p1", "p2", "p3" };

  Data data(std::move(vecs), std::move(names));

  Problem prob("test_4pt");
  prob.set_data(std::move(data));
  prob.set_numberOfClusters(2);

  prob.fillDistanceMatrix();

  // Cluster assignment: 0->0, 1->0, 2->1, 3->1
  prob.clusters_ind = { 0, 0, 1, 1 };
  // Medoids: cluster 0 -> point 0, cluster 1 -> point 2
  prob.centroids_ind = { 0, 2 };

  return prob;
}

// ---------------------------------------------------------------------------
// Dunn Index tests
// ---------------------------------------------------------------------------
TEST_CASE("Dunn Index: known 4-point problem", "[scores][dunn]")
{
  auto prob = make_4point_problem();
  double dunn = scores::dunnIndex(prob);

  // min_inter = min(d(0,2), d(0,3), d(1,2), d(1,3)) = min(5,6,4,5) = 4
  // max_intra = max(d(0,1), d(2,3))                 = max(1,1) = 1
  // Dunn = 4 / 1 = 4.0
  REQUIRE_THAT(dunn, WithinAbs(4.0, 1e-12));
}

TEST_CASE("Dunn Index: throws when not clustered", "[scores][dunn]")
{
  Problem prob("empty");
  REQUIRE_THROWS_AS(scores::dunnIndex(prob), std::runtime_error);
}

TEST_CASE("Dunn Index: well-separated > poorly-separated", "[scores][dunn]")
{
  // Good clustering
  auto prob_good = make_4point_problem();
  double dunn_good = scores::dunnIndex(prob_good);

  // Bad clustering: assign all 4 points to a single cluster (well, 2-cluster but with
  // the inter-cluster being tiny)
  // Use: {0,1,2,3} where 0,2 are cluster 0 and 1,3 are cluster 1
  // min_inter = min(d(0,1), d(0,3), d(2,1), d(2,3)) = min(1,6,4,1) = 1
  // max_intra = max(d(0,2), d(1,3)) = max(5,5) = 5
  // Dunn_bad = 1/5 = 0.2
  auto prob_bad = make_4point_problem();
  prob_bad.clusters_ind = { 0, 1, 0, 1 };
  prob_bad.centroids_ind = { 0, 1 };
  double dunn_bad = scores::dunnIndex(prob_bad);

  REQUIRE(dunn_good > dunn_bad);
}

// ---------------------------------------------------------------------------
// Inertia tests
// ---------------------------------------------------------------------------
TEST_CASE("Inertia: known 4-point problem", "[scores][inertia]")
{
  auto prob = make_4point_problem();
  double result = scores::inertia(prob);

  // d(0, medoid0=0) = 0
  // d(1, medoid0=0) = 1
  // d(2, medoid1=2) = 0
  // d(3, medoid1=2) = 1
  // total = 0 + 1 + 0 + 1 = 2.0
  REQUIRE_THAT(result, WithinAbs(2.0, 1e-12));
}

TEST_CASE("Inertia: throws when not clustered", "[scores][inertia]")
{
  Problem prob("empty");
  REQUIRE_THROWS_AS(scores::inertia(prob), std::runtime_error);
}

TEST_CASE("Inertia: better clustering has lower inertia", "[scores][inertia]")
{
  auto prob_good = make_4point_problem();
  double inertia_good = scores::inertia(prob_good);

  // Suboptimal: medoid for cluster 0 is point 1, for cluster 1 is point 3
  auto prob_worse = make_4point_problem();
  prob_worse.clusters_ind = { 0, 0, 1, 1 };
  prob_worse.centroids_ind = { 1, 3 }; // non-optimal medoids
  double inertia_worse = scores::inertia(prob_worse);

  // With suboptimal medoids inertia should be >= optimal inertia
  REQUIRE(inertia_worse >= inertia_good);
}

// ---------------------------------------------------------------------------
// Calinski-Harabasz Index tests
// ---------------------------------------------------------------------------
TEST_CASE("Calinski-Harabasz Index: known 4-point problem", "[scores][ch]")
{
  auto prob = make_4point_problem();
  double ch = scores::calinskiHarabaszIndex(prob);

  // Row sums: 0+1+5+6=12, 1+0+4+5=10, 5+4+0+1=10, 6+5+1+0=12
  // Tie between index 1 and 2 (both sum=10), argmin picks first -> overall_medoid = 1
  //
  // W = sum d(i, medoid_c)^2
  //   cluster 0: d(0,0)^2 + d(1,0)^2 = 0 + 1 = 1
  //   cluster 1: d(2,2)^2 + d(3,2)^2 = 0 + 1 = 1
  //   W = 2
  //
  // B = sum_c |c| * d(medoid_c, overall_medoid=1)^2
  //   cluster 0: 2 * d(0,1)^2 = 2 * 1 = 2
  //   cluster 1: 2 * d(2,1)^2 = 2 * 16 = 32
  //   B = 34
  //
  // CH = (B/(k-1)) / (W/(N-k)) = (34/1) / (2/2) = 34.0
  REQUIRE_THAT(ch, WithinAbs(34.0, 1e-10));
}

TEST_CASE("Calinski-Harabasz Index: throws when not clustered", "[scores][ch]")
{
  Problem prob("empty");
  REQUIRE_THROWS_AS(scores::calinskiHarabaszIndex(prob), std::runtime_error);
}

TEST_CASE("Calinski-Harabasz Index: throws with 1 cluster", "[scores][ch]")
{
  std::vector<std::vector<data_t>> vecs = { { 1.0 }, { 2.0 } };
  std::vector<std::string> names = { "a", "b" };
  Data data(std::move(vecs), std::move(names));
  Problem prob("one_cluster");
  prob.set_data(std::move(data));
  prob.set_numberOfClusters(1);
  prob.clusters_ind = { 0, 0 };
  prob.centroids_ind = { 0 };
  REQUIRE_THROWS_AS(scores::calinskiHarabaszIndex(prob), std::runtime_error);
}

TEST_CASE("Calinski-Harabasz Index: better clustering has higher CH", "[scores][ch]")
{
  // Well-separated clusters
  auto prob_good = make_4point_problem();
  double ch_good = scores::calinskiHarabaszIndex(prob_good);

  // Bad clustering: mix the points across clusters
  // cluster 0: {0,2}, cluster 1: {1,3} — inter-cluster distances are small
  auto prob_bad = make_4point_problem();
  prob_bad.clusters_ind = { 0, 1, 0, 1 };
  prob_bad.centroids_ind = { 0, 1 };
  double ch_bad = scores::calinskiHarabaszIndex(prob_bad);

  // Better clustering should have higher CH
  REQUIRE(ch_good > ch_bad);
}

// ---------------------------------------------------------------------------
// Adjusted Rand Index tests
// ---------------------------------------------------------------------------
TEST_CASE("ARI: perfect agreement", "[scores][ari]")
{
  std::vector<int> labels = { 0, 0, 1, 1 };
  double ari = scores::adjustedRandIndex(labels, labels);
  REQUIRE_THAT(ari, WithinAbs(1.0, 1e-12));
}

TEST_CASE("ARI: permuted labels still gives 1.0", "[scores][ari]")
{
  // {0,0,1,1} and {1,1,0,0} are equivalent clusterings (permutation invariant)
  std::vector<int> true_labels = { 0, 0, 1, 1 };
  std::vector<int> pred_labels = { 1, 1, 0, 0 };
  double ari = scores::adjustedRandIndex(true_labels, pred_labels);
  REQUIRE_THAT(ari, WithinAbs(1.0, 1e-12));
}

TEST_CASE("ARI: low agreement gives near-zero ARI", "[scores][ari]")
{
  // true={0,0,0,1,1,1}, pred={0,1,0,1,0,1} — alternating, very poor agreement
  std::vector<int> true_labels = { 0, 0, 0, 1, 1, 1 };
  std::vector<int> pred_labels = { 0, 1, 0, 1, 0, 1 };
  double ari = scores::adjustedRandIndex(true_labels, pred_labels);
  // Should be close to 0 (or even negative)
  REQUIRE(ari < 0.1);
}

TEST_CASE("ARI: throws on size mismatch", "[scores][ari]")
{
  std::vector<int> a = { 0, 0, 1 };
  std::vector<int> b = { 0, 1 };
  REQUIRE_THROWS_AS(scores::adjustedRandIndex(a, b), std::invalid_argument);
}

TEST_CASE("ARI: 6-point two-cluster known result", "[scores][ari]")
{
  // Perfect match
  std::vector<int> true_labels = { 0, 0, 0, 1, 1, 1 };
  std::vector<int> pred_labels = { 0, 0, 0, 1, 1, 1 };
  REQUIRE_THAT(scores::adjustedRandIndex(true_labels, pred_labels), WithinAbs(1.0, 1e-12));
}

// ---------------------------------------------------------------------------
// Normalized Mutual Information tests
// ---------------------------------------------------------------------------
TEST_CASE("NMI: perfect agreement", "[scores][nmi]")
{
  std::vector<int> labels = { 0, 0, 1, 1 };
  double nmi = scores::normalizedMutualInformation(labels, labels);
  REQUIRE_THAT(nmi, WithinAbs(1.0, 1e-12));
}

TEST_CASE("NMI: permuted labels gives 1.0", "[scores][nmi]")
{
  std::vector<int> true_labels = { 0, 0, 1, 1 };
  std::vector<int> pred_labels = { 1, 1, 0, 0 };
  double nmi = scores::normalizedMutualInformation(true_labels, pred_labels);
  REQUIRE_THAT(nmi, WithinAbs(1.0, 1e-12));
}

TEST_CASE("NMI: low agreement gives low NMI", "[scores][nmi]")
{
  std::vector<int> true_labels = { 0, 0, 0, 1, 1, 1 };
  std::vector<int> pred_labels = { 0, 1, 0, 1, 0, 1 };
  double nmi = scores::normalizedMutualInformation(true_labels, pred_labels);
  // Should be well below 1.0
  REQUIRE(nmi < 0.5);
  REQUIRE(nmi >= 0.0);
}

TEST_CASE("NMI: throws on size mismatch", "[scores][nmi]")
{
  std::vector<int> a = { 0, 0, 1 };
  std::vector<int> b = { 0, 1 };
  REQUIRE_THROWS_AS(scores::normalizedMutualInformation(a, b), std::invalid_argument);
}

TEST_CASE("NMI: value is in [0, 1] for all test cases", "[scores][nmi]")
{
  // Various clusterings
  std::vector<std::pair<std::vector<int>, std::vector<int>>> cases = {
    { { 0, 0, 1, 1 }, { 0, 0, 1, 1 } },
    { { 0, 0, 1, 1 }, { 1, 1, 0, 0 } },
    { { 0, 1, 2, 0 }, { 0, 0, 1, 1 } },
    { { 0, 0, 0, 1, 1, 1 }, { 0, 1, 0, 1, 0, 1 } },
  };
  for (auto &[t, p] : cases) {
    double nmi = scores::normalizedMutualInformation(t, p);
    REQUIRE(nmi >= 0.0);
    REQUIRE(nmi <= 1.0 + 1e-10);
  }
}
