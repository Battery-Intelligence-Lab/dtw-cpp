/**
 * @file unit_test_scores_new.cpp
 * @brief Unit tests for the cluster quality metrics:
 *   Silhouette, Davies-Bouldin, Dunn, Inertia, Calinski-Harabasz (internal),
 *   Adjusted Rand Index, Normalized Mutual Information (external),
 *   including hand-computed values and the degenerate partitions.
 *
 * @author Volkan Kumtepeli
 * @author Claude 4.6
 * @date 02 Apr 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>
#include <string>
#include <stdexcept>
#include <cmath>
#include <utility>

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
  prob.set_n_clusters(2);

  prob.fill_distance_matrix();

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
  double dunn = scores::dunn(prob);

  // min_inter = min(d(0,2), d(0,3), d(1,2), d(1,3)) = min(5,6,4,5) = 4
  // max_intra = max(d(0,1), d(2,3))                 = max(1,1) = 1
  // Dunn = 4 / 1 = 4.0
  REQUIRE_THAT(dunn, WithinAbs(4.0, 1e-12));
}

TEST_CASE("Dunn Index: throws when not clustered", "[scores][dunn]")
{
  Problem prob("empty");
  REQUIRE_THROWS_AS(scores::dunn(prob), dtwc::InvalidInput);
}

TEST_CASE("Dunn Index: well-separated > poorly-separated", "[scores][dunn]")
{
  // Good clustering
  auto prob_good = make_4point_problem();
  double dunn_good = scores::dunn(prob_good);

  // Bad clustering: assign all 4 points to a single cluster (well, 2-cluster but with
  // the inter-cluster being tiny)
  // Use: {0,1,2,3} where 0,2 are cluster 0 and 1,3 are cluster 1
  // min_inter = min(d(0,1), d(0,3), d(2,1), d(2,3)) = min(1,6,4,1) = 1
  // max_intra = max(d(0,2), d(1,3)) = max(5,5) = 5
  // Dunn_bad = 1/5 = 0.2
  auto prob_bad = make_4point_problem();
  prob_bad.clusters_ind = { 0, 1, 0, 1 };
  prob_bad.centroids_ind = { 0, 1 };
  double dunn_bad = scores::dunn(prob_bad);

  REQUIRE(dunn_good > dunn_bad);
}

// ---------------------------------------------------------------------------
// Davies-Bouldin Index tests
// ---------------------------------------------------------------------------
TEST_CASE("Davies-Bouldin Index: known 4-point problem", "[scores][dbi]")
{
  // S_0 = (d(0,0) + d(1,0)) / 2 = 0.5, S_1 = (d(2,2) + d(3,2)) / 2 = 0.5,
  // M_01 = d(0,2) = 5, R_01 = (0.5 + 0.5) / 5 = 0.2, DBI = (0.2 + 0.2) / 2 = 0.2.
  auto prob = make_4point_problem();
  REQUIRE_THAT(scores::davies_bouldin(prob), WithinAbs(0.2, 1e-12));
}

TEST_CASE("Davies-Bouldin Index: mixed clusters score worse than separated ones", "[scores][dbi]")
{
  // Clusters {0,2} and {1,3}, medoids 0 and 1: S_0 = S_1 = (0 + 5) / 2 = 2.5,
  // M_01 = d(0,1) = 1, R_01 = 5 / 1 = 5, DBI = 5 (lower is better; separated: 0.2).
  auto prob = make_4point_problem();
  prob.clusters_ind = { 0, 1, 0, 1 };
  prob.centroids_ind = { 0, 1 };
  REQUIRE_THAT(scores::davies_bouldin(prob), WithinAbs(5.0, 1e-12));
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
  REQUIRE_THROWS_AS(scores::inertia(prob), dtwc::InvalidInput);
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
  double ch = scores::calinski_harabasz(prob);

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
  REQUIRE_THROWS_AS(scores::calinski_harabasz(prob), dtwc::InvalidInput);
}

TEST_CASE("Calinski-Harabasz Index: throws with 1 cluster", "[scores][ch]")
{
  std::vector<std::vector<data_t>> vecs = { { 1.0 }, { 2.0 } };
  std::vector<std::string> names = { "a", "b" };
  Data data(std::move(vecs), std::move(names));
  Problem prob("one_cluster");
  prob.set_data(std::move(data));
  prob.set_n_clusters(1);
  prob.clusters_ind = { 0, 0 };
  prob.centroids_ind = { 0 };
  REQUIRE_THROWS_AS(scores::calinski_harabasz(prob), dtwc::InvalidInput);
}

TEST_CASE("Calinski-Harabasz Index: better clustering has higher CH", "[scores][ch]")
{
  // Well-separated clusters
  auto prob_good = make_4point_problem();
  double ch_good = scores::calinski_harabasz(prob_good);

  // Bad clustering: mix the points across clusters
  // cluster 0: {0,2}, cluster 1: {1,3} — inter-cluster distances are small
  auto prob_bad = make_4point_problem();
  prob_bad.clusters_ind = { 0, 1, 0, 1 };
  prob_bad.centroids_ind = { 0, 1 };
  double ch_bad = scores::calinski_harabasz(prob_bad);

  // Better clustering should have higher CH
  REQUIRE(ch_good > ch_bad);
}

// ---------------------------------------------------------------------------
// Adjusted Rand Index tests
// ---------------------------------------------------------------------------
TEST_CASE("ARI: perfect agreement", "[scores][ari]")
{
  std::vector<index_t> labels = { 0, 0, 1, 1 };
  double ari = scores::adjusted_rand(labels, labels);
  REQUIRE_THAT(ari, WithinAbs(1.0, 1e-12));
}

TEST_CASE("ARI: permuted labels still gives 1.0", "[scores][ari]")
{
  // {0,0,1,1} and {1,1,0,0} are equivalent clusterings (permutation invariant)
  std::vector<index_t> true_labels = { 0, 0, 1, 1 };
  std::vector<index_t> pred_labels = { 1, 1, 0, 0 };
  double ari = scores::adjusted_rand(true_labels, pred_labels);
  REQUIRE_THAT(ari, WithinAbs(1.0, 1e-12));
}

TEST_CASE("ARI: low agreement gives near-zero ARI", "[scores][ari]")
{
  // true={0,0,0,1,1,1}, pred={0,1,0,1,0,1} — alternating, very poor agreement
  std::vector<index_t> true_labels = { 0, 0, 0, 1, 1, 1 };
  std::vector<index_t> pred_labels = { 0, 1, 0, 1, 0, 1 };
  double ari = scores::adjusted_rand(true_labels, pred_labels);
  // Should be close to 0 (or even negative)
  REQUIRE(ari < 0.1);
}

TEST_CASE("ARI: throws on size mismatch", "[scores][ari]")
{
  std::vector<index_t> a = { 0, 0, 1 };
  std::vector<index_t> b = { 0, 1 };
  REQUIRE_THROWS_AS(scores::adjusted_rand(a, b), InvalidInput);
}

TEST_CASE("ARI: 6-point two-cluster known result", "[scores][ari]")
{
  // Perfect match
  std::vector<index_t> true_labels = { 0, 0, 0, 1, 1, 1 };
  std::vector<index_t> pred_labels = { 0, 0, 0, 1, 1, 1 };
  REQUIRE_THAT(scores::adjusted_rand(true_labels, pred_labels), WithinAbs(1.0, 1e-12));
}

// ---------------------------------------------------------------------------
// Normalized Mutual Information tests
// ---------------------------------------------------------------------------
TEST_CASE("NMI: perfect agreement", "[scores][nmi]")
{
  std::vector<index_t> labels = { 0, 0, 1, 1 };
  double nmi = scores::normalized_mutual_info(labels, labels);
  REQUIRE_THAT(nmi, WithinAbs(1.0, 1e-12));
}

TEST_CASE("NMI: permuted labels gives 1.0", "[scores][nmi]")
{
  std::vector<index_t> true_labels = { 0, 0, 1, 1 };
  std::vector<index_t> pred_labels = { 1, 1, 0, 0 };
  double nmi = scores::normalized_mutual_info(true_labels, pred_labels);
  REQUIRE_THAT(nmi, WithinAbs(1.0, 1e-12));
}

TEST_CASE("NMI: low agreement gives low NMI", "[scores][nmi]")
{
  std::vector<index_t> true_labels = { 0, 0, 0, 1, 1, 1 };
  std::vector<index_t> pred_labels = { 0, 1, 0, 1, 0, 1 };
  double nmi = scores::normalized_mutual_info(true_labels, pred_labels);
  // Should be well below 1.0
  REQUIRE(nmi < 0.5);
  REQUIRE(nmi >= 0.0);
}

TEST_CASE("NMI: throws on size mismatch", "[scores][nmi]")
{
  std::vector<index_t> a = { 0, 0, 1 };
  std::vector<index_t> b = { 0, 1 };
  REQUIRE_THROWS_AS(scores::normalized_mutual_info(a, b), InvalidInput);
}

TEST_CASE("NMI: value is in [0, 1] for all test cases", "[scores][nmi]")
{
  // Various clusterings
  std::vector<std::pair<std::vector<index_t>, std::vector<index_t>>> cases = {
    { { 0, 0, 1, 1 }, { 0, 0, 1, 1 } },
    { { 0, 0, 1, 1 }, { 1, 1, 0, 0 } },
    { { 0, 1, 2, 0 }, { 0, 0, 1, 1 } },
    { { 0, 0, 0, 1, 1, 1 }, { 0, 1, 0, 1, 0, 1 } },
  };
  for (auto &[t, p] : cases) {
    double nmi = scores::normalized_mutual_info(t, p);
    REQUIRE(nmi >= 0.0);
    REQUIRE(nmi <= 1.0 + 1e-10);
  }
}

// ---------------------------------------------------------------------------
// Hand-computed partitions. Constant series of length L (or series of length 1)
// make a DTW distance a diagonal L1 sum: d({a,a,a}, {b,b,b}) = 3|a - b| and
// d({a}, {b}) = |a - b|. Every expected value below is derived from the
// definition of the score, not from the implementation.
// ---------------------------------------------------------------------------

/// A Problem over `series` clustered by hand: series i is in cluster labels[i]
/// and cluster c has medoid medoids[c], so there are medoids.size() clusters.
static Problem make_clustered(std::vector<std::vector<data_t>> series,
                              std::vector<index_t> labels,
                              std::vector<index_t> medoids)
{
  std::vector<std::string> names;
  for (std::size_t i = 0; i < series.size(); ++i)
    names.push_back("s" + std::to_string(i));

  Problem prob("scores_by_hand");
  prob.set_data(Data(std::move(series), std::move(names)));
  prob.set_n_clusters(static_cast<int>(medoids.size()));
  prob.clusters_ind = std::move(labels);
  prob.centroids_ind = std::move(medoids);
  return prob;
}

// ---------------------------------------------------------------------------
// Silhouette: s(i) = (b - a) / max(a, b), a = mean distance to the |C_i| - 1
// peers, b = the smallest mean distance to another cluster (Rousseeuw 1987).
// ---------------------------------------------------------------------------
TEST_CASE("Silhouette: hand-computed values for a 4-point, 2-cluster case", "[scores][silhouette]")
{
  // Series {0,0,0} {1,1,1} {10,10,10} {11,11,11}; clusters {0,1} and {2,3}.
  //   a(0) = d(0,1) = 3,  b(0) = (d(0,2) + d(0,3)) / 2 = (30 + 33) / 2 = 31.5
  //   a(1) = d(1,0) = 3,  b(1) = (d(1,2) + d(1,3)) / 2 = (27 + 30) / 2 = 28.5
  // and points 2, 3 mirror points 1, 0.
  const std::vector<std::vector<data_t>> series = {
    { 0, 0, 0 }, { 1, 1, 1 }, { 10, 10, 10 }, { 11, 11, 11 } };
  const double s_outer = (31.5 - 3.0) / 31.5;
  const double s_inner = (28.5 - 3.0) / 28.5;

  SECTION("clusters {0,1} and {2,3}")
  {
    auto prob = make_clustered(series, { 0, 0, 1, 1 }, { 0, 2 });
    const auto sil = scores::silhouette(prob);
    REQUIRE(sil.size() == 4);
    REQUIRE_THAT(sil[0], WithinAbs(s_outer, 1e-10));
    REQUIRE_THAT(sil[1], WithinAbs(s_inner, 1e-10));
    REQUIRE_THAT(sil[2], WithinAbs(s_inner, 1e-10));
    REQUIRE_THAT(sil[3], WithinAbs(s_outer, 1e-10));
  }

  SECTION("the same partition under swapped cluster labels scores the same")
  {
    auto prob = make_clustered(series, { 1, 1, 0, 0 }, { 2, 0 });
    const auto sil = scores::silhouette(prob);
    REQUIRE(sil.size() == 4);
    REQUIRE_THAT(sil[0], WithinAbs(s_outer, 1e-10));
    REQUIRE_THAT(sil[1], WithinAbs(s_inner, 1e-10));
    REQUIRE_THAT(sil[2], WithinAbs(s_inner, 1e-10));
    REQUIRE_THAT(sil[3], WithinAbs(s_outer, 1e-10));
  }
}

TEST_CASE("Silhouette: b(i) is the nearest other cluster, not the mean over all others",
          "[scores][silhouette]")
{
  // Length-1 series 0 2 | 10 12 | 100 102 in three clusters; every a(i) = 2.
  //   point 0: b = min((10 + 12) / 2, (100 + 102) / 2) = 11,  s = 9 / 11
  //   point 1: b = min((8 + 10) / 2, (98 + 100) / 2)   = 9,   s = 7 / 9
  //   point 2: b = min((10 + 8) / 2, (90 + 92) / 2)    = 9,   s = 7 / 9
  //   point 3: b = min((12 + 10) / 2, (88 + 90) / 2)   = 11,  s = 9 / 11
  //   point 4: b = min((90 + 88) / 2, (100 + 98) / 2)  = 89,  s = 87 / 89
  //   point 5: b = min((92 + 90) / 2, (102 + 100) / 2) = 91,  s = 89 / 91
  // A mean over all other points would give b(0) = 56.
  auto prob = make_clustered({ { 0 }, { 2 }, { 10 }, { 12 }, { 100 }, { 102 } },
                             { 0, 0, 1, 1, 2, 2 }, { 0, 2, 4 });
  const auto sil = scores::silhouette(prob);
  REQUIRE(sil.size() == 6);
  const double expected[] = { 9.0 / 11, 7.0 / 9, 7.0 / 9, 9.0 / 11, 87.0 / 89, 89.0 / 91 };
  for (std::size_t i = 0; i < sil.size(); ++i) {
    INFO("point " << i);
    REQUIRE_THAT(sil[i], WithinAbs(expected[i], 1e-12));
  }
}

TEST_CASE("Silhouette: a singleton cluster scores 0 and does not disturb the others",
          "[scores][silhouette]")
{
  // {50,50,50} is alone in cluster 0: s = 0 by definition. {1,1,1} twice in
  // cluster 1 has a = 0 and b = 147 > 0, so s = 1.
  auto prob = make_clustered({ { 50, 50, 50 }, { 1, 1, 1 }, { 1, 1, 1 } }, { 0, 1, 1 }, { 0, 1 });
  const auto sil = scores::silhouette(prob);
  REQUIRE(sil.size() == 3);
  REQUIRE_THAT(sil[0], WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(sil[1], WithinAbs(1.0, 1e-10));
  REQUIRE_THAT(sil[2], WithinAbs(1.0, 1e-10));
}

TEST_CASE("Silhouette: a misassigned point scores -1 and pulls its new cluster-mates down",
          "[scores][silhouette]")
{
  // {100,100,100} (point 2) is put with the two {0,0,0}; its true twins are
  // points 3 and 4. d({0,0,0}, {100,100,100}) = 300.
  //   point 2: a = (300 + 300) / 2 = 300, b = 0,          s = (0 - 300) / 300 = -1
  //   points 0, 1: a = (0 + 300) / 2 = 150, b = 300,      s = 150 / 300 = 0.5
  //   points 3, 4: a = 0, b = (300 + 300 + 0) / 3 = 200,  s = 1
  auto prob = make_clustered({ { 0, 0, 0 }, { 0, 0, 0 }, { 100, 100, 100 },
                               { 100, 100, 100 }, { 100, 100, 100 } },
                             { 0, 0, 0, 1, 1 }, { 0, 3 });
  const auto sil = scores::silhouette(prob);
  REQUIRE(sil.size() == 5);
  const double expected[] = { 0.5, 0.5, -1.0, 1.0, 1.0 };
  for (std::size_t i = 0; i < sil.size(); ++i) {
    INFO("point " << i);
    REQUIRE_THAT(sil[i], WithinAbs(expected[i], 1e-12));
  }
}

TEST_CASE("Silhouette: one realised cluster throws instead of reporting about +1",
          "[scores][silhouette][degenerate]")
{
  // With one cluster b(i) has no candidate: a min left at DBL_MAX would give
  // s(i) = (MAX - a) / MAX ~ 1, a "perfect" score for three visibly different series.
  auto prob = make_clustered({ { 0, 0, 0 }, { 5, 5, 5 }, { 50, 50, 50 } }, { 0, 0, 0 }, { 0 });
  REQUIRE_THROWS_AS(scores::silhouette(prob), InvalidInput);
}

TEST_CASE("Silhouette: a = b = 0 scores 0, not NaN", "[scores][silhouette][degenerate]")
{
  // Four identical series in two clusters: every distance is 0, so s = 0/0.
  // Rousseeuw's convention is 0; a NaN would poison any downstream mean.
  auto prob = make_clustered({ { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 }, { 1, 1, 1 } },
                             { 0, 0, 1, 1 }, { 0, 2 });
  const auto sil = scores::silhouette(prob);
  REQUIRE(sil.size() == 4);
  for (const double s : sil) {
    REQUIRE_FALSE(std::isnan(s));
    REQUIRE_THAT(s, WithinAbs(0.0, 1e-15));
  }
}

// ---------------------------------------------------------------------------
// Davies-Bouldin (IEEE TPAMI 1(2):224-227, 1979): R_ij = (S_i + S_j) / M_ij,
// DB = (1/k) sum_i max_{j != i} R_ij. Axiom (3): R_ij = 0 iff S_i = S_j = 0, and
// R_ij decreases strictly in M_ij, so M_ij -> 0 with spread gives +inf (the worst
// value), never a skipped pair.
// ---------------------------------------------------------------------------
TEST_CASE("Davies-Bouldin Index: coincident medoids with real spread give +inf, not 0",
          "[scores][dbi][degenerate]")
{
  // Medoids 0 and 2 are the same series (M_01 = 0) while both clusters spread
  // (S > 0). Skipping the pair would leave the maximum at 0 and report the worst
  // configuration as a perfect DBI.
  auto prob = make_clustered({ { 0, 0, 0 }, { 8, 8, 8 }, { 0, 0, 0 }, { 9, 9, 9 } },
                             { 0, 0, 1, 1 }, { 0, 2 });
  prob.fill_distance_matrix();
  REQUIRE(prob.dist_by_ind(0, 2) == 0.0);
  REQUIRE(prob.dist_by_ind(1, 0) > 0.0);

  const double dbi = scores::davies_bouldin(prob);
  REQUIRE(std::isinf(dbi));
  REQUIRE(dbi > 0.0);
}

TEST_CASE("Davies-Bouldin Index: coincident medoids with zero spread give 0 (axiom 3)",
          "[scores][dbi][degenerate]")
{
  // M_ij = 0 and S_i = S_j = 0 is 0/0; axiom (3) fixes the limit at 0.
  auto prob = make_clustered({ { 2, 2, 2 }, { 2, 2, 2 }, { 2, 2, 2 }, { 2, 2, 2 } },
                             { 0, 0, 1, 1 }, { 0, 2 });
  const double dbi = scores::davies_bouldin(prob);
  REQUIRE_FALSE(std::isnan(dbi));
  REQUIRE_THAT(dbi, WithinAbs(0.0, 1e-15));
}

TEST_CASE("Scores use the realised label set, not the declared n_clusters", "[scores][degenerate]")
{
  // n_clusters() says 3 but the labels realise only cluster 0. Dunn once counted
  // the declared clusters, passed its Nc >= 2 guard, found no inter-cluster pair
  // and returned DBL_MAX / max_intra as an ordinary finite number.
  auto prob = make_clustered({ { 0, 0, 0 }, { 1, 1, 1 }, { 2, 2, 2 } }, { 0, 0, 0 }, { 0, 0, 0 });
  REQUIRE(prob.n_clusters() == 3);
  REQUIRE_THROWS_AS(scores::dunn(prob), InvalidInput);
  REQUIRE_THROWS_AS(scores::davies_bouldin(prob), InvalidInput);
  REQUIRE_THROWS_AS(scores::silhouette(prob), InvalidInput);

  SECTION("an empty declared cluster does not change a realised 2-cluster score")
  {
    // The same partition declared with k = 2 and with k = 3 (cluster 2 empty)
    // must score digit-identically; the 1/k normalisers used to differ.
    const std::vector<std::vector<data_t>> series = { { 0, 0 }, { 0, 0 }, { 10, 10 }, { 10, 10 } };
    auto tight = make_clustered(series, { 0, 0, 1, 1 }, { 0, 2 });
    auto padded = make_clustered(series, { 0, 0, 1, 1 }, { 0, 2, 2 });

    REQUIRE(scores::davies_bouldin(padded) == scores::davies_bouldin(tight));
    REQUIRE(scores::dunn(padded) == scores::dunn(tight));
    REQUIRE(scores::calinski_harabasz(padded) == scores::calinski_harabasz(tight));
    REQUIRE(scores::silhouette(padded) == scores::silhouette(tight));
  }
}

// ---------------------------------------------------------------------------
// Dunn, Calinski-Harabasz and inertia beyond the known 4-point problem.
// ---------------------------------------------------------------------------
TEST_CASE("Dunn Index: a zero intra-cluster diameter gives +inf", "[scores][dunn]")
{
  // Both clusters are tight (max_intra = 0) and far apart: min_inter / 0+ = +inf.
  auto prob = make_clustered({ { 0, 0, 0 }, { 0, 0, 0 }, { 100, 100, 100 }, { 100, 100, 100 } },
                             { 0, 0, 1, 1 }, { 0, 2 });
  const double dunn = scores::dunn(prob);
  REQUIRE(std::isinf(dunn));
  REQUIRE(dunn > 0.0);
}

TEST_CASE("Calinski-Harabasz Index: as many clusters as points throws", "[scores][ch]")
{
  // N - k = 0 would divide by zero in W / (N - k).
  auto prob = make_clustered({ { 0.0, 1.0 }, { 2.0, 3.0 }, { 4.0, 5.0 } }, { 0, 1, 2 }, { 0, 1, 2 });
  REQUIRE_THROWS_AS(scores::calinski_harabasz(prob), InvalidInput);
}

TEST_CASE("Inertia: sums the distance to the assigned medoid, not to the nearest one",
          "[scores][inertia]")
{
  // Length-1 series 0 1 100 101 with medoids 0 and 2.
  //   labels {0,0,1,1}: 0 + 1 + 0 + 1 = 2
  //   labels {0,1,1,1}: 0 + d(1,2) + 0 + 1 = 0 + 99 + 0 + 1 = 100, though
  //   series 1 is nearer to medoid 0.
  const std::vector<std::vector<data_t>> series = { { 0.0 }, { 1.0 }, { 100.0 }, { 101.0 } };
  auto good = make_clustered(series, { 0, 0, 1, 1 }, { 0, 2 });
  auto bad = make_clustered(series, { 0, 1, 1, 1 }, { 0, 2 });
  REQUIRE_THAT(scores::inertia(good), WithinAbs(2.0, 1e-10));
  REQUIRE_THAT(scores::inertia(bad), WithinAbs(100.0, 1e-10));
}

// ---------------------------------------------------------------------------
// ARI and NMI on label vectors.
// ---------------------------------------------------------------------------
TEST_CASE("ARI: a partition and its fully split image score -0.5", "[scores][ari]")
{
  // labels {0,0,1,1}, pred {0,1,0,1}: every pair together in one is apart in the other.
  // sum C(n_ij,2) = 0, sum C(a_i,2) = sum C(b_j,2) = 2, C(4,2) = 6, so
  // expected = 2*2/6, max = 2 and ARI = (0 - 2/3) / (2 - 2/3) = -1/2.
  const std::vector<index_t> labels = { 0, 0, 1, 1 };
  const std::vector<index_t> pred = { 0, 1, 0, 1 };
  REQUIRE_THAT(scores::adjusted_rand(labels, pred), WithinAbs(-0.5, 1e-10));
}

TEST_CASE("ARI: hand-computed 6-point case, in either argument order", "[scores][ari]")
{
  // labels {0,0,0,1,1,1}, pred {0,0,1,1,1,1}: contingency table [[2,1],[0,3]].
  //   sum C(n_ij,2) = 1 + 0 + 0 + 3 = 4, sum C(a_i,2) = 3 + 3 = 6,
  //   sum C(b_j,2) = 1 + 6 = 7, C(6,2) = 15
  //   expected = 6 * 7 / 15 = 2.8, max = (6 + 7) / 2 = 6.5
  //   ARI = (4 - 2.8) / (6.5 - 2.8) = 1.2 / 3.7
  const std::vector<index_t> labels = { 0, 0, 0, 1, 1, 1 };
  const std::vector<index_t> pred = { 0, 0, 1, 1, 1, 1 };
  REQUIRE_THAT(scores::adjusted_rand(labels, pred), WithinAbs(1.2 / 3.7, 1e-10));
  REQUIRE_THAT(scores::adjusted_rand(pred, labels), WithinAbs(1.2 / 3.7, 1e-10));
}

TEST_CASE("NMI: hand-computed 6-point case, in either argument order", "[scores][nmi]")
{
  // The ARI case above: labels {0,0,0,1,1,1}, pred {0,0,1,1,1,1}, contingency
  // table [[2,1],[0,3]] / 6, row marginals (1/2, 1/2), column marginals (1/3, 2/3).
  //   MI = (1/3) ln 2 + (1/6) ln(1/2) + (1/2) ln(3/2) = (1/6) ln 2 + (1/2) ln(3/2)
  //   H(labels) = ln 2,  H(pred) = -(1/3) ln(1/3) - (2/3) ln(2/3)
  //   NMI = MI / ((H(labels) + H(pred)) / 2) = 0.4787040...
  // The entropies differ, so this also pins the arithmetic-mean normaliser
  // (the geometric mean would give 0.4791388..., the larger entropy 0.4591479...).
  const std::vector<index_t> labels = { 0, 0, 0, 1, 1, 1 };
  const std::vector<index_t> pred = { 0, 0, 1, 1, 1, 1 };
  const double mi = std::log(2.0) / 6.0 + 0.5 * std::log(1.5);
  const double h_labels = std::log(2.0);
  const double h_pred = -(1.0 / 3.0) * std::log(1.0 / 3.0) - (2.0 / 3.0) * std::log(2.0 / 3.0);
  const double expected = mi / (0.5 * (h_labels + h_pred));
  REQUIRE_THAT(scores::normalized_mutual_info(labels, pred), WithinAbs(expected, 1e-12));
  REQUIRE_THAT(scores::normalized_mutual_info(pred, labels), WithinAbs(expected, 1e-12));
}

TEST_CASE("ARI and NMI: two constant labelings agree completely, not 0/0", "[scores][ari][nmi]")
{
  // One cluster on each side makes both ARI's denominator and NMI's entropies
  // zero; n = 1 also has C(n,2) = 0 pairs.
  for (const std::size_t n : { std::size_t{ 4 }, std::size_t{ 2 }, std::size_t{ 1 } }) {
    CAPTURE(n);
    const std::vector<index_t> constant(n, 0);
    REQUIRE_THAT(scores::adjusted_rand(constant, constant), WithinAbs(1.0, 1e-12));
    REQUIRE_THAT(scores::normalized_mutual_info(constant, constant), WithinAbs(1.0, 1e-12));
  }
}

TEST_CASE("ARI and NMI count labels that agree in their low 32 bits apart",
          "[scores][ari][nmi][index_t]")
{
  // A relabelling scores exactly 1. Labels 0 and 2^32 share their low 32 bits
  // and -1 is negative: a key packing two labels into 32-bit halves merges them.
  constexpr index_t high = index_t{ 1 } << 32;
  const std::vector<index_t> truth = { 0, 0, 1, 1, 2, 2 };
  const std::vector<index_t> pred = { 0, 0, high, high, -1, -1 };
  REQUIRE_THAT(scores::adjusted_rand(truth, pred), WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(scores::adjusted_rand(pred, truth), WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(scores::normalized_mutual_info(truth, pred), WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(scores::normalized_mutual_info(pred, truth), WithinAbs(1.0, 1e-12));
}
