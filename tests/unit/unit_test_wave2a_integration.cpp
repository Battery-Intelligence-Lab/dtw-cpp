/**
 * @file unit_test_wave2a_integration.cpp
 * @brief Adversarial integration tests for all Wave 2A features.
 *
 * @details Wave 2A added:
 *   1. Deferred dense allocation (Problem::set_data doesn't allocate O(N^2))
 *   2. FastCLARA bugfixes (ndim, missing_strategy propagation) + improved sample size
 *   3. Shared medoid utilities (algorithms/detail/medoid_utils.hpp)
 *   4. Hierarchical clustering (single/complete/average, build_dendrogram + cut_dendrogram)
 *   5. CLARANS (experimental, budget-gated)
 *
 * All tests use Catch2 with MSVC/C++17.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#include <dtwc.hpp>
#include <algorithms/hierarchical.hpp>
#include <algorithms/clarans.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using Catch::Matchers::WithinAbs;
using namespace dtwc;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr double NaN_val = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------------
// Temporary output directory (avoids creating stray result CSVs)
// ---------------------------------------------------------------------------
static fs::path g_tmp_dir()
{
  static fs::path dir = fs::temp_directory_path() / "dtwc_wave2a_integration_test";
  static bool created = [] {
    fs::create_directories(dir);
    return true;
  }();
  (void)created;
  return dir;
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a univariate Problem from raw vectors.
static Problem make_problem_uv(std::vector<std::vector<double>> vecs, int Nc,
                                core::MissingStrategy ms = core::MissingStrategy::Error)
{
  std::vector<std::string> names;
  names.reserve(vecs.size());
  for (size_t i = 0; i < vecs.size(); ++i)
    names.push_back("s" + std::to_string(i));

  Data d(std::move(vecs), std::move(names));
  Problem prob("wave2a_uv");
  prob.set_data(std::move(d));
  prob.set_numberOfClusters(Nc);
  prob.missing_strategy = ms;
  prob.verbose = false;
  prob.output_folder = g_tmp_dir();
  return prob;
}

/// Build a multivariate Problem: ndim channels, each series has n_steps * ndim flat values.
static Problem make_problem_mv(int N, int n_steps, int ndim,
                                double cluster_sep = 50.0, int Nc = 2,
                                core::MissingStrategy ms = core::MissingStrategy::Error,
                                unsigned seed = 42)
{
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> noise(0.0, 1.0);

  std::vector<std::vector<double>> vecs;
  std::vector<std::string> names;
  vecs.reserve(N);
  names.reserve(N);

  for (int i = 0; i < N; ++i) {
    double base = (i % Nc) * cluster_sep;
    std::vector<double> flat;
    flat.reserve(n_steps * ndim);
    for (int t = 0; t < n_steps; ++t)
      for (int d = 0; d < ndim; ++d)
        flat.push_back(base + noise(rng));
    vecs.push_back(std::move(flat));
    names.push_back("mv_" + std::to_string(i));
  }

  Data data(std::move(vecs), std::move(names), static_cast<size_t>(ndim));
  Problem prob("wave2a_mv");
  prob.set_data(std::move(data));
  prob.set_numberOfClusters(Nc);
  prob.missing_strategy = ms;
  prob.verbose = false;
  prob.output_folder = g_tmp_dir();
  return prob;
}

/// Build a well-separated univariate problem: n_clusters groups, separation 100 units.
static Problem make_separated_problem(int n_per_cluster, int n_clusters, unsigned seed = 7)
{
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> noise(0.0, 0.5);

  std::vector<std::vector<double>> vecs;
  std::vector<std::string> names;

  for (int c = 0; c < n_clusters; ++c) {
    for (int i = 0; i < n_per_cluster; ++i) {
      double base = c * 100.0;
      std::vector<double> ts;
      for (int j = 0; j < 10; ++j)
        ts.push_back(base + noise(rng));
      vecs.push_back(std::move(ts));
      names.push_back("g" + std::to_string(c) + "_" + std::to_string(i));
    }
  }

  Data data(std::move(vecs), std::move(names));
  Problem prob("separated");
  prob.set_data(std::move(data));
  prob.verbose = false;
  prob.output_folder = g_tmp_dir();
  return prob;
}

// ===========================================================================
// Test 1: Deferred allocation smoke (N=5000)
// ===========================================================================
TEST_CASE("Wave2A: deferred allocation smoke — N=5000 matrix size==0 after set_data",
          "[wave2a][deferred]")
{
  Data data;
  data.p_vec.reserve(5000);
  data.p_names.reserve(5000);
  for (int i = 0; i < 5000; ++i) {
    data.p_vec.push_back({ static_cast<double>(i), static_cast<double>(i + 1) });
    data.p_names.push_back("s" + std::to_string(i));
  }

  Problem prob;
  prob.set_data(std::move(data));
  prob.verbose = false;
  prob.output_folder = g_tmp_dir();

  // Dense matrix must NOT be allocated yet (deferred).
  REQUIRE(prob.distance_matrix().size() == 0);

  // distByInd works on-demand (lazy compute).
  double d01 = prob.distByInd(0, 1);
  double d02 = prob.distByInd(0, 2);
  REQUIRE(d01 >= 0.0);
  REQUIRE(d02 >= 0.0);
  REQUIRE(std::isfinite(d01));
  REQUIRE(std::isfinite(d02));

  // Verify symmetry of cached distances.
  double d10 = prob.distByInd(1, 0);
  REQUIRE_THAT(d01, WithinAbs(d10, 1e-12));

  // We asked for 3 unique off-diagonal pairs: (0,1),(0,2),(1,0).
  // The matrix should have some cached entries but NOT the full N*(N-1)/2.
  size_t computed = prob.distance_matrix().count_computed();
  // N^2 / 2 = ~12.5M; computed must be tiny by comparison.
  REQUIRE(computed < 100); // only the few pairs we explicitly queried

  // Now fill the full matrix — should work on a much smaller problem to keep
  // the test fast. We recreate a tiny problem for the fill step.
  Data small_data;
  for (int i = 0; i < 5; ++i) {
    small_data.p_vec.push_back({ static_cast<double>(i) });
    small_data.p_names.push_back("t" + std::to_string(i));
  }
  Problem small_prob;
  small_prob.set_data(std::move(small_data));
  small_prob.verbose = false;
  REQUIRE(small_prob.distance_matrix().size() == 0);

  small_prob.fillDistanceMatrix();
  REQUIRE(small_prob.distance_matrix().size() == 5);
  REQUIRE(small_prob.isDistanceMatrixFilled());
}

// ===========================================================================
// Test 2: FastCLARA with ndim=2
// ===========================================================================
TEST_CASE("Wave2A: FastCLARA with multivariate data ndim=2 produces valid result",
          "[wave2a][fast_clara][mv]")
{
  constexpr int N = 60;
  constexpr int n_steps = 15;
  constexpr int ndim = 2;
  constexpr int k = 3;

  Problem prob = make_problem_mv(N, n_steps, ndim, 100.0, k);

  algorithms::CLARAOptions opts;
  opts.n_clusters  = k;
  opts.n_samples   = 3;
  opts.random_seed = 42;

  core::ClusteringResult result;
  REQUIRE_NOTHROW(result = algorithms::fast_clara(prob, opts));

  // Labels must be in [0, k).
  REQUIRE(result.labels.size() == static_cast<size_t>(N));
  for (int l : result.labels) {
    REQUIRE(l >= 0);
    REQUIRE(l < k);
  }

  // Medoids must be distinct and valid.
  REQUIRE(result.medoid_indices.size() == static_cast<size_t>(k));
  std::set<int> unique_m(result.medoid_indices.begin(), result.medoid_indices.end());
  REQUIRE(unique_m.size() == static_cast<size_t>(k));
  for (int m : result.medoid_indices) {
    REQUIRE(m >= 0);
    REQUIRE(m < N);
  }

  // Cost must be non-negative and finite.
  REQUIRE(result.total_cost >= 0.0);
  REQUIRE(std::isfinite(result.total_cost));
}

// ===========================================================================
// Test 3: FastCLARA with MissingStrategy::ZeroCost + NaN data must not crash
// ===========================================================================
TEST_CASE("Wave2A: FastCLARA with ZeroCost strategy on NaN data does not crash",
          "[wave2a][fast_clara][nan][zerocost]")
{
  // Build 30 series of length 10 with ~20% NaN values.
  std::mt19937_64 rng(55);
  std::normal_distribution<double> gauss(0.0, 1.0);
  std::uniform_real_distribution<double> uni(0.0, 1.0);

  constexpr int N = 30;
  constexpr int L = 10;
  constexpr int k = 3;

  std::vector<std::vector<double>> vecs(N, std::vector<double>(L));
  for (auto &v : vecs)
    for (auto &x : v)
      x = (uni(rng) < 0.20) ? NaN_val : gauss(rng);

  Problem prob = make_problem_uv(vecs, k, core::MissingStrategy::ZeroCost);

  algorithms::CLARAOptions opts;
  opts.n_clusters  = k;
  opts.n_samples   = 2;
  opts.random_seed = 99;

  core::ClusteringResult result;
  REQUIRE_NOTHROW(result = algorithms::fast_clara(prob, opts));

  REQUIRE(result.labels.size() == static_cast<size_t>(N));
  REQUIRE(result.medoid_indices.size() == static_cast<size_t>(k));
  REQUIRE(std::isfinite(result.total_cost));
  REQUIRE(result.total_cost >= 0.0);
}

// ===========================================================================
// Test 4: Hierarchical single/complete/average — N=20, all 3 linkages
//   Each dendrogram must have N-1 merges; cut(2) must produce valid labels.
// ===========================================================================
TEST_CASE("Wave2A: hierarchical all 3 linkages on 20 points",
          "[wave2a][hierarchical][linkage]")
{
  constexpr int N = 20;

  // Simple data: two tight groups [0..9] and [100..109].
  std::vector<std::vector<double>> vecs;
  for (int i = 0; i < 10; ++i)
    vecs.push_back({ static_cast<double>(i) });
  for (int i = 0; i < 10; ++i)
    vecs.push_back({ 100.0 + static_cast<double>(i) });

  Problem prob = make_problem_uv(vecs, 2);
  prob.fillDistanceMatrix();

  for (auto linkage : { algorithms::Linkage::Single,
                         algorithms::Linkage::Complete,
                         algorithms::Linkage::Average }) {
    algorithms::HierarchicalOptions opts;
    opts.linkage    = linkage;
    opts.max_points = 200; // generous guard

    algorithms::Dendrogram dend;
    REQUIRE_NOTHROW(dend = algorithms::build_dendrogram(prob, opts));

    REQUIRE(dend.n_points == N);
    REQUIRE(dend.merges.size() == static_cast<size_t>(N - 1));

    // All merge distances must be non-negative.
    for (const auto &step : dend.merges)
      REQUIRE(step.distance >= 0.0);

    // New sizes must be between 2 and N.
    for (const auto &step : dend.merges) {
      REQUIRE(step.new_size >= 2);
      REQUIRE(step.new_size <= N);
    }

    // cut(2) must produce valid labels.
    core::ClusteringResult res2;
    REQUIRE_NOTHROW(res2 = algorithms::cut_dendrogram(dend, prob, 2));

    REQUIRE(res2.labels.size() == static_cast<size_t>(N));
    REQUIRE(res2.medoid_indices.size() == 2u);

    for (int l : res2.labels) {
      REQUIRE(l >= 0);
      REQUIRE(l < 2);
    }

    // Both clusters must be non-empty.
    int cnt0 = static_cast<int>(std::count(res2.labels.begin(), res2.labels.end(), 0));
    int cnt1 = static_cast<int>(std::count(res2.labels.begin(), res2.labels.end(), 1));
    REQUIRE(cnt0 > 0);
    REQUIRE(cnt1 > 0);
  }
}

// ===========================================================================
// Test 5: Hierarchical cut consistency
//   cut(k) labels cover all N points, medoids are within-cluster points.
// ===========================================================================
TEST_CASE("Wave2A: hierarchical cut consistency — labels cover all points, medoids in-cluster",
          "[wave2a][hierarchical][cut_consistency]")
{
  // Build a simple 10-point problem.
  std::vector<std::vector<double>> vecs;
  for (int i = 0; i < 10; ++i)
    vecs.push_back({ static_cast<double>(i * 5) });

  Problem prob = make_problem_uv(vecs, 3);
  prob.fillDistanceMatrix();

  algorithms::Dendrogram dend = algorithms::build_dendrogram(
    prob, { algorithms::Linkage::Average, 200 });

  for (int k = 1; k <= 10; ++k) {
    core::ClusteringResult res;
    REQUIRE_NOTHROW(res = algorithms::cut_dendrogram(dend, prob, k));

    REQUIRE(res.labels.size() == 10u);
    REQUIRE(res.medoid_indices.size() == static_cast<size_t>(k));

    // Labels must cover [0, k).
    std::set<int> seen_labels(res.labels.begin(), res.labels.end());
    REQUIRE(static_cast<int>(seen_labels.size()) == k);
    REQUIRE(*seen_labels.begin() == 0);
    REQUIRE(*seen_labels.rbegin() == k - 1);

    // Each medoid must belong to its claimed cluster.
    for (int c = 0; c < k; ++c) {
      int med = res.medoid_indices[c];
      REQUIRE(med >= 0);
      REQUIRE(med < 10);
      REQUIRE(res.labels[med] == c); // medoid is in its own cluster
    }

    // No two clusters share a medoid.
    std::set<int> unique_meds(res.medoid_indices.begin(), res.medoid_indices.end());
    REQUIRE(unique_meds.size() == static_cast<size_t>(k));
  }
}

// ===========================================================================
// Test 6: Hierarchical max_points guard — N=100, max_points=50 must throw
// ===========================================================================
TEST_CASE("Wave2A: hierarchical max_points guard throws when N > max_points",
          "[wave2a][hierarchical][guard]")
{
  Data data;
  for (int i = 0; i < 100; ++i) {
    data.p_vec.push_back({ static_cast<double>(i) });
    data.p_names.push_back("p" + std::to_string(i));
  }

  Problem prob;
  prob.set_data(std::move(data));
  prob.verbose = false;
  prob.fillDistanceMatrix();

  algorithms::HierarchicalOptions opts;
  opts.max_points = 50; // N=100 exceeds this

  REQUIRE_THROWS_AS(algorithms::build_dendrogram(prob, opts), std::runtime_error);
}

// ===========================================================================
// Test 7: CLARANS vs FastPAM quality on well-separated 30-point data
//   On well-separated data, CLARANS cost must be <= FastPAM cost * 1.5
//   (i.e., CLARANS should find a near-optimal solution).
// ===========================================================================
TEST_CASE("Wave2A: CLARANS vs FastPAM quality on well-separated 30-point data",
          "[wave2a][clarans][quality]")
{
  constexpr int n_per = 10;
  constexpr int k = 3;

  auto prob_pam    = make_separated_problem(n_per, k, 77);
  auto prob_claran = make_separated_problem(n_per, k, 77);

  // FastPAM (optimal baseline).
  auto pam_result = fast_pam(prob_pam, k);

  // CLARANS with multiple restarts.
  algorithms::CLARANSOptions opts;
  opts.n_clusters   = k;
  opts.num_local    = 5;
  opts.max_neighbor = -1; // auto
  opts.max_dtw_evals = -1; // no budget limit
  opts.random_seed  = 42;

  auto clarans_result = algorithms::clarans(prob_claran, opts);

  // Both must produce valid results.
  REQUIRE(pam_result.labels.size() == static_cast<size_t>(n_per * k));
  REQUIRE(clarans_result.labels.size() == static_cast<size_t>(n_per * k));

  // CLARANS cost should be reasonably close to FastPAM on well-separated data.
  // We allow 50% slack since CLARANS is randomized (not guaranteed optimal).
  REQUIRE(clarans_result.total_cost <= pam_result.total_cost * 1.5 + 1.0);

  // Basic sanity: labels and medoids valid.
  for (int l : clarans_result.labels) {
    REQUIRE(l >= 0);
    REQUIRE(l < k);
  }
  std::set<int> um(clarans_result.medoid_indices.begin(), clarans_result.medoid_indices.end());
  REQUIRE(um.size() == static_cast<size_t>(k));
}

// ===========================================================================
// Test 8: CLARANS budget enforcement
//   max_dtw_evals=100 must produce a valid result (not crash, not garbage).
// ===========================================================================
TEST_CASE("Wave2A: CLARANS budget enforcement — max_dtw_evals=100 gives valid result",
          "[wave2a][clarans][budget]")
{
  auto prob = make_separated_problem(10, 3, 13);
  const int N = static_cast<int>(prob.size()); // 30

  algorithms::CLARANSOptions opts;
  opts.n_clusters    = 3;
  opts.num_local     = 5;
  opts.max_dtw_evals = 100; // very tight budget
  opts.random_seed   = 7;

  core::ClusteringResult result;
  REQUIRE_NOTHROW(result = algorithms::clarans(prob, opts));

  REQUIRE(result.labels.size() == static_cast<size_t>(N));
  REQUIRE(result.medoid_indices.size() == 3u);
  REQUIRE(result.total_cost >= 0.0);
  REQUIRE(std::isfinite(result.total_cost));

  for (int l : result.labels) {
    REQUIRE(l >= 0);
    REQUIRE(l < 3);
  }

  // Medoids must be valid and distinct.
  std::set<int> um(result.medoid_indices.begin(), result.medoid_indices.end());
  REQUIRE(um.size() == 3u);
  for (int m : result.medoid_indices) {
    REQUIRE(m >= 0);
    REQUIRE(m < N);
  }
}

// ===========================================================================
// Test 9: Full pipeline — hierarchical → cut(3) → silhouette + Dunn + CH all finite
// ===========================================================================
TEST_CASE("Wave2A: full pipeline hierarchical→cut(3)→score metrics all finite",
          "[wave2a][pipeline][scores]")
{
  // 15 points in 3 well-separated clusters.
  constexpr int n_per = 5;
  constexpr int k = 3;

  std::vector<std::vector<double>> vecs;
  for (int c = 0; c < k; ++c)
    for (int i = 0; i < n_per; ++i)
      vecs.push_back({ c * 100.0 + i * 0.1, c * 100.0 + i * 0.2 });

  Problem prob = make_problem_uv(vecs, k);
  prob.fillDistanceMatrix();

  // Build dendrogram with average linkage.
  auto dend = algorithms::build_dendrogram(
    prob, { algorithms::Linkage::Average, 200 });

  REQUIRE(dend.n_points == n_per * k);
  REQUIRE(dend.merges.size() == static_cast<size_t>(n_per * k - 1));

  // Cut to k clusters.
  auto cr = algorithms::cut_dendrogram(dend, prob, k);
  REQUIRE(cr.labels.size() == static_cast<size_t>(n_per * k));
  REQUIRE(cr.medoid_indices.size() == static_cast<size_t>(k));

  // Inject labels + medoids into Problem for score computation.
  prob.set_numberOfClusters(k);
  prob.clusters_ind  = cr.labels;
  prob.centroids_ind = cr.medoid_indices;

  // Compute scores.
  auto sil = scores::silhouette(prob);
  double dunn = scores::dunnIndex(prob);
  double ch   = scores::calinskiHarabaszIndex(prob);

  // All scores must be finite.
  for (double s : sil)
    REQUIRE(std::isfinite(s));
  REQUIRE(std::isfinite(dunn));
  REQUIRE(std::isfinite(ch));

  // Silhouette values must be in [-1, 1].
  for (double s : sil) {
    REQUIRE(s >= -1.0 - 1e-9);
    REQUIRE(s <= 1.0 + 1e-9);
  }

  // Dunn index must be non-negative.
  REQUIRE(dunn >= 0.0);

  // CH index for well-separated data must be positive.
  REQUIRE(ch > 0.0);
}

// ===========================================================================
// Test 10: Deferred allocation + FastCLARA
//   Run FastCLARA on N=500; verify parent Problem's dense matrix was NOT
//   fully filled (count_computed < N*(N-1)/2).
// ===========================================================================
TEST_CASE("Wave2A: FastCLARA on N=500 does NOT fill parent dense matrix",
          "[wave2a][deferred][fast_clara][lazy]")
{
  constexpr int N = 500;
  constexpr int k = 4;

  // Build N series in 4 well-separated groups.
  std::vector<std::vector<double>> vecs;
  std::vector<std::string> names;
  for (int i = 0; i < N; ++i) {
    double base = (i % k) * 200.0;
    vecs.push_back({ base, base + 1.0, base + 2.0 });
    names.push_back("s" + std::to_string(i));
  }

  Data data(std::move(vecs), std::move(names));
  Problem prob("wave2a_deferred_clara");
  prob.set_data(std::move(data));
  prob.verbose = false;
  prob.output_folder = g_tmp_dir();

  // The matrix must start empty.
  REQUIRE(prob.distance_matrix().size() == 0);

  algorithms::CLARAOptions opts;
  opts.n_clusters  = k;
  opts.n_samples   = 3;
  opts.sample_size = 50; // sub-problem of 50 points, not full N
  opts.random_seed = 11;

  core::ClusteringResult result;
  REQUIRE_NOTHROW(result = algorithms::fast_clara(prob, opts));

  // Result must be valid.
  REQUIRE(result.labels.size() == static_cast<size_t>(N));
  REQUIRE(result.medoid_indices.size() == static_cast<size_t>(k));
  REQUIRE(result.total_cost >= 0.0);
  REQUIRE(std::isfinite(result.total_cost));

  // Parent distance matrix must NOT have been fully filled.
  // Full matrix would require N*(N-1)/2 = 124750 off-diagonal pairs.
  // CLARA only touches N*k (2000) for assignment, plus sub-problem distances.
  // We allow a generous 10x budget to tolerate any incidental caching.
  const size_t full_pairs = static_cast<size_t>(N) * (N - 1) / 2;
  const size_t computed   = prob.distance_matrix().count_computed();

  // Note: distance_matrix().size() == N only after distByInd triggers lazy resize.
  // CLARA calls distByInd on the *parent* prob for the N*k assignment step.
  // So some entries will be computed — but far fewer than N*(N-1)/2.
  INFO("Computed entries: " << computed << " / " << full_pairs * 2 << " (full matrix)");
  REQUIRE(computed < full_pairs / 5); // less than 20% of the full matrix
}

// ===========================================================================
// Test 11: Deferred allocation — distByInd after set_data is correct and cached
// ===========================================================================
TEST_CASE("Wave2A: deferred distByInd is cached after first call",
          "[wave2a][deferred][cache]")
{
  std::vector<std::vector<double>> vecs = {
    { 0.0, 1.0, 2.0 },
    { 10.0, 11.0, 12.0 },
    { 5.0, 6.0, 7.0 }
  };
  Problem prob = make_problem_uv(vecs, 2);

  // Matrix not yet allocated.
  REQUIRE(prob.distance_matrix().size() == 0);

  // First call triggers lazy computation.
  double d01 = prob.distByInd(0, 1);
  REQUIRE(std::isfinite(d01));
  REQUIRE(d01 > 0.0);

  // Symmetry.
  double d10 = prob.distByInd(1, 0);
  REQUIRE_THAT(d01, WithinAbs(d10, 1e-12));

  // Self-distance = 0.
  double d00 = prob.distByInd(0, 0);
  REQUIRE_THAT(d00, WithinAbs(0.0, 1e-12));

  // Triangle inequality: d(0,1) <= d(0,2) + d(2,1)
  double d02 = prob.distByInd(0, 2);
  double d21 = prob.distByInd(2, 1);
  REQUIRE(d01 <= d02 + d21 + 1e-9);
}

// ===========================================================================
// Test 12: FastCLARA respects missing_strategy propagation to sub-problems
//   This is one of the Wave 2A bugfixes. Verify that ZeroCost strategy is
//   correctly propagated: the sub-Problem used internally must not throw
//   when data has NaN values.
// ===========================================================================
TEST_CASE("Wave2A: FastCLARA propagates missing_strategy to sub-problems",
          "[wave2a][fast_clara][missing_strategy][propagation]")
{
  // 20 series, some with NaN, using ZeroCost.
  std::mt19937_64 rng(333);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::normal_distribution<double> gauss(0.0, 5.0);

  constexpr int N = 20;
  constexpr int L = 12;

  std::vector<std::vector<double>> vecs(N, std::vector<double>(L));
  for (auto &v : vecs)
    for (auto &x : v)
      x = (uni(rng) < 0.15) ? NaN_val : gauss(rng);

  Problem prob = make_problem_uv(vecs, 2, core::MissingStrategy::ZeroCost);

  algorithms::CLARAOptions opts;
  opts.n_clusters  = 2;
  opts.n_samples   = 3;
  opts.sample_size = 10;
  opts.random_seed = 77;

  // Must not throw (if missing_strategy were NOT propagated, the sub-problem
  // would use Error strategy and throw on NaN).
  core::ClusteringResult result;
  REQUIRE_NOTHROW(result = algorithms::fast_clara(prob, opts));

  REQUIRE(result.labels.size() == static_cast<size_t>(N));
  REQUIRE(std::isfinite(result.total_cost));
}

// ===========================================================================
// Test 13: CLARANS medoids are self-assigned (invariant check)
// ===========================================================================
TEST_CASE("Wave2A: CLARANS medoids are self-assigned to their own cluster",
          "[wave2a][clarans][self_assignment]")
{
  auto prob = make_separated_problem(8, 3, 42);

  algorithms::CLARANSOptions opts;
  opts.n_clusters  = 3;
  opts.num_local   = 2;
  opts.random_seed = 42;

  auto result = algorithms::clarans(prob, opts);

  for (int c = 0; c < 3; ++c) {
    int med = result.medoid_indices[c];
    REQUIRE(result.labels[med] == c);
  }
}

// ===========================================================================
// Test 14: Hierarchical dendrogram merge distances are non-decreasing
//   (correct agglomerative property — each merge can only be >= the previous).
// ===========================================================================
TEST_CASE("Wave2A: hierarchical merge distances are non-decreasing",
          "[wave2a][hierarchical][monotone]")
{
  std::vector<std::vector<double>> vecs;
  for (int i = 0; i < 15; ++i)
    vecs.push_back({ static_cast<double>(i) });

  Problem prob = make_problem_uv(vecs, 2);
  prob.fillDistanceMatrix();

  for (auto linkage : { algorithms::Linkage::Single,
                         algorithms::Linkage::Complete,
                         algorithms::Linkage::Average }) {
    algorithms::HierarchicalOptions opts;
    opts.linkage    = linkage;
    opts.max_points = 200;

    auto dend = algorithms::build_dendrogram(prob, opts);
    REQUIRE(dend.merges.size() == 14u);

    for (size_t i = 1; i < dend.merges.size(); ++i) {
      // Each merge distance must be >= the previous (agglomerative property).
      REQUIRE(dend.merges[i].distance >= dend.merges[i - 1].distance - 1e-9);
    }
  }
}

// ===========================================================================
// Test 15: FastCLARA ndim propagation — sub-problem must use same ndim as parent
//   Specifically, sub-problem distances must be consistent with parent distances.
// ===========================================================================
TEST_CASE("Wave2A: FastCLARA ndim=2 sub-problem distances consistent with parent",
          "[wave2a][fast_clara][ndim][propagation]")
{
  constexpr int N = 40;
  constexpr int n_steps = 8;
  constexpr int ndim = 2;
  constexpr int k = 2;

  Problem prob = make_problem_mv(N, n_steps, ndim, 200.0, k, core::MissingStrategy::Error, 55);

  algorithms::CLARAOptions opts;
  opts.n_clusters  = k;
  opts.n_samples   = 2;
  opts.sample_size = 20;
  opts.random_seed = 33;

  core::ClusteringResult result;
  REQUIRE_NOTHROW(result = algorithms::fast_clara(prob, opts));

  REQUIRE(result.labels.size() == static_cast<size_t>(N));
  REQUIRE(result.medoid_indices.size() == static_cast<size_t>(k));

  // Verify total_cost matches manual recomputation via parent prob.distByInd.
  double recomputed = 0.0;
  for (int p = 0; p < N; ++p) {
    int med = result.medoid_indices[result.labels[p]];
    recomputed += prob.distByInd(p, med);
  }
  REQUIRE_THAT(result.total_cost, WithinAbs(recomputed, 1e-6));
}
