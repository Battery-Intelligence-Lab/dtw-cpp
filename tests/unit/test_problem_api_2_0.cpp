/**
 * @file test_problem_api_2_0.cpp
 * @brief Task 1.6 tests: 2.0 Problem/scores rename shims, result write-back,
 *        and the variant_params -> dtw_fn_ rebind invariant.
 *
 * @details Pins the LIVE public entry points introduced/changed by Task 1.6 of
 *          the DTWC++ 2.0 refactor (docs/api-contract-2.0.md, FROZEN):
 *            1. Deprecated 1.x camelCase shims forward to the canonical
 *               snake_case names (compile with a deprecation warning, which we
 *               suppress locally; assert identical behaviour).
 *            2. scores::silhouette(prob) works in pure C++ immediately after
 *               fast_pam(prob, k) with NO manual wiring — the result write-back
 *               moved into core (was binding-only in 1.x).
 *            3. set_variant(...) rebinds the bound DTW function dtw_fn_ — a
 *               direct behavioural check: a known distance changes from the
 *               registered Standard value to the registered ADTW value.
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <numeric>
#include <string>
#include <vector>

// Local, cross-compiler suppression of -Wdeprecated-declarations so this test
// can call the deprecated 1.x shims on purpose without failing a -Werror build.
#if defined(__clang__)
#  define DTWC_PUSH_NO_DEPRECATED _Pragma("clang diagnostic push") _Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"")
#  define DTWC_POP_NO_DEPRECATED  _Pragma("clang diagnostic pop")
#elif defined(__GNUC__)
#  define DTWC_PUSH_NO_DEPRECATED _Pragma("GCC diagnostic push") _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")
#  define DTWC_POP_NO_DEPRECATED  _Pragma("GCC diagnostic pop")
#elif defined(_MSC_VER)
#  define DTWC_PUSH_NO_DEPRECATED __pragma(warning(push)) __pragma(warning(disable : 4996))
#  define DTWC_POP_NO_DEPRECATED  __pragma(warning(pop))
#else
#  define DTWC_PUSH_NO_DEPRECATED
#  define DTWC_POP_NO_DEPRECATED
#endif

using Catch::Matchers::WithinAbs;
using namespace dtwc;

// ---------------------------------------------------------------------------
// Helper: two clearly separated groups of 3 flat-ish series each (length 3).
// Group L ~ 0, Group H ~ 100 -> fast_pam with k=2 recovers the two groups and
// intra-cluster DTW distances are ~0 while inter-cluster ones are ~200-300.
// ---------------------------------------------------------------------------
static Problem make_two_group_problem()
{
  std::vector<std::vector<data_t>> vecs = {
    { 0.0, 0.0, 0.0 }, { 0.0, 0.0, 1.0 }, { 0.0, 1.0, 0.0 },       // group L (near 0)
    { 100.0, 100.0, 100.0 }, { 100.0, 100.0, 101.0 }, { 100.0, 101.0, 100.0 }, // group H (near 100)
  };
  std::vector<std::string> names = { "L0", "L1", "L2", "H0", "H1", "H2" };
  Data data(std::move(vecs), std::move(names));
  Problem prob("api_2_0_two_group");
  prob.set_data(std::move(data));
  return prob;
}

// ===========================================================================
// Test 1: Deprecated 1.x shims forward to the canonical 2.0 names.
// exercises: Problem::{set_numberOfClusters,cluster_size,distByInd,maxDistance}
//            and scores::{daviesBouldinIndex,dunnIndex,adjustedRandIndex}
//            deprecated shims -> their snake_case canonical implementations.
// ===========================================================================
TEST_CASE("Task 1.6: deprecated shims forward to canonical names", "[api_2_0][deprecated]")
{
  SECTION("set_numberOfClusters == set_n_clusters, cluster_size() == n_clusters()")
  {
    Problem p_old = make_two_group_problem();
    Problem p_new = make_two_group_problem();

    DTWC_PUSH_NO_DEPRECATED
    p_old.set_numberOfClusters(2); // deprecated -> forwards to set_n_clusters
    DTWC_POP_NO_DEPRECATED
    p_new.set_n_clusters(2);       // canonical

    DTWC_PUSH_NO_DEPRECATED
    const auto k_old = p_old.cluster_size(); // deprecated -> forwards to n_clusters()
    DTWC_POP_NO_DEPRECATED
    REQUIRE(k_old == p_new.n_clusters());
    REQUIRE(p_new.n_clusters() == 2);
  }

  SECTION("distByInd == dist_by_ind, maxDistance == max_distance")
  {
    Problem prob = make_two_group_problem();
    prob.fill_distance_matrix();

    DTWC_PUSH_NO_DEPRECATED
    const double d_old = prob.distByInd(0, 3); // deprecated -> dist_by_ind
    const double md_old = prob.maxDistance();  // deprecated -> max_distance
    DTWC_POP_NO_DEPRECATED

    REQUIRE_THAT(d_old, WithinAbs(prob.dist_by_ind(0, 3), 1e-12));
    REQUIRE_THAT(md_old, WithinAbs(prob.max_distance(), 1e-12));
  }

  SECTION("scores deprecated aliases forward to canonical")
  {
    Problem prob = make_two_group_problem();
    (void)fast_pam(prob, 2); // write-back populates centroids_ind/clusters_ind

    DTWC_PUSH_NO_DEPRECATED
    const double dbi_old = scores::daviesBouldinIndex(prob); // -> davies_bouldin
    const double dunn_old = scores::dunnIndex(prob);         // -> dunn
    DTWC_POP_NO_DEPRECATED
    REQUIRE_THAT(dbi_old, WithinAbs(scores::davies_bouldin(prob), 1e-12));
    REQUIRE_THAT(dunn_old, WithinAbs(scores::dunn(prob), 1e-12));

    const std::vector<int> a = { 0, 0, 1, 1 };
    const std::vector<int> b = { 1, 1, 0, 0 };
    DTWC_PUSH_NO_DEPRECATED
    const double ari_old = scores::adjustedRandIndex(a, b); // -> adjusted_rand
    DTWC_POP_NO_DEPRECATED
    REQUIRE_THAT(ari_old, WithinAbs(scores::adjusted_rand(a, b), 1e-12));
  }
}

// ===========================================================================
// Test 2: result write-back — silhouette works right after fast_pam, no wiring.
// exercises: dtwc::fast_pam() write-back -> dtwc::scores::silhouette() LIVE path
//            (cluster then score in pure C++; no manual centroids_ind assignment).
// Registered band: two well-separated groups => mean silhouette > 0.9.
// ===========================================================================
TEST_CASE("Task 1.6: silhouette works after fast_pam with no manual wiring", "[api_2_0][write_back][silhouette]")
{
  Problem prob = make_two_group_problem();

  const auto result = fast_pam(prob, 2);

  // Write-back landed the result into prob (was binding-only in 1.x).
  REQUIRE(prob.n_clusters() == 2);
  REQUIRE(prob.centroids_ind == result.medoid_indices);
  REQUIRE(prob.clusters_ind == result.labels);
  REQUIRE(prob.centroids_ind.size() == 2);
  REQUIRE(prob.clusters_ind.size() == 6);

  // LIVE score path: silhouette reads prob state that fast_pam wrote. Before the
  // 2.0 write-back this returned the degenerate all-(-1) fill (centroids empty).
  const auto sil = scores::silhouette(prob);
  REQUIRE(sil.size() == 6);
  const double mean = std::accumulate(sil.begin(), sil.end(), 0.0) / static_cast<double>(sil.size());
  REQUIRE(mean > 0.9); // registered pass band for two clearly separated groups
}

// ===========================================================================
// Test 3: set_variant() rebinds the bound DTW function dtw_fn_ (behavioural).
// exercises: dtwc::Problem::set_variant() -> refresh_distance_matrix() ->
//            rebind_dtw_fn(); prove by computing a known distance before/after.
//
// Registered oracle (hand-derived, L1, full DTW band=-1, x={0,0} vs y={0,1,2}):
//   Standard DTW  = C(1,2) = 3.0
//       C(i,j) = |x_i-y_j| + min(C(i-1,j-1), C(i-1,j), C(i,j-1))
//   ADTW (penalty=1.0) = C(1,2) = 4.0
//       C(i,j) = |x_i-y_j| + min(C(i-1,j-1), C(i-1,j)+p, C(i,j-1)+p)
//   (both values registered here BEFORE the run; see warping_adtw.hpp recurrence.)
// ===========================================================================
TEST_CASE("Task 1.6: set_variant rebinds dtw_fn_ (Standard=3.0 -> ADTW=4.0)", "[api_2_0][rebind][variant]")
{
  std::vector<std::vector<data_t>> vecs = { { 0.0, 0.0 }, { 0.0, 1.0, 2.0 } };
  std::vector<std::string> names = { "x", "y" };
  Data data(std::move(vecs), std::move(names));
  Problem prob("api_2_0_rebind");
  prob.set_data(std::move(data));
  prob.set_band(-1); // full DTW (also the default); matches the hand-derived oracle

  // Bound function is Standard DTW at construction (default variant).
  const double d_std = prob.dtw_function()(prob.series(0), prob.series(1));
  REQUIRE_THAT(d_std, WithinAbs(3.0, 1e-9)); // registered Standard value

  // Writing the variant through the setter MUST rebind dtw_fn_ to ADTW.
  core::DTWVariantParams vp;
  vp.variant = core::DTWVariant::ADTW;
  vp.adtw_penalty = 1.0;
  prob.set_variant(vp);

  const double d_adtw = prob.dtw_function()(prob.series(0), prob.series(1));
  REQUIRE_THAT(d_adtw, WithinAbs(4.0, 1e-9)); // registered ADTW value
  REQUIRE(d_adtw != d_std);                   // the rebind actually changed the function
}
