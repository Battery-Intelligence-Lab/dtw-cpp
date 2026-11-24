/**
 * @file unit_test_fast_pam.cpp
 * @brief Unit tests for FasterPAM k-medoids clustering.
 *
 * @details Tests verify correctness, convergence, and quality of FastPAM
 * against basic invariants and compared to Lloyd-style k-medoids.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include <dtwc.hpp>
#include <mip/mip.hpp> // highs_solver_available

#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <set>
#include <string>
#include <system_error>
#include <vector>

#ifndef DTWC_TEST_DATA_DIR
#define DTWC_TEST_DATA_DIR "./data"
#endif

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::MessageMatches;
using Catch::Matchers::WithinAbs;
using namespace dtwc;

// ---------------------------------------------------------------------------
// Helper: build a Problem with N synthetic time series.
// Creates distinct patterns so clustering is meaningful.
// ---------------------------------------------------------------------------
static Problem make_synthetic_problem(int N)
{
  std::vector<std::vector<data_t>> vecs;
  std::vector<std::string> names;

  for (int i = 0; i < N; ++i) {
    // Create time series with group structure:
    //   Group 0: i in [0, N/3)       -> low baseline, gentle slope
    //   Group 1: i in [N/3, 2*N/3)   -> medium baseline, steeper slope
    //   Group 2: i in [2*N/3, N)     -> high baseline, negative slope
    int group = (i * 3) / N;
    double baseline = group * 50.0;
    double slope = (group == 2) ? -2.0 : (group + 1) * 1.5;
    double noise_offset = i * 0.1;  // deterministic per-series offset

    std::vector<data_t> ts;
    int len = 20 + (i % 5);  // varying lengths
    for (int j = 0; j < len; ++j) {
      ts.push_back(baseline + slope * j + noise_offset);
    }
    vecs.push_back(std::move(ts));
    names.push_back("ts_" + std::to_string(i));
  }

  Data data(std::move(vecs), std::move(names));
  Problem prob("fast_pam_test");
  prob.set_data(std::move(data));
  return prob;
}

// ---------------------------------------------------------------------------
// Helper: build a small Problem for edge-case tests.
// ---------------------------------------------------------------------------
static Problem make_small_problem(int N = 5)
{
  std::vector<std::vector<data_t>> vecs;
  std::vector<std::string> names;

  for (int i = 0; i < N; ++i) {
    std::vector<data_t> ts;
    for (int j = 0; j <= i + 2; ++j) {
      ts.push_back(static_cast<data_t>(i * 10 + j));
    }
    vecs.push_back(std::move(ts));
    names.push_back("ts_" + std::to_string(i));
  }

  Data data(std::move(vecs), std::move(names));
  Problem prob("fast_pam_small");
  prob.set_data(std::move(data));
  return prob;
}

// ---------------------------------------------------------------------------
// Helper: compute Lloyd-style cost for comparison.
// Runs Problem's existing cluster_by_kmedoids_lloyd which is Lloyd iteration.
// ---------------------------------------------------------------------------
static double lloyd_cost(int N, int k)
{
  Problem prob = make_synthetic_problem(N);
  prob.set_n_clusters(k);
  prob.set_n_repetitions(5);
  prob.set_max_iter(100);
  prob.fill_distance_matrix();
  prob.init_fun = init::Kmeanspp;
  prob.cluster_by_kmedoids_lloyd();
  return prob.find_total_cost();
}


// ===========================================================================
// The invariants of a FastPAM result, on 15 and 12 grouped series (k = 3) and
// 10 small ones (k = 2): convergence, valid and distinct medoids, labels in
// [0, k), each medoid in its own cluster, total_cost the recomputed cost.
// ===========================================================================
TEST_CASE("FastPAM result invariants and cost consistency", "[fast_pam]")
{
  struct Input
  {
    bool synthetic;
    int N;
    int k;
  };
  for (const Input input : { Input{ true, 15, 3 }, Input{ true, 12, 3 }, Input{ false, 10, 2 } }) {
    const int N = input.N;
    const int k = input.k;
    CAPTURE(input.synthetic, N, k);

    Problem prob = input.synthetic ? make_synthetic_problem(N) : make_small_problem(N);
    const auto result = fast_pam(prob, k);

    REQUIRE(result.converged);
    REQUIRE(result.iterations < 100);
    REQUIRE(result.total_cost > 0.0);

    REQUIRE(result.medoid_indices.size() == static_cast<size_t>(k));
    for (index_t m : result.medoid_indices) {
      REQUIRE(m >= 0);
      REQUIRE(m < N);
    }
    const std::set<index_t> unique_medoids(result.medoid_indices.begin(), result.medoid_indices.end());
    REQUIRE(unique_medoids.size() == static_cast<size_t>(k));

    REQUIRE(result.labels.size() == static_cast<size_t>(N));
    for (index_t label : result.labels) {
      REQUIRE(label >= 0);
      REQUIRE(label < k);
    }
    for (int c = 0; c < k; ++c)
      REQUIRE(result.labels[result.medoid_indices[c]] == c);

    double recomputed_cost = 0.0;
    for (int p = 0; p < N; ++p)
      recomputed_cost += prob.dist_by_ind(p, result.medoid_indices[result.labels[p]]);
    REQUIRE_THAT(result.total_cost, WithinAbs(recomputed_cost, 1e-10));
  }
}

// ===========================================================================
// Test 2: FastPAM finds cost <= Lloyd iteration.
// ===========================================================================
TEST_CASE("FastPAM cost <= Lloyd cost", "[fast_pam][quality]")
{
  constexpr int N = 15;
  constexpr int k = 3;

  Problem prob_fp = make_synthetic_problem(N);
  double fp_cost = fast_pam(prob_fp, k).total_cost;

  double ll_cost = lloyd_cost(N, k);

  // FastPAM should find an equal or better (lower) cost than Lloyd.
  // Allow a small tolerance for floating-point arithmetic.
  REQUIRE(fp_cost <= ll_cost + 1e-9);
}

TEST_CASE("seeded FastPAM BUILD samples proportional to k-median distance",
          "[fast_pam][seeded][initialization]")
{
  // Conditional on first medoid 0, the remaining singleton-series distances
  // are 1 and 3. K-median++ therefore selects point 2 with probability 3/4;
  // incorrectly squaring the weights would move that probability to 9/10.
  Problem prob("fast_pam_d_sampling");
  prob.set_data(Data(std::vector<std::vector<data_t>>{{0.0}, {1.0}, {3.0}},
                     std::vector<std::string>{"zero", "one", "three"}));

  int conditioned = 0;
  int selected_far = 0;
  constexpr std::uint64_t seed_count = 4096;
  for (std::uint64_t seed = 0; seed < seed_count; ++seed) {
    // max_iter=0 observes the deterministic seeded BUILD result before SWAP.
    const auto result = fast_pam(prob, 2, 0, seed);
    if (result.medoid_indices.front() != 0) continue;
    ++conditioned;
    if (result.medoid_indices.back() == 2) ++selected_far;
  }

  REQUIRE(conditioned > 1000);
  const double far_fraction = static_cast<double>(selected_far) / conditioned;
  CAPTURE(conditioned, selected_far, far_fraction);
  REQUIRE(far_fraction > 0.70);
  REQUIRE(far_fraction < 0.80);
}

TEST_CASE("A Soft-DTW Problem clusters on the divergence: the exact methods and FastPAM agree",
          "[fast_pam][softdtw]")
{
  // The raw Soft-DTW of similar series is negative, sdtw(x, x) too; the divergence
  // sdtw(x, y) - (sdtw(x, x) + sdtw(y, y)) / 2 is >= 0 and 0 between equal series. Series 0, 1 and 3 are equal,
  // 4 and 5 close. FastPAM reaches the optimum the exact methods prove here.
  const std::vector<std::vector<data_t>> series{ { 3, 0 }, { 3, 0 }, { 0, 3 }, { 3, 0 }, { 0, 0 }, { 0.2, 0 }, { 1, 3 } };
  const double gamma = 1.0;
  Problem prob("softdtw_divergence");
  prob.set_data(Data(std::vector<std::vector<data_t>>(series), { "a", "b", "c", "d", "e", "f", "g" }));
  core::DTWVariantParams params;
  params.variant = core::DTWVariant::SoftDTW;
  params.sdtw_gamma = gamma;
  prob.set_variant(params);
  prob.fill_distance_matrix();
  const auto sdtw = [&](std::size_t i, std::size_t j) {
    return distance::soft_dtw<data_t>(std::span<const data_t>(series[i]), std::span<const data_t>(series[j]), gamma);
  };
  for (std::size_t i = 0; i < series.size(); ++i)
    for (std::size_t j = 0; j < series.size(); ++j) {
      CAPTURE(i, j);
      const double d = prob.dist_by_ind(static_cast<index_t>(i), static_cast<index_t>(j));
      CHECK(d >= 0.0);
      if (i == j)
        CHECK(d == 0.0);
      else
        CHECK_THAT(d, WithinAbs(std::max(0.0, sdtw(i, j) - (sdtw(i, i) + sdtw(j, j)) / 2), 1e-12));
    }
  CHECK(prob.dist_by_ind(0, 1) == 0.0);

  std::vector<Method> methods{ Method::LRCore, Method::PAM };
  if (highs_solver_available()) methods.push_back(Method::MIP);
  prob.set_n_clusters(3);
  double optimum = 0.0;
  for (const Method method : methods) {
    CAPTURE(name_of(method_names, method));
    prob.set_method(method);
    const auto result = prob.cluster();
    for (std::size_t c = 0; c < result.medoid_indices.size(); ++c)
      CHECK(result.labels[static_cast<std::size_t>(result.medoid_indices[c])] == static_cast<index_t>(c));
    if (method == Method::LRCore) optimum = result.total_cost;
    CHECK(result.total_cost > 0.0);
    CHECK_THAT(result.total_cost, WithinAbs(optimum, 1e-12));
  }
}

TEST_CASE("seeded FastPAM completes the medoid set when all weights are zero",
          "[fast_pam][seeded][degenerate]")
{
  // Exercises dtwc::fast_pam. Identical series drive every BUILD sampling
  // weight to exactly zero, so core::distance_sampling_weights returns
  // total == 0 and core::kmedoids_pp must complete the distinct medoid set
  // deterministically instead of constructing an invalid weighted distribution.
  // The k-means++ siblings of this case are in unit_test_clustering_algorithms.cpp.
  Problem prob("fast_pam_identical_series");
  prob.set_data(Data(
    std::vector<std::vector<data_t>>{
      {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0},
      {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}},
    std::vector<std::string>{"a", "b", "c", "d"}));

  // max_iter=0 observes the deterministic seeded BUILD result before SWAP.
  const auto result = fast_pam(prob, 3, 0, 7);

  REQUIRE(result.medoid_indices.size() == 3);
  const std::set<index_t> unique(result.medoid_indices.begin(),
                                 result.medoid_indices.end());
  CHECK(unique.size() == 3);
  // The fallback fills upward from index 0, so whichever index the seed drew
  // first, both 0 and 1 must be completed into the set. This holds for all four
  // possible first draws and is therefore independent of the seed.
  CHECK(unique.count(0) == 1);
  CHECK(unique.count(1) == 1);
  CHECK(result.total_cost == 0.0);
}

// ===========================================================================
TEST_CASE("FastPAM converged: no swap lowers the cost by more than 1e-9 of it on distances of two scales",
          "[fast_pam][convergence]")
{
  // Distances 1 and e = 1e-12. Points 2, 3 and 4 are 1 apart; 0 and 1 are e apart, near 2 and 3 (0 at 2e, 1 at e) and
  // 1 from 4. For k = 2, {1, 4} costs 3e and {0, 4} 5e. A BUILD of 4 and one of 2, 3 costs about 1 (the other is 1
  // from both); the first swap takes 0 in for it and the cost falls to 5e, where swapping 0 for 1 still lowers it by
  // 2e, 40 % of it.
  const double e = 1e-12;
  const std::vector<std::vector<double>> D{ { 0, e, 2 * e, 2 * e, 1 }, { e, 0, e, e, 1 }, { 2 * e, e, 0, 1, 1 },
                                            { 2 * e, e, 1, 0, 1 }, { 1, 1, 1, 1, 0 } };
  Problem prob("two_scales");
  prob.set_data(Data(std::vector<std::vector<data_t>>{ { 0 }, { 1 }, { 2 }, { 3 }, { 4 } },
                     std::vector<std::string>{ "a", "b", "c", "d", "e" }));
  auto &matrix = prob.writable_distance_matrix();
  matrix.resize(D.size());
  for (std::size_t i = 0; i < D.size(); ++i)
    for (std::size_t j = i; j < D.size(); ++j) matrix.set(i, j, D[i][j]);
  prob.fill_distance_matrix(); // commits the writes
  const auto cost = [&](const std::vector<index_t> &medoids) { // the objective, recomputed directly
    double sum = 0.0;
    for (const auto &row : D) {
      double nearest = row[static_cast<std::size_t>(medoids[0])];
      for (const index_t m : medoids) nearest = std::min(nearest, row[static_cast<std::size_t>(m)]);
      sum += nearest;
    }
    return sum;
  };

  for (std::uint64_t seed = 0; seed < 16; ++seed) { // BUILD follows the seed; several of these start near 1
    CAPTURE(seed);
    const auto result = fast_pam(prob, 2, 100, seed);
    REQUIRE(result.converged);
    const double returned = cost(result.medoid_indices);
    for (std::size_t slot = 0; slot < result.medoid_indices.size(); ++slot)
      for (index_t candidate = 0; candidate < static_cast<index_t>(D.size()); ++candidate) {
        if (std::count(result.medoid_indices.begin(), result.medoid_indices.end(), candidate) != 0) continue;
        auto swapped = result.medoid_indices;
        swapped[slot] = candidate;
        CAPTURE(result.medoid_indices, swapped, returned);
        CHECK(returned - cost(swapped) <= 1e-9 * returned);
      }
  }
}

// ===========================================================================
// Test 6: k=1 produces a single cluster with all points.
// ===========================================================================
TEST_CASE("FastPAM k=1 assigns all points to one cluster", "[fast_pam][k1]")
{
  constexpr int N = 8;
  constexpr int k = 1;

  Problem prob = make_small_problem(N);
  auto result = fast_pam(prob, k);

  REQUIRE(result.medoid_indices.size() == 1);
  REQUIRE(result.labels.size() == static_cast<size_t>(N));

  for (index_t label : result.labels) {
    REQUIRE(label == 0);
  }

  REQUIRE(result.converged);
}

// ===========================================================================
// Test 7: k=N makes every point a medoid.
// ===========================================================================
TEST_CASE("FastPAM k=N makes every point a medoid", "[fast_pam][kN]")
{
  constexpr int N = 5;
  constexpr int k = N;

  Problem prob = make_small_problem(N);
  auto result = fast_pam(prob, k);

  REQUIRE(result.medoid_indices.size() == static_cast<size_t>(N));
  REQUIRE_THAT(result.total_cost, WithinAbs(0.0, 1e-10));

  // Every point should be a medoid.
  std::set<index_t> medoid_set(result.medoid_indices.begin(), result.medoid_indices.end());
  REQUIRE(medoid_set.size() == static_cast<size_t>(N));
}

// ===========================================================================
// Test 10: Invalid inputs throw.
// ===========================================================================
TEST_CASE("FastPAM throws on invalid inputs", "[fast_pam][errors]")
{
  SECTION("k = 0 throws")
  {
    Problem prob = make_small_problem(5);
    REQUIRE_THROWS_AS(fast_pam(prob, 0), std::runtime_error);
  }

  SECTION("k > N throws")
  {
    Problem prob = make_small_problem(5);
    REQUIRE_THROWS_AS(fast_pam(prob, 10), std::runtime_error);
  }

  SECTION("empty problem throws")
  {
    Problem prob("empty");
    REQUIRE_THROWS_AS(fast_pam(prob, 1), std::runtime_error);
  }
}

// ===========================================================================
// max_iter: 0 is the BUILD-only oracle (the seeded start, no SWAP); a negative
// count has no meaning and is refused here, once, for every binding.
// ===========================================================================
namespace {

/// Eight series [i, i+0.01, i-0.02, i+0.03]: ambiguous enough that the seeded BUILD
/// medoids differ from the SWAP optimum. The Python and MATLAB suites pin the same
/// medoids on the same series.
Problem make_offset_problem()
{
  std::vector<std::vector<data_t>> vecs;
  std::vector<std::string> names;
  for (int i = 0; i < 8; ++i) {
    const double base = i;
    vecs.push_back({ base, base + 0.01, base - 0.02, base + 0.03 });
    names.push_back(std::to_string(i));
  }
  Problem prob("fast_pam_offset");
  prob.set_data(Data(std::move(vecs), std::move(names)));
  return prob;
}

} // namespace

TEST_CASE("FastPAM max_iter = 0 returns the seeded BUILD medoids and runs no SWAP",
          "[fast_pam][max_iter]")
{
  auto prob = make_offset_problem();

  const auto build_29 = fast_pam(prob, 3, 0, 29);
  CHECK(build_29.medoid_indices == std::vector<index_t>{ 4, 2, 7 });
  CHECK(build_29.iterations == 0);
  CHECK_FALSE(build_29.converged);

  const auto build_42 = fast_pam(prob, 3, 0, 42);
  CHECK(build_42.medoid_indices == std::vector<index_t>{ 6, 2, 5 });

  // The default seed is 42.
  CHECK(fast_pam(prob, 3, 0).medoid_indices == build_42.medoid_indices);

  // With a SWAP budget seed 29 leaves {4, 2, 7} for {4, 1, 7}.
  const auto swapped = fast_pam(prob, 3, 100, 29);
  CHECK(swapped.medoid_indices == std::vector<index_t>{ 4, 1, 7 });
  CHECK_THAT(swapped.total_cost, WithinAbs(20.0, 1e-9));
  CHECK(swapped.converged);
}

TEST_CASE("FastPAM refuses a negative max_iter before it touches the Problem",
          "[fast_pam][max_iter][errors]")
{
  const int bad = GENERATE(-1, -100, std::numeric_limits<int>::min());
  CAPTURE(bad);
  auto prob = make_offset_problem();
  prob.set_n_clusters(2);

  REQUIRE_THROWS_MATCHES(fast_pam(prob, 3, bad), InvalidInput,
                         MessageMatches(ContainsSubstring("fast_pam: max_iter")
                                        && ContainsSubstring("got " + std::to_string(bad))));
  CHECK_FALSE(prob.is_distance_matrix_filled()); // refused before any distance was computed
  CHECK(prob.labels().empty());
  CHECK(prob.n_clusters() == 2);
}

// ===========================================================================
// Test 11: 2.0 write-back — fast_pam stores the result INTO prob.
// (1.x asserted non-mutation; the binding auto-wire moved into
//  core, so pure-C++ users get prob.centroids_ind/clusters_ind/n_clusters set.)
// ===========================================================================
TEST_CASE("FastPAM writes result back into Problem", "[fast_pam][write_back]")
{
  constexpr int N = 8;
  constexpr int k = 2;

  Problem prob = make_small_problem(N);

  // Set up some unrelated initial state on prob (k=3) to prove fast_pam overwrites it.
  prob.set_n_clusters(3);

  auto result = fast_pam(prob, k);

  // prob now holds the fast_pam result (labels/medoids/k), with NO manual wiring.
  REQUIRE(prob.n_clusters() == k);
  REQUIRE(prob.centroids_ind == result.medoid_indices);
  REQUIRE(prob.clusters_ind == result.labels);
  REQUIRE(static_cast<int>(prob.centroids_ind.size()) == k);
  REQUIRE(static_cast<int>(prob.clusters_ind.size()) == N);
}
