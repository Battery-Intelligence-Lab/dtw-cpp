/**
 * @file unit_test_wave1a_integration.cpp
 * @brief Wave 1A end-to-end integration tests + performance sanity checks.
 *
 * @details Verifies that all Wave 1A components work correctly together:
 *   - missing_utils.hpp (bitwise NaN check, interpolation)
 *   - MissingStrategy enum (Error / ZeroCost / AROW / Interpolate)
 *   - DTW-AROW algorithm wired through Problem
 *   - 5 new metrics: Dunn, inertia, CH, ARI, NMI
 *   - Problem::cluster() → metrics pipeline with missing data
 *
 * @section performance Performance sanity (not a formal benchmark)
 *   One test prints chrono timings for 1000 pairwise DTW computations
 *   on length-100 series with 10% NaN under ZeroCost / AROW / Interpolate.
 *   The goal is to confirm AROW is not catastrophically slower (same O(n*m)
 *   complexity, different recurrence branch).
 *
 * @date 02 Apr 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using Catch::Matchers::WithinAbs;
using namespace dtwc;

// ---------------------------------------------------------------------------
// Test-local constants
// ---------------------------------------------------------------------------
static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate N series of length L drawn from a Gaussian centred at `mean` with
/// given standard deviation.  `nan_rate` fraction of values are replaced with
/// NaN (uniformly at random, seeded deterministically).
static std::vector<std::vector<double>> make_gaussian_series(
    int N, int L, double mean, double stddev, double nan_rate = 0.0,
    uint64_t seed = 42)
{
  std::mt19937_64 rng(seed);
  std::normal_distribution<double> gauss(mean, stddev);
  std::uniform_real_distribution<double> uniform(0.0, 1.0);

  std::vector<std::vector<double>> out(N, std::vector<double>(L));
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < L; ++j) {
      out[i][j] = gauss(rng);
      if (nan_rate > 0.0 && uniform(rng) < nan_rate)
        out[i][j] = NaN;
    }
  }
  return out;
}

/// A temporary directory for test output (created once, cleaned on test exit).
/// The tests call cluster() which writes CSV files; we redirect them here so
/// they don't fail trying to create ".\results\" in the working directory.
static fs::path g_tmp_output_dir()
{
  static fs::path dir = fs::temp_directory_path() / "dtwc_wave1a_test_output";
  static bool created = [&] {
    fs::create_directories(dir);
    return true;
  }();
  (void)created;
  return dir;
}

/// Build a Problem from a flat list of series.  No NaN assumed unless you pass
/// nan_rate > 0 (which uses make_gaussian_series internally).
static Problem make_problem(
    std::vector<std::vector<double>> vecs, int Nc,
    core::MissingStrategy strategy = core::MissingStrategy::Error)
{
  std::vector<std::string> names;
  names.reserve(vecs.size());
  for (size_t i = 0; i < vecs.size(); ++i)
    names.push_back("s" + std::to_string(i));

  Data d(std::move(vecs), std::move(names));

  Problem prob("integration");
  prob.set_data(std::move(d));
  prob.set_numberOfClusters(Nc);
  prob.missing_strategy = strategy;
  prob.verbose = false;
  prob.output_folder = g_tmp_output_dir(); // avoid failure writing result CSVs
  return prob;
}

/// Returns true if all values in v are finite (not NaN, not inf).
static bool all_finite(const std::vector<double> &v)
{
  return std::all_of(v.begin(), v.end(), [](double x) {
    return std::isfinite(x);
  });
}

// ---------------------------------------------------------------------------
// 1. Full clustering pipeline with missing data (ZeroCost)
// ---------------------------------------------------------------------------

TEST_CASE("Integration: ZeroCost pipeline — full metrics on 20 series with NaN",
          "[wave1a][integration][zerocost]")
{
  // 20 series of length 20, 10% NaN, 2 clusters
  auto vecs_a = make_gaussian_series(10, 20, 0.0, 1.0, 0.10, 1);
  auto vecs_b = make_gaussian_series(10, 20, 20.0, 1.0, 0.10, 2);
  std::vector<std::vector<double>> all_vecs;
  all_vecs.insert(all_vecs.end(), vecs_a.begin(), vecs_a.end());
  all_vecs.insert(all_vecs.end(), vecs_b.begin(), vecs_b.end());

  auto prob = make_problem(all_vecs, 2, core::MissingStrategy::ZeroCost);
  prob.maxIter = 20;
  prob.N_repetition = 1;

  // Fill distance matrix (required before manual cluster ops)
  REQUIRE_NOTHROW(prob.fillDistanceMatrix());
  REQUIRE(prob.isDistanceMatrixFilled());

  // Check distance matrix is finite everywhere
  const int N = static_cast<int>(prob.size());
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      REQUIRE(std::isfinite(prob.distByInd(i, j)));

  // Cluster
  prob.cluster();
  REQUIRE(prob.clusters_ind.size() == static_cast<size_t>(N));
  REQUIRE(prob.centroids_ind.size() == 2u);

  // --- Silhouette ---
  auto sil = scores::silhouette(prob);
  REQUIRE(sil.size() == static_cast<size_t>(N));
  REQUIRE(all_finite(sil));
  for (double s : sil)
    REQUIRE(s >= -1.0 - 1e-9);

  // --- Davies-Bouldin ---
  double dbi = scores::daviesBouldinIndex(prob);
  REQUIRE(std::isfinite(dbi));
  REQUIRE(dbi >= 0.0);

  // --- Dunn ---
  double dunn = scores::dunnIndex(prob);
  REQUIRE(std::isfinite(dunn));
  REQUIRE(dunn >= 0.0);

  // --- Inertia ---
  double inert = scores::inertia(prob);
  REQUIRE(std::isfinite(inert));
  REQUIRE(inert >= 0.0);

  // --- Calinski-Harabasz ---
  double ch = scores::calinskiHarabaszIndex(prob);
  REQUIRE(std::isfinite(ch));
  REQUIRE(ch >= 0.0);
}

// ---------------------------------------------------------------------------
// 2. AROW clustering pipeline + AROW >= ZeroCost
// ---------------------------------------------------------------------------

TEST_CASE("Integration: AROW pipeline — finite metrics; AROW >= ZeroCost distances",
          "[wave1a][integration][arow]")
{
  auto vecs_a = make_gaussian_series(10, 20, 0.0, 1.0, 0.10, 10);
  auto vecs_b = make_gaussian_series(10, 20, 15.0, 1.0, 0.10, 11);
  std::vector<std::vector<double>> all_vecs;
  all_vecs.insert(all_vecs.end(), vecs_a.begin(), vecs_a.end());
  all_vecs.insert(all_vecs.end(), vecs_b.begin(), vecs_b.end());

  // Build both problems sharing the same underlying data.
  auto prob_zero = make_problem(all_vecs, 2, core::MissingStrategy::ZeroCost);
  auto prob_arow = make_problem(all_vecs, 2, core::MissingStrategy::AROW);

  prob_zero.verbose = false;
  prob_arow.verbose = false;

  REQUIRE_NOTHROW(prob_zero.fillDistanceMatrix());
  REQUIRE_NOTHROW(prob_arow.fillDistanceMatrix());

  const int N = static_cast<int>(prob_zero.size());

  // All AROW distances must be finite and >= ZeroCost distances (up to rounding).
  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      double d_zero = prob_zero.distByInd(i, j);
      double d_arow = prob_arow.distByInd(i, j);
      REQUIRE(std::isfinite(d_arow));
      REQUIRE(d_arow >= d_zero - 1e-9);
    }
  }

  // AROW pipeline: cluster, then compute all metrics
  prob_arow.maxIter = 20;
  prob_arow.N_repetition = 1;
  prob_arow.cluster();

  REQUIRE(prob_arow.clusters_ind.size() == static_cast<size_t>(N));

  double sil_mean = 0.0;
  auto sils = scores::silhouette(prob_arow);
  for (double s : sils) {
    REQUIRE(std::isfinite(s));
    sil_mean += s;
  }
  sil_mean /= static_cast<double>(N);

  REQUIRE(std::isfinite(scores::daviesBouldinIndex(prob_arow)));
  REQUIRE(std::isfinite(scores::dunnIndex(prob_arow)));
  REQUIRE(std::isfinite(scores::inertia(prob_arow)));
  REQUIRE(std::isfinite(scores::calinskiHarabaszIndex(prob_arow)));
}

// ---------------------------------------------------------------------------
// 3. Interpolate pipeline
// ---------------------------------------------------------------------------

TEST_CASE("Integration: Interpolate pipeline — finite distances and metrics",
          "[wave1a][integration][interpolate]")
{
  auto vecs_a = make_gaussian_series(10, 20, 0.0, 1.0, 0.10, 20);
  auto vecs_b = make_gaussian_series(10, 20, 15.0, 1.0, 0.10, 21);
  std::vector<std::vector<double>> all_vecs;
  all_vecs.insert(all_vecs.end(), vecs_a.begin(), vecs_a.end());
  all_vecs.insert(all_vecs.end(), vecs_b.begin(), vecs_b.end());

  auto prob = make_problem(all_vecs, 2, core::MissingStrategy::Interpolate);
  prob.maxIter = 20;
  prob.N_repetition = 1;
  prob.verbose = false;

  REQUIRE_NOTHROW(prob.fillDistanceMatrix());

  const int N = static_cast<int>(prob.size());
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      REQUIRE(std::isfinite(prob.distByInd(i, j)));

  prob.cluster();

  auto sils = scores::silhouette(prob);
  REQUIRE(all_finite(sils));

  REQUIRE(std::isfinite(scores::daviesBouldinIndex(prob)));
  REQUIRE(std::isfinite(scores::dunnIndex(prob)));
  REQUIRE(std::isfinite(scores::inertia(prob)));
  REQUIRE(std::isfinite(scores::calinskiHarabaszIndex(prob)));
}

// ---------------------------------------------------------------------------
// 4. Metrics agree on a clearly-separated two-group dataset
// ---------------------------------------------------------------------------

TEST_CASE("Integration: metrics agree on well-separated 2-cluster data",
          "[wave1a][integration][quality]")
{
  // 2 clearly separated groups, no NaN, 10 points each, series length 5.
  // Group A: all values near 0; Group B: all values near 100.
  // With 2 clusters the clustering should be near-perfect.

  const int n_each = 10;
  const int L = 5;
  const double sep = 100.0;

  auto vecs_a = make_gaussian_series(n_each, L, 0.0, 0.1, 0.0, 30);
  auto vecs_b = make_gaussian_series(n_each, L, sep, 0.1, 0.0, 31);

  std::vector<std::vector<double>> all_vecs;
  all_vecs.insert(all_vecs.end(), vecs_a.begin(), vecs_a.end());
  all_vecs.insert(all_vecs.end(), vecs_b.begin(), vecs_b.end());

  auto prob = make_problem(all_vecs, 2, core::MissingStrategy::Error);
  prob.maxIter = 50;
  prob.N_repetition = 3;
  prob.verbose = false;

  prob.fillDistanceMatrix();
  prob.cluster();

  const int N = 2 * n_each;

  // --- Silhouette should be very high (> 0.9 for perfectly separated data) ---
  auto sils = scores::silhouette(prob);
  double mean_sil = std::accumulate(sils.begin(), sils.end(), 0.0) / N;
  REQUIRE(mean_sil > 0.9);

  // --- Dunn should be large (large inter-cluster / small intra-cluster) ---
  double dunn = scores::dunnIndex(prob);
  REQUIRE(dunn > 10.0); // inter ~ 100, intra ~ 0.1*L at most

  // --- CH should be large ---
  double ch = scores::calinskiHarabaszIndex(prob);
  REQUIRE(ch > 10.0);

  // --- DBI should be small (< 0.1) ---
  double dbi = scores::daviesBouldinIndex(prob);
  REQUIRE(dbi < 0.1);

  // --- Inertia should be small relative to the total inter-cluster distance ---
  // Each cluster has 10 points, L1 DTW on length-5 series with stddev=0.1.
  // Expected per-pair DTW ~ L * stddev * sqrt(2/pi) ~ 5 * 0.1 * 0.8 ~ 0.4.
  // Inertia sums over 20 points => expected ~ 20 * 0.4 = 8.  Use 20 as safe upper bound.
  double inert = scores::inertia(prob);
  REQUIRE(inert < 20.0); // 20 points, very tight clusters
}

// ---------------------------------------------------------------------------
// 5. ARI / NMI with clustering output vs known ground truth
// ---------------------------------------------------------------------------

TEST_CASE("Integration: ARI and NMI > 0 for reasonable clustering of separated data",
          "[wave1a][integration][ari_nmi]")
{
  // Use the same clearly-separated dataset (no NaN).
  const int n_each = 10;
  const int L = 5;

  auto vecs_a = make_gaussian_series(n_each, L, 0.0, 0.5, 0.0, 40);
  auto vecs_b = make_gaussian_series(n_each, L, 50.0, 0.5, 0.0, 41);

  std::vector<std::vector<double>> all_vecs;
  all_vecs.insert(all_vecs.end(), vecs_a.begin(), vecs_a.end());
  all_vecs.insert(all_vecs.end(), vecs_b.begin(), vecs_b.end());

  auto prob = make_problem(all_vecs, 2, core::MissingStrategy::Error);
  prob.maxIter = 50;
  prob.N_repetition = 3;
  prob.verbose = false;

  prob.fillDistanceMatrix();
  prob.cluster();

  // Ground truth: first 10 points are group 0, last 10 are group 1.
  std::vector<int> ground_truth(2 * n_each);
  for (int i = 0; i < n_each; ++i) ground_truth[i] = 0;
  for (int i = n_each; i < 2 * n_each; ++i) ground_truth[i] = 1;

  // Predicted labels from clustering (may use different label integers — ARI/NMI are
  // invariant to permutation).
  const std::vector<int> &pred = prob.clusters_ind;

  double ari = scores::adjustedRandIndex(ground_truth, pred);
  double nmi = scores::normalizedMutualInformation(ground_truth, pred);

  // For well-separated data, ARI and NMI must be high.
  REQUIRE(ari > 0.5);
  REQUIRE(nmi > 0.5);

  // Bounds check
  REQUIRE(ari <= 1.0 + 1e-9);
  REQUIRE(nmi >= 0.0);
  REQUIRE(nmi <= 1.0 + 1e-9);
}

// ---------------------------------------------------------------------------
// 6. Error strategy works on clean data; throws on NaN data
// ---------------------------------------------------------------------------

TEST_CASE("Integration: MissingStrategy::Error — clean data OK, NaN data throws",
          "[wave1a][integration][error_strategy]")
{
  // Clean data: should work fine
  auto vecs_clean = make_gaussian_series(10, 15, 0.0, 1.0, 0.0, 50);
  auto prob_clean = make_problem(vecs_clean, 2, core::MissingStrategy::Error);
  prob_clean.verbose = false;
  REQUIRE_NOTHROW(prob_clean.fillDistanceMatrix());
  REQUIRE(prob_clean.isDistanceMatrixFilled());

  // Data with NaN: Error strategy should throw
  auto vecs_nan = make_gaussian_series(10, 15, 0.0, 1.0, 0.10, 51);
  auto prob_nan = make_problem(vecs_nan, 2, core::MissingStrategy::Error);
  prob_nan.verbose = false;
  REQUIRE_THROWS(prob_nan.fillDistanceMatrix());
}

// ---------------------------------------------------------------------------
// 7. ARI/NMI with ZeroCost clustering output
// ---------------------------------------------------------------------------

TEST_CASE("Integration: ZeroCost clustering — ARI and NMI > 0",
          "[wave1a][integration][zerocost_ari_nmi]")
{
  const int n_each = 10;
  const int L = 10;

  auto vecs_a = make_gaussian_series(n_each, L, 0.0, 0.5, 0.10, 60);
  auto vecs_b = make_gaussian_series(n_each, L, 30.0, 0.5, 0.10, 61);

  std::vector<std::vector<double>> all_vecs;
  all_vecs.insert(all_vecs.end(), vecs_a.begin(), vecs_a.end());
  all_vecs.insert(all_vecs.end(), vecs_b.begin(), vecs_b.end());

  auto prob = make_problem(all_vecs, 2, core::MissingStrategy::ZeroCost);
  prob.maxIter = 30;
  prob.N_repetition = 2;
  prob.verbose = false;

  prob.fillDistanceMatrix();
  prob.cluster();

  std::vector<int> ground_truth(2 * n_each);
  for (int i = 0; i < n_each; ++i) ground_truth[i] = 0;
  for (int i = n_each; i < 2 * n_each; ++i) ground_truth[i] = 1;

  double ari = scores::adjustedRandIndex(ground_truth, prob.clusters_ind);
  double nmi = scores::normalizedMutualInformation(ground_truth, prob.clusters_ind);

  REQUIRE(ari > 0.0);
  REQUIRE(nmi > 0.0);
}

// ---------------------------------------------------------------------------
// 8. Performance sanity: ZeroCost vs AROW vs Interpolate
//    (not a pass/fail benchmark — just prints timings)
// ---------------------------------------------------------------------------

TEST_CASE("Performance sanity: 1000 pairwise DTW on length-100 series with 10% NaN",
          "[wave1a][integration][perf]")
{
  constexpr int N_pairs = 1000;
  constexpr int L = 100;
  constexpr double nan_rate = 0.10;

  // Generate N_pairs pairs of series
  auto xs = make_gaussian_series(N_pairs, L, 0.0, 1.0, nan_rate, 100);
  auto ys = make_gaussian_series(N_pairs, L, 1.0, 1.0, nan_rate, 200);

  using Clock = std::chrono::high_resolution_clock;
  using Ms = std::chrono::duration<double, std::milli>;

  // --- ZeroCost ---
  double zerocost_total = 0.0;
  auto t0 = Clock::now();
  for (int i = 0; i < N_pairs; ++i)
    zerocost_total += dtwMissing_L<double>(xs[i], ys[i]);
  auto t1 = Clock::now();
  double ms_zero = Ms(t1 - t0).count();

  // --- AROW ---
  double arow_total = 0.0;
  t0 = Clock::now();
  for (int i = 0; i < N_pairs; ++i)
    arow_total += dtwAROW_L<double>(xs[i], ys[i]);
  t1 = Clock::now();
  double ms_arow = Ms(t1 - t0).count();

  // --- Interpolate: preprocess then standard DTW ---
  double interp_total = 0.0;
  t0 = Clock::now();
  for (int i = 0; i < N_pairs; ++i) {
    auto xi = interpolate_linear(xs[i]);
    auto yi = interpolate_linear(ys[i]);
    interp_total += dtwFull_L<double>(xi, yi);
  }
  t1 = Clock::now();
  double ms_interp = Ms(t1 - t0).count();

  std::cout << "\n[perf] " << N_pairs << " pairs × length " << L
            << " series, " << (nan_rate * 100) << "% NaN:\n"
            << "  ZeroCost  : " << ms_zero << " ms  (sum=" << zerocost_total << ")\n"
            << "  AROW      : " << ms_arow << " ms  (sum=" << arow_total << ")\n"
            << "  Interpolate: " << ms_interp << " ms  (sum=" << interp_total << ")\n";

  // All results must be finite
  REQUIRE(std::isfinite(zerocost_total));
  REQUIRE(std::isfinite(arow_total));
  REQUIRE(std::isfinite(interp_total));

  // AROW >= ZeroCost in aggregate (not guaranteed pair-by-pair due to summation
  // but the property holds mathematically — here we verify it holds in sum).
  REQUIRE(arow_total >= zerocost_total - 1e-6);

  // AROW should not be more than 20x slower than ZeroCost (same O(n*m) loop).
  // This is a very loose bound to avoid CI flakiness; on real hardware the
  // ratio is typically < 2x.
  REQUIRE(ms_arow < ms_zero * 20.0 + 100.0); // +100ms guard for measurement noise
}
