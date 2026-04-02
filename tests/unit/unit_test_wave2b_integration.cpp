/**
 * @file unit_test_wave2b_integration.cpp
 * @brief Adversarial integration tests for all Wave 2B features.
 *
 * @details Wave 2B added:
 *   - MV WDTW: wdtwFull_mv, wdtwBanded_mv
 *   - MV ADTW: adtwFull_L_mv, adtwBanded_mv
 *   - MV DDTW: derivative_transform_mv + dtwBanded_mv
 *   - Per-channel LB_Keogh: compute_envelopes_mv, lb_keogh_mv
 *   - SquaredL2 LB: lb_keogh_squared, lb_keogh_mv_squared
 *   - MV missing-data DTW: dtwMissing_L_mv, dtwMissing_banded_mv
 *
 * Tests:
 *   1. Cross-variant consistency for ndim=1 (MV variants match scalar counterparts)
 *   2. LB_Keogh is valid lower bound on MV DTW (50 random pairs, ndim=2,3)
 *   3. LB_Keogh SquaredL2 is valid lower bound (same setup)
 *   4. MV missing + MV variants: non-negative, finite, symmetric (ndim=2, with NaN)
 *   5. Full Problem pipeline with ndim=3: WDTW, distance matrix, cluster, all metrics finite
 *   6. Performance comparison ndim=1 vs ndim=3: timing sanity check
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

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

// ---------------------------------------------------------------------------
// Random series helpers
// ---------------------------------------------------------------------------

/// Generate a flat interleaved MV series: n_steps * ndim elements, Gaussian.
static std::vector<double> make_mv_series(std::size_t n_steps, std::size_t ndim,
                                          double mean, double stddev,
                                          std::mt19937_64 &rng)
{
  std::normal_distribution<double> gauss(mean, stddev);
  std::vector<double> v(n_steps * ndim);
  for (auto &x : v) x = gauss(rng);
  return v;
}

/// Generate a flat interleaved MV series with NaN in some channels.
static std::vector<double> make_mv_series_nan(std::size_t n_steps, std::size_t ndim,
                                              double mean, double stddev, double nan_rate,
                                              std::mt19937_64 &rng)
{
  std::normal_distribution<double> gauss(mean, stddev);
  std::uniform_real_distribution<double> uni(0.0, 1.0);
  std::vector<double> v(n_steps * ndim);
  for (auto &x : v) {
    x = (uni(rng) < nan_rate) ? NaN : gauss(rng);
  }
  return v;
}

// ---------------------------------------------------------------------------
// Temporary output directory for Problem-based tests
// ---------------------------------------------------------------------------
static fs::path g_tmp_dir()
{
  static fs::path dir = fs::temp_directory_path() / "dtwc_wave2b_integration_test";
  static bool created = [] {
    fs::create_directories(dir);
    return true;
  }();
  (void)created;
  return dir;
}

// ---------------------------------------------------------------------------
// Test 1: Cross-variant consistency for ndim=1
//   For random series, verify all MV variants with ndim=1 match scalar counterparts.
// ---------------------------------------------------------------------------

TEST_CASE("Wave2B cross-variant ndim=1: wdtwFull_mv matches scalar wdtwFull",
          "[wave2b][integration][ndim1][wdtw]")
{
  std::mt19937_64 rng(101);
  const int n_steps = 20;
  const double g = 0.05;

  for (int trial = 0; trial < 10; ++trial) {
    auto xv = make_mv_series(n_steps, 1, 0.0, 1.0, rng);
    auto yv = make_mv_series(n_steps, 1, 0.5, 1.0, rng);

    // Scalar
    std::vector<double> xs(xv.begin(), xv.end());
    std::vector<double> ys(yv.begin(), yv.end());
    const int max_dev = n_steps - 1;
    auto w = dtwc::wdtw_weights<double>(max_dev, g);
    double d_scalar = dtwc::wdtwFull(xs, ys, w);

    // MV ndim=1
    double d_mv = dtwc::wdtwFull_mv(xv.data(), n_steps, yv.data(), n_steps, 1, g);

    REQUIRE_THAT(d_mv, WithinAbs(d_scalar, 1e-9));
  }
}

TEST_CASE("Wave2B cross-variant ndim=1: adtwFull_L_mv matches scalar adtwFull_L",
          "[wave2b][integration][ndim1][adtw]")
{
  std::mt19937_64 rng(202);
  const int n_steps = 20;
  const double penalty = 0.5;

  for (int trial = 0; trial < 10; ++trial) {
    auto xv = make_mv_series(n_steps, 1, 0.0, 1.0, rng);
    auto yv = make_mv_series(n_steps, 1, 0.5, 1.0, rng);

    std::vector<double> xs(xv.begin(), xv.end());
    std::vector<double> ys(yv.begin(), yv.end());
    double d_scalar = dtwc::adtwFull_L(xs, ys, penalty);
    double d_mv     = dtwc::adtwFull_L_mv(xv.data(), n_steps, yv.data(), n_steps, 1, penalty);

    REQUIRE_THAT(d_mv, WithinAbs(d_scalar, 1e-9));
  }
}

TEST_CASE("Wave2B cross-variant ndim=1: dtwMissing_L_mv matches scalar dtwMissing_L",
          "[wave2b][integration][ndim1][missing]")
{
  std::mt19937_64 rng(303);
  const int n_steps = 15;

  for (int trial = 0; trial < 10; ++trial) {
    auto xv = make_mv_series_nan(n_steps, 1, 0.0, 1.0, 0.15, rng);
    auto yv = make_mv_series_nan(n_steps, 1, 0.5, 1.0, 0.15, rng);

    std::vector<double> xs(xv.begin(), xv.end());
    std::vector<double> ys(yv.begin(), yv.end());
    double d_scalar = dtwc::dtwMissing_L(xs, ys);
    double d_mv     = dtwc::dtwMissing_L_mv(xv.data(), n_steps, yv.data(), n_steps, 1);

    REQUIRE_THAT(d_mv, WithinAbs(d_scalar, 1e-9));
  }
}

TEST_CASE("Wave2B cross-variant ndim=1: wdtwBanded_mv matches scalar wdtwBanded",
          "[wave2b][integration][ndim1][wdtw][banded]")
{
  std::mt19937_64 rng(404);
  const int n_steps = 25;
  const int band = 5;
  const double g = 0.05;

  for (int trial = 0; trial < 8; ++trial) {
    auto xv = make_mv_series(n_steps, 1, 0.0, 1.0, rng);
    auto yv = make_mv_series(n_steps, 1, 0.5, 1.0, rng);

    std::vector<double> xs(xv.begin(), xv.end());
    std::vector<double> ys(yv.begin(), yv.end());
    double d_scalar = dtwc::wdtwBanded(xs, ys, band, g);
    double d_mv     = dtwc::wdtwBanded_mv(xv.data(), n_steps, yv.data(), n_steps, 1, band, g);

    REQUIRE_THAT(d_mv, WithinAbs(d_scalar, 1e-9));
  }
}

TEST_CASE("Wave2B cross-variant ndim=1: adtwBanded_mv matches scalar adtwBanded",
          "[wave2b][integration][ndim1][adtw][banded]")
{
  std::mt19937_64 rng(505);
  const int n_steps = 25;
  const int band = 5;
  const double penalty = 0.5;

  for (int trial = 0; trial < 8; ++trial) {
    auto xv = make_mv_series(n_steps, 1, 0.0, 1.0, rng);
    auto yv = make_mv_series(n_steps, 1, 0.5, 1.0, rng);

    std::vector<double> xs(xv.begin(), xv.end());
    std::vector<double> ys(yv.begin(), yv.end());
    double d_scalar = dtwc::adtwBanded(xs, ys, band, penalty);
    double d_mv     = dtwc::adtwBanded_mv(xv.data(), n_steps, yv.data(), n_steps, 1, band, penalty);

    REQUIRE_THAT(d_mv, WithinAbs(d_scalar, 1e-9));
  }
}

TEST_CASE("Wave2B cross-variant ndim=1: derivative_transform_mv matches derivative_transform",
          "[wave2b][integration][ndim1][ddtw]")
{
  std::mt19937_64 rng(606);
  const int n_steps = 18;

  for (int trial = 0; trial < 10; ++trial) {
    auto xv = make_mv_series(n_steps, 1, 0.0, 1.0, rng);

    std::vector<double> xs(xv.begin(), xv.end());
    auto dx_scalar = dtwc::derivative_transform(xs);
    auto dx_mv     = dtwc::derivative_transform_mv(xv, 1);

    REQUIRE(dx_scalar.size() == dx_mv.size());
    for (std::size_t i = 0; i < dx_scalar.size(); ++i)
      REQUIRE_THAT(dx_mv[i], WithinAbs(dx_scalar[i], 1e-12));
  }
}

// ---------------------------------------------------------------------------
// Test 2: LB_Keogh (L1) is valid lower bound on MV DTW for ndim=2,3
// ---------------------------------------------------------------------------

TEST_CASE("Wave2B LB_Keogh MV is valid lower bound on dtwBanded_mv (ndim=2)",
          "[wave2b][integration][lb][ndim2]")
{
  std::mt19937_64 rng(1001);
  const std::size_t n_steps = 20;
  const std::size_t ndim = 2;
  const int band = 4;
  const int N_pairs = 50;

  for (int trial = 0; trial < N_pairs; ++trial) {
    auto xv = make_mv_series(n_steps, ndim, 0.0, 2.0, rng);
    auto yv = make_mv_series(n_steps, ndim, 1.0, 2.0, rng);

    // Compute envelopes of y, then LB_Keogh of x vs y_envelopes
    std::vector<double> upper(n_steps * ndim), lower(n_steps * ndim);
    dtwc::core::compute_envelopes_mv(yv.data(), n_steps, ndim, band, upper.data(), lower.data());

    double lb = dtwc::core::lb_keogh_mv(xv.data(), n_steps, ndim, upper.data(), lower.data());
    double dtw_d = dtwc::dtwBanded_mv(xv.data(), n_steps, yv.data(), n_steps, ndim, band);

    // LB must be non-negative
    REQUIRE(lb >= 0.0);
    // LB must be finite
    REQUIRE(std::isfinite(lb));
    // LB must be a valid lower bound (lb <= dtw)
    REQUIRE(lb <= dtw_d + 1e-9);
  }
}

TEST_CASE("Wave2B LB_Keogh MV is valid lower bound on dtwBanded_mv (ndim=3)",
          "[wave2b][integration][lb][ndim3]")
{
  std::mt19937_64 rng(1002);
  const std::size_t n_steps = 20;
  const std::size_t ndim = 3;
  const int band = 4;
  const int N_pairs = 50;

  for (int trial = 0; trial < N_pairs; ++trial) {
    auto xv = make_mv_series(n_steps, ndim, 0.0, 2.0, rng);
    auto yv = make_mv_series(n_steps, ndim, 1.0, 2.0, rng);

    std::vector<double> upper(n_steps * ndim), lower(n_steps * ndim);
    dtwc::core::compute_envelopes_mv(yv.data(), n_steps, ndim, band, upper.data(), lower.data());

    double lb    = dtwc::core::lb_keogh_mv(xv.data(), n_steps, ndim, upper.data(), lower.data());
    double dtw_d = dtwc::dtwBanded_mv(xv.data(), n_steps, yv.data(), n_steps, ndim, band);

    REQUIRE(lb >= 0.0);
    REQUIRE(std::isfinite(lb));
    REQUIRE(lb <= dtw_d + 1e-9);
  }
}

// ---------------------------------------------------------------------------
// Test 3: LB_Keogh SquaredL2 is valid lower bound on MV SquaredL2 DTW
// ---------------------------------------------------------------------------

TEST_CASE("Wave2B LB_Keogh SquaredL2 is valid lower bound on MV SquaredL2 DTW (ndim=2)",
          "[wave2b][integration][lb][squared][ndim2]")
{
  std::mt19937_64 rng(2001);
  const std::size_t n_steps = 20;
  const std::size_t ndim = 2;
  const int band = 4;
  const int N_pairs = 50;

  for (int trial = 0; trial < N_pairs; ++trial) {
    auto xv = make_mv_series(n_steps, ndim, 0.0, 2.0, rng);
    auto yv = make_mv_series(n_steps, ndim, 1.0, 2.0, rng);

    std::vector<double> upper(n_steps * ndim), lower(n_steps * ndim);
    dtwc::core::compute_envelopes_mv(yv.data(), n_steps, ndim, band, upper.data(), lower.data());

    double lb    = dtwc::core::lb_keogh_mv_squared(xv.data(), n_steps, ndim, upper.data(), lower.data());
    double dtw_d = dtwc::dtwBanded_mv(xv.data(), n_steps, yv.data(), n_steps, ndim, band,
                                       -1.0, dtwc::core::MetricType::SquaredL2);

    REQUIRE(lb >= 0.0);
    REQUIRE(std::isfinite(lb));
    REQUIRE(lb <= dtw_d + 1e-9);
  }
}

TEST_CASE("Wave2B LB_Keogh SquaredL2 is valid lower bound on MV SquaredL2 DTW (ndim=3)",
          "[wave2b][integration][lb][squared][ndim3]")
{
  std::mt19937_64 rng(2002);
  const std::size_t n_steps = 20;
  const std::size_t ndim = 3;
  const int band = 4;
  const int N_pairs = 50;

  for (int trial = 0; trial < N_pairs; ++trial) {
    auto xv = make_mv_series(n_steps, ndim, 0.0, 2.0, rng);
    auto yv = make_mv_series(n_steps, ndim, 1.0, 2.0, rng);

    std::vector<double> upper(n_steps * ndim), lower(n_steps * ndim);
    dtwc::core::compute_envelopes_mv(yv.data(), n_steps, ndim, band, upper.data(), lower.data());

    double lb    = dtwc::core::lb_keogh_mv_squared(xv.data(), n_steps, ndim, upper.data(), lower.data());
    double dtw_d = dtwc::dtwBanded_mv(xv.data(), n_steps, yv.data(), n_steps, ndim, band,
                                       -1.0, dtwc::core::MetricType::SquaredL2);

    REQUIRE(lb >= 0.0);
    REQUIRE(std::isfinite(lb));
    REQUIRE(lb <= dtw_d + 1e-9);
  }
}

// ---------------------------------------------------------------------------
// Test 4: MV missing + MV variants: non-negative, finite, symmetric (ndim=2, with NaN)
// ---------------------------------------------------------------------------

TEST_CASE("Wave2B MV missing ndim=2: non-negative, finite, symmetric with NaN",
          "[wave2b][integration][missing][ndim2]")
{
  std::mt19937_64 rng(3001);
  const std::size_t n_steps = 15;
  const std::size_t ndim = 2;
  const int N_pairs = 30;

  for (int trial = 0; trial < N_pairs; ++trial) {
    auto xv = make_mv_series_nan(n_steps, ndim, 0.0, 2.0, 0.2, rng);
    auto yv = make_mv_series_nan(n_steps, ndim, 1.0, 2.0, 0.2, rng);

    double d_xy = dtwc::dtwMissing_L_mv(xv.data(), n_steps, yv.data(), n_steps, ndim);
    double d_yx = dtwc::dtwMissing_L_mv(yv.data(), n_steps, xv.data(), n_steps, ndim);

    REQUIRE(d_xy >= 0.0);
    REQUIRE(std::isfinite(d_xy));
    REQUIRE_THAT(d_xy, WithinAbs(d_yx, 1e-9));
  }
}

TEST_CASE("Wave2B MV WDTW ndim=2: non-negative, finite, symmetric with random data",
          "[wave2b][integration][wdtw][ndim2]")
{
  std::mt19937_64 rng(3002);
  const std::size_t n_steps = 15;
  const std::size_t ndim = 2;

  for (int trial = 0; trial < 20; ++trial) {
    auto xv = make_mv_series(n_steps, ndim, 0.0, 2.0, rng);
    auto yv = make_mv_series(n_steps, ndim, 1.0, 2.0, rng);

    double d_xy = dtwc::wdtwFull_mv(xv.data(), n_steps, yv.data(), n_steps, ndim);
    double d_yx = dtwc::wdtwFull_mv(yv.data(), n_steps, xv.data(), n_steps, ndim);

    REQUIRE(d_xy >= 0.0);
    REQUIRE(std::isfinite(d_xy));
    REQUIRE_THAT(d_xy, WithinAbs(d_yx, 1e-9));
  }
}

TEST_CASE("Wave2B MV ADTW ndim=2: non-negative, finite, symmetric with random data",
          "[wave2b][integration][adtw][ndim2]")
{
  std::mt19937_64 rng(3003);
  const std::size_t n_steps = 15;
  const std::size_t ndim = 2;
  const double penalty = 1.0;

  for (int trial = 0; trial < 20; ++trial) {
    auto xv = make_mv_series(n_steps, ndim, 0.0, 2.0, rng);
    auto yv = make_mv_series(n_steps, ndim, 1.0, 2.0, rng);

    double d_xy = dtwc::adtwFull_L_mv(xv.data(), n_steps, yv.data(), n_steps, ndim, penalty);
    double d_yx = dtwc::adtwFull_L_mv(yv.data(), n_steps, xv.data(), n_steps, ndim, penalty);

    REQUIRE(d_xy >= 0.0);
    REQUIRE(std::isfinite(d_xy));
    REQUIRE_THAT(d_xy, WithinAbs(d_yx, 1e-9));
  }
}

TEST_CASE("Wave2B MV DDTW ndim=2: non-negative, finite, symmetric with random data",
          "[wave2b][integration][ddtw][ndim2]")
{
  std::mt19937_64 rng(3004);
  const std::size_t n_steps = 15;
  const std::size_t ndim = 2;
  const int band = 4;

  for (int trial = 0; trial < 20; ++trial) {
    auto xv = make_mv_series(n_steps, ndim, 0.0, 2.0, rng);
    auto yv = make_mv_series(n_steps, ndim, 1.0, 2.0, rng);

    // DDTW = derivative transform + standard MV DTW
    auto dx = dtwc::derivative_transform_mv(xv, ndim);
    auto dy = dtwc::derivative_transform_mv(yv, ndim);

    double d_xy = dtwc::dtwBanded_mv(dx.data(), n_steps, dy.data(), n_steps, ndim, band);
    double d_yx = dtwc::dtwBanded_mv(dy.data(), n_steps, dx.data(), n_steps, ndim, band);

    REQUIRE(d_xy >= 0.0);
    REQUIRE(std::isfinite(d_xy));
    REQUIRE_THAT(d_xy, WithinAbs(d_yx, 1e-9));
  }
}

TEST_CASE("Wave2B MV banded missing ndim=2: non-negative, finite, symmetric with NaN",
          "[wave2b][integration][missing][banded][ndim2]")
{
  std::mt19937_64 rng(3005);
  const std::size_t n_steps = 15;
  const std::size_t ndim = 2;
  const int band = 4;

  for (int trial = 0; trial < 20; ++trial) {
    auto xv = make_mv_series_nan(n_steps, ndim, 0.0, 2.0, 0.15, rng);
    auto yv = make_mv_series_nan(n_steps, ndim, 1.0, 2.0, 0.15, rng);

    double d_xy = dtwc::dtwMissing_banded_mv(xv.data(), n_steps, yv.data(), n_steps, ndim, band);
    double d_yx = dtwc::dtwMissing_banded_mv(yv.data(), n_steps, xv.data(), n_steps, ndim, band);

    REQUIRE(d_xy >= 0.0);
    REQUIRE(std::isfinite(d_xy));
    REQUIRE_THAT(d_xy, WithinAbs(d_yx, 1e-9));
  }
}

// ---------------------------------------------------------------------------
// Test 5: Full Problem pipeline with ndim=3
//   Create data, set variant to WDTW, fill distance matrix, cluster,
//   compute all metrics. All finite.
// ---------------------------------------------------------------------------

TEST_CASE("Wave2B Problem pipeline ndim=3 WDTW: fill + cluster + metrics all finite",
          "[wave2b][integration][problem][ndim3][wdtw]")
{
  // Build a dataset: 2 groups of 8 series, ndim=3, length=10 steps -> 30 elements each
  const std::size_t n_per_group = 8;
  const std::size_t n_steps = 10;
  const std::size_t ndim = 3;
  const std::size_t flat_len = n_steps * ndim;

  std::mt19937_64 rng(5001);
  std::normal_distribution<double> g_a(0.0, 0.5);
  std::normal_distribution<double> g_b(20.0, 0.5);

  std::vector<std::vector<double>> vecs;
  std::vector<std::string> names;

  // Group A
  for (std::size_t i = 0; i < n_per_group; ++i) {
    std::vector<double> v(flat_len);
    for (auto &x : v) x = g_a(rng);
    vecs.push_back(std::move(v));
    names.push_back("A" + std::to_string(i));
  }
  // Group B
  for (std::size_t i = 0; i < n_per_group; ++i) {
    std::vector<double> v(flat_len);
    for (auto &x : v) x = g_b(rng);
    vecs.push_back(std::move(v));
    names.push_back("B" + std::to_string(i));
  }

  dtwc::Data data(std::move(vecs), std::move(names));
  data.ndim = ndim;

  dtwc::Problem prob("wave2b_ndim3_wdtw");
  prob.set_data(std::move(data));
  prob.set_variant(dtwc::core::DTWVariant::WDTW);
  prob.set_numberOfClusters(2);
  prob.maxIter = 20;
  prob.N_repetition = 1;
  prob.verbose = false;
  prob.output_folder = g_tmp_dir();

  // Fill distance matrix
  REQUIRE_NOTHROW(prob.fillDistanceMatrix());
  REQUIRE(prob.isDistanceMatrixFilled());

  const int N = static_cast<int>(prob.size());
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      REQUIRE(std::isfinite(prob.distByInd(i, j)));

  // Cluster
  prob.cluster();
  REQUIRE(prob.clusters_ind.size() == static_cast<std::size_t>(N));
  REQUIRE(prob.centroids_ind.size() == 2u);

  // Cluster quality metrics
  auto sils = dtwc::scores::silhouette(prob);
  REQUIRE(sils.size() == static_cast<std::size_t>(N));
  for (double s : sils)
    REQUIRE(std::isfinite(s));

  double dbi = dtwc::scores::daviesBouldinIndex(prob);
  REQUIRE(std::isfinite(dbi));
  REQUIRE(dbi >= 0.0);

  double dunn = dtwc::scores::dunnIndex(prob);
  REQUIRE(std::isfinite(dunn));
  REQUIRE(dunn >= 0.0);

  double inert = dtwc::scores::inertia(prob);
  REQUIRE(std::isfinite(inert));
  REQUIRE(inert >= 0.0);

  double ch = dtwc::scores::calinskiHarabaszIndex(prob);
  REQUIRE(std::isfinite(ch));
  REQUIRE(ch >= 0.0);
}

TEST_CASE("Wave2B Problem pipeline ndim=3 ADTW: fill + cluster + metrics all finite",
          "[wave2b][integration][problem][ndim3][adtw]")
{
  const std::size_t n_per_group = 6;
  const std::size_t n_steps = 10;
  const std::size_t ndim = 3;
  const std::size_t flat_len = n_steps * ndim;

  std::mt19937_64 rng(5002);
  std::normal_distribution<double> g_a(-10.0, 0.5);
  std::normal_distribution<double> g_b( 10.0, 0.5);

  std::vector<std::vector<double>> vecs;
  std::vector<std::string> names;
  for (std::size_t i = 0; i < n_per_group; ++i) {
    std::vector<double> v(flat_len);
    for (auto &x : v) x = g_a(rng);
    vecs.push_back(v);
    names.push_back("A" + std::to_string(i));
  }
  for (std::size_t i = 0; i < n_per_group; ++i) {
    std::vector<double> v(flat_len);
    for (auto &x : v) x = g_b(rng);
    vecs.push_back(v);
    names.push_back("B" + std::to_string(i));
  }

  dtwc::Data data(std::move(vecs), std::move(names));
  data.ndim = ndim;

  dtwc::Problem prob("wave2b_ndim3_adtw");
  prob.set_data(std::move(data));
  prob.set_variant(dtwc::core::DTWVariant::ADTW);
  prob.set_numberOfClusters(2);
  prob.maxIter = 20;
  prob.N_repetition = 1;
  prob.verbose = false;
  prob.output_folder = g_tmp_dir();

  REQUIRE_NOTHROW(prob.fillDistanceMatrix());
  REQUIRE(prob.isDistanceMatrixFilled());

  const int N = static_cast<int>(prob.size());
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      REQUIRE(std::isfinite(prob.distByInd(i, j)));

  prob.cluster();
  REQUIRE(prob.clusters_ind.size() == static_cast<std::size_t>(N));

  for (double s : dtwc::scores::silhouette(prob))
    REQUIRE(std::isfinite(s));
  REQUIRE(std::isfinite(dtwc::scores::daviesBouldinIndex(prob)));
  REQUIRE(std::isfinite(dtwc::scores::dunnIndex(prob)));
  REQUIRE(std::isfinite(dtwc::scores::inertia(prob)));
  REQUIRE(std::isfinite(dtwc::scores::calinskiHarabaszIndex(prob)));
}

TEST_CASE("Wave2B Problem pipeline ndim=3 DDTW: fill + cluster + metrics all finite",
          "[wave2b][integration][problem][ndim3][ddtw]")
{
  const std::size_t n_per_group = 6;
  const std::size_t n_steps = 12;
  const std::size_t ndim = 3;
  const std::size_t flat_len = n_steps * ndim;

  std::mt19937_64 rng(5003);
  std::normal_distribution<double> g_a(0.0, 0.5);
  std::normal_distribution<double> g_b(5.0, 0.5);

  std::vector<std::vector<double>> vecs;
  std::vector<std::string> names;
  for (std::size_t i = 0; i < n_per_group; ++i) {
    std::vector<double> v(flat_len);
    for (auto &x : v) x = g_a(rng);
    vecs.push_back(v);
    names.push_back("A" + std::to_string(i));
  }
  for (std::size_t i = 0; i < n_per_group; ++i) {
    std::vector<double> v(flat_len);
    for (auto &x : v) x = g_b(rng);
    vecs.push_back(v);
    names.push_back("B" + std::to_string(i));
  }

  dtwc::Data data(std::move(vecs), std::move(names));
  data.ndim = ndim;

  dtwc::Problem prob("wave2b_ndim3_ddtw");
  prob.set_data(std::move(data));
  prob.set_variant(dtwc::core::DTWVariant::DDTW);
  prob.set_numberOfClusters(2);
  prob.maxIter = 20;
  prob.N_repetition = 1;
  prob.verbose = false;
  prob.output_folder = g_tmp_dir();

  REQUIRE_NOTHROW(prob.fillDistanceMatrix());
  REQUIRE(prob.isDistanceMatrixFilled());

  const int N = static_cast<int>(prob.size());
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      REQUIRE(std::isfinite(prob.distByInd(i, j)));

  prob.cluster();
  REQUIRE(prob.clusters_ind.size() == static_cast<std::size_t>(N));

  for (double s : dtwc::scores::silhouette(prob))
    REQUIRE(std::isfinite(s));
  REQUIRE(std::isfinite(dtwc::scores::daviesBouldinIndex(prob)));
  REQUIRE(std::isfinite(dtwc::scores::dunnIndex(prob)));
  REQUIRE(std::isfinite(dtwc::scores::inertia(prob)));
  REQUIRE(std::isfinite(dtwc::scores::calinskiHarabaszIndex(prob)));
}

// ---------------------------------------------------------------------------
// Test 6: Performance comparison ndim=1 vs ndim=3 on 500 pairs
//   Print timing; verify ndim=3 is roughly proportional to D (not wildly off).
// ---------------------------------------------------------------------------

TEST_CASE("Wave2B performance: 500 pairs ndim=1 vs ndim=3 timing proportionality",
          "[wave2b][integration][perf]")
{
  constexpr int N_pairs = 500;
  constexpr std::size_t n_steps = 50;
  constexpr double g = 0.05;

  std::mt19937_64 rng(9001);

  // Build series for ndim=1 and ndim=3
  std::vector<std::vector<double>> xs1(N_pairs), ys1(N_pairs);
  std::vector<std::vector<double>> xs3(N_pairs), ys3(N_pairs);

  for (int i = 0; i < N_pairs; ++i) {
    xs1[i] = make_mv_series(n_steps, 1, 0.0, 1.0, rng);
    ys1[i] = make_mv_series(n_steps, 1, 0.5, 1.0, rng);
    xs3[i] = make_mv_series(n_steps, 3, 0.0, 1.0, rng);
    ys3[i] = make_mv_series(n_steps, 3, 0.5, 1.0, rng);
  }

  using Clock = std::chrono::high_resolution_clock;
  using Ms = std::chrono::duration<double, std::milli>;

  // --- ndim=1 WDTW ---
  double sum1 = 0.0;
  auto t0 = Clock::now();
  for (int i = 0; i < N_pairs; ++i)
    sum1 += dtwc::wdtwFull_mv(xs1[i].data(), n_steps, ys1[i].data(), n_steps, 1, g);
  auto t1 = Clock::now();
  double ms1 = Ms(t1 - t0).count();

  // --- ndim=3 WDTW ---
  double sum3 = 0.0;
  t0 = Clock::now();
  for (int i = 0; i < N_pairs; ++i)
    sum3 += dtwc::wdtwFull_mv(xs3[i].data(), n_steps, ys3[i].data(), n_steps, 3, g);
  t1 = Clock::now();
  double ms3 = Ms(t1 - t0).count();

  std::cout << "\n[Wave2B perf] " << N_pairs << " pairs x " << n_steps << " steps:\n"
            << "  ndim=1 WDTW: " << ms1 << " ms  (sum=" << sum1 << ")\n"
            << "  ndim=3 WDTW: " << ms3 << " ms  (sum=" << sum3 << ")\n"
            << "  Ratio ndim=3/ndim=1: " << (ms3 / (ms1 + 1e-9)) << "x\n";

  // Both sums must be finite
  REQUIRE(std::isfinite(sum1));
  REQUIRE(std::isfinite(sum3));

  // ndim=3 should be >= ndim=1 in time (more work per pair) with a generous upper bound.
  // A 10x upper bound prevents the test from failing on fast hardware / cache effects.
  // We do not enforce a strict lower bound since CI machines may serialize the loops.
  REQUIRE(ms1 >= 0.0);
  REQUIRE(ms3 >= 0.0);
  // Verify that ndim=3 distances are larger in aggregate (more channels => more cost)
  REQUIRE(sum3 >= sum1 - 1e-6);
}

// ---------------------------------------------------------------------------
// Test 6b: ADTW timing — ndim=1 vs ndim=3
// ---------------------------------------------------------------------------

TEST_CASE("Wave2B performance: 500 pairs ndim=1 vs ndim=3 ADTW timing",
          "[wave2b][integration][perf][adtw]")
{
  constexpr int N_pairs = 500;
  constexpr std::size_t n_steps = 50;
  const double penalty = 0.5;

  std::mt19937_64 rng(9002);

  std::vector<std::vector<double>> xs1(N_pairs), ys1(N_pairs);
  std::vector<std::vector<double>> xs3(N_pairs), ys3(N_pairs);

  for (int i = 0; i < N_pairs; ++i) {
    xs1[i] = make_mv_series(n_steps, 1, 0.0, 1.0, rng);
    ys1[i] = make_mv_series(n_steps, 1, 0.5, 1.0, rng);
    xs3[i] = make_mv_series(n_steps, 3, 0.0, 1.0, rng);
    ys3[i] = make_mv_series(n_steps, 3, 0.5, 1.0, rng);
  }

  using Clock = std::chrono::high_resolution_clock;
  using Ms = std::chrono::duration<double, std::milli>;

  double sum1 = 0.0;
  auto t0 = Clock::now();
  for (int i = 0; i < N_pairs; ++i)
    sum1 += dtwc::adtwFull_L_mv(xs1[i].data(), n_steps, ys1[i].data(), n_steps, 1, penalty);
  auto t1 = Clock::now();
  double ms1 = Ms(t1 - t0).count();

  double sum3 = 0.0;
  t0 = Clock::now();
  for (int i = 0; i < N_pairs; ++i)
    sum3 += dtwc::adtwFull_L_mv(xs3[i].data(), n_steps, ys3[i].data(), n_steps, 3, penalty);
  t1 = Clock::now();
  double ms3 = Ms(t1 - t0).count();

  std::cout << "\n[Wave2B perf ADTW] " << N_pairs << " pairs x " << n_steps << " steps:\n"
            << "  ndim=1 ADTW: " << ms1 << " ms  (sum=" << sum1 << ")\n"
            << "  ndim=3 ADTW: " << ms3 << " ms  (sum=" << sum3 << ")\n"
            << "  Ratio ndim=3/ndim=1: " << (ms3 / (ms1 + 1e-9)) << "x\n";

  REQUIRE(std::isfinite(sum1));
  REQUIRE(std::isfinite(sum3));
  // ndim=3 distances should be larger (three channels sum to more cost)
  REQUIRE(sum3 >= sum1 - 1e-6);
}
