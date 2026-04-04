/**
 * @file unit_test_simd.cpp
 * @brief Comprehensive tests for SIMD implementations of LB_Keogh, z_normalize,
 *        and multi-pair DTW.
 *
 * @details Each SIMD function is tested against a scalar reference implementation
 *          written directly in this file, ensuring independence from the production
 *          dispatch path. Tests cover various lengths (including non-power-of-2 for
 *          tail handling), edge cases, and mathematical properties.
 *
 * @author Volkan Kumtepeli
 * @date 29 Mar 2026
 */

#include <core/lower_bound_impl.hpp>
#include <core/z_normalize.hpp>
#include <warping.hpp>
#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <vector>
#include <random>
#include <cmath>
#include <limits>
#include <numeric>
#include <algorithm>

using Catch::Approx;

// ============================================================================
//  Scalar reference implementations (independent of production code)
// ============================================================================

/// Scalar LB_Keogh: sum of excess distances outside the envelope.
static double lb_keogh_scalar(const double *q, std::size_t n,
                              const double *u, const double *l)
{
  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double eu = q[i] - u[i];
    double el = l[i] - q[i];
    double e = std::max(0.0, std::max(eu, el));
    sum += e;
  }
  return sum;
}

/// Scalar z-normalize: 3-pass (mean, stddev, normalize).
/// Matches the production scalar path semantics: population stddev, threshold 1e-10.
static void z_normalize_scalar(double *series, std::size_t n)
{
  if (n == 0) return;
  if (n == 1) { series[0] = 0.0; return; }

  double sum = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    sum += series[i];
  double mean = sum / static_cast<double>(n);

  double sq_sum = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    double d = series[i] - mean;
    sq_sum += d * d;
  }
  double stddev = std::sqrt(sq_sum / static_cast<double>(n));

  if (stddev > 1e-10) {
    double inv_stddev = 1.0 / stddev;
    for (std::size_t i = 0; i < n; ++i)
      series[i] = (series[i] - mean) * inv_stddev;
  } else {
    for (std::size_t i = 0; i < n; ++i)
      series[i] = 0.0;
  }
}

/// Scalar full DTW with L1 metric (independent of production dtwFull_L).
static double dtw_scalar(const double *x, std::size_t nx,
                         const double *y, std::size_t ny)
{
  constexpr double INF = std::numeric_limits<double>::max();
  if (nx == 0 || ny == 0) return INF;

  // Full matrix DTW (simple, correct reference)
  std::vector<double> prev(ny, INF), curr(ny, INF);

  prev[0] = std::abs(x[0] - y[0]);
  for (std::size_t j = 1; j < ny; ++j)
    prev[j] = prev[j - 1] + std::abs(x[0] - y[j]);

  for (std::size_t i = 1; i < nx; ++i) {
    curr[0] = prev[0] + std::abs(x[i] - y[0]);
    for (std::size_t j = 1; j < ny; ++j) {
      double cost = std::abs(x[i] - y[j]);
      curr[j] = cost + std::min({prev[j - 1], prev[j], curr[j - 1]});
    }
    std::swap(prev, curr);
  }
  return prev[ny - 1];
}

// ============================================================================
//  Helpers
// ============================================================================

/// Generate a random vector of given length using a deterministic seed.
static std::vector<double> random_series(std::size_t n, unsigned seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<double> dist(-10.0, 10.0);
  std::vector<double> v(n);
  for (auto &x : v)
    x = dist(rng);
  return v;
}

/// Compute envelopes using a simple scalar loop (independent reference).
static void compute_envelopes_scalar(const double *series, std::size_t n,
                                     int band, double *upper, double *lower)
{
  std::size_t w = static_cast<std::size_t>(std::max(band, 0));
  for (std::size_t p = 0; p < n; ++p) {
    std::size_t lo = (p >= w) ? p - w : 0;
    std::size_t hi = std::min(p + w + 1, n);
    double mx = series[lo], mn = series[lo];
    for (std::size_t j = lo + 1; j < hi; ++j) {
      if (series[j] > mx) mx = series[j];
      if (series[j] < mn) mn = series[j];
    }
    upper[p] = mx;
    lower[p] = mn;
  }
}

/// Lengths that exercise SIMD tail handling at various widths (1, 2, 4, 8 lanes).
static const std::vector<std::size_t> test_lengths = {
    1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33,
    63, 64, 65, 100, 128, 255, 256, 500, 1000, 4000};

// ============================================================================
//  LB_Keogh Tests
// ============================================================================

TEST_CASE("LB_Keogh: various lengths match scalar reference", "[simd][lb_keogh]")
{
  for (std::size_t n : test_lengths) {
    CAPTURE(n);
    auto query = random_series(n, 42 + static_cast<unsigned>(n));
    auto candidate = random_series(n, 1337 + static_cast<unsigned>(n));

    std::vector<double> upper(n), lower(n);
    compute_envelopes_scalar(candidate.data(), n, 3, upper.data(), lower.data());

    double expected = lb_keogh_scalar(query.data(), n, upper.data(), lower.data());
    double actual = dtwc::core::lb_keogh(query.data(), n, upper.data(), lower.data());

    REQUIRE(actual == Approx(expected).epsilon(1e-12));
  }
}

TEST_CASE("LB_Keogh: identity (series inside its own envelope) gives zero",
          "[simd][lb_keogh]")
{
  for (std::size_t n : {1ul, 7ul, 16ul, 64ul, 500ul, 4000ul}) {
    CAPTURE(n);
    auto series = random_series(n, 99 + static_cast<unsigned>(n));

    std::vector<double> upper(n), lower(n);
    compute_envelopes_scalar(series.data(), n, 5, upper.data(), lower.data());

    double result = dtwc::core::lb_keogh(series.data(), n, upper.data(), lower.data());
    REQUIRE(result == Approx(0.0).margin(1e-12));
  }
}

TEST_CASE("LB_Keogh: non-negativity for random pairs", "[simd][lb_keogh]")
{
  std::mt19937 seed_rng(12345);
  for (int trial = 0; trial < 50; ++trial) {
    std::size_t n = 10 + (seed_rng() % 500);
    CAPTURE(trial, n);

    auto query = random_series(n, seed_rng());
    auto candidate = random_series(n, seed_rng());

    std::vector<double> upper(n), lower(n);
    compute_envelopes_scalar(candidate.data(), n, 4, upper.data(), lower.data());

    double result = dtwc::core::lb_keogh(query.data(), n, upper.data(), lower.data());
    REQUIRE(result >= 0.0);
  }
}

TEST_CASE("LB_Keogh: known hand-computed value", "[simd][lb_keogh]")
{
  // Envelope: upper = {5, 5, 5}, lower = {1, 1, 1}
  // Query: {0, 3, 7}
  // Contributions: max(0, max(0-5, 1-0)) = max(0, 1) = 1
  //                max(0, max(3-5, 1-3)) = max(0, 0) = 0  (inside envelope)
  //                max(0, max(7-5, 1-7)) = max(0, 2) = 2
  // Total = 3.0
  double query[] = {0.0, 3.0, 7.0};
  double upper[] = {5.0, 5.0, 5.0};
  double lower[] = {1.0, 1.0, 1.0};

  double result = dtwc::core::lb_keogh(query, std::size_t(3), upper, lower);
  REQUIRE(result == Approx(3.0).epsilon(1e-15));
}

TEST_CASE("LB_Keogh: query entirely above envelope", "[simd][lb_keogh]")
{
  // Query all 100, envelope [0, 10] => each contribution = 90, total = n * 90
  std::size_t n = 37; // non-power-of-2
  std::vector<double> query(n, 100.0);
  std::vector<double> upper(n, 10.0);
  std::vector<double> lower(n, 0.0);

  double expected = static_cast<double>(n) * 90.0;
  double result = dtwc::core::lb_keogh(query.data(), n, upper.data(), lower.data());
  REQUIRE(result == Approx(expected).epsilon(1e-12));
}

TEST_CASE("LB_Keogh: query entirely below envelope", "[simd][lb_keogh]")
{
  std::size_t n = 65;
  std::vector<double> query(n, -5.0);
  std::vector<double> upper(n, 10.0);
  std::vector<double> lower(n, 0.0);

  // Each contribution: max(0, max(-5-10, 0-(-5))) = max(0, 5) = 5
  double expected = static_cast<double>(n) * 5.0;
  double result = dtwc::core::lb_keogh(query.data(), n, upper.data(), lower.data());
  REQUIRE(result == Approx(expected).epsilon(1e-12));
}

TEST_CASE("LB_Keogh: single element", "[simd][lb_keogh]")
{
  double query[] = {3.0};
  double upper[] = {1.0};
  double lower[] = {0.0};

  // 3 > 1 => contribution = 2
  double result = dtwc::core::lb_keogh(query, std::size_t(1), upper, lower);
  REQUIRE(result == Approx(2.0).epsilon(1e-15));
}

// ============================================================================
//  z_normalize Tests
// ============================================================================

TEST_CASE("z_normalize: various lengths match scalar reference", "[simd][z_normalize]")
{
  for (std::size_t n : test_lengths) {
    CAPTURE(n);
    auto data_prod = random_series(n, 77 + static_cast<unsigned>(n));
    auto data_ref = data_prod; // copy for scalar reference

    dtwc::core::z_normalize(data_prod.data(), n);
    z_normalize_scalar(data_ref.data(), n);

    for (std::size_t i = 0; i < n; ++i) {
      CAPTURE(i);
      REQUIRE(data_prod[i] == Approx(data_ref[i]).epsilon(1e-10));
    }
  }
}

TEST_CASE("z_normalize: statistical properties (mean=0, stddev=1)", "[simd][z_normalize]")
{
  for (std::size_t n : {5ul, 16ul, 64ul, 255ul, 1000ul, 4000ul}) {
    CAPTURE(n);
    auto data = random_series(n, 200 + static_cast<unsigned>(n));
    dtwc::core::z_normalize(data.data(), n);

    // Compute mean
    double mean = 0.0;
    for (std::size_t i = 0; i < n; ++i)
      mean += data[i];
    mean /= static_cast<double>(n);

    // Compute population stddev
    double var = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
      double d = data[i] - mean;
      var += d * d;
    }
    double stddev = std::sqrt(var / static_cast<double>(n));

    REQUIRE(mean == Approx(0.0).margin(1e-10));
    REQUIRE(stddev == Approx(1.0).epsilon(1e-10));
  }
}

TEST_CASE("z_normalize: constant series becomes all zeros", "[simd][z_normalize]")
{
  for (std::size_t n : {1ul, 2ul, 8ul, 33ul, 100ul}) {
    CAPTURE(n);
    std::vector<double> data(n, 42.0);
    dtwc::core::z_normalize(data.data(), n);

    for (std::size_t i = 0; i < n; ++i) {
      CAPTURE(i);
      REQUIRE(data[i] == Approx(0.0).margin(1e-15));
    }
  }
}

TEST_CASE("z_normalize: single element becomes zero", "[simd][z_normalize]")
{
  double data[] = {999.0};
  dtwc::core::z_normalize(data, std::size_t(1));
  REQUIRE(data[0] == Approx(0.0).margin(1e-15));
}

TEST_CASE("z_normalize: two elements exact result", "[simd][z_normalize]")
{
  // Series: {a, b} with a != b
  // mean = (a+b)/2, stddev = |b-a|/2
  // normalized: ((a - mean) / stddev, (b - mean) / stddev) = (-1, 1)
  double data[] = {3.0, 7.0};
  dtwc::core::z_normalize(data, std::size_t(2));
  REQUIRE(data[0] == Approx(-1.0).epsilon(1e-12));
  REQUIRE(data[1] == Approx(1.0).epsilon(1e-12));

  // Reversed: {7, 3} -> {1, -1}
  double data2[] = {7.0, 3.0};
  dtwc::core::z_normalize(data2, std::size_t(2));
  REQUIRE(data2[0] == Approx(1.0).epsilon(1e-12));
  REQUIRE(data2[1] == Approx(-1.0).epsilon(1e-12));
}

TEST_CASE("z_normalize: empty series does not crash", "[simd][z_normalize]")
{
  double *ptr = nullptr;
  // Should be a no-op
  dtwc::core::z_normalize(ptr, std::size_t(0));
  SUCCEED("Empty series handled without crash");
}

TEST_CASE("z_normalize: near-zero variance treated as constant", "[simd][z_normalize]")
{
  // All values essentially the same (within 1e-15 of each other)
  std::size_t n = 64;
  std::vector<double> data(n);
  for (std::size_t i = 0; i < n; ++i)
    data[i] = 1.0 + static_cast<double>(i) * 1e-16;

  dtwc::core::z_normalize(data.data(), n);

  for (std::size_t i = 0; i < n; ++i) {
    CAPTURE(i);
    REQUIRE(data[i] == Approx(0.0).margin(1e-10));
  }
}

TEST_CASE("z_normalize: large values maintain precision", "[simd][z_normalize]")
{
  std::size_t n = 100;
  std::vector<double> data(n);
  // Large offset + small variation
  for (std::size_t i = 0; i < n; ++i)
    data[i] = 1e12 + static_cast<double>(i);

  auto ref = data;
  dtwc::core::z_normalize(data.data(), n);
  z_normalize_scalar(ref.data(), n);

  for (std::size_t i = 0; i < n; ++i) {
    CAPTURE(i);
    REQUIRE(data[i] == Approx(ref[i]).epsilon(1e-6)); // Relaxed for large offset
  }
}

TEST_CASE("z_normalize: negative values", "[simd][z_normalize]")
{
  std::vector<double> data = {-10.0, -5.0, 0.0, 5.0, 10.0};
  auto ref = data;
  dtwc::core::z_normalize(data.data(), data.size());
  z_normalize_scalar(ref.data(), ref.size());

  for (std::size_t i = 0; i < data.size(); ++i) {
    CAPTURE(i);
    REQUIRE(data[i] == Approx(ref[i]).epsilon(1e-12));
  }
}

// Multi-pair SIMD DTW tests removed (Highway dependency removed)
