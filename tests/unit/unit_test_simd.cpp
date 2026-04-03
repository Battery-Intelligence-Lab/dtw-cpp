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

#ifdef DTWC_HAS_HIGHWAY
#include <simd/multi_pair_dtw.hpp>
#endif

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

// ============================================================================
//  Multi-pair DTW Tests (only when SIMD is available)
// ============================================================================

#ifdef DTWC_HAS_HIGHWAY

TEST_CASE("Multi-pair DTW: matches scalar for 4 random pairs",
          "[simd][multi_pair_dtw]")
{
  std::mt19937 rng(54321);
  std::uniform_int_distribution<std::size_t> len_dist(10, 100);
  std::uniform_real_distribution<double> val_dist(-5.0, 5.0);

  std::vector<std::vector<double>> xs(4), ys(4);
  for (int p = 0; p < 4; ++p) {
    std::size_t nx = len_dist(rng), ny = len_dist(rng);
    xs[p].resize(nx);
    ys[p].resize(ny);
    for (auto &v : xs[p]) v = val_dist(rng);
    for (auto &v : ys[p]) v = val_dist(rng);
  }

  const double *x_ptrs[4] = {xs[0].data(), xs[1].data(), xs[2].data(), xs[3].data()};
  const double *y_ptrs[4] = {ys[0].data(), ys[1].data(), ys[2].data(), ys[3].data()};
  std::size_t x_lens[4] = {xs[0].size(), xs[1].size(), xs[2].size(), xs[3].size()};
  std::size_t y_lens[4] = {ys[0].size(), ys[1].size(), ys[2].size(), ys[3].size()};

  auto result = dtwc::simd::dtw_multi_pair(x_ptrs, y_ptrs, x_lens, y_lens, 4);

  for (int p = 0; p < 4; ++p) {
    CAPTURE(p, x_lens[p], y_lens[p]);
    double expected = dtw_scalar(xs[p].data(), xs[p].size(),
                                 ys[p].data(), ys[p].size());
    REQUIRE(result.distances[p] == Approx(expected).epsilon(1e-10));
  }
}

TEST_CASE("Multi-pair DTW: fewer than 4 pairs (1, 2, 3)",
          "[simd][multi_pair_dtw]")
{
  // Create 4 pairs but test with n_pairs = 1, 2, 3
  std::vector<double> a = {1.0, 2.0, 3.0, 4.0, 5.0};
  std::vector<double> b = {2.0, 3.0, 4.0, 5.0, 6.0};
  std::vector<double> c = {1.0, 1.0, 1.0, 1.0, 1.0};
  std::vector<double> d = {5.0, 5.0, 5.0, 5.0, 5.0};

  const double *x_ptrs[4] = {a.data(), b.data(), c.data(), a.data()};
  const double *y_ptrs[4] = {b.data(), c.data(), d.data(), d.data()};
  std::size_t x_lens[4] = {a.size(), b.size(), c.size(), a.size()};
  std::size_t y_lens[4] = {b.size(), c.size(), d.size(), d.size()};

  for (std::size_t n_pairs : {1ul, 2ul, 3ul}) {
    CAPTURE(n_pairs);
    auto result = dtwc::simd::dtw_multi_pair(x_ptrs, y_ptrs, x_lens, y_lens, n_pairs);

    for (std::size_t p = 0; p < n_pairs; ++p) {
      CAPTURE(p);
      double expected = dtw_scalar(x_ptrs[p], x_lens[p], y_ptrs[p], y_lens[p]);
      REQUIRE(result.distances[p] == Approx(expected).epsilon(1e-10));
    }
  }
}

TEST_CASE("Multi-pair DTW: equal series give zero distance",
          "[simd][multi_pair_dtw]")
{
  auto series = random_series(50, 11111);

  const double *x_ptrs[4] = {series.data(), series.data(), series.data(), series.data()};
  const double *y_ptrs[4] = {series.data(), series.data(), series.data(), series.data()};
  std::size_t lens[4] = {series.size(), series.size(), series.size(), series.size()};

  auto result = dtwc::simd::dtw_multi_pair(x_ptrs, y_ptrs, lens, lens, 4);

  for (int p = 0; p < 4; ++p) {
    CAPTURE(p);
    REQUIRE(result.distances[p] == Approx(0.0).margin(1e-12));
  }
}

TEST_CASE("Multi-pair DTW: different length pairs in same batch",
          "[simd][multi_pair_dtw]")
{
  auto x0 = random_series(10, 100);
  auto y0 = random_series(20, 101);
  auto x1 = random_series(50, 102);
  auto y1 = random_series(30, 103);
  auto x2 = random_series(5, 104);
  auto y2 = random_series(80, 105);
  auto x3 = random_series(100, 106);
  auto y3 = random_series(7, 107);

  const double *x_ptrs[4] = {x0.data(), x1.data(), x2.data(), x3.data()};
  const double *y_ptrs[4] = {y0.data(), y1.data(), y2.data(), y3.data()};
  std::size_t x_lens[4] = {x0.size(), x1.size(), x2.size(), x3.size()};
  std::size_t y_lens[4] = {y0.size(), y1.size(), y2.size(), y3.size()};

  auto result = dtwc::simd::dtw_multi_pair(x_ptrs, y_ptrs, x_lens, y_lens, 4);

  for (int p = 0; p < 4; ++p) {
    CAPTURE(p, x_lens[p], y_lens[p]);
    double expected = dtw_scalar(x_ptrs[p], x_lens[p], y_ptrs[p], y_lens[p]);
    REQUIRE(result.distances[p] == Approx(expected).epsilon(1e-10));
  }
}

TEST_CASE("Multi-pair DTW: empty series returns max()",
          "[simd][multi_pair_dtw]")
{
  auto non_empty = random_series(10, 999);
  std::vector<double> empty_series;

  // Pair 0: both empty; Pair 1: x empty; Pair 2: y empty; Pair 3: normal
  const double *x_ptrs[4] = {nullptr, nullptr, non_empty.data(), non_empty.data()};
  const double *y_ptrs[4] = {nullptr, non_empty.data(), nullptr, non_empty.data()};
  std::size_t x_lens[4] = {0, 0, non_empty.size(), non_empty.size()};
  std::size_t y_lens[4] = {0, non_empty.size(), 0, non_empty.size()};

  auto result = dtwc::simd::dtw_multi_pair(x_ptrs, y_ptrs, x_lens, y_lens, 4);

  constexpr double max_val = std::numeric_limits<double>::max();
  REQUIRE(result.distances[0] == max_val);
  REQUIRE(result.distances[1] == max_val);
  REQUIRE(result.distances[2] == max_val);
  // Pair 3: same series -> 0
  REQUIRE(result.distances[3] == Approx(0.0).margin(1e-12));
}

TEST_CASE("Multi-pair DTW: symmetry dtw(x,y) == dtw(y,x)",
          "[simd][multi_pair_dtw]")
{
  auto a = random_series(40, 5000);
  auto b = random_series(60, 5001);
  auto c = random_series(25, 5002);
  auto d = random_series(35, 5003);

  // Forward: (a,b), (c,d)
  const double *xf[4] = {a.data(), c.data(), a.data(), c.data()};
  const double *yf[4] = {b.data(), d.data(), b.data(), d.data()};
  std::size_t xl[4] = {a.size(), c.size(), a.size(), c.size()};
  std::size_t yl[4] = {b.size(), d.size(), b.size(), d.size()};

  auto fwd = dtwc::simd::dtw_multi_pair(xf, yf, xl, yl, 4);

  // Reverse: (b,a), (d,c)
  const double *xr[4] = {b.data(), d.data(), b.data(), d.data()};
  const double *yr[4] = {a.data(), c.data(), a.data(), c.data()};
  std::size_t xrl[4] = {b.size(), d.size(), b.size(), d.size()};
  std::size_t yrl[4] = {a.size(), c.size(), a.size(), c.size()};

  auto rev = dtwc::simd::dtw_multi_pair(xr, yr, xrl, yrl, 4);

  for (int p = 0; p < 4; ++p) {
    CAPTURE(p);
    REQUIRE(fwd.distances[p] == Approx(rev.distances[p]).epsilon(1e-12));
  }
}

TEST_CASE("Multi-pair DTW: large series (n=1000) stress test",
          "[simd][multi_pair_dtw]")
{
  auto x0 = random_series(1000, 7000);
  auto y0 = random_series(1000, 7001);
  auto x1 = random_series(800, 7002);
  auto y1 = random_series(1200, 7003);

  const double *x_ptrs[4] = {x0.data(), x1.data(), x0.data(), x1.data()};
  const double *y_ptrs[4] = {y0.data(), y1.data(), y1.data(), y0.data()};
  std::size_t x_lens[4] = {x0.size(), x1.size(), x0.size(), x1.size()};
  std::size_t y_lens[4] = {y0.size(), y1.size(), y1.size(), y0.size()};

  auto result = dtwc::simd::dtw_multi_pair(x_ptrs, y_ptrs, x_lens, y_lens, 4);

  for (int p = 0; p < 4; ++p) {
    CAPTURE(p, x_lens[p], y_lens[p]);
    double expected = dtw_scalar(x_ptrs[p], x_lens[p], y_ptrs[p], y_lens[p]);
    REQUIRE(result.distances[p] == Approx(expected).epsilon(1e-10));
  }
}

TEST_CASE("Multi-pair DTW: deterministic (same input -> same output)",
          "[simd][multi_pair_dtw]")
{
  auto x = random_series(50, 8000);
  auto y = random_series(60, 8001);

  const double *x_ptrs[4] = {x.data(), x.data(), x.data(), x.data()};
  const double *y_ptrs[4] = {y.data(), y.data(), y.data(), y.data()};
  std::size_t xl[4] = {x.size(), x.size(), x.size(), x.size()};
  std::size_t yl[4] = {y.size(), y.size(), y.size(), y.size()};

  auto r1 = dtwc::simd::dtw_multi_pair(x_ptrs, y_ptrs, xl, yl, 4);
  auto r2 = dtwc::simd::dtw_multi_pair(x_ptrs, y_ptrs, xl, yl, 4);
  auto r3 = dtwc::simd::dtw_multi_pair(x_ptrs, y_ptrs, xl, yl, 4);

  for (int p = 0; p < 4; ++p) {
    CAPTURE(p);
    // All 4 lanes should give the same result (same pair repeated)
    REQUIRE(r1.distances[p] == r1.distances[0]);
    // Repeated calls should be identical
    REQUIRE(r1.distances[p] == r2.distances[p]);
    REQUIRE(r1.distances[p] == r3.distances[p]);
  }
}

TEST_CASE("Multi-pair DTW: matches production dtwFull_L",
          "[simd][multi_pair_dtw]")
{
  // Cross-check against the production scalar DTW (dtwFull_L) as well
  auto x0 = random_series(30, 9000);
  auto y0 = random_series(40, 9001);
  auto x1 = random_series(20, 9002);
  auto y1 = random_series(25, 9003);

  const double *x_ptrs[4] = {x0.data(), x1.data(), x0.data(), x1.data()};
  const double *y_ptrs[4] = {y0.data(), y1.data(), y1.data(), y0.data()};
  std::size_t x_lens[4] = {x0.size(), x1.size(), x0.size(), x1.size()};
  std::size_t y_lens[4] = {y0.size(), y1.size(), y1.size(), y0.size()};

  auto result = dtwc::simd::dtw_multi_pair(x_ptrs, y_ptrs, x_lens, y_lens, 4);

  // Compare against production dtwFull_L
  std::vector<std::vector<double>> xv = {x0, x1, x0, x1};
  std::vector<std::vector<double>> yv = {y0, y1, y1, y0};

  for (int p = 0; p < 4; ++p) {
    CAPTURE(p);
    double prod = dtwc::dtwFull_L<double>(xv[p], yv[p]);
    REQUIRE(result.distances[p] == Approx(prod).epsilon(1e-10));
  }
}

TEST_CASE("Multi-pair DTW: single-element series", "[simd][multi_pair_dtw]")
{
  double a[] = {3.0};
  double b[] = {7.0};
  double c[] = {-1.0};
  double d[] = {2.0};

  const double *x_ptrs[4] = {a, b, c, d};
  const double *y_ptrs[4] = {b, a, d, c};
  std::size_t lens[4] = {1, 1, 1, 1};

  auto result = dtwc::simd::dtw_multi_pair(x_ptrs, y_ptrs, lens, lens, 4);

  // dtw of single elements = |x - y|
  REQUIRE(result.distances[0] == Approx(4.0).epsilon(1e-12));  // |3-7|
  REQUIRE(result.distances[1] == Approx(4.0).epsilon(1e-12));  // |7-3|
  REQUIRE(result.distances[2] == Approx(3.0).epsilon(1e-12));  // |-1-2|
  REQUIRE(result.distances[3] == Approx(3.0).epsilon(1e-12));  // |2-(-1)|
}

#endif // DTWC_HAS_HIGHWAY
