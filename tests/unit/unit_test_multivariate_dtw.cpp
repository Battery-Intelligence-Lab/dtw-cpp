/**
 * @file unit_test_multivariate_dtw.cpp
 * @brief Unit tests for multivariate DTW functions and MV distance functors.
 *
 * Tests cover:
 *   - MVL1Dist and MVSquaredL2Dist metric functors
 *   - dtwFull_L_mv: correctness, ndim=1 parity with scalar, symmetry, known values
 *   - dtwBanded_mv: correctness, ndim=1 parity with scalar, large-band matches unbanded
 *   - Edge cases: zero length, same-pointer identity, different lengths
 *   - D=1 performance parity (timing, not a hard assertion)
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>
#include <cmath>
#include <chrono>
#include <random>
#include <iostream>

using Catch::Matchers::WithinAbs;

// =========================================================================
//  Metric functor tests
// =========================================================================

TEST_CASE("MVL1Dist: ndim=1 matches scalar L1", "[mv][metric]")
{
  double a = 3.0, b = 7.0;
  REQUIRE(dtwc::detail::MVL1Dist{}(&a, &b, 1) == std::abs(a - b));
}

TEST_CASE("MVL1Dist: ndim=3", "[mv][metric]")
{
  // |1-4| + |2-1| + |3-6| = 3 + 1 + 3 = 7
  double a[] = {1.0, 2.0, 3.0};
  double b[] = {4.0, 1.0, 6.0};
  REQUIRE(dtwc::detail::MVL1Dist{}(a, b, 3) == 7.0);
}

TEST_CASE("MVL1Dist: ndim=2 symmetric", "[mv][metric]")
{
  double a[] = {1.0, 5.0};
  double b[] = {3.0, 2.0};
  REQUIRE(dtwc::detail::MVL1Dist{}(a, b, 2) == dtwc::detail::MVL1Dist{}(b, a, 2));
}

TEST_CASE("MVSquaredL2Dist: ndim=1 matches scalar", "[mv][metric]")
{
  double a = 3.0, b = 7.0;
  // (3-7)^2 = 16
  REQUIRE(dtwc::detail::MVSquaredL2Dist{}(&a, &b, 1) == 16.0);
}

TEST_CASE("MVSquaredL2Dist: ndim=3", "[mv][metric]")
{
  // (1-4)^2 + (2-1)^2 + (3-6)^2 = 9 + 1 + 9 = 19
  double a[] = {1.0, 2.0, 3.0};
  double b[] = {4.0, 1.0, 6.0};
  REQUIRE(dtwc::detail::MVSquaredL2Dist{}(a, b, 3) == 19.0);
}

TEST_CASE("MVSquaredL2Dist: zero distance for identical points", "[mv][metric]")
{
  double a[] = {2.0, 3.0, 4.0};
  REQUIRE(dtwc::detail::MVSquaredL2Dist{}(a, a, 3) == 0.0);
}

// =========================================================================
//  MV DTW correctness — ndim=1 must match scalar DTW exactly
// =========================================================================

TEST_CASE("MV DTW: ndim=1 matches standard dtwFull_L", "[mv][dtw]")
{
  std::vector<double> x = {1, 3, 4, 2, 5};
  std::vector<double> y = {2, 4, 3, 5, 1};
  double d_std = dtwc::dtwFull_L(x, y);
  double d_mv  = dtwc::dtwFull_L_mv(x.data(), 5, y.data(), 5, 1);
  REQUIRE_THAT(d_mv, WithinAbs(d_std, 1e-10));
}

TEST_CASE("MV DTW banded: ndim=1 matches standard dtwBanded", "[mv][dtw]")
{
  std::vector<double> x = {1, 3, 4, 2, 5};
  std::vector<double> y = {2, 4, 3, 5, 1};
  double d_std = dtwc::dtwBanded(x, y, 2);
  double d_mv  = dtwc::dtwBanded_mv(x.data(), 5, y.data(), 5, 1, 2);
  REQUIRE_THAT(d_mv, WithinAbs(d_std, 1e-10));
}

// =========================================================================
//  MV DTW correctness — ndim=2
// =========================================================================

TEST_CASE("MV DTW: 2D identical series = 0", "[mv][dtw]")
{
  // 3 timesteps, 2 features each
  double x[] = {1,2, 3,4, 5,6};
  double y[] = {1,2, 3,4, 5,6};
  REQUIRE(dtwc::dtwFull_L_mv(x, 3, y, 3, 2) == 0.0);
}

TEST_CASE("MV DTW: 2D known distance (L1)", "[mv][dtw]")
{
  // x: (0,0), (1,1)
  // y: (1,1), (2,2)
  // Point distances (L1): d(x0,y0)=2, d(x0,y1)=4, d(x1,y0)=0, d(x1,y1)=2
  // DTW cost matrix:
  //   C(0,0) = 2
  //   C(1,0) = 2 + 0 = 2
  //   C(0,1) = 2 + 4 = 6
  //   C(1,1) = min(C(0,0), C(1,0), C(0,1)) + d(x1,y1) = min(2,2,6) + 2 = 4
  double x[] = {0,0, 1,1};
  double y[] = {1,1, 2,2};
  REQUIRE_THAT(dtwc::dtwFull_L_mv(x, 2, y, 2, 2), WithinAbs(4.0, 1e-10));
}

TEST_CASE("MV DTW: 3D series known distance", "[mv][dtw]")
{
  // x: (1,0,0), (0,1,0)
  // y: (0,0,1), (1,0,0)
  // L1 distances:
  //   d(x0,y0) = |1-0|+|0-0|+|0-1| = 2
  //   d(x0,y1) = |1-1|+|0-0|+|0-0| = 0
  //   d(x1,y0) = |0-0|+|1-0|+|0-1| = 2
  //   d(x1,y1) = |0-1|+|1-0|+|0-0| = 2
  // DTW cost matrix:
  //   C(0,0) = 2
  //   C(1,0) = 2 + 2 = 4
  //   C(0,1) = 2 + 0 = 2
  //   C(1,1) = min(C(0,0), C(1,0), C(0,1)) + 2 = min(2,4,2) + 2 = 4
  double x[] = {1,0,0, 0,1,0};
  double y[] = {0,0,1, 1,0,0};
  REQUIRE_THAT(dtwc::dtwFull_L_mv(x, 2, y, 2, 3), WithinAbs(4.0, 1e-10));
}

TEST_CASE("MV DTW: symmetry (dtwFull_L_mv)", "[mv][dtw]")
{
  double x[] = {1,2,3, 4,5,6, 7,8,9};
  double y[] = {9,8,7, 6,5,4, 3,2,1};
  double d1 = dtwc::dtwFull_L_mv(x, 3, y, 3, 3);
  double d2 = dtwc::dtwFull_L_mv(y, 3, x, 3, 3);
  REQUIRE_THAT(d1, WithinAbs(d2, 1e-10));
}

// =========================================================================
//  MV DTW: banded variants
// =========================================================================

TEST_CASE("MV DTW banded: large band matches unbanded (ndim=2)", "[mv][dtw]")
{
  // band=100 >> series length => banded should equal full
  double x[] = {1,2, 3,4, 5,6, 7,8};
  double y[] = {2,1, 4,3, 6,5, 8,7};
  double d_full = dtwc::dtwFull_L_mv(x, 4, y, 4, 2);
  double d_band = dtwc::dtwBanded_mv(x, 4, y, 4, 2, 100);
  REQUIRE_THAT(d_full, WithinAbs(d_band, 1e-10));
}

TEST_CASE("MV DTW banded: ndim=2 symmetry", "[mv][dtw]")
{
  double x[] = {0,1, 2,3, 4,5};
  double y[] = {5,4, 3,2, 1,0};
  double d1 = dtwc::dtwBanded_mv(x, 3, y, 3, 2, 2);
  double d2 = dtwc::dtwBanded_mv(y, 3, x, 3, 2, 2);
  REQUIRE_THAT(d1, WithinAbs(d2, 1e-10));
}

// =========================================================================
//  MV DTW: SquaredL2 metric
// =========================================================================

TEST_CASE("MV DTW SquaredL2: ndim=2 known value", "[mv][dtw]")
{
  // x: (0,0), (1,1)
  // y: (1,1), (2,2)
  // Squared L2 distances:
  //   d(x0,y0) = 1+1 = 2
  //   d(x0,y1) = 4+4 = 8
  //   d(x1,y0) = 0+0 = 0
  //   d(x1,y1) = 1+1 = 2
  // C(0,0)=2, C(1,0)=2, C(0,1)=10, C(1,1)=min(2,2,10)+2=4
  double x[] = {0,0, 1,1};
  double y[] = {1,1, 2,2};
  double d = dtwc::dtwFull_L_mv(x, 2, y, 2, 2, -1.0, dtwc::core::MetricType::SquaredL2);
  REQUIRE_THAT(d, WithinAbs(4.0, 1e-10));
}

TEST_CASE("MV DTW SquaredL2: ndim=1 matches standard banded", "[mv][dtw]")
{
  std::vector<double> x = {1, 3, 4, 2, 5, 6};
  std::vector<double> y = {2, 4, 3, 5, 1, 7};
  double d_std = dtwc::dtwBanded(x, y, 3, -1.0, dtwc::core::MetricType::SquaredL2);
  double d_mv  = dtwc::dtwBanded_mv(x.data(), 6, y.data(), 6, 1, 3, -1.0,
                                     dtwc::core::MetricType::SquaredL2);
  REQUIRE_THAT(d_mv, WithinAbs(d_std, 1e-10));
}

// =========================================================================
//  MV DTW: edge cases
// =========================================================================

TEST_CASE("MV DTW: empty series returns max", "[mv][dtw][edge]")
{
  double x[] = {1.0, 2.0};
  constexpr double maxVal = std::numeric_limits<double>::max();
  // Pass nullptr with 0 steps to simulate empty series
  REQUIRE(dtwc::dtwFull_L_mv(x, 2, static_cast<double*>(nullptr), 0, 2) == maxVal);
  REQUIRE(dtwc::dtwFull_L_mv(static_cast<double*>(nullptr), 0, x, 2, 2) == maxVal);
}

TEST_CASE("MV DTW: different lengths (ndim=2)", "[mv][dtw][edge]")
{
  double x[] = {0,0, 1,1, 2,2};  // 3 timesteps
  double y[] = {0,0, 2,2};        // 2 timesteps
  double d = dtwc::dtwFull_L_mv(x, 3, y, 2, 2);
  REQUIRE(d >= 0.0);
  REQUIRE(d < std::numeric_limits<double>::max());
}

TEST_CASE("MV DTW: same pointer same length = 0", "[mv][dtw][edge]")
{
  double x[] = {1,2, 3,4, 5,6};
  REQUIRE(dtwc::dtwFull_L_mv(x, 3, x, 3, 2) == 0.0);
}

TEST_CASE("MV DTW banded: negative band falls back to full", "[mv][dtw][edge]")
{
  double x[] = {1,2, 3,4, 5,6};
  double y[] = {2,1, 4,3, 6,5};
  double d_full = dtwc::dtwFull_L_mv(x, 3, y, 3, 2);
  double d_band = dtwc::dtwBanded_mv(x, 3, y, 3, 2, -1);
  REQUIRE_THAT(d_full, WithinAbs(d_band, 1e-10));
}

// =========================================================================
//  D=1 performance parity (informational, not a hard timing assertion)
// =========================================================================

TEST_CASE("MV DTW: D=1 performance parity", "[mv][dtw][perf]")
{
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist_rng(0.0, 100.0);
  const size_t N = 200;
  std::vector<double> x(N), y(N);
  for (auto &v : x) v = dist_rng(rng);
  for (auto &v : y) v = dist_rng(rng);

  // Warmup
  for (int i = 0; i < 100; ++i) {
    dtwc::dtwFull_L(x.data(), N, y.data(), N);
    dtwc::dtwFull_L_mv(x.data(), N, y.data(), N, 1);
  }

  const int ITERS = 1000;

  auto t0 = std::chrono::high_resolution_clock::now();
  double sum_std = 0;
  for (int i = 0; i < ITERS; ++i)
    sum_std += dtwc::dtwFull_L(x.data(), N, y.data(), N);
  auto t1 = std::chrono::high_resolution_clock::now();
  double sum_mv = 0;
  for (int i = 0; i < ITERS; ++i)
    sum_mv += dtwc::dtwFull_L_mv(x.data(), N, y.data(), N, 1);
  auto t2 = std::chrono::high_resolution_clock::now();

  // Results must match (ndim=1 dispatches to existing scalar path)
  REQUIRE_THAT(sum_mv, WithinAbs(sum_std, 1e-6));

  auto ms_std = std::chrono::duration<double, std::milli>(t1 - t0).count();
  auto ms_mv  = std::chrono::duration<double, std::milli>(t2 - t1).count();
  std::cout << "[perf] Standard dtwFull_L: " << ms_std << " ms, "
            << "MV dtwFull_L_mv(D=1): " << ms_mv << " ms  "
            << "(ratio: " << ms_mv / ms_std << "x)\n";
  // MV(D=1) dispatches to existing scalar path, so timing should be ~1x.
  // We don't assert on timing — just print for observation.
}
