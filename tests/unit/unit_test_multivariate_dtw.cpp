/**
 * @file unit_test_multivariate_dtw.cpp
 * @brief Unit tests for multivariate DTW functions.
 * @author Volkan Kumtepeli
 *
 * Tests cover:
 *   - dtwFull_L_mv: correctness, ndim=1 parity with scalar, symmetry, known values
 *   - dtwBanded_mv: correctness, ndim=1 parity with scalar, large-band matches unbanded
 *   - Edge cases: zero length, same-pointer identity, different lengths
 *   - D=1 performance parity (timing, not a hard assertion)
 */

#include <dtwc.hpp>

#include "../support/deterministic_series.hpp"
#include "../support/dtw_oracle.hpp"
#include "../support/dtw_route_bound.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>
#include <cmath>
#include <limits>
#include <random>

using Catch::Matchers::WithinAbs;

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

// =========================================================================
//  Problem multivariate DTW integration
// =========================================================================

TEST_CASE("Problem: multivariate DTW distance matrix", "[mv][problem]")
{
  dtwc::Data data;
  data.ndim = 2;
  data.p_vec = {
    {0,0, 1,1},     // series 0: [(0,0), (1,1)]
    {0,0, 1,1},     // series 1: identical to 0
    {10,10, 11,11}  // series 2: far away
  };
  data.p_names = {"a", "b", "c"};

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.set_verbose(false);
  prob.fill_distance_matrix();

  REQUIRE(prob.dist_by_ind(0, 1) == 0.0);  // identical
  REQUIRE(prob.dist_by_ind(0, 2) > 0.0);   // different
  REQUIRE(prob.dist_by_ind(0, 2) == prob.dist_by_ind(1, 2));  // symmetry
}

TEST_CASE("Problem: ndim=1 backward compat", "[mv][problem]")
{
  dtwc::Data data;
  data.p_vec = {{1,2,3}, {4,5,6}};
  data.p_names = {"a", "b"};

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.set_verbose(false);
  prob.fill_distance_matrix();

  double d = prob.dist_by_ind(0, 1);
  REQUIRE(d > 0.0);
  // Should match standard DTW
  double d_std = dtwc::dtwBanded(std::vector<double>{1,2,3}, std::vector<double>{4,5,6}, prob.band);
  REQUIRE(d == d_std);
}

// =========================================================================
//  derivative_transform_mv tests
// =========================================================================

TEST_CASE("derivative_transform_mv: ndim=1 unchanged", "[mv][ddtw]")
{
  std::vector<double> x = {1, 3, 6, 10};
  auto dx_old = dtwc::derivative_transform(x);
  auto dx_new = dtwc::derivative_transform_mv(x, 1);
  REQUIRE(dx_old.size() == dx_new.size());
  for (size_t i = 0; i < dx_old.size(); ++i)
    CHECK(std::abs(dx_new[i] - dx_old[i]) < 1e-10);
}

TEST_CASE("derivative_transform_mv: ndim=2 per-channel", "[mv][ddtw]")
{
  // 4 timesteps x 2 features: [(1,10), (3,20), (6,30), (10,40)]
  std::vector<double> x = {1,10, 3,20, 6,30, 10,40};
  auto dx = dtwc::derivative_transform_mv(x, 2);
  REQUIRE(dx.size() == 8);

  auto ch0 = dtwc::derivative_transform(std::vector<double>{1,3,6,10});
  auto ch1 = dtwc::derivative_transform(std::vector<double>{10,20,30,40});

  for (size_t t = 0; t < 4; ++t) {
    CHECK(std::abs(dx[t*2+0] - ch0[t]) < 1e-10);
    CHECK(std::abs(dx[t*2+1] - ch1[t]) < 1e-10);
  }
}

TEST_CASE("derivative_transform_mv: empty and single", "[mv][ddtw]")
{
  auto dx_empty = dtwc::derivative_transform_mv(std::vector<double>{}, 3);
  REQUIRE(dx_empty.empty());

  auto dx_single = dtwc::derivative_transform_mv(std::vector<double>{1,2,3}, 3);
  REQUIRE(dx_single.size() == 3);
  for (auto v : dx_single) CHECK(v == 0.0);
}

// =========================================================================
//  Channel counts above the three of the oracle table, against the oracle
// =========================================================================

TEST_CASE("MV DTW: ndim 2, 5, 10 and 50 equal the oracle, full and banded", "[mv][dtw][oracle]")
{
  namespace ts = dtwc::test_support;
  using dtwc::core::MetricType;
  constexpr std::pair<std::size_t, std::size_t> shapes[] = { { 6, 11 }, { 11, 6 }, { 9, 9 } };
  std::size_t checks = 0;
  for (const std::size_t ndim : { 2, 5, 10, 50 })
    for (const auto metric : { MetricType::L1, MetricType::L2, MetricType::SquaredL2 })
      for (const auto [nx, ny] : shapes) {
        const auto x = ts::benchmark_series(nx * ndim, 31), y = ts::benchmark_series(ny * ndim, 32);
        // 0 binds on 9x9 and leaves the unequal shapes no path; 5 is the longest length difference.
        for (const int band : { -1, 0, 5, 100 }) {
          ts::OracleSpec spec;
          spec.metric = metric == MetricType::L1   ? ts::OracleMetric::L1
                        : metric == MetricType::L2 ? ts::OracleMetric::L2
                                                   : ts::OracleMetric::SquaredL2;
          spec.band = band;
          spec.ndim = ndim;
          const double want = ts::dtw_oracle(spec, x, y);
          const double got = band < 0 ? dtwc::dtwFull_L_mv(x.data(), nx, y.data(), ny, ndim, -1.0, metric)
                                      : dtwc::dtwBanded_mv(x.data(), nx, y.data(), ny, ndim, band, -1.0, metric);
          INFO("ndim " << ndim << ", band " << band << ", " << nx << "x" << ny << ": got " << got
                       << ", oracle " << want);
          // The library's sentinel for no path is max(); the oracle's is infinity.
          CHECK((std::isinf(want) ? got == std::numeric_limits<double>::max()
                                  : ts::dtw_routes_agree<double>(got, want, nx, ny)));
          ++checks;
        }
      }
  CHECK(checks == 4 * 3 * 3 * 4);
}
