/**
 * @file unit_test_multivariate_dtw.cpp
 * @brief Unit tests for multivariate DTW functions and derivative_transform_mv.
 * @author Volkan Kumtepeli
 */

#include <dtwc.hpp>

#include "../support/deterministic_series.hpp"
#include "../support/dtw_oracle.hpp"
#include "../support/dtw_route_bound.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

using Catch::Matchers::WithinAbs;

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
  REQUIRE_THAT(dtwc::dtwBanded_mv(x, 2, y, 2, 2), WithinAbs(4.0, 1e-10));
}

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
  double d = dtwc::dtwBanded_mv(x, 2, y, 2, 2, -1, -1.0, dtwc::core::MetricType::SquaredL2);
  REQUIRE_THAT(d, WithinAbs(4.0, 1e-10));
}

TEST_CASE("MV DTW: empty series returns max", "[mv][dtw][edge]")
{
  double x[] = {1.0, 2.0};
  constexpr double maxVal = std::numeric_limits<double>::max();
  // Pass nullptr with 0 steps to simulate empty series
  REQUIRE(dtwc::dtwBanded_mv(x, 2, static_cast<double*>(nullptr), 0, 2) == maxVal);
  REQUIRE(dtwc::dtwBanded_mv(static_cast<double*>(nullptr), 0, x, 2, 2) == maxVal);
}

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

// Channel counts above the three of the oracle table, against the oracle.
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
          const double got = band < 0 ? dtwc::dtwBanded_mv(x.data(), nx, y.data(), ny, ndim, -1, -1.0, metric)
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
