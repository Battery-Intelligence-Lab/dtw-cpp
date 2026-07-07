/**
 * @file unit_test_time_series.cpp
 * @brief Unit tests for TimeSeriesView and TimeSeries.
 *
 * @date 28 Mar 2026
 * @author Volkan Kumtepeli
 */

#include <core/time_series.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>
#include <string>

using Catch::Matchers::WithinAbs;
using namespace dtwc::core;

TEST_CASE("TimeSeriesView construction and access", "[TimeSeries]")
{
  double data[] = { 1.0, 2.0, 3.0, 4.0, 5.0 };
  TimeSeriesView<double> view{ data, 5 };

  REQUIRE(view.length == 5);
  REQUIRE_FALSE(view.empty());
  REQUIRE(view[0] == 1.0);
  REQUIRE(view[4] == 5.0);
}

TEST_CASE("TimeSeriesView empty", "[TimeSeries]")
{
  TimeSeriesView<double> view{ nullptr, 0 };
  REQUIRE(view.empty());
  REQUIRE(view.length == 0);
  REQUIRE(view.begin() == view.end());
}

TEST_CASE("TimeSeriesView iteration", "[TimeSeries]")
{
  double data[] = { 10.0, 20.0, 30.0 };
  TimeSeriesView<double> view{ data, 3 };

  double sum = 0.0;
  for (auto v : view)
    sum += v;
  REQUIRE_THAT(sum, WithinAbs(60.0, 1e-12));
}

TEST_CASE("TimeSeries construction", "[TimeSeries]")
{
  TimeSeries<double> ts;
  ts.data = { 1.0, 2.0, 3.0 };
  ts.name = "test_series";

  REQUIRE(ts.size() == 3);
  REQUIRE_FALSE(ts.empty());
  REQUIRE(ts.name == "test_series");
  REQUIRE(ts[0] == 1.0);
  REQUIRE(ts[2] == 3.0);
}

TEST_CASE("TimeSeries empty", "[TimeSeries]")
{
  TimeSeries<double> ts;
  REQUIRE(ts.empty());
  REQUIRE(ts.size() == 0);
}

TEST_CASE("TimeSeries implicit conversion to view", "[TimeSeries]")
{
  TimeSeries<double> ts;
  ts.data = { 5.0, 10.0, 15.0 };

  TimeSeriesView<double> view = ts.view();
  REQUIRE(view.length == 3);
  REQUIRE(view[0] == 5.0);
  REQUIRE(view[1] == 10.0);
  REQUIRE(view[2] == 15.0);
}

TEST_CASE("TimeSeries explicit view() method", "[TimeSeries]")
{
  TimeSeries<double> ts;
  ts.data = { 100.0, 200.0 };

  auto view = ts.view();
  REQUIRE(view.length == 2);
  REQUIRE(view[0] == 100.0);
  REQUIRE(view[1] == 200.0);
}

TEST_CASE("TimeSeries mutable indexing", "[TimeSeries]")
{
  TimeSeries<double> ts;
  ts.data = { 0.0, 0.0, 0.0 };

  ts[1] = 42.0;
  REQUIRE(ts[1] == 42.0);

  // Verify the view reflects the mutation
  TimeSeriesView<double> view = ts.view();
  REQUIRE(view[1] == 42.0);
}

TEST_CASE("TimeSeries with float type", "[TimeSeries]")
{
  TimeSeries<float> ts;
  ts.data = { 1.5f, 2.5f, 3.5f };

  TimeSeriesView<float> view = ts.view();
  REQUIRE(view.length == 3);
  REQUIRE(view[0] == 1.5f);
}

// --- Regression tests for Task 0.10: TimeSeries::view() drops ndim ---
// Bug (time_series.hpp:64,66): view()/operator TimeSeriesView built
// `{ data.data(), data.size() }`, so the view's ndim defaulted to 1 and its
// `length` was the FLAT buffer size instead of the timestep count. A
// multivariate (ndim=3) series therefore round-tripped as a univariate series
// of length flat_size -> silent corruption (at()/flat_size() mis-address).
// The unfixed code fails every REQUIRE below: TimeSeries had no ndim field at
// all, view.ndim == 1 (not 3), and view.length == 6 (not 2).
TEST_CASE("TimeSeries::view() preserves ndim for multivariate series", "[TimeSeries][mv]")
{
  TimeSeries<double> ts;
  ts.ndim = 3;                                       // 2 timesteps x 3 features
  ts.data = { 1.0, 2.0, 3.0, 4.0, 5.0, 6.0 };

  auto view = ts.view();
  REQUIRE(view.ndim == 3);        // unfixed: 1 (dropped)
  REQUIRE(view.length == 2);      // timesteps; unfixed: 6 (flat size)
  REQUIRE(view.flat_size() == 6);
  // at(i) must address timestep i (ndim-strided), not scalar i.
  REQUIRE(view.at(0)[0] == 1.0);
  REQUIRE(view.at(0)[2] == 3.0);
  REQUIRE(view.at(1)[0] == 4.0);
  REQUIRE(view.at(1)[2] == 6.0);
}

// Same defect via the explicit conversion operator path.
TEST_CASE("TimeSeries explicit conversion preserves ndim", "[TimeSeries][mv]")
{
  TimeSeries<double> ts;
  ts.ndim = 2;                                       // 3 timesteps x 2 features
  ts.data = { 10.0, 11.0, 20.0, 21.0, 30.0, 31.0 };

  auto view = static_cast<TimeSeriesView<double>>(ts);
  REQUIRE(view.ndim == 2);        // unfixed: 1 (dropped)
  REQUIRE(view.length == 3);      // timesteps; unfixed: 6 (flat size)
  REQUIRE(view.at(2)[1] == 31.0);
}

// Univariate default (ndim==1) must be unchanged by the fix.
TEST_CASE("TimeSeries::view() univariate unchanged", "[TimeSeries][mv]")
{
  TimeSeries<double> ts;
  ts.data = { 7.0, 8.0, 9.0, 10.0 };

  auto view = ts.view();
  REQUIRE(view.ndim == 1);
  REQUIRE(view.length == 4);
  REQUIRE(view.flat_size() == 4);
}
