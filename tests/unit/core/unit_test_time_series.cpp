/**
 * @file unit_test_time_series.cpp
 * @brief Unit tests for TimeSeriesView and TimeSeries.
 *
 * @date 28 Mar 2026
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

  TimeSeriesView<double> view = ts;
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
  TimeSeriesView<double> view = ts;
  REQUIRE(view[1] == 42.0);
}

TEST_CASE("TimeSeries with float type", "[TimeSeries]")
{
  TimeSeries<float> ts;
  ts.data = { 1.5f, 2.5f, 3.5f };

  TimeSeriesView<float> view = ts;
  REQUIRE(view.length == 3);
  REQUIRE(view[0] == 1.5f);
}
