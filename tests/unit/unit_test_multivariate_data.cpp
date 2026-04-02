/**
 * @file unit_test_multivariate_data.cpp
 * @brief Unit tests for multivariate support: Data::ndim, series_length,
 *        validate_ndim, and TimeSeriesView ndim/at()/flat_size().
 *
 * @date 02 Apr 2026
 */

#include <dtwc.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("Data: ndim defaults to 1", "[data][mv]")
{
  dtwc::Data data;
  REQUIRE(data.ndim == 1);
}

TEST_CASE("Data: series_length with ndim=1", "[data][mv]")
{
  dtwc::Data data;
  data.p_vec = { { 1.0, 2.0, 3.0 } };
  data.p_names = { "a" };
  REQUIRE(data.series_length(0) == 3);
}

TEST_CASE("Data: series_length with ndim=3", "[data][mv]")
{
  dtwc::Data data;
  data.ndim = 3;
  data.p_vec = { { 1, 2, 3, 4, 5, 6 } };
  data.p_names = { "a" };
  REQUIRE(data.series_length(0) == 2);
}

TEST_CASE("Data: validate_ndim catches bad size", "[data][mv]")
{
  dtwc::Data data;
  data.ndim = 3;
  data.p_vec = { { 1, 2, 3, 4, 5 } };
  data.p_names = { "a" };
  REQUIRE_THROWS_AS(data.validate_ndim(), std::runtime_error);
}

TEST_CASE("Data: validate_ndim passes for valid", "[data][mv]")
{
  dtwc::Data data;
  data.ndim = 2;
  data.p_vec = { { 1, 2, 3, 4 }, { 5, 6, 7, 8 } };
  data.p_names = { "a", "b" };
  REQUIRE_NOTHROW(data.validate_ndim());
}

TEST_CASE("Data: constructor validates ndim", "[data][mv]")
{
  using data_t = dtwc::data_t;
  std::vector<std::vector<data_t>> bad_vec = { { 1, 2, 3, 4, 5 } };
  std::vector<std::string> names = { "a" };
  // ndim=3 but flat size 5 is not divisible
  REQUIRE_THROWS_AS(dtwc::Data(std::move(bad_vec), std::move(names), 3), std::runtime_error);
}

TEST_CASE("Data: constructor with ndim=2 succeeds", "[data][mv]")
{
  using data_t = dtwc::data_t;
  std::vector<std::vector<data_t>> vecs = { { 1, 2, 3, 4 } };
  std::vector<std::string> names = { "a" };
  dtwc::Data data(std::move(vecs), std::move(names), 2);
  REQUIRE(data.ndim == 2);
  REQUIRE(data.series_length(0) == 2);
}

TEST_CASE("TimeSeriesView: ndim and at()", "[data][mv]")
{
  double arr[] = { 1, 2, 3, 4, 5, 6 };
  dtwc::core::TimeSeriesView<double> view{ arr, 2, 3 };
  REQUIRE(view.length == 2);
  REQUIRE(view.ndim == 3);
  REQUIRE(view.at(0)[0] == 1.0);
  REQUIRE(view.at(0)[2] == 3.0);
  REQUIRE(view.at(1)[0] == 4.0);
  REQUIRE(view.at(1)[2] == 6.0);
  REQUIRE(view.flat_size() == 6);
}

TEST_CASE("TimeSeriesView: ndim=1 backward compat", "[data][mv]")
{
  double arr[] = { 1, 2, 3 };
  dtwc::core::TimeSeriesView<double> view{ arr, 3 };
  REQUIRE(view.ndim == 1);
  REQUIRE(view[0] == 1.0);
  REQUIRE(view[2] == 3.0);
  REQUIRE(view.flat_size() == 3);
}

TEST_CASE("TimeSeriesView: end() spans full flat buffer", "[data][mv]")
{
  double arr[] = { 1, 2, 3, 4 };
  dtwc::core::TimeSeriesView<double> view{ arr, 2, 2 };
  REQUIRE(view.end() == arr + 4);
}

TEST_CASE("TimeSeriesView: operator== compares ndim", "[data][mv]")
{
  double a[] = { 1, 2, 3, 4 };
  dtwc::core::TimeSeriesView<double> v1{ a, 2, 2 };
  dtwc::core::TimeSeriesView<double> v2{ a, 4, 1 };
  REQUIRE(v1 != v2); // same data pointer, different ndim/length interpretation
}
