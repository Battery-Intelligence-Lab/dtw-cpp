/**
 * @file unit_test_barycenter.cpp
 * @brief Independent oracles for DBA, SSG, soft-DTW barycenters and clustering.
 */

#include <dtwc.hpp>
#include <algorithms/barycenter.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::WithinRel;
using namespace dtwc;

namespace {

Problem make_problem(std::vector<std::vector<data_t>> values, std::size_t ndim = 1)
{
  std::vector<std::string> names;
  for (std::size_t i = 0; i < values.size(); ++i) names.push_back("s" + std::to_string(i));
  Problem problem("barycenter_test");
  Data data(std::move(values), std::move(names));
  data.ndim = ndim;
  problem.set_data(std::move(data));
  return problem;
}

double squared_dtw(const std::vector<double>& x, const std::vector<double>& y)
{
  std::vector<double> d(x.size() * y.size(), 1e300);
  auto at = [&](std::size_t i, std::size_t j) -> double& { return d[i * y.size() + j]; };
  for (std::size_t i = 0; i < x.size(); ++i) {
    for (std::size_t j = 0; j < y.size(); ++j) {
      const double local = (x[i] - y[j]) * (x[i] - y[j]);
      if (i == 0 && j == 0) at(i, j) = local;
      else at(i, j) = local + std::min({i ? at(i - 1, j) : 1e300,
                                       j ? at(i, j - 1) : 1e300,
                                       (i && j) ? at(i - 1, j - 1) : 1e300});
    }
  }
  return at(x.size() - 1, y.size() - 1);
}

} // namespace

TEST_CASE("DBA and SSG recover the scalar arithmetic mean",
          "[barycenter][dba][ssg]")
{
  auto problem = make_problem({{0.0}, {2.0}, {4.0}});
  const std::vector<int> indices{0, 1, 2};

  algorithms::BarycenterOptions options;
  options.max_iter = 200;
  options.tolerance = 1e-9;
  options.method = algorithms::BarycenterMethod::DBA;
  const auto dba = algorithms::dtw_barycenter(problem, indices, 1, options);
  REQUIRE_THAT(dba[0], WithinAbs(2.0, 1e-12));

  options.method = algorithms::BarycenterMethod::SSG;
  options.learning_rate = 0.25;
  options.learning_rate_decay = 0.002;
  options.random_seed = 17;
  const auto ssg = algorithms::dtw_barycenter(problem, indices, 1, options);
  REQUIRE_THAT(ssg[0], WithinAbs(2.0, 0.08));
}

TEST_CASE("soft-DTW barycenter descends to the scalar optimum",
          "[barycenter][soft_dtw]")
{
  auto problem = make_problem({{0.0}, {2.0}});
  algorithms::BarycenterOptions options;
  options.method = algorithms::BarycenterMethod::SoftDTW;
  options.max_iter = 100;
  options.learning_rate = 0.5;
  options.gamma = 0.5;
  options.tolerance = 1e-10;
  const auto center = algorithms::dtw_barycenter(problem, {0, 1}, 1, options);
  REQUIRE_THAT(center[0], WithinAbs(1.0, 1e-6));
}

TEST_CASE("soft-DTW production adjoint matches finite differences",
          "[barycenter][soft_dtw][gradient]")
{
  constexpr double gradient_relative_tolerance = 1e-5;
  constexpr double finite_difference_step = 1e-6;
  const std::vector<data_t> x{0.2, -0.4, 1.3, 0.7, -0.8};
  const std::vector<data_t> y{-0.1, 0.5, -0.7, 1.0, 1.6, 0.3, -1.2};

  for (const double gamma : {0.1, 1.0}) {
    const auto analytic = algorithms::detail::soft_dtw_squared_value_gradient(x, y, gamma);
    REQUIRE(analytic.gradient.size() == x.size());

    for (std::size_t i = 0; i < x.size(); ++i) {
      const double step = finite_difference_step * std::max(1.0, std::abs(x[i]));
      auto plus = x;
      auto minus = x;
      plus[i] += step;
      minus[i] -= step;
      const double finite_difference =
        (algorithms::detail::soft_dtw_squared_value_gradient(plus, y, gamma).value
         - algorithms::detail::soft_dtw_squared_value_gradient(minus, y, gamma).value)
        / (2.0 * step);

      CAPTURE(gamma, i, analytic.gradient[i], finite_difference);
      REQUIRE_THAT(analytic.gradient[i],
                   WithinRel(finite_difference, gradient_relative_tolerance));
    }
  }
}

TEST_CASE("barycenter reduces hard squared-DTW objective from its initializer",
          "[barycenter][objective]")
{
  const std::vector<std::vector<double>> values{
    {0.0, 0.0, 1.0, 2.0}, {0.0, 1.0, 2.0, 2.0}, {0.0, 0.5, 1.5, 2.0}};
  auto problem = make_problem(values);
  algorithms::BarycenterOptions options;
  options.method = algorithms::BarycenterMethod::DBA;
  const auto center = algorithms::dtw_barycenter(problem, {0, 1, 2}, 4, options);
  double before = 0.0;
  double after = 0.0;
  for (const auto& value : values) {
    before += squared_dtw(values.front(), value);
    after += squared_dtw(center, value);
  }
  REQUIRE(after <= before + 1e-12);
}

TEST_CASE("barycenter k-means separates two waveform groups",
          "[barycenter][kmeans]")
{
  std::vector<std::vector<data_t>> values;
  for (int i = 0; i < 8; ++i)
    values.push_back({0.0 + i * 0.01, 1.0, 2.0, 1.0, 0.0});
  for (int i = 0; i < 8; ++i)
    values.push_back({100.0 + i * 0.01, 101.0, 102.0, 101.0, 100.0});
  auto problem = make_problem(std::move(values));

  algorithms::BarycenterClusteringOptions options;
  options.n_clusters = 2;
  options.method = algorithms::BarycenterMethod::SSG;
  options.random_seed = 3;
  options.max_iter = 20;
  options.barycenter_max_iter = 30;
  const auto result = algorithms::barycenter_kmeans(problem, options);

  REQUIRE(result.labels.size() == 16);
  REQUIRE(result.barycenters.size() == 2);
  REQUIRE(result.barycenters[0].size() == 5);
  REQUIRE(result.barycenters[1].size() == 5);
  REQUIRE(std::all_of(result.labels.begin(), result.labels.begin() + 8,
                      [&](int label) { return label == result.labels[0]; }));
  REQUIRE(std::all_of(result.labels.begin() + 8, result.labels.end(),
                      [&](int label) { return label == result.labels[8]; }));
  REQUIRE(result.labels[0] != result.labels[8]);
  REQUIRE(result.total_cost >= 0.0);
}

TEST_CASE("barycenter rejects unsupported multivariate and invalid inputs",
          "[barycenter][errors]")
{
  auto multivariate = make_problem({{0.0, 1.0, 2.0, 3.0}, {1.0, 2.0, 3.0, 4.0}}, 2);
  REQUIRE_THROWS_AS(algorithms::dtw_barycenter(multivariate, {0, 1}, 2), InvalidInput);

  auto problem = make_problem({{0.0}, {1.0}});
  REQUIRE_THROWS_AS(algorithms::dtw_barycenter(problem, {}, 1), InvalidInput);
  REQUIRE_THROWS_AS(algorithms::dtw_barycenter(problem, {9}, 1), InvalidInput);
  REQUIRE_THROWS_AS(algorithms::dtw_barycenter(problem, {0}, 0), InvalidInput);
}
