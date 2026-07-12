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
#include <cstdint>
#include <limits>
#include <numeric>
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
  // SSG applies the full squared-DTW gradient, including its factor of two.
  options.learning_rate = 0.125;
  options.learning_rate_decay = 0.002;
  options.random_seed = 17;
  // A short stochastic run retains a material last-epoch order bias.  Drive
  // this scalar Robbins-Monro case far enough that it tests recovery of the
  // mean, while the mixed-length fingerprint below separately pins RNG order.
  options.max_iter = 1600;
  options.tolerance = 0.0;
  const auto ssg = algorithms::dtw_barycenter(problem, indices, 1, options);
  REQUIRE_THAT(ssg[0], WithinAbs(2.0, 0.01));
}

TEST_CASE("SSG retains warping-path multiplicity in its stochastic gradient",
          "[barycenter][ssg][gradient]")
{
  // Resampling the only input to length two initializes the centre at {0, 10}.
  // Its unique optimal path is (0,0),(0,1),(0,2),(1,3), so coordinate zero
  // has valence 3 and aligned sum 6.  For squared DTW the component gradient
  // is 2 * (V*z - W*x), hence one eta=0.1 step gives 0 + 2*0.1*6 = 1.2.
  // Replacing the aligned sum by its mean (the old implementation) gives 0.2
  // and therefore cannot satisfy this oracle.
  auto problem = make_problem({{0.0, 2.0, 4.0, 10.0}});
  algorithms::BarycenterOptions options;
  options.method = algorithms::BarycenterMethod::SSG;
  options.max_iter = 1;
  options.learning_rate = 0.1;
  options.learning_rate_decay = 0.0;
  options.tolerance = 0.0;

  const auto center = algorithms::dtw_barycenter(problem, {0}, 2, options);

  REQUIRE(center.size() == 2);
  REQUIRE_THAT(center[0], WithinAbs(1.2, 1e-12));
  REQUIRE_THAT(center[1], WithinAbs(10.0, 1e-12));

  // With eta=0.2, the fixed-path Lipschitz cap is active at
  // 1/(2*max_valence)=1/6.  It is one scalar for the entire gradient, so the
  // multiplicity remains present and coordinate zero moves exactly to 2.0.
  options.learning_rate = 0.2;
  const auto capped_center = algorithms::dtw_barycenter(problem, {0}, 2, options);
  REQUIRE_THAT(capped_center[0], WithinAbs(2.0, 1e-12));
  REQUIRE_THAT(capped_center[1], WithinAbs(10.0, 1e-12));
}

TEST_CASE("default SSG learning rate stays finite on unequal-length series",
          "[barycenter][ssg][stability]")
{
  const std::vector<std::vector<double>> values{
    {0.0, 10.0},
    {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
    {-1.0, 1.0, 3.0, 5.0, 7.0, 9.0, 11.0}};
  auto problem = make_problem(values);
  algorithms::BarycenterOptions options;
  options.method = algorithms::BarycenterMethod::SSG;
  double before = 0.0;
  for (const auto& value : values) before += squared_dtw(values.front(), value);

  for (const std::uint64_t seed : {0ULL, 1ULL, 11ULL, 42ULL, 999ULL}) {
    options.random_seed = seed;
    const auto center = algorithms::dtw_barycenter(problem, {0, 1, 2}, 2, options);
    double after = 0.0;
    for (const auto& value : values) after += squared_dtw(center, value);

    CAPTURE(seed, center, before, after);
    REQUIRE(std::all_of(center.begin(), center.end(),
                        [](double value) { return std::isfinite(value); }));
    REQUIRE(after < before);
  }
}

TEST_CASE("default SSG learning rate handles extreme path valence",
          "[barycenter][ssg][stability][valence]")
{
  for (const std::size_t length : {100U, 1000U}) {
    std::vector<data_t> ramp(length);
    std::iota(ramp.begin(), ramp.end(), 0.0);
    auto problem = make_problem({ramp});
    algorithms::BarycenterOptions options;
    options.method = algorithms::BarycenterMethod::SSG;

    const auto center = algorithms::dtw_barycenter(problem, {0}, 2, options);
    const std::vector<data_t> initial{ramp.front(), ramp.back()};
    const double before = squared_dtw(initial, ramp);
    const double after = squared_dtw(center, ramp);

    CAPTURE(length, center, before, after);
    REQUIRE(std::all_of(center.begin(), center.end(),
                        [](double value) { return std::isfinite(value); }));
    REQUIRE(std::isfinite(after));
    REQUIRE(after < before);
  }
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

TEST_CASE("soft-DTW recurrences agree exactly when their local costs coincide",
          "[barycenter][soft_dtw][cross_implementation]")
{
  // On binary values, |x-y| == (x-y)^2 for every pair, so the public L1
  // soft-DTW and the barycenter's squared-cost recurrence receive the same
  // complete local-cost matrix. This compares the independent forward passes
  // on unequal lengths rather than duplicating either recurrence in the test.
  const std::vector<data_t> binary_x{0.0, 1.0, 1.0, 0.0, 1.0};
  const std::vector<data_t> binary_y{1.0, 0.0, 1.0};
  for (const double gamma : {0.1, 1.0, 2.5}) {
    const double l1_value = soft_dtw<double>(binary_x, binary_y, gamma);
    const double squared_value =
      algorithms::detail::soft_dtw_squared_value_gradient(
        binary_x, binary_y, gamma).value;
    CAPTURE(gamma, l1_value, squared_value);
    REQUIRE_THAT(squared_value, WithinAbs(l1_value, 1e-12));
  }

  // Also pin the intentional public semantic difference. If either engine is
  // silently changed to the other's local cost, this sensitivity check fails.
  const std::vector<data_t> nonbinary_x{0.0, 2.0};
  const std::vector<data_t> nonbinary_y{0.0, 0.0};
  const double l1_value = soft_dtw<double>(nonbinary_x, nonbinary_y, 1.0);
  const double squared_value =
    algorithms::detail::soft_dtw_squared_value_gradient(
      nonbinary_x, nonbinary_y, 1.0).value;
  REQUIRE(std::abs(squared_value - l1_value) > 1.0);
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

TEST_CASE("barycenter k-means updates its center before testing convergence",
          "[barycenter][kmeans][convergence][m24]")
{
  auto problem = make_problem({{0.0}, {2.0}});
  const algorithms::BarycenterClusteringOptions defaults;
  REQUIRE(defaults.tolerance > 0.0);

  for (const double tolerance : {defaults.tolerance, 0.0}) {
    CAPTURE(tolerance);
    algorithms::BarycenterClusteringOptions options;
    options.n_clusters = 1;
    options.method = algorithms::BarycenterMethod::DBA;
    options.tolerance = tolerance;

    const auto result = algorithms::barycenter_kmeans(problem, options);

    CHECK(result.converged);
    CHECK(result.iterations == 1);
    REQUIRE(result.labels.size() == 2);
    CHECK(result.labels == std::vector<int>{0, 0});
    REQUIRE(result.barycenters.size() == 1);
    REQUIRE(result.barycenters[0].size() == 1);
    CHECK_THAT(result.barycenters[0][0], WithinAbs(1.0, 1e-12));
    CHECK_THAT(result.total_cost, WithinAbs(2.0, 1e-12));
  }
}

TEST_CASE("barycenters reject non-finite input and finite-input overflow loudly",
          "[barycenter][errors][nonfinite][m24]")
{
  const char* const input_error =
    "dtw_barycenter: all values must be finite.";
  for (const double bad_value : {
         std::numeric_limits<double>::infinity(),
         std::numeric_limits<double>::quiet_NaN()}) {
    auto problem = make_problem({{0.0}, {bad_value}});
    algorithms::BarycenterClusteringOptions clustering_options;
    clustering_options.n_clusters = 1;

    CHECK_THROWS_WITH(
      algorithms::dtw_barycenter(problem, {0, 1}, 1), input_error);
    CHECK_THROWS_WITH(
      algorithms::barycenter_kmeans(problem, clustering_options), input_error);
  }

  const double largest = std::numeric_limits<double>::max();
  auto overflow_problem = make_problem({{largest}, {-largest}});
  algorithms::BarycenterOptions barycenter_options;
  for (const auto method : {
         algorithms::BarycenterMethod::DBA,
         algorithms::BarycenterMethod::SSG}) {
    CAPTURE(static_cast<int>(method));
    barycenter_options.method = method;
    CHECK_THROWS_WITH(
      algorithms::dtw_barycenter(overflow_problem, {0, 1}, 1, barycenter_options),
      "dtw_barycenter: computed squared-DTW cost is non-finite; "
      "rescale input values to a smaller magnitude.");
  }

  auto update_overflow = make_problem({{largest}, {largest}});
  barycenter_options.method = algorithms::BarycenterMethod::DBA;
  CHECK_THROWS_WITH(
    algorithms::dtw_barycenter(update_overflow, {0, 1}, 1, barycenter_options),
    "dtw_barycenter: computed barycenter update is non-finite; "
    "rescale input values to a smaller magnitude.");

  barycenter_options.method = algorithms::BarycenterMethod::SoftDTW;
  CHECK_THROWS_WITH(
    algorithms::dtw_barycenter(overflow_problem, {0, 1}, 1, barycenter_options),
    "dtw_barycenter: computed soft-DTW value or gradient is non-finite; "
    "rescale input values to a smaller magnitude.");

  algorithms::BarycenterClusteringOptions clustering_options;
  clustering_options.n_clusters = 1;
  clustering_options.method = algorithms::BarycenterMethod::DBA;
  CHECK_THROWS_WITH(
    algorithms::barycenter_kmeans(overflow_problem, clustering_options),
    "barycenter_kmeans: computed assignment cost is non-finite; "
    "rescale input values to a smaller magnitude.");

  auto initialization_overflow = make_problem({{largest}, {-largest}, {0.0}});
  clustering_options.n_clusters = 2;
  CHECK_THROWS_WITH(
    algorithms::barycenter_kmeans(initialization_overflow, clustering_options),
    "barycenter_kmeans: computed initialization distance total is non-finite; "
    "rescale input values to a smaller magnitude.");
}

TEST_CASE("barycenter k-means mixed-length no-op fingerprint",
          "[barycenter][kmeans][fingerprint]")
{
  auto problem = make_problem({
    {-7.20, -6.70, -5.95, -6.45},
    {0.15, 0.85, 1.75, 1.05, 0.10},
    {10.20, 10.95, 12.10, 11.15, 10.05, 9.80},
    {-6.90, -6.25, -5.55, -5.90, -6.60},
    {0.00, 0.55, 1.25, 1.90, 1.15, 0.25},
    {9.75, 10.45, 11.60, 12.05, 10.85},
    {-7.45, -6.85, -6.10, -5.70, -6.20, -7.00},
    {0.35, 1.10, 1.95, 1.30},
    {10.55, 11.20, 12.35, 11.55, 10.40, 9.95, 9.70}
  });

  algorithms::BarycenterClusteringOptions options;
  options.n_clusters = 3;
  options.max_iter = 8;
  options.barycenter_max_iter = 9;
  options.target_length = 5;
  options.method = algorithms::BarycenterMethod::SSG;
  options.learning_rate = 0.075;
  options.learning_rate_decay = 0.03;
  options.tolerance = 0.0;
  options.random_seed = 123456789ULL;

  const auto result = algorithms::barycenter_kmeans(problem, options);
  // Re-recorded under the portable-v1 seeded schedule after the original
  // serial/workspace oracle exposed vendor RNG drift. Exact equality now pins
  // assignment order, cluster-local streams, centres, and final inertia on
  // every standard library.
  const std::vector<int> expected_labels{0, 2, 1, 0, 2, 1, 0, 2, 1};
  const std::vector<std::vector<data_t>> expected_barycenters{
    {-7.0173908226487498, -6.10105517828847255, -5.73290735028454357,
     -6.01109573627146609, -6.67454216978708459},
    {10.23402574201830717, 11.2577535371416424, 12.00215999834467695,
     10.76294845388155785, 10.07312117090305037},
    {0.25560480948132586, 1.04430561746080963, 1.86198585437672093,
     1.1628100586225909, 0.53901004613194803}
  };

  REQUIRE(result.labels == expected_labels);
  REQUIRE(result.barycenters == expected_barycenters);
  REQUIRE(result.total_cost == 3.9239742509803337);
  REQUIRE(result.iterations == 1);
  REQUIRE(result.converged);
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

TEST_CASE("barycenter entry points reject unsupported Problem DTW configuration",
          "[barycenter][errors][configuration]")
{
  auto problem = make_problem({{0.0, 1.0, 0.0}, {0.0, 2.0, 0.0}});
  algorithms::BarycenterClusteringOptions clustering_options;
  clustering_options.n_clusters = 1;

  SECTION("non-Standard variant")
  {
    for (const auto variant : {core::DTWVariant::DDTW, core::DTWVariant::WDTW,
                               core::DTWVariant::ADTW, core::DTWVariant::SoftDTW,
                               core::DTWVariant::MSM, core::DTWVariant::TWE}) {
      CAPTURE(static_cast<int>(variant));
      problem.set_variant(variant);

      REQUIRE_THROWS_AS(
        algorithms::dtw_barycenter(problem, {0, 1}, 3), InvalidInput);
      REQUIRE_THROWS_WITH(
        algorithms::dtw_barycenter(problem, {0, 1}, 3),
        "dtw_barycenter: only DTWVariant::Standard is supported; "
        "set the Problem variant to DTWVariant::Standard.");
      REQUIRE_THROWS_AS(
        algorithms::barycenter_kmeans(problem, clustering_options), InvalidInput);
      REQUIRE_THROWS_WITH(
        algorithms::barycenter_kmeans(problem, clustering_options),
        "barycenter_kmeans: only DTWVariant::Standard is supported; "
        "set the Problem variant to DTWVariant::Standard.");
    }
  }

  SECTION("non-default band")
  {
    problem.set_band(1);

    REQUIRE_THROWS_AS(
      algorithms::dtw_barycenter(problem, {0, 1}, 3), InvalidInput);
    REQUIRE_THROWS_WITH(
      algorithms::dtw_barycenter(problem, {0, 1}, 3),
      "dtw_barycenter: banded DTW is not supported; set the Problem band to -1.");
    REQUIRE_THROWS_AS(
      algorithms::barycenter_kmeans(problem, clustering_options), InvalidInput);
    REQUIRE_THROWS_WITH(
      algorithms::barycenter_kmeans(problem, clustering_options),
      "barycenter_kmeans: banded DTW is not supported; set the Problem band to -1.");
  }
}
