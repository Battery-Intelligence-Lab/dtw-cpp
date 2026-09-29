/**
 * @file unit_test_invalid_public_selectors.cpp
 * @brief Every declared value of the clustering selectors is accepted and dispatches.
 *
 * Method, Solver, algorithms::Linkage, algorithms::BarycenterMethod and Device
 * reach the library from a user only as a name (bindings, CLI, config files),
 * and each name table rejects an unknown spelling; every switch over them is
 * exhaustive at compile time (-Werror=switch, /we4062). What is left to prove
 * here is that each declared value runs.
 */

#include <dtwc.hpp>

#include <algorithms/barycenter.hpp>
#include <algorithms/hierarchical.hpp>
#include <base/env.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace {

using namespace dtwc;

Problem make_problem()
{
  constexpr std::size_t n = 4;
  std::vector<std::vector<data_t>> series;
  std::vector<std::string> names;
  for (std::size_t i = 0; i < n; ++i) {
    series.push_back({static_cast<double>(i * i + i)});
    names.push_back("s" + std::to_string(i));
  }
  Problem problem("f1_selector_fixture");
  problem.set_verbose(false);
  problem.set_data(Data(std::move(series), std::move(names)));
  problem.set_n_clusters(2);
  return problem;
}

} // namespace

TEST_CASE("F1 all declared Method values remain accepted",
          "[f1][enum][valid][method]")
{
  constexpr std::array values{
    Method::Kmedoids, Method::MIP, Method::LRCore, Method::TADPole
  };
  for (const Method value : values) {
    auto problem = make_problem();
    CHECK_NOTHROW(problem.set_method(value));
    CHECK(problem.method() == value);
  }
}

TEST_CASE("F1 all declared Solver values remain accepted",
          "[f1][enum][valid][solver]")
{
  constexpr std::array values{Solver::Gurobi, Solver::HiGHS};
  for (const Solver value : values) {
    auto problem = make_problem();
    CHECK_NOTHROW((void)problem.set_solver(value));
  }
}

TEST_CASE("F1 all declared Linkage values dispatch",
          "[f1][enum][valid][linkage]")
{
  using algorithms::Linkage;
  constexpr std::array values{Linkage::Single, Linkage::Complete,
                              Linkage::Average};
  for (const Linkage value : values) {
    auto problem = make_problem();
    problem.fill_distance_matrix();
    algorithms::HierarchicalOptions options;
    options.linkage = value;
    const auto dendrogram = algorithms::build_dendrogram(problem, options);
    CHECK(dendrogram.n_points == 4);
    CHECK(dendrogram.merges.size() == 3);
  }
}

TEST_CASE("F1 all declared BarycenterMethod values dispatch at both entry points",
          "[f1][enum][valid][barycenter]")
{
  using algorithms::BarycenterMethod;
  constexpr std::array values{
    BarycenterMethod::SSG,
    BarycenterMethod::DBA,
    BarycenterMethod::SoftDTW
  };
  for (const BarycenterMethod value : values) {
    auto problem = make_problem();
    algorithms::BarycenterOptions barycenter;
    barycenter.method = value;
    barycenter.max_iter = 1;
    const auto center = algorithms::dtw_barycenter(
      problem, {0, 1, 2, 3}, 1, barycenter);
    REQUIRE(center.size() == 1);
    CHECK(std::isfinite(center[0]));

    algorithms::BarycenterClusteringOptions clustering;
    clustering.n_clusters = 2;
    clustering.max_iter = 1;
    clustering.barycenter_max_iter = 1;
    clustering.method = value;
    const auto result = algorithms::barycenter_kmeans(problem, clustering);
    CHECK(result.labels.size() == 4);
    CHECK(result.barycenters.size() == 2);
    CHECK(std::isfinite(result.total_cost));
  }
}

TEST_CASE("F1 all declared Device values report their exact names",
          "[f1][enum][valid][device]")
{
  CHECK(to_string(Device::CPU) == "cpu");
  CHECK(to_string(Device::GPU) == "gpu");
}
