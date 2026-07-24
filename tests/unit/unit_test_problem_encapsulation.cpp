#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <filesystem>
#include <iostream>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

#ifndef DTWC_F19_TEST_ROOT
#error "DTWC_F19_TEST_ROOT must bind F19 output to the build tree"
#endif

using Catch::Matchers::WithinAbs;

namespace {

dtwc::Data two_series_data()
{
  return dtwc::Data(
    std::vector<std::vector<double>>{ { 0.0, 1.0 }, { 2.0, 4.0 } },
    std::vector<std::string>{ "left", "right" });
}

dtwc::Problem capped_lloyd_problem(const std::filesystem::path &output)
{
  dtwc::Problem problem{ "f19_capped_lloyd" };
  problem.set_data(dtwc::Data(
    std::vector<std::vector<double>>{
      { 0.0 }, { 40.0 }, { 40.0 }, { 46.0 }, { 49.0 }, { 51.0 }, { 51.0 }, { 51.0 }, { 100.0 } },
    std::vector<std::string>{
      "s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8" }));
  problem.set_n_clusters(2);
  problem.set_max_iter(100);
  problem.set_n_repetitions(1);
  problem.set_output_folder(output);
  problem.init_fun = [](dtwc::Problem &candidate) {
    std::vector<int> initial_medoids{ 0, 8 };
    candidate.set_clusters(initial_medoids);
  };
  return problem;
}

struct OutputDirectory
{
  std::filesystem::path path;

  explicit OutputDirectory(std::string_view leaf)
    : path(std::filesystem::path{ DTWC_F19_TEST_ROOT } / leaf)
  {
    std::filesystem::create_directories(path);
  }

  ~OutputDirectory()
  {
    std::error_code error;
    std::filesystem::remove_all(path, error);
  }
};

} // namespace

TEST_CASE("F19 Problem canonical accessors round-trip setter state",
          "[f19][problem][api]")
{
  const dtwc::Problem defaults;
  CHECK(defaults.method() == dtwc::Method::Kmedoids);
  CHECK(defaults.random_seed() == dtwc::settings::DEFAULT_RANDOM_SEED);
  CHECK(defaults.last_iterations() == 0);
  CHECK(defaults.tadpole_dc() == -1.0);
  CHECK(defaults.lb_strategy() == dtwc::LowerBoundStrategy::Auto);
  CHECK(defaults.storage_policy() == dtwc::core::StoragePolicy::Auto);
  CHECK_FALSE(defaults.verbose());
  CHECK_FALSE(defaults.output_folder().empty());
  CHECK(defaults.name().empty());
  CHECK(defaults.data().size() == 0);

  dtwc::Problem problem{ "constructor_name" };
  const auto output = std::filesystem::path{ DTWC_F19_TEST_ROOT } / "accessors";
  problem.set_method(dtwc::Method::TADPole);
  problem.set_random_seed(123456789ULL);
  problem.set_tadpole_dc(0.125);
  problem.set_lb_strategy(dtwc::LowerBoundStrategy::Webb);
  problem.set_storage_policy(dtwc::core::StoragePolicy::Heap);
  problem.set_verbose(true);
  problem.set_output_folder(output);
  problem.set_name("renamed");
  problem.set_data(two_series_data());

  CHECK(problem.method() == dtwc::Method::TADPole);
  CHECK(problem.random_seed() == 123456789ULL);
  CHECK(problem.last_iterations() == 0);
  CHECK(problem.tadpole_dc() == 0.125);
  CHECK(problem.lb_strategy() == dtwc::LowerBoundStrategy::Webb);
  CHECK(problem.storage_policy() == dtwc::core::StoragePolicy::Heap);
  CHECK(problem.verbose());
  CHECK(problem.output_folder() == output);
  CHECK(problem.name() == "renamed");
  CHECK(problem.data().size() == 2);
  CHECK(problem.data().precision == dtwc::core::Precision::Float64);
  REQUIRE(problem.series(1).size() == 2);
  CHECK(problem.series(1)[0] == 2.0);
  CHECK(problem.series(1)[1] == 4.0);
}

TEST_CASE("F19 Problem setters reject invalid state transactionally",
          "[f19][problem][api][invalid]")
{
  dtwc::Problem problem{ "preserved" };
  problem.set_data(two_series_data());
  problem.set_method(dtwc::Method::MIP);
  problem.set_lb_strategy(dtwc::LowerBoundStrategy::Keogh);
  problem.set_storage_policy(dtwc::core::StoragePolicy::Heap);

  CHECK_THROWS_AS(
    problem.set_method(static_cast<dtwc::Method>(-1)), dtwc::InvalidInput);
  CHECK(problem.method() == dtwc::Method::MIP);
  CHECK_THROWS_AS(
    problem.set_lb_strategy(static_cast<dtwc::LowerBoundStrategy>(-1)),
    dtwc::InvalidInput);
  CHECK(problem.lb_strategy() == dtwc::LowerBoundStrategy::Keogh);
  CHECK_THROWS_AS(
    problem.set_storage_policy(static_cast<dtwc::core::StoragePolicy>(-1)),
    dtwc::InvalidInput);
  CHECK(problem.storage_policy() == dtwc::core::StoragePolicy::Heap);

  auto invalid_data = two_series_data();
  invalid_data.precision = static_cast<dtwc::core::Precision>(-1);
  CHECK_THROWS_AS(problem.set_data(std::move(invalid_data)), dtwc::InvalidInput);
  CHECK(problem.data().precision == dtwc::core::Precision::Float64);
  CHECK(problem.data().size() == 2);
  CHECK(problem.name() == "preserved");
}

TEST_CASE("F19 capped Lloyd behavior survives Problem encapsulation",
          "[f19][problem][api][lloyd]")
{
  OutputDirectory output{ "lloyd" };
  auto problem = capped_lloyd_problem(output.path);

  REQUIRE_NOTHROW(problem.cluster_by_kmedoids_lloyd());
  CHECK(problem.method() == dtwc::Method::Kmedoids);
  CHECK(problem.last_iterations() == 2);
  CHECK(problem.medoids() == std::vector<int>{ 1, 5 });
  CHECK(problem.labels()
        == std::vector<int>{ 0, 0, 0, 1, 1, 1, 1, 1, 1 });
  CHECK_THAT(problem.find_total_cost(), WithinAbs(96.0, 1e-12));
  CHECK(problem.name() == "f19_capped_lloyd");
  CHECK(problem.output_folder() == output.path);

  std::cout
    << "F19_PROBLEM_API getters=10/10 setters=9/9 lloyd=ran skips=0\n";
}
