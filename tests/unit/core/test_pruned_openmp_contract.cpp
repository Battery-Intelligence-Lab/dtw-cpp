/**
 * @file test_pruned_openmp_contract.cpp
 * @brief Mutation guard for the pruned distance-matrix exception boundary.
 *
 * OpenMP structured blocks cannot propagate C++ exceptions.  The shared
 * run_openmp() boundary captures a failure inside the worker and rethrows it
 * on the caller thread in canonical-index order.  A raw `omp parallel` in the
 * pruned translation unit would bypass that contract and can terminate the
 * process, so this source-level guard deliberately complements the runtime
 * behavior tests in unit_test_pruned_distance_matrix.cpp.
 */

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace {

std::string read_pruned_source()
{
  const auto repo_root = std::filesystem::path{DTWC_TEST_DATA_DIR}.parent_path();
  const auto source_path = repo_root / "dtwc" / "core" / "pruned_distance_matrix.cpp";
  std::ifstream source(source_path, std::ios::binary);
  REQUIRE(source.is_open());
  return {std::istreambuf_iterator<char>{source}, std::istreambuf_iterator<char>{}};
}

} // namespace

TEST_CASE("Pruned matrix code uses the shared OpenMP exception boundary",
          "[pruned_distance_matrix][openmp][m43]")
{
  const std::string source = read_pruned_source();

  REQUIRE(source.find("#pragma omp parallel") == std::string::npos);
  REQUIRE(source.find("run_openmp(") != std::string::npos);
}

TEST_CASE("Pruned nearest-neighbor thresholds use one standard atomic representation",
          "[pruned_distance_matrix][openmp][atomic][m44]")
{
  const std::string source = read_pruned_source();

  REQUIRE(source.find("std::vector<double> nn_dist") == std::string::npos);
  REQUIRE(source.find("reinterpret_cast<volatile uint64_t") == std::string::npos);
  REQUIRE(source.find("std::atomic<double>") != std::string::npos);
  REQUIRE(source.find(".load(std::memory_order_relaxed)") != std::string::npos);
}
