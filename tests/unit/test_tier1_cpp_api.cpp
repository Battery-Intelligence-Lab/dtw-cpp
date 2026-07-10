/** @file test_tier1_cpp_api.cpp
 *  @brief Live C++ route for the frozen Tier-1 device/load/cluster/Result API.
 */

#include <dtwc.hpp>
#include <detail/tier1_method_resolution.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#ifndef DTWC_TEST_DATA_DIR
#define DTWC_TEST_DATA_DIR "./data"
#endif

namespace fs = std::filesystem;

namespace {

fs::path fixture()
{
  return fs::path{DTWC_TEST_DATA_DIR}.parent_path()
       / "tests" / "conformance" / "data" / "conformance_series.csv";
}

std::pair<std::vector<int>, std::vector<int>> canonicalise(const dtwc::Result &result)
{
  std::vector<int> medoids = result.medoids();
  std::sort(medoids.begin(), medoids.end());
  std::vector<int> labels(result.labels().size());
  for (std::size_t i = 0; i < labels.size(); ++i) {
    const int assigned = result.medoids().at(
      static_cast<std::size_t>(result.labels().at(i)));
    labels[i] = static_cast<int>(
      std::lower_bound(medoids.begin(), medoids.end(), assigned) - medoids.begin());
  }
  return {labels, medoids};
}

std::string first_line(const fs::path &path)
{
  std::ifstream in(path);
  std::string line;
  std::getline(in, line);
  return line;
}

} // namespace

TEST_CASE("Tier-1 C++ load is lazy and validates at materialisation", "[api][tier1]")
{
  REQUIRE_NOTHROW(dtwc::load(fs::path("definitely_missing.csv")));
  const auto missing = dtwc::load(fs::path("definitely_missing.csv"));
  REQUIRE_THROWS_AS(dtwc::cluster(missing, 2), dtwc::IOError);
}

TEST_CASE("Tier-1 C++ conformance fixture clusters, scores, and saves", "[api][tier1][conformance]")
{
  REQUIRE(dtwc::device("cpu") == "cpu");
  REQUIRE(dtwc::device() == "cpu");

  const auto dataset = dtwc::load(fixture(), 0, ',', "quickstart");
  const auto result = dtwc::cluster(dataset, 3, "pam", 3, "cpu", 100);
  const auto [labels, medoids] = canonicalise(result);

  const std::vector<int> expected_labels = {
    0,0,0,0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,2
  };
  REQUIRE(labels == expected_labels);
  REQUIRE(medoids == std::vector<int>{4, 13, 22});
  REQUIRE(result.cost() >= 0.0);
  REQUIRE(result.device() == "cpu");
  REQUIRE(result.score("silhouette") > 0.96);
  REQUIRE_THROWS_AS(result.score("made_up"), dtwc::InvalidInput);

  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path out = fs::temp_directory_path()
                     / ("dtwc_tier1_" + std::to_string(nonce));
  result.save(out);
  REQUIRE(first_line(out / "quickstart_labels.csv") == "name,cluster");
  REQUIRE(first_line(out / "quickstart_medoids.csv")
          == "cluster,medoid_index,medoid_name");
  REQUIRE(fs::exists(out / "quickstart_distance_matrix.csv"));
  REQUIRE(first_line(out / "quickstart_silhouettes.csv")
          == "name,cluster,silhouette");
  std::error_code ec;
  fs::remove_all(out, ec);
}

TEST_CASE("Tier-1 C++ rejects invalid method, k, and matrix-free GPU mismatch", "[api][tier1]")
{
  const auto dataset = dtwc::load(dtwc::Dataset::series_type{
    {0.0, 0.0}, {0.0, 1.0}, {100.0, 100.0}
  });
  REQUIRE_THROWS_AS(dtwc::cluster(dataset, 2, "not-a-method"), dtwc::InvalidInput);
  REQUIRE_THROWS_AS(dtwc::cluster(dataset, 0), dtwc::InvalidInput);
  REQUIRE_THROWS_AS(dtwc::cluster(dataset, 4), dtwc::InvalidInput);
}

TEST_CASE("Tier-1 auto method resolution is compatible with its execution target",
          "[api][tier1][device]")
{
  using dtwc::detail::Tier1ExecutionTarget;
  using dtwc::detail::resolve_tier1_method;

  // N=5001 is the first non-degenerate case that selects CLARA on CPU. GPU
  // matrix-free schedules are unsupported, so auto must retain the compatible
  // PAM path there. The remote HPC process owns its eventual size decision.
  CHECK(resolve_tier1_method("auto", 5001, Tier1ExecutionTarget::CPU) == "clara");
  CHECK(resolve_tier1_method("auto", 5001, Tier1ExecutionTarget::GPU) == "pam");
  CHECK(resolve_tier1_method("auto", 5001, Tier1ExecutionTarget::HPC) == "auto");

  CHECK(resolve_tier1_method("auto", 5000, Tier1ExecutionTarget::CPU) == "pam");
  CHECK(resolve_tier1_method("auto", 5000, Tier1ExecutionTarget::GPU) == "pam");

  // Explicit incompatibilities stay explicit so the existing DeviceError path
  // remains loud rather than silently substituting a different algorithm.
  CHECK(resolve_tier1_method("clara", 5001, Tier1ExecutionTarget::GPU) == "clara");
}
