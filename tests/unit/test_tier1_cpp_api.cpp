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

dtwc::Dataset::series_type seed_sensitive_series()
{
  // Eight translated, non-constant waveforms have no separated-cluster
  // structure. PAM's BUILD seed therefore changes both its initial medoids and
  // its local optimum without relying on a degenerate length-1 shortcut.
  const std::vector<double> base{0.0, 0.01, -0.02, 0.03};
  dtwc::Dataset::series_type series;
  for (int offset = 0; offset < 8; ++offset) {
    auto waveform = base;
    for (double &value : waveform) value += static_cast<double>(offset);
    series.push_back(std::move(waveform));
  }
  return series;
}

dtwc::Problem seed_sensitive_problem()
{
  auto series = seed_sensitive_series();
  std::vector<std::string> names;
  names.reserve(series.size());
  for (std::size_t i = 0; i < series.size(); ++i)
    names.push_back(std::to_string(i));
  dtwc::Problem problem("tier1_seed_fixture");
  problem.set_data(dtwc::Data(std::move(series), std::move(names)));
  return problem;
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

TEST_CASE("Tier-1 C++ PAM uses the shared local seed without touching legacy RNG",
          "[api][tier1][seed]")
{
  REQUIRE(dtwc::settings::DEFAULT_RANDOM_SEED == 42);

  auto init_29_problem = seed_sensitive_problem();
  auto init_42_problem = seed_sensitive_problem();
  const auto init_29 = dtwc::fast_pam_seeded(init_29_problem, 3, 29, 0);
  const auto init_42 = dtwc::fast_pam_seeded(init_42_problem, 3, 42, 0);
  CHECK(init_29.medoid_indices == std::vector<int>{4, 2, 7});
  CHECK(init_42.medoid_indices == std::vector<int>{6, 2, 5});

  auto final_29_problem = seed_sensitive_problem();
  auto final_42_problem = seed_sensitive_problem();
  const auto final_29 = dtwc::fast_pam_seeded(final_29_problem, 3, 29);
  const auto final_42 = dtwc::fast_pam_seeded(final_42_problem, 3, 42);
  CHECK(final_29.medoid_indices == std::vector<int>{4, 1, 7});
  CHECK(final_29.total_cost == 20.0);
  CHECK(final_42.medoid_indices == std::vector<int>{6, 2, 5});
  CHECK(final_42.total_cost == 24.0);

  // Tier-1 owns an invocation-local engine.  Its result is the seed-42 oracle,
  // and the call must neither consume nor reseed mutable Tier-2 randGenerator.
  const auto legacy_rng_original = dtwc::randGenerator;
  dtwc::randGenerator.seed(8675309);
  const auto legacy_rng_before = dtwc::randGenerator;
  const auto result = dtwc::cluster(
    dtwc::load(seed_sensitive_series()), 3, "pam", -1, "cpu", 100);
  CHECK(result.medoids() == final_42.medoid_indices);
  CHECK(result.labels() == final_42.labels);
  CHECK(result.cost() == final_42.total_cost);
  CHECK(dtwc::randGenerator == legacy_rng_before);
  dtwc::randGenerator = legacy_rng_original;
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
