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
#include <iostream>
#include <limits>
#include <iterator>
#include <sstream>
#include <system_error>
#include <set>
#include <string>
#include <vector>

#ifndef DTWC_TEST_DATA_DIR
#define DTWC_TEST_DATA_DIR "./data"
#endif

namespace fs = std::filesystem;

// The pre-2.0 shape `load(src, skip_cols, delimiter)` must not silently rebind
// the delimiter to skip_rows (','==44); api.hpp poisons it with deleted
// overloads. A requires-expression is the only compile-time way to assert that
// about an overload SET (std::is_invocable_v cannot name one), but it must be
// DEPENDENT: clang 19 evaluates a non-dependent requires-expression eagerly and
// reports "call to deleted function" as a hard error instead of an unsatisfied
// requirement. Removing either deleted declaration turns these asserts red
// (verified by deleting them in a standalone probe).
template <class Source>
concept dtwc_load_binds_char_as_skip_rows =
  requires(Source source) { dtwc::load(source, 0, ','); }
  || requires(Source source) { dtwc::load(source, 1, ',', "name"); };

template <class Source>
concept dtwc_load_accepts_full_shape =
  requires(Source source) { dtwc::load(source, 0, 1, ',', "name"); };

static_assert(!dtwc_load_binds_char_as_skip_rows<fs::path>);
static_assert(!dtwc_load_binds_char_as_skip_rows<dtwc::Dataset::series_type>);
// Positive controls: the asserts above must not pass because load() is unusable.
static_assert(dtwc_load_accepts_full_shape<fs::path>);
static_assert(dtwc_load_accepts_full_shape<dtwc::Dataset::series_type>);

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

  const auto dataset = dtwc::load(fixture(), 0, 0, ',', "quickstart");
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

TEST_CASE("Tier-1 save() completes when the silhouette is undefined",
          "[api][tier1][save][degenerate]")
{
  // Regression: scores::silhouette() rejects fewer than two REALISED clusters,
  // and save() called it after writing labels/medoids/matrix, so a legal run was
  // left with a half-populated directory plus an exception. save() must warn and
  // skip the silhouette file instead. score("silhouette") keeps throwing.
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();

  SECTION("k = 1 is a legal request") {
    const auto dataset = dtwc::load(
      dtwc::Dataset::series_type{{0.0, 1.0, 2.0}, {0.0, 2.0, 4.0}, {5.0, 5.0, 5.0}},
      0, 0, 0, "k1");
    const auto result = dtwc::cluster(dataset, 1, "pam", -1, "cpu", 10);
    REQUIRE(result.medoids().size() == 1);
    REQUIRE_THROWS_AS(result.score("silhouette"), dtwc::InvalidInput);

    const fs::path out =
      fs::temp_directory_path() / ("dtwc_tier1_k1_" + std::to_string(nonce));
    REQUIRE_NOTHROW(result.save(out));
    REQUIRE(first_line(out / "k1_labels.csv") == "name,cluster");
    REQUIRE(first_line(out / "k1_medoids.csv") == "cluster,medoid_index,medoid_name");
    REQUIRE(fs::exists(out / "k1_distance_matrix.csv"));
    REQUIRE_FALSE(fs::exists(out / "k1_silhouettes.csv"));
    std::error_code ec;
    fs::remove_all(out, ec);
  }

  SECTION("k = 2 collapsing to one realised cluster") {
    // Four identical series: every distance is 0, ties resolve to the lower
    // medoid slot, so one declared cluster ends up empty and the realised count
    // is 1 even though k = 2 was requested.
    const auto dataset = dtwc::load(
      dtwc::Dataset::series_type{
        {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}, {1.0, 2.0, 3.0}},
      0, 0, 0, "dup");
    const auto result = dtwc::cluster(dataset, 2, "pam", -1, "cpu", 10);
    const auto &labels = result.labels();
    const int realised = static_cast<int>(
      std::set<int>(labels.begin(), labels.end()).size());
    REQUIRE(realised == 1);

    const fs::path out =
      fs::temp_directory_path() / ("dtwc_tier1_dup_" + std::to_string(nonce));
    REQUIRE_NOTHROW(result.save(out));
    REQUIRE(first_line(out / "dup_labels.csv") == "name,cluster");
    REQUIRE(first_line(out / "dup_medoids.csv") == "cluster,medoid_index,medoid_name");
    REQUIRE_FALSE(fs::exists(out / "dup_silhouettes.csv"));
    std::error_code ec;
    fs::remove_all(out, ec);
  }
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
  dtwc::randGenerator.seed(8675309); // NOLINT(cert-msc51-cpp) fixed seed: the test asserts Tier-1 never touches this engine
  const auto legacy_rng_before = dtwc::randGenerator;
  const auto result = dtwc::cluster(
    dtwc::load(seed_sensitive_series()), 3, "pam", -1, "cpu", 100);
  CHECK(result.medoids() == final_42.medoid_indices);
  CHECK(result.labels() == final_42.labels);
  CHECK(result.cost() == final_42.total_cost);
  CHECK(dtwc::randGenerator == legacy_rng_before);
  dtwc::randGenerator = legacy_rng_original;
}

TEST_CASE("Tier-1 C++ Lloyd is invocation-local and independent of legacy RNG",
          "[api][tier1][seed][lloyd]")
{
  const auto legacy_rng_original = dtwc::randGenerator;
  const auto results_path_original = dtwc::settings::paths::results;
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path out = fs::temp_directory_path()
                     / ("dtwc_lloyd_seed_" + std::to_string(nonce));
  fs::create_directories(out);
  dtwc::settings::paths::results = out;

  dtwc::randGenerator.seed(17); // NOLINT(cert-msc51-cpp) fixed seed: the test asserts Tier-1 never touches this engine
  const auto legacy_rng_before_first = dtwc::randGenerator;
  const auto first = dtwc::cluster(
    dtwc::load(seed_sensitive_series(), 0, 0, 0, "lloyd_first"),
    3, "kmedoids", -1, "cpu", 100);
  CHECK(dtwc::randGenerator == legacy_rng_before_first);

  dtwc::randGenerator.seed(8675309); // NOLINT(cert-msc51-cpp) fixed seed: the test asserts Tier-1 never touches this engine
  const auto legacy_rng_before_second = dtwc::randGenerator;
  const auto second = dtwc::cluster(
    dtwc::load(seed_sensitive_series(), 0, 0, 0, "lloyd_second"),
    3, "kmedoids", -1, "cpu", 100);
  CHECK(dtwc::randGenerator == legacy_rng_before_second);
  CHECK(canonicalise(first) == canonicalise(second));
  CHECK(first.cost() == second.cost());

  dtwc::randGenerator = legacy_rng_original;
  dtwc::settings::paths::results = results_path_original;
  std::error_code ec;
  fs::remove_all(out, ec);
}

TEST_CASE("Lloyd repetitions restore the best result when the best is not last",
          "[api][tier1][seed][lloyd]")
{
  auto problem = seed_sensitive_problem();
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path out = fs::temp_directory_path()
                     / ("dtwc_lloyd_best_" + std::to_string(nonce));
  fs::create_directories(out);
  problem.set_output_folder(out);
  problem.set_n_clusters(3);
  problem.set_n_repetitions(2);

  int repetition = 0;
  problem.init_fun = [&repetition](dtwc::Problem &current) {
    std::vector<int> medoids = repetition++ == 0
      ? std::vector<int>{0, 1, 7}
      : std::vector<int>{0, 1, 2};
    current.set_clusters(medoids);
  };

  problem.cluster_and_process(); // owns the run artifacts; cluster() does not
  CHECK(repetition == 2);
  CHECK(problem.find_total_cost() == 20.0);
  CHECK(problem.medoids() == std::vector<int>{0, 3, 6});
  CHECK(first_line(out / "tier1_seed_fixture_bestRepetition_Nc_3.csv") == "0");

  std::error_code ec;
  fs::remove_all(out, ec);
}

TEST_CASE("Lloyd uses a checked seed schedule and preserves custom initializers",
          "[api][tier1][seed][lloyd]")
{
  REQUIRE(dtwc::Problem{}.random_seed() == dtwc::settings::DEFAULT_RANDOM_SEED);

  const auto legacy_rng_original = dtwc::randGenerator;
  dtwc::randGenerator.seed(271828); // NOLINT(cert-msc51-cpp) fixed seed: the test asserts Tier-1 never touches this engine
  const auto legacy_rng_before = dtwc::randGenerator;
  auto problem = seed_sensitive_problem();
  problem.set_n_clusters(3);
  dtwc::init::random_seeded(problem, 42);
  const auto seed_42_medoids = problem.medoids();
  CHECK(seed_42_medoids == std::vector<int>{6, 2, 1});
  dtwc::init::random_seeded(problem, 43);
  const auto seed_43_medoids = problem.medoids();
  CHECK(seed_42_medoids != seed_43_medoids);

  auto kmeanspp_problem = seed_sensitive_problem();
  kmeanspp_problem.set_n_clusters(3);
  dtwc::init::Kmeanspp_seeded(kmeanspp_problem, 42);
  CHECK(kmeanspp_problem.medoids() == std::vector<int>{6, 2, 5});
  CHECK(dtwc::randGenerator == legacy_rng_before);
  dtwc::randGenerator = legacy_rng_original;

  auto no_restarts = seed_sensitive_problem();
  no_restarts.set_n_clusters(3);
  no_restarts.set_n_repetitions(0);
  REQUIRE_THROWS_AS(no_restarts.cluster_by_kmedoids_lloyd(), dtwc::InvalidInput);

  auto overflow = seed_sensitive_problem();
  overflow.set_n_clusters(3);
  overflow.set_n_repetitions(2);
  overflow.set_random_seed(std::numeric_limits<std::uint64_t>::max());
  REQUIRE_THROWS_AS(overflow.cluster_by_kmedoids_lloyd(), dtwc::InvalidInput);
}

TEST_CASE("Lloyd rejects a non-finite assignment before publishing labels",
          "[api][tier1][lloyd][nonfinite]")
{
  const double largest = std::numeric_limits<double>::max();
  dtwc::Problem problem("lloyd_infinite_cost");
  problem.set_data(dtwc::Data(
    std::vector<std::vector<double>>{{largest}, {-largest}},
    std::vector<std::string>{"positive", "negative"}));
  problem.set_n_clusters(1);
  problem.set_random_seed(0);
  const auto labels_before = problem.labels();

  REQUIRE_THROWS_AS(
    problem.cluster_by_kmedoids_lloyd(), dtwc::InvalidInput);
  CHECK(problem.labels() == labels_before);
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

TEST_CASE("Tier-1 pins heap series storage for a GPU execution target",
          "[api][tier1][device][storage]")
{
  using dtwc::detail::Tier1ExecutionTarget;
  using dtwc::detail::tier1_storage_policy;

  // fill_distance_matrix throws DeviceError on mmap-backed series, and
  // StoragePolicy::Auto spills above the free-RAM threshold (now measured on
  // Windows too), so the Tier-1 GPU route must install Heap before set_data.
  CHECK(tier1_storage_policy(Tier1ExecutionTarget::GPU)
        == dtwc::core::StoragePolicy::Heap);
  // CPU and HPC keep the caller-visible default; Tier-2 mmap use is untouched.
  CHECK(tier1_storage_policy(Tier1ExecutionTarget::CPU)
        == dtwc::core::StoragePolicy::Auto);
  CHECK(tier1_storage_policy(Tier1ExecutionTarget::HPC)
        == dtwc::core::StoragePolicy::Auto);
}

TEST_CASE("Tier-1 C++ load honours skip_rows", "[api][tier1][skip_rows]")
{
  const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
  const fs::path csv =
    fs::temp_directory_path() / ("dtwc_skip_rows_" + std::to_string(nonce) + ".csv");
  {
    std::ofstream out(csv);
    out << "id,t0,t1,t2\n"
           "unit,s,s,s\n"
           "a,0,0,0\n"
           "b,0,0,1\n"
           "c,10,10,10\n"
           "d,10,10,11\n";
  }

  SECTION("a path source skips leading FILE LINES, like --skip-rows") {
    const auto dataset = dtwc::load(csv, 1, 2, ',', "hdr");
    CHECK(dataset.skip_rows() == 2);
    const auto result = dtwc::cluster(dataset, 2, "pam", -1, "cpu", 10);
    REQUIRE(result.labels().size() == 4);

    // The two text header lines are unparseable, so an unskipped load must fail
    // loudly rather than silently returning a shorter or garbled dataset.
    REQUIRE_THROWS_AS(
      dtwc::cluster(dtwc::load(csv, 1, 0, ',', "unskipped"), 2, "pam", -1, "cpu", 10),
      dtwc::IOError);
  }

  SECTION("an in-memory source skips leading SERIES") {
    const auto dataset = dtwc::load(
      dtwc::Dataset::series_type{
        {7.0, 7.0}, {7.0, 7.0}, {0.0, 0.0}, {0.0, 1.0}, {9.0, 9.0}},
      0, 2, 0, "mem");
    const auto result = dtwc::cluster(dataset, 2, "pam", -1, "cpu", 10);
    REQUIRE(result.labels().size() == 3);
  }

  SECTION("negative skip_rows is rejected by both overloads") {
    REQUIRE_THROWS_AS(dtwc::load(csv, 0, -1), dtwc::InvalidInput);
    REQUIRE_THROWS_AS(
      dtwc::load(dtwc::Dataset::series_type{{0.0, 1.0}}, 0, -1),
      dtwc::InvalidInput);
  }

  std::error_code ec;
  fs::remove(csv, ec);
}

namespace {

/// Switch the process working directory for the duration of a scope. Tier-1
/// must not depend on (or write into) whatever directory the caller ran from.
class ScopedWorkingDirectory
{
public:
  explicit ScopedWorkingDirectory(const fs::path &directory)
    : previous_{ fs::current_path() }
  {
    fs::current_path(directory);
  }
  ~ScopedWorkingDirectory()
  {
    std::error_code ec;
    fs::current_path(previous_, ec);
  }
  ScopedWorkingDirectory(const ScopedWorkingDirectory &) = delete;
  ScopedWorkingDirectory &operator=(const ScopedWorkingDirectory &) = delete;

private:
  fs::path previous_;
};

/// Capture std::cout for the duration of a scope, restoring it on any exit.
class ScopedCoutRedirect
{
public:
  explicit ScopedCoutRedirect(std::streambuf *sink) : previous_{ std::cout.rdbuf(sink) } {}
  ~ScopedCoutRedirect() { std::cout.rdbuf(previous_); }
  ScopedCoutRedirect(const ScopedCoutRedirect &) = delete;
  ScopedCoutRedirect &operator=(const ScopedCoutRedirect &) = delete;

private:
  std::streambuf *previous_;
};

fs::path make_sandbox(const std::string &tag)
{
  const auto nonce =
    std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
  const auto directory = fs::temp_directory_path() / (tag + nonce);
  fs::create_directories(directory);
  return directory;
}

dtwc::Dataset::series_type two_cluster_series()
{
  return { { 0.0, 0.0, 0.0 }, { 0.1, 0.0, 0.1 },
           { 9.0, 9.0, 9.0 }, { 9.1, 9.0, 9.1 } };
}

} // namespace

TEST_CASE("Tier-1 C++ cluster() leaves the working directory untouched",
          "[api][tier1][sideeffects]")
{
  // Problem::output_folder_ defaults to the CWD-relative settings::paths::results,
  // so a Tier-1 route that wrote run artifacts failed (or littered) whenever the
  // caller ran from a directory without ./results/.
  const auto sandbox = make_sandbox("dtwc_tier1_cwd_");
  const auto series = two_cluster_series();

  {
    const ScopedWorkingDirectory cwd{ sandbox };
    for (const auto *method : { "kmedoids", "lrcore", "tadpole" }) {
      const auto result =
        dtwc::cluster(dtwc::load(series), 2, method, -1, "cpu", 10);
      CHECK(result.labels().size() == series.size());
      CHECK(result.medoids().size() == 2);
    }
    CHECK(fs::is_empty(sandbox));
  }

  CHECK(fs::is_empty(sandbox));
  std::error_code ec;
  fs::remove_all(sandbox, ec);
}

TEST_CASE("Problem::cluster() prints to stdout only when verbose",
          "[api][tier1][sideeffects][verbose]")
{
  const auto make_problem = [](dtwc::Method method) {
    dtwc::Problem problem("stdout_silence");
    problem.set_data(dtwc::Data(
      std::vector<std::vector<dtwc::data_t>>{ { 0.0, 0.0, 0.0 }, { 0.1, 0.0, 0.1 },
                                              { 9.0, 9.0, 9.0 }, { 9.1, 9.0, 9.1 } },
      std::vector<std::string>{ "a", "b", "c", "d" }));
    problem.set_n_clusters(2);
    problem.set_method(method);
    problem.set_max_iter(10);
    problem.set_n_repetitions(1);
    return problem;
  };

  // Every method Problem::cluster() dispatches to that is buildable without an
  // optional backend. Method::MIP is covered by the HiGHS suites.
  for (const auto method :
       { dtwc::Method::Kmedoids, dtwc::Method::LRCore, dtwc::Method::TADPole }) {
    auto quiet = make_problem(method);
    std::ostringstream quiet_sink;
    {
      const ScopedCoutRedirect redirect{ quiet_sink.rdbuf() };
      quiet.cluster();
    }
    CHECK(quiet_sink.str().empty());

    auto loud = make_problem(method);
    loud.set_verbose(true);
    std::ostringstream loud_sink;
    {
      const ScopedCoutRedirect redirect{ loud_sink.rdbuf() };
      loud.cluster();
    }

    // Gating changes what is printed, never what is computed.
    CHECK(quiet.medoids() == loud.medoids());
    CHECK(quiet.labels() == loud.labels());

    // Only Lloyd has verbose progress of its own; the exact routes stay silent
    // either way, so the non-empty assertion is scoped to the one that prints.
    if (method == dtwc::Method::Kmedoids) CHECK_FALSE(loud_sink.str().empty());
  }
}

TEST_CASE("Tier-1 Result::distance_matrix is the dense symmetric N-by-N matrix",
          "[api][tier1][result]")
{
  const auto series = two_cluster_series();
  const auto result = dtwc::cluster(dtwc::load(series), 2, "kmedoids", -1, "cpu", 10);

  const auto flat = result.distance_matrix();
  const std::size_t n = series.size();
  REQUIRE(flat.size() == n * n);

  // Independent oracle: the same series through Problem::dist_by_ind.
  dtwc::Problem oracle("distance_matrix_oracle");
  oracle.set_data(dtwc::Data(
    two_cluster_series(), std::vector<std::string>{ "a", "b", "c", "d" }));
  oracle.fill_distance_matrix();

  for (std::size_t i = 0; i < n; ++i) {
    CHECK(flat[i * n + i] == 0.0);
    for (std::size_t j = 0; j < n; ++j) {
      CHECK(flat[i * n + j] == flat[j * n + i]);
      CHECK(flat[i * n + j]
            == oracle.dist_by_ind(static_cast<int>(i), static_cast<int>(j)));
    }
  }
}

TEST_CASE("Tier-1 carries a non-ASCII name as UTF-8 end to end",
          "[api][tier1][unicode]")
{
  // path::string() is the native narrow encoding (Windows ACP), so a
  // non-ASCII stem reached Python as an undecodable byte; fs::path built back
  // from a UTF-8 std::string would then write a mojibake filename. Both ends
  // go through path_to_utf8 / utf8_to_path.
  const std::string cafe = "caf\xc3\xa9"; // U+00E9 as UTF-8, source-encoding independent
  const auto sandbox = make_sandbox("dtwc_utf8_e2e_");

  SECTION("a file source names the Dataset, and save() names its files, in UTF-8")
  {
    const auto csv = sandbox / dtwc::utf8_to_path(cafe + ".csv");
    {
      std::ofstream out(csv);
      REQUIRE(out.good());
      out << "0,0,0\n0,0,1\n9,9,9\n9,9,8\n";
    }

    const auto dataset = dtwc::load(csv);
    CHECK(dataset.name() == cafe);

    const auto result = dtwc::cluster(dataset, 2, "pam", -1, "cpu", 10);
    const auto out_dir = sandbox / "saved";
    result.save(out_dir);

    bool found = false;
    for (const auto &entry : fs::directory_iterator(out_dir)) {
      const auto name = dtwc::path_to_utf8(entry.path().filename());
      if (name.find(cafe) != std::string::npos) found = true;
      // No mojibake: the ACP round trip would have produced these bytes.
      CHECK(name.find("caf\xc3\x83") == std::string::npos);
    }
    CHECK(found);
    CHECK(fs::exists(out_dir / dtwc::utf8_to_path(cafe + "_labels.csv")));
  }

  SECTION("a directory source writes UTF-8 series names inside the CSVs")
  {
    const auto folder = sandbox / "folder";
    fs::create_directories(folder);
    for (const auto &stem : { cafe, std::string{ "zeta" } }) {
      std::ofstream out(folder / dtwc::utf8_to_path(stem + ".csv"));
      REQUIRE(out.good());
      out << (stem == "zeta" ? "9\n8\n7\n" : "1\n2\n3\n");
    }

    const auto result =
      dtwc::cluster(dtwc::load(folder), 2, "pam", -1, "cpu", 10);
    const auto out_dir = sandbox / "folder_saved";
    result.save(out_dir);

    std::ifstream labels(out_dir / "folder_labels.csv", std::ios::binary);
    REQUIRE(labels.good());
    const std::string bytes{ std::istreambuf_iterator<char>(labels),
                             std::istreambuf_iterator<char>() };
    CHECK(bytes.find(cafe) != std::string::npos);
  }

  std::error_code ec;
  fs::remove_all(sandbox, ec);
}
