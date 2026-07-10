/**
 * @file unit_test_cli_args.cpp
 * @brief Regression tests for dtwc_cl CLI argument parsing (Task 0.9).
 *
 * @details These pin the audit "cli-ux" findings
 * (handoff-2026-06-01-adversarial-audit.md, line 26):
 *
 *   1. `--metric` was consumed ONLY by the CUDA path; on the CPU path
 *      resolve_dtw_fn() always binds MetricType::L1, so a non-L1 metric was
 *      silently ignored (computed L1). Now rejected (no-silent-fallback).
 *   2. `std::stoi(device.substr(5))` on a bad "cuda:N" (e.g. "cuda:abc")
 *      threw std::invalid_argument uncaught -> propagated out of main() ->
 *      std::terminate. Now a clean validation error.
 *   3. `device.rfind("cuda", 0)` was case-sensitive, so "CUDA:0" silently fell
 *      back to CPU; and any unknown device likewise silently ran on CPU. Now
 *      case-insensitive, and unknown devices are a hard error.
 *
 * We compile the CLI translation unit with DTWC_CL_NO_MAIN so that main() and
 * its CLI11 dependency are excluded and the pure parse helpers
 * (parse_device / validate_metric_for_device) become directly callable. This
 * exercises the REAL production code, not a copy.
 *
 * Why the UNFIXED code fails these tests: before the fix, parse_device and
 * validate_metric_for_device did not exist (device handling was inline raw
 * rfind + unchecked std::stoi in main), so this translation unit would not even
 * compile against the old dtwc_cl.cpp — and the buggy behaviours above are
 * exactly what the assertions below forbid.
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#define DTWC_CL_NO_MAIN
#include "../../dtwc/dtwc_cl.cpp" // pulls in parse_device / validate_metric_for_device
#include "../../dtwc/algorithms/tadpole.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>
#include <string>
#include <variant>

// ---------------------------------------------------------------------------
// parse_device
// ---------------------------------------------------------------------------

TEST_CASE("parse_device accepts cpu", "[cli][device]")
{
  const DeviceSpec d = parse_device("cpu");
  REQUIRE(d.valid);
  REQUIRE_FALSE(d.is_cuda);
}

TEST_CASE("parse_device accepts cuda and cuda:N", "[cli][device]")
{
  const DeviceSpec bare = parse_device("cuda");
  REQUIRE(bare.valid);
  REQUIRE(bare.is_cuda);
  REQUIRE(bare.cuda_id == 0);

  const DeviceSpec three = parse_device("cuda:3");
  REQUIRE(three.valid);
  REQUIRE(three.is_cuda);
  REQUIRE(three.cuda_id == 3);
}

// Bug #3: case-sensitive rfind meant "CUDA:0" silently fell back to CPU.
TEST_CASE("parse_device is case-insensitive (CUDA:0 is a CUDA request)", "[cli][device]")
{
  const DeviceSpec upper = parse_device("CUDA:0");
  REQUIRE(upper.valid);
  REQUIRE(upper.is_cuda); // was FALSE (silent CPU) before the fix
  REQUIRE(upper.cuda_id == 0);

  REQUIRE(parse_device("Cuda").is_cuda);
  REQUIRE(parse_device("CUDA:2").cuda_id == 2);
}

// Bug #2: std::stoi("abc") threw uncaught -> std::terminate.
TEST_CASE("parse_device rejects a non-numeric cuda id without crashing", "[cli][device]")
{
  const DeviceSpec bad = parse_device("cuda:abc");
  REQUIRE_FALSE(bad.valid);
  REQUIRE_FALSE(bad.error.empty());
  REQUIRE_FALSE(bad.is_cuda);

  // "cuda:" with an empty id must also be rejected (not accepted as cuda:0).
  REQUIRE_FALSE(parse_device("cuda:").valid);
  // A negative / signed id is non-numeric under our digit check -> rejected.
  REQUIRE_FALSE(parse_device("cuda:-1").valid);
}

// Bug #3: unknown device silently fell back to CPU (no validation).
TEST_CASE("parse_device rejects unknown devices (no silent CPU fallback)", "[cli][device]")
{
  const DeviceSpec gpu = parse_device("gpu"); // CLI surface is cpu/cuda only
  REQUIRE_FALSE(gpu.valid);
  REQUIRE_FALSE(gpu.error.empty());

  REQUIRE_FALSE(parse_device("foo").valid);
  REQUIRE_FALSE(parse_device("").valid);
}

// ---------------------------------------------------------------------------
// validate_metric_for_device
// ---------------------------------------------------------------------------

// Bug #1: --metric was a no-op on the CPU path (silently computed L1).
TEST_CASE("validate_metric_for_device rejects a non-L1 metric on the CPU path", "[cli][metric]")
{
  // Non-L1 on CPU is unsupported and must error (was silently ignored before).
  REQUIRE_FALSE(validate_metric_for_device("squared_euclidean", /*is_cuda=*/false).empty());

  // The default L1 metric is fine on CPU.
  REQUIRE(validate_metric_for_device("l1", /*is_cuda=*/false).empty());

  // Any metric is fine on the CUDA path (that path consumes it).
  REQUIRE(validate_metric_for_device("squared_euclidean", /*is_cuda=*/true).empty());
  REQUIRE(validate_metric_for_device("l1", /*is_cuda=*/true).empty());
}

TEST_CASE("CLI distance config rejects YAML transformer bypasses before work",
          "[cli][config][distance]")
{
  CHECK(validate_cli_distance_configuration(
          "standard", "l1", "dependent", "error", false).empty());
  CHECK(validate_cli_distance_configuration(
          "standard", "squared_euclidean", "dependent", "error", true).empty());

  CHECK(validate_cli_distance_configuration(
          "unknown", "l1", "dependent", "error", false)
        == "unsupported --variant 'unknown'");
  CHECK(validate_cli_distance_configuration(
          "standard", "unknown", "dependent", "error", true)
        == "unsupported --metric 'unknown'");
  CHECK(validate_cli_distance_configuration(
          "standard", "l1", "unknown", "error", false)
        == "unsupported --mv-mode 'unknown'");
  CHECK(validate_cli_distance_configuration(
          "standard", "l1", "dependent", "unknown", false)
        == "unsupported --missing-strategy 'unknown'");
  CHECK(validate_cli_distance_configuration(
          "twe", "l1", "dependent", "zero_cost", false)
        == "non-standard --variant cannot be combined with a non-error "
           "--missing-strategy");
  CHECK(validate_cli_distance_configuration(
          "twe", "l1", "independent", "error", false)
        == "--mv-mode independent requires --variant standard and "
           "--missing-strategy error");
  CHECK(validate_cli_distance_configuration(
          "standard", "squared_euclidean", "dependent", "error", false)
        == "metric 'squared_euclidean' is unsupported on the cpu path "
           "(only 'l1' is implemented on CPU; use --device cuda for "
           "'squared_euclidean').");
  CHECK(validate_cli_distance_configuration(
          "twe", "l1", "dependent", "error", true)
        == "--device cuda supports --variant standard only");
}

// ---------------------------------------------------------------------------
// CLI distance-matrix storage routing (Task 8.1 M11)
// ---------------------------------------------------------------------------

namespace {

struct ScratchDirectory
{
  std::filesystem::path path;

  explicit ScratchDirectory(std::string_view name)
    : path(std::filesystem::temp_directory_path() / std::string(name))
  {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
    std::filesystem::create_directories(path);
  }

  ~ScratchDirectory()
  {
    std::error_code ec;
    std::filesystem::remove_all(path, ec);
  }
};

dtwc::Problem tiny_storage_problem()
{
  dtwc::Problem prob{"cli_storage"};
  std::vector<std::vector<dtwc::data_t>> series{
    { 0.0, 1.0, 2.0 },
    { 0.0, 2.0, 3.0 },
    { 1.0, 2.0, 4.0 },
  };
  std::vector<std::string> names{ "a", "b", "c" };
  prob.set_data(dtwc::Data(std::move(series), std::move(names)));
  return prob;
}

dtwc::Problem seed_sensitive_pam_problem()
{
  const std::vector<dtwc::data_t> base{0.0, 0.01, -0.02, 0.03};
  std::vector<std::vector<dtwc::data_t>> series;
  std::vector<std::string> names;
  for (int offset = 0; offset < 8; ++offset) {
    auto waveform = base;
    for (auto &value : waveform) value += static_cast<dtwc::data_t>(offset);
    series.push_back(std::move(waveform));
    names.push_back(std::to_string(offset));
  }
  dtwc::Problem prob{"cli_seed_fixture"};
  prob.set_data(dtwc::Data(std::move(series), std::move(names)));
  return prob;
}

} // namespace

TEST_CASE("CLI PAM honors default seed 42 and explicit seed override 29",
          "[cli][seed][pam]")
{
  REQUIRE(dtwc::settings::DEFAULT_RANDOM_SEED == 42);

  auto default_problem = seed_sensitive_pam_problem();
  const auto default_result = run_cli_pam(default_problem, 3, 100);
  CHECK(default_result.medoid_indices == std::vector<int>{6, 2, 5});
  CHECK(default_result.labels == std::vector<int>{1, 1, 1, 1, 2, 2, 0, 0});
  CHECK(default_result.total_cost == 24.0);

  auto override_problem = seed_sensitive_pam_problem();
  override_problem.set_random_seed(/* --seed */ 29);
  const auto override_result = run_cli_pam(override_problem, 3, 100);
  CHECK(override_result.medoid_indices == std::vector<int>{4, 1, 7});
  CHECK(override_result.labels == std::vector<int>{1, 1, 1, 0, 0, 0, 2, 2});
  CHECK(override_result.total_cost == 20.0);
}

TEST_CASE("CLI PAM n_init retains the best deterministic restart",
          "[cli][seed][pam][n_init]")
{
  constexpr auto base_seed = dtwc::settings::DEFAULT_RANDOM_SEED;

  auto one_problem = seed_sensitive_pam_problem();
  const auto one = run_cli_pam(one_problem, 3, 100);
  CHECK(one.total_cost == 24.0);

  auto improving_problem = seed_sensitive_pam_problem();
  improving_problem.set_random_seed(base_seed + 1);
  const auto improving = run_cli_pam(improving_problem, 3, 100);
  CHECK(improving.total_cost == 20.0);

  // --n-init=2 must try base_seed and base_seed+1, then retain the lower-cost
  // result rather than merely returning the final or the first restart.
  for (int repetition = 0; repetition < 3; ++repetition) {
    auto problem = seed_sensitive_pam_problem();
    problem.set_n_repetitions(2); // mirrors main()'s parsed CLI state
    const auto result = run_cli_pam(problem, 3, 100);

    CAPTURE(repetition);
    CHECK(result.medoid_indices == improving.medoid_indices);
    CHECK(result.labels == improving.labels);
    CHECK(result.total_cost == improving.total_cost);
    CHECK(result.iterations == improving.iterations);
    CHECK(result.converged == improving.converged);
  }

  auto invalid_problem = seed_sensitive_pam_problem();
  invalid_problem.set_n_repetitions(0);
  REQUIRE_THROWS_WITH(
    run_cli_pam(invalid_problem, 3, 100),
    "run_cli_pam: n_init must be at least 1.");

  auto overflow_problem = seed_sensitive_pam_problem();
  overflow_problem.set_n_repetitions(2);
  overflow_problem.set_random_seed(std::numeric_limits<std::uint64_t>::max());
  REQUIRE_THROWS_WITH(
    run_cli_pam(overflow_problem, 3, 100),
    "run_cli_pam: random_seed + n_init - 1 overflows uint64.");
}

TEST_CASE("CLI TADPole threshold uses mmap or fails before dense allocation",
          "[cli][storage][mmap][tadpole]")
{
  ScratchDirectory scratch{"dtwc_cli_tadpole_storage"};
  auto prob = tiny_storage_problem();
  const auto cache = scratch.path / "tadpole_distmat.cache";

#ifdef DTWC_HAS_MMAP
  const auto selected = configure_cli_distance_storage(
    prob, "tadpole", /*mmap_threshold=*/0, cache);

  REQUIRE(selected == cache);
  REQUIRE(std::holds_alternative<dtwc::core::MmapDistanceMatrix>(prob.distance_matrix()));
  REQUIRE(std::filesystem::exists(cache));

  // Exercise the real TADPole exact/fallback schedule. Its lazy cache writes
  // must populate the mapped matrix rather than replacing it with dense storage.
  const auto result = dtwc::algorithms::tadpole(
    prob, /*n_clusters=*/2, /*dc=*/3.0, /*prune=*/false);
  REQUIRE(result.labels.size() == prob.size());
  REQUIRE(std::holds_alternative<dtwc::core::MmapDistanceMatrix>(prob.distance_matrix()));
  REQUIRE(std::get<dtwc::core::MmapDistanceMatrix>(prob.distance_matrix())
            .is_computed(0, 1));
#else
  REQUIRE_THROWS_WITH(
    configure_cli_distance_storage(prob, "tadpole", /*mmap_threshold=*/0, cache),
    Catch::Matchers::ContainsSubstring("requires memory-mapped distance storage")
      && Catch::Matchers::ContainsSubstring("DTWC_ENABLE_LLFIO=ON"));

  // The capability error must happen before dist_by_ind can lazily allocate N^2
  // packed doubles on the heap.
  REQUIRE(std::holds_alternative<dtwc::core::DenseDistanceMatrix>(prob.distance_matrix()));
  REQUIRE(prob.dense_distance_matrix().size() == 0);
  REQUIRE_FALSE(std::filesystem::exists(cache));
#endif
}

TEST_CASE("CLI OneBatch keeps its own O(Nm) storage when mmap threshold fires",
          "[cli][storage][onebatch]")
{
  ScratchDirectory scratch{"dtwc_cli_onebatch_storage"};
  auto prob = tiny_storage_problem();
  const auto cache = scratch.path / "onebatch_distmat.cache";

  const auto selected = configure_cli_distance_storage(
    prob, "onebatch", /*mmap_threshold=*/0, cache);

  REQUIRE_FALSE(selected.has_value());
  REQUIRE(std::holds_alternative<dtwc::core::DenseDistanceMatrix>(prob.distance_matrix()));
  REQUIRE(prob.dense_distance_matrix().size() == 0);
  REQUIRE_FALSE(std::filesystem::exists(cache));
}

TEST_CASE("CLI mmap storage binds the selected pointwise metric",
          "[cli][storage][mmap][fingerprint]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in (DTWC_ENABLE_LLFIO=OFF)");
#else
  ScratchDirectory scratch{"dtwc_cli_metric_fingerprint"};
  const auto cache = scratch.path / "metric_distmat.cache";

  {
    auto squared = tiny_storage_problem();
    REQUIRE(configure_cli_distance_storage(
              squared, "pam", /*mmap_threshold=*/0, cache,
              dtwc::core::MetricType::SquaredL2) == cache);
    // Simulate an externally produced GPU entry; the cache identity, not CPU
    // dispatch, is what this CLI helper owns.
    std::get<dtwc::core::MmapDistanceMatrix>(squared.distance_matrix())
      .set(0, 1, 7.0);
  }

  auto l1 = tiny_storage_problem();
  REQUIRE_THROWS_WITH(
    configure_cli_distance_storage(
      l1, "pam", /*mmap_threshold=*/0, cache, dtwc::core::MetricType::L1),
    Catch::Matchers::ContainsSubstring("fingerprint mismatch"));
#endif
}

TEST_CASE("CLI rejects legacy CSV checkpoint plus mmap before either is opened",
          "[cli][storage][mmap][checkpoint]")
{
  ScratchDirectory scratch{"dtwc_cli_checkpoint_mmap"};
  auto prob = tiny_storage_problem();
  const auto cache = scratch.path / "checkpoint_distmat.cache";

  REQUIRE_THROWS_WITH(
    configure_cli_distance_storage(
      prob, "pam", /*mmap_threshold=*/0, cache, dtwc::core::MetricType::L1,
      /*legacy_checkpoint_requested=*/true),
    Catch::Matchers::ContainsSubstring("cannot be combined")
      && Catch::Matchers::ContainsSubstring("resumes automatically"));
  REQUIRE_FALSE(std::filesystem::exists(cache));
  REQUIRE(std::holds_alternative<dtwc::core::DenseDistanceMatrix>(
    prob.distance_matrix()));
}

TEST_CASE("CLI rejects legacy precomputed CSV plus mmap before false success",
          "[cli][storage][mmap][dist-matrix]")
{
  ScratchDirectory scratch{"dtwc_cli_precomputed_mmap"};
  auto prob = tiny_storage_problem();
  const auto cache = scratch.path / "precomputed_distmat.cache";

  REQUIRE_THROWS_WITH(
    configure_cli_distance_storage(
      prob, "pam", /*mmap_threshold=*/0, cache, dtwc::core::MetricType::L1,
      /*legacy_checkpoint_requested=*/false,
      /*legacy_distance_matrix_requested=*/true),
    Catch::Matchers::ContainsSubstring("--dist-matrix")
      && Catch::Matchers::ContainsSubstring("cannot be combined")
      && Catch::Matchers::ContainsSubstring("importing"));
  REQUIRE_FALSE(std::filesystem::exists(cache));
  REQUIRE(std::holds_alternative<dtwc::core::DenseDistanceMatrix>(
    prob.distance_matrix()));
}

// ---------------------------------------------------------------------------
// CLI / TOML flag deprecation registry (Task 2.3, api-contract-2.0.md §4/§7)
//
// These pin the SSOT table (cli_renames) and the warning formatter that
// dtwc_cl main() iterates in its post-parse handler to (a) accept an old flag /
// old TOML-or-YAML key, (b) emit exactly one stderr warning per use, and (c)
// yield precedence to the canonical spelling. The unit test cannot link CLI11
// (DTWC_CL_NO_MAIN excludes it), so it drives the CLI11-free mechanism directly;
// the live CLI11 routing + stderr emission is exercised end-to-end against the
// built dtwc_cl binary (Task 2.3 verification).
// ---------------------------------------------------------------------------

// "new canonical flag works": the canonical spellings are NOT flagged as
// deprecated, and the two renamed concepts map old -> new correctly.
TEST_CASE("cli_renames maps deprecated flags to contract-canonical names", "[cli][deprecation]")
{
  // Deprecated CLI flag spellings resolve to the 2.0 canonical flag.
  REQUIRE(canonical_flag_for("--clusters") == "--n-clusters"); // §1.5/§2.1 n_clusters
  REQUIRE(canonical_flag_for("--restart") == "--resume");      // §2.7

  // TOML/YAML "old key acceptance": the bare-key form (no leading dashes, i.e.
  // how it appears in a --config TOML or --yaml-config file) resolves the same.
  REQUIRE(canonical_flag_for("clusters") == "--n-clusters");
  REQUIRE(canonical_flag_for("restart") == "--resume");

  // Canonical / unknown spellings are NOT deprecated (no false warning).
  REQUIRE(canonical_flag_for("--n-clusters").empty());
  REQUIRE(canonical_flag_for("n-clusters").empty());
  REQUIRE(canonical_flag_for("--resume").empty());
  REQUIRE(canonical_flag_for("--method").empty());
  REQUIRE(canonical_flag_for("--skip-cols").empty()); // caller flag stays canonical
}

// "old flag works AND emits the deprecation warning" — pins the exact one-line
// stderr message the post-parse handler prints (the warning mechanism).
TEST_CASE("format_deprecation_warning is the exact one-line stderr message", "[cli][deprecation]")
{
  REQUIRE(format_deprecation_warning("--clusters", "--n-clusters")
          == "[dtwc] warning: '--clusters' is deprecated, use '--n-clusters' instead");
  REQUIRE(format_deprecation_warning("--restart", "--resume")
          == "[dtwc] warning: '--restart' is deprecated, use '--resume' instead");
}

// The rename table is the de-facto CLI API surface: complete + internally
// consistent (both spellings are long flags, they differ, and the lookup
// round-trips for every entry).
TEST_CASE("cli_renames table is complete and internally consistent", "[cli][deprecation]")
{
  const auto &t = cli_renames();
  REQUIRE(t.size() == 2);
  for (const auto &r : t) {
    REQUIRE(r.old_flag.rfind("--", 0) == 0);
    REQUIRE(r.new_flag.rfind("--", 0) == 0);
    REQUIRE(r.old_flag != r.new_flag);
    REQUIRE(canonical_flag_for(r.old_flag) == r.new_flag);
  }
}
