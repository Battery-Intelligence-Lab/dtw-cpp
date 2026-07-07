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

#include <catch2/catch_test_macros.hpp>

#include <string>

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
