/**
 * @file unit_test_cli_args.cpp
 * @brief dtwc_cl's run-time contracts, pinned where they now live: cli::bind()
 *        (through parse_config), dtwc::run() and its Parquet metadata planner.
 *
 * @details This file used to compile dtwc_cl.cpp with DTWC_CL_NO_MAIN to
 * reach its file-local helpers. Those helpers are gone: the device grammar is
 * detail::parse_device, the selectors are types, and the pipeline is run(). The
 * contracts they carried are asserted here through the production entry points;
 * the device matrix is test_run_resolution.cpp's, and the streamed Parquet
 * names are checked by the real-binary tests of Arrow builds.
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#include "cli/config.hpp"
#include "cli/run.hpp"

#include "../support/scratch_directory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <filesystem>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace fs = std::filesystem;
using Catch::Matchers::ContainsSubstring;
using dtwc::Method;
using dtwc::Device;
using dtwc::detail::ParquetLayout;
using dtwc::detail::plan_parquet_load;
using dtwc::test_support::ScratchDirectory;

namespace {

std::size_t ram_limit(const std::string &text) { return dtwc::parse_config({ { "ram_limit", text } }).ram_limit; }

/// A run that writes nothing unless a test names an output directory.
dtwc::Config quiet_config(int k, Method method)
{
  dtwc::Config config;
  config.k = k;
  config.method = method;
  config.output.clear();
  return config;
}

dtwc::Data tiny_series()
{
  return dtwc::Data(std::vector<std::vector<double>>{ { 0.0, 1.0, 2.0 }, { 0.0, 2.0, 3.0 }, { 1.0, 2.0, 4.0 } },
                    std::vector<std::string>{ "a", "b", "c" });
}

/// Eight translated waveforms: PAM's BUILD seed changes its local optimum.
dtwc::Data seed_sensitive_series()
{
  const std::vector<double> base{ 0.0, 0.01, -0.02, 0.03 };
  std::vector<std::vector<double>> series;
  std::vector<std::string> names;
  for (int offset = 0; offset < 8; ++offset) {
    auto waveform = base;
    for (auto &value : waveform) value += static_cast<double>(offset);
    series.push_back(std::move(waveform));
    names.push_back(std::to_string(offset));
  }
  return dtwc::Data(std::move(series), std::move(names));
}

} // namespace

// ---------------------------------------------------------------------------
// --ram-limit, read by cli::bind()
// ---------------------------------------------------------------------------

TEST_CASE("--ram-limit reads a size and rejects anything else", "[cli][parquet][ram]")
{
  CHECK(ram_limit("") == 0);
  CHECK(ram_limit("0") == 0);
  CHECK(ram_limit("1") == 1);
  CHECK(ram_limit("2K") == 2ULL * 1024ULL);
  CHECK(ram_limit("1.5MiB") == 1572864ULL);
  CHECK(ram_limit("3gb") == 3ULL * 1024ULL * 1024ULL * 1024ULL);
  // A plain byte count is exact past 2^53, up to size_t's maximum.
  if constexpr (std::numeric_limits<std::size_t>::digits > 53)
    CHECK(ram_limit("9007199254740993") == 9007199254740993ULL);
  CHECK(ram_limit(std::to_string(std::numeric_limits<std::size_t>::max())) == std::numeric_limits<std::size_t>::max());
  // A fraction of a byte rounds up, so a nonzero value never turns the cap off.
  CHECK(ram_limit("0.1B") == 1);
  CHECK(ram_limit("1.1K") == 1127); // 1126.4 bytes

  for (const std::string malformed : { "-1G", "nan", "inf", "1GBjunk", "G", "1Q", "+1G", " 1G",
                                       "999999999999999999999999T" }) {
    CAPTURE(malformed);
    CHECK_THROWS_AS(ram_limit(malformed), dtwc::InvalidInput);
  }
}

// A cap on an input no reader can apply it to must fail: the CLI once accepted
// --ram-limit for CSV / Arrow, printed the cap and loaded everything.
TEST_CASE("--ram-limit is rejected where no reader can honour it", "[cli][parquet][ram]")
{
  auto config = quiet_config(2, Method::PAM);
  config.ram_limit = 1ULL << 30;
  CHECK_THROWS_MATCHES(dtwc::run(config, tiny_series()), dtwc::InvalidInput,
                       Catch::Matchers::MessageMatches(ContainsSubstring("cannot be honoured for this input")));
  config.input = "never_read.csv"; // refused before the file is opened
  CHECK_THROWS_WITH(dtwc::run(config), ContainsSubstring("cannot be honoured for this input"));

  config.ram_limit = 0;
  config.input.clear();
  CHECK(dtwc::run(config, tiny_series()).labels().size() == 3);
}

// ---------------------------------------------------------------------------
// Parquet metadata planning: decided before any payload is read
// ---------------------------------------------------------------------------

TEST_CASE("Parquet plan selects streaming before payload materialization", "[cli][parquet][ram][streaming]")
{
  const auto plan = plan_parquet_load(Method::CLARA, Device::CPU, 6001, /*bytes=*/4097, /*cap=*/4096,
                                      ParquetLayout::ListColumn);
  CHECK(plan.method == Method::CLARA);
  CHECK(plan.stream_payload);

  const auto boundary =
    plan_parquet_load(Method::CLARA, Device::CPU, 6001, 4096, 4096, ParquetLayout::ListColumn);
  CHECK_FALSE(boundary.stream_payload);
}

TEST_CASE("Parquet plan resolves auto from metadata, for the device", "[cli][parquet][ram][auto]")
{
  const auto pam = plan_parquet_load(Method::Auto, Device::CPU, 5000, 100, 1000, ParquetLayout::ListColumn);
  CHECK(pam.method == Method::PAM);
  CHECK_FALSE(pam.stream_payload);

  const auto clara = plan_parquet_load(Method::Auto, Device::CPU, 5001, 1001, 1000, ParquetLayout::ListColumn);
  CHECK(clara.method == Method::CLARA);
  CHECK(clara.stream_payload);

  // On a GPU `auto` is pam, whose matrix the GPU fills, at any N: over the cap
  // that is the loud "cannot stream", never a CPU-only CLARA.
  CHECK(plan_parquet_load(Method::Auto, Device::GPU, 5001, 100, 1000, ParquetLayout::ListColumn).method
        == Method::PAM);
  CHECK_THROWS_WITH(plan_parquet_load(Method::Auto, Device::GPU, 5001, 1001, 1000, ParquetLayout::ListColumn),
                    ContainsSubstring("method 'pam' cannot stream"));
}

TEST_CASE("Parquet RAM limit rejects every route that cannot honor it", "[cli][parquet][ram][loudness]")
{
  CHECK_THROWS_WITH(plan_parquet_load(Method::PAM, Device::CPU, 20, 1001, 1000, ParquetLayout::ListColumn),
                    ContainsSubstring("method 'pam' cannot stream"));
  CHECK_THROWS_WITH(plan_parquet_load(Method::CLARA, Device::CPU, 1, 1001, 1000, ParquetLayout::ScalarColumn),
                    ContainsSubstring("list-per-row"));
  CHECK_THROWS_WITH(plan_parquet_load(Method::CLARA, Device::CPU, 100, 1001, 1000, ParquetLayout::Directory),
                    ContainsSubstring("single Parquet file"));
  CHECK_THROWS_AS(plan_parquet_load(Method::CLARA, Device::CPU, 0, 1001, 1000, ParquetLayout::ListColumn),
                  dtwc::InvalidInput);
}

TEST_CASE("Parquet plan treats a zero RAM limit as uncapped", "[cli][parquet][ram][streaming]")
{
  // `ram_limit == 0` means "no cap", never "a cap of zero bytes"; it returns
  // before the method and layout rejections.
  constexpr auto huge = std::numeric_limits<std::size_t>::max();
  CHECK_FALSE(plan_parquet_load(Method::CLARA, Device::CPU, 6001, huge, 0, ParquetLayout::ListColumn)
                .stream_payload);
  CHECK_NOTHROW(plan_parquet_load(Method::PAM, Device::CPU, 20, huge, 0, ParquetLayout::ScalarColumn));
  CHECK_NOTHROW(plan_parquet_load(Method::CLARA, Device::CPU, 20, huge, 0, ParquetLayout::Directory));
}

// ---------------------------------------------------------------------------
// PAM seeds and restarts (--seed, --n-init)
// ---------------------------------------------------------------------------

TEST_CASE("run's PAM honors default seed 42 and explicit seed override 29", "[cli][seed][pam]")
{
  REQUIRE(dtwc::Config{}.seed == 42);
  const auto default_result = dtwc::run(quiet_config(3, Method::PAM), seed_sensitive_series());
  CHECK(default_result.medoids() == std::vector<dtwc::index_t>{ 6, 2, 5 });
  CHECK(default_result.labels() == std::vector<dtwc::index_t>{ 1, 1, 1, 1, 2, 2, 0, 0 });
  CHECK(default_result.cost() == 24.0);

  auto seed_29 = quiet_config(3, Method::PAM);
  seed_29.seed = 29;
  const auto override_result = dtwc::run(seed_29, seed_sensitive_series());
  CHECK(override_result.medoids() == std::vector<dtwc::index_t>{ 4, 1, 7 });
  CHECK(override_result.labels() == std::vector<dtwc::index_t>{ 1, 1, 1, 0, 0, 0, 2, 2 });
  CHECK(override_result.cost() == 20.0);
}

TEST_CASE("run's PAM n_init retains the best deterministic restart", "[cli][seed][pam][n_init]")
{
  auto improving_config = quiet_config(3, Method::PAM);
  improving_config.seed = 43;
  const auto improving = dtwc::run(improving_config, seed_sensitive_series());
  CHECK(improving.cost() == 20.0);

  // --n-init 2 tries seeds 42 and 43 and keeps the lower cost, not the first or
  // the last restart (the Problem holds the last one's labels until run() sets
  // the kept result).
  auto config = quiet_config(3, Method::PAM);
  config.n_init = 2;
  for (int repetition = 0; repetition < 3; ++repetition) {
    const auto result = dtwc::run(config, seed_sensitive_series());
    CAPTURE(repetition);
    CHECK(result.medoids() == improving.medoids());
    CHECK(result.labels() == improving.labels());
    CHECK(result.cost() == improving.cost());
    CHECK(result.iterations() == improving.iterations());
    CHECK(result.converged() == improving.converged());
  }

  config.n_init = 0;
  CHECK_THROWS_WITH(dtwc::run(config, seed_sensitive_series()), "--n-init must be a positive integer");
}

// ---------------------------------------------------------------------------
// Distance-matrix storage routing: --mmap-threshold 0 always maps
// ---------------------------------------------------------------------------

namespace {

dtwc::Config mapped_config(Method method, const ScratchDirectory &scratch, const std::string &name)
{
  auto config = quiet_config(2, method);
  config.output = scratch.path.string();
  config.name = name;
  config.mmap_threshold = 0;
  return config;
}

bool cache_exists(const ScratchDirectory &scratch, const std::string &name)
{
  return fs::exists(scratch.path / (name + ".dtwm"));
}

} // namespace

TEST_CASE("run's TADPole threshold uses mmap or fails before dense allocation", "[cli][storage][mmap][tadpole]")
{
  const ScratchDirectory scratch{ "cli_tadpole_storage" };
  auto config = mapped_config(Method::TADPole, scratch, "tadpole");
  config.tadpole_dc = 3.0;
#ifdef DTWC_HAS_MMAP
  // TADPole is not exempt: it reads the matrix when a complete one is there.
  const auto result = dtwc::run(config, tiny_series());
  CHECK(result.labels().size() == 3);
  CHECK(cache_exists(scratch, "tadpole"));
#else
  CHECK_THROWS_WITH(dtwc::run(config, tiny_series()),
                    ContainsSubstring("requires memory-mapped distance storage")
                      && ContainsSubstring("DTWC_ENABLE_LLFIO=ON"));
  CHECK_FALSE(cache_exists(scratch, "tadpole"));
#endif
}

TEST_CASE("run's OneBatch keeps its own O(Nm) storage when the mmap threshold fires", "[cli][storage][onebatch]")
{
  const ScratchDirectory scratch{ "cli_onebatch_storage" };
  CHECK(dtwc::run(mapped_config(Method::OneBatch, scratch, "onebatch"), tiny_series()).labels().size() == 3);
  CHECK_FALSE(cache_exists(scratch, "onebatch"));
}

TEST_CASE("run's non-full FastCLARA does not open an unused parent matrix", "[cli][storage][clara]")
{
  const ScratchDirectory scratch{ "cli_clara_storage" };
  auto config = mapped_config(Method::CLARA, scratch, "clara");
  config.sample_size = 2; // < N = 3
  CHECK(dtwc::run(config, tiny_series()).labels().size() == 3);
  CHECK_FALSE(cache_exists(scratch, "clara"));

  config.checkpoint = (scratch.path / "ckpt").string();
  CHECK_THROWS_MATCHES(dtwc::run(config, tiny_series()), dtwc::InvalidInput, // bad input, not a runtime_error
                       Catch::Matchers::MessageMatches(ContainsSubstring("unused O(N^2) state")));
  config.checkpoint.clear();
  config.dist_matrix = (scratch.path / "never_read.csv").string();
  CHECK_THROWS_WITH(dtwc::run(config, tiny_series()), ContainsSubstring("unused O(N^2) state"));
}

TEST_CASE("run's mmap storage binds the pointwise metric", "[cli][storage][mmap][fingerprint]")
{
#ifndef DTWC_HAS_MMAP
  SKIP("mmap support not compiled in (DTWC_ENABLE_LLFIO=OFF)");
#else
  const ScratchDirectory scratch{ "cli_metric_fingerprint" };
  auto config = mapped_config(Method::PAM, scratch, "metric");
  config.metric = dtwc::core::MetricType::SquaredL2;
  CHECK(dtwc::run(config, tiny_series()).labels().size() == 3);
  REQUIRE(cache_exists(scratch, "metric"));

  config.metric = dtwc::core::MetricType::L1; // the same cache file, other distances
  CHECK_THROWS_WITH(dtwc::run(config, tiny_series()), ContainsSubstring("fingerprint mismatch"));
#endif
}

TEST_CASE("run's mmap storage with --checkpoint maps the checkpoint file", "[cli][storage][mmap][checkpoint]")
{
  // One file: the mapped matrix is the checkpoint, <checkpoint>/<name>.dtwm.
  const ScratchDirectory scratch{ "cli_checkpoint_mmap" };
  auto config = mapped_config(Method::PAM, scratch, "checkpoint");
  config.checkpoint = (scratch.path / "ckpt").string();
#ifdef DTWC_HAS_MMAP
  CHECK(dtwc::run(config, tiny_series()).labels().size() == 3);
  CHECK(fs::file_size(scratch.path / "ckpt" / "checkpoint.dtwm") == 48 + 6 * sizeof(double));
#else
  CHECK_THROWS_WITH(dtwc::run(config, tiny_series()), ContainsSubstring("requires memory-mapped distance storage"));
#endif
  CHECK_FALSE(cache_exists(scratch, "checkpoint"));
}

TEST_CASE("run rejects a legacy precomputed CSV plus mmap before false success", "[cli][storage][mmap][dist-matrix]")
{
  const ScratchDirectory scratch{ "cli_precomputed_mmap" };
  auto config = mapped_config(Method::PAM, scratch, "precomputed");
  config.dist_matrix = (scratch.path / "never_read.csv").string();
  CHECK_THROWS_WITH(dtwc::run(config, tiny_series()),
                    ContainsSubstring("--dist-matrix") && ContainsSubstring("cannot be combined")
                      && ContainsSubstring("importing"));
  CHECK_FALSE(cache_exists(scratch, "precomputed"));
}
