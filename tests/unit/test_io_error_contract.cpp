/**
 * @file test_io_error_contract.cpp
 * @brief api-contract-2.0.md §5's file rule at three seams where it had slipped.
 *
 * @details
 *  - Tier-1 load: GT-4 made every reader error an `IOError`, and `load` added its
 *    `load: failed to read '<path>': ` prefix only to other exceptions, so the
 *    prefix vanished. The type stays `IOError`; the prefix is back.
 *  - `ignoreBOM`: a partial byte-order mark that cannot be handed back named no file.
 *  - `dtwc_cl`: Parquet / Arrow IPC input on a build without Arrow was
 *    `InvalidInput`. A format this build cannot read is `IOError`.
 *
 * dtwc_cl's pipeline is dtwc::run (IF-2 S3), which the third case drives; both
 * it and Python's load() read through dtwc::read_data, which the last case drives.
 *
 * @date 24 Sep 2026
 */

#include "cli/run.hpp"
#include "dtwc.hpp"

#include "../support/scratch_directory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <istream>
#include <streambuf>
#include <string>
#include <vector>
#include <system_error>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::MessageMatches;
using Catch::Matchers::StartsWith;
using dtwc::test_support::ScratchDirectory;
namespace fs = std::filesystem;

namespace {

/// Holds one byte at a time, as a pipe's reader may: unget() cannot hand back
/// a byte read before the current one.
class OneByteAtATime : public std::streambuf
{
  std::string bytes_;
  std::size_t next_ = 0;
  char current_ = 0;

protected:
  int_type underflow() override
  {
    if (next_ == bytes_.size()) return traits_type::eof();
    current_ = bytes_[next_++];
    setg(&current_, &current_, &current_ + 1);
    return traits_type::to_int_type(current_);
  }

public:
  explicit OneByteAtATime(std::string bytes) : bytes_(std::move(bytes)) {}
};

} // namespace

TEST_CASE("Tier-1 load: a reader IOError keeps its type and names the load",
          "[io][error][load]")
{
  const ScratchDirectory dir{ "io_error_contract" };
  const fs::path missing = dir.path / "missing.csv";
  const fs::path bad = dir.path / "bad.csv";
  std::ofstream(bad, std::ios::binary) << "1,2,x\n4,5,6\n";

  const std::string prefix = "load: failed to read '";
  CHECK_THROWS_MATCHES(dtwc::cluster(dtwc::load(missing), 1), dtwc::IOError,
                       MessageMatches(StartsWith(prefix + missing.string() + "': ")
                                      && ContainsSubstring("could not be opened")));
  CHECK_THROWS_MATCHES(dtwc::cluster(dtwc::load(bad), 1), dtwc::IOError,
                       MessageMatches(StartsWith(prefix + bad.string() + "': ")
                                      && ContainsSubstring("invalid numeric field 'x'")));
}

TEST_CASE("ignoreBOM: a partial mark that cannot be handed back names the file",
          "[io][error][bom]")
{
  OneByteAtATime bytes("\xEF\xBB" "x\n");
  std::istream in(&bytes);
  CHECK_THROWS_MATCHES(dtwc::ignoreBOM(in, "staged/series.csv"), dtwc::IOError,
                       MessageMatches(ContainsSubstring("'staged/series.csv'")
                                      && ContainsSubstring("partial UTF-8 byte-order mark")));
}

TEST_CASE("dtwc_cl: an input format this build cannot read is IOError",
          "[cli][io][error]")
{
  // The format is judged by the name alone, before any file is opened: none exists.
  const auto run_on = [](const char *input) {
    dtwc::Config config;
    config.input = input;
    config.k = 2;
    config.output.clear();
    (void)dtwc::run(config);
  };
  // Python's load() calls read_data directly, without run().
  const auto read_on = [](const char *input) { (void)dtwc::read_data(input); };
  const auto check = [](const auto &reach) {
    // Control: a format every build reads fails in its reader, naming the file.
    CHECK_THROWS_MATCHES(reach("missing.csv"), dtwc::IOError,
                         MessageMatches(StartsWith("load: failed to read 'missing.csv': ")));
#ifndef DTWC_HAS_PARQUET
    CHECK_THROWS_MATCHES(reach("missing.parquet"), dtwc::IOError,
                         MessageMatches(ContainsSubstring("Parquet input")
                                        && ContainsSubstring("-DDTWC_ENABLE_ARROW=ON")));
#endif
#ifndef DTWC_HAS_ARROW
    CHECK_THROWS_MATCHES(reach("missing.arrow"), dtwc::IOError,
                         MessageMatches(ContainsSubstring("Arrow IPC input")
                                        && ContainsSubstring("-DDTWC_ENABLE_ARROW=ON")));
#endif
  };
  check(run_on);
  check(read_on);
}

TEST_CASE("read_data: a reader option the format cannot honour is InvalidInput",
          "[io][error]")
{
  // Refused before any file is opened (none exists): ignoring it would be silent.
  CHECK_THROWS_MATCHES(dtwc::read_data("missing.csv", 0, 0, '\0', "v"), dtwc::InvalidInput,
                       MessageMatches(ContainsSubstring("--column selects a Parquet column")));
#ifdef DTWC_HAS_PARQUET
  CHECK_THROWS_MATCHES(dtwc::read_data("missing.parquet", 0, 1), dtwc::InvalidInput,
                       MessageMatches(ContainsSubstring("skip_rows")));
#endif
}

TEST_CASE("read_data: a folder is text whatever its name", "[io][load]")
{
  // Only a file's extension names its format: a folder named runs.arrow that
  // holds CSV files is not an Arrow file to open (or to refuse without Arrow).
  const ScratchDirectory dir{ "read_data_folder" };
  const fs::path folder = dir.path / "runs.arrow";
  fs::create_directories(folder);
  std::ofstream(folder / "a.csv", std::ios::binary) << "1\n2\n";
  const auto data = dtwc::read_data(folder);
  REQUIRE(data.size() == 1);
  CHECK(data.p_names.front() == "a");
  CHECK(data.p_vec.front() == std::vector<double>{ 1.0, 2.0 });
}
