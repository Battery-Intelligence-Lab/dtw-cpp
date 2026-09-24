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
 *    `InvalidInput`, while `.dtws` without llfio is `IOError`. A format this build
 *    cannot read is `IOError`.
 *
 * The CLI translation unit is compiled with DTWC_CL_NO_MAIN, as in
 * unit_test_cli_args.cpp, so its helpers are the production code.
 *
 * @date 24 Sep 2026
 */

#define DTWC_CL_NO_MAIN
#include "../../dtwc/dtwc_cl.cpp" // require_input_format_is_built

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <istream>
#include <streambuf>
#include <string>
#include <system_error>

using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::MessageMatches;
using Catch::Matchers::StartsWith;

namespace {

struct Scratch
{
  fs::path path = fs::temp_directory_path()
    / ("dtwc_io_error_contract_" + std::to_string(reinterpret_cast<std::uintptr_t>(this)));
  Scratch() { fs::remove_all(path); fs::create_directories(path); }
  ~Scratch() { std::error_code ec; fs::remove_all(path, ec); }
};

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
  const Scratch dir;
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
  CHECK_NOTHROW(require_input_format_is_built(false, false));
#ifdef DTWC_HAS_PARQUET
  CHECK_NOTHROW(require_input_format_is_built(true, false));
#else
  CHECK_THROWS_MATCHES(require_input_format_is_built(true, false), dtwc::IOError,
                       MessageMatches(ContainsSubstring("Parquet input")
                                      && ContainsSubstring("-DDTWC_ENABLE_ARROW=ON")));
#endif
#ifdef DTWC_HAS_ARROW
  CHECK_NOTHROW(require_input_format_is_built(false, true));
#else
  CHECK_THROWS_MATCHES(require_input_format_is_built(false, true), dtwc::IOError,
                       MessageMatches(ContainsSubstring("Arrow IPC input")
                                      && ContainsSubstring("-DDTWC_ENABLE_ARROW=ON")));
#endif
}
