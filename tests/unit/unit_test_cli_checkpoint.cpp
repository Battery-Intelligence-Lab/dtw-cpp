/**
 * @file unit_test_cli_checkpoint.cpp
 * @brief Real-binary gate for the dtwc_cl checkpoint flags.
 *
 * @details A unit test on Problem::checkpoint proves the option is honoured,
 * never that the CLI reaches it, so this drives the built dtwc_cl executable:
 *   - `--checkpoint <dir> --checkpoint-interval 2` writes `<dir>/<name>.dtwm`;
 *   - a second identical run resumes from it;
 *   - a run on other data exits 1 naming the mismatch, never recomputing
 *     over the checkpoint;
 *   - above --mmap-threshold the mapped matrix is that same file;
 *   - `--checkpoint-interval` without `--checkpoint` exits 1 with a message.
 *
 * The fixture series are written into a temporary directory so the test does
 * not depend on the size or contents of any tracked dataset.
 *
 * @author Volkan Kumtepeli
 * @date 02 Sep 2026
 */

#include "../support/scratch_directory.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#elif defined(__APPLE__)
#include <cstdint>
#include <mach-o/dyld.h>
#include <sys/wait.h>
#else
#include <sys/wait.h>
#endif

namespace fs = std::filesystem;

namespace {

/// Directory holding this test executable, which is also where dtwc_cl is
/// linked. Asked of the OS rather than of argv[0]: Catch2 strips the directory
/// from argv[0], and the working directory is the source root, not the build.
fs::path executable_directory()
{
#ifdef _WIN32
  std::wstring buffer(4096, L'\0');
  const DWORD length =
    GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
  REQUIRE(length > 0);
  REQUIRE(length < buffer.size());
  return fs::path(buffer.substr(0, length)).parent_path();
#elif defined(__APPLE__)
  std::uint32_t size = 0;
  _NSGetExecutablePath(nullptr, &size); // reports the required size
  std::string buffer(size, '\0');
  REQUIRE(_NSGetExecutablePath(buffer.data(), &size) == 0);
  return fs::canonical(fs::path(buffer.c_str())).parent_path();
#else
  return fs::canonical("/proc/self/exe").parent_path();
#endif
}

/// Path of the dtwc_cl binary built alongside this test.
fs::path cli_executable()
{
  const fs::path directory = executable_directory();
  fs::path candidate = directory / "dtwc_cl";
  if (!fs::exists(candidate)) candidate.replace_extension(".exe");
  INFO("dtwc_cl not found next to the test executable: " << candidate.string());
  REQUIRE(fs::is_regular_file(candidate));
  return candidate;
}

std::string read_text(const fs::path &path)
{
  std::ifstream input(path, std::ios::binary);
  REQUIRE(input.good());
  return std::string((std::istreambuf_iterator<char>(input)),
                     std::istreambuf_iterator<char>());
}

struct CommandResult {
  int exit_code{ 0 };
  std::string out;
  std::string err;
};

/// Run a command, capturing its streams through files: the streams matter as
/// much as the status, and redirection is the one capture route that behaves
/// identically under cmd.exe and a POSIX shell.
CommandResult run(const std::vector<std::string> &argv, const fs::path &scratch)
{
  const fs::path out_path = scratch / "stdout.txt";
  const fs::path err_path = scratch / "stderr.txt";
  std::string command;
  for (const auto &argument : argv) command += '"' + argument + "\" ";
  command += "> \"" + out_path.string() + "\" 2> \"" + err_path.string() + '"';
#ifdef _WIN32
  // cmd.exe /C strips the outer quote pair; the extra pair keeps every quoted
  // argument intact.
  command = '"' + command + '"';
#endif
  const int status = std::system(command.c_str());
#ifdef _WIN32
  const int exit_code = status;
#else
  const int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
  return { exit_code, read_text(out_path), read_text(err_path) };
}

/// Six short integer-valued series, one CSV per series, matching the
/// index-column + header-row layout the CLI reads with --skip-rows/--skip-cols.
/// `shift` changes every value, so two shifts are other data of the same shape.
void write_fixture_series(const fs::path &directory, int shift = 0)
{
  fs::create_directories(directory);
  for (int s = 0; s < 6; ++s) {
    std::ofstream file(directory / ("series_" + std::to_string(s) + ".csv"));
    REQUIRE(file.good());
    file << "t,value\n";
    for (int t = 0; t < 16; ++t)
      file << t << ',' << (t * 7 + s * 13) % 11 + shift << '\n';
  }
}

/// The checkpoint of six series: a 48-byte header and 21 packed doubles.
constexpr std::uintmax_t checkpoint_bytes = 48 + 21 * sizeof(double);

std::vector<std::string> cli_args(const fs::path &data, const fs::path &scratch, const fs::path &checkpoint_dir)
{
  return { cli_executable().string(),
           "-i", data.string(),
           "-k", "2",
           "-o", (scratch / "out").string(),
           "--name", "cli_ckpt",
           "--skip-rows", "1",
           "--skip-cols", "1",
           "--max-iter", "1",
           "--n-init", "1",
           "--checkpoint", checkpoint_dir.string(),
           "--verbose" };
}

} // anonymous namespace


TEST_CASE("dtwc_cl --checkpoint-interval publishes and resumes a checkpoint",
          "[cli][checkpoint]")
{
  const dtwc::test_support::ScratchDirectory scratch{ "cli_ckpt" };
  const fs::path data = scratch.path / "series";
  const fs::path checkpoint_dir = scratch.path / "ckpt";
  const fs::path checkpoint = checkpoint_dir / "cli_ckpt.dtwm";
  write_fixture_series(data);
  auto argv = cli_args(data, scratch.path, checkpoint_dir);
  argv.insert(argv.end(), { "--checkpoint-interval", "2" });

  const CommandResult first = run(argv, scratch.path);
  INFO("stdout:\n" << first.out << "\nstderr:\n" << first.err);
  REQUIRE(first.exit_code == 0);
  REQUIRE(first.out.find("No checkpoint in") != std::string::npos);
  REQUIRE(fs::file_size(checkpoint) == checkpoint_bytes);
  REQUIRE_FALSE(fs::exists(fs::path(checkpoint).concat(".tmp")));

  const CommandResult second = run(argv, scratch.path);
  INFO("stdout:\n" << second.out << "\nstderr:\n" << second.err);
  REQUIRE(second.exit_code == 0);
  REQUIRE(second.out.find("Resumed from checkpoint") != std::string::npos);
  REQUIRE(std::vector<fs::path>(fs::directory_iterator(checkpoint_dir), fs::directory_iterator())
          == std::vector<fs::path>{ checkpoint });
}


TEST_CASE("dtwc_cl refuses a checkpoint of other data", "[cli][checkpoint]")
{
  // It used to recompute every distance, exit 0, and overwrite the checkpoint.
  const dtwc::test_support::ScratchDirectory scratch{ "cli_ckpt_other" };
  const fs::path checkpoint_dir = scratch.path / "ckpt";
  write_fixture_series(scratch.path / "series");
  write_fixture_series(scratch.path / "other", 1);

  REQUIRE(run(cli_args(scratch.path / "series", scratch.path, checkpoint_dir), scratch.path).exit_code == 0);
  const auto saved = read_text(checkpoint_dir / "cli_ckpt.dtwm");

  const CommandResult other = run(cli_args(scratch.path / "other", scratch.path, checkpoint_dir), scratch.path);
  INFO("stdout:\n" << other.out << "\nstderr:\n" << other.err);
  REQUIRE(other.exit_code == 1);
  REQUIRE(other.err.find("fingerprint mismatch") != std::string::npos);
  REQUIRE(read_text(checkpoint_dir / "cli_ckpt.dtwm") == saved);
}


#ifdef DTWC_HAS_MMAP
TEST_CASE("dtwc_cl maps its checkpoint above --mmap-threshold", "[cli][checkpoint][mmap]")
{
  // One file for both storages: a checkpoint written in RAM is reopened mapped.
  const dtwc::test_support::ScratchDirectory scratch{ "cli_ckpt_mapped" };
  const fs::path checkpoint_dir = scratch.path / "ckpt";
  write_fixture_series(scratch.path / "series");
  const auto in_ram = cli_args(scratch.path / "series", scratch.path, checkpoint_dir);
  REQUIRE(run(in_ram, scratch.path).exit_code == 0);

  auto mapped = in_ram;
  mapped.insert(mapped.end(), { "--mmap-threshold", "0" });
  const CommandResult result = run(mapped, scratch.path);
  INFO("stdout:\n" << result.out << "\nstderr:\n" << result.err);
  REQUIRE(result.exit_code == 0);
  REQUIRE(result.out.find("Using memory-mapped distance matrix:") != std::string::npos);
  REQUIRE(result.out.find("cli_ckpt.dtwm") != std::string::npos);
  REQUIRE(fs::file_size(checkpoint_dir / "cli_ckpt.dtwm") == checkpoint_bytes);
  REQUIRE_FALSE(fs::exists(scratch.path / "out" / "cli_ckpt.dtwm"));
}
#endif


TEST_CASE("dtwc_cl rejects --checkpoint-interval without --checkpoint",
          "[cli][checkpoint]")
{
  const dtwc::test_support::ScratchDirectory scratch{ "cli_ckpt_bad" };
  const fs::path data = scratch.path / "series";
  write_fixture_series(data);

  const CommandResult result = run(
    { cli_executable().string(),
      "-i", data.string(),
      "-k", "2",
      "-o", (scratch.path / "out").string(),
      "--name", "cli_ckpt",
      "--skip-rows", "1",
      "--skip-cols", "1",
      "--checkpoint-interval", "2" },
    scratch.path);

  INFO("stdout:\n" << result.out << "\nstderr:\n" << result.err);
  REQUIRE(result.exit_code == 1);
  REQUIRE(result.err.find("--checkpoint-interval requires --checkpoint")
          != std::string::npos);
}
