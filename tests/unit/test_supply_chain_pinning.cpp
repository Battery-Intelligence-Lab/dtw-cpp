/**
 * @file test_supply_chain_pinning.cpp
 * @brief Regression tests for Task 0.12 — build supply-chain pinning.
 *
 * Targets the "Build supply chain" HIGH findings of the 2026-06-01 adversarial
 * audit (handoff-2026-06-01-adversarial-audit.md). These are properties of the
 * build configuration files themselves, so the tests assert on the on-disk
 * text of `cmake/Dependencies.cmake` and the CI workflow that carried the
 * Codecov step. Each SECTION pins one bug; the comment states why the UNFIXED
 * tree fails it. None of the assertions were weakened to pass.
 *
 * How the files are located: every test target is compiled with
 * `-DDTWC_TEST_DATA_DIR="<repo>/data"` (cmake/Coverage.cmake) and ctest runs it
 * with WORKING_DIRECTORY = <repo>. The repo root is therefore the parent of
 * DTWC_TEST_DATA_DIR — a compile-time absolute path, not a runtime
 * repo-relative dependency. A "./data" fallback keeps the file self-contained.
 *
 * @date 2026-07-07
 */

#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <regex>
#include <sstream>
#include <string>

#ifndef DTWC_TEST_DATA_DIR
#define DTWC_TEST_DATA_DIR "./data"
#endif

namespace {

namespace fs = std::filesystem;

fs::path repo_root()
{
  // DTWC_TEST_DATA_DIR == "<repo>/data" -> parent is the repo root.
  return fs::path{ DTWC_TEST_DATA_DIR }.parent_path();
}

std::string read_file(const fs::path &p)
{
  std::ifstream in(p, std::ios::binary);
  REQUIRE(in.good()); // if this fails, the build-config file is missing/moved
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

bool contains(const std::string &haystack, const std::string &needle)
{
  return haystack.find(needle) != std::string::npos;
}

std::size_t regex_count(const std::string &text, const std::regex &re)
{
  auto begin = std::sregex_iterator(text.begin(), text.end(), re);
  auto end = std::sregex_iterator();
  return static_cast<std::size_t>(std::distance(begin, end));
}

} // namespace

TEST_CASE("Dependencies.cmake pins the supply chain", "[supply-chain][build]")
{
  const std::string deps = read_file(repo_root() / "cmake" / "Dependencies.cmake");

  SECTION("llfio is pinned to a commit SHA, not the moving `develop` branch")
  {
    // Bug (audit): `GIT_TAG develop` builds whatever upstream HEAD is at
    // configure time — an upstream force-push/hijack silently changes what we
    // compile. UNFIXED tree contains "GIT_TAG develop" and NO 40-hex SHA after
    // GIT_TAG, so both assertions below fail.
    REQUIRE_FALSE(contains(deps, "GIT_TAG develop"));
    REQUIRE(std::regex_search(deps, std::regex{ R"(GIT_TAG\s+[0-9a-f]{40})" }));
  }

  SECTION("every pinned CPM URL tarball carries a URL_HASH SHA256")
  {
    // Bug (audit): "all CPM URL tarballs no URL_HASH" — a tampered/edited
    // mirror tarball would be accepted silently. UNFIXED tree has ZERO real
    // `URL_HASH SHA256=<64-hex>` directives (count 0 < 4). The `[0-9a-f]{64}`
    // anchor deliberately does NOT match the "SHA256=<hash>" placeholders in
    // the yaml-cpp / Arrow OPEN comments, so only genuine directives count.
    const std::regex url_hash{ R"(URL_HASH\s+SHA256=[0-9a-f]{64})" };
    REQUIRE(regex_count(deps, url_hash) >= 4);
  }

  SECTION("quickcpplib pre-clone is pinned to a commit SHA")
  {
    // Bug (audit): the pre-clone did `git clone --depth 1 <default branch>`,
    // patched and then executed the result at configure time — arbitrary
    // moving code. UNFIXED tree has no `_dtwc_qcl_sha` pin. Fixed tree assigns
    // a 40-hex SHA and detaches onto it.
    REQUIRE(std::regex_search(
      deps, std::regex{ R"(set\(_dtwc_qcl_sha\s+"[0-9a-f]{40}"\))" }));
    REQUIRE(contains(deps, "checkout --detach"));
  }

  SECTION("llfio is optional, not a hard-required dependency")
  {
    // Bug (audit): "llfio REQUIRED violates optional-deps rule" — configure
    // aborted via `message(FATAL_ERROR ... llfio is a required dependency)`.
    // UNFIXED tree still contains that message and has no DTWC_ENABLE_LLFIO
    // guard, so both assertions below fail.
    REQUIRE_FALSE(contains(deps, "llfio is a required dependency"));
    REQUIRE(contains(deps, "DTWC_ENABLE_LLFIO"));
  }
}

TEST_CASE("CI coverage upload does not pipe a remote script into a shell",
  "[supply-chain][ci]")
{
  const std::string wf =
    read_file(repo_root() / ".github" / "workflows" / "documentation.yml");

  // Bug (audit): `bash <(curl https://codecov.io/bash) -f coverage.info` with
  // CODECOV_TOKEN in the environment = remote code execution + token exposure
  // on every PR run. UNFIXED tree contains "codecov.io/bash" and "bash <(curl";
  // the fixed tree uses the maintained official action instead.
  REQUIRE_FALSE(contains(wf, "codecov.io/bash"));
  REQUIRE_FALSE(contains(wf, "bash <(curl"));
  REQUIRE(contains(wf, "codecov/codecov-action@"));
}
