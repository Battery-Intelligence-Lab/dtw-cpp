/**
 * @file test_supply_chain_pinning.cpp
 * @brief Regression tests for Task 0.12 — build supply-chain pinning.
 *
 * Targets the "Build supply chain" HIGH findings of the 2026-06-01 adversarial
 * audit (handoff-2026-06-01-adversarial-audit.md). These are properties of the
 * build configuration files themselves, so the tests assert on the on-disk
 * text of `cmake/Dependencies.cmake`, `dtwc/mip/CMakeLists.txt`, and the CI
 * workflow that carried the Codecov step. Each SECTION pins one bug; the
 * comment states why the UNFIXED tree fails it. None of the assertions were
 * weakened to pass.
 *
 * Text-scan caveat: these assertions are cheap sentinels over build files, not
 * behavioural proofs. Where a directive has a runtime/configure consequence
 * (e.g. the R3 llfio guard below) the comment names the live gate that was
 * actually executed to prove the behaviour.
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
#include <iostream>
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
    // the Arrow OPEN comment, so only genuine directives count.
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

TEST_CASE("mip-solvers links llfio only under the optional-deps guard",
  "[supply-chain][build][R3]")
{
  // Task R3 (2026-07-07): the "llfio REQUIRED violates optional-deps rule" HIGH
  // was only half-closed. Phase 0 (task 0.12) added option(DTWC_ENABLE_LLFIO)
  // and guarded the dtwc++ link, but dtwc/mip/CMakeLists.txt still listed
  // `llfio_hl` in the UNCONDITIONAL PRIVATE link list of the mip-solvers OBJECT
  // library, so a no-optional-deps configuration (-DDTWC_ENABLE_LLFIO=OFF, or
  // llfio absent) referenced a target that does not exist.
  //
  // This is a TEXT-SCAN SENTINEL only: it pins the fixed wording so a later edit
  // cannot silently reintroduce the unconditional link. It is NOT the
  // behavioural proof. The REAL gate is a live CMake configure that R3 actually
  // ran (explicitly granted):
  //     cmake --preset clang-win -B build/llfio-off-check -DDTWC_ENABLE_LLFIO=OFF
  // which exited 0 ("Generating done"), printed
  //     "llfio:    OFF (DTWC_ENABLE_LLFIO=OFF) — mmap disabled."
  // and produced a compile_commands.json in which NO mip translation unit
  // (mip_Gurobi/mip_Highs/benders) carries an llfio include path or a
  // -DDTWC_HAS_MMAP define. The exercised LIVE entry point is CMake's
  // configure+generate of dtwc/mip/CMakeLists.txt — the very file read below —
  // not a stubbed or dead function.
  std::string mip = read_file(repo_root() / "dtwc" / "mip" / "CMakeLists.txt");

  // Scan directives, not prose: strip `# ... EOL` comments so the explanatory
  // comment block (which legitimately names llfio_hl before the guard) cannot
  // defeat the "nothing precedes the guard" check below.
  mip = std::regex_replace(mip, std::regex{ R"(#[^\n]*)" }, "");

  const std::string guard = "if(TARGET llfio_hl)";
  const auto guard_pos = mip.find(guard);
  REQUIRE(guard_pos != std::string::npos); // conditional guard must exist

  // No llfio_hl directive may precede the guard. In the UNFIXED tree llfio_hl
  // sat in the base `target_link_libraries(mip-solvers PRIVATE ...)` list above
  // any guard, so this assertion fails there.
  REQUIRE(mip.substr(0, guard_pos).find("llfio_hl") == std::string::npos);

  // The guarded body links the target AND mirrors dtwc++'s DTWC_HAS_MMAP
  // define. Both are required: Problem::distMat_t is
  // std::variant<DenseDistanceMatrix, MmapDistanceMatrix> (Problem.hpp:88) and
  // every mip source includes Problem.hpp, so mip-solvers must see the same
  // DTWC_HAS_MMAP state as dtwc++ or Problem's layout diverges across TUs (ODR).
  const std::string after = mip.substr(guard_pos);
  REQUIRE(contains(after, "target_link_libraries(mip-solvers PRIVATE llfio_hl)"));
  REQUIRE(contains(after, "DTWC_HAS_MMAP"));
}

TEST_CASE("CMake presets match the portable project floor and host",
  "[build][cmake][presets][F16]")
{
  const std::string root = read_file(repo_root() / "CMakeLists.txt");
  const std::string presets = read_file(repo_root() / "CMakePresets.json");
  const std::string pyproject = read_file(repo_root() / "pyproject.toml");

  // These are fixed scalar sentinels, not a replacement JSON/CMake parser.
  // CMake's string(JSON) configure guard and live --list-presets/configure
  // probes are the independent behavioral arbiters.
  CHECK(root.starts_with("cmake_minimum_required(VERSION 3.26)"));
  CHECK(regex_count(root,
          std::regex{ R"(cmake_minimum_required\s*\(\s*VERSION\s+3\.26\s*\))" })
    == 1);
  CHECK(regex_count(
          root, std::regex{ R"(cmake_minimum_required\s*\()" })
    == 1);
  CHECK(regex_count(presets, std::regex{ R"("version"\s*:\s*6\s*,)" }) == 1);
  CHECK(regex_count(
          presets, std::regex{ R"("cmakeMinimumRequired"\s*:)" })
    == 1);
  CHECK(regex_count(presets,
          std::regex{
            R"("cmakeMinimumRequired"\s*:\s*\{\s*"major"\s*:\s*3\s*,\s*"minor"\s*:\s*26\s*,\s*"patch"\s*:\s*0\s*\})" })
    == 1);
  CHECK(regex_count(pyproject,
          std::regex{
            R"((^|[\r\n])[ \t]*cmake\.version\s*=\s*">=3\.26")" })
    == 1);
  CHECK(regex_count(presets,
          std::regex{ R"("binaryDir"\s*:\s*"\$\{sourceDir\}/build")" })
    == 1);

  // Exactly three configure presets own a C++ compiler: Windows Clang, Linux
  // GCC, and the deliberate macOS system compiler. Hidden `default` must not
  // smuggle a machine-specific compiler into inheriting presets.
  CHECK(regex_count(
          presets, std::regex{ R"("CMAKE_CXX_COMPILER"\s*:)" })
    == 3);
  CHECK(regex_count(presets, std::regex{ R"("CMAKE_C_COMPILER"\s*:)" }) == 1);
  CHECK(regex_count(
          presets, std::regex{ R"("[A-Za-z]:[/\\])" })
    == 0);
  CHECK_FALSE(contains(presets, "C:/Program Files/LLVM"));

  const std::string default_name = R"("name": "default")";
  const std::string win_name = R"("name": "clang-win")";
  const std::string debug_name = R"("name": "clang-win-debug")";
  const std::string msvc_name = R"("name": "msvc")";
  const std::string gcc_name = R"("name": "gcc-linux")";
  const std::string mac_name = R"("name": "clang-macos")";

  const auto default_pos = presets.find(default_name);
  const auto win_pos = presets.find(win_name, default_pos + default_name.size());
  const auto debug_pos = presets.find(debug_name, win_pos + win_name.size());
  const auto msvc_pos = presets.find(msvc_name, debug_pos + debug_name.size());
  const auto gcc_pos = presets.find(gcc_name, msvc_pos + msvc_name.size());
  const auto mac_pos = presets.find(mac_name, gcc_pos + gcc_name.size());
  const auto build_pos = presets.find("\"buildPresets\"", mac_pos);
  const auto test_pos = presets.find("\"testPresets\"", build_pos);

  REQUIRE(default_pos != std::string::npos);
  REQUIRE(win_pos != std::string::npos);
  REQUIRE(debug_pos != std::string::npos);
  REQUIRE(msvc_pos != std::string::npos);
  REQUIRE(gcc_pos != std::string::npos);
  REQUIRE(mac_pos != std::string::npos);
  REQUIRE(build_pos != std::string::npos);
  REQUIRE(test_pos != std::string::npos);

  const std::string default_block =
    presets.substr(default_pos, win_pos - default_pos);
  const std::string win_block = presets.substr(win_pos, debug_pos - win_pos);
  const std::string debug_block =
    presets.substr(debug_pos, msvc_pos - debug_pos);
  const std::string msvc_block = presets.substr(msvc_pos, gcc_pos - msvc_pos);
  const std::string gcc_block = presets.substr(gcc_pos, mac_pos - gcc_pos);
  const std::string mac_block = presets.substr(mac_pos, build_pos - mac_pos);
  const std::string build_block =
    presets.substr(build_pos, test_pos - build_pos);
  const std::string test_block = presets.substr(test_pos);

  const auto has_host_condition = [](const std::string &block,
                                    const std::string &host) {
    const std::string pattern =
      R"("condition"\s*:\s*\{\s*"type"\s*:\s*"equals"\s*,\s*"lhs"\s*:\s*"\$\{hostSystemName\}"\s*,\s*"rhs"\s*:\s*")"
      + host + R"("\s*\})";
    return regex_count(block, std::regex{ pattern }) == 1;
  };

  CHECK(contains(default_block, R"("hidden": true)"));
  CHECK_FALSE(contains(default_block, "CMAKE_CXX_COMPILER"));
  CHECK_FALSE(contains(default_block, "CMAKE_C_COMPILER"));
  CHECK_FALSE(contains(default_block, "CMAKE_TOOLCHAIN_FILE"));
  CHECK(regex_count(win_block,
          std::regex{
            R"("CMAKE_CXX_COMPILER"\s*:\s*"clang\+\+"\s*[,}])" })
    == 1);
  CHECK(contains(win_block, R"("inherits": "default")"));
  CHECK(has_host_condition(win_block, "Windows"));
  CHECK(contains(debug_block, R"("inherits": "clang-win")"));
  CHECK(has_host_condition(debug_block, "Windows"));
  CHECK(contains(msvc_block, R"("inherits": "default")"));
  CHECK(has_host_condition(msvc_block, "Windows"));
  CHECK(regex_count(gcc_block,
          std::regex{ R"("CMAKE_CXX_COMPILER"\s*:\s*"g\+\+"\s*[,}])" })
    == 1);
  CHECK(contains(gcc_block, R"("inherits": "default")"));
  CHECK(has_host_condition(gcc_block, "Linux"));
  CHECK(regex_count(mac_block,
          std::regex{
            R"("CMAKE_C_COMPILER"\s*:\s*"/usr/bin/clang"\s*[,}])" })
    == 1);
  CHECK(regex_count(mac_block,
          std::regex{
            R"("CMAKE_CXX_COMPILER"\s*:\s*"/usr/bin/clang\+\+"\s*[,}])" })
    == 1);
  CHECK(contains(mac_block, R"("inherits": "default")"));
  CHECK(has_host_condition(mac_block, "Darwin"));

  // Conditions must cover configure, build, and test entry points. Build/test
  // presets do not automatically inherit their configure preset's condition.
  CHECK(regex_count(presets,
          std::regex{ R"("lhs"\s*:\s*"\$\{hostSystemName\}")" })
    == 10);
  CHECK(regex_count(
          presets, std::regex{ R"("type"\s*:\s*"equals")" })
    == 10);
  CHECK(regex_count(
          presets, std::regex{ R"("rhs"\s*:\s*"Windows")" })
    == 6);
  CHECK(regex_count(presets, std::regex{ R"("rhs"\s*:\s*"Linux")" }) == 1);
  CHECK(regex_count(
          presets, std::regex{ R"("rhs"\s*:\s*"Darwin")" })
    == 3);

  CHECK(regex_count(build_block,
          std::regex{
            R"("configurePreset"\s*:\s*"clang-win"\s*[,}])" })
    == 1);
  CHECK(regex_count(build_block,
          std::regex{
            R"("configurePreset"\s*:\s*"clang-win-debug"\s*[,}])" })
    == 1);
  CHECK(regex_count(build_block,
          std::regex{
            R"("configurePreset"\s*:\s*"clang-macos"\s*[,}])" })
    == 1);
  CHECK(regex_count(test_block,
          std::regex{
            R"("configurePreset"\s*:\s*"clang-win"\s*[,}])" })
    == 1);
  CHECK(regex_count(test_block,
          std::regex{
            R"("configurePreset"\s*:\s*"clang-macos"\s*[,}])" })
    == 1);

  const auto build_win_pos = build_block.find(win_name);
  const auto build_debug_pos =
    build_block.find(debug_name, build_win_pos + win_name.size());
  const auto build_mac_pos =
    build_block.find(mac_name, build_debug_pos + debug_name.size());
  const auto test_win_pos = test_block.find(win_name);
  const auto test_mac_pos =
    test_block.find(mac_name, test_win_pos + win_name.size());
  REQUIRE(build_win_pos != std::string::npos);
  REQUIRE(build_debug_pos != std::string::npos);
  REQUIRE(build_mac_pos != std::string::npos);
  REQUIRE(test_win_pos != std::string::npos);
  REQUIRE(test_mac_pos != std::string::npos);

  const std::string build_win_block =
    build_block.substr(build_win_pos, build_debug_pos - build_win_pos);
  const std::string build_debug_block =
    build_block.substr(build_debug_pos, build_mac_pos - build_debug_pos);
  const std::string build_mac_block = build_block.substr(build_mac_pos);
  const std::string test_win_block =
    test_block.substr(test_win_pos, test_mac_pos - test_win_pos);
  const std::string test_mac_block = test_block.substr(test_mac_pos);
  CHECK(has_host_condition(build_win_block, "Windows"));
  CHECK(has_host_condition(build_debug_block, "Windows"));
  CHECK(has_host_condition(build_mac_block, "Darwin"));
  CHECK(has_host_condition(test_win_block, "Windows"));
  CHECK(has_host_condition(test_mac_block, "Darwin"));

  std::cout
    << "F16_CMAKE_PRESETS floor=3.26.0 compiler=clang++ "
       "host_conditions=ran metadata_guard=ran skips=0\n";
}
