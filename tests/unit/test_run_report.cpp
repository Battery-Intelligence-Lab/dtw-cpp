/**
 * @file test_run_report.cpp
 * @brief The run report of dtwc::run: its eight sections in order and its stage table, the report of a failed run
 *        beside the run's unchanged error, and `report = false`, which makes none.
 *
 * @details The report's numbers are the run's own and its machine's: no case compares a time or a machine fact, only
 *          what the text must hold whatever they are (the headings once each, the stage rows each once and in the
 *          run's order, their wall seconds summing to no more than the total row's).
 *
 * @author Volkan Kumtepeli
 * @author Claude Opus 5.5 (RR1b)
 * @date 03 Nov 2026
 */

#include <dtwc.hpp>
#include <cli/report.hpp>
#include <cli/run.hpp>
#include <config.hpp>

#include "../support/scratch_directory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace fs = std::filesystem;
using Catch::Matchers::ContainsSubstring;

/// `n` series of 20 steps in three groups.
dtwc::Data three_groups(std::size_t n)
{
  std::vector<std::vector<double>> series(n);
  std::vector<std::string> names(n);
  for (std::size_t i = 0; i < n; ++i) {
    for (std::size_t v = 0; v < 20; ++v)
      series[i].push_back(10.0 * double(i % 3) + std::sin(0.3 * double(v) + 0.01 * double(i)));
    names[i] = "s" + std::to_string(i);
  }
  return dtwc::Data(std::move(series), std::move(names), 1);
}

dtwc::Config pam_into(const fs::path &output)
{
  dtwc::Config config;
  config.method = dtwc::Method::PAM;
  config.k = 3;
  config.name = "rr";
  config.output = dtwc::path_to_utf8(output);
  config.ram_limit = std::size_t{ 1 } << 30;
  return config;
}

/// A text file's lines, a CR before each newline dropped.
std::string read_text(const fs::path &path)
{
  std::ifstream in(path, std::ios::binary);
  std::ostringstream text;
  text << in.rdbuf();
  std::string out = text.str();
  out.erase(std::remove(out.begin(), out.end(), '\r'), out.end());
  return out;
}

std::vector<std::string> lines_of(const std::string &text)
{
  std::vector<std::string> lines;
  std::istringstream stream(text);
  for (std::string line; std::getline(stream, line);) lines.push_back(line);
  return lines;
}

/// The stage table's rows: each row's name and wall seconds, the total row last.
std::vector<std::pair<std::string, double>> stage_rows(const std::vector<std::string> &lines)
{
  std::vector<std::pair<std::string, double>> rows;
  bool in_table = false;
  const std::regex row("^   ([a-z]+) +([0-9]+\\.[0-9]{3}) ");
  for (const auto &line : lines) {
    if (line.rfind("5. ", 0) == 0) in_table = true;
    else if (line.rfind("6. ", 0) == 0) break;
    std::smatch m;
    if (in_table && std::regex_search(line, m, row)) rows.emplace_back(m[1].str(), std::stod(m[2].str()));
  }
  return rows;
}

} // namespace

TEST_CASE("The run report: eight sections in order, the stages once each, their walls within the total",
          "[run][report]")
{
  const dtwc::test_support::ScratchDirectory scratch{ "run_report" };
  const auto config = pam_into(scratch.path / "out");
  const auto result = dtwc::run(config, three_groups(60));
  const std::string &text = result.report();
  REQUIRE_FALSE(text.empty());
  CHECK(read_text(scratch.path / "out" / "rr_report.txt") == text); // the file is the text

  // Result::save writes the same text last, beside the result files.
  const auto saved = scratch.path / "saved";
  result.save(saved);
  CHECK(read_text(saved / "rr_report.txt") == text);
  CHECK(fs::exists(saved / "rr_labels.csv"));
  CHECK(fs::exists(saved / "rr_silhouettes.csv"));

  const auto lines = lines_of(text);
  std::vector<int> headings;
  for (const auto &line : lines) {
    CHECK(line.size() <= 100);
    CHECK(std::all_of(line.begin(), line.end(), [](char c) { return c >= ' ' && c <= '~'; })); // plain ASCII
    if (line.size() > 2 && line[0] >= '1' && line[0] <= '9' && line.compare(1, 2, ". ") == 0)
      headings.push_back(line[0] - '0');
  }
  CHECK(headings == std::vector<int>{ 1, 2, 3, 4, 5, 6, 7, 8 });
  CHECK_THAT(text, ContainsSubstring("   status         completed\n"));

  // PAM fills the matrix in a stage of its own, then clusters; the silhouettes are the write's.
  const auto rows = stage_rows(lines);
  std::vector<std::string> names;
  double sum = 0.0;
  for (std::size_t i = 0; i + 1 < rows.size(); ++i) {
    names.push_back(rows[i].first);
    sum += rows[i].second;
  }
  CHECK(names == std::vector<std::string>{ "check", "plan", "read", "storage", "fill", "cluster", "write",
                                           "silhouettes" });
  REQUIRE(rows.size() == names.size() + 1);
  CHECK(rows.back().first == "total");
  CHECK(sum <= rows.back().second + 1e-9);
}

TEST_CASE("A run that fails past its checks writes its report up to the failing stage and throws its error unchanged",
          "[run][report]")
{
  const dtwc::test_support::ScratchDirectory scratch{ "run_report_failed" };
  // k above N: refused once the series are read, in stage read.
  auto config = pam_into(scratch.path / "out");
  config.k = 7;
  std::string reported, unreported;
  try {
    (void)dtwc::run(config, three_groups(6));
    FAIL("k = 7 of 6 series ran");
  } catch (const dtwc::InvalidInput &e) {
    reported = e.what();
  }
  auto quiet = config;
  quiet.report = false;
  quiet.output = dtwc::path_to_utf8(scratch.path / "quiet");
  try {
    (void)dtwc::run(quiet, three_groups(6));
    FAIL("k = 7 of 6 series ran");
  } catch (const dtwc::InvalidInput &e) {
    unreported = e.what();
  }
  std::string direct; // the error itself, as the check throws it
  try {
    dtwc::require_clusterable(7, 6);
  } catch (const dtwc::InvalidInput &e) {
    direct = e.what();
  }
  CHECK(reported == direct);
  CHECK(unreported == direct);
  CHECK_FALSE(fs::exists(scratch.path / "quiet" / "rr_report.txt"));

  const auto path = scratch.path / "out" / "rr_report.txt";
  REQUIRE(fs::exists(path));
  const std::string text = read_text(path);
  CHECK_THAT(text, ContainsSubstring("   status         FAILED in stage read: InvalidInput\n"));
  // The method the run was asked for, though it failed before resolving it (a defect found 2026-11-03: "method auto").
  CHECK_THAT(text, ContainsSubstring("   configuration  method pam, k 7,"));
  CHECK_THAT(text, ContainsSubstring("   error          InvalidInput: " + reported + "\n"));
  const auto rows = stage_rows(lines_of(text));
  REQUIRE(rows.size() >= 2);
  CHECK(rows[rows.size() - 2].first == "read"); // no row after the failing stage's, then the total

  // A failure in `check` (no k) writes no report, though its output directory exists.
  auto unchecked = pam_into(scratch.path / "checked");
  unchecked.k = 0;
  fs::create_directories(scratch.path / "checked");
  CHECK_THROWS_AS(dtwc::run(unchecked, three_groups(6)), dtwc::InvalidInput);
  CHECK_FALSE(fs::exists(scratch.path / "checked" / "rr_report.txt"));
}

TEST_CASE("The fill is a stage of its own for PAM, Lloyd's k-medoids and CLARA of every series, and for no other method",
          "[run][report]")
{
  const dtwc::test_support::ScratchDirectory scratch{ "run_report_fill" };
  const auto stages_of = [&](dtwc::Method method, const char *directory) {
    auto config = pam_into(scratch.path / directory);
    config.method = method;
    if (method == dtwc::Method::CLARA) config.sample_size = 60; // every series
    const auto result = dtwc::run(config, three_groups(60));
    std::vector<std::string> names;
    for (const auto &row : stage_rows(lines_of(result.report()))) names.push_back(row.first);
    return names;
  };
  const std::vector<std::string> filled{ "check", "plan", "read", "storage", "fill",
                                         "cluster", "write", "silhouettes", "total" };
  CHECK(stages_of(dtwc::Method::PAM, "pam") == filled);
  CHECK(stages_of(dtwc::Method::Kmedoids, "lloyd") == filled);
  CHECK(stages_of(dtwc::Method::CLARA, "clara") == filled);
  const std::vector<std::string> inside{ "check", "plan", "read", "storage", "cluster", "write", "silhouettes", "total" };
  CHECK(stages_of(dtwc::Method::Hierarchical, "hierarchical") == inside);
}

TEST_CASE("The report's command row double-quotes what the key table writes as a TOML literal, which cmd.exe would "
          "hand on with its single quotes",
          "[run][report]")
{
  dtwc::Config config;
  // Never opened: command_of only spells the keys. The key table writes a string of over 100 characters with a
  // backslash as '''...''' and a one-character string as 'x'.
  const std::string long_path = "C:\\" + std::string(100, 'd') + "\\series.csv";
  config.input = long_path;
  config.name = "x";
  config.k = 3;
  const std::string command = dtwc::detail::command_of(config, dtwc::Method::PAM);
  CHECK_THAT(command, ContainsSubstring(" --input \"" + long_path + "\""));
  CHECK_THAT(command, ContainsSubstring(" --name \"x\""));
  CHECK_THAT(command, ContainsSubstring(" --n-clusters 3"));
  CHECK_THAT(command, ContainsSubstring(" --method \"pam\""));
  CHECK_THAT(command, !ContainsSubstring("'"));
}

TEST_CASE("report = false: no report text and no report file, the results as ever", "[run][report]")
{
  const dtwc::test_support::ScratchDirectory scratch{ "run_report_off" };
  auto config = pam_into(scratch.path / "out");
  config.report = false;
  const auto result = dtwc::run(config, three_groups(60));
  CHECK(result.report().empty());
  CHECK_FALSE(fs::exists(scratch.path / "out" / "rr_report.txt"));
  CHECK(fs::exists(scratch.path / "out" / "rr_labels.csv"));
}
