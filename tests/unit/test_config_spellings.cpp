/**
 * @file test_config_spellings.cpp
 * @brief Contract of cli::bind(): every spelling of every dtwc::Config key reads
 *        to one Config, and Config{} holds dtwc_cl's defaults.
 *
 * @details The golden text is tests/conformance/config_all_fields.toml: every key
 * at a non-default value, in to_config_text() form. A command line (short flags,
 * aliases, odd case, `--ram-limit 1G`), that file through `--config`, its YAML
 * twin and parse_config() pairs must each render to it, and Config{} must differ
 * from it on every line, so a key the golden leaves at its default fails.
 *
 * The second half drives the real dtwc_cl, whose options are bind()'s since step
 * S3: every option, alias and choice its --help lists must read the same through
 * bind(), and the defaults its --verbose echo reports must be Config{}'s (its
 * --print-config, against tests/conformance/config_defaults.toml and the golden
 * file, is test_cli_device_matrix's).
 *
 * @date 24 Sep 2026
 */

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <CLI/CLI.hpp>

#include "base/error.hpp"
#include "cli/config.hpp"

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <utility>
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

fs::path conformance_dir() { return fs::path{ DTWC_TEST_DATA_DIR }.parent_path() / "tests" / "conformance"; }

std::string read_text(const fs::path &path)
{
  std::ifstream input(path, std::ios::binary);
  REQUIRE(input.good());
  return { std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>() };
}

std::vector<std::string> lines_of(const std::string &text)
{
  std::vector<std::string> lines;
  std::istringstream stream(text);
  for (std::string line; std::getline(stream, line);) lines.push_back(line);
  return lines;
}

/// The golden file without its comment lines.
std::string golden_text()
{
  std::string text;
  for (const std::string &line : lines_of(read_text(conformance_dir() / "config_all_fields.toml")))
    if (!line.empty() && line.front() != '#') text += line + '\n';
  return text;
}

/// A Config from a dtwc_cl argument list, read through bind().
dtwc::Config parse_args(std::vector<std::string> args)
{
  dtwc::Config config;
  CLI::App app{ "DTWC++ -- Dynamic Time Warping Clustering" };
  dtwc::cli::bind(app, config);
  args.insert(args.begin(), "dtwc_cl");
  std::vector<const char *> argv;
  for (const std::string &arg : args) argv.push_back(arg.c_str());
  app.parse(static_cast<int>(argv.size()), argv.data());
  return config;
}

/// key -> value, quotes removed, of a to_config_text() rendering.
std::map<std::string, std::string> values_of(const std::string &text)
{
  std::map<std::string, std::string> values;
  for (const std::string &line : lines_of(text)) {
    const auto equals = line.find(" = ");
    REQUIRE(equals != std::string::npos);
    std::string value = line.substr(equals + 3);
    if (value.size() >= 2 && (value.front() == '"' || value.front() == '\'') && value.back() == value.front())
      value = value.substr(1, value.size() - 2);
    values[line.substr(0, equals)] = value;
  }
  return values;
}

/// Restores std::cerr however the scope ends.
struct CapturedStderr
{
  std::ostringstream text;
  std::streambuf *previous = std::cerr.rdbuf(text.rdbuf());
  ~CapturedStderr() { std::cerr.rdbuf(previous); }
};

// ---- the real dtwc_cl (as tests/unit/unit_test_cli_checkpoint.cpp drives it) ----

/// Directory holding this test executable, which is also where dtwc_cl is linked.
fs::path executable_directory()
{
#ifdef _WIN32
  std::wstring buffer(4096, L'\0');
  const DWORD length = GetModuleFileNameW(nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
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

fs::path cli_executable()
{
  fs::path candidate = executable_directory() / "dtwc_cl";
  if (!fs::exists(candidate)) candidate.replace_extension(".exe");
  INFO("dtwc_cl not found next to the test executable: " << candidate.string());
  REQUIRE(fs::is_regular_file(candidate));
  return candidate;
}

struct CommandResult
{
  int exit_code{ 0 };
  std::string out;
  std::string err;
};

/// Run a command, capturing its streams through files (the one capture route
/// that behaves the same under cmd.exe and a POSIX shell).
CommandResult run(const std::vector<std::string> &argv, const fs::path &scratch)
{
  const fs::path out_path = scratch / "stdout.txt";
  const fs::path err_path = scratch / "stderr.txt";
  std::string command;
  for (const auto &argument : argv) command += '"' + argument + "\" ";
  command += "> \"" + out_path.string() + "\" 2> \"" + err_path.string() + '"';
#ifdef _WIN32
  command = '"' + command + '"'; // cmd.exe /C strips the outer quote pair
#endif
  const int status = std::system(command.c_str());
#ifdef _WIN32
  const int exit_code = status;
#else
  const int exit_code = WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
  return { exit_code, read_text(out_path), read_text(err_path) };
}

fs::path fresh_scratch(const std::string &name)
{
  const fs::path scratch = fs::temp_directory_path() / name;
  fs::remove_all(scratch);
  fs::create_directories(scratch);
  return scratch;
}

std::string trimmed(const std::string &text)
{
  const auto first = text.find_first_not_of(' ');
  return first == std::string::npos ? std::string{} : text.substr(first, text.find_last_not_of(' ') - first + 1);
}

bool is_option_name(const std::string &word)
{
  const std::size_t dashes = word.rfind("--", 0) == 0 ? 2 : 1;
  return word.size() > dashes && word.front() == '-' && std::isalpha(static_cast<unsigned char>(word[dashes])) != 0;
}

/// One option line of dtwc_cl --help: its names and the entries of its {..} groups.
struct HelpOption
{
  std::vector<std::string> names;
  std::vector<std::string> choices;
};

std::vector<HelpOption> help_options(const std::string &help)
{
  std::vector<HelpOption> options;
  for (const std::string &line : lines_of(help)) {
    HelpOption option;
    std::istringstream words(line);
    for (std::string word; words >> word;) {
      if (word.back() == ',') word.pop_back();
      if (!is_option_name(word)) break;
      option.names.push_back(word);
    }
    if (option.names.empty()) continue;
    for (auto open = line.find('{'); open != std::string::npos; open = line.find('{', open + 1)) {
      std::istringstream entries(line.substr(open + 1, line.find('}', open) - open - 1));
      for (std::string entry; std::getline(entries, entry, ',');) option.choices.push_back(entry);
    }
    options.push_back(option);
  }
  return options;
}

} // namespace

TEST_CASE("every spelling of every Config key renders to the golden text", "[config][spellings]")
{
  const std::string golden = golden_text();
  REQUIRE(values_of(golden).size() == 49);

  SECTION("the golden file itself, through --config")
  {
    const std::string toml = (conformance_dir() / "config_all_fields.toml").string();
    CHECK(dtwc::to_config_text(parse_args({ "--config", toml })) == golden);
  }

  SECTION("its YAML twin, through --config")
  {
    const std::string yaml = (conformance_dir() / "config_all_fields.yaml").string();
#ifdef DTWC_HAS_YAML
    CHECK(dtwc::to_config_text(parse_args({ "--config", yaml })) == golden);
#else
    CHECK_THROWS_WITH(parse_args({ "--config", yaml }),
                      Catch::Matchers::ContainsSubstring("built without YAML support; use TOML"));
#endif
  }

  SECTION("a command line of short flags, aliases and odd case")
  {
    const dtwc::Config config = parse_args({
      "-i", "series.csv", "-o", "out", "--name", "golden", "--column", "values",
      "--data-type", "FP32", "--ram-limit", "1G",
      "-k", "5", "-m", "CLARA", "-b", "7", "--metric", "L2SQ", "--variant", "WDTW",
      "--max-iter", "25", "--n-init", "4", "--dc", "0.25",
      "--wdtw-g", "0.125", "--adtw-penalty", "2.5", "--sdtw-gamma", "0.5", "--msm-c", "3",
      "--twe-nu", "0.01", "--twe-lambda", "0.75", "--mv-mode", "Independent", "--missing-strategy", "AROW",
      "--sample-size", "40", "--n-samples", "3", "--seed", "7", "--batch-size", "64",
      "--batch-weighting", "debias", "--linkage", "Complete",
      "--skip-rows", "1", "--skip-cols", "2", "--delimiter", ";",
      "--dist-matrix", "dist.csv", "--checkpoint", "ckpt", "--checkpoint-interval", "10", "--resume",
      "--mmap-threshold", "1000",
      "--solver", "GUROBI", "--mip-gap", "1e-3", "--time-limit", "60", "--no-warm-start",
      "--numeric-focus", "3", "--mip-focus", "1", "--verbose-solver", "--benders", "yes",
      "--max-benders-iter", "50", "--lr-max-nodes", "5000",
      "-d", "CUDA:2", "--gpu-dtype", "double", "-v" });
    CHECK(dtwc::to_config_text(config) == golden);
  }

  SECTION("parse_config pairs, as Python keywords and MATLAB name-value pairs give them")
  {
    const dtwc::Config config = dtwc::parse_config({
      { "i", "series.csv" }, { "output", "out" }, { "name", "golden" }, { "column", "values" },
      { "data_precision", "f32" }, { "ram_limit", "1GiB" },
      { "k", "5" }, { "method", "clara" }, { "band", "7" }, { "metric", "sqeuclidean" }, { "variant", "wdtw" },
      { "max_iter", "25" }, { "n_init", "4" }, { "dc", "0.25" },
      { "wdtw_g", "0.125" }, { "adtw_penalty", "2.5" }, { "sdtw_gamma", "0.5" }, { "msm_c", "3" },
      { "twe_nu", "0.01" }, { "twe_lambda", "0.75" }, { "mv_mode", "independent" }, { "missing_strategy", "arow" },
      { "sample_size", "40" }, { "n_samples", "3" }, { "seed", "7" }, { "batch_size", "64" },
      { "batch_weighting", "debiased" }, { "linkage", "complete" },
      { "skip_rows", "1" }, { "skip_cols", "2" }, { "delimiter", ";" },
      { "dist_matrix", "dist.csv" }, { "checkpoint", "ckpt" }, { "checkpoint_interval", "10" }, { "resume", "true" },
      { "mmap_threshold", "1000" },
      { "solver", "gurobi" }, { "mip_gap", "0.001" }, { "time_limit", "60" }, { "no_warm_start", "true" },
      { "numeric_focus", "3" }, { "mip_focus", "1" }, { "verbose_solver", "true" }, { "benders", "on" },
      { "max_benders_iter", "50" }, { "lr_max_nodes", "5000" },
      { "device", "gpu:2" }, { "gpu_precision", "fp64" }, { "verbose", "true" } });
    CHECK(dtwc::to_config_text(config) == golden);
  }
}

TEST_CASE("Config{} differs from the golden text on every line", "[config][spellings]")
{
  const auto golden = lines_of(golden_text());
  const auto defaults = lines_of(dtwc::to_config_text(dtwc::Config{}));
  REQUIRE(defaults.size() == golden.size());
  for (std::size_t i = 0; i < golden.size(); ++i) {
    INFO("line " << i + 1 << ": golden '" << golden[i] << "', Config{} '" << defaults[i] << "'");
    CHECK(golden[i].substr(0, golden[i].find(" = ")) == defaults[i].substr(0, defaults[i].find(" = ")));
    CHECK(golden[i] != defaults[i]);
  }
}

TEST_CASE("an empty command line and no pairs give Config{}", "[config][spellings]")
{
  const std::string defaults = dtwc::to_config_text(dtwc::Config{});
  CHECK(dtwc::to_config_text(parse_args({})) == defaults);
  CHECK(dtwc::to_config_text(dtwc::parse_config({})) == defaults);
}

TEST_CASE("to_config_text round-trips awkward values through --config", "[config][spellings]")
{
  dtwc::Config config;
  config.name = "a # b = c";
  config.output = "";
  config.column = "'";
  config.dist_matrix = "line\none\x01";
  config.variant.wdtw_g = 0.1;
  config.variant.twe_nu = 1e-300;
  config.variant.sdtw_gamma = 123456.789;
  config.mip.lr_max_nodes = 9007199254740993; // 2^53 + 1: an integer a double would round
  const std::vector<std::string> inputs{ R"(C:\data dir\my "series", v2.csv)", "donn\xc3\xa9" "es/s\xc3\xa9rie.csv" };
  for (const std::string &input : inputs) {
    for (const char delimiter : { ',', '\t', '"', '#' }) {
      config.input = input;
      config.delimiter = delimiter;
      INFO("input '" << input << "', delimiter code " << static_cast<int>(delimiter));
      const std::string text = dtwc::to_config_text(config);
      dtwc::Config back;
      CLI::App app;
      dtwc::cli::bind(app, back);
      std::istringstream stream(text);
      app.parse_from_stream(stream); // the reader --config uses
      CHECK(dtwc::to_config_text(back) == text);
      CHECK(back.input == config.input);
      CHECK(back.delimiter == delimiter);
      CHECK(back.dist_matrix == config.dist_matrix);
      CHECK(back.variant.twe_nu == config.variant.twe_nu);
    }
  }
  // Readable, not CLI11's 'B"(..)"' escape: UTF-8 stays as it is, a tab is \t.
  const auto lines = lines_of(dtwc::to_config_text(config));
  CHECK(std::count(lines.begin(), lines.end(), "input = \"donn\xc3\xa9" "es/s\xc3\xa9rie.csv\"") == 1);
  config.delimiter = '\t';
  const auto tabbed = lines_of(dtwc::to_config_text(config));
  CHECK(std::count(tabbed.begin(), tabbed.end(), R"(delimiter = "\t")") == 1);
  CHECK(std::count(tabbed.begin(), tabbed.end(), R"(dist-matrix = "line\none\u0001")") == 1);
}

TEST_CASE("flags beat the file; unknown keys and unreadable values are errors", "[config][spellings]")
{
  const std::string toml = (conformance_dir() / "config_all_fields.toml").string();
  const auto values = values_of(dtwc::to_config_text(parse_args({ "--config", toml, "-k", "9", "--method", "pam" })));
  CHECK(values.at("n-clusters") == "9");
  CHECK(values.at("method") == "pam");
  CHECK(values.at("band") == "7"); // the file still supplies the rest

  CHECK_THROWS_MATCHES(dtwc::parse_config({ { "max_iterations", "5" } }), dtwc::InvalidInput,
                       Catch::Matchers::Message("INI was not able to parse max-iterations"));
  // One pair is one line: a key cannot smuggle in a second setting.
  CHECK_THROWS_AS(dtwc::parse_config({ { "k\nband", "3" } }), dtwc::InvalidInput);
  CHECK_THROWS_AS(dtwc::parse_config({ { "config", "other.toml" } }), dtwc::InvalidInput);
  CHECK_THROWS_AS(dtwc::parse_config({ { "method", "kmeans" } }), dtwc::InvalidInput);
  CHECK_THROWS_AS(dtwc::parse_config({ { "k", "0" } }), dtwc::InvalidInput);
  CHECK_THROWS_AS(dtwc::parse_config({ { "seed", "-1" } }), dtwc::InvalidInput); // unsigned: the type says no
  CHECK_THROWS_AS(dtwc::parse_config({ { "mmap_threshold", "-1" } }), dtwc::InvalidInput);
  CHECK_THROWS_AS(dtwc::parse_config({ { "benders", "of" } }), dtwc::InvalidInput);
  CHECK_THROWS_WITH(dtwc::parse_config({ { "ram_limit", "1.5" } }),
                    Catch::Matchers::ContainsSubstring("whole positive byte count"));
  CHECK_THROWS_AS(dtwc::parse_config({ { "delimiter", ";;" } }), dtwc::InvalidInput);
  CHECK_THROWS_AS(dtwc::parse_config({ { "device", "tpu" } }), dtwc::DeviceError);
  CHECK(values_of(dtwc::to_config_text(dtwc::parse_config({ { "ram_limit", "1.5G" } }))).at("ram-limit")
        == "1610612736");
}

TEST_CASE("the deprecated spellings warn and yield to the canonical ones", "[config][spellings]")
{
  std::map<std::string, std::string> alone;
  std::map<std::string, std::string> both;
  std::string warnings;
  {
    CapturedStderr captured;
    alone = values_of(dtwc::to_config_text(parse_args({ "--clusters", "6", "--restart" })));
    both = values_of(dtwc::to_config_text(parse_args({ "--clusters", "6", "-k", "2" })));
    warnings = captured.text.str();
  }
  CHECK(alone.at("n-clusters") == "6");
  CHECK(alone.at("resume") == "true");
  CHECK(both.at("n-clusters") == "2");
  CHECK(warnings
        == "[dtwc] warning: '--clusters' is deprecated, use '--n-clusters' instead\n"
           "[dtwc] warning: '--restart' is deprecated, use '--resume' instead\n"
           "[dtwc] warning: '--clusters' is deprecated, use '--n-clusters' instead\n");
  CHECK(alone.count("clusters") == 0); // the text form never writes a deprecated key
  CHECK(alone.count("restart") == 0);
}

TEST_CASE("bind() reads every option, alias and choice dtwc_cl --help lists", "[config][cli]")
{
  const fs::path scratch = fresh_scratch("dtwc_test_config_spellings_help");
  const CommandResult help = run({ cli_executable().string(), "--help" }, scratch);
  INFO("dtwc_cl --help:\n" << help.out << help.err);
  REQUIRE(help.exit_code == 0);
  const std::vector<HelpOption> listed = help_options(help.out);
  REQUIRE(listed.size() >= 40);

  dtwc::Config config;
  CLI::App app;
  dtwc::cli::bind(app, config);
  std::set<std::string> keys_listed;
  std::size_t choices = 0;
  for (const HelpOption &option : listed) {
    const std::string first_long = *std::find_if(option.names.begin(), option.names.end(),
                                                 [](const std::string &name) { return name.rfind("--", 0) == 0; });
    INFO("dtwc_cl option " << first_long);
    if (first_long == "--help" || first_long == "--version" || first_long == "--print-config")
      continue; // dtwc_cl's own, not Config keys

    const CLI::Option *bound = app.get_option_no_throw(first_long);
    REQUIRE(bound != nullptr);
    std::set<std::string> bound_names;
    for (const std::string &name : bound->get_snames()) bound_names.insert("-" + name);
    for (const std::string &name : bound->get_lnames()) bound_names.insert("--" + name);
    CHECK(bound_names == std::set<std::string>(option.names.begin(), option.names.end()));
    if (first_long == "--config") continue;

    const std::string key = first_long.substr(2);
    keys_listed.insert(key);
    for (const std::string &choice : option.choices) {
      const auto arrow = choice.find("->");
      const std::string spelling = choice.substr(0, arrow);
      const std::string canonical = arrow == std::string::npos ? choice : choice.substr(arrow + 2);
      INFO("choice '" << choice << "'");
      CHECK(values_of(dtwc::to_config_text(parse_args({ first_long, spelling }))).at(key) == canonical);
      ++choices;
    }
  }
  CHECK(choices >= 100);

  // dtwc_cl lists every key bind() has: since IF-2 S3 its options are bind()'s.
  std::set<std::string> unlisted;
  for (const auto &entry : values_of(dtwc::to_config_text(dtwc::Config{})))
    if (keys_listed.count(entry.first) == 0) unlisted.insert(entry.first);
  CHECK(unlisted.empty());
  fs::remove_all(scratch);
}

TEST_CASE("Config{} holds the defaults dtwc_cl reports", "[config][cli]")
{
  const fs::path scratch = fresh_scratch("dtwc_test_config_spellings_defaults");
  const fs::path input = conformance_dir() / "data" / "conformance_series.csv";
  // The "  Label: value" lines dtwc_cl -v prints before it reads the data.
  const auto echo = [&](const std::vector<std::string> &extra) {
    std::vector<std::string> argv{ cli_executable().string(), "-i", input.string(),
                                   "-o", (scratch / "out").string(), "-v" };
    argv.insert(argv.end(), extra.begin(), extra.end());
    const CommandResult result = run(argv, scratch);
    INFO("stdout:\n" << result.out << "\nstderr:\n" << result.err);
    REQUIRE(result.exit_code == 0);
    std::map<std::string, std::string> settings;
    for (const std::string &line : lines_of(result.out)) {
      if (line.rfind("=== System", 0) == 0) break;
      const auto colon = line.find(':');
      if (line.rfind("  ", 0) == 0 && colon != std::string::npos)
        settings[trimmed(line.substr(0, colon))] = trimmed(line.substr(colon + 1));
    }
    return settings;
  };
  const auto defaults = values_of(dtwc::to_config_text(dtwc::Config{}));

  const auto plain = echo({});
  CHECK(plain.at("Name") == defaults.at("name"));
  CHECK(plain.at("Clusters") == defaults.at("n-clusters"));
  CHECK(plain.at("Method") == defaults.at("method"));
  CHECK(plain.at("Band") == "full"); // how dtwc_cl prints band -1
  CHECK(defaults.at("band") == "-1");
  CHECK(plain.at("Metric") == defaults.at("metric"));
  CHECK(plain.at("Variant") == defaults.at("variant"));
  CHECK(plain.at("Missing") == defaults.at("missing-strategy"));
  CHECK(plain.at("MaxIter") == defaults.at("max-iter"));
  CHECK(plain.at("N-init") == defaults.at("n-init"));
  CHECK(plain.at("Device") == defaults.at("device"));
  CHECK(plain.at("Dtype") == defaults.at("dtype"));
  CHECK(plain.at("GPU Prec") == defaults.at("gpu-precision"));

  const auto clara = echo({ "-m", "clara" });
  CHECK(clara.at("CLARA sample_size") == "auto"); // how dtwc_cl prints sample size -1
  CHECK(defaults.at("sample-size") == "-1");
  CHECK(clara.at("CLARA n_samples") == defaults.at("n-samples"));
  CHECK(clara.at("CLARA seed") == defaults.at("seed"));

  const auto hierarchical = echo({ "-m", "hierarchical" });
  CHECK(hierarchical.at("Linkage") == defaults.at("linkage"));
  fs::remove_all(scratch);
}
