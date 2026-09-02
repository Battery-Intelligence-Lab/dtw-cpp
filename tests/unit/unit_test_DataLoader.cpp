/**
 * @file unit_test_DataLoader.cpp
 * @brief Unit test file for time DataLoader class
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 29 Dec 2023
 */

#include <DataLoader.hpp>
#include <settings.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <concepts>
#include <filesystem>
#include <fstream>
#include <system_error>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using Catch::Matchers::WithinAbs;

using namespace dtwc;

// This fixture deliberately compiles and executes the retained 1.x names.
// Keep those calls local and warning-suppressed; ordinary repository code uses
// the canonical 2.0 spellings.
#if defined(__clang__)
#  define DTWC_PUSH_NO_DEPRECATED \
    _Pragma("clang diagnostic push") \
    _Pragma("clang diagnostic ignored \"-Wdeprecated-declarations\"")
#  define DTWC_POP_NO_DEPRECATED _Pragma("clang diagnostic pop")
#elif defined(__GNUC__)
#  define DTWC_PUSH_NO_DEPRECATED \
    _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wdeprecated-declarations\"")
#  define DTWC_POP_NO_DEPRECATED _Pragma("GCC diagnostic pop")
#elif defined(_MSC_VER)
#  define DTWC_PUSH_NO_DEPRECATED \
    __pragma(warning(push)) __pragma(warning(disable : 4996))
#  define DTWC_POP_NO_DEPRECATED __pragma(warning(pop))
#else
#  define DTWC_PUSH_NO_DEPRECATED
#  define DTWC_POP_NO_DEPRECATED
#endif

#ifndef DTWC_TEST_DATA_DIR
#define DTWC_TEST_DATA_DIR "./data"
#endif

namespace {
fs::path dummy_data_path() { return fs::path{DTWC_TEST_DATA_DIR} / "dummy"; }

using loader_setter_t = DataLoader &(DataLoader::*)(int);
using path_setter_t = void (*)(const settings::fs::path &);
using cstring_path_setter_t = void (*)(const char *);

static_assert(std::same_as<
              decltype(static_cast<loader_setter_t>(&DataLoader::start_column)),
              loader_setter_t>);
static_assert(std::same_as<
              decltype(static_cast<loader_setter_t>(&DataLoader::start_row)),
              loader_setter_t>);

DTWC_PUSH_NO_DEPRECATED
static_assert(std::same_as<
              decltype(static_cast<loader_setter_t>(&DataLoader::startColumn)),
              loader_setter_t>);
static_assert(std::same_as<
              decltype(static_cast<loader_setter_t>(&DataLoader::startRow)),
              loader_setter_t>);
DTWC_POP_NO_DEPRECATED

static_assert(std::same_as<
              decltype(static_cast<path_setter_t>(
                &settings::paths::set_data_path)),
              path_setter_t>);
static_assert(std::same_as<
              decltype(static_cast<cstring_path_setter_t>(
                &settings::paths::set_data_path)),
              cstring_path_setter_t>);
static_assert(std::same_as<
              decltype(static_cast<path_setter_t>(
                &settings::paths::set_results_path)),
              path_setter_t>);
static_assert(std::same_as<
              decltype(static_cast<cstring_path_setter_t>(
                &settings::paths::set_results_path)),
              cstring_path_setter_t>);

DTWC_PUSH_NO_DEPRECATED
static_assert(std::same_as<
              decltype(static_cast<path_setter_t>(
                &settings::paths::setDataPath)),
              path_setter_t>);
static_assert(std::same_as<
              decltype(static_cast<cstring_path_setter_t>(
                &settings::paths::setDataPath)),
              cstring_path_setter_t>);
static_assert(std::same_as<
              decltype(static_cast<path_setter_t>(
                &settings::paths::setResultsPath)),
              path_setter_t>);
static_assert(std::same_as<
              decltype(static_cast<cstring_path_setter_t>(
                &settings::paths::setResultsPath)),
              cstring_path_setter_t>);
DTWC_POP_NO_DEPRECATED

class PathSettingsGuard
{
  settings::fs::path data_{ settings::paths::data };
  settings::fs::path results_{ settings::paths::results };

public:
  PathSettingsGuard() = default;
  PathSettingsGuard(const PathSettingsGuard &) = delete;
  PathSettingsGuard &operator=(const PathSettingsGuard &) = delete;
  ~PathSettingsGuard()
  {
    settings::paths::data = std::move(data_);
    settings::paths::results = std::move(results_);
  }
};
}

TEST_CASE("DataLoader class functionality", "[DataLoader]")
{
  SECTION("DataLoader Default Constructor")
  {
    // Test if the DataLoader is constructed with default parameters
    DataLoader loader;

    REQUIRE(loader.startColumn() == 0);
    REQUIRE(loader.startRow() == 0);
    REQUIRE(loader.n_data() == -1);
    REQUIRE(loader.delimiter() == ',');
    REQUIRE(loader.path() == ".");
  }

  SECTION("DataLoader Path Constructor")
  {
    // Test if the DataLoader is constructed with the correct path
    DataLoader loader1("test.csv"), loader2("test.tsv");

    REQUIRE(loader1.startColumn() == 0);
    REQUIRE(loader1.startRow() == 0);
    REQUIRE(loader1.n_data() == -1);
    REQUIRE(loader1.delimiter() == ',');
    REQUIRE(loader1.path() == "test.csv");

    REQUIRE(loader2.startColumn() == 0);
    REQUIRE(loader2.startRow() == 0);
    REQUIRE(loader2.n_data() == -1);
    REQUIRE(loader2.delimiter() == '\t');
    REQUIRE(loader2.path() == "test.tsv");
  }

  SECTION("DataLoader Path and Ndata Constructor")
  {
    // Test if DataLoader is constructed with correct path and Ndata
    int Ndata = 100;
    DataLoader loader1("test.csv", Ndata), loader2("test.tsv", Ndata);

    REQUIRE(loader1.startColumn() == 0);
    REQUIRE(loader1.startRow() == 0);
    REQUIRE(loader1.n_data() == Ndata);
    REQUIRE(loader1.delimiter() == ',');
    REQUIRE(loader1.path() == "test.csv");

    REQUIRE(loader2.startColumn() == 0);
    REQUIRE(loader2.startRow() == 0);
    REQUIRE(loader2.n_data() == Ndata);
    REQUIRE(loader2.delimiter() == '\t');
    REQUIRE(loader2.path() == "test.tsv");
  }

  SECTION("Method Chaining")
  {
    fs::path testPath = "test.csv";
    DataLoader loader;
    loader.start_column(1).start_row(3).n_data(100).path(testPath).verbosity(1);
    // Test if the method chaining correctly sets the properties
    REQUIRE(loader.startColumn() == 1);
    REQUIRE(loader.startRow() == 3);
    REQUIRE(loader.n_data() == 100);
    REQUIRE(loader.delimiter() == ',');
    REQUIRE(loader.path() == testPath);
    REQUIRE(loader.verbosity() == 1);
  }

  SECTION("Count directory mode")
  {
    DataLoader loader(dummy_data_path());
    REQUIRE(loader.count() == 25);
  }

  SECTION("Count directory mode with Ndata limit")
  {
    DataLoader loader(dummy_data_path(), 5);
    REQUIRE(loader.count() == 5);
  }

  SECTION("Count matches load for directory")
  {
    DataLoader loader(dummy_data_path());
    loader.verbosity(0);
    REQUIRE(loader.count() == loader.load().size());
  }

  SECTION("Count matches load with Ndata limit")
  {
    DataLoader loader(dummy_data_path(), 10);
    loader.verbosity(0);
    REQUIRE(loader.count() == 10);
    REQUIRE(loader.count() == loader.load().size());
  }

  SECTION("Count throws on non-existent file")
  {
    DataLoader loader("nonexistent_file.csv");
    REQUIRE_THROWS_AS(loader.count(), std::runtime_error);
  }
}

TEST_CASE("F21 canonical C++ loader and path names preserve legacy state",
          "[DataLoader][f21][public-api]")
{
  DataLoader canonical;
  DataLoader legacy;

  REQUIRE(&canonical.start_column(3) == &canonical);
  REQUIRE(canonical.startColumn() == 3);
  REQUIRE(canonical.startRow() == 0);
  REQUIRE(&canonical.start_row(7) == &canonical);
  REQUIRE(canonical.startColumn() == 3);
  REQUIRE(canonical.startRow() == 7);

  DTWC_PUSH_NO_DEPRECATED
  DataLoader *legacy_column_receiver = &legacy.startColumn(3);
  DTWC_POP_NO_DEPRECATED
  REQUIRE(legacy_column_receiver == &legacy);
  REQUIRE(legacy.startColumn() == 3);
  REQUIRE(legacy.startRow() == 0);

  DTWC_PUSH_NO_DEPRECATED
  DataLoader *legacy_row_receiver = &legacy.startRow(7);
  DTWC_POP_NO_DEPRECATED
  REQUIRE(legacy_row_receiver == &legacy);
  REQUIRE(legacy.startColumn() == 3);
  REQUIRE(legacy.startRow() == 7);
  REQUIRE(canonical.startColumn() == legacy.startColumn());
  REQUIRE(canonical.startRow() == legacy.startRow());

  REQUIRE(&canonical.start_column(11).start_row(13) == &canonical);
  REQUIRE(canonical.startColumn() == 11);
  REQUIRE(canonical.startRow() == 13);

  DTWC_PUSH_NO_DEPRECATED
  DataLoader *legacy_chain_receiver = &legacy.startColumn(11).startRow(13);
  DTWC_POP_NO_DEPRECATED
  REQUIRE(legacy_chain_receiver == &legacy);
  REQUIRE(legacy.startColumn() == 11);
  REQUIRE(legacy.startRow() == 13);
  REQUIRE(canonical.startColumn() == legacy.startColumn());
  REQUIRE(canonical.startRow() == legacy.startRow());

  const auto original_data = settings::paths::data;
  const auto original_results = settings::paths::results;
  {
    PathSettingsGuard restore_paths;
    const auto poison = [](const settings::fs::path &data,
                           const settings::fs::path &results) {
      settings::paths::data = data;
      settings::paths::results = results;
    };
    const auto require_paths = [](const settings::fs::path &data,
                                  const settings::fs::path &results) {
      REQUIRE(settings::paths::data == data);
      REQUIRE(settings::paths::results == results);
    };

    poison("poison-data-1", "poison-results-1");
    settings::paths::set_data_path(settings::fs::path{ "canonical-data-path" });
    require_paths("canonical-data-path", "poison-results-1");

    poison("poison-data-2", "poison-results-2");
    settings::paths::set_results_path(
      settings::fs::path{ "canonical-results-path" });
    require_paths("poison-data-2", "canonical-results-path");

    poison("poison-data-3", "poison-results-3");
    DTWC_PUSH_NO_DEPRECATED
    settings::paths::setDataPath(settings::fs::path{ "legacy-data-path" });
    DTWC_POP_NO_DEPRECATED
    require_paths("legacy-data-path", "poison-results-3");

    poison("poison-data-4", "poison-results-4");
    DTWC_PUSH_NO_DEPRECATED
    settings::paths::setResultsPath(
      settings::fs::path{ "legacy-results-path" });
    DTWC_POP_NO_DEPRECATED
    require_paths("poison-data-4", "legacy-results-path");

    poison("poison-data-5", "poison-results-5");
    {
      const std::string path = "canonical-cstring-data";
      settings::paths::set_data_path(path.c_str());
      require_paths("canonical-cstring-data", "poison-results-5");
    }
    REQUIRE(settings::paths::data == "canonical-cstring-data");

    poison("poison-data-6", "poison-results-6");
    {
      const std::string path = "canonical-cstring-results";
      settings::paths::set_results_path(path.c_str());
      require_paths("poison-data-6", "canonical-cstring-results");
    }
    REQUIRE(settings::paths::results == "canonical-cstring-results");

    poison("poison-data-7", "poison-results-7");
    {
      const std::string path = "legacy-cstring-data";
      DTWC_PUSH_NO_DEPRECATED
      settings::paths::setDataPath(path.c_str());
      DTWC_POP_NO_DEPRECATED
      require_paths("legacy-cstring-data", "poison-results-7");
    }
    REQUIRE(settings::paths::data == "legacy-cstring-data");

    poison("poison-data-8", "poison-results-8");
    {
      const std::string path = "legacy-cstring-results";
      DTWC_PUSH_NO_DEPRECATED
      settings::paths::setResultsPath(path.c_str());
      DTWC_POP_NO_DEPRECATED
      require_paths("poison-data-8", "legacy-cstring-results");
    }
    REQUIRE(settings::paths::results == "legacy-cstring-results");
  }

  REQUIRE(settings::paths::data == original_data);
  REQUIRE(settings::paths::results == original_results);

  std::cout
    << "F21_CPP_NAMES canonical=4/4 legacy=4/4 overloads=12/12 "
       "loader_state=22/22 path_state=16/16 cstring_copy=4/4 "
       "skips=0 verdict=PASS\n";
}

#ifdef DTWC_HAS_MMAP
TEST_CASE("default_series_cache_path is collision-free across concurrent loads",
          "[DataLoader][mmap][concurrency]")
{
  // Audit 2026-09-02 (B): the temp-path generator used a plain
  // `static std::size_t counter` incremented with `counter++`, so two
  // concurrent load_stored() calls could observe the same value and route two
  // different data sets to the same mapped .dtws file. One relaxed atomic
  // fetch_add per load call restores uniqueness at no per-series cost.
  constexpr int n_threads = 8;
  constexpr int per_thread = 256;
  std::vector<std::vector<std::string>> produced(n_threads);
  std::vector<std::thread> workers;
  workers.reserve(n_threads);
  for (int t = 0; t < n_threads; ++t)
    workers.emplace_back([&produced, t] {
      produced[static_cast<std::size_t>(t)].reserve(per_thread);
      for (int i = 0; i < per_thread; ++i)
        produced[static_cast<std::size_t>(t)].push_back(
          dtwc::detail::default_series_cache_path().string());
    });
  for (auto &worker : workers) worker.join();

  std::set<std::string> unique;
  for (const auto &batch : produced) unique.insert(batch.begin(), batch.end());
  CHECK(unique.size()
        == static_cast<std::size_t>(n_threads) * per_thread);

  // Cross-process evidence. The former "unique" component was
  // `reinterpret_cast<uintptr_t>(&counter)` — one static address, identical in
  // every process of the same image, so two processes generated the same first
  // temp path. This line is printed so two runs of this binary can be compared
  // directly; nothing in-process can observe another process's counter.
  std::cout << "DTWC_SERIES_CACHE_FIRST="
            << dtwc::detail::default_series_cache_path().filename().string()
            << '\n';
}
#endif // DTWC_HAS_MMAP

namespace {

/// RAII folder of N single-column CSV files with deterministic names.
struct SeriesFolder
{
  std::filesystem::path root;

  SeriesFolder(std::string_view name, int n_files)
    : root(std::filesystem::temp_directory_path() / std::string(name))
  {
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
    std::filesystem::create_directories(root);
    for (int i = 0; i < n_files; ++i) {
      std::ofstream out(root / ("s" + std::to_string(i) + ".csv"));
      out << (i + 1) << "\n" << (i + 2) << "\n";
    }
  }
  ~SeriesFolder()
  {
    std::error_code ec;
    std::filesystem::remove_all(root, ec);
  }
};

/// Capture everything written to std::cout while alive.
struct CoutCapture
{
  std::ostringstream sink;
  std::streambuf *previous{ std::cout.rdbuf(sink.rdbuf()) };
  ~CoutCapture() { std::cout.rdbuf(previous); }
};

} // namespace

TEST_CASE("Loaders agree on one Ndata contract and reject Ndata < -1",
          "[DataLoader][fileOperations][Ndata]")
{
  // Audit 2026-09-02 A11: for Ndata == 0 the three loaders disagreed --
  // count() returned 1, the folder load returned ALL series, and the batch
  // load returned 0. Ndata < -1 was never rejected anywhere. One predicate:
  // a negative Ndata means "all", otherwise stop at exactly Ndata series.
  SeriesFolder folder{ "dtwc_ndata_contract", 4 };

  for (const int requested : { 0, 1, 3, 4, 7, -1 }) {
    CAPTURE(requested);
    const std::size_t expect = requested < 0
      ? 4u : std::min<std::size_t>(static_cast<std::size_t>(requested), 4u);

    DataLoader loader;
    loader.path(folder.root).n_data(requested).verbosity(0);
    CHECK(loader.count() == expect);
    CHECK(loader.load().size() == expect);
  }

  DataLoader bad;
  CHECK_THROWS_AS(bad.n_data(-2), std::runtime_error);
}

TEST_CASE("DataLoader extension matching is case-insensitive and keeps an "
          "explicit delimiter", "[DataLoader][delimiter]")
{
  // Audit 2026-09-02 A13: the extension comparison was case-sensitive, so
  // "data.TSV" silently kept the ',' default; and path() overwrote a delimiter
  // the caller had just set explicitly.
  CHECK(DataLoader{}.path("a.TSV").delimiter() == '\t');
  CHECK(DataLoader{}.path("a.Csv").delimiter() == ',');
  CHECK(DataLoader{}.path("a.TXT").delimiter() == '\t');

  DataLoader explicit_delim;
  explicit_delim.delimiter('|').path("a.csv");
  CHECK(explicit_delim.delimiter() == '|');
}

TEST_CASE("Folder and batch loaders honour verbosity(0)",
          "[DataLoader][verbosity]")
{
  // Audit 2026-09-02: fileOperations printed "Reading data:" and
  // "N time-series data are read." unconditionally, so verbosity(0) (and
  // api.cpp's verbosity(0)) could not silence the loader.
  SeriesFolder folder{ "dtwc_verbosity_contract", 2 };
  {
    CoutCapture capture;
    DataLoader loader;
    loader.path(folder.root).verbosity(0).load();
    CHECK(capture.sink.str().empty());
  }

  const auto batch = folder.root / "batch.csv";
  {
    std::ofstream out(batch);
    out << "1,2,3\n4,5,6\n";
  }
  {
    CoutCapture capture;
    DataLoader loader;
    loader.path(batch).verbosity(0).load();
    CHECK(capture.sink.str().empty());
  }
}
