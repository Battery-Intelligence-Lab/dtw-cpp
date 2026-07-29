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

#include <concepts>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

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
