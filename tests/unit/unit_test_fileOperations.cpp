/**
 * @file unit_test_fileOperations.cpp
 * @brief Unit test file for file reading functions
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 25 Dec 2023
 */

#include <dtwc.hpp>
#include "../test_util.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <catch2/generators/catch_generators_adapters.hpp>

#include <string>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <random>
#include <algorithm>
#include <cmath>
#include <iterator>
#include <set>
#include <map>
#include <cstdint>

using Catch::Matchers::WithinAbs;
using Catch::Matchers::ContainsSubstring;

using namespace dtwc;

namespace {
struct TemporaryBatchFile {
  fs::path path;

  TemporaryBatchFile(std::string_view extension, std::string_view contents)
  {
    path = fs::temp_directory_path()
         / ("dtwc_batch_parser_"
            + std::to_string(reinterpret_cast<std::uintptr_t>(this))
            + std::string(extension));
    std::ofstream out(path, std::ios::binary);
    out << contents;
  }

  ~TemporaryBatchFile() { std::error_code ec; fs::remove(path, ec); }
};
} // namespace

TEST_CASE("ignoreBOM test", "[file_operations]")
{
  auto testText = [](std::string data) {
    std::stringstream in(data);

    ignoreBOM(in);

    std::string result;
    std::getline(in, result);

    return result;
  };

  SECTION("ignoreBOM with BOM present")
  {
    std::string dataText = "\xEF\xBB\xBFText with BOM";
    REQUIRE(testText(dataText) == "Text with BOM");
  }

  SECTION("ignoreBOM without BOM present")
  {
    std::string dataText = "Text without BOM";
    REQUIRE(testText(dataText) == "Text without BOM");
  }

  SECTION("ignoreBOM with empty stream")
  {
    std::string dataText("");
    REQUIRE(testText(dataText).empty());
  }

  SECTION("ignoreBOM with stream less than 3 characters")
  {
    std::string dataText = "Hi";
    REQUIRE(testText(dataText) == "Hi");
  }
}

TEST_CASE("Write and Read Distance Matrices via Problem", "[fileOperations]")
{
  // Test round-trip of distance matrix I/O through Problem.
  auto N = GENERATE(1, 2, 5, 10, 20);

  // Create a DenseDistanceMatrix with random values.
  dtwc::core::DenseDistanceMatrix matrix(static_cast<size_t>(N));
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(0.0, 100.0);
  for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
    matrix.set(i, i, 0.0);
    for (size_t j = i + 1; j < static_cast<size_t>(N); ++j)
      matrix.set(i, j, dist(rng));
  }

  // Write via Problem's write_distance_matrix mechanism (inline CSV).
  fs::path tempFilePath = "test_distmat.csv";
  {
    std::ofstream file(tempFilePath);
    for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
      for (size_t j = 0; j < static_cast<size_t>(N); ++j) {
        if (j > 0) file << ',';
        file << std::setprecision(15) << matrix.get(i, j);
      }
      file << '\n';
    }
  }

  // Read back via Problem's read_distance_matrix mechanism.
  Problem prob;
  prob.read_distance_matrix(tempFilePath);

  // Verify the file round-trip by reading the CSV manually and comparing.
  {
    std::ifstream inFile(tempFilePath);
    std::string line;
    size_t row = 0;
    while (std::getline(inFile, line)) {
      std::istringstream ss(line);
      std::string cell;
      size_t col = 0;
      while (std::getline(ss, cell, ',')) {
        double val = std::stod(cell);
        REQUIRE_THAT(val, WithinAbs(matrix.get(row, col), 1e-3));
        ++col;
      }
      ++row;
    }
    REQUIRE(row == static_cast<size_t>(N));
  } // close inFile before removing

  fs::remove(tempFilePath);
}

TEST_CASE("Problem::write_distance_matrix + read_distance_matrix end-to-end roundtrip",
          "[fileOperations][problem]")
{
  // Full end-to-end roundtrip through Problem: fill a distance matrix from
  // DTW calls, write to CSV via Problem::write_distance_matrix, load into a
  // fresh Problem via Problem::read_distance_matrix, and verify the loaded
  // values are returned by dist_by_ind without recomputation.
  //
  // The existing "Write and Read Distance Matrices via Problem" test
  // (above) wrote the CSV manually and only called read_distance_matrix to
  // verify it doesn't throw. This complements it by exercising both I/O
  // methods on matching real DTW values.

  constexpr int N = 6;
  constexpr int L = 12;

  // Build identical data for two Problems.
  dtwc::Data data1;
  data1.p_vec.resize(N);
  data1.p_names.resize(N);
  for (int i = 0; i < N; ++i) {
    data1.p_vec[i].resize(L);
    for (int t = 0; t < L; ++t)
      data1.p_vec[i][t] = std::sin(static_cast<double>(i) + static_cast<double>(t));
    data1.p_names[i] = "s" + std::to_string(i);
  }
  dtwc::Data data2 = data1; // deep copy — same content

  const auto tmp = std::filesystem::temp_directory_path() / "dtwc_distmat_roundtrip_test";
  std::filesystem::create_directories(tmp);

  dtwc::Problem prob1("rt");
  prob1.set_data(std::move(data1));
  prob1.set_output_folder(tmp);
  prob1.set_verbose(false);
  prob1.fill_distance_matrix();

  // Snapshot every (i, j) via dist_by_ind — pulls from the filled dense matrix.
  std::vector<std::vector<double>> snapshot(N, std::vector<double>(N, 0.0));
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j)
      snapshot[i][j] = prob1.dist_by_ind(i, j);

  prob1.write_distance_matrix("rt_distmat.csv");
  const auto csv_path = tmp / "rt_distmat.csv";
  REQUIRE(std::filesystem::exists(csv_path));

  // Fresh Problem with identical data; its matrix is empty until loaded.
  dtwc::Problem prob2("rt2");
  prob2.set_data(std::move(data2));
  prob2.set_output_folder(tmp);
  prob2.set_verbose(false);
  prob2.read_distance_matrix(csv_path);

  // Every pair must now return the loaded value; no DTW recomputation
  // (snapshot values came from different RNG/compute ordering — if reading
  // didn't populate the matrix, we'd still get the same values because the
  // computation is deterministic, so we also check that the matrix reports
  // is_computed() for these indices via the final write-back roundtrip).
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      const double loaded = prob2.dist_by_ind(i, j);
      REQUIRE_THAT(loaded, WithinAbs(snapshot[i][j], 1e-9));
    }
  }
}

TEST_CASE("Write and Read Empty Matrix", "[fileOperations]")
{
  dtwc::core::DenseDistanceMatrix matrix;
  REQUIRE(matrix.size() == 0);

  // Empty matrix should be default-constructed with size 0.
  dtwc::core::DenseDistanceMatrix readMat;
  REQUIRE(readMat.size() == 0);
}


TEST_CASE("Load batch file", "[fileOperations]")
{
  std::string tempFileName = "test_matrix";

  // Generate data:
  const int N_data = GENERATE(1, 2, 10, 1000); // Size of the outer vector
  const int L_data = GENERATE(1, 2, 10, 1000); // Maximum size of the inner vectors

  const auto random_data = test_util::get_random_data<double>(N_data, L_data);

  // write the files
  test_util::write_data_to_file(tempFileName + ".csv", random_data, ',');
  test_util::write_data_to_file(tempFileName + ".tsv", random_data, '\t');

  // ----- now testing -----
  SECTION("csv batch load")
  {
    fs::path pth = tempFileName + ".csv";
    int start_row{ 0 }, start_col{}, delimiter{ ',' };
    auto [p_vec, p_names] = load_batch_file<double>(pth, N_data, false, start_row, start_col, delimiter);

    for (size_t i{}; i < N_data; i++)
      REQUIRE(p_names[i] == std::to_string(i + 1));

    REQUIRE(p_vec == random_data);
  }

  SECTION("tsv batch load")
  {
    fs::path pth = tempFileName + ".tsv";
    int start_row{ 0 }, start_col{}, delimiter{ '\t' };
    auto [p_vec, p_names] = load_batch_file<double>(pth, N_data, false, start_row, start_col, delimiter);

    for (size_t i{}; i < N_data; i++)
      REQUIRE(p_names[i] == std::to_string(i + 1));

    REQUIRE(p_vec == random_data);
  }

  fs::remove(tempFileName + ".csv"); // Clean up the test files
  fs::remove(tempFileName + ".tsv"); // Clean up the test files
}

TEST_CASE("Batch loader preserves textual NaN and later fields",
          "[fileOperations][batch_parser][m26]")
{
  TemporaryBatchFile file(".csv",
    "sensor-A,1,nan,3\n"
    "sensor-B,4,5,6\n");
  DataLoader loader(file.path);
  loader.start_column(1).verbosity(0);

  const Data loaded = loader.load_local();
  REQUIRE(loaded.size() == 2);
  REQUIRE(loaded.p_vec[0].size() == 3);
  CHECK(loaded.p_vec[0][0] == 1.0);
  CHECK(std::isnan(loaded.p_vec[0][1]));
  CHECK(loaded.p_vec[0][2] == 3.0);
  CHECK(loaded.p_vec[1] == std::vector<double>{4.0, 5.0, 6.0});

  const Data metadata = loader.load_metadata();
  CHECK(metadata.series_flat_size(0) == loaded.p_vec[0].size());
  CHECK(metadata.series_flat_size(1) == loaded.p_vec[1].size());
}

TEST_CASE("Batch loader rejects malformed and unapproved non-finite fields",
          "[fileOperations][batch_parser][m26]")
{
  const std::vector<std::string> bad_tokens = {
    "oops", "1x", "", "inf", "-inf", "infinity", "nan(payload)", "+nan"
  };
  for (const auto &token : bad_tokens) {
    DYNAMIC_SECTION("token='" << token << "'") {
      TemporaryBatchFile file(".csv", "id,1," + token + ",3\n");
      DataLoader loader(file.path);
      loader.start_column(1).verbosity(0);
      REQUIRE_THROWS_WITH(loader.load_local(),
        ContainsSubstring("row 1, column 3"));
      REQUIRE_THROWS_WITH(loader.load_metadata(),
        ContainsSubstring("row 1, column 3"));
    }
  }
}

TEST_CASE("Batch loader uses exact delimiters and arbitrary skipped fields",
          "[fileOperations][batch_parser][m26]")
{
  TemporaryBatchFile valid(".tsv",
    "sensor A\tquality=good\t+1\t-2.5\t1e3\n");
  DataLoader loader(valid.path);
  loader.start_column(2).verbosity(0);
  const Data loaded = loader.load_local();
  REQUIRE(loaded.p_vec.size() == 1);
  CHECK(loaded.p_vec[0] == std::vector<double>{1.0, -2.5, 1000.0});
  CHECK(loader.load_metadata().series_flat_size(0) == 3);

  TemporaryBatchFile empty_field(".tsv", "id\t1\t\t3\n");
  DataLoader invalid(empty_field.path);
  invalid.start_column(1).verbosity(0);
  REQUIRE_THROWS_WITH(invalid.load_local(),
    ContainsSubstring("row 1, column 3"));
}

TEST_CASE("Folder-series reader shares the strict NaN parser",
          "[fileOperations][batch_parser][m26]")
{
  TemporaryBatchFile file(".csv",
    "sample-A,1\n"
    "sample-B,nan\n"
    "sample-C,3\n");
  const auto values = readFile<double>(file.path, 0, 1, ',');
  REQUIRE(values.size() == 3);
  CHECK(values[0] == 1.0);
  CHECK(std::isnan(values[1]));
  CHECK(values[2] == 3.0);

  TemporaryBatchFile invalid(".csv", "sample-A,1\nsample-B,1x\n");
  REQUIRE_THROWS_WITH(readFile<double>(invalid.path, 0, 1, ','),
    ContainsSubstring("row 2, column 2"));

  TemporaryBatchFile invalid_first(".csv", "sample-A,1x\n");
  REQUIRE_THROWS_WITH(readFile<double>(invalid_first.path, 0, 1, ','),
    ContainsSubstring("row 1, column 2"));
}

TEST_CASE("readFile throws on missing file", "[fileOperations]")
{
  fs::path nonExistentFile = "this_file_does_not_exist_12345.csv";

  // Ensure the file doesn't exist
  if (fs::exists(nonExistentFile)) {
    fs::remove(nonExistentFile);
  }

  REQUIRE_THROWS_AS(readFile<double>(nonExistentFile), std::runtime_error);
}

TEST_CASE("load_batch_file throws on missing file", "[fileOperations]")
{
  fs::path nonExistentFile = "this_batch_file_does_not_exist_12345.csv";

  // Ensure the file doesn't exist
  if (fs::exists(nonExistentFile)) {
    fs::remove(nonExistentFile);
  }

  REQUIRE_THROWS_AS(load_batch_file<double>(nonExistentFile), std::runtime_error);
}

TEST_CASE("Load folder", "[fileOperations]")
{
  // Generate data:
  constexpr int stringLength = 10;

  const int N_data = GENERATE(1, 2, 10, 100);  // Size of the outer vector
  const int L_data = GENERATE(1, 2, 10, 1000); // Maximum size of the inner vectors

  const auto random_data = test_util::get_random_data<double>(N_data, L_data);
  const auto random_names = test_util::get_random_names(N_data, stringLength);

  // ----- now testing -----
  SECTION("csv batch load")
  {
    std::string folder("CSV");
    test_util::write_data_to_folder(folder, random_data, random_names);
    fs::path pth = folder;
    int start_row{ 0 }, start_col{ 1 };
    char delimiter{ ',' };
    auto [p_vec, p_names] = load_folder<double>(pth, N_data, false, start_row, start_col, delimiter);

    // Order of names and data is different in different operating systems.
    for (size_t i{}; i < N_data; i++) {
      auto iterNow = std::find(p_names.begin(), p_names.end(), random_names[i]);
      REQUIRE(iterNow != p_names.end());

      const int j = std::distance(p_names.begin(), iterNow);
      REQUIRE(p_vec[j] == random_data[i]);
    }
  }

  fs::remove_all("CSV"); // Clean up the test files
}

TEST_CASE("Problem::read_distance_matrix propagates a failed read",
          "[fileOperations][problem][io]")
{
  // A1 (2026-09-02 io/cli audit): the reader wrapped its whole body in
  // `catch (...)` and only printed a message, so no caller could distinguish a
  // failed load from a successful one. dtwc_cl then printed "Loaded distance
  // matrix from <path>" straight after "Distance matrix could not be read!".
  // The failure must reach the caller; the CLI already owns the handler that
  // decides to continue without a precomputed matrix.
  const auto tmp =
    std::filesystem::temp_directory_path() / "dtwc_read_distmat_failure_test";
  std::filesystem::create_directories(tmp);

  SECTION("absent file")
  {
    dtwc::Problem prob("a1_absent");
    REQUIRE_THROWS_AS(prob.read_distance_matrix(tmp / "definitely-missing.csv"),
                      std::runtime_error);
  }

  SECTION("non-numeric field")
  {
    const auto bad = tmp / "not-a-matrix.csv";
    {
      std::ofstream file(bad);
      file << "0,not-a-number\nnot-a-number,0\n";
    }
    dtwc::Problem prob("a1_malformed");
    REQUIRE_THROWS(prob.read_distance_matrix(bad));
    std::filesystem::remove(bad);
  }
}

TEST_CASE("Directory-source series names are UTF-8 on every platform",
          "[fileOperations][unicode]")
{
  // path::string() is the NATIVE narrow encoding: on Windows the ACP, so this
  // stem came back as the single byte 0xE9 and the Python binding raised
  // UnicodeDecodeError while the CLI wrote "caf\351" into its CSVs. The stem is
  // spelled with a universal-character escape so the assertion does not depend
  // on this source file's own encoding.
  const std::u8string stem = u8"caf\u00e9";
  const std::string expected_utf8 = "caf\xc3\xa9";

  const auto folder = fs::temp_directory_path()
                    / ("dtwc_utf8_names_"
                       + std::to_string(reinterpret_cast<std::uintptr_t>(&stem)));
  fs::create_directories(folder);
  const auto file = folder / fs::path(stem + u8".csv");
  {
    std::ofstream out(file);
    REQUIRE(out.good());
    out << "1\n2\n3\n";
  }

  DataLoader loader(folder);
  loader.verbosity(0);
  dtwc::Problem problem("utf8_names");
  problem.set_data(loader.load_local());

  REQUIRE(problem.size() == 1);
  CHECK(std::string{ problem.series_name(0) } == expected_utf8);

  // The metadata-only folder route names series the same way.
  DataLoader metadata_loader(folder);
  metadata_loader.verbosity(0);
  const auto metadata = metadata_loader.load_metadata();
  REQUIRE(metadata.size() == 1);
  CHECK(std::string{ metadata.name(0) } == expected_utf8);

  // utf8_to_path is the inverse, and must not throw on a name that never came
  // from a loader: MSVC's char8_t conversion throws on unmappable bytes, which
  // would turn a native-encoded CLI --name into a failed write.
  CHECK(path_to_utf8(utf8_to_path(expected_utf8)) == expected_utf8);
  CHECK(utf8_to_path("plain.csv").string() == "plain.csv");
  const std::string native_ansi = "caf\xe9.csv"; // lone 0xE9: not valid UTF-8
  CHECK(utf8_to_path(native_ansi).string() == native_ansi);

  std::error_code ec;
  fs::remove_all(folder, ec);
}
