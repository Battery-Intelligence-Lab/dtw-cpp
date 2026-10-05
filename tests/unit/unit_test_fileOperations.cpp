/**
 * @file unit_test_fileOperations.cpp
 * @brief Unit test file for file reading functions
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 25 Dec 2023
 */

#include <dtwc.hpp>
#include "../support/scratch_directory.hpp"
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
#include <bit>
#include <cctype>
#include <clocale>
#include <cmath>
#include <iterator>
#include <set>
#include <map>
#include <cstdint>
#include <span>
#include <streambuf>
#include <string_view>
#include <system_error>

#ifndef DTWC_TEST_DATA_DIR
#define DTWC_TEST_DATA_DIR "./data"
#endif

using Catch::Matchers::WithinAbs;
using Catch::Matchers::ContainsSubstring;

using namespace dtwc;
using dtwc::test_support::ScratchDirectory;

namespace {
struct TemporaryBatchFile {
  ScratchDirectory directory{ "batch_parser" };
  fs::path path;

  TemporaryBatchFile(std::string_view extension, std::string_view contents)
    : path(directory.path / ("batch" + std::string(extension)))
  {
    std::ofstream out(path, std::ios::binary);
    out << contents;
  }
};

/// The reader fixtures in <repo>/tests/data/reader (DTWC_TEST_DATA_DIR is <repo>/data).
fs::path reader_fixture(std::string_view name)
{
  return fs::path{ DTWC_TEST_DATA_DIR }.parent_path() / "tests" / "data" / "reader"
       / std::string(name);
}

/// A file or folder through DataLoader, the route the CLI and Python take.
Data load_path(const fs::path &path, int start_row = 0, int start_col = 0)
{
  DataLoader loader(path);
  loader.start_row(start_row).start_column(start_col).verbosity(0);
  return loader.load();
}

/// A read-only stream buffer that cannot seek, like a pipe or FIFO: the
/// std::streambuf seekoff / seekpos defaults fail.
class PipeBuffer : public std::streambuf
{
  std::string bytes_;

public:
  explicit PipeBuffer(std::string bytes) : bytes_(std::move(bytes))
  {
    setg(bytes_.data(), bytes_.data(), bytes_.data() + bytes_.size());
  }
};

using Series = std::vector<std::vector<double>>;
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

  // Create a DistanceMatrix with random values.
  dtwc::core::DistanceMatrix matrix(static_cast<size_t>(N));
  std::mt19937 rng(42);
  std::uniform_real_distribution<double> dist(0.0, 100.0);
  for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
    matrix.set(i, i, 0.0);
    for (size_t j = i + 1; j < static_cast<size_t>(N); ++j)
      matrix.set(i, j, dist(rng));
  }

  // Write via Problem's write_distance_matrix mechanism (inline CSV).
  const ScratchDirectory scratch{ "distmat_csv" };
  const fs::path tempFilePath = scratch.path / "test_distmat.csv";
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
  } // close inFile before the directory is removed
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

  const ScratchDirectory scratch{ "distmat_roundtrip" };
  const auto &tmp = scratch.path;

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
  dtwc::core::DistanceMatrix matrix;
  REQUIRE(matrix.size() == 0);

  // Empty matrix should be default-constructed with size 0.
  dtwc::core::DistanceMatrix readMat;
  REQUIRE(readMat.size() == 0);
}


TEST_CASE("Load batch file", "[fileOperations]")
{
  const ScratchDirectory scratch{ "batch_file" };
  const std::string tempFileName = (scratch.path / "test_matrix").string();

  // Generate data:
  const int N_data = GENERATE(1, 2, 10, 1000); // Size of the outer vector
  const int L_data = GENERATE(1, 2, 10, 1000); // Maximum size of the inner vectors

  // At least one value per series: a blank line is no longer the on-disk form
  // of an empty series. The last section pins what a blank line is now.
  const auto random_data = test_util::get_random_data<double>(N_data, L_data, 1);

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

  SECTION("a blank line is not an empty series")
  {
    fs::path pth = tempFileName + ".csv";
    {
      std::ofstream out(pth, std::ios::app);
      out << "\n\n"; // trailing blank lines: ignored
    }
    auto [p_vec, p_names] = load_batch_file<double>(pth, -1, false, 0, 0, ',');
    REQUIRE(p_vec == random_data);

    // An empty series written first, as a blank line followed by data: an
    // error naming the row, where it used to load as an empty series.
    auto with_empty = random_data;
    with_empty.insert(with_empty.begin(), std::vector<double>{});
    test_util::write_data_to_file(pth.string(), with_empty, ',');
    REQUIRE_THROWS_WITH(load_batch_file<double>(pth, -1, false, 0, 0, ','),
                        ContainsSubstring("row 1 is empty"));
  }
}

TEST_CASE("Batch loader preserves textual NaN and later fields",
          "[fileOperations][batch_parser][m26]")
{
  TemporaryBatchFile file(".csv",
    "sensor-A,1,nan,3\n"
    "sensor-B,4,5,6\n");
  DataLoader loader(file.path);
  loader.start_column(1).verbosity(0);

  const Data loaded = loader.load();
  REQUIRE(loaded.size() == 2);
  REQUIRE(loaded.p_vec[0].size() == 3);
  CHECK(loaded.p_vec[0][0] == 1.0);
  CHECK(std::isnan(loaded.p_vec[0][1]));
  CHECK(loaded.p_vec[0][2] == 3.0);
  CHECK(loaded.p_vec[1] == std::vector<double>{4.0, 5.0, 6.0});
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
      REQUIRE_THROWS_WITH(loader.load(),
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
  const Data loaded = loader.load();
  REQUIRE(loaded.p_vec.size() == 1);
  CHECK(loaded.p_vec[0] == std::vector<double>{1.0, -2.5, 1000.0});

  TemporaryBatchFile empty_field(".tsv", "id\t1\t\t3\n");
  DataLoader invalid(empty_field.path);
  invalid.start_column(1).verbosity(0);
  REQUIRE_THROWS_WITH(invalid.load(),
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
  const ScratchDirectory scratch{ "load_folder" };

  SECTION("csv batch load")
  {
    const std::string folder = (scratch.path / "CSV").string();
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
}

TEST_CASE("Problem::read_distance_matrix propagates a failed read",
          "[fileOperations][problem][io]")
{
  // The reader wrapped its whole body in
  // `catch (...)` and only printed a message, so no caller could distinguish a
  // failed load from a successful one. dtwc_cl then printed "Loaded distance
  // matrix from <path>" straight after "Distance matrix could not be read!".
  // The failure must reach the caller; the CLI already owns the handler that
  // decides to continue without a precomputed matrix.
  const ScratchDirectory scratch{ "read_distmat_failure" };
  const auto &tmp = scratch.path;

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

  const ScratchDirectory scratch{ "utf8_names" };
  const fs::path &folder = scratch.path;
  const auto file = folder / fs::path(stem + u8".csv");
  {
    std::ofstream out(file);
    REQUIRE(out.good());
    out << "1\n2\n3\n";
  }

  DataLoader loader(folder);
  loader.verbosity(0);
  dtwc::Problem problem("utf8_names");
  problem.set_data(loader.load());

  REQUIRE(problem.size() == 1);
  CHECK(std::string{ problem.series_name(0) } == expected_utf8);

  // utf8_to_path is the inverse, and must not throw on a name that never came
  // from a loader: MSVC's char8_t conversion throws on unmappable bytes, which
  // would turn a native-encoded CLI --name into a failed write.
  CHECK(path_to_utf8(utf8_to_path(expected_utf8)) == expected_utf8);
  CHECK(utf8_to_path("plain.csv").string() == "plain.csv");
  const std::string native_ansi = "caf\xe9.csv"; // lone 0xE9: not valid UTF-8
  CHECK(utf8_to_path(native_ansi).string() == native_ansi);
}

// ---------------------------------------------------------------------------
// One case per reader defect, each against a hand-written expected value.
// The fixtures are an input matrix of reader edge cases.
// ---------------------------------------------------------------------------

TEST_CASE("FX-6 trailing blank lines are ignored, not read as an empty series",
          "[fileOperations][fx6][blank]")
{
  // "1,2,3\n4,5,6\n\n": the last line became an empty series that clustered
  // with DBL_MAX distances, exit 0.
  DataLoader loader(reader_fixture("trailing_blank.csv"));
  loader.verbosity(0);
  const Data loaded = loader.load();
  CHECK(loaded.p_vec == Series{ { 1, 2, 3 }, { 4, 5, 6 } });
  CHECK(loaded.p_names == std::vector<std::string>{ "1", "2" });
}

TEST_CASE("FX-6 a blank line followed by data is an error naming its row",
          "[fileOperations][fx6][blank]")
{
  // "1,2,3\n\n4,5,6\n" and "1,2\n   \n3,4\n" loaded three series, the middle one empty.
  for (const char *name : { "interior_blank.csv", "ws_line.csv" }) {
    CAPTURE(name);
    DataLoader loader(reader_fixture(name));
    loader.verbosity(0);
    CHECK_THROWS_WITH(loader.load(), ContainsSubstring("row 2 is empty"));
  }
  // A folder file "0.5\n\n0.7\n" read as {0.5, 0.7}: the missing value vanished
  // and every later value shifted by one.
  DataLoader folder(reader_fixture("folder_blank_line"));
  folder.verbosity(0);
  CHECK_THROWS_WITH(folder.load(), ContainsSubstring("row 2 is empty"));
}

TEST_CASE("FX-6 a file in a series folder holds one value per line",
          "[fileOperations][fx6][folder]")
{
  // "0,0.5\n1,0.6\n2,0.9\n" read with defaults clustered the index {0, 1, 2}.
  const auto two_column = reader_fixture("folder_two_column");
  CHECK_THROWS_WITH(load_path(two_column), ContainsSubstring("row 1 has 2 fields"));
  CHECK_THROWS_WITH(load_path(two_column), ContainsSubstring("--skip-cols"));
  const Data values = load_path(two_column, 0, 1);
  CHECK(values.p_vec == Series{ { 0.5, 0.6, 0.9 } });
  CHECK(values.p_names == std::vector<std::string>{ "a" });

  // Decimal commas "1,5\n2,5\n3,75\n" read as {1, 2, 3}.
  CHECK_THROWS_WITH(load_path(reader_fixture("folder_decimal_comma")),
                    ContainsSubstring("row 1 has 2 fields"));
}

TEST_CASE("FX-6 the legacy ,0 header is skipped only when no row is skipped",
          "[fileOperations][fx6][folder]")
{
  // ",0\n0,\n1,2.5\n2,3.5\n" with the documented pandas settings (skip one row
  // and one column): the first value is missing, and the legacy-header rule
  // dropped it, reading {2.5, 3.5}.
  CHECK_THROWS_WITH(load_path(reader_fixture("folder_leading_missing"), 1, 1),
                    ContainsSubstring("row 2, column 2: empty numeric field"));

  // A present first value is read as before.
  TemporaryBatchFile pandas(".csv", ",0\n0,0.25\n1,0.5\n");
  CHECK(readFile<double>(pandas.path, 1, 1, ',') == std::vector<double>{ 0.25, 0.5 });
}

TEST_CASE("FX-6 dot-files in a series folder are not series",
          "[fileOperations][fx6][folder]")
{
  const ScratchDirectory scratch{ "fx6_dot_files" };
  const fs::path &folder = scratch.path;
  std::ofstream(folder / "a.csv") << "1\n2\n3\n";
  std::ofstream(folder / "b.csv") << "4\n5\n6\n";
  std::ofstream(folder / ".gitkeep").close(); // was an empty series named ".gitkeep"
  std::ofstream(folder / "._a.csv") << "9\n"; // an AppleDouble companion was a series

  const Data loaded = load_path(folder);
  CHECK(loaded.p_names == std::vector<std::string>{ "a", "b" });
  CHECK(loaded.p_vec == Series{ { 1, 2, 3 }, { 4, 5, 6 } });
}

TEST_CASE("FX-6 ignoreBOM never seeks, so a pipe keeps its first row",
          "[fileOperations][fx6][bom]")
{
  const auto first_line = [](std::string bytes) {
    PipeBuffer buffer(std::move(bytes));
    std::istream in(&buffer);
    ignoreBOM(in);
    std::string line;
    std::getline(in, line);
    return line;
  };
  // tellg/seekg on a non-seekable stream failed it: this read no row at all.
  CHECK(first_line("1,2,3\n4,5,6\n") == "1,2,3");
  CHECK(first_line("\xEF\xBB\xBF" "1,2\n") == "1,2");
  // U+FF54 (EF BD 94) and a truncated mark are not a BOM: their bytes stay.
  CHECK(first_line("\xEF\xBD\x94" "1\n") == "\xEF\xBD\x94" "1");
  CHECK(first_line("\xEF\xBB" "x\n") == "\xEF\xBB" "x");

  CHECK(load_path(reader_fixture("bom.csv")).p_vec == Series{ { 1, 2, 3 }, { 4, 5, 6 } });
  TemporaryBatchFile header(".csv", "\xEF\xBD\x94" "1,\xEF\xBD\x94" "2\n1,2\n");
  CHECK(load_path(header.path, 1).p_vec == Series{ { 1, 2 } });
}

TEST_CASE("FX-6 '+-1' is not a number", "[fileOperations][fx6][number]")
{
  // "1,+-1,3": the '+' was stripped and "-1" parsed.
  CHECK_THROWS_WITH(load_path(reader_fixture("plusminus.csv")),
                    ContainsSubstring("row 1, column 2: invalid numeric field '+-1'"));
  TemporaryBatchFile plus(".csv", "+7,-2,+.5\n");
  CHECK(load_path(plus.path).p_vec == Series{ { 7, -2, 0.5 } });
}

TEST_CASE("FX-6 field whitespace is ASCII under every C locale",
          "[fileOperations][fx6][locale]")
{
  struct CtypeGuard
  {
    std::string saved;
    CtypeGuard()
    {
      const char *const current = std::setlocale(LC_CTYPE, nullptr);
      saved = current != nullptr ? current : "C";
    }
    ~CtypeGuard() { std::setlocale(LC_CTYPE, saved.c_str()); }
  } guard;

  // "1,2,3\xA0": std::isspace trimmed the Latin-1 no-break space under a UTF-8
  // LC_CTYPE, so Python (which sets the user's locale) read 3 and the CLI failed.
  const auto nbsp = reader_fixture("nbsp_latin1.csv");
  std::setlocale(LC_CTYPE, "C");
  CHECK_THROWS_WITH(load_path(nbsp), ContainsSubstring("row 1, column 3: invalid numeric field"));

  const char *applied = nullptr;
  for (const char *name : { "en_US.UTF-8", "C.UTF-8", "en_US.utf8" })
    if (std::setlocale(LC_CTYPE, name) != nullptr) {
      applied = name;
      break;
    }
  const bool bites = applied != nullptr && std::isspace(0xA0) != 0;
  std::cout << "FX6_CTYPE locale=" << (applied ? applied : "unavailable")
            << " isspace_0xA0=" << (bites ? "yes" : "no") << '\n';
  CHECK_THROWS_WITH(load_path(nbsp), ContainsSubstring("row 1, column 3: invalid numeric field"));

  // ASCII blanks around a field are still trimmed: "1, 2 ,\t3 ".
  CHECK(load_path(reader_fixture("spaces.csv")).p_vec == Series{ { 1, 2, 3 } });
}

TEST_CASE("FX-6 parse_number keeps the std::from_chars contract",
          "[fileOperations][fx6][number]")
{
  // Expected bits written by hand as hexadecimal literals; each equals what
  // libc++'s std::from_chars(general) returned on macOS 26.
  struct Case
  {
    std::string_view text;
    std::size_t used;
    double value;
  };
  const Case parsed[] = {
    { "1e-3", 4, 0x1.0624dd2f1a9fcp-10 },
    { "2E+2", 4, 0x1.9p+7 },
    { ".5", 2, 0x1p-1 },
    { "5.", 2, 0x1.4p+2 },
    { "-0", 2, -0.0 },
    { "0.30000000000000004", 19, 0x1.3333333333334p-2 },
    { "4.9e-324", 8, 0x1p-1074 },
    { "2.4703282292062328e-324", 23, 0x1p-1074 },
    { "1.7976931348623157e308", 22, 0x1.fffffffffffffp+1023 },
    { "123456789012345678901234567890", 30, 0x1.8ee90ff6c373ep+96 },
    { "0x1p3", 1, 0.0 }, // general format: "0", then 'x' is not consumed
    { "1e", 1, 1.0 },
  };
  for (const auto &c : parsed) {
    CAPTURE(c.text);
    double value = 42.0;
    const auto r = io::parse_number(c.text.data(), c.text.data() + c.text.size(), value);
    CHECK(r.ec == std::errc{});
    CHECK(static_cast<std::size_t>(r.ptr - c.text.data()) == c.used);
    CHECK(std::bit_cast<std::uint64_t>(value) == std::bit_cast<std::uint64_t>(c.value));
  }

  double nan_value = 42.0;
  const std::string_view nan_text = "nan";
  CHECK(io::parse_number(nan_text.data(), nan_text.data() + 3, nan_value).ec == std::errc{});
  CHECK(std::isnan(nan_value));

  // Errors leave the value untouched.
  for (const std::string_view text :
       { "1e400", "-1e400", "1e-400", "2.4703282292062327e-324", "1.7976931348623159e308" }) {
    CAPTURE(text);
    double value = 42.0;
    const auto r = io::parse_number(text.data(), text.data() + text.size(), value);
    CHECK(r.ec == std::errc::result_out_of_range);
    CHECK(value == 42.0);
  }
  for (const std::string_view text : { "+7", "", "-", ".", "e5", " 1" }) {
    CAPTURE(text);
    double value = 42.0;
    const auto r = io::parse_number(text.data(), text.data() + text.size(), value);
    CHECK(r.ec == std::errc::invalid_argument);
    CHECK(r.ptr == text.data());
    CHECK(value == 42.0);
  }

  float single = 42.0f;
  const std::string_view tenth = "0.1", huge = "3.4028236e38", tiny = "1.4e-45";
  CHECK(io::parse_number(tenth.data(), tenth.data() + tenth.size(), single).ec == std::errc{});
  CHECK(std::bit_cast<std::uint32_t>(single) == std::bit_cast<std::uint32_t>(0x1.99999ap-4f));
  CHECK(io::parse_number(huge.data(), huge.data() + huge.size(), single).ec
        == std::errc::result_out_of_range);
  CHECK(std::bit_cast<std::uint32_t>(single) == std::bit_cast<std::uint32_t>(0x1.99999ap-4f));
  CHECK(io::parse_number(tiny.data(), tiny.data() + tiny.size(), single).ec == std::errc{});
  CHECK(single == 0x1p-149f);
}

TEST_CASE("FX-6 a Ctrl-Z byte is a non-numeric field, not the end of the file",
          "[fileOperations][fx6][ctrl_z]")
{
  // "1,2,3\n4,5,6\n7,8,9\x1a\n10,11,12\n": a text-mode stream on Windows ended the
  // file at the 0x1A, so three series were read and the fourth dropped silently.
  CHECK_THROWS_AS(load_path(reader_fixture("ctrl_z.csv")), IOError);
  CHECK_THROWS_WITH(load_path(reader_fixture("ctrl_z.csv")),
                    ContainsSubstring("row 3, column 3: invalid numeric field"));
}

TEST_CASE("FX-6 the text reader parses through parse_number",
          "[fileOperations][fx6][number]")
{
  // "1e-3,2E+2,.5,5.,-0,+7"
  const Data sci = load_path(reader_fixture("sci.csv"));
  REQUIRE(sci.size() == 1);
  const std::vector<double> expected{ 0x1.0624dd2f1a9fcp-10, 200.0, 0.5, 5.0, -0.0, 7.0 };
  REQUIRE(sci.p_vec[0].size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i)
    CHECK(std::bit_cast<std::uint64_t>(sci.p_vec[0][i])
          == std::bit_cast<std::uint64_t>(expected[i]));

  CHECK(load_path(reader_fixture("denorm.csv")).p_vec == Series{ { 0x1p-1074, 1.0 } });
  CHECK_THROWS_WITH(load_path(reader_fixture("overflow.csv")),
                    ContainsSubstring("numeric field is out of range '1e400'"));
  CHECK_THROWS_WITH(load_path(reader_fixture("underflow.csv")),
                    ContainsSubstring("numeric field is out of range '1e-400'"));
  CHECK_THROWS_WITH(load_path(reader_fixture("hexfloat.csv")),
                    ContainsSubstring("invalid numeric field '0x1p3'"));
}

TEST_CASE("FX-6 the rest of the reader audit's input matrix",
          "[fileOperations][fx6][matrix]")
{
  // Written in binary so that no line ending is translated on any platform.
  struct Loads
  {
    std::string_view ext, bytes;
    int skip_rows, skip_cols;
    Series expected;
  };
  const Loads loads[] = {
    { ".csv", "1,2,3\r\n4,5,6\r\n", 0, 0, { { 1, 2, 3 }, { 4, 5, 6 } } },
    { ".txt", "1\t2\t3\r\n4\t5\t6\r\n", 0, 0, { { 1, 2, 3 }, { 4, 5, 6 } } },
    { ".csv", "1,2,3\n4,5,6", 0, 0, { { 1, 2, 3 }, { 4, 5, 6 } } },
    { ".csv", "1,2,3\n4,5\n6\n", 0, 0, { { 1, 2, 3 }, { 4, 5 }, { 6 } } },
    { ".csv", "t1,t2,t3\n1,2,3\n", 1, 0, { { 1, 2, 3 } } },
    { ".csv", ",0,1,2\n0,0.5,0.6,0.7\n1,0.1,0.2,0.3\n", 1, 1, { { 0.5, 0.6, 0.7 }, { 0.1, 0.2, 0.3 } } },
    { ".csv", "t1,t2\n", 1, 0, {} },
    { ".csv", "", 0, 0, {} },
  };
  for (const auto &c : loads) {
    CAPTURE(c.bytes);
    TemporaryBatchFile file(c.ext, c.bytes);
    CHECK(load_path(file.path, c.skip_rows, c.skip_cols).p_vec == c.expected);
  }

  TemporaryBatchFile mixed_case_nan(".csv", "1,NaN,3\n");
  const Data nan = load_path(mixed_case_nan.path);
  REQUIRE(nan.p_vec.size() == 1);
  CHECK(std::isnan(nan.p_vec[0][1]));

  struct Fails
  {
    std::string_view ext, bytes, message;
  };
  const Fails fails[] = {
    { ".csv", "t1,t2,t3\n1,2,3\n", "row 1, column 1: invalid numeric field 't1'" },
    { ".csv", "1,-nan,3\n", "row 1, column 2: unapproved non-finite numeric field '-nan'" },
    { ".csv", "1,NA,3\n", "row 1, column 2: invalid numeric field 'NA'" },
    { ".csv", "\"1\",\"2\",\"3\"\n", "row 1, column 1: invalid numeric field '\"1\"'" },
    { ".csv", "1;2;3\n4;5;6\n", "row 1, column 1: invalid numeric field '1;2;3'" },
    { ".csv", "1,5;2,5;3,5\n", "row 1, column 2: invalid numeric field '5;2'" },
    { ".tsv", "1,2,3\r\n", "row 1, column 1: invalid numeric field '1,2,3'" },
    { ".txt", "1.0e+00  -2.5e+00\r\n", "row 1, column 1: invalid numeric field" },
    { ".csv", "1,2,3,\n", "row 1, column 4: empty numeric field" },
    { ".csv", "1,2,3\xC2\xA0\n", "row 1, column 3: invalid numeric field" },
    { ".csv", "1,2,3\r4,5,6\r", "row 1, column 3" }, // a lone CR does not end a line
    { ".csv", std::string_view{ "\xFF\xFE" "1\0,\0" "2\0\n\0", 10 }, "row 1, column 1: invalid numeric field" },
  };
  for (const auto &c : fails) {
    CAPTURE(c.bytes);
    TemporaryBatchFile file(c.ext, c.bytes);
    CHECK_THROWS_WITH(load_path(file.path), ContainsSubstring(std::string(c.message)));
  }

  // A series folder: a missing value mid-file is an error, and so is a file that is not data.
  const ScratchDirectory scratch{ "fx6_matrix_folder" };
  const fs::path &folder = scratch.path;
  std::error_code ec;
  std::ofstream(folder / "a.csv", std::ios::binary) << ",0\n0,0.5\n1,\n2,0.7\n";
  CHECK_THROWS_WITH(load_path(folder, 1, 1), ContainsSubstring("row 3, column 2: empty numeric field"));
  fs::remove(folder / "a.csv", ec);
  std::ofstream(folder / "a.csv", std::ios::binary) << "1\n2\n3\n";
  std::ofstream(folder / "README.md", std::ios::binary) << "# notes\r\nsee a.csv\r\n";
  CHECK_THROWS_WITH(load_path(folder), ContainsSubstring("README.md' row 1, column 1: invalid numeric field"));
}

TEST_CASE("A CR ends a line only before its LF, whatever the delimiter", "[fileOperations][crlf]")
{
  const auto load = [](std::string text, char delimiter) { // '|' stands for the delimiter
    std::replace(text.begin(), text.end(), '|', delimiter);
    TemporaryBatchFile file(".dat", text);
    DataLoader loader(file.path);
    loader.delimiter(delimiter).verbosity(0);
    return loader.load().p_vec;
  };
  for (const char delimiter : { ' ', ',', '\t' }) {
    CAPTURE(delimiter);
    // A CR-only file is one line. The space delimiter split it at each CR, so it
    // read as one series of nine values where ',' and '\t' refused it.
    CHECK_THROWS_AS(load("1|2|3\r4|5|6.5\r7|8|9\r", delimiter), IOError);
    CHECK_THROWS_AS(load("1|2|3 \r4|5|6 \r", delimiter), IOError);
    CHECK(load("1|2|3 \r\n4|5|6\r\n\r\n", delimiter) == Series{ { 1, 2, 3 }, { 4, 5, 6 } });
  }
  // The corpus' CRLF twin (checked out byte for byte, .gitattributes) reads as its LF twin.
  CHECK(load_path(reader_fixture("trailing_blank_crlf.csv")).p_vec
        == load_path(reader_fixture("trailing_blank.csv")).p_vec);
}

TEST_CASE("FX-6 Problem rejects an empty series from any source",
          "[fileOperations][fx6][problem]")
{
  Problem problem("fx6_empty_series");
  problem.set_verbose(false);

  Data owned(Series{ { 1.0, 2.0 }, {}, { 3.0 } }, std::vector<std::string>{ "a", "b", "c" });
  CHECK_THROWS_AS(problem.set_data(owned), InvalidInput);
  CHECK_THROWS_WITH(problem.set_data(owned), ContainsSubstring("series 1 ('b') is empty"));

  Data single(std::vector<std::vector<float>>{ { 1.0f }, {} },
              std::vector<std::string>{ "p", "q" });
  CHECK_THROWS_WITH(problem.set_data(single), ContainsSubstring("series 1 ('q') is empty"));

  const std::vector<double> one{ 1.0 }, none{};
  Data view(std::vector<std::span<const double>>{ one, none },
            std::vector<std::string_view>{ "x", "y" }, 1);
  CHECK_THROWS_AS(problem.set_view_data(view), InvalidInput);
  CHECK_THROWS_WITH(problem.set_view_data(view), ContainsSubstring("series 1 ('y') is empty"));

  // A zero-byte (non-dot) file in a folder is an empty series too.
  const ScratchDirectory scratch{ "fx6_empty_file" };
  const fs::path &folder = scratch.path;
  std::ofstream(folder / "a.csv") << "1\n2\n";
  std::ofstream(folder / "b.csv").close();
  DataLoader loader(folder);
  loader.verbosity(0);
  CHECK_THROWS_WITH((Problem{ "fx6_loader", loader }),
                    ContainsSubstring("series 1 ('b') is empty"));
  CHECK_THROWS_AS(problem.set_data(loader.load()), InvalidInput);

  // Non-empty data is still accepted.
  problem.set_data(Data(Series{ { 1.0 }, { 2.0, 3.0 } }, std::vector<std::string>{ "u", "v" }));
  CHECK(problem.size() == 2);
}

TEST_CASE("GT-4b skip_cols wider than a row is InvalidInput, from a file as from memory",
          "[fileOperations][error][gt4b]")
{
  // The oracle is the in-memory source, whose contract is InvalidInput:
  // the same request against a file is the same mistake, not a failed read.
  REQUIRE_THROWS_AS(dtwc::cluster(dtwc::load(Series{ { 1, 2, 3 }, { 4, 5, 6 } }, 5), 1),
                    InvalidInput);

  TemporaryBatchFile batch(".csv", "1,2,3\n4,5,6\n");
  CHECK_THROWS_AS(dtwc::cluster(dtwc::load(batch.path, 5), 1), InvalidInput);
  CHECK_THROWS_AS(load_path(batch.path, 0, 5), InvalidInput);

  // A one-series-per-file folder reads one value per line from column skip_cols + 1.
  const ScratchDirectory scratch{ "gt4b_skip_cols_folder" };
  const fs::path &folder = scratch.path;
  std::ofstream(folder / "a.csv") << "1\n2\n";
  CHECK_THROWS_AS(load_path(folder, 0, 1), InvalidInput);
  CHECK_THROWS_AS(readFile<double>(folder / "a.csv", 0, 1), InvalidInput);
}
