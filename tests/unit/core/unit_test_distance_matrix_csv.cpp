/**
 * @file unit_test_distance_matrix_csv.cpp
 * @brief F14 byte contract for every native distance-matrix CSV route.
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <bit>
#include <clocale>
#include <cstdlib>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

#ifndef DTWC_F14_TEST_ROOT
#error "DTWC_F14_TEST_ROOT must name the configured build-local test root"
#endif

namespace fs = std::filesystem;

namespace {

constexpr std::string_view kExpected =
  ",-0,1.0000000000000002\n"
  "-0,1.7976931348623157e+308,-1.25\n"
  "1.0000000000000002,-1.25,0\n";
constexpr std::string_view kPositiveInfinityMessage =
  "distance-matrix CSV: computed non-finite value at row 0, column 1.";
constexpr std::string_view kNegativeInfinityMessage =
  "distance-matrix CSV: computed non-finite value at row 1, column 2.";

fs::path test_root()
{
  const fs::path root{DTWC_F14_TEST_ROOT};
  if (!root.is_absolute()
      || root.filename() != "f14-distance-matrix-csv-unit") {
    throw std::runtime_error("unsafe F14 unit-test root: " + root.string());
  }
  fs::create_directories(root);
  return root;
}

fs::path fresh_path(std::string_view name)
{
  const fs::path root = test_root();
  const fs::path path = root / std::string(name);
  if (path.parent_path() != root)
    throw std::runtime_error("F14 scratch path escaped its build root");
  std::error_code error;
  fs::remove_all(path, error);
  if (error)
    throw std::runtime_error(
      "cannot reset F14 scratch path: " + error.message());
  return path;
}

std::string read_binary(const fs::path &path)
{
  std::ifstream input(path, std::ios::in | std::ios::binary);
  if (!input.is_open())
    throw std::runtime_error("cannot read F14 artifact: " + path.string());
  return {
    std::istreambuf_iterator<char>(input),
    std::istreambuf_iterator<char>()};
}

void seed_binary(const fs::path &path, std::string_view bytes)
{
  std::ofstream output(
    path, std::ios::out | std::ios::binary | std::ios::trunc);
  output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (!output)
    throw std::runtime_error("cannot seed F14 artifact: " + path.string());
}

double negative_zero()
{
  return std::bit_cast<double>(UINT64_C(0x8000000000000000));
}

template <class Matrix>
void populate_contract_matrix(Matrix &matrix)
{
  matrix.set(0, 1, negative_zero());
  matrix.set(0, 2, std::nextafter(1.0, 2.0));
  matrix.set(1, 1, std::numeric_limits<double>::max());
  matrix.set(1, 2, -1.25);
  matrix.set(2, 2, 0.0);
}

template <class Matrix>
void check_contract_bits(const Matrix &matrix)
{
  CHECK_FALSE(matrix.is_computed(0, 0));
  CHECK(std::bit_cast<std::uint64_t>(matrix.get(0, 1))
        == UINT64_C(0x8000000000000000));
  CHECK(std::bit_cast<std::uint64_t>(matrix.get(0, 2))
        == UINT64_C(0x3ff0000000000001));
  CHECK(std::bit_cast<std::uint64_t>(matrix.get(1, 1))
        == UINT64_C(0x7fefffffffffffff));
  CHECK(std::bit_cast<std::uint64_t>(matrix.get(1, 2))
        == UINT64_C(0xbff4000000000000));
  CHECK(std::bit_cast<std::uint64_t>(matrix.get(2, 2))
        == UINT64_C(0x0000000000000000));
}

dtwc::core::DenseDistanceMatrix contract_dense()
{
  dtwc::core::DenseDistanceMatrix matrix(3);
  populate_contract_matrix(matrix);
  return matrix;
}

void configure_dense_problem(dtwc::Problem &problem)
{
  problem.set_data(dtwc::Data(
    std::vector<std::vector<double>>{{0.0}, {1.0}, {2.0}},
    std::vector<std::string>{"a", "b", "c"}));
  auto &matrix = problem.dense_distance_matrix();
  matrix.resize(3);
  populate_contract_matrix(matrix);
  problem.set_output_folder(test_root());
}

#ifdef DTWC_HAS_MMAP
void configure_mmap_problem(
  dtwc::Problem &problem, const fs::path &cache_path)
{
  problem.set_data(dtwc::Data(
    std::vector<std::vector<double>>{{0.0}, {1.0}, {2.0}},
    std::vector<std::string>{"a", "b", "c"}));
  problem.use_mmap_distance_matrix(cache_path);
  auto &matrix =
    std::get<dtwc::core::MmapDistanceMatrix>(problem.distance_matrix());
  populate_contract_matrix(matrix);
  problem.set_output_folder(test_root());
}
#endif

struct Rejection
{
  bool typed = false;
  std::string message;
};

template <class Function>
Rejection capture_rejection(Function &&function)
{
  try {
    std::forward<Function>(function)();
  } catch (const dtwc::InvalidInput &error) {
    return {true, error.what()};
  } catch (const std::exception &error) {
    return {false, std::string("wrong exception: ") + error.what()};
  }
  return {false, "<no exception>"};
}

class comma_numpunct final : public std::numpunct<char>
{
protected:
  char do_decimal_point() const override { return ','; }
  char do_thousands_sep() const override { return '.'; }
  std::string do_grouping() const override { return "\3"; }
};

std::locale hostile_locale()
{
  return std::locale(std::locale::classic(), new comma_numpunct);
}

class global_locale_guard
{
public:
  explicit global_locale_guard(const std::locale &replacement)
    : previous_(std::locale())
  {
    std::locale::global(replacement);
  }

  ~global_locale_guard() { std::locale::global(previous_); }

  global_locale_guard(const global_locale_guard &) = delete;
  global_locale_guard &operator=(const global_locale_guard &) = delete;

private:
  std::locale previous_;
};

class cout_capture
{
public:
  cout_capture() : previous_(std::cout.rdbuf(buffer_.rdbuf())) {}
  ~cout_capture() { std::cout.rdbuf(previous_); }

  std::string str() const { return buffer_.str(); }

  cout_capture(const cout_capture &) = delete;
  cout_capture &operator=(const cout_capture &) = delete;

private:
  std::ostringstream buffer_;
  std::streambuf *previous_;
};

void check_literal_bytes(std::string_view bytes)
{
  CHECK(bytes == kExpected);
  CHECK(bytes.size() == 83);
  CHECK(std::count(bytes.begin(), bytes.end(), '\n') == 3);
  CHECK(std::count(bytes.begin(), bytes.end(), '\r') == 0);
  CHECK(std::count(bytes.begin(), bytes.end(), ',') == 6);
  REQUIRE_FALSE(bytes.empty());
  CHECK(bytes.back() == '\n');
}

} // namespace

TEST_CASE("F14 independent matrix fixture retains every registered bit",
          "[f14][csv][oracle]")
{
  const auto matrix = contract_dense();
  check_contract_bits(matrix);
  CHECK(kExpected.size() == 83);
  CHECK(kExpected.substr(0, 22) == ",-0,1.0000000000000002");
  CHECK(kExpected.substr(23, 32)
        == "-0,1.7976931348623157e+308,-1.25");
  CHECK(kExpected.substr(56, 26) == "1.0000000000000002,-1.25,0");
}

TEST_CASE("F14 dense stream emits the exact independent literal",
          "[f14][csv][dense][stream]")
{
  const auto matrix = contract_dense();
  std::ostringstream output;
  output << matrix;
  check_literal_bytes(output.str());
}

TEST_CASE("F14 dense stream ignores and preserves hostile caller state",
          "[f14][csv][dense][state]")
{
  const auto matrix = contract_dense();
  std::ostringstream output;
  const std::locale locale = hostile_locale();
  output.imbue(locale);
  output.setf(std::ios::scientific, std::ios::floatfield);
  output.setf(std::ios::showpos | std::ios::showpoint | std::ios::uppercase);
  output.precision(3);
  output.fill('#');
  output.width(9);

  const auto flags = output.flags();
  const auto precision = output.precision();
  const auto fill = output.fill();
  const auto width = output.width();
  output << matrix;

  check_literal_bytes(output.str());
  CHECK(output.flags() == flags);
  CHECK(output.precision() == precision);
  CHECK(output.fill() == fill);
  CHECK(output.width() == width);
  CHECK(output.getloc() == locale);
}

TEST_CASE("F14 dense file is binary locale-free and bit-roundtrippable",
          "[f14][csv][dense][file]")
{
  const auto path = fresh_path("dense.csv");
  const auto matrix = contract_dense();
  {
    global_locale_guard guard(hostile_locale());
    dtwc::io::write_csv(matrix, path);
  }
  const std::string bytes = read_binary(path);
  check_literal_bytes(bytes);

  dtwc::core::DenseDistanceMatrix loaded;
  std::string read_error;
  try {
    dtwc::io::read_csv(loaded, path);
  } catch (const std::exception &error) {
    read_error = error.what();
  }
  CHECK(read_error.empty());
  if (read_error.empty()) {
    CHECK(loaded.size() == 3);
    check_contract_bits(loaded);
  }
}

TEST_CASE("F14 dense Problem visitor and print route are literal-identical",
          "[f14][csv][dense][problem]")
{
  dtwc::Problem problem("f14_dense");
  configure_dense_problem(problem);
  const auto path = fresh_path("dense_problem.csv");
  problem.write_distance_matrix(path.filename().string());
  check_literal_bytes(read_binary(path));

  cout_capture capture;
  problem.print_distance_matrix();
  check_literal_bytes(capture.str());
}

TEST_CASE("F14 positive infinity rejects before any dense output",
          "[f14][csv][dense][nonfinite]")
{
  auto matrix = contract_dense();
  matrix.set(0, 1, std::numeric_limits<double>::infinity());

  std::ostringstream stream;
  stream << "prefix";
  const std::locale locale = hostile_locale();
  stream.imbue(locale);
  stream.setf(std::ios::scientific, std::ios::floatfield);
  stream.setf(std::ios::showpos | std::ios::showpoint | std::ios::uppercase);
  stream.precision(3);
  stream.fill('#');
  stream.width(9);
  const auto flags = stream.flags();
  const auto precision = stream.precision();
  const auto fill = stream.fill();
  const auto width = stream.width();
  const auto stream_rejection =
    capture_rejection([&] { stream << matrix; });
  CHECK(stream_rejection.typed);
  CHECK(stream_rejection.message == kPositiveInfinityMessage);
  CHECK(stream.str() == "prefix");
  CHECK(stream.flags() == flags);
  CHECK(stream.precision() == precision);
  CHECK(stream.fill() == fill);
  CHECK(stream.width() == width);
  CHECK(stream.getloc() == locale);

  const auto existing = fresh_path("positive_existing.csv");
  seed_binary(existing, "seed");
  const auto file_rejection =
    capture_rejection([&] { dtwc::io::write_csv(matrix, existing); });
  CHECK(file_rejection.typed);
  CHECK(file_rejection.message == kPositiveInfinityMessage);
  CHECK(read_binary(existing) == "seed");

  const auto missing = fresh_path("positive_missing.csv");
  const auto missing_rejection =
    capture_rejection([&] { dtwc::io::write_csv(matrix, missing); });
  CHECK(missing_rejection.typed);
  CHECK(missing_rejection.message == kPositiveInfinityMessage);
  CHECK_FALSE(fs::exists(missing));
}

TEST_CASE("F14 negative infinity rejects before dense Problem output",
          "[f14][csv][dense][nonfinite]")
{
  dtwc::Problem problem("f14_negative");
  configure_dense_problem(problem);
  problem.dense_distance_matrix().set(
    1, 2, -std::numeric_limits<double>::infinity());

  const auto path = fresh_path("negative_existing.csv");
  seed_binary(path, "seed");
  const auto file_rejection = capture_rejection(
    [&] { problem.write_distance_matrix(path.filename().string()); });
  CHECK(file_rejection.typed);
  CHECK(file_rejection.message == kNegativeInfinityMessage);
  CHECK(read_binary(path) == "seed");

  cout_capture capture;
  const auto print_rejection =
    capture_rejection([&] { problem.print_distance_matrix(); });
  CHECK(print_rejection.typed);
  CHECK(print_rejection.message == kNegativeInfinityMessage);
  CHECK(capture.str().empty());
}

TEST_CASE("F14 zero-size dense routes emit zero bytes",
          "[f14][csv][dense][empty]")
{
  const dtwc::core::DenseDistanceMatrix matrix;
  std::ostringstream stream;
  stream << matrix;
  CHECK(stream.str().empty());

  const auto path = fresh_path("dense_empty.csv");
  dtwc::io::write_csv(matrix, path);
  CHECK(read_binary(path).empty());

  dtwc::Problem problem("f14_empty");
  cout_capture capture;
  problem.print_distance_matrix();
  CHECK(capture.str().empty());
}

TEST_CASE("F14 native Result save matches the registered dense stream bytes",
          "[f14][csv][result]")
{
  const auto dataset = dtwc::load(
    std::vector<std::vector<double>>{{0.0}, {1.0}, {3.0}},
    0, 0, ',', "f14_result");
  const auto result = dtwc::cluster(dataset, 2, "pam", -1, "cpu", 100);
  const auto directory = fresh_path("result");
  result.save(directory);
  const std::string bytes =
    read_binary(directory / "f14_result_distance_matrix.csv");
  constexpr std::string_view expected =
    "0,1,3\n"
    "1,0,2\n"
    "3,2,0\n";
  CHECK(bytes == expected);
  CHECK(bytes.size() == 18);
  CHECK(std::count(bytes.begin(), bytes.end(), '\n') == 3);
  CHECK(std::count(bytes.begin(), bytes.end(), '\r') == 0);
  REQUIRE_FALSE(bytes.empty());
  CHECK(bytes.back() == '\n');
}

#ifdef DTWC_HAS_MMAP

TEST_CASE("F14 mmap stream is literal and hostile-state independent",
          "[f14][csv][mmap][stream]")
{
  const auto cache = fresh_path("stream.dtwcache");
  dtwc::core::MmapDistanceMatrix matrix(cache, 3);
  populate_contract_matrix(matrix);
  check_contract_bits(matrix);

  std::ostringstream ordinary;
  ordinary << matrix;
  check_literal_bytes(ordinary.str());

  std::ostringstream hostile;
  const std::locale locale = hostile_locale();
  hostile.imbue(locale);
  hostile.setf(std::ios::scientific, std::ios::floatfield);
  hostile.setf(std::ios::showpos | std::ios::showpoint | std::ios::uppercase);
  hostile.precision(3);
  hostile.fill('#');
  hostile.width(9);
  const auto flags = hostile.flags();
  const auto precision = hostile.precision();
  const auto fill = hostile.fill();
  const auto width = hostile.width();
  hostile << matrix;
  check_literal_bytes(hostile.str());
  CHECK(hostile.flags() == flags);
  CHECK(hostile.precision() == precision);
  CHECK(hostile.fill() == fill);
  CHECK(hostile.width() == width);
  CHECK(hostile.getloc() == locale);
}

TEST_CASE("F14 mmap Problem visitor and print route are literal-identical",
          "[f14][csv][mmap][problem]")
{
  dtwc::Problem problem("f14_mmap");
  configure_mmap_problem(problem, fresh_path("problem.dtwcache"));
  const auto path = fresh_path("mmap_problem.csv");
  {
    global_locale_guard guard(hostile_locale());
    problem.write_distance_matrix(path.filename().string());
  }
  check_literal_bytes(read_binary(path));

  cout_capture capture;
  problem.print_distance_matrix();
  check_literal_bytes(capture.str());
}

TEST_CASE("F14 mmap empty and nonfinite routes execute without partial output",
          "[f14][csv][mmap][nonfinite][empty]")
{
  {
    dtwc::core::MmapDistanceMatrix empty(fresh_path("empty.dtwcache"), 0);
    std::ostringstream output;
    output << empty;
    CHECK(output.str().empty());
  }
  {
    dtwc::Problem empty_problem("f14_mmap_empty");
    empty_problem.use_mmap_distance_matrix(
      fresh_path("empty_problem.dtwcache"));
    empty_problem.set_output_folder(test_root());
    const auto empty_path = fresh_path("mmap_empty.csv");
    empty_problem.write_distance_matrix(empty_path.filename().string());
    CHECK(read_binary(empty_path).empty());
    cout_capture capture;
    empty_problem.print_distance_matrix();
    CHECK(capture.str().empty());
  }

  dtwc::Problem problem("f14_mmap_nonfinite");
  configure_mmap_problem(problem, fresh_path("nonfinite.dtwcache"));
  auto &matrix =
    std::get<dtwc::core::MmapDistanceMatrix>(problem.distance_matrix());
  matrix.set(1, 2, -std::numeric_limits<double>::infinity());

  std::ostringstream stream;
  stream << "prefix";
  const std::locale locale = hostile_locale();
  stream.imbue(locale);
  stream.setf(std::ios::scientific, std::ios::floatfield);
  stream.setf(std::ios::showpos | std::ios::showpoint | std::ios::uppercase);
  stream.precision(3);
  stream.fill('#');
  stream.width(9);
  const auto flags = stream.flags();
  const auto precision = stream.precision();
  const auto fill = stream.fill();
  const auto width = stream.width();
  const auto stream_rejection =
    capture_rejection([&] { stream << matrix; });
  CHECK(stream_rejection.typed);
  CHECK(stream_rejection.message == kNegativeInfinityMessage);
  CHECK(stream.str() == "prefix");
  CHECK(stream.flags() == flags);
  CHECK(stream.precision() == precision);
  CHECK(stream.fill() == fill);
  CHECK(stream.width() == width);
  CHECK(stream.getloc() == locale);

  const auto path = fresh_path("mmap_nonfinite.csv");
  seed_binary(path, "seed");
  const auto file_rejection = capture_rejection(
    [&] { problem.write_distance_matrix(path.filename().string()); });
  CHECK(file_rejection.typed);
  CHECK(file_rejection.message == kNegativeInfinityMessage);
  CHECK(read_binary(path) == "seed");

  cout_capture capture;
  const auto print_rejection =
    capture_rejection([&] { problem.print_distance_matrix(); });
  CHECK(print_rejection.typed);
  CHECK(print_rejection.message == kNegativeInfinityMessage);
  CHECK(capture.str().empty());
}

#endif

TEST_CASE("F14 CSV read is independent of the C numeric locale",
          "[f14][csv][dense][locale]")
{
  // read_csv used std::stod, which honours LC_NUMERIC: under a comma-decimal
  // locale it stopped at the '.' and read "1.5" back as 1. std::from_chars is
  // locale-independent by specification. The writer already used to_chars.
  struct c_numeric_locale_guard
  {
    std::string saved;
    c_numeric_locale_guard()
    {
      const char *const current = std::setlocale(LC_NUMERIC, nullptr);
      saved = current != nullptr ? current : "C";
    }
    ~c_numeric_locale_guard() { std::setlocale(LC_NUMERIC, saved.c_str()); }
  } guard;

  const char *applied = nullptr;
  for (const char *name : { "de-DE", "German", "de_DE.UTF-8", "de_DE" })
    if (std::setlocale(LC_NUMERIC, name) != nullptr) {
      applied = name;
      break;
    }

  if (applied == nullptr) {
    std::cout << "F14_LOCALE_NUMERIC locale=unavailable "
                 "reason=no_German_LC_NUMERIC_locale_installed\n";
    CHECK(std::strtod("1.5", nullptr) == 1.5);
    return;
  }

  // Without this the case would prove nothing: the locale must actually bite.
  const bool comma_decimal = std::strtod("1.5", nullptr) != 1.5;

  const auto path = fresh_path("locale.csv");
  dtwc::core::DenseDistanceMatrix matrix;
  matrix.resize(2);
  matrix.set(0, 0, 0.0);
  matrix.set(1, 0, 1.5);
  matrix.set(1, 1, 0.0);
  dtwc::io::write_csv(matrix, path);
  const std::string bytes = read_binary(path);

  dtwc::core::DenseDistanceMatrix loaded;
  dtwc::io::read_csv(loaded, path);

  std::cout << "F14_LOCALE_NUMERIC locale=" << applied << " comma_decimal="
            << (comma_decimal ? "yes" : "no") << " ran\n";
  CHECK(comma_decimal);
  CHECK(bytes.find("1.5") != std::string::npos);
  CHECK(bytes.find("1,5") == std::string::npos);
  REQUIRE(loaded.size() == 2);
  CHECK(loaded.get(1, 0) == 1.5);
  CHECK(loaded.get(0, 1) == 1.5);
}

TEST_CASE("F14 focused route marker", "[f14][csv][marker]")
{
  CHECK(true);
#ifdef DTWC_HAS_MMAP
  std::cout << "F14_CSV_CONTRACT dense=ran mmap=ran skips=0\n";
#else
  std::cout << "F14_CSV_CONTRACT dense=ran mmap=unavailable skips=0\n";
#endif
}
