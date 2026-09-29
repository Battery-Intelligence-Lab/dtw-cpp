/**
 * @file unit_test_mmap_distance_matrix.cpp
 * @brief The mapped storage of core::DistanceMatrix and the `.dtwm` file it maps.
 *
 * @details The oracle for the file is its documented byte layout, read back
 * with plain streams: a 48-byte header {magic "DTWM", uint32 version 4, uint64 N,
 * 32-byte fingerprint} and N(N+1)/2 doubles.
 *
 * @date 29 Sep 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h> // GetFileAttributesW: the sparse-file oracle
#else
#include <csignal>
#include <sys/resource.h>
#endif

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using dtwc::core::DistanceMatrix;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::MessageMatches;
namespace fs = std::filesystem;

namespace {

/// A fresh path under the test temp directory, removed on scope exit.
struct TempFile
{
  fs::path path;
  explicit TempFile(const std::string &name)
    : path(fs::temp_directory_path() / ("dtwc_mmap_test_" + name + ".dtwm"))
  {
    fs::remove(path);
  }
  ~TempFile()
  {
    std::error_code ignored;
    fs::remove(path, ignored);
  }
  TempFile(const TempFile &) = delete;
  TempFile &operator=(const TempFile &) = delete;
};

std::vector<char> file_bytes(const fs::path &path)
{
  std::ifstream in(path, std::ios::binary);
  REQUIRE(in.is_open());
  return { std::istreambuf_iterator<char>{ in }, std::istreambuf_iterator<char>{} };
}

template <typename T>
T read_at(const std::vector<char> &bytes, std::size_t offset)
{
  REQUIRE(offset + sizeof(T) <= bytes.size());
  T value{};
  std::memcpy(&value, bytes.data() + offset, sizeof value);
  return value;
}

DistanceMatrix::fingerprint_type fingerprint_of(std::uint8_t byte)
{
  DistanceMatrix::fingerprint_type f{};
  f.fill(byte);
  return f;
}

} // namespace

TEST_CASE("tri_index is symmetric and packs the lower triangle row by row", "[DistanceMatrix][tri_index]")
{
  for (std::size_t i = 0; i < 20; ++i)
    for (std::size_t j = 0; j <= i; ++j) {
      REQUIRE(dtwc::core::tri_index(i, j) == dtwc::core::tri_index(j, i));
      REQUIRE(dtwc::core::tri_index(i, j) == i * (i + 1) / 2 + j);
    }
  REQUIRE(dtwc::core::packed_size(0) == 0);
  REQUIRE(dtwc::core::packed_size(1) == 1);
  REQUIRE(dtwc::core::packed_size(4) == 10);
}

#ifndef DTWC_HAS_MMAP

TEST_CASE("map() on a build without llfio is IOError naming the build option", "[DistanceMatrix][mmap]")
{
  TempFile tmp("no_llfio");
  REQUIRE_THROWS_MATCHES(DistanceMatrix::map(tmp.path, 3, {}), dtwc::IOError,
                         MessageMatches(ContainsSubstring("DTWC_ENABLE_LLFIO=ON")));
  REQUIRE_FALSE(fs::exists(tmp.path));
}

#else

TEST_CASE("A new mapped matrix is every entry NaN under its header", "[DistanceMatrix][mmap]")
{
  TempFile tmp("create");
  const auto fingerprint = fingerprint_of(0x5a);
  {
    auto m = DistanceMatrix::map(tmp.path, 10, fingerprint);
    REQUIRE(m.is_mapped());
    REQUIRE(m.size() == 10);
    REQUIRE(m.count_computed() == 0);
  }
  const auto bytes = file_bytes(tmp.path);
  REQUIRE(bytes.size() == 48 + 55 * sizeof(double));
  REQUIRE(std::string(bytes.data(), 4) == "DTWM");
  REQUIRE(read_at<std::uint32_t>(bytes, 4) == 4);
  REQUIRE(read_at<std::uint64_t>(bytes, 8) == 10);
  REQUIRE(read_at<DistanceMatrix::fingerprint_type>(bytes, 16) == fingerprint);
  for (std::size_t k = 0; k < 55; ++k)
    REQUIRE(std::isnan(read_at<double>(bytes, 48 + k * sizeof(double))));
}

TEST_CASE("A mapped matrix persists bit for bit and reopens with its values", "[DistanceMatrix][mmap]")
{
  TempFile tmp("persist");
  const auto fingerprint = fingerprint_of(0x11);
  const std::vector<double> values{ 0.0, -0.0, 1.5, -2.25, 1e300, 5e-324, 7.0 };
  const std::size_t n = 5;
  {
    auto m = DistanceMatrix::map(tmp.path, n, fingerprint);
    for (std::size_t k = 0; k < values.size(); ++k) m.raw()[k] = values[k];
    m.set(4, 3, 42.0);
  }
  const auto bytes = file_bytes(tmp.path);
  for (std::size_t k = 0; k < values.size(); ++k)
    REQUIRE(std::bit_cast<std::uint64_t>(read_at<double>(bytes, 48 + k * sizeof(double)))
            == std::bit_cast<std::uint64_t>(values[k]));

  // One layout, two routes. The stream read comes first: Windows refuses a
  // second open that does not share delete access while llfio holds the file.
  const auto streamed = DistanceMatrix::read(tmp.path, n, fingerprint);
  const auto reopened = DistanceMatrix::map(tmp.path, n, fingerprint);
  REQUIRE_FALSE(streamed.is_mapped());
  for (std::size_t k = 0; k < reopened.packed_count(); ++k) {
    REQUIRE(std::bit_cast<std::uint64_t>(reopened.raw()[k]) == std::bit_cast<std::uint64_t>(streamed.raw()[k]));
    if (k < values.size())
      REQUIRE(std::bit_cast<std::uint64_t>(reopened.raw()[k]) == std::bit_cast<std::uint64_t>(values[k]));
  }
  REQUIRE(reopened.get(3, 4) == 42.0);
  REQUIRE(reopened.count_computed() == values.size() + 1);
}

TEST_CASE("Writing a matrix over the file it maps flushes it in place", "[DistanceMatrix][mmap]")
{
  // A rename over a mapped file fails on Windows and would orphan the map on POSIX.
  TempFile tmp("write_self");
  const auto fingerprint = fingerprint_of(0x22);
  {
    auto m = DistanceMatrix::map(tmp.path, 3, fingerprint);
    m.set(2, 0, 9.5);
    REQUIRE_NOTHROW(m.write(tmp.path, fingerprint));
    REQUIRE(m.is_mapped());
    REQUIRE_FALSE(fs::exists(fs::path(tmp.path).concat(".tmp")));
    m.set(2, 1, 3.0); // still the same mapping
  }
  const auto read_back = DistanceMatrix::read(tmp.path, 3, fingerprint);
  REQUIRE(read_back.get(0, 2) == 9.5);
  REQUIRE(read_back.get(1, 2) == 3.0);
}

TEST_CASE("A moved or resized matrix lets go of its file", "[DistanceMatrix][mmap]")
{
  TempFile tmp("release");
  auto m = DistanceMatrix::map(tmp.path, 4, {});
  m.set(1, 0, 2.0);
  DistanceMatrix moved(std::move(m));
  REQUIRE(moved.is_mapped());
  REQUIRE(moved.get(0, 1) == 2.0);
  REQUIRE(m.size() == 0); // NOLINT(bugprone-use-after-move): the moved-from state is the subject
  REQUIRE(m.raw() == nullptr);
  REQUIRE_FALSE(m.is_mapped());

  moved.resize(4);
  REQUIRE_FALSE(moved.is_mapped());
  REQUIRE(moved.count_computed() == 0);
  // With the handle closed the file can go on every platform.
  std::error_code ec;
  REQUIRE(fs::remove(tmp.path, ec));
  REQUIRE_FALSE(ec);
}

TEST_CASE("Zero and one series map and reopen", "[DistanceMatrix][mmap]")
{
  TempFile zero("n0");
  TempFile one("n1");
  { auto m = DistanceMatrix::map(zero.path, 0, {}); }
  {
    auto m = DistanceMatrix::map(one.path, 1, {});
    m.set(0, 0, 0.0);
  }
  REQUIRE(fs::file_size(zero.path) == 48);
  REQUIRE(fs::file_size(one.path) == 56);
  REQUIRE(DistanceMatrix::map(zero.path, 0, {}).size() == 0);
  REQUIRE(DistanceMatrix::map(one.path, 1, {}).get(0, 0) == 0.0);
}

// A crafted N whose packed size wraps in 64 bits (2^62: N(N+1)/2 * 8 == 0 mod
// 2^64) must not pass the length check on a header-only file.
TEST_CASE("open rejects N that overflows packed size", "[DistanceMatrix][mmap][security]")
{
  TempFile tmp("overflow");
  std::array<char, 48> header{};
  std::memcpy(header.data(), "DTWM", 4);
  const std::uint32_t version = 4;
  std::memcpy(header.data() + 4, &version, sizeof version);
  const std::uint64_t bad_n = std::uint64_t{ 1 } << 62;
  std::memcpy(header.data() + 8, &bad_n, sizeof bad_n);
  std::ofstream(tmp.path, std::ios::binary).write(header.data(), header.size());

  REQUIRE_THROWS_MATCHES(DistanceMatrix::map(tmp.path, 3, {}), dtwc::IOError,
                         MessageMatches(ContainsSubstring("not the size")));
  REQUIRE_THROWS_AS(DistanceMatrix::read(tmp.path, 3, {}), dtwc::IOError);
}

TEST_CASE("A distance matrix stored for other data is InvalidInput, CSV and cache alike",
          "[DistanceMatrix][mmap][fingerprint][error][gt4b]")
{
  // The oracle is read_distance_matrix: a CSV matrix computed for other series
  // is InvalidInput. A cache computed for other series is the same request.
  const auto problem = [](std::vector<std::vector<double>> series) {
    std::vector<std::string> names;
    for (std::size_t i = 0; i < series.size(); ++i) names.push_back("s" + std::to_string(i));
    dtwc::Problem prob("gt4b_other_data");
    prob.set_data(dtwc::Data(std::move(series), std::move(names)));
    return prob;
  };
  TempFile cache("other_data");
  const fs::path csv = fs::path(cache.path).replace_extension(".csv");
  {
    auto writer = problem({ { 0, 1 }, { 1, 2 }, { 2, 4 }, { 9, 9 } });
    writer.use_mmap_distance_matrix(cache.path);
    writer.fill_distance_matrix();
  }
  std::ofstream(csv) << "0,1,2,3\n1,0,1,2\n2,1,0,1\n3,2,1,0\n";

  auto reader = problem({ { 0, 1, 2 }, { 1, 2, 3 }, { 5, 6, 7 } });
  REQUIRE_THROWS_AS(reader.read_distance_matrix(csv), dtwc::InvalidInput);
  REQUIRE_THROWS_AS(reader.use_mmap_distance_matrix(cache.path), dtwc::InvalidInput);
  REQUIRE_FALSE(reader.distance_matrix().is_mapped());
  auto same_n = problem({ { 0, 1 }, { 1, 2 }, { 2, 4 }, { 9, 8 } }); // one value differs
  REQUIRE_THROWS_MATCHES(same_n.use_mmap_distance_matrix(cache.path), dtwc::InvalidInput,
                         MessageMatches(ContainsSubstring("fingerprint mismatch")));
  fs::remove(csv);
}

#ifdef _WIN32
TEST_CASE("A new mapped matrix is not a sparse file on Windows", "[DistanceMatrix][mmap]")
{
  // llfio's default sets FILE_ATTRIBUTE_SPARSE_FILE, and random reads from a
  // filled sparse file measured 1.9x slower.
  TempFile tmp("sparse");
  {
    auto m = DistanceMatrix::map(tmp.path, 50, {});
    m.set(3, 7, 42.0);
    m.sync();
  }
  const DWORD attributes = GetFileAttributesW(tmp.path.c_str());
  REQUIRE(attributes != INVALID_FILE_ATTRIBUTES);
  REQUIRE((attributes & FILE_ATTRIBUTE_SPARSE_FILE) == 0);
  REQUIRE(DistanceMatrix::map(tmp.path, 50, {}).get(7, 3) == 42.0);
}
#else
namespace {
/// Lowers RLIMIT_FSIZE's soft limit, as `ulimit -f` or a batch scheduler does,
/// and ignores SIGXFSZ so an oversized file fails with EFBIG rather than killing
/// the test; both are restored on scope exit.
class FileSizeLimit
{
  rlimit previous_{};
  void (*previous_handler_)(int) = SIG_DFL;

public:
  explicit FileSizeLimit(rlim_t bytes)
  {
    REQUIRE(getrlimit(RLIMIT_FSIZE, &previous_) == 0);
    previous_handler_ = std::signal(SIGXFSZ, SIG_IGN);
    rlimit lowered = previous_;
    lowered.rlim_cur = std::min(bytes, previous_.rlim_max);
    const int lowered_status = setrlimit(RLIMIT_FSIZE, &lowered);
    if (lowered_status != 0) std::signal(SIGXFSZ, previous_handler_);
    REQUIRE(lowered_status == 0);
  }
  ~FileSizeLimit()
  {
    setrlimit(RLIMIT_FSIZE, &previous_);
    std::signal(SIGXFSZ, previous_handler_);
  }
  FileSizeLimit(const FileSizeLimit &) = delete;
  FileSizeLimit &operator=(const FileSizeLimit &) = delete;
};
} // namespace

TEST_CASE("Creating a cache over a file-size quota is IOError naming the path",
          "[DistanceMatrix][mmap][error][gt4b]")
{
  // llfio's own error is no dtwc::Error: Python saw RuntimeError and MATLAB
  // dtwc:runtime (contract §5: IOError).
  TempFile tmp("quota");
  const FileSizeLimit limit(64 * 1024);
  CHECK_THROWS_MATCHES(DistanceMatrix::map(tmp.path, 200, {}), dtwc::IOError, // about 160 KB
                       MessageMatches(ContainsSubstring(tmp.path.filename().string())));
}
#endif

#endif // DTWC_HAS_MMAP
