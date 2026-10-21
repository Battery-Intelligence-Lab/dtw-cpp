/**
 * @file unit_test_mmap_distance_matrix.cpp
 * @brief The mapped storage of core::DistanceMatrix and the `.dtwm` file it maps.
 *
 * @details The oracle for the file is its documented byte layout, read back
 * with plain streams: a 48-byte header {magic "DTWM", uint32 version 4, uint64 N,
 * 32-byte fingerprint} and N(N+1)/2 doubles.
 *
 * @author Volkan Kumtepeli
 * @date 29 Sep 2026
 */

#include <dtwc.hpp>

#include "../../support/scratch_directory.hpp"

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
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

using dtwc::core::DistanceMatrix;
using Catch::Matchers::ContainsSubstring;
using Catch::Matchers::MessageMatches;
namespace fs = std::filesystem;
using dtwc::test_support::ScratchDirectory;

namespace {

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

/// The packed values' bits, widened to double.
std::vector<std::uint64_t> bits_of(const DistanceMatrix &m)
{
  std::vector<std::uint64_t> bits;
  m.visit([&](const auto *values) {
    for (std::size_t k = 0; k < m.packed_count(); ++k)
      bits.push_back(std::bit_cast<std::uint64_t>(static_cast<double>(values[k])));
  });
  return bits;
}

/// A version-5 header: the version-4 layout above (bytes 0-47, version 5), then the bytes of a value, a zero word
/// and the rows written.
std::vector<char> v5_header(std::size_t n, const DistanceMatrix::fingerprint_type &fingerprint, std::uint32_t bytes,
                            std::uint64_t rows)
{
  std::vector<char> header(64, 0);
  std::memcpy(header.data(), "DTWM", 4);
  const std::uint32_t version = 5;
  const std::uint64_t n64 = n;
  std::memcpy(header.data() + 4, &version, sizeof version);
  std::memcpy(header.data() + 8, &n64, sizeof n64);
  std::memcpy(header.data() + 16, fingerprint.data(), fingerprint.size());
  std::memcpy(header.data() + 48, &bytes, sizeof bytes);
  std::memcpy(header.data() + 56, &rows, sizeof rows);
  return header;
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
  const ScratchDirectory tmp_dir{ "mmap_no_llfio" };
  const fs::path tmp = tmp_dir.path / "no_llfio.dtwm";
  REQUIRE_THROWS_MATCHES(DistanceMatrix::map(tmp, 3, {}), dtwc::IOError,
                         MessageMatches(ContainsSubstring("DTWC_ENABLE_LLFIO=ON")));
  REQUIRE_FALSE(fs::exists(tmp));
}

#else

TEST_CASE("A new mapped matrix has no row written: its version-5 header says 0", "[DistanceMatrix][mmap]")
{
  // No NaN is written into a new file: its header's rows written, 0, makes every pair not computed.
  for (const auto width : { dtwc::core::Precision::Float64, dtwc::core::Precision::Float32 }) {
    const ScratchDirectory tmp_dir{ "mmap_create" };
    const fs::path tmp = tmp_dir.path / "create.dtwm";
    const auto fingerprint = fingerprint_of(0x5a);
    const std::size_t value = dtwc::core::value_bytes(width);
    {
      auto m = DistanceMatrix::map(tmp, 10, fingerprint, width);
      REQUIRE(m.is_mapped());
      REQUIRE(m.size() == 10);
      REQUIRE(m.width() == width);
      REQUIRE(m.rows_written() == 0);
      REQUIRE(m.count_computed() == 0);
      REQUIRE_FALSE(m.all_computed("test", false));
    }
    const auto bytes = file_bytes(tmp);
    REQUIRE(bytes.size() == 64 + 55 * value);
    REQUIRE(std::vector<char>(bytes.begin(), bytes.begin() + 64)
            == v5_header(10, fingerprint, static_cast<std::uint32_t>(value), 0));
    REQUIRE(DistanceMatrix::read(tmp, 10, fingerprint, width).count_computed() == 0);
  }
}

TEST_CASE("Rows written: a wait that leaves rows on the device records them, and no reader looks past them",
          "[DistanceMatrix][mmap]")
{
  const ScratchDirectory tmp_dir{ "mmap_rows" };
  const fs::path tmp = tmp_dir.path / "rows.dtwm";
  const auto fingerprint = fingerprint_of(0x33);
  {
    auto m = DistanceMatrix::map(tmp, 4, fingerprint);
    m.clear_rows(0, 2);
    m.set(1, 0, 2.5);
    m.flush_rows(0, 2, false); // started only: nothing recorded
    REQUIRE(m.rows_written() == 0);
    m.flush_rows(2, 4, true); // rows 0 and 1 not known on the device: nothing recorded
    REQUIRE(m.rows_written() == 0);
    m.flush_rows(0, 2, true);
    REQUIRE(m.rows_written() == 2);
    m.set(3, 0, 7.0); // a row past the written ones
  }
  REQUIRE(read_at<std::uint64_t>(file_bytes(tmp), 56) == 2);
  const auto read = DistanceMatrix::read(tmp, 4, fingerprint);
  REQUIRE(read.get(0, 1) == 2.5);
  REQUIRE(std::isnan(read.get(0, 3))); // row 3 was never written: not computed, though the file holds 7
  const auto mapped = DistanceMatrix::map(tmp, 4, fingerprint);
  REQUIRE(mapped.rows_written() == 2);
  REQUIRE(mapped.count_computed() == 1);
  REQUIRE_FALSE(mapped.all_computed("test", false));
  const auto full = dtwc::io::to_full_matrix(mapped);
  REQUIRE(full[0 * 4 + 1] == 2.5);
  REQUIRE(std::isnan(full[3 * 4 + 0]));
  std::ostringstream csv;
  csv << mapped;
  REQUIRE(csv.str() == ",2.5,,\n2.5,,,\n,,,\n,,,\n");
}

TEST_CASE("A version-4 file is read and mapped as float64, and a mapped one stays version 4", "[DistanceMatrix][mmap]")
{
  // Version 4: a 48-byte header (bytes 0-47 of version 5's) and doubles, every row written.
  const ScratchDirectory tmp_dir{ "mmap_v4" };
  const fs::path tmp = tmp_dir.path / "v4.dtwm";
  const auto fingerprint = fingerprint_of(0x44);
  auto bytes = v5_header(3, fingerprint, 8, 0);
  bytes.resize(48);
  const std::uint32_t v4 = 4;
  std::memcpy(bytes.data() + 4, &v4, sizeof v4);
  const double values[] = { 0.0, 1.5, 0.0, std::nan(""), 2.25, 0.0 };
  for (const double v : values) bytes.insert(bytes.end(), reinterpret_cast<const char *>(&v),
                                             reinterpret_cast<const char *>(&v) + sizeof v);
  std::ofstream(tmp, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));

  const auto read = DistanceMatrix::read(tmp, 3, fingerprint);
  REQUIRE(read.rows_written() == 3);
  REQUIRE(read.get(1, 0) == 1.5);
  REQUIRE_FALSE(read.is_computed(2, 0));
  REQUIRE_THROWS_MATCHES(DistanceMatrix::read(tmp, 3, fingerprint, dtwc::core::Precision::Float32),
                         dtwc::InvalidInput, MessageMatches(ContainsSubstring("holds float64 distances")));
  {
    auto mapped = DistanceMatrix::map(tmp, 3, fingerprint);
    REQUIRE(mapped.rows_written() == 3);
    REQUIRE(mapped.get(2, 1) == 2.25);
    mapped.set(2, 0, 4.0);
    mapped.write(tmp, fingerprint); // its own file: flushed in place
  }
  const auto after = file_bytes(tmp);
  REQUIRE(after.size() == 48 + 6 * sizeof(double));
  REQUIRE(read_at<std::uint32_t>(after, 4) == 4);
  REQUIRE(read_at<double>(after, 48 + 3 * sizeof(double)) == 4.0);
}

TEST_CASE("A file of the other width is InvalidInput naming both; a bad header word is IOError",
          "[DistanceMatrix][mmap][error]")
{
  const ScratchDirectory tmp_dir{ "mmap_width" };
  const auto fingerprint = fingerprint_of(0x66);
  const fs::path f64 = tmp_dir.path / "f64.dtwm", f32 = tmp_dir.path / "f32.dtwm";
  DistanceMatrix(4, dtwc::core::Precision::Float64).write(f64, fingerprint);
  DistanceMatrix(4, dtwc::core::Precision::Float32).write(f32, fingerprint);
  REQUIRE(fs::file_size(f64) == 64 + 10 * 8);
  REQUIRE(fs::file_size(f32) == 64 + 10 * 4);
  REQUIRE_THROWS_MATCHES(DistanceMatrix::read(f64, 4, fingerprint, dtwc::core::Precision::Float32), dtwc::InvalidInput,
                         MessageMatches(ContainsSubstring("holds float64 distances, but this Problem stores float32")
                                        && ContainsSubstring("set matrix_dtype float64")));
  REQUIRE_THROWS_MATCHES(DistanceMatrix::map(f32, 4, fingerprint), dtwc::InvalidInput,
                         MessageMatches(ContainsSubstring("holds float32 distances, but this Problem stores float64")));
  REQUIRE(DistanceMatrix::map(f32, 4, fingerprint, dtwc::core::Precision::Float32).width()
          == dtwc::core::Precision::Float32);

  auto bad = file_bytes(f32);
  const std::uint32_t three = 3;
  std::memcpy(bad.data() + 48, &three, sizeof three); // three bytes a value
  std::ofstream(f32, std::ios::binary | std::ios::trunc).write(bad.data(), static_cast<std::streamsize>(bad.size()));
  REQUIRE_THROWS_AS(DistanceMatrix::read(f32, 4, fingerprint, dtwc::core::Precision::Float32), dtwc::IOError);
  const std::uint32_t four = 4;
  const std::uint64_t past_n = 5;
  std::memcpy(bad.data() + 48, &four, sizeof four);
  std::memcpy(bad.data() + 56, &past_n, sizeof past_n); // more rows written than N
  std::ofstream(f32, std::ios::binary | std::ios::trunc).write(bad.data(), static_cast<std::streamsize>(bad.size()));
  REQUIRE_THROWS_AS(DistanceMatrix::read(f32, 4, fingerprint, dtwc::core::Precision::Float32), dtwc::IOError);
}

TEST_CASE("A mapped matrix persists bit for bit and reopens with its values", "[DistanceMatrix][mmap]")
{
  const ScratchDirectory tmp_dir{ "mmap_persist" };
  const fs::path tmp = tmp_dir.path / "persist.dtwm";
  const auto fingerprint = fingerprint_of(0x11);
  const std::vector<double> values{ 0.0, -0.0, 1.5, -2.25, 1e300, 5e-324, 7.0 };
  const std::size_t n = 5;
  {
    auto m = DistanceMatrix::map(tmp, n, fingerprint);
    m.clear_rows(0, n);
    m.visit([&](auto *packed) {
      for (std::size_t k = 0; k < values.size(); ++k)
        packed[k] = static_cast<std::remove_cvref_t<decltype(*packed)>>(values[k]);
    });
    m.set(4, 3, 42.0);
    m.flush_rows(0, n, true); // on the device: every row written
  }
  const auto bytes = file_bytes(tmp);
  for (std::size_t k = 0; k < values.size(); ++k)
    REQUIRE(std::bit_cast<std::uint64_t>(read_at<double>(bytes, 64 + k * sizeof(double)))
            == std::bit_cast<std::uint64_t>(values[k]));

  // One layout, two routes. The stream read comes first: Windows refuses a
  // second open that does not share delete access while llfio holds the file.
  const auto streamed = DistanceMatrix::read(tmp, n, fingerprint);
  const auto reopened = DistanceMatrix::map(tmp, n, fingerprint);
  REQUIRE_FALSE(streamed.is_mapped());
  const auto reopened_bits = bits_of(reopened);
  REQUIRE(reopened_bits == bits_of(streamed));
  for (std::size_t k = 0; k < values.size(); ++k)
    REQUIRE(reopened_bits[k] == std::bit_cast<std::uint64_t>(values[k]));
  REQUIRE(reopened.get(3, 4) == 42.0);
  REQUIRE(reopened.count_computed() == values.size() + 1);
}

TEST_CASE("Writing a matrix over the file it maps flushes it in place", "[DistanceMatrix][mmap]")
{
  // A rename over a mapped file fails on Windows and would orphan the map on POSIX.
  const ScratchDirectory tmp_dir{ "mmap_write_self" };
  const fs::path tmp = tmp_dir.path / "write_self.dtwm";
  const auto fingerprint = fingerprint_of(0x22);
  {
    auto m = DistanceMatrix::map(tmp, 3, fingerprint);
    m.clear_rows(0, 3);
    m.flush_rows(0, 3, true); // every row written, NaN
    m.set(2, 0, 9.5);
    REQUIRE_NOTHROW(m.write(tmp, fingerprint));
    REQUIRE(m.is_mapped());
    REQUIRE_FALSE(fs::exists(fs::path(tmp).concat(".tmp")));
    m.set(2, 1, 3.0); // still the same mapping
  }
  const auto read_back = DistanceMatrix::read(tmp, 3, fingerprint);
  REQUIRE(read_back.get(0, 2) == 9.5);
  REQUIRE(read_back.get(1, 2) == 3.0);
}

TEST_CASE("A moved or resized matrix lets go of its file", "[DistanceMatrix][mmap]")
{
  const ScratchDirectory tmp_dir{ "mmap_release" };
  const fs::path tmp = tmp_dir.path / "release.dtwm";
  auto m = DistanceMatrix::map(tmp, 4, {});
  m.set(1, 0, 2.0);
  DistanceMatrix moved(std::move(m));
  REQUIRE(moved.is_mapped());
  REQUIRE(moved.get(0, 1) == 2.0);
  REQUIRE(m.size() == 0); // NOLINT(bugprone-use-after-move): the moved-from state is the subject
  m.visit([](const auto *values) { REQUIRE(values == nullptr); });
  REQUIRE_FALSE(m.is_mapped());

  moved.resize(4);
  REQUIRE_FALSE(moved.is_mapped());
  REQUIRE(moved.count_computed() == 0);
  // With the handle closed the file can go on every platform.
  std::error_code ec;
  REQUIRE(fs::remove(tmp, ec));
  REQUIRE_FALSE(ec);
}

TEST_CASE("Zero and one series map and reopen", "[DistanceMatrix][mmap]")
{
  const ScratchDirectory zero_dir{ "mmap_n0" };
  const fs::path zero = zero_dir.path / "n0.dtwm";
  const ScratchDirectory one_dir{ "mmap_n1" };
  const fs::path one = one_dir.path / "n1.dtwm";
  { auto m = DistanceMatrix::map(zero, 0, {}); }
  {
    auto m = DistanceMatrix::map(one, 1, {});
    m.set(0, 0, 0.0);
  }
  REQUIRE(fs::file_size(zero) == 64);
  REQUIRE(fs::file_size(one) == 72);
  REQUIRE(DistanceMatrix::map(zero, 0, {}).size() == 0);
  REQUIRE(DistanceMatrix::map(one, 1, {}).get(0, 0) == 0.0);
}

// A crafted N whose packed size wraps in 64 bits (2^62: N(N+1)/2 * 8 == 0 mod
// 2^64) must not pass the length check on a header-only file.
TEST_CASE("open rejects N that overflows packed size", "[DistanceMatrix][mmap][security]")
{
  const ScratchDirectory tmp_dir{ "mmap_overflow" };
  const fs::path tmp = tmp_dir.path / "overflow.dtwm";
  std::array<char, 48> header{};
  std::memcpy(header.data(), "DTWM", 4);
  const std::uint32_t version = 4;
  std::memcpy(header.data() + 4, &version, sizeof version);
  const std::uint64_t bad_n = std::uint64_t{ 1 } << 62;
  std::memcpy(header.data() + 8, &bad_n, sizeof bad_n);
  std::ofstream(tmp, std::ios::binary).write(header.data(), header.size());

  REQUIRE_THROWS_MATCHES(DistanceMatrix::map(tmp, 3, {}), dtwc::IOError,
                         MessageMatches(ContainsSubstring("not the size")));
  REQUIRE_THROWS_AS(DistanceMatrix::read(tmp, 3, {}), dtwc::IOError);
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
  const ScratchDirectory cache_dir{ "mmap_other_data" };
  const fs::path cache = cache_dir.path / "other_data.dtwm";
  const fs::path csv = fs::path(cache).replace_extension(".csv");
  {
    auto writer = problem({ { 0, 1 }, { 1, 2 }, { 2, 4 }, { 9, 9 } });
    writer.use_mmap_distance_matrix(cache);
    writer.fill_distance_matrix();
  }
  std::ofstream(csv) << "0,1,2,3\n1,0,1,2\n2,1,0,1\n3,2,1,0\n";

  auto reader = problem({ { 0, 1, 2 }, { 1, 2, 3 }, { 5, 6, 7 } });
  REQUIRE_THROWS_AS(reader.read_distance_matrix(csv), dtwc::InvalidInput);
  REQUIRE_THROWS_AS(reader.use_mmap_distance_matrix(cache), dtwc::InvalidInput);
  REQUIRE_FALSE(reader.distance_matrix().is_mapped());
  auto same_n = problem({ { 0, 1 }, { 1, 2 }, { 2, 4 }, { 9, 8 } }); // one value differs
  REQUIRE_THROWS_MATCHES(same_n.use_mmap_distance_matrix(cache), dtwc::InvalidInput,
                         MessageMatches(ContainsSubstring("fingerprint mismatch")));
  fs::remove(csv);
}

#ifdef _WIN32
TEST_CASE("A new mapped matrix is not a sparse file on Windows", "[DistanceMatrix][mmap]")
{
  // llfio's default sets FILE_ATTRIBUTE_SPARSE_FILE, and random reads from a
  // filled sparse file measured 1.9x slower.
  const ScratchDirectory tmp_dir{ "mmap_sparse" };
  const fs::path tmp = tmp_dir.path / "sparse.dtwm";
  {
    auto m = DistanceMatrix::map(tmp, 50, {});
    m.set(3, 7, 42.0);
    m.sync();
  }
  const DWORD attributes = GetFileAttributesW(tmp.c_str());
  REQUIRE(attributes != INVALID_FILE_ATTRIBUTES);
  REQUIRE((attributes & FILE_ATTRIBUTE_SPARSE_FILE) == 0);
  REQUIRE(DistanceMatrix::map(tmp, 50, {}).get(7, 3) == 42.0);
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
  // dtwc:runtime; the contract is IOError.
  const ScratchDirectory tmp_dir{ "mmap_quota" };
  const fs::path tmp = tmp_dir.path / "quota.dtwm";
  const FileSizeLimit limit(64 * 1024);
  CHECK_THROWS_MATCHES(DistanceMatrix::map(tmp, 200, {}), dtwc::IOError, // about 160 KB
                       MessageMatches(ContainsSubstring(tmp.filename().string())));
}
#endif

#endif // DTWC_HAS_MMAP
