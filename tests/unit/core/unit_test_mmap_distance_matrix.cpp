/**
 * @file unit_test_mmap_distance_matrix.cpp
 * @brief Unit tests for MmapDistanceMatrix class (memory-mapped distance matrix).
 *
 * @date 08 Apr 2026
 */

#include <core/mmap_distance_matrix.hpp>
#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <array>
#include <atomic>
#include <barrier>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc::core;
namespace fs = std::filesystem;

#ifndef DTWC_HAS_MMAP

TEST_CASE("MmapDistanceMatrix tests require LLFIO", "[MmapDistanceMatrix][mmap]")
{
  SKIP("mmap support not compiled in (DTWC_ENABLE_LLFIO=OFF)");
}

#else

namespace {

/// RAII helper to create a unique temp file path and remove it on destruction.
struct TempFile {
  fs::path path;

  TempFile()
  {
    // Generate a unique filename in the system temp directory
    path = fs::temp_directory_path() / ("dtwc_mmap_test_" + std::to_string(reinterpret_cast<uintptr_t>(this)) + ".bin");
    // Ensure no leftover from a previous failed run
    fs::remove(path);
  }

  ~TempFile()
  {
    std::error_code ec;
    fs::remove(path, ec); // best-effort cleanup
  }

  TempFile(const TempFile &) = delete;
  TempFile &operator=(const TempFile &) = delete;
};

std::vector<std::uint8_t> read_file_bytes(const fs::path &path)
{
  std::ifstream input(path, std::ios::binary);
  REQUIRE(input.is_open());
  return {std::istreambuf_iterator<char>{input},
          std::istreambuf_iterator<char>{}};
}

void write_file_bytes(const fs::path &path,
                      const std::vector<std::uint8_t> &bytes)
{
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  REQUIRE(output.is_open());
  output.write(reinterpret_cast<const char *>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  REQUIRE(output.good());
}

std::uint64_t payload_bits(const fs::path &path, std::size_t packed_index)
{
  const auto bytes = read_file_bytes(path);
  const std::size_t offset = MmapDistanceMatrix::header_size
                           + packed_index * sizeof(double);
  REQUIRE(offset + sizeof(std::uint64_t) <= bytes.size());
  std::uint64_t bits{};
  std::memcpy(&bits, bytes.data() + offset, sizeof(bits));
  return bits;
}

void write_payload_bits(const fs::path &path, std::size_t packed_index,
                        std::uint64_t bits)
{
  auto bytes = read_file_bytes(path);
  const std::size_t offset = MmapDistanceMatrix::header_size
                           + packed_index * sizeof(double);
  REQUIRE(offset + sizeof(bits) <= bytes.size());
  std::memcpy(bytes.data() + offset, &bits, sizeof(bits));
  write_file_bytes(path, bytes);
}

struct OpenAttempt
{
  bool returned{false};
  bool value_exposed{false};
  std::string error;
};

OpenAttempt try_open_and_read(
  const fs::path &path,
  const MmapDistanceMatrix::fingerprint_type &fingerprint,
  std::size_t i, std::size_t j)
{
  OpenAttempt result;
  try {
    auto matrix = MmapDistanceMatrix::open(path, fingerprint);
    result.returned = true;
    (void)matrix.get(i, j);
    result.value_exposed = true;
  } catch (const std::exception &error) {
    result.error = error.what();
  } catch (...) {
    result.error = "non-standard exception";
  }
  return result;
}

void require_payload_rejected_without_file_mutation(
  const fs::path &path,
  const MmapDistanceMatrix::fingerprint_type &fingerprint,
  std::size_t i, std::size_t j,
  const std::vector<std::uint8_t> &corrupted_bytes)
{
  const OpenAttempt attempt = try_open_and_read(path, fingerprint, i, j);
  INFO("open error: " << attempt.error);
  CHECK_FALSE(attempt.returned);
  CHECK_FALSE(attempt.value_exposed);
  CHECK(attempt.error.find("payload integrity") != std::string::npos);
  CHECK(read_file_bytes(path) == corrupted_bytes);
}

void require_header_rejected_without_file_mutation(
  const fs::path &path,
  const MmapDistanceMatrix::fingerprint_type &fingerprint,
  std::string_view expected_error,
  const std::vector<std::uint8_t> &corrupted_bytes)
{
  const OpenAttempt attempt = try_open_and_read(path, fingerprint, 0, 0);
  INFO("open error: " << attempt.error);
  CHECK_FALSE(attempt.returned);
  CHECK_FALSE(attempt.value_exposed);
  CHECK(attempt.error.find(expected_error) != std::string::npos);
  CHECK(read_file_bytes(path) == corrupted_bytes);
}

dtwc::Data make_problem_data(std::size_t n)
{
  std::vector<std::vector<double>> series;
  std::vector<std::string> names;
  series.reserve(n);
  names.reserve(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double value = static_cast<double>(i);
    series.push_back({value, value + 1.0, value + 3.0, value + 6.0});
    names.push_back("s" + std::to_string(i));
  }
  return dtwc::Data(std::move(series), std::move(names));
}

struct DenseSentinel
{
  const double *address{};
  std::uint64_t bits{};
};

DenseSentinel install_dense_sentinel(dtwc::Problem &problem, double value)
{
  auto &matrix = problem.dense_distance_matrix();
  matrix.resize(problem.size());
  matrix.set(0, 1, value);
  return {matrix.raw(), std::bit_cast<std::uint64_t>(value)};
}

void require_problem_bind_rejected_transactionally(
  dtwc::Problem &problem, const fs::path &path,
  const DenseSentinel &sentinel,
  const std::vector<std::uint8_t> &corrupted_bytes)
{
  bool threw = false;
  std::string error;
  try {
    problem.use_mmap_distance_matrix(path);
  } catch (const std::exception &exception) {
    threw = true;
    error = exception.what();
  } catch (...) {
    threw = true;
    error = "non-standard exception";
  }
  INFO("bind error: " << error);
  CHECK(threw);
  CHECK(error.find("payload integrity") != std::string::npos);
  const bool remains_dense = std::holds_alternative<DenseDistanceMatrix>(
    problem.distance_matrix());
  CHECK(remains_dense);
  if (remains_dense) {
    const auto &matrix = problem.dense_distance_matrix();
    CHECK(matrix.raw() == sentinel.address);
    CHECK(std::bit_cast<std::uint64_t>(matrix.get(0, 1)) == sentinel.bits);
  }
  CHECK(read_file_bytes(path) == corrupted_bytes);
}

std::string mmap_source()
{
  const auto repo_root = fs::path{DTWC_TEST_DATA_DIR}.parent_path();
  std::ifstream source(repo_root / "dtwc" / "core" / "mmap_distance_matrix.hpp",
                       std::ios::binary);
  REQUIRE(source.is_open());
  return {std::istreambuf_iterator<char>{source},
          std::istreambuf_iterator<char>{}};
}

} // namespace

// ============================================================================
// Basic operations
// ============================================================================

TEST_CASE("MmapDistanceMatrix create N=10", "[MmapDistanceMatrix]")
{
  TempFile tmp;
  MmapDistanceMatrix dm(tmp.path, 10);

  REQUIRE(dm.size() == 10);
  REQUIRE(dm.packed_count() == 10 * 11 / 2); // 55
}

TEST_CASE("MmapDistanceMatrix all entries uncomputed after creation", "[MmapDistanceMatrix]")
{
  TempFile tmp;
  MmapDistanceMatrix dm(tmp.path, 10);

  for (size_t i = 0; i < 10; ++i)
    for (size_t j = 0; j < 10; ++j)
      REQUIRE_FALSE(dm.is_computed(i, j));

  REQUIRE(dm.count_computed() == 0);
  REQUIRE_FALSE(dm.all_computed());
}

// ============================================================================
// Set and get with symmetry
// ============================================================================

TEST_CASE("MmapDistanceMatrix set and get with symmetry", "[MmapDistanceMatrix]")
{
  TempFile tmp;
  MmapDistanceMatrix dm(tmp.path, 10);

  dm.set(3, 7, 42.0);
  REQUIRE_THAT(dm.get(3, 7), WithinAbs(42.0, 1e-12));
  REQUIRE_THAT(dm.get(7, 3), WithinAbs(42.0, 1e-12));
  REQUIRE(dm.is_computed(3, 7));
  REQUIRE(dm.is_computed(7, 3));
}

// ============================================================================
// Diagonal
// ============================================================================

TEST_CASE("MmapDistanceMatrix diagonal set(0,0,0.0)", "[MmapDistanceMatrix]")
{
  TempFile tmp;
  MmapDistanceMatrix dm(tmp.path, 5);

  dm.set(0, 0, 0.0);
  REQUIRE(dm.is_computed(0, 0));
  REQUIRE_THAT(dm.get(0, 0), WithinAbs(0.0, 1e-12));
}

TEST_CASE("MmapDistanceMatrix all diagonal entries", "[MmapDistanceMatrix]")
{
  TempFile tmp;
  MmapDistanceMatrix dm(tmp.path, 4);

  for (size_t i = 0; i < 4; ++i)
    dm.set(i, i, 0.0);

  for (size_t i = 0; i < 4; ++i) {
    REQUIRE(dm.is_computed(i, i));
    REQUIRE_THAT(dm.get(i, i), WithinAbs(0.0, 1e-12));
  }
}

// ============================================================================
// max()
// ============================================================================

TEST_CASE("MmapDistanceMatrix max returns max of computed values", "[MmapDistanceMatrix]")
{
  TempFile tmp;
  MmapDistanceMatrix dm(tmp.path, 5);

  dm.set(0, 1, 2.0);
  dm.set(0, 2, 8.5);
  dm.set(1, 2, 4.0);

  REQUIRE_THAT(dm.max(), WithinAbs(8.5, 1e-12));
}

TEST_CASE("MmapDistanceMatrix max on unfilled matrix returns 0", "[MmapDistanceMatrix]")
{
  TempFile tmp;
  MmapDistanceMatrix dm(tmp.path, 5);

  REQUIRE_THAT(dm.max(), WithinAbs(0.0, 1e-12));
}

// ============================================================================
// count_computed and all_computed
// ============================================================================

TEST_CASE("MmapDistanceMatrix count_computed and all_computed", "[MmapDistanceMatrix]")
{
  TempFile tmp;
  MmapDistanceMatrix dm(tmp.path, 3);

  REQUIRE(dm.count_computed() == 0);
  REQUIRE_FALSE(dm.all_computed());

  // Fill all 6 entries: (0,0), (0,1), (0,2), (1,1), (1,2), (2,2)
  dm.set(0, 0, 0.0);
  dm.set(0, 1, 1.0);
  dm.set(0, 2, 2.0);
  dm.set(1, 1, 0.0);
  dm.set(1, 2, 3.0);
  dm.set(2, 2, 0.0);

  REQUIRE(dm.count_computed() == 6);
  REQUIRE(dm.all_computed());
}

// ============================================================================
// Persistence (warm-start)
// ============================================================================

TEST_CASE("MmapDistanceMatrix persistence: create, write, sync, destroy, reopen", "[MmapDistanceMatrix]")
{
  TempFile tmp;

  // Phase 1: create and write values
  {
    MmapDistanceMatrix dm(tmp.path, 10);
    dm.set(0, 0, 0.0);
    dm.set(3, 7, 42.0);
    dm.set(5, 9, 99.5);
    dm.set(0, 9, 1.25);
    dm.sync();
  } // dm destroyed here, file remains

  // Phase 2: reopen and verify values persist
  {
    auto dm = MmapDistanceMatrix::open(tmp.path);
    REQUIRE(dm.size() == 10);
    REQUIRE(dm.packed_count() == 55);

    // Verify written values
    REQUIRE_THAT(dm.get(0, 0), WithinAbs(0.0, 1e-12));
    REQUIRE_THAT(dm.get(3, 7), WithinAbs(42.0, 1e-12));
    REQUIRE_THAT(dm.get(7, 3), WithinAbs(42.0, 1e-12)); // symmetry
    REQUIRE_THAT(dm.get(5, 9), WithinAbs(99.5, 1e-12));
    REQUIRE_THAT(dm.get(0, 9), WithinAbs(1.25, 1e-12));

    // Verify unwritten values are still uncomputed
    REQUIRE_FALSE(dm.is_computed(1, 2));
    REQUIRE_FALSE(dm.is_computed(4, 6));

    REQUIRE(dm.count_computed() == 4);
  }
}

TEST_CASE("MmapDistanceMatrix persistence: incremental warm-start", "[MmapDistanceMatrix]")
{
  TempFile tmp;

  // Phase 1: write some values
  {
    MmapDistanceMatrix dm(tmp.path, 5);
    dm.set(0, 1, 10.0);
    dm.set(2, 3, 20.0);
    dm.sync();
  }

  // Phase 2: reopen, add more values
  {
    auto dm = MmapDistanceMatrix::open(tmp.path);
    REQUIRE_THAT(dm.get(0, 1), WithinAbs(10.0, 1e-12));
    REQUIRE_THAT(dm.get(2, 3), WithinAbs(20.0, 1e-12));

    dm.set(3, 4, 30.0);
    dm.set(0, 0, 0.0);
    dm.sync();
  }

  // Phase 3: verify all values
  {
    auto dm = MmapDistanceMatrix::open(tmp.path);
    REQUIRE_THAT(dm.get(0, 1), WithinAbs(10.0, 1e-12));
    REQUIRE_THAT(dm.get(2, 3), WithinAbs(20.0, 1e-12));
    REQUIRE_THAT(dm.get(3, 4), WithinAbs(30.0, 1e-12));
    REQUIRE_THAT(dm.get(0, 0), WithinAbs(0.0, 1e-12));
    REQUIRE(dm.count_computed() == 4);
  }
}

// ============================================================================
// Mutable payload integrity (M53 preregistration)
// ============================================================================

TEST_CASE("Authenticated mmap payload requires a new format version",
          "[MmapDistanceMatrix][mmap][integrity][version][m53]")
{
  CHECK(MmapDistanceMatrix::version >= 3);
}

TEST_CASE("MmapDistanceMatrix rejects a finite payload bit flip before exposure",
          "[MmapDistanceMatrix][mmap][integrity][m53]")
{
  TempFile tmp;
  MmapDistanceMatrix::fingerprint_type fingerprint{};
  fingerprint.fill(0x53u);
  const std::size_t index = tri_index(3, 1);

  {
    MmapDistanceMatrix matrix(tmp.path, 4, fingerprint);
    matrix.set(3, 1, 42.0);
    matrix.sync();
  }

  const auto pristine = read_file_bytes(tmp.path);
  const std::uint64_t original_bits = payload_bits(tmp.path, index);
  const std::uint64_t corrupted_bits = original_bits ^ std::uint64_t{1};
  REQUIRE(std::isfinite(std::bit_cast<double>(corrupted_bits)));
  REQUIRE(corrupted_bits != original_bits);
  write_payload_bits(tmp.path, index, corrupted_bits);
  const auto corrupted = read_file_bytes(tmp.path);
  REQUIRE(corrupted != pristine);

  require_payload_rejected_without_file_mutation(
    tmp.path, fingerprint, 3, 1, corrupted);
}

TEST_CASE("MmapDistanceMatrix authenticates NaN computed-state transitions",
          "[MmapDistanceMatrix][mmap][integrity][status][m53]")
{
  MmapDistanceMatrix::fingerprint_type fingerprint{};
  fingerprint.fill(0x35u);
  const std::size_t index = tri_index(1, 0);

  SECTION("one bit turns an uncomputed NaN into a finite trusted value")
  {
    TempFile tmp;
    {
      MmapDistanceMatrix matrix(tmp.path, 2, fingerprint);
      matrix.sync();
    }

    const auto pristine = read_file_bytes(tmp.path);
    const std::uint64_t original_bits = payload_bits(tmp.path, index);
    REQUIRE(std::isnan(std::bit_cast<double>(original_bits)));

    std::optional<std::uint64_t> finite_bits;
    for (unsigned bit = 0; bit < 64; ++bit) {
      const std::uint64_t candidate = original_bits ^ (std::uint64_t{1} << bit);
      if (std::isfinite(std::bit_cast<double>(candidate))) {
        finite_bits = candidate;
        break;
      }
    }
    REQUIRE(finite_bits.has_value());
    REQUIRE(std::popcount(original_bits ^ *finite_bits) == 1);
    write_payload_bits(tmp.path, index, *finite_bits);
    const auto corrupted = read_file_bytes(tmp.path);
    REQUIRE(corrupted != pristine);

    require_payload_rejected_without_file_mutation(
      tmp.path, fingerprint, 1, 0, corrupted);
  }

  SECTION("one bit turns a computed finite value into uncomputed NaN")
  {
    TempFile tmp;
    constexpr std::uint64_t finite_bits = 0x7fe8000000000000ull;
    constexpr std::uint64_t status_bit = std::uint64_t{1} << 52;
    const double finite_value = std::bit_cast<double>(finite_bits);
    REQUIRE(std::isfinite(finite_value));
    REQUIRE(std::isnan(std::bit_cast<double>(finite_bits ^ status_bit)));

    {
      MmapDistanceMatrix matrix(tmp.path, 2, fingerprint);
      matrix.set(1, 0, finite_value);
      matrix.sync();
    }

    const auto pristine = read_file_bytes(tmp.path);
    REQUIRE(payload_bits(tmp.path, index) == finite_bits);
    write_payload_bits(tmp.path, index, finite_bits ^ status_bit);
    const auto corrupted = read_file_bytes(tmp.path);
    REQUIRE(corrupted != pristine);

    require_payload_rejected_without_file_mutation(
      tmp.path, fingerprint, 1, 0, corrupted);
  }
}

TEST_CASE("Problem lazy mmap warm-start validates payload before cache publication",
          "[MmapDistanceMatrix][mmap][integrity][problem][lazy][m53]")
{
  TempFile tmp;
  double expected{};
  {
    dtwc::Problem source{"m53_lazy_source"};
    source.set_data(make_problem_data(4));
    source.use_mmap_distance_matrix(tmp.path);
    expected = source.dist_by_ind(0, 1);
    auto &mapped = std::get<MmapDistanceMatrix>(source.distance_matrix());
    mapped.sync();
    REQUIRE(mapped.count_computed() == 1);
  }

  // Valid explicit-sync warm reopen remains unchanged.
  {
    dtwc::Problem control{"m53_lazy_control"};
    control.set_data(make_problem_data(4));
    control.use_mmap_distance_matrix(tmp.path);
    REQUIRE(std::bit_cast<std::uint64_t>(control.dist_by_ind(0, 1))
            == std::bit_cast<std::uint64_t>(expected));
    REQUIRE_FALSE(std::get<MmapDistanceMatrix>(control.distance_matrix())
                    .is_computed(0, 2));
  }

  const auto pristine = read_file_bytes(tmp.path);
  const std::size_t index = tri_index(1, 0);
  const std::uint64_t original_bits = payload_bits(tmp.path, index);
  write_payload_bits(tmp.path, index, original_bits ^ std::uint64_t{1});
  const auto corrupted = read_file_bytes(tmp.path);
  REQUIRE(corrupted != pristine);

  dtwc::Problem target{"m53_lazy_target"};
  target.set_data(make_problem_data(4));
  const DenseSentinel sentinel = install_dense_sentinel(target, 753.0);
  require_problem_bind_rejected_transactionally(
    target, tmp.path, sentinel, corrupted);
}

TEST_CASE("Problem full parallel mmap fill authenticates every persisted value",
          "[MmapDistanceMatrix][mmap][integrity][problem][parallel][m53]")
{
  TempFile tmp;
  constexpr std::size_t n = 12;
  double expected{};
  {
    dtwc::Problem source{"m53_full_source"};
    source.set_data(make_problem_data(n));
    source.set_distance_strategy(dtwc::DistanceMatrixStrategy::BruteForce);
    source.use_mmap_distance_matrix(tmp.path);
    source.fill_distance_matrix();
    auto &mapped = std::get<MmapDistanceMatrix>(source.distance_matrix());
    REQUIRE(mapped.all_computed());
    expected = mapped.get(0, n - 1);
    mapped.sync();
  }

  // A complete parallel fill with a consistent digest is a valid warm start.
  {
    dtwc::Problem control{"m53_full_control"};
    control.set_data(make_problem_data(n));
    control.set_distance_strategy(dtwc::DistanceMatrixStrategy::BruteForce);
    control.use_mmap_distance_matrix(tmp.path);
    REQUIRE(control.is_distance_matrix_filled());
    REQUIRE(std::bit_cast<std::uint64_t>(control.dist_by_ind(0, n - 1))
            == std::bit_cast<std::uint64_t>(expected));
  }

  const auto pristine = read_file_bytes(tmp.path);
  const std::size_t index = tri_index(n - 1, 0);
  const std::uint64_t original_bits = payload_bits(tmp.path, index);
  write_payload_bits(tmp.path, index, original_bits ^ std::uint64_t{1});
  const auto corrupted = read_file_bytes(tmp.path);
  REQUIRE(corrupted != pristine);

  dtwc::Problem target{"m53_full_target"};
  target.set_data(make_problem_data(n));
  target.set_distance_strategy(dtwc::DistanceMatrixStrategy::BruteForce);
  const DenseSentinel sentinel = install_dense_sentinel(target, 953.0);
  require_problem_bind_rejected_transactionally(
    target, tmp.path, sentinel, corrupted);
}

TEST_CASE("Mmap payload authentication survives normal destructor persistence",
          "[MmapDistanceMatrix][mmap][integrity][durability][m53]")
{
  TempFile tmp;
  MmapDistanceMatrix::fingerprint_type fingerprint{};
  fingerprint.fill(0x19u);
  {
    MmapDistanceMatrix matrix(tmp.path, 4, fingerprint);
    matrix.set(0, 3, 19.5);
    matrix.set(1, 2, 29.5);
    // Intentionally no explicit sync: ordinary RAII teardown is a supported
    // warm-reopen path and must persist data and integrity consistently.
  }

  const auto reopened = MmapDistanceMatrix::open(tmp.path, fingerprint);
  REQUIRE(std::bit_cast<std::uint64_t>(reopened.get(0, 3))
          == std::bit_cast<std::uint64_t>(19.5));
  REQUIRE(std::bit_cast<std::uint64_t>(reopened.get(1, 2))
          == std::bit_cast<std::uint64_t>(29.5));
  REQUIRE(reopened.count_computed() == 2);
}

TEST_CASE("Mmap disjoint parallel sets retain O(1) integrity updates",
          "[MmapDistanceMatrix][mmap][integrity][parallel][m53]")
{
  TempFile tmp;
  constexpr std::size_t n = 64;
  constexpr std::size_t worker_count = 4;
  MmapDistanceMatrix::fingerprint_type fingerprint{};
  fingerprint.fill(0xa5u);

  {
    MmapDistanceMatrix matrix(tmp.path, n, fingerprint);
    std::array<std::thread, worker_count> workers;
    for (std::size_t worker = 0; worker < worker_count; ++worker) {
      workers[worker] = std::thread([&, worker] {
        for (std::size_t i = worker; i < n; i += worker_count) {
          for (std::size_t j = 0; j <= i; ++j) {
            matrix.set(i, j, static_cast<double>(tri_index(i, j)) + 0.25);
          }
        }
      });
    }
    for (auto &worker : workers) worker.join();
    matrix.sync();
  }

  const auto reopened = MmapDistanceMatrix::open(tmp.path, fingerprint);
  REQUIRE(reopened.all_computed());
  REQUIRE(reopened.count_computed() == packed_size(n));
  REQUIRE(std::bit_cast<std::uint64_t>(reopened.get(63, 0))
          == std::bit_cast<std::uint64_t>(
               static_cast<double>(tri_index(63, 0)) + 0.25));
  REQUIRE(std::bit_cast<std::uint64_t>(reopened.get(42, 17))
          == std::bit_cast<std::uint64_t>(
               static_cast<double>(tri_index(42, 17)) + 0.25));
}

TEST_CASE("Mmap set path contains no full scan or blocking lock",
          "[MmapDistanceMatrix][mmap][integrity][source_guard][m53]")
{
  const std::string source = mmap_source();
  const std::size_t begin = source.find("void set(size_t i, size_t j, double v)");
  const std::size_t end = source.find("bool is_computed", begin);
  REQUIRE(begin != std::string::npos);
  REQUIRE(end != std::string::npos);
  const std::string set_body = source.substr(begin, end - begin);

  CHECK(set_body.find("for (") == std::string::npos);
  CHECK(set_body.find("while (") == std::string::npos);
  CHECK(set_body.find("mutex") == std::string::npos);
  CHECK(set_body.find("lock_guard") == std::string::npos);
  CHECK(set_body.find("sync()") == std::string::npos);
}

TEST_CASE("Mmap immutable header checksum failures remain loud and non-mutating",
          "[MmapDistanceMatrix][mmap][integrity][header][m53]")
{
  MmapDistanceMatrix::fingerprint_type fingerprint{};
  fingerprint.fill(0x71u);

  SECTION("stored header CRC bit")
  {
    TempFile tmp;
    {
      MmapDistanceMatrix matrix(tmp.path, 2, fingerprint);
      matrix.set(0, 1, 7.0);
      matrix.sync();
    }
    auto corrupted = read_file_bytes(tmp.path);
    REQUIRE(corrupted.size() >= MmapDistanceMatrix::header_size);
    corrupted[60] ^= 0x01u;
    write_file_bytes(tmp.path, corrupted);
    require_header_rejected_without_file_mutation(
      tmp.path, fingerprint, "header CRC mismatch", corrupted);
  }

  SECTION("fingerprint field with repaired header CRC")
  {
    TempFile tmp;
    {
      MmapDistanceMatrix matrix(tmp.path, 2, fingerprint);
      matrix.set(0, 1, 7.0);
      matrix.sync();
    }
    auto corrupted = read_file_bytes(tmp.path);
    REQUIRE(corrupted.size() >= MmapDistanceMatrix::header_size);
    corrupted[20] ^= 0x01u;
    const std::uint32_t repaired_crc = detail::crc32_naive(corrupted.data(), 60);
    std::memcpy(corrupted.data() + 60, &repaired_crc, sizeof(repaired_crc));
    write_file_bytes(tmp.path, corrupted);
    require_header_rejected_without_file_mutation(
      tmp.path, fingerprint, "fingerprint mismatch", corrupted);
  }
}

// ============================================================================
// Edge cases: N=0, N=1
// ============================================================================

TEST_CASE("MmapDistanceMatrix N=0", "[MmapDistanceMatrix]")
{
  TempFile tmp;
  MmapDistanceMatrix dm(tmp.path, 0);

  REQUIRE(dm.size() == 0);
  REQUIRE(dm.packed_count() == 0);
  REQUIRE(dm.count_computed() == 0);
  REQUIRE(dm.all_computed()); // vacuously true
  REQUIRE_THAT(dm.max(), WithinAbs(0.0, 1e-12));
}

TEST_CASE("MmapDistanceMatrix N=1", "[MmapDistanceMatrix]")
{
  TempFile tmp;
  MmapDistanceMatrix dm(tmp.path, 1);

  REQUIRE(dm.size() == 1);
  REQUIRE(dm.packed_count() == 1);
  REQUIRE_FALSE(dm.is_computed(0, 0));

  dm.set(0, 0, 0.0);
  REQUIRE(dm.is_computed(0, 0));
  REQUIRE_THAT(dm.get(0, 0), WithinAbs(0.0, 1e-12));
  REQUIRE(dm.all_computed());
}

TEST_CASE("MmapDistanceMatrix N=1 persistence", "[MmapDistanceMatrix]")
{
  TempFile tmp;

  {
    MmapDistanceMatrix dm(tmp.path, 1);
    dm.set(0, 0, 0.0);
    dm.sync();
  }

  {
    auto dm = MmapDistanceMatrix::open(tmp.path);
    REQUIRE(dm.size() == 1);
    REQUIRE_THAT(dm.get(0, 0), WithinAbs(0.0, 1e-12));
    REQUIRE(dm.all_computed());
  }
}

// ============================================================================
// Large: N=1000
// ============================================================================

TEST_CASE("MmapDistanceMatrix large N=1000", "[MmapDistanceMatrix]")
{
  TempFile tmp;

  {
    MmapDistanceMatrix dm(tmp.path, 1000);
    REQUIRE(dm.size() == 1000);
    REQUIRE(dm.packed_count() == 1000 * 1001 / 2); // 500500

    // Set diagonal entries
    for (size_t i = 0; i < 1000; ++i)
      dm.set(i, i, 0.0);

    // Set a few off-diagonal entries
    dm.set(0, 999, 123.456);
    dm.set(500, 501, 789.0);
    dm.set(42, 777, 3.14159);

    REQUIRE(dm.count_computed() == 1003);
    dm.sync();
  }

  // Reopen and verify
  {
    auto dm = MmapDistanceMatrix::open(tmp.path);
    REQUIRE(dm.size() == 1000);

    // Check diagonal
    for (size_t i = 0; i < 1000; ++i)
      REQUIRE_THAT(dm.get(i, i), WithinAbs(0.0, 1e-12));

    // Check off-diagonal
    REQUIRE_THAT(dm.get(0, 999), WithinAbs(123.456, 1e-12));
    REQUIRE_THAT(dm.get(999, 0), WithinAbs(123.456, 1e-12)); // symmetry
    REQUIRE_THAT(dm.get(500, 501), WithinAbs(789.0, 1e-12));
    REQUIRE_THAT(dm.get(42, 777), WithinAbs(3.14159, 1e-12));

    // Uncomputed entry
    REQUIRE_FALSE(dm.is_computed(1, 2));

    REQUIRE(dm.count_computed() == 1003);
  }
}

// ============================================================================
// Raw pointer access
// ============================================================================

TEST_CASE("MmapDistanceMatrix raw pointer", "[MmapDistanceMatrix]")
{
  TempFile tmp;
  MmapDistanceMatrix dm(tmp.path, 2);

  dm.set(0, 0, 1.0);
  dm.set(0, 1, 2.0);
  dm.set(1, 1, 3.0);

  const double *raw = dm.raw();
  REQUIRE(dm.packed_count() == 3);
  // Packed: tri(0,0)=0, tri(1,0)=1, tri(1,1)=2
  REQUIRE_THAT(raw[0], WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(raw[1], WithinAbs(2.0, 1e-12));
  REQUIRE_THAT(raw[2], WithinAbs(3.0, 1e-12));
}

// ============================================================================
// Error handling
// ============================================================================

TEST_CASE("MmapDistanceMatrix open nonexistent file throws", "[MmapDistanceMatrix]")
{
  fs::path nonexistent = fs::temp_directory_path() / "dtwc_mmap_nonexistent_test_12345.bin";
  fs::remove(nonexistent); // ensure it doesn't exist
  REQUIRE_THROWS_AS(MmapDistanceMatrix::open(nonexistent), std::runtime_error);
}

// Regression for audit CRITICAL #4 (mmap_distance_matrix.hpp validate_header).
// Before the fix, validate_header computed `expected = header_size + packed_size(n)*8`
// with NO overflow guard. A crafted header with n = 2^62 makes packed_size(n) = 2^61,
// and 2^61 * 8 == 2^64 == 0 (mod 2^64), so `expected` wraps to header_size. The
// truncation check `file_len < expected` then PASSES on a header-only file, and open()
// returns a matrix reporting size()==2^62 backed by that mapping -> OOB reads.
// The unfixed code does NOT throw here; the fix routes validate_header through a
// checked file_size() that throws on the multiplication overflow.
TEST_CASE("MmapDistanceMatrix open rejects N that overflows packed size", "[MmapDistanceMatrix][security]")
{
  TempFile tmp;

  const uint64_t bad_n = uint64_t{ 1 } << 62; // packed_size = 2^61; *8 wraps to 0 mod 2^64

  // Build a v2 header that passes magic/version/endian/elem_size/fingerprint/CRC checks so
  // that the ONLY thing standing between the file and acceptance is the size check.
  std::array<uint8_t, MmapDistanceMatrix::header_size> hdr{};
  std::memcpy(hdr.data() + 0, MmapDistanceMatrix::magic, 4);
  const uint16_t ver = MmapDistanceMatrix::version;
  std::memcpy(hdr.data() + 4, &ver, 2);
  const uint32_t em = MmapDistanceMatrix::endian_marker;
  std::memcpy(hdr.data() + 6, &em, 4);
  hdr[10] = MmapDistanceMatrix::elem_size;
  hdr[11] = MmapDistanceMatrix::fingerprint_algorithm;
  std::memcpy(hdr.data() + 12, &bad_n, 8);
  hdr[MmapDistanceMatrix::publication_state_offset] =
    MmapDistanceMatrix::publication_state_ready;
  // Fingerprint and reserved bytes remain zero; only overflow is under test.
  const uint32_t crc = detail::crc32_naive(hdr.data(), 60);
  std::memcpy(hdr.data() + 60, &crc, 4);

  {
    std::ofstream f(tmp.path, std::ios::binary);
    f.write(reinterpret_cast<const char *>(hdr.data()), static_cast<std::streamsize>(hdr.size()));
  }

  REQUIRE_THROWS_AS(MmapDistanceMatrix::open(tmp.path), std::runtime_error);
}

TEST_CASE("MmapDistanceMatrix rejects corrupted fingerprint metadata before data access",
          "[MmapDistanceMatrix][mmap][fingerprint][security]")
{
  TempFile tmp;
  MmapDistanceMatrix::fingerprint_type fingerprint{};
  fingerprint.fill(0x5au);

  {
    MmapDistanceMatrix dm(tmp.path, 2, fingerprint);
    dm.set(0, 1, 123.0);
    dm.sync();
  }

  // Flip one fingerprint byte without repairing the CRC. Header integrity must
  // fail before open() can expose the persisted 123.0 computed entry.
  {
    std::fstream file(tmp.path, std::ios::in | std::ios::out | std::ios::binary);
    file.seekg(20);
    char byte{};
    file.read(&byte, 1);
    byte = static_cast<char>(static_cast<unsigned char>(byte) ^ 0x01u);
    file.seekp(20);
    file.write(&byte, 1);
  }

  REQUIRE_THROWS_WITH(
    MmapDistanceMatrix::open(tmp.path, fingerprint),
    Catch::Matchers::ContainsSubstring("header CRC mismatch"));
}

TEST_CASE("MmapDistanceMatrix rejects a well-formed unexpected fingerprint",
          "[MmapDistanceMatrix][mmap][fingerprint]")
{
  TempFile tmp;
  MmapDistanceMatrix::fingerprint_type stored{};
  MmapDistanceMatrix::fingerprint_type expected{};
  stored.fill(0x11u);
  expected.fill(0x22u);

  {
    MmapDistanceMatrix dm(tmp.path, 2, stored);
    dm.set(0, 1, 123.0);
    dm.sync();
  }

  REQUIRE_THROWS_WITH(
    MmapDistanceMatrix::open(tmp.path, expected),
    Catch::Matchers::ContainsSubstring("fingerprint mismatch"));

  // The convenience overload is not an unchecked escape hatch: it explicitly
  // expects the all-zero identity used by low-level unbound matrices.
  REQUIRE_THROWS_WITH(
    MmapDistanceMatrix::open(tmp.path),
    Catch::Matchers::ContainsSubstring("fingerprint mismatch"));
}

TEST_CASE("MmapDistanceMatrix rejects legacy version-1 cache headers loudly",
          "[MmapDistanceMatrix][mmap][version]")
{
  TempFile tmp;

  // The released pre-fingerprint layout was a 32-byte v1 header. N=0 makes
  // that header a complete valid v1 file, so accepting it would silently trust
  // metadata that cannot bind distances to their source data/configuration.
  std::array<uint8_t, 32> legacy{};
  std::memcpy(legacy.data(), MmapDistanceMatrix::magic, 4);
  const uint16_t legacy_version = 1;
  std::memcpy(legacy.data() + 4, &legacy_version, 2);
  const uint32_t endian = MmapDistanceMatrix::endian_marker;
  std::memcpy(legacy.data() + 6, &endian, 4);
  legacy[10] = MmapDistanceMatrix::elem_size;
  const uint64_t n = 0;
  std::memcpy(legacy.data() + 12, &n, 8);
  const uint32_t crc = detail::crc32_naive(legacy.data(), 20);
  std::memcpy(legacy.data() + 20, &crc, 4);

  {
    std::ofstream out(tmp.path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(legacy.data()),
              static_cast<std::streamsize>(legacy.size()));
  }

  REQUIRE_THROWS_WITH(
    MmapDistanceMatrix::open(tmp.path),
    Catch::Matchers::ContainsSubstring("unsupported version 1"));
}

TEST_CASE("MmapDistanceMatrix rejects a published header whose data initialization is incomplete",
          "[MmapDistanceMatrix][mmap][durability]")
{
  TempFile tmp;

  // Reproduce the crash window in the pre-M15 constructor: the complete v2
  // header has reached disk, but the newly extended packed region still
  // contains filesystem-provided zero bits rather than NaN sentinels. Without
  // a publication state, open() accepts all three zeros as computed distances.
  constexpr size_t n = 2;
  std::vector<uint8_t> bytes(
    MmapDistanceMatrix::header_size + 3 * sizeof(double), 0);
  std::memcpy(bytes.data() + 0, MmapDistanceMatrix::magic, 4);
  const uint16_t ver = MmapDistanceMatrix::version;
  std::memcpy(bytes.data() + 4, &ver, 2);
  const uint32_t endian = MmapDistanceMatrix::endian_marker;
  std::memcpy(bytes.data() + 6, &endian, 4);
  bytes[10] = MmapDistanceMatrix::elem_size;
  bytes[11] = MmapDistanceMatrix::fingerprint_algorithm;
  const uint64_t n64 = n;
  std::memcpy(bytes.data() + 12, &n64, 8);
  // Byte 52 is the v2 publication state. Zero means initializing. The
  // fingerprint, state, remaining reserved bytes, and data tail stay zero.
  bytes[MmapDistanceMatrix::publication_state_offset] =
    MmapDistanceMatrix::publication_state_initializing;
  const uint32_t crc = detail::crc32_naive(bytes.data(), 60);
  std::memcpy(bytes.data() + 60, &crc, 4);

  {
    std::ofstream out(tmp.path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(bytes.data()),
              static_cast<std::streamsize>(bytes.size()));
  }

  REQUIRE_THROWS_WITH(
    MmapDistanceMatrix::open(tmp.path),
    Catch::Matchers::ContainsSubstring("initialization incomplete"));
}

TEST_CASE("MmapDistanceMatrix allows exactly one concurrent creator per cache path",
          "[MmapDistanceMatrix][mmap][race]")
{
  TempFile tmp;
  std::barrier start_line(3);
  std::barrier finish_line(2);
  std::atomic<int> successes{ 0 };
  std::atomic<int> failures{ 0 };
  std::array<std::string, 2> errors;

  auto create = [&](size_t slot) {
    start_line.arrive_and_wait();
    try {
      MmapDistanceMatrix matrix(tmp.path, 128);
      successes.fetch_add(1, std::memory_order_relaxed);
      // Keep the winning mapping alive until both creation attempts finish;
      // the loser must fail at atomic path creation, not after winner teardown.
      finish_line.arrive_and_wait();
    } catch (const std::exception &error) {
      errors[slot] = error.what();
      failures.fetch_add(1, std::memory_order_relaxed);
      finish_line.arrive_and_wait();
    }
  };

  std::thread first(create, 0);
  std::thread second(create, 1);
  start_line.arrive_and_wait();
  first.join();
  second.join();

  INFO("creator 0: " << errors[0]);
  INFO("creator 1: " << errors[1]);
  REQUIRE(successes.load(std::memory_order_relaxed) == 1);
  REQUIRE(failures.load(std::memory_order_relaxed) == 1);

  // The winner must leave one fully initialized, reopenable cache. The losing
  // creator cannot truncate it, alias it, or publish its own header/data.
  const auto reopened = MmapDistanceMatrix::open(tmp.path);
  REQUIRE(reopened.size() == 128);
  REQUIRE(reopened.count_computed() == 0);
}

// ============================================================================
// Free functions: tri_index and packed_size
// ============================================================================

TEST_CASE("tri_index symmetry", "[MmapDistanceMatrix][tri_index]")
{
  REQUIRE(tri_index(3, 7) == tri_index(7, 3));
  REQUIRE(tri_index(0, 0) == 0);
  REQUIRE(tri_index(1, 0) == tri_index(0, 1));
}

TEST_CASE("packed_size", "[MmapDistanceMatrix][packed_size]")
{
  REQUIRE(packed_size(0) == 0);
  REQUIRE(packed_size(1) == 1);
  REQUIRE(packed_size(2) == 3);
  REQUIRE(packed_size(3) == 6);
  REQUIRE(packed_size(10) == 55);
  REQUIRE(packed_size(1000) == 500500);
}

#endif // DTWC_HAS_MMAP
