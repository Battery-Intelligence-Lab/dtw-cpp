/**
 * @file unit_test_mmap_distance_matrix.cpp
 * @brief Unit tests for MmapDistanceMatrix class (memory-mapped distance matrix).
 *
 * @date 08 Apr 2026
 */

#include <core/mmap_distance_matrix.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <array>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <thread>
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
