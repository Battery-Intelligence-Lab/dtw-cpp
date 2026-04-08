/**
 * @file unit_test_mmap_distance_matrix.cpp
 * @brief Unit tests for MmapDistanceMatrix class (memory-mapped distance matrix).
 *
 * @date 08 Apr 2026
 */

#ifdef DTWC_HAS_MMAP

#include <core/mmap_distance_matrix.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <filesystem>
#include <string>

using Catch::Matchers::WithinAbs;
using namespace dtwc::core;
namespace fs = std::filesystem;

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

#else // !DTWC_HAS_MMAP

#include <catch2/catch_test_macros.hpp>

TEST_CASE("MmapDistanceMatrix SKIPPED — DTWC_HAS_MMAP not defined", "[MmapDistanceMatrix]")
{
  SKIP("llfio not available — mmap tests skipped");
}

#endif // DTWC_HAS_MMAP
