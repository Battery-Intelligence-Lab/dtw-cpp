/**
 * @file test_dense_distance_matrix_adversarial.cpp
 * @brief Adversarial tests for DistanceMatrix.
 *
 * These tests are written from the SPECIFICATION, not the implementation.
 * Spec: "Dense symmetric distance matrix with flat array storage.
 *        set() enforces symmetry. Uncomputed entries are written as empty fields."
 *
 * If a test fails, the CODE is wrong, not the test.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include <core/distance_matrix.hpp>
#include <core/matrix_io.hpp>
#include <dtwc.hpp>

#include "../../support/scratch_directory.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>

using Catch::Matchers::WithinAbs;
using dtwc::test_support::ScratchDirectory;
using namespace dtwc::core;

// ============================================================================
// Area 1: DistanceMatrix Edge Cases & Contracts
// ============================================================================

TEST_CASE("Adversarial: Symmetry enforcement via set()", "[adversarial][DistanceMatrix]")
{
  DistanceMatrix dm(5);

  SECTION("set(i,j,v) makes get(j,i)==v AND get(i,j)==v")
  {
    dm.set(1, 3, 42.0);
    REQUIRE_THAT(dm.get(1, 3), WithinAbs(42.0, 1e-15));
    REQUIRE_THAT(dm.get(3, 1), WithinAbs(42.0, 1e-15));
  }

  SECTION("set(j,i,v) also symmetric")
  {
    dm.set(4, 0, 7.77);
    REQUIRE_THAT(dm.get(0, 4), WithinAbs(7.77, 1e-15));
    REQUIRE_THAT(dm.get(4, 0), WithinAbs(7.77, 1e-15));
  }

  SECTION("Overwriting preserves symmetry")
  {
    dm.set(2, 3, 1.0);
    dm.set(3, 2, 2.0); // overwrite from other direction
    REQUIRE_THAT(dm.get(2, 3), WithinAbs(2.0, 1e-15));
    REQUIRE_THAT(dm.get(3, 2), WithinAbs(2.0, 1e-15));
  }

  SECTION("All corners symmetric in non-square index")
  {
    dm.set(0, 4, 99.0);
    REQUIRE_THAT(dm.get(4, 0), WithinAbs(99.0, 1e-15));
    dm.set(4, 0, 11.0);
    REQUIRE_THAT(dm.get(0, 4), WithinAbs(11.0, 1e-15));
  }
}

TEST_CASE("Adversarial: NaN sentinel after resize", "[adversarial][DistanceMatrix]")
{
  SECTION("All entries NaN after construction including diagonal")
  {
    DistanceMatrix dm(4);
    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        REQUIRE_FALSE(dm.is_computed(i, j));
      }
    }
  }

  SECTION("All entries uncomputed after resize including diagonal")
  {
    DistanceMatrix dm(3);
    dm.set(0, 0, 0.0);
    dm.set(1, 2, 5.0);

    dm.resize(4);

    for (size_t i = 0; i < 4; ++i) {
      for (size_t j = 0; j < 4; ++j) {
        REQUIRE_FALSE(dm.is_computed(i, j));
      }
    }
  }
}

TEST_CASE("Adversarial: Diagonal self-distance", "[adversarial][DistanceMatrix]")
{
  DistanceMatrix dm(3);

  dm.set(1, 1, 0.0);

  SECTION("is_computed returns true after setting diagonal")
  {
    REQUIRE(dm.is_computed(1, 1));
  }

  SECTION("get returns 0.0 for set diagonal")
  {
    REQUIRE_THAT(dm.get(1, 1), WithinAbs(0.0, 1e-15));
  }

  SECTION("Unset diagonal entries remain NaN")
  {
    REQUIRE_FALSE(dm.is_computed(0, 0));
    REQUIRE_FALSE(dm.is_computed(2, 2));
  }
}

TEST_CASE("Adversarial: max() skips NaN", "[adversarial][DistanceMatrix]")
{
  DistanceMatrix dm(5);

  SECTION("max of partially-filled matrix returns max of computed values only")
  {
    dm.set(0, 1, 3.0);
    dm.set(2, 3, 7.5);
    dm.set(1, 4, 2.0);

    REQUIRE_THAT(dm.max(), WithinAbs(7.5, 1e-12));
  }

  SECTION("max of empty matrix (size 0) returns 0.0")
  {
    DistanceMatrix empty;
    REQUIRE_THAT(empty.max(), WithinAbs(0.0, 1e-15));
  }

  SECTION("max of all-NaN matrix returns 0.0")
  {
    // dm has size 5 but no values set — all NaN
    REQUIRE_THAT(dm.max(), WithinAbs(0.0, 1e-15));
  }

  SECTION("max with only zero distances returns 0.0")
  {
    // DTW distances are always >= 0. Zero distances (identical series) are valid.
    DistanceMatrix dm2(3);
    dm2.set(0, 1, 0.0);
    dm2.set(1, 2, 0.0);

    REQUIRE_THAT(dm2.max(), WithinAbs(0.0, 1e-12));
  }
}

TEST_CASE("Adversarial: resize clears old data", "[adversarial][DistanceMatrix]")
{
  DistanceMatrix dm(3);
  dm.set(0, 1, 100.0);
  dm.set(2, 2, 50.0);

  SECTION("Resize larger clears all old data")
  {
    dm.resize(5);

    // Old positions must be uncomputed
    REQUIRE_FALSE(dm.is_computed(0, 1));
    REQUIRE_FALSE(dm.is_computed(1, 0));
    REQUIRE_FALSE(dm.is_computed(2, 2));
  }

  SECTION("Resize smaller then larger — old data must NOT persist")
  {
    dm.resize(2);
    dm.resize(5);

    for (size_t i = 0; i < 5; ++i)
      for (size_t j = 0; j < 5; ++j)
        REQUIRE_FALSE(dm.is_computed(i, j));
  }

  SECTION("Resize to same size clears data")
  {
    dm.resize(3);
    REQUIRE_FALSE(dm.is_computed(0, 1));
    REQUIRE_FALSE(dm.is_computed(2, 2));
  }
}

TEST_CASE("Adversarial: Zero-size matrix", "[adversarial][DistanceMatrix]")
{
  SECTION("Constructor with 0 does not crash")
  {
    DistanceMatrix dm(0);
    REQUIRE(dm.size() == 0);
  }

  SECTION("max() on zero-size returns 0.0")
  {
    DistanceMatrix dm(0);
    REQUIRE_THAT(dm.max(), WithinAbs(0.0, 1e-15));
  }

  SECTION("resize(0) does not crash")
  {
    DistanceMatrix dm(5);
    dm.resize(0);
    REQUIRE(dm.size() == 0);
  }
}

TEST_CASE("Adversarial: Single element matrix", "[adversarial][DistanceMatrix]")
{
  DistanceMatrix dm(1);

  REQUIRE(dm.size() == 1);
  REQUIRE_FALSE(dm.is_computed(0, 0)); // initially uncomputed

  dm.set(0, 0, 5.0);
  REQUIRE_THAT(dm.get(0, 0), WithinAbs(5.0, 1e-15));
  REQUIRE(dm.is_computed(0, 0));
}

TEST_CASE("Adversarial: Large indices boundary", "[adversarial][DistanceMatrix]")
{
  constexpr size_t N = 1000;
  DistanceMatrix dm(N);

  SECTION("Set and get at (0, N-1)")
  {
    dm.set(0, N - 1, 1.23);
    REQUIRE_THAT(dm.get(0, N - 1), WithinAbs(1.23, 1e-12));
    REQUIRE_THAT(dm.get(N - 1, 0), WithinAbs(1.23, 1e-12));
  }

  SECTION("Set and get at (N-1, 0)")
  {
    dm.set(N - 1, 0, 4.56);
    REQUIRE_THAT(dm.get(N - 1, 0), WithinAbs(4.56, 1e-12));
    REQUIRE_THAT(dm.get(0, N - 1), WithinAbs(4.56, 1e-12));
  }

  SECTION("Set and get at (N/2, N/2)")
  {
    dm.set(500, 500, 7.89);
    REQUIRE_THAT(dm.get(500, 500), WithinAbs(7.89, 1e-12));
    REQUIRE(dm.is_computed(500, 500));
  }
}

// ============================================================================
// Area 2: Distance Matrix I/O Round-Trip
// ============================================================================

TEST_CASE("Adversarial: Full matrix CSV round-trip", "[adversarial][DistanceMatrix][IO]")
{
  constexpr size_t N = 10;
  DistanceMatrix dm(N);

  // Fill with deterministic values
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i; j < N; ++j)
      dm.set(i, j, static_cast<double>(i * N + j) * 0.1);

  const ScratchDirectory scratch("full_roundtrip");
  const auto csv = scratch.path / "matrix.csv";
  dtwc::io::write_csv(dm,csv);

  DistanceMatrix dm2;
  dtwc::io::read_csv(dm2,csv);

  REQUIRE(dm2.size() == N);
  for (size_t i = 0; i < N; ++i)
    for (size_t j = 0; j < N; ++j)
      REQUIRE_THAT(dm2.get(i, j), WithinAbs(dm.get(i, j), 1e-12));
}

TEST_CASE("Adversarial: Partial matrix CSV round-trip (empty-field preservation)",
  "[adversarial][DistanceMatrix][IO]")
{
  constexpr size_t N = 5;
  DistanceMatrix dm(N);

  // Set only some entries, leave others as NaN
  dm.set(0, 1, 1.0);
  dm.set(2, 3, 2.0);
  dm.set(4, 4, 0.0);

  const ScratchDirectory scratch("partial_roundtrip");
  const auto csv = scratch.path / "matrix.csv";
  dtwc::io::write_csv(dm,csv);

  DistanceMatrix dm2;
  dtwc::io::read_csv(dm2,csv);

  REQUIRE(dm2.size() == N);

  // Computed entries must match
  REQUIRE_THAT(dm2.get(0, 1), WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(dm2.get(1, 0), WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(dm2.get(2, 3), WithinAbs(2.0, 1e-12));
  REQUIRE_THAT(dm2.get(3, 2), WithinAbs(2.0, 1e-12));
  REQUIRE_THAT(dm2.get(4, 4), WithinAbs(0.0, 1e-12));

  // Uncomputed entries must remain uncomputed after round-trip
  REQUIRE_FALSE(dm2.is_computed(0, 0));
  REQUIRE_FALSE(dm2.is_computed(0, 2));
  REQUIRE_FALSE(dm2.is_computed(3, 4));
}

TEST_CASE("Adversarial: Small values in distance matrix roundtrip", "[adversarial][DistanceMatrix][IO]")
{
  DistanceMatrix dm(3);
  dm.set(0, 1, 42.5);
  dm.set(1, 2, 0.001);

  const ScratchDirectory scratch("small_roundtrip");
  const auto csv = scratch.path / "matrix.csv";
  dtwc::io::write_csv(dm,csv);

  DistanceMatrix dm2;
  dtwc::io::read_csv(dm2,csv);

  REQUIRE_THAT(dm2.get(0, 1), WithinAbs(42.5, 1e-12));
  REQUIRE_THAT(dm2.get(1, 0), WithinAbs(42.5, 1e-12));
  REQUIRE_THAT(dm2.get(1, 2), WithinAbs(0.001, 1e-12));
}

TEST_CASE("Adversarial: Very large values (1e300)", "[adversarial][DistanceMatrix][IO]")
{
  DistanceMatrix dm(2);
  dm.set(0, 1, 1e300);

  const ScratchDirectory scratch("large_roundtrip");
  const auto csv = scratch.path / "matrix.csv";
  dtwc::io::write_csv(dm,csv);

  DistanceMatrix dm2;
  dtwc::io::read_csv(dm2,csv);

  // Relative tolerance for very large values
  REQUIRE_THAT(dm2.get(0, 1), WithinAbs(1e300, 1e287));
  REQUIRE_THAT(dm2.get(1, 0), WithinAbs(1e300, 1e287));
}

TEST_CASE("Adversarial: Very small values (1e-300)", "[adversarial][DistanceMatrix][IO]")
{
  DistanceMatrix dm(2);
  dm.set(0, 1, 1e-300);

  const ScratchDirectory scratch("small_roundtrip");
  const auto csv = scratch.path / "matrix.csv";
  dtwc::io::write_csv(dm,csv);

  DistanceMatrix dm2;
  dtwc::io::read_csv(dm2,csv);

  // Must not underflow to zero
  REQUIRE(dm2.get(0, 1) != 0.0);
  REQUIRE_THAT(dm2.get(0, 1), WithinAbs(1e-300, 1e-312));
}

TEST_CASE("Adversarial: operator<< matches write_csv output",
  "[adversarial][DistanceMatrix][IO]")
{
  constexpr size_t N = 4;
  DistanceMatrix dm(N);
  for (size_t i = 0; i < N; ++i)
    for (size_t j = i; j < N; ++j)
      dm.set(i, j, static_cast<double>(i * 10 + j) * 1.1);

  // write_csv output
  const ScratchDirectory scratch("operator_vs_csv");
  const auto csv = scratch.path / "matrix.csv";
  dtwc::io::write_csv(dm,csv);

  std::ifstream file(csv);
  std::string csv_content((std::istreambuf_iterator<char>(file)),
    std::istreambuf_iterator<char>());
  file.close();

  // operator<< output
  std::ostringstream oss;
  oss << dm;
  std::string stream_content = oss.str();

  // Spec says both should produce identical CSV format
  REQUIRE(csv_content == stream_content);
}

TEST_CASE("Adversarial: read_csv with trailing newline", "[adversarial][DistanceMatrix][IO]")
{
  // Write a symmetric CSV manually with a trailing blank line
  const ScratchDirectory scratch("trailing_newline");
  const auto csv = scratch.path / "matrix.csv";
  {
    std::ofstream f(csv);
    f << "0.0,2.5\n2.5,0.0\n\n"; // trailing blank line
  }

  DistanceMatrix dm;
  dtwc::io::read_csv(dm,csv);

  // Should have 2 rows, not 3 — trailing blank line must be ignored
  REQUIRE(dm.size() == 2);
  REQUIRE_THAT(dm.get(0, 0), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(dm.get(0, 1), WithinAbs(2.5, 1e-12));
  REQUIRE_THAT(dm.get(1, 0), WithinAbs(2.5, 1e-12));
  REQUIRE_THAT(dm.get(1, 1), WithinAbs(0.0, 1e-12));
}

TEST_CASE("Adversarial: Empty matrix write/read", "[adversarial][DistanceMatrix][IO]")
{
  DistanceMatrix dm(0);

  const ScratchDirectory scratch("empty_roundtrip");
  const auto csv = scratch.path / "matrix.csv";

  SECTION("write_csv on empty matrix does not crash")
  {
    REQUIRE_NOTHROW(dtwc::io::write_csv(dm,csv));
  }

  SECTION("read_csv of empty file does not crash")
  {
    {
      std::ofstream f(csv);
      // write nothing
    }
    DistanceMatrix dm2;
    REQUIRE_NOTHROW(dtwc::io::read_csv(dm2,csv));
    REQUIRE(dm2.size() == 0);
  }
}

TEST_CASE("Adversarial: read_csv of asymmetric CSV is rejected, not overwritten",
  "[adversarial][DistanceMatrix][IO]")
{
  // The storage is symmetric, so an asymmetric file cannot be represented. The
  // reader used to let the later of the two values win (0,1) = 10, (0,2) = 20,
  // (1,2) = 30, silently discarding 1, 2 and 3 (FX-11). It now throws before
  // touching the destination.
  const ScratchDirectory scratch("asymmetric_csv");
  const auto csv = scratch.path / "matrix.csv";
  {
    std::ofstream f(csv);
    f << "0.0,1.0,2.0\n"
      << "10.0,0.0,3.0\n"
      << "20.0,30.0,0.0\n";
  }

  DistanceMatrix dm;
  REQUIRE_THROWS_AS(dtwc::io::read_csv(dm, csv), dtwc::InvalidInput);
  REQUIRE_THROWS_WITH(dtwc::io::read_csv(dm, csv),
    Catch::Matchers::ContainsSubstring("row 2, column 1 is 10 but row 1, column 2 is 1"));
  REQUIRE(dm.size() == 0);

  // The same file made symmetric reads every pair.
  {
    std::ofstream f(csv);
    f << "0.0,10.0,20.0\n"
      << "10.0,0.0,30.0\n"
      << "20.0,30.0,0.0\n";
  }
  dtwc::io::read_csv(dm, csv);
  REQUIRE(dm.size() == 3);
  REQUIRE_THAT(dm.get(0, 1), WithinAbs(10.0, 1e-12));
  REQUIRE_THAT(dm.get(1, 0), WithinAbs(10.0, 1e-12));
  REQUIRE_THAT(dm.get(0, 2), WithinAbs(20.0, 1e-12));
  REQUIRE_THAT(dm.get(2, 0), WithinAbs(20.0, 1e-12));
  REQUIRE_THAT(dm.get(1, 2), WithinAbs(30.0, 1e-12));
  REQUIRE_THAT(dm.get(2, 1), WithinAbs(30.0, 1e-12));
}
