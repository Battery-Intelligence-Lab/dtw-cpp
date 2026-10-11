/**
 * @file unit_test_distance_matrix.cpp
 * @brief Unit tests for DistanceMatrix class.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include <core/distance_matrix.hpp>
#include <core/matrix_io.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <type_traits>
#include <utility>

using Catch::Matchers::WithinAbs;
using namespace dtwc::core;

TEST_CASE("DistanceMatrix default constructor", "[DistanceMatrix]")
{
  DistanceMatrix dm;
  REQUIRE(dm.size() == 0);
}

TEST_CASE("DistanceMatrix sized constructor", "[DistanceMatrix]")
{
  DistanceMatrix dm(5);
  REQUIRE(dm.size() == 5);

  // All entries should be uncomputed (NaN sentinel)
  for (size_t i = 0; i < 5; ++i)
    for (size_t j = 0; j < 5; ++j)
      REQUIRE_FALSE(dm.is_computed(i, j));
}

TEST_CASE("DistanceMatrix set and get", "[DistanceMatrix]")
{
  DistanceMatrix dm(4);

  dm.set(0, 1, 3.5);
  REQUIRE_THAT(dm.get(0, 1), WithinAbs(3.5, 1e-12));
}

TEST_CASE("DistanceMatrix symmetry", "[DistanceMatrix]")
{
  DistanceMatrix dm(4);

  dm.set(1, 3, 7.25);
  REQUIRE_THAT(dm.get(1, 3), WithinAbs(7.25, 1e-12));
  REQUIRE_THAT(dm.get(3, 1), WithinAbs(7.25, 1e-12));
}

TEST_CASE("DistanceMatrix is_computed with NaN sentinel", "[DistanceMatrix]")
{
  DistanceMatrix dm(3);

  // Initially all entries are uncomputed (NaN)
  REQUIRE_FALSE(dm.is_computed(0, 1));
  REQUIRE_FALSE(dm.is_computed(1, 0));

  // After setting, entry is computed
  dm.set(0, 1, 2.0);
  REQUIRE(dm.is_computed(0, 1));
  REQUIRE(dm.is_computed(1, 0)); // symmetric

  // Unset entry should still be uncomputed
  REQUIRE_FALSE(dm.is_computed(0, 2));
}

TEST_CASE("DistanceMatrix zero distance is computed", "[DistanceMatrix]")
{
  DistanceMatrix dm(3);

  dm.set(0, 0, 0.0);
  REQUIRE(dm.is_computed(0, 0));
  REQUIRE_THAT(dm.get(0, 0), WithinAbs(0.0, 1e-12));
}

TEST_CASE("DistanceMatrix resize", "[DistanceMatrix]")
{
  DistanceMatrix dm(3);
  dm.set(0, 1, 5.0);

  dm.resize(5);
  REQUIRE(dm.size() == 5);

  // After resize, all entries should be uncomputed (NaN)
  for (size_t i = 0; i < 5; ++i)
    for (size_t j = 0; j < 5; ++j)
      REQUIRE_FALSE(dm.is_computed(i, j));
}

TEST_CASE("DistanceMatrix max skips NaN entries", "[DistanceMatrix]")
{
  DistanceMatrix dm(3);
  dm.set(0, 1, 2.0);
  dm.set(0, 2, 8.5);
  dm.set(1, 2, 4.0);

  // max() should return 8.5, ignoring NaN entries
  REQUIRE_THAT(dm.max(), WithinAbs(8.5, 1e-12));
}

TEST_CASE("DistanceMatrix max empty", "[DistanceMatrix]")
{
  DistanceMatrix dm;
  REQUIRE_THAT(dm.max(), WithinAbs(0.0, 1e-12));
}

TEST_CASE("DistanceMatrix max on sized-but-unfilled matrix", "[DistanceMatrix]")
{
  // A matrix that has been sized but has no computed entries
  DistanceMatrix dm(5);
  // All entries are NaN, so max should return 0.0 (the fallback)
  REQUIRE_THAT(dm.max(), WithinAbs(0.0, 1e-12));
}

TEST_CASE("DistanceMatrix diagonal self-distance", "[DistanceMatrix]")
{
  DistanceMatrix dm(4);

  // Set diagonal to zero (self-distance)
  for (size_t i = 0; i < 4; ++i)
    dm.set(i, i, 0.0);

  for (size_t i = 0; i < 4; ++i) {
    REQUIRE(dm.is_computed(i, i));
    REQUIRE_THAT(dm.get(i, i), WithinAbs(0.0, 1e-12));
  }
}

TEST_CASE("DistanceMatrix visit hands the packed values at their width", "[DistanceMatrix]")
{
  for (const auto width : { Precision::Float64, Precision::Float32 }) {
    DistanceMatrix dm(2, width);
    REQUIRE(dm.width() == width);
    dm.set(0, 0, 1.0);
    dm.set(0, 1, 2.0);
    dm.set(1, 1, 3.0);
    // Packed lower-triangular: tri(0,0) = 0, tri(1,0) = 1, tri(1,1) = 2.
    REQUIRE(dm.packed_count() == 3);
    dm.visit([&](const auto *values) {
      using T = std::remove_cvref_t<decltype(*values)>;
      REQUIRE(std::is_same_v<T, float> == (width == Precision::Float32));
      REQUIRE(values[0] == 1.0);
      REQUIRE(values[1] == 2.0); // (1,0) == (0,1)
      REQUIRE(values[2] == 3.0);
    });
    REQUIRE(dm.get(0, 1) == 2.0);
    REQUIRE(dm.get(1, 0) == 2.0);
    dm.resize(3); // a resize keeps the width
    REQUIRE(dm.width() == width);
  }
}

TEST_CASE("A float32 matrix stores each distance rounded to the nearest float", "[DistanceMatrix][float32]")
{
  // The oracle is IEEE binary32 rounding to nearest even, static_cast<float> of the double.
  constexpr double float_max = 0x1.fffffep127;              // 2^128 - 2^104, the largest float
  constexpr double rounds_to_inf = 0x1.ffffffp127;          // 2^128 - 2^103: the tie, which rounds to even, inf
  const double values[] = { 0.1, 1.0 / 3.0, 1e-40, 3e-46, float_max, 0x1.fffffefffffffp127, 1e300, -0.0, 5.0 };
  DistanceMatrix dm(6, Precision::Float32);
  for (std::size_t k = 0; k < std::size(values); ++k) {
    const std::size_t i = k / 2 + 1, j = k % 2; // pairs (1,0) (1,1) (2,0) ... in rows 1 to 5, distinct
    dm.set(i, j, values[k]);
    const double stored = dm.get(i, j);
    REQUIRE(std::bit_cast<std::uint64_t>(stored)
            == std::bit_cast<std::uint64_t>(static_cast<double>(static_cast<float>(values[k]))));
  }
  REQUIRE(dm.get(2, 0) == static_cast<double>(1e-40f)); // a subnormal float, kept
  REQUIRE(dm.get(2, 1) == 0.0);                          // below 2^-150: zero
  REQUIRE(dm.get(3, 0) == float_max);
  REQUIRE(dm.get(3, 1) == float_max);                    // below the tie: the largest float
  REQUIRE(std::isinf(dm.get(4, 0)));                     // beyond: inf
  REQUIRE_FALSE(dm.is_computed(0, 0));                   // NaN stays NaN: not computed
  REQUIRE_FALSE(dm.is_computed(4, 2));
  dm.set(4, 3, rounds_to_inf);
  REQUIRE(std::isinf(dm.get(4, 3)));

  // first_infinite names the first ±inf entry in packed order, rows [r0, r1).
  REQUIRE(dm.first_infinite(0, 6) == std::pair<std::size_t, std::size_t>{ 4, 0 });
  REQUIRE(dm.first_infinite(0, 4) == std::pair<std::size_t, std::size_t>{ 6, 6 }); // none: (n, n)
  REQUIRE(dm.count_computed() == std::size(values) + 1);
  dm.clear_rows(4, 5);
  REQUIRE(dm.first_infinite(0, 6) == std::pair<std::size_t, std::size_t>{ 6, 6 });
  REQUIRE(dm.max() == float_max);
}

TEST_CASE("DistanceMatrix to_full_matrix", "[DistanceMatrix]")
{
  DistanceMatrix dm(3);
  dm.set(0, 1, 5.0);
  dm.set(0, 2, 3.0);
  dm.set(1, 2, 7.0);

  // Flat row-major std::vector<double> since Eigen was dropped; the matrix is
  // symmetric, so element (i, j) is at i * n + j and the layout question is moot.
  const auto full = dtwc::io::to_full_matrix(dm);
  REQUIRE(full.size() == 3 * 3);
  const auto at = [&full](size_t i, size_t j) { return full[i * 3 + j]; };
  REQUIRE_THAT(at(0, 1), WithinAbs(5.0, 1e-12));
  REQUIRE_THAT(at(1, 0), WithinAbs(5.0, 1e-12));
  REQUIRE_THAT(at(0, 2), WithinAbs(3.0, 1e-12));
  REQUIRE_THAT(at(2, 0), WithinAbs(3.0, 1e-12));
  REQUIRE_THAT(at(1, 2), WithinAbs(7.0, 1e-12));
  REQUIRE_THAT(at(2, 1), WithinAbs(7.0, 1e-12));
  // Symmetry is the property that makes the row-major/column-major choice safe,
  // so assert it rather than assume it. Off-diagonal only: the diagonal is
  // whatever get(i, i) returns, which is NaN while unset — unchanged from the
  // Eigen version, which copied the same value, and NaN is not equal to itself.
  for (size_t i = 0; i < 3; ++i)
    for (size_t j = 0; j < 3; ++j)
      if (i != j) REQUIRE_THAT(at(i, j), WithinAbs(at(j, i), 1e-12));
}

TEST_CASE("DistanceMatrix uncomputed entries are not computed", "[DistanceMatrix]")
{
  DistanceMatrix dm(3);

  // Verify that fresh entries are marked as not computed.
  REQUIRE_FALSE(dm.is_computed(0, 1));
  REQUIRE_FALSE(dm.is_computed(2, 0));

  // After setting a value, it should be marked as computed.
  dm.set(0, 1, 0.0);
  REQUIRE(dm.is_computed(0, 1));
  REQUIRE(dm.is_computed(1, 0));
}

TEST_CASE("DistanceMatrix zero distance is valid and computed", "[DistanceMatrix]")
{
  // Zero is a valid DTW distance (identical series). The NaN sentinel must not
  // confuse 0.0 with "uncomputed".
  DistanceMatrix dm(3);

  dm.set(0, 1, 0.0);
  REQUIRE(dm.is_computed(0, 1));
  REQUIRE_THAT(dm.get(0, 1), WithinAbs(0.0, 1e-12));
}
