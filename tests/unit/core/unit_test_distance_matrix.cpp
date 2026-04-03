/**
 * @file unit_test_distance_matrix.cpp
 * @brief Unit tests for DenseDistanceMatrix class.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include <core/distance_matrix.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>

using Catch::Matchers::WithinAbs;
using namespace dtwc::core;

TEST_CASE("DenseDistanceMatrix default constructor", "[DistanceMatrix]")
{
  DenseDistanceMatrix dm;
  REQUIRE(dm.size() == 0);
}

TEST_CASE("DenseDistanceMatrix sized constructor", "[DistanceMatrix]")
{
  DenseDistanceMatrix dm(5);
  REQUIRE(dm.size() == 5);

  // All entries should be uncomputed (NaN sentinel)
  for (size_t i = 0; i < 5; ++i)
    for (size_t j = 0; j < 5; ++j)
      REQUIRE_FALSE(dm.is_computed(i, j));
}

TEST_CASE("DenseDistanceMatrix set and get", "[DistanceMatrix]")
{
  DenseDistanceMatrix dm(4);

  dm.set(0, 1, 3.5);
  REQUIRE_THAT(dm.get(0, 1), WithinAbs(3.5, 1e-12));
}

TEST_CASE("DenseDistanceMatrix symmetry", "[DistanceMatrix]")
{
  DenseDistanceMatrix dm(4);

  dm.set(1, 3, 7.25);
  REQUIRE_THAT(dm.get(1, 3), WithinAbs(7.25, 1e-12));
  REQUIRE_THAT(dm.get(3, 1), WithinAbs(7.25, 1e-12));
}

TEST_CASE("DenseDistanceMatrix is_computed with NaN sentinel", "[DistanceMatrix]")
{
  DenseDistanceMatrix dm(3);

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

TEST_CASE("DenseDistanceMatrix zero distance is computed", "[DistanceMatrix]")
{
  DenseDistanceMatrix dm(3);

  dm.set(0, 0, 0.0);
  REQUIRE(dm.is_computed(0, 0));
  REQUIRE_THAT(dm.get(0, 0), WithinAbs(0.0, 1e-12));
}

TEST_CASE("DenseDistanceMatrix resize", "[DistanceMatrix]")
{
  DenseDistanceMatrix dm(3);
  dm.set(0, 1, 5.0);

  dm.resize(5);
  REQUIRE(dm.size() == 5);

  // After resize, all entries should be uncomputed (NaN)
  for (size_t i = 0; i < 5; ++i)
    for (size_t j = 0; j < 5; ++j)
      REQUIRE_FALSE(dm.is_computed(i, j));
}

TEST_CASE("DenseDistanceMatrix max skips NaN entries", "[DistanceMatrix]")
{
  DenseDistanceMatrix dm(3);
  dm.set(0, 1, 2.0);
  dm.set(0, 2, 8.5);
  dm.set(1, 2, 4.0);

  // max() should return 8.5, ignoring NaN entries
  REQUIRE_THAT(dm.max(), WithinAbs(8.5, 1e-12));
}

TEST_CASE("DenseDistanceMatrix max empty", "[DistanceMatrix]")
{
  DenseDistanceMatrix dm;
  REQUIRE_THAT(dm.max(), WithinAbs(0.0, 1e-12));
}

TEST_CASE("DenseDistanceMatrix max on sized-but-unfilled matrix", "[DistanceMatrix]")
{
  // A matrix that has been sized but has no computed entries
  DenseDistanceMatrix dm(5);
  // All entries are NaN, so max should return 0.0 (the fallback)
  REQUIRE_THAT(dm.max(), WithinAbs(0.0, 1e-12));
}

TEST_CASE("DenseDistanceMatrix diagonal self-distance", "[DistanceMatrix]")
{
  DenseDistanceMatrix dm(4);

  // Set diagonal to zero (self-distance)
  for (size_t i = 0; i < 4; ++i)
    dm.set(i, i, 0.0);

  for (size_t i = 0; i < 4; ++i) {
    REQUIRE(dm.is_computed(i, i));
    REQUIRE_THAT(dm.get(i, i), WithinAbs(0.0, 1e-12));
  }
}

TEST_CASE("DenseDistanceMatrix raw pointer access", "[DistanceMatrix]")
{
  DenseDistanceMatrix dm(2);
  dm.set(0, 0, 1.0);
  dm.set(0, 1, 2.0);
  dm.set(1, 1, 3.0);

  const double *raw = dm.raw();
  // Row-major flat storage: [0,0]=1.0  [0,1]=2.0  [1,0]=2.0  [1,1]=3.0
  REQUIRE_THAT(raw[0], WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(raw[1], WithinAbs(2.0, 1e-12));
  REQUIRE_THAT(raw[2], WithinAbs(2.0, 1e-12)); // symmetric
  REQUIRE_THAT(raw[3], WithinAbs(3.0, 1e-12));
}

TEST_CASE("DenseDistanceMatrix uncomputed entries are not computed", "[DistanceMatrix]")
{
  DenseDistanceMatrix dm(3);

  // Verify that fresh entries are marked as not computed.
  REQUIRE_FALSE(dm.is_computed(0, 1));
  REQUIRE_FALSE(dm.is_computed(2, 0));

  // After setting a value, it should be marked as computed.
  dm.set(0, 1, 0.0);
  REQUIRE(dm.is_computed(0, 1));
  REQUIRE(dm.is_computed(1, 0));
}

TEST_CASE("DenseDistanceMatrix negative distance is valid and computed", "[DistanceMatrix]")
{
  // With the NaN sentinel, negative values should be treated as valid computed entries
  // (unlike the old -1.0 sentinel which incorrectly conflated negative distances
  // with uncomputed entries)
  DenseDistanceMatrix dm(3);

  dm.set(0, 1, -5.0);
  REQUIRE(dm.is_computed(0, 1));
  REQUIRE_THAT(dm.get(0, 1), WithinAbs(-5.0, 1e-12));
}
