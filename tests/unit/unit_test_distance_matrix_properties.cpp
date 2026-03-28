/**
 * @file unit_test_distance_matrix_properties.cpp
 * @brief Property-based tests for DTW distance matrices.
 *
 * Verifies diagonal zeros, symmetry, non-negativity, and consistency
 * between fillDistanceMatrix and individual pair computations.
 *
 * @date 28 Mar 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../test_util.hpp"

#include <cmath>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

namespace {

/**
 * @brief Build a Problem from the dummy dataset with N series.
 */
Problem make_problem(int N_data)
{
  dtwc::DataLoader dl{ settings::dataPath / "dummy", N_data };
  dl.startColumn(1).startRow(1);
  dtwc::Problem prob{ "dist_mat_test", dl };
  return prob;
}

} // anonymous namespace


// ---------------------------------------------------------------------------
// 1. Diagonal is all zeros
// ---------------------------------------------------------------------------
TEST_CASE("Distance matrix diagonal is all zeros", "[Phase1][distance_matrix]")
{
  constexpr int N = 10;
  auto prob = make_problem(N);
  prob.fillDistanceMatrix();

  for (int i = 0; i < N; ++i) {
    double d = prob.distByInd(i, i);
    REQUIRE_THAT(d, WithinAbs(0.0, 1e-15));
  }
}

// ---------------------------------------------------------------------------
// 2. Symmetry: d(i,j) == d(j,i) for all i,j
// ---------------------------------------------------------------------------
TEST_CASE("Distance matrix is symmetric", "[Phase1][distance_matrix]")
{
  constexpr int N = 10;
  auto prob = make_problem(N);
  prob.fillDistanceMatrix();

  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      double dij = prob.distByInd(i, j);
      double dji = prob.distByInd(j, i);
      REQUIRE_THAT(dij, WithinAbs(dji, 1e-15));
    }
  }
}

// ---------------------------------------------------------------------------
// 3. Non-negativity: d(i,j) >= 0 for all i,j
// ---------------------------------------------------------------------------
TEST_CASE("Distance matrix entries are non-negative", "[Phase1][distance_matrix]")
{
  constexpr int N = 10;
  auto prob = make_problem(N);
  prob.fillDistanceMatrix();

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      double d = prob.distByInd(i, j);
      REQUIRE(d >= 0.0);
    }
  }
}

// ---------------------------------------------------------------------------
// 4. Identity of indiscernibles: d(i,j)==0 implies series are identical
//    (the converse: distinct random series should have d > 0)
// ---------------------------------------------------------------------------
TEST_CASE("Distinct series have positive distance", "[Phase1][distance_matrix]")
{
  constexpr int N = 10;
  auto prob = make_problem(N);
  prob.fillDistanceMatrix();

  // Check that at least some off-diagonal entry is > 0.
  // (Dummy data has 25 distinct series, so the first 10 should not all be identical.)
  bool found_positive = false;
  for (int i = 0; i < N && !found_positive; ++i) {
    for (int j = i + 1; j < N && !found_positive; ++j) {
      if (prob.distByInd(i, j) > 0.0)
        found_positive = true;
    }
  }
  REQUIRE(found_positive);
}

// ---------------------------------------------------------------------------
// 5. fillDistanceMatrix gives same results as computing pairs individually
// ---------------------------------------------------------------------------
TEST_CASE("fillDistanceMatrix matches individual pair computation", "[Phase1][distance_matrix]")
{
  constexpr int N = 8;

  // Method A: compute individual pairs before fillDistanceMatrix.
  auto probA = make_problem(N);
  std::vector<std::vector<double>> pairwise(N, std::vector<double>(N, 0.0));
  for (int i = 0; i < N; ++i)
    for (int j = i; j < N; ++j)
      pairwise[i][j] = pairwise[j][i] = probA.distByInd(i, j);

  // Method B: use fillDistanceMatrix.
  auto probB = make_problem(N);
  probB.fillDistanceMatrix();

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      REQUIRE_THAT(probB.distByInd(i, j), WithinAbs(pairwise[i][j], 1e-15));
    }
  }
}

// ---------------------------------------------------------------------------
// 6. Distance matrix is marked as filled after fillDistanceMatrix
// ---------------------------------------------------------------------------
TEST_CASE("isDistanceMatrixFilled flag is set correctly", "[Phase1][distance_matrix]")
{
  constexpr int N = 5;
  auto prob = make_problem(N);

  REQUIRE_FALSE(prob.isDistanceMatrixFilled());
  prob.fillDistanceMatrix();
  REQUIRE(prob.isDistanceMatrixFilled());
}

// ---------------------------------------------------------------------------
// 7. DTW full vs banded consistency: with default band=-1, banded == full
// ---------------------------------------------------------------------------
TEST_CASE("Default band produces same distances as dtwFull", "[Phase1][distance_matrix]")
{
  constexpr int N = 5;
  auto prob = make_problem(N);

  // Default band is -1 (full DTW).
  REQUIRE(prob.band == settings::DEFAULT_BAND_LENGTH);
  REQUIRE(prob.band == -1);

  prob.fillDistanceMatrix();

  // Compare against dtwFull computed directly on the data vectors.
  for (int i = 0; i < N; ++i) {
    for (int j = i; j < N; ++j) {
      double expected = dtwFull<data_t>(prob.p_vec(i), prob.p_vec(j));
      REQUIRE_THAT(prob.distByInd(i, j), WithinAbs(expected, 1e-12));
    }
  }
}

// ---------------------------------------------------------------------------
// 8. Re-filling the distance matrix is a no-op when already filled
// ---------------------------------------------------------------------------
TEST_CASE("Repeated fillDistanceMatrix is idempotent", "[Phase1][distance_matrix]")
{
  constexpr int N = 5;
  auto prob = make_problem(N);

  prob.fillDistanceMatrix();
  REQUIRE(prob.isDistanceMatrixFilled());

  // Save a few values.
  double d01 = prob.distByInd(0, 1);
  double d23 = prob.distByInd(2, 3);

  // Call again -- should be a no-op.
  REQUIRE_NOTHROW(prob.fillDistanceMatrix());
  REQUIRE(prob.isDistanceMatrixFilled());

  REQUIRE_THAT(prob.distByInd(0, 1), WithinAbs(d01, 1e-15));
  REQUIRE_THAT(prob.distByInd(2, 3), WithinAbs(d23, 1e-15));
}
