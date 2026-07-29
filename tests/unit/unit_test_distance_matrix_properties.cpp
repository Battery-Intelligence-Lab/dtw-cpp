/**
 * @file unit_test_distance_matrix_properties.cpp
 * @brief Property-based tests for DTW distance matrices.
 *
 * Verifies diagonal zeros, symmetry, non-negativity, and consistency
 * between fill_distance_matrix and individual pair computations.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../test_util.hpp"

#include <cmath>
#include <vector>

#ifndef DTWC_TEST_DATA_DIR
#define DTWC_TEST_DATA_DIR "./data"
#endif

static struct TestDataInit2 {
  TestDataInit2() { dtwc::settings::paths::set_data_path(DTWC_TEST_DATA_DIR); }
} test_data_init2_;

using Catch::Matchers::WithinAbs;
using namespace dtwc;

namespace {

/**
 * @brief Build a Problem from the dummy dataset with N series.
 */
Problem make_problem(int N_data)
{
  dtwc::DataLoader dl{ settings::paths::data / "dummy", N_data };
  dl.start_column(1).start_row(1);
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
  prob.fill_distance_matrix();

  for (int i = 0; i < N; ++i) {
    double d = prob.dist_by_ind(i, i);
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
  prob.fill_distance_matrix();

  for (int i = 0; i < N; ++i) {
    for (int j = i + 1; j < N; ++j) {
      double dij = prob.dist_by_ind(i, j);
      double dji = prob.dist_by_ind(j, i);
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
  prob.fill_distance_matrix();

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      double d = prob.dist_by_ind(i, j);
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
  prob.fill_distance_matrix();

  // Check that at least some off-diagonal entry is > 0.
  // (Dummy data has 25 distinct series, so the first 10 should not all be identical.)
  bool found_positive = false;
  for (int i = 0; i < N && !found_positive; ++i) {
    for (int j = i + 1; j < N && !found_positive; ++j) {
      if (prob.dist_by_ind(i, j) > 0.0)
        found_positive = true;
    }
  }
  REQUIRE(found_positive);
}

// ---------------------------------------------------------------------------
// 5. fill_distance_matrix gives same results as computing pairs individually
// ---------------------------------------------------------------------------
TEST_CASE("fill_distance_matrix matches individual pair computation", "[Phase1][distance_matrix]")
{
  constexpr int N = 8;

  // Method A: compute individual pairs before fill_distance_matrix.
  auto probA = make_problem(N);
  std::vector<std::vector<double>> pairwise(N, std::vector<double>(N, 0.0));
  for (int i = 0; i < N; ++i)
    for (int j = i; j < N; ++j)
      pairwise[i][j] = pairwise[j][i] = probA.dist_by_ind(i, j);

  // Method B: use fill_distance_matrix.
  auto probB = make_problem(N);
  probB.fill_distance_matrix();

  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < N; ++j) {
      REQUIRE_THAT(probB.dist_by_ind(i, j), WithinAbs(pairwise[i][j], 1e-15));
    }
  }
}

// ---------------------------------------------------------------------------
// 6. Distance matrix is marked as filled after fill_distance_matrix
// ---------------------------------------------------------------------------
TEST_CASE("is_distance_matrix_filled flag is set correctly", "[Phase1][distance_matrix]")
{
  constexpr int N = 5;
  auto prob = make_problem(N);

  REQUIRE_FALSE(prob.is_distance_matrix_filled());
  prob.fill_distance_matrix();
  REQUIRE(prob.is_distance_matrix_filled());
}

// ---------------------------------------------------------------------------
// 7. DTW full vs banded consistency: with default band=-1, banded == full
// ---------------------------------------------------------------------------
TEST_CASE("Default band produces same distances as dtwFull", "[Phase1][distance_matrix]")
{
  constexpr int N = 5;
  auto prob = make_problem(N);

  // Default band is -1 (full DTW).
  REQUIRE(prob.band == settings::DEFAULT_BAND);
  REQUIRE(prob.band == -1);

  prob.fill_distance_matrix();

  // Compare against dtwFull computed directly on the data vectors.
  for (int i = 0; i < N; ++i) {
    for (int j = i; j < N; ++j) {
      double expected = dtwFull<data_t>(prob.p_vec(i), prob.p_vec(j));
      REQUIRE_THAT(prob.dist_by_ind(i, j), WithinAbs(expected, 1e-12));
    }
  }
}

// ---------------------------------------------------------------------------
// 8. LowerBoundStrategy variants produce the same distance matrix
//    (pruning reduces work, never changes results).
// ---------------------------------------------------------------------------
TEST_CASE("LowerBoundStrategy variants yield identical results", "[Phase1][distance_matrix][lb_strategy]")
{
  // Explicitly set distance_strategy=Pruned so we exercise the pruned path
  // regardless of the Auto-gate threshold. Dummy dataset caps at 25 series.
  constexpr int N = 25;
  const LowerBoundStrategy strategies[] = {
    LowerBoundStrategy::Auto,
    LowerBoundStrategy::None,
    LowerBoundStrategy::Kim,
    LowerBoundStrategy::Keogh,
    LowerBoundStrategy::KimKeogh,
  };

  auto prob_ref = make_problem(N);
  const int actual_N = static_cast<int>(prob_ref.size());
  prob_ref.band = 3;
  prob_ref.distance_strategy = DistanceMatrixStrategy::Pruned;
  prob_ref.set_lb_strategy(LowerBoundStrategy::Auto);
  prob_ref.fill_distance_matrix();

  for (auto strat : strategies) {
    auto prob = make_problem(N);
    prob.band = 3;
    prob.distance_strategy = DistanceMatrixStrategy::Pruned;
    prob.set_lb_strategy(strat);
    prob.fill_distance_matrix();

    for (int i = 0; i < actual_N; ++i) {
      for (int j = 0; j < actual_N; ++j) {
        CAPTURE(static_cast<int>(strat), i, j);
        REQUIRE_THAT(prob.dist_by_ind(i, j),
                     WithinAbs(prob_ref.dist_by_ind(i, j), 1e-12));
      }
    }
  }
}

// ---------------------------------------------------------------------------
// 9. Re-filling the distance matrix is a no-op when already filled
// ---------------------------------------------------------------------------
TEST_CASE("Repeated fill_distance_matrix is idempotent", "[Phase1][distance_matrix]")
{
  constexpr int N = 5;
  auto prob = make_problem(N);

  prob.fill_distance_matrix();
  REQUIRE(prob.is_distance_matrix_filled());

  // Save a few values.
  double d01 = prob.dist_by_ind(0, 1);
  double d23 = prob.dist_by_ind(2, 3);

  // Call again -- should be a no-op.
  REQUIRE_NOTHROW(prob.fill_distance_matrix());
  REQUIRE(prob.is_distance_matrix_filled());

  REQUIRE_THAT(prob.dist_by_ind(0, 1), WithinAbs(d01, 1e-15));
  REQUIRE_THAT(prob.dist_by_ind(2, 3), WithinAbs(d23, 1e-15));
}
