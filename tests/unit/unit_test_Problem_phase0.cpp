/**
 * @file unit_test_Problem_phase0.cpp
 * @brief Phase 0 bug-detection tests for Problem class and related functionality.
 *
 * @details These tests verify known bugs and properties of the Problem class.
 * Some tests are EXPECTED TO FAIL on unmodified code -- they prove bugs exist.
 *
 * EXPECTED FAILURES on unmodified code:
 *   - "[Phase0] writeMedoids throws std::runtime_error on bad path"
 *     Problem_IO.cpp line 42 does `throw 1` (an int) instead of
 *     `throw std::runtime_error(...)`. The test uses REQUIRE_THROWS_AS(...,
 *     std::runtime_error) which will FAIL until the bug is fixed.
 *
 * @date 28 Mar 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>
#include <string>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

// ---------------------------------------------------------------------------
// Helper: build a small Problem with N dummy time series (no file I/O needed).
// ---------------------------------------------------------------------------
static Problem make_small_problem(int N = 5)
{
  std::vector<std::vector<data_t>> vecs;
  std::vector<std::string> names;
  for (int i = 0; i < N; ++i) {
    // Create simple, distinct time series of varying lengths.
    std::vector<data_t> ts;
    for (int j = 0; j <= i + 2; ++j)
      ts.push_back(static_cast<data_t>(i * 10 + j));
    vecs.push_back(std::move(ts));
    names.push_back("ts_" + std::to_string(i));
  }

  Data data(std::move(vecs), std::move(names));
  Problem prob("phase0_test");
  prob.set_data(std::move(data));
  return prob;
}

// ---------------------------------------------------------------------------
// Test 1: Exception type -- writeMedoids throws int, not std::runtime_error
// ---------------------------------------------------------------------------
TEST_CASE("[Phase0] writeMedoids throws std::runtime_error on bad path",
          "[Phase0][Problem][exception]")
{
  // Build a small problem and set the output folder to an invalid path
  // so that writeMedoids fails to open the file.
  Problem prob = make_small_problem(5);
  prob.set_numberOfClusters(2);
  prob.N_repetition = 1;
  prob.maxIter = 1;

  // Point output to a non-existent directory that cannot be created.
  // On both Windows and Unix this should fail to open a file.
  prob.output_folder = "/nonexistent_dir_phase0_test/deep/nested/path";

  // cluster_by_kMedoidsLloyd() eventually calls writeMedoids() which
  // currently does `throw 1` (an int).
  // After the fix it should throw std::runtime_error.
  //
  // EXPECTED TO FAIL on unmodified code: REQUIRE_THROWS_AS expects
  // std::runtime_error but gets int.
  REQUIRE_THROWS_AS(prob.cluster_by_kMedoidsLloyd(), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Test 4: fillDistanceMatrix properties
// ---------------------------------------------------------------------------
TEST_CASE("[Phase0] fillDistanceMatrix properties",
          "[Phase0][Problem][distanceMatrix]")
{
  constexpr int N = 5;
  Problem prob = make_small_problem(N);

  // Fill the distance matrix.
  prob.fillDistanceMatrix();
  REQUIRE(prob.isDistanceMatrixFilled());

  SECTION("diagonal is zero")
  {
    for (int i = 0; i < N; ++i) {
      const double d = prob.distByInd(i, i);
      REQUIRE_THAT(d, WithinAbs(0.0, 1e-15));
    }
  }

  SECTION("symmetry: distMat(i,j) == distMat(j,i)")
  {
    for (int i = 0; i < N; ++i) {
      for (int j = i + 1; j < N; ++j) {
        const double dij = prob.distByInd(i, j);
        const double dji = prob.distByInd(j, i);
        REQUIRE_THAT(dij, WithinAbs(dji, 1e-15));
      }
    }
  }

  SECTION("non-negativity")
  {
    for (int i = 0; i < N; ++i) {
      for (int j = 0; j < N; ++j) {
        REQUIRE(prob.distByInd(i, j) >= 0.0);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Test 5: Data::size() type and correctness
// ---------------------------------------------------------------------------
TEST_CASE("[Phase0] Data::size() returns correct value",
          "[Phase0][Data][size]")
{
  SECTION("empty Data has size 0")
  {
    Data data;
    REQUIRE(data.size() == 0);
  }

  SECTION("Data with 3 entries has size 3")
  {
    std::vector<std::vector<data_t>> vecs = { { 1.0 }, { 2.0, 3.0 }, { 4.0, 5.0, 6.0 } };
    std::vector<std::string> names = { "a", "b", "c" };
    Data data(std::move(vecs), std::move(names));
    REQUIRE(data.size() == 3);
  }

  SECTION("Data with 1 entry has size 1")
  {
    std::vector<std::vector<data_t>> vecs = { { 42.0 } };
    std::vector<std::string> names = { "only" };
    Data data(std::move(vecs), std::move(names));
    REQUIRE(data.size() == 1);
  }

  SECTION("size() is usable as both int and size_t comparison")
  {
    // Data::size() currently returns int. This section verifies it works
    // in contexts expecting either int or size_t, so the test remains
    // valid if the return type is later changed to size_t.
    std::vector<std::vector<data_t>> vecs = { { 1.0 }, { 2.0 } };
    std::vector<std::string> names = { "x", "y" };
    Data data(std::move(vecs), std::move(names));

    int int_size = data.size();
    size_t size_t_size = static_cast<size_t>(data.size());

    REQUIRE(int_size == 2);
    REQUIRE(size_t_size == 2u);
    REQUIRE(data.p_vec.size() == static_cast<size_t>(data.size()));
  }
}
