/**
 * @file unit_test_deferred_allocation.cpp
 * @brief Tests for deferred dense distance-matrix allocation.
 *
 * Verifies that Problem::set_data() does NOT immediately allocate the O(N^2)
 * distance matrix, and that the matrix is allocated lazily on the first call
 * to fill_distance_matrix(). Also verifies that dist_by_ind() works both before
 * and after allocation.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;

TEST_CASE("Deferred allocation: set_data does not allocate dense matrix", "[problem][deferred]")
{
  dtwc::Data data;
  // 1000 series — would be 8MB if allocated densely
  for (int i = 0; i < 1000; ++i) {
    data.p_vec.push_back({ static_cast<double>(i), static_cast<double>(i + 1) });
    data.p_names.push_back("s" + std::to_string(i));
  }

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.set_verbose(false);

  // Dense matrix should NOT be allocated yet
  REQUIRE(prob.dense_distance_matrix().size() == 0);
}

TEST_CASE("Deferred allocation: fill_distance_matrix allocates and fills", "[problem][deferred]")
{
  dtwc::Data data;
  data.p_vec = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
  data.p_names = { "a", "b", "c" };

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.set_verbose(false);

  REQUIRE(prob.dense_distance_matrix().size() == 0);

  prob.fill_distance_matrix();

  REQUIRE(prob.dense_distance_matrix().size() == 3);
  REQUIRE(prob.dist_by_ind(0, 1) > 0.0);
  REQUIRE(prob.dist_by_ind(0, 0) == 0.0);
}

TEST_CASE("Deferred allocation: dist_by_ind works without fill", "[problem][deferred]")
{
  dtwc::Data data;
  data.p_vec = { { 1, 2, 3 }, { 4, 5, 6 } };
  data.p_names = { "a", "b" };

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.set_verbose(false);

  // No fill_distance_matrix() called — dist_by_ind should still work
  double d = prob.dist_by_ind(0, 1);
  REQUIRE(d > 0.0);
  REQUIRE_THAT(d, WithinAbs(prob.dist_by_ind(1, 0), 1e-12)); // symmetry
}

TEST_CASE("Deferred allocation: FastPAM still works", "[problem][deferred]")
{
  dtwc::Data data;
  for (int i = 0; i < 20; ++i) {
    data.p_vec.push_back({ static_cast<double>(i % 2 == 0 ? 0 : 100),
                           static_cast<double>(i % 2 == 0 ? 1 : 101) });
    data.p_names.push_back("s" + std::to_string(i));
  }

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.set_verbose(false);

  // FastPAM calls fill_distance_matrix internally — should work
  auto result = dtwc::fast_pam(prob, 2);
  REQUIRE(result.labels.size() == 20);
  REQUIRE(result.medoid_indices.size() == 2);
  REQUIRE(result.total_cost >= 0.0);
}

TEST_CASE("Deferred allocation: set_variant works after set_data", "[problem][deferred]")
{
  dtwc::Data data;
  data.p_vec = { { 1, 2, 3 }, { 4, 5, 6 } };
  data.p_names = { "a", "b" };

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.set_variant(dtwc::core::DTWVariant::WDTW);
  prob.set_verbose(false);

  // Should work — rebind_dtw_fn was called by set_variant
  double d = prob.dist_by_ind(0, 1);
  REQUIRE(d > 0.0);
}

TEST_CASE("Deferred allocation: existing tests backward compat", "[problem][deferred]")
{
  // Small problem with fill
  dtwc::Data data;
  data.p_vec = { { 0 }, { 1 }, { 5 }, { 6 } };
  data.p_names = { "a", "b", "c", "d" };

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.set_verbose(false);
  prob.fill_distance_matrix();

  // Should work exactly as before
  REQUIRE_THAT(prob.dist_by_ind(0, 1), WithinAbs(1.0, 1e-12));
  REQUIRE_THAT(prob.dist_by_ind(0, 2), WithinAbs(5.0, 1e-12));
  REQUIRE_THAT(prob.dist_by_ind(2, 3), WithinAbs(1.0, 1e-12));
}

TEST_CASE("Deferred allocation: set_data then set_data resets matrix", "[problem][deferred]")
{
  dtwc::Data data1;
  data1.p_vec = { { 1, 2, 3 }, { 4, 5, 6 } };
  data1.p_names = { "a", "b" };

  dtwc::Problem prob;
  prob.set_data(std::move(data1));
  prob.set_verbose(false);
  prob.fill_distance_matrix();
  REQUIRE(prob.dense_distance_matrix().size() == 2);

  // Now set new data — matrix should be reset
  dtwc::Data data2;
  data2.p_vec = { { 10 }, { 20 }, { 30 } };
  data2.p_names = { "x", "y", "z" };
  prob.set_data(std::move(data2));

  // Matrix must be cleared (deferred again)
  REQUIRE(prob.dense_distance_matrix().size() == 0);

  // Fill again — should work with new data
  prob.fill_distance_matrix();
  REQUIRE(prob.dense_distance_matrix().size() == 3);
  REQUIRE_THAT(prob.dist_by_ind(0, 1), WithinAbs(10.0, 1e-12));
}
