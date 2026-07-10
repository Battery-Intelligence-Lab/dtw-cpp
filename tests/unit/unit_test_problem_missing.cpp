/**
 * @file unit_test_problem_missing.cpp
 * @brief Tests for Problem::missing_strategy wiring (Error / ZeroCost / Interpolate).
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#include <dtwc.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cmath>
#include <cstring>
#include <limits>

#ifdef _OPENMP
#include <omp.h>
#endif

using Catch::Matchers::ContainsSubstring;

namespace {
#ifdef _OPENMP
struct OmpThreadLimitGuard {
  int previous{omp_get_max_threads()};
  ~OmpThreadLimitGuard() { omp_set_num_threads(previous); }
};
#endif
} // namespace

static bool is_nan_bits(double d)
{
  uint64_t bits;
  std::memcpy(&bits, &d, sizeof(bits));
  return (bits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL
      && (bits & 0x000FFFFFFFFFFFFFULL) != 0;
}

TEST_CASE("Problem: MissingStrategy::Error throws on the caller thread",
          "[problem][missing][m40]")
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  dtwc::Data data;
  data.p_vec   = { {1.0, 2.0, 3.0}, {1.0, nan, 3.0} };
  data.p_names = { "a", "b" };

#ifdef _OPENMP
  OmpThreadLimitGuard thread_guard;
#endif
  for (const int threads : {1, 4}) {
    DYNAMIC_SECTION("OMP threads=" << threads) {
#ifdef _OPENMP
      omp_set_num_threads(threads);
#else
      (void)threads;
#endif
      dtwc::Problem prob;
      prob.set_data(data);
      prob.missing_strategy = dtwc::core::MissingStrategy::Error;
      prob.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
      prob.verbose = false;

      REQUIRE_THROWS_WITH(prob.fill_distance_matrix(),
        ContainsSubstring("NaN detected in series 'b' (index 1)"));
      CHECK_FALSE(prob.is_distance_matrix_filled());
      CHECK(prob.dense_distance_matrix().count_computed()
            < prob.dense_distance_matrix().packed_count());
    }
  }
}

TEST_CASE("Problem: OpenMP worker exceptions rethrow without publishing a full cache",
          "[problem][missing][m40]")
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
#ifdef _OPENMP
  OmpThreadLimitGuard thread_guard;
#endif
  for (const int threads : {1, 4}) {
    DYNAMIC_SECTION("OMP threads=" << threads) {
#ifdef _OPENMP
      omp_set_num_threads(threads);
#else
      (void)threads;
#endif
      dtwc::Data data;
      data.p_vec = {
        {0.0, 1.0, 2.0},
        {nan, nan, nan},
        {2.0, 1.0, 0.0},
        {4.0, 5.0, 6.0}
      };
      data.p_names = {"ordinary", "all-missing", "reverse", "offset"};
      dtwc::Problem prob;
      prob.set_data(std::move(data));
      prob.missing_strategy = dtwc::core::MissingStrategy::Interpolate;
      prob.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
      prob.verbose = false;

      REQUIRE_THROWS_WITH(prob.fill_distance_matrix(),
        ContainsSubstring("interpolate_linear: all values are NaN"));
      CHECK_FALSE(prob.is_distance_matrix_filled());
      CHECK(prob.dense_distance_matrix().count_computed()
            < prob.dense_distance_matrix().packed_count());
    }
  }
}

TEST_CASE("Problem: ordinary and ZeroCost distance fingerprints survive exception hardening",
          "[problem][missing][m40]")
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
#ifdef _OPENMP
  OmpThreadLimitGuard thread_guard;
#endif
  for (const int threads : {1, 4}) {
    DYNAMIC_SECTION("OMP threads=" << threads) {
#ifdef _OPENMP
      omp_set_num_threads(threads);
#else
      (void)threads;
#endif
      std::vector<std::vector<double>> ordinary{
        {0.0, 1.0, 2.0}, {0.0, 2.0, 2.0}, {2.0, 1.0, 0.0}
      };
      auto missing = ordinary;
      dtwc::Problem standard;
      standard.set_data(dtwc::Data(
        std::move(ordinary), std::vector<std::string>{"a", "b", "c"}));
      standard.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
      standard.verbose = false;
      standard.fill_distance_matrix();
      CHECK(standard.dist_by_ind(0, 1) == 1.0);
      CHECK(standard.dist_by_ind(0, 2) == 4.0);
      CHECK(standard.dist_by_ind(1, 2) == 5.0);
      CHECK(standard.is_distance_matrix_filled());

      missing[1][1] = nan;
      dtwc::Problem zero_cost;
      zero_cost.set_data(dtwc::Data(
        std::move(missing), std::vector<std::string>{"a", "b", "c"}));
      zero_cost.missing_strategy = dtwc::core::MissingStrategy::ZeroCost;
      zero_cost.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
      zero_cost.verbose = false;
      zero_cost.fill_distance_matrix();
      CHECK(zero_cost.dist_by_ind(0, 1) == 0.0);
      CHECK(zero_cost.dist_by_ind(0, 2) == 4.0);
      CHECK(zero_cost.dist_by_ind(1, 2) == 4.0);
      CHECK(zero_cost.is_distance_matrix_filled());
    }
  }
}

TEST_CASE("Problem: MissingStrategy::ZeroCost computes finite distances", "[problem][missing]")
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  dtwc::Data data;
  data.p_vec   = { {1.0, 2.0, 3.0}, {1.0, nan, 3.0}, {4.0, 5.0, 6.0} };
  data.p_names = { "a", "b", "c" };

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.missing_strategy = dtwc::core::MissingStrategy::ZeroCost;
  prob.verbose = false;
  prob.fillDistanceMatrix();

  const double d01 = prob.distByInd(0, 1);
  REQUIRE(d01 >= 0.0);
  REQUIRE_FALSE(is_nan_bits(d01));

  // ZeroCost: NaN position contributes 0, so d(a,b) should be less than d(a,c)
  const double d02 = prob.distByInd(0, 2);
  REQUIRE(d01 < d02);
}

TEST_CASE("Problem: MissingStrategy::Interpolate fills NaN and computes", "[problem][missing]")
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  dtwc::Data data;
  data.p_vec   = { {1.0, 2.0, 3.0}, {1.0, nan, 3.0} };
  data.p_names = { "a", "b" };

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.missing_strategy = dtwc::core::MissingStrategy::Interpolate;
  prob.verbose = false;
  prob.fillDistanceMatrix();

  // After interpolation, b becomes {1.0, 2.0, 3.0} — identical to a
  const double d01 = prob.distByInd(0, 1);
  REQUIRE(d01 < 1e-10);
}

TEST_CASE("Problem: No NaN with Error strategy works normally", "[problem][missing]")
{
  dtwc::Data data;
  data.p_vec   = { {1.0, 2.0, 3.0}, {4.0, 5.0, 6.0} };
  data.p_names = { "a", "b" };

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.missing_strategy = dtwc::core::MissingStrategy::Error;
  prob.verbose = false;
  prob.fillDistanceMatrix();

  const double d = prob.distByInd(0, 1);
  REQUIRE(d > 0.0);
}
