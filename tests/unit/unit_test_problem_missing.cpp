/**
 * @file unit_test_problem_missing.cpp
 * @brief Tests for Problem::missing_strategy wiring (Error / ZeroCost / Interpolate).
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#include <dtwc.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>
#include <cstring>
#include <limits>

static bool is_nan_bits(double d)
{
  uint64_t bits;
  std::memcpy(&bits, &d, sizeof(bits));
  return (bits & 0x7FF0000000000000ULL) == 0x7FF0000000000000ULL
      && (bits & 0x000FFFFFFFFFFFFFULL) != 0;
}

TEST_CASE("Problem: MissingStrategy::Error throws on NaN", "[problem][missing]")
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  dtwc::Data data;
  data.p_vec   = { {1.0, 2.0, 3.0}, {1.0, nan, 3.0} };
  data.p_names = { "a", "b" };

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.missing_strategy = dtwc::core::MissingStrategy::Error;
  prob.verbose = false;

  REQUIRE_THROWS_AS(prob.fillDistanceMatrix(), std::runtime_error);
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
