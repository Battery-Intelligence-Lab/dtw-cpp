/**
 * @file unit_test_mv_missing.cpp
 * @brief Unit tests for multivariate missing-data DTW (per-channel NaN handling).
 *
 * @details Covers dtwMissing_L_mv and dtwMissing_banded_mv with the zero-cost
 * philosophy: if channel d is NaN in either series at a timestep pair (i,j),
 * that channel contributes 0 to the pointwise cost. Other channels still
 * contribute normally.
 *
 * Also covers Problem::ZeroCost wiring for ndim > 1.
 *
 * @author Volkan Kumtepeli
 * @date 02 Apr 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <limits>
#include <vector>
#include <cmath>

using Catch::Matchers::WithinAbs;

static const double NaN = std::numeric_limits<double>::quiet_NaN();

// =========================================================================
//  MissingMVL1Dist functor tests
// =========================================================================

TEST_CASE("MissingMVL1Dist: no NaN matches MVL1Dist", "[mv][missing][functor]")
{
  double a[] = {1.0, 2.0, 3.0};
  double b[] = {4.0, 1.0, 6.0};
  // |1-4| + |2-1| + |3-6| = 3 + 1 + 3 = 7
  REQUIRE(dtwc::detail::MissingMVL1Dist{}(a, b, 3) == 7.0);
}

TEST_CASE("MissingMVL1Dist: NaN in a skips that channel", "[mv][missing][functor]")
{
  double a[] = {NaN, 2.0};
  double b[] = {4.0, 1.0};
  // channel 0: NaN -> skip; channel 1: |2-1| = 1
  REQUIRE(dtwc::detail::MissingMVL1Dist{}(a, b, 2) == 1.0);
}

TEST_CASE("MissingMVL1Dist: NaN in b skips that channel", "[mv][missing][functor]")
{
  double a[] = {1.0, 2.0};
  double b[] = {NaN, 5.0};
  // channel 0: NaN -> skip; channel 1: |2-5| = 3
  REQUIRE(dtwc::detail::MissingMVL1Dist{}(a, b, 2) == 3.0);
}

TEST_CASE("MissingMVL1Dist: all NaN returns 0", "[mv][missing][functor]")
{
  double a[] = {NaN, NaN};
  double b[] = {1.0, 2.0};
  REQUIRE(dtwc::detail::MissingMVL1Dist{}(a, b, 2) == 0.0);
}

TEST_CASE("MissingMVL1Dist: symmetry with NaN", "[mv][missing][functor]")
{
  double a[] = {NaN, 3.0};
  double b[] = {2.0, NaN};
  // channel 0: a is NaN -> 0; channel 1: b is NaN -> 0; total = 0
  REQUIRE(dtwc::detail::MissingMVL1Dist{}(a, b, 2) == 0.0);
  REQUIRE(dtwc::detail::MissingMVL1Dist{}(b, a, 2) == 0.0);
}

// =========================================================================
//  MissingMVSquaredL2Dist functor tests
// =========================================================================

TEST_CASE("MissingMVSquaredL2Dist: no NaN matches MVSquaredL2Dist", "[mv][missing][functor]")
{
  double a[] = {1.0, 2.0};
  double b[] = {3.0, 5.0};
  // (1-3)^2 + (2-5)^2 = 4 + 9 = 13
  REQUIRE(dtwc::detail::MissingMVSquaredL2Dist{}(a, b, 2) == 13.0);
}

TEST_CASE("MissingMVSquaredL2Dist: NaN skips channel", "[mv][missing][functor]")
{
  double a[] = {NaN, 2.0};
  double b[] = {3.0, 5.0};
  // channel 0: NaN -> skip; channel 1: (2-5)^2 = 9
  REQUIRE(dtwc::detail::MissingMVSquaredL2Dist{}(a, b, 2) == 9.0);
}

// =========================================================================
//  dtwMissing_L_mv: no NaN matches standard MV DTW
// =========================================================================

TEST_CASE("MV Missing: no NaN matches standard MV DTW", "[mv][missing]")
{
  double x[] = {1,2, 3,4, 5,6};
  double y[] = {2,3, 4,5, 6,7};
  double d_std  = dtwc::dtwFull_L_mv(x, 3, y, 3, 2);
  double d_miss = dtwc::dtwMissing_L_mv(x, 3, y, 3, 2);
  REQUIRE_THAT(d_miss, WithinAbs(d_std, 1e-10));
}

TEST_CASE("MV Missing: ndim=1 matches scalar missing DTW", "[mv][missing]")
{
  std::vector<double> x = {1, NaN, 3};
  std::vector<double> y = {2, 3, 4};
  double d_scalar = dtwc::dtwMissing_L(x, y);
  double d_mv     = dtwc::dtwMissing_L_mv(x.data(), 3, y.data(), 3, 1);
  REQUIRE_THAT(d_mv, WithinAbs(d_scalar, 1e-10));
}

// =========================================================================
//  dtwMissing_L_mv: all NaN = 0
// =========================================================================

TEST_CASE("MV Missing: all NaN = 0", "[mv][missing]")
{
  double x[] = {NaN, NaN, NaN, NaN};
  double y[] = {1.0, 2.0, 3.0, 4.0};
  // every channel in x is NaN so all per-timestep costs are 0
  REQUIRE(dtwc::dtwMissing_L_mv(x, 2, y, 2, 2) == 0.0);
}

TEST_CASE("MV Missing: both series all NaN = 0", "[mv][missing]")
{
  double x[] = {NaN, NaN};
  double y[] = {NaN, NaN};
  REQUIRE(dtwc::dtwMissing_L_mv(x, 1, y, 1, 2) == 0.0);
}

// =========================================================================
//  dtwMissing_L_mv: NaN reduces distance (one channel NaN)
// =========================================================================

TEST_CASE("MV Missing: one channel NaN reduces distance", "[mv][missing]")
{
  // ndim=2, 2 timesteps each
  // x_clean = [(1,10), (2,20)], y = [(1,100), (2,200)]
  // x_nan   = [(1,NaN), (2,20)]  — channel 1 missing at t=0
  double x_clean[] = {1,10,  2,20};
  double x_nan[]   = {1,NaN, 2,20};
  double y[]       = {1,100, 2,200};

  double d_clean = dtwc::dtwMissing_L_mv(x_clean, 2, y, 2, 2);
  double d_nan   = dtwc::dtwMissing_L_mv(x_nan,   2, y, 2, 2);

  // NaN in channel 1 at t=0 contributes 0 instead of |10-100|=90, so d_nan < d_clean
  REQUIRE(d_nan < d_clean);
  REQUIRE(d_nan >= 0.0);
}

// =========================================================================
//  dtwMissing_L_mv: symmetry
// =========================================================================

TEST_CASE("MV Missing: symmetry (NaN in x and y)", "[mv][missing]")
{
  double x[] = {1,NaN, 3,4};
  double y[] = {NaN,2, 3,5};
  double d1 = dtwc::dtwMissing_L_mv(x, 2, y, 2, 2);
  double d2 = dtwc::dtwMissing_L_mv(y, 2, x, 2, 2);
  REQUIRE_THAT(d1, WithinAbs(d2, 1e-10));
}

TEST_CASE("MV Missing: symmetry (no NaN)", "[mv][missing]")
{
  double x[] = {1,2, 3,4, 5,6};
  double y[] = {6,5, 4,3, 2,1};
  double d1 = dtwc::dtwMissing_L_mv(x, 3, y, 3, 2);
  double d2 = dtwc::dtwMissing_L_mv(y, 3, x, 3, 2);
  REQUIRE_THAT(d1, WithinAbs(d2, 1e-10));
}

// =========================================================================
//  dtwMissing_L_mv: non-negative
// =========================================================================

TEST_CASE("MV Missing: non-negative with mixed NaN pattern", "[mv][missing]")
{
  // ndim=3, 2 timesteps
  double x[] = {NaN, 1,   NaN,
                2,   NaN, 3};
  double y[] = {4,   NaN, 5,
                NaN, NaN, 6};
  REQUIRE(dtwc::dtwMissing_L_mv(x, 2, y, 2, 3) >= 0.0);
}

// =========================================================================
//  dtwMissing_L_mv: hand-computed values
// =========================================================================

TEST_CASE("MV Missing: hand-computed L1 (ndim=2, 2 steps)", "[mv][missing]")
{
  // ndim=2, nx=ny=2
  // x = [(1,NaN), (3,4)], y = [(2,100), (3,5)]
  //
  // Pointwise costs (L1, MissingMVL1Dist):
  //   cost(x[0], y[0]) = |1-2| + skip(NaN) = 1
  //   cost(x[0], y[1]) = |1-3| + skip(NaN) = 2
  //   cost(x[1], y[0]) = |3-2| + |4-100| = 1 + 96 = 97
  //   cost(x[1], y[1]) = |3-3| + |4-5| = 0 + 1 = 1
  //
  // DTW cost matrix (2x2, indices: x-row, y-col):
  //   C(0,0) = cost(x[0],y[0]) = 1
  //   C(1,0) = C(0,0) + cost(x[1],y[0]) = 1 + 97 = 98
  //   C(0,1) = C(0,0) + cost(x[0],y[1]) = 1 + 2 = 3
  //   C(1,1) = min(C(0,0), C(1,0), C(0,1)) + cost(x[1],y[1])
  //          = min(1, 98, 3) + 1 = 1 + 1 = 2
  //
  // Expected result: 2
  double x[] = {1,NaN, 3,4};
  double y[] = {2,100, 3,5};
  REQUIRE_THAT(dtwc::dtwMissing_L_mv(x, 2, y, 2, 2), WithinAbs(2.0, 1e-10));
}

TEST_CASE("MV Missing: hand-computed SquaredL2 (ndim=2, 2 steps)", "[mv][missing]")
{
  // ndim=2, nx=ny=2
  // x = [(1,NaN), (3,4)], y = [(2,100), (3,5)]
  //
  // Pointwise costs (SquaredL2, MissingMVSquaredL2Dist):
  //   cost(x[0], y[0]) = (1-2)^2 + skip(NaN) = 1
  //   cost(x[0], y[1]) = (1-3)^2 + skip(NaN) = 4
  //   cost(x[1], y[0]) = (3-2)^2 + (4-100)^2 = 1 + 9216 = 9217
  //   cost(x[1], y[1]) = (3-3)^2 + (4-5)^2 = 0 + 1 = 1
  //
  // DTW cost matrix:
  //   C(0,0) = 1
  //   C(1,0) = 1 + 9217 = 9218
  //   C(0,1) = 1 + 4 = 5
  //   C(1,1) = min(1, 9218, 5) + 1 = 1 + 1 = 2
  double x[] = {1,NaN, 3,4};
  double y[] = {2,100, 3,5};
  double d = dtwc::dtwMissing_L_mv(x, 2, y, 2, 2, -1.0, dtwc::core::MetricType::SquaredL2);
  REQUIRE_THAT(d, WithinAbs(2.0, 1e-10));
}

// =========================================================================
//  dtwMissing_L_mv: NaN does not increase distance vs clean version
// =========================================================================

TEST_CASE("MV Missing: adding NaN does not increase distance", "[mv][missing]")
{
  // If we replace a channel value with NaN, the cost should not increase
  // because we zero out a previously positive cost contribution.
  double x_clean[] = {1,10,  2,20};
  double x_nan[]   = {1,NaN, 2,20};
  double y[]       = {1,5,   2,30};

  double d_clean = dtwc::dtwMissing_L_mv(x_clean, 2, y, 2, 2);
  double d_nan   = dtwc::dtwMissing_L_mv(x_nan,   2, y, 2, 2);

  REQUIRE(d_nan <= d_clean + 1e-10);
}

// =========================================================================
//  dtwMissing_L_mv: edge cases
// =========================================================================

TEST_CASE("MV Missing: empty series returns max", "[mv][missing][edge]")
{
  double x[] = {1.0, 2.0};
  constexpr double maxVal = std::numeric_limits<double>::max();
  REQUIRE(dtwc::dtwMissing_L_mv(x, 1, static_cast<double*>(nullptr), 0, 2) == maxVal);
  REQUIRE(dtwc::dtwMissing_L_mv(static_cast<double*>(nullptr), 0, x, 1, 2) == maxVal);
}

TEST_CASE("MV Missing: same pointer same timesteps = 0", "[mv][missing][edge]")
{
  double x[] = {1,NaN, 3,4, 5,6};
  // same-pointer identity shortcut should apply
  REQUIRE(dtwc::dtwMissing_L_mv(x, 3, x, 3, 2) == 0.0);
}

// =========================================================================
//  dtwMissing_banded_mv: basic dispatch
// =========================================================================

TEST_CASE("MV Missing banded: negative band delegates to unbanded", "[mv][missing][banded]")
{
  double x[] = {1,NaN, 3,4};
  double y[] = {NaN,2, 3,5};
  double d_full  = dtwc::dtwMissing_L_mv(x, 2, y, 2, 2);
  double d_banded = dtwc::dtwMissing_banded_mv(x, 2, y, 2, 2, -1);
  REQUIRE_THAT(d_full, WithinAbs(d_banded, 1e-10));
}

TEST_CASE("MV Missing banded: ndim=1 delegates to scalar missing banded", "[mv][missing][banded]")
{
  std::vector<double> x = {1, NaN, 3, 4, 5};
  std::vector<double> y = {2, 3, NaN, 5, 6};
  int band = 2;
  double d_scalar = dtwc::dtwMissing_banded(x, y, band);
  double d_mv     = dtwc::dtwMissing_banded_mv(x.data(), 5, y.data(), 5, 1, band);
  REQUIRE_THAT(d_mv, WithinAbs(d_scalar, 1e-10));
}

TEST_CASE("MV Missing banded: non-negative with NaN", "[mv][missing][banded]")
{
  double x[] = {NaN,1, 2,NaN, 3,4};
  double y[] = {1,NaN, NaN,2, 4,3};
  REQUIRE(dtwc::dtwMissing_banded_mv(x, 3, y, 3, 2, 2) >= 0.0);
}

// =========================================================================
//  Problem: ZeroCost with ndim=2
// =========================================================================

TEST_CASE("Problem ZeroCost ndim=2: distances are finite and non-negative", "[mv][missing][problem]")
{
  dtwc::Data data;
  data.ndim = 2;
  data.p_vec = {
    {1,2,  3,4},           // series 0: clean
    {1,NaN, 3,4},           // series 1: NaN in channel 1 at t=0
    {100,200, 300,400}      // series 2: far away
  };
  data.p_names = {"a", "b", "c"};

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.missing_strategy = dtwc::core::MissingStrategy::ZeroCost;
  prob.verbose = false;
  prob.fillDistanceMatrix();

  double d01 = prob.distByInd(0, 1);
  double d02 = prob.distByInd(0, 2);
  double d12 = prob.distByInd(1, 2);

  REQUIRE(d01 >= 0.0);
  REQUIRE(d02 >= 0.0);
  REQUIRE(d12 >= 0.0);
  REQUIRE(!dtwc::is_missing(d01));
  REQUIRE(!dtwc::is_missing(d02));
  REQUIRE(!dtwc::is_missing(d12));
}

TEST_CASE("Problem ZeroCost ndim=2: NaN reduces distance vs clean", "[mv][missing][problem]")
{
  // Series 0 and 1 differ only in channel 1 at t=0 (NaN vs large value).
  // Series 2 is far from both series 0 and 1 in channel 1.
  // d(series_nan, series_far) should be < d(series_clean, series_far)
  // because the NaN in channel 1 eliminates a large cost contribution.
  dtwc::Data data_clean;
  data_clean.ndim = 2;
  data_clean.p_vec = {
    {1,10,    3,20},    // series 0: x_clean
    {1,10000, 3,20000}  // series 1: far in channel 1
  };
  data_clean.p_names = {"x_clean", "far"};

  dtwc::Problem prob_clean;
  prob_clean.set_data(std::move(data_clean));
  prob_clean.missing_strategy = dtwc::core::MissingStrategy::ZeroCost;
  prob_clean.verbose = false;
  prob_clean.fillDistanceMatrix();
  double d_clean = prob_clean.distByInd(0, 1);

  dtwc::Data data_nan;
  data_nan.ndim = 2;
  data_nan.p_vec = {
    {1,NaN,   3,20},    // series 0: x_nan (channel 1 at t=0 is NaN)
    {1,10000, 3,20000}  // series 1: far in channel 1
  };
  data_nan.p_names = {"x_nan", "far"};

  dtwc::Problem prob_nan;
  prob_nan.set_data(std::move(data_nan));
  prob_nan.missing_strategy = dtwc::core::MissingStrategy::ZeroCost;
  prob_nan.verbose = false;
  prob_nan.fillDistanceMatrix();
  double d_nan = prob_nan.distByInd(0, 1);

  // NaN removes the large channel-1 cost at t=0, so d_nan < d_clean
  REQUIRE(d_nan < d_clean);
}

TEST_CASE("Problem ZeroCost ndim=2: identical series gives 0", "[mv][missing][problem]")
{
  dtwc::Data data;
  data.ndim = 2;
  data.p_vec = {
    {1,2, 3,4, 5,6},
    {1,2, 3,4, 5,6}
  };
  data.p_names = {"a", "b"};

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.missing_strategy = dtwc::core::MissingStrategy::ZeroCost;
  prob.verbose = false;
  prob.fillDistanceMatrix();

  REQUIRE_THAT(prob.distByInd(0, 1), WithinAbs(0.0, 1e-10));
}

TEST_CASE("Problem ZeroCost ndim=1 still works correctly", "[mv][missing][problem]")
{
  // Regression test: ndim=1 should still use the scalar path
  const double nan = std::numeric_limits<double>::quiet_NaN();
  dtwc::Data data;
  data.ndim = 1;
  data.p_vec = { {1.0, 2.0, 3.0}, {1.0, nan, 3.0} };
  data.p_names = { "a", "b" };

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.missing_strategy = dtwc::core::MissingStrategy::ZeroCost;
  prob.verbose = false;
  prob.fillDistanceMatrix();

  double d = prob.distByInd(0, 1);
  REQUIRE(d >= 0.0);
  REQUIRE(!dtwc::is_missing(d));
}

// =========================================================================
//  Phase 2 regression: dtwMissing_banded_mv is now a first-class banded
//  path (previously fell back to unbanded MV per the TODO on line 324).
//  Tight band must produce a higher distance than unbanded for a shifted-peak
//  pattern — proves the band is actually enforced.
// =========================================================================

TEST_CASE("dtwMissing_banded_mv actually restricts the warping path", "[mv][missing][banded]")
{
  std::vector<double> x = {0,0, 0,0, 9,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0};
  std::vector<double> y = {0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 9,0, 0,0};

  const double d_unbanded = dtwc::dtwMissing_L_mv<double>(
      x.data(), 10, y.data(), 10, 2);
  const double d_banded1  = dtwc::dtwMissing_banded_mv<double>(
      x.data(), 10, y.data(), 10, 2, 1);

  INFO("d_unbanded=" << d_unbanded << " d_banded1=" << d_banded1);
  REQUIRE(d_banded1 > d_unbanded); // band=1 forces extra cost vs free warp.
}
