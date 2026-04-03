/**
 * @file unit_test_mv_lower_bounds.cpp
 * @brief Unit tests for per-channel multivariate LB_Keogh and SquaredL2 LB variants.
 * @author Volkan Kumtepeli
 *
 * @details Tests cover:
 *   - compute_envelopes_mv: ndim=1 parity with scalar, ndim=2 per-channel correctness
 *   - lb_keogh_mv: ndim=1 parity with scalar, valid lower bound on MV DTW, zero for identical
 *   - lb_keogh_squared: valid lower bound on univariate SquaredL2 DTW
 *   - lb_keogh_mv_squared: valid lower bound on multivariate SquaredL2 DTW
 */

#include <dtwc.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc::core;

// =========================================================================
//  compute_envelopes_mv
// =========================================================================

TEST_CASE("MV envelopes: ndim=1 matches scalar", "[mv][lb]")
{
  std::vector<double> s = {1, 3, 2, 5, 4};
  std::vector<double> u1(5), l1(5), u2(5), l2(5);
  compute_envelopes(s.data(), 5, 1, u1.data(), l1.data());
  compute_envelopes_mv(s.data(), 5, 1, 1, u2.data(), l2.data());
  for (int i = 0; i < 5; ++i) {
    REQUIRE(u1[i] == u2[i]);
    REQUIRE(l1[i] == l2[i]);
  }
}

TEST_CASE("MV envelopes: ndim=2 per-channel", "[mv][lb]")
{
  // 3 timesteps x 2 features: [(1,10), (3,20), (2,30)]
  double s[] = {1,10, 3,20, 2,30};
  double u[6], l[6];
  compute_envelopes_mv(s, 3, 2, 1, u, l);

  // Channel 0: {1,3,2}, band=1
  //   t0: window [0,1] -> max(1,3)=3, min(1,3)=1
  //   t1: window [0,2] -> max(1,3,2)=3, min(1,3,2)=1
  //   t2: window [1,2] -> max(3,2)=3, min(3,2)=2
  REQUIRE(u[0*2+0] == 3.0); REQUIRE(l[0*2+0] == 1.0);
  REQUIRE(u[1*2+0] == 3.0); REQUIRE(l[1*2+0] == 1.0);
  REQUIRE(u[2*2+0] == 3.0); REQUIRE(l[2*2+0] == 2.0);

  // Channel 1: {10,20,30}, band=1
  //   t0: window [0,1] -> max(10,20)=20, min(10,20)=10
  //   t1: window [0,2] -> max(10,20,30)=30, min(10,20,30)=10
  //   t2: window [1,2] -> max(20,30)=30, min(20,30)=20
  REQUIRE(u[0*2+1] == 20.0); REQUIRE(l[0*2+1] == 10.0);
  REQUIRE(u[1*2+1] == 30.0); REQUIRE(l[1*2+1] == 10.0);
  REQUIRE(u[2*2+1] == 30.0); REQUIRE(l[2*2+1] == 20.0);
}

TEST_CASE("MV envelopes: empty series is a no-op", "[mv][lb][edge]")
{
  double u[4], l[4];
  // Must not crash; output is undefined but the call must return cleanly.
  compute_envelopes_mv(static_cast<double*>(nullptr), 0, 2, 1, u, l);
}

TEST_CASE("MV envelopes: band=0 produces point envelopes", "[mv][lb]")
{
  // band=0 means each point's envelope equals itself
  double s[] = {1,2, 3,4, 5,6};
  double u[6], l[6];
  compute_envelopes_mv(s, 3, 2, 0, u, l);
  for (int i = 0; i < 6; ++i) {
    REQUIRE(u[i] == s[i]);
    REQUIRE(l[i] == s[i]);
  }
}

// =========================================================================
//  lb_keogh_mv
// =========================================================================

TEST_CASE("MV lb_keogh: ndim=1 matches scalar", "[mv][lb]")
{
  double s[] = {1, 3, 2, 5, 4};
  double q[] = {2, 4, 1, 6, 3};
  double u[5], l_[5];
  compute_envelopes(s, 5, 1, u, l_);

  double lb1 = lb_keogh(q, 5, u, l_);
  double lb2 = lb_keogh_mv(q, 5, 1, u, l_);
  REQUIRE_THAT(lb1, WithinAbs(lb2, 1e-10));
}

TEST_CASE("MV lb_keogh: valid lower bound on MV DTW", "[mv][lb]")
{
  // 4 timesteps x 2 features
  double x[] = {0,0, 5,5, 2,2, 8,8};
  double y[] = {1,1, 4,4, 3,3, 7,7};
  double u[8], l_[8];
  compute_envelopes_mv(y, 4, 2, 1, u, l_);

  double lb = lb_keogh_mv(x, 4, 2, u, l_);
  double dtw_d = dtwc::dtwBanded_mv(x, 4, y, 4, 2, 1);

  REQUIRE(lb <= dtw_d + 1e-10);  // LB <= DTW
  REQUIRE(lb >= 0.0);
}

TEST_CASE("MV lb_keogh: zero for identical series", "[mv][lb]")
{
  double s[] = {1,2, 3,4, 5,6};
  double u[6], l_[6];
  compute_envelopes_mv(s, 3, 2, 1, u, l_);
  REQUIRE(lb_keogh_mv(s, 3, 2, u, l_) == 0.0);
}

TEST_CASE("MV lb_keogh: non-negative for random series", "[mv][lb]")
{
  // LB must always be >= 0 regardless of input
  double x[] = {-5.0, 3.0, -1.0, 2.0, -4.0, 7.0};
  double y[] = { 1.0,-2.0,  4.0,-3.0,  6.0,-5.0};
  double u[6], l_[6];
  compute_envelopes_mv(y, 3, 2, 1, u, l_);
  double lb = lb_keogh_mv(x, 3, 2, u, l_);
  REQUIRE(lb >= 0.0);
}

TEST_CASE("MV lb_keogh: large band = no pruning (LB trivially zero for identical)", "[mv][lb]")
{
  double s[] = {1,2, 3,4};
  double u[4], l_[4];
  // band >= n_steps-1 means envelopes cover the whole series: every point within every other's window
  compute_envelopes_mv(s, 2, 2, 100, u, l_);
  REQUIRE(lb_keogh_mv(s, 2, 2, u, l_) == 0.0);
}

// =========================================================================
//  lb_keogh_squared
// =========================================================================

TEST_CASE("lb_keogh_squared: valid lower bound on SquaredL2 DTW", "[lb][squared]")
{
  double x[] = {0, 5, 2, 8};
  double y[] = {1, 4, 3, 7};
  double u[4], l_[4];
  compute_envelopes(y, 4, 1, u, l_);

  double lb = lb_keogh_squared(x, 4, u, l_);
  double dtw_d = dtwc::dtwBanded(x, 4, y, 4, 1, -1.0, dtwc::core::MetricType::SquaredL2);

  REQUIRE(lb <= dtw_d + 1e-10);
  REQUIRE(lb >= 0.0);
}

TEST_CASE("lb_keogh_squared: zero for series within envelope", "[lb][squared]")
{
  // If the query is identical to the candidate, LB must be zero
  double s[] = {1, 3, 2, 5};
  double u[4], l_[4];
  compute_envelopes(s, 4, 1, u, l_);
  REQUIRE(lb_keogh_squared(s, 4, u, l_) == 0.0);
}

TEST_CASE("lb_keogh_squared: known value", "[lb][squared]")
{
  // Upper = [2], Lower = [0]; query = [3]
  // excess = 3-2 = 1; sum = 1^2 = 1
  double q[] = {3.0};
  double u[] = {2.0};
  double l_[] = {0.0};
  REQUIRE(lb_keogh_squared(q, 1, u, l_) == 1.0);
}

// =========================================================================
//  lb_keogh_mv_squared
// =========================================================================

TEST_CASE("MV lb_keogh_squared: ndim=1 matches scalar lb_keogh_squared", "[mv][lb][squared]")
{
  double x[] = {0, 5, 2, 8};
  double y[] = {1, 4, 3, 7};
  double u[4], l_[4];
  compute_envelopes(y, 4, 1, u, l_);

  double lb1 = lb_keogh_squared(x, 4, u, l_);
  double lb2 = lb_keogh_mv_squared(x, 4, 1, u, l_);
  REQUIRE_THAT(lb1, WithinAbs(lb2, 1e-10));
}

TEST_CASE("MV lb_keogh_squared: valid lower bound on MV SquaredL2 DTW", "[mv][lb][squared]")
{
  double x[] = {0,0, 5,5, 2,2, 8,8};
  double y[] = {1,1, 4,4, 3,3, 7,7};
  double u[8], l_[8];
  compute_envelopes_mv(y, 4, 2, 1, u, l_);

  double lb = lb_keogh_mv_squared(x, 4, 2, u, l_);
  double dtw_d = dtwc::dtwBanded_mv(x, 4, y, 4, 2, 1, -1.0, dtwc::core::MetricType::SquaredL2);

  REQUIRE(lb <= dtw_d + 1e-10);
  REQUIRE(lb >= 0.0);
}

TEST_CASE("MV lb_keogh_squared: zero for identical series", "[mv][lb][squared]")
{
  double s[] = {1,2, 3,4, 5,6};
  double u[6], l_[6];
  compute_envelopes_mv(s, 3, 2, 1, u, l_);
  REQUIRE(lb_keogh_mv_squared(s, 3, 2, u, l_) == 0.0);
}

TEST_CASE("MV lb_keogh_squared: non-negative for arbitrary input", "[mv][lb][squared]")
{
  double x[] = {-5.0, 3.0, -1.0, 2.0, -4.0, 7.0};
  double y[] = { 1.0,-2.0,  4.0,-3.0,  6.0,-5.0};
  double u[6], l_[6];
  compute_envelopes_mv(y, 3, 2, 1, u, l_);
  double lb = lb_keogh_mv_squared(x, 3, 2, u, l_);
  REQUIRE(lb >= 0.0);
}
