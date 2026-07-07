/**
 * @file unit_test_dtw_api.cpp
 * @brief Unit tests for the unified DTW API (dtwc::core).
 *
 * @details Verifies that the new dtw_distance / dtw_runtime wrappers
 *          produce identical results to the underlying warping.hpp
 *          functions they delegate to.
 *
 * @author Volkan Kumtepeli
 * @date 28 Mar 2026
 */

#include <core/dtw.hpp>
#include <core/dtw_cost.hpp>
#include <core/lower_bounds.hpp>
#include <soft_dtw.hpp>
#include <warping.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cstddef>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc::core;

// ----- Compile-time lower-bound compatibility checks --------------------

static_assert(lb_keogh_valid<L1Metric>,
              "LB_Keogh must be valid for L1Metric");
static_assert(lb_keogh_valid<L2Metric>,
              "LB_Keogh must be valid for L2Metric (identical to L1 for scalars)");
static_assert(lb_keogh_valid<SquaredL2Metric>,
              "LB_Keogh must be valid for SquaredL2Metric");
static_assert(lb_kim_valid<L1Metric>,
              "LB_Kim must be valid for L1Metric");
static_assert(lb_kim_valid<L2Metric>,
              "LB_Kim must be valid for L2Metric");

// ----- dtw_distance (vector overload) -----------------------------------

TEST_CASE("dtw_distance matches dtwFull_L for unconstrained DTW",
          "[dtw_api][dtw_distance]")
{
  const std::vector<double> x{1, 2, 3};
  const std::vector<double> y{3, 4, 5, 6, 7};
  constexpr double ground_truth = 13.0;

  const double result = dtw_distance(x, y);
  const double ref = dtwc::dtwFull_L<double>(x, y);

  REQUIRE_THAT(result, WithinAbs(ground_truth, 1e-15));
  REQUIRE_THAT(result, WithinAbs(ref, 1e-15));
}

TEST_CASE("dtw_distance matches dtwBanded for banded DTW",
          "[dtw_api][dtw_distance]")
{
  const std::vector<double> x{1, 2, 3};
  const std::vector<double> y{3, 4, 5, 6, 7};
  constexpr double ground_truth = 13.0;

  const int band = 100;
  const double result = dtw_distance(x, y, band);
  const double ref = dtwc::dtwBanded<double>(x, y, band);

  REQUIRE_THAT(result, WithinAbs(ground_truth, 1e-15));
  REQUIRE_THAT(result, WithinAbs(ref, 1e-15));
}

// ----- dtw_distance (pointer overload) ----------------------------------

TEST_CASE("dtw_distance pointer overload matches vector overload",
          "[dtw_api][dtw_distance]")
{
  const std::vector<double> x{1, 2, 3};
  const std::vector<double> y{3, 4, 5, 6, 7};

  const double vec_result = dtw_distance(x, y);
  const double ptr_result = dtw_distance(x.data(), x.size(),
                                         y.data(), y.size());

  REQUIRE_THAT(ptr_result, WithinAbs(vec_result, 1e-15));
}

TEST_CASE("dtw_distance pointer overload with band matches vector overload",
          "[dtw_api][dtw_distance]")
{
  const std::vector<double> x{1, 2, 3};
  const std::vector<double> y{3, 4, 5, 6, 7};
  const int band = 100;

  const double vec_result = dtw_distance(x, y, band);
  const double ptr_result = dtw_distance(x.data(), x.size(),
                                         y.data(), y.size(), band);

  REQUIRE_THAT(ptr_result, WithinAbs(vec_result, 1e-15));
}

// ----- dtw_runtime (runtime-dispatched) ---------------------------------

TEST_CASE("dtw_runtime with None constraint matches dtwFull_L",
          "[dtw_api][dtw_runtime]")
{
  const std::vector<double> x{1, 2, 3};
  const std::vector<double> y{3, 4, 5, 6, 7};

  DTWOptions opts;
  opts.constraint = ConstraintType::None;

  const double result = dtw_runtime(x.data(), x.size(),
                                    y.data(), y.size(), opts);
  const double ref = dtwc::dtwFull_L<double>(x, y);

  REQUIRE_THAT(result, WithinAbs(ref, 1e-15));
}

TEST_CASE("dtw_runtime with SakoeChibaBand matches dtwBanded",
          "[dtw_api][dtw_runtime]")
{
  const std::vector<double> x{1, 2, 3};
  const std::vector<double> y{3, 4, 5, 6, 7};
  const int band = 100;

  DTWOptions opts;
  opts.constraint = ConstraintType::SakoeChibaBand;
  opts.band = band;

  const double result = dtw_runtime(x.data(), x.size(),
                                    y.data(), y.size(), opts);
  const double ref = dtwc::dtwBanded<double>(x, y, band);

  REQUIRE_THAT(result, WithinAbs(ref, 1e-15));
}

// ----- SquaredL2 metric tests -------------------------------------------

TEST_CASE("dtw_distance with SquaredL2 metric",
          "[dtw_api][dtw_distance][SquaredL2]")
{
  const std::vector<double> x{1, 2, 3};
  const std::vector<double> y{3, 4, 5, 6, 7};
  constexpr double sq_ground_truth = 35.0;

  const double result = dtw_distance(x, y, -1, MetricType::SquaredL2);
  REQUIRE_THAT(result, WithinAbs(sq_ground_truth, 1e-15));

  // Must differ from L1
  const double l1_result = dtw_distance(x, y, -1, MetricType::L1);
  REQUIRE_THAT(l1_result, WithinAbs(13.0, 1e-15));
  REQUIRE(result != l1_result);
}

TEST_CASE("dtw_runtime with SquaredL2 metric matches dtwFull_L SquaredL2",
          "[dtw_api][dtw_runtime][SquaredL2]")
{
  const std::vector<double> x{1, 2, 3};
  const std::vector<double> y{3, 4, 5, 6, 7};

  DTWOptions opts;
  opts.constraint = ConstraintType::None;
  opts.metric = MetricType::SquaredL2;

  const double result = dtw_runtime(x.data(), x.size(),
                                    y.data(), y.size(), opts);
  const double ref = dtwc::dtwFull_L<double>(x, y, -1.0, MetricType::SquaredL2);

  REQUIRE_THAT(result, WithinAbs(ref, 1e-15));
  REQUIRE_THAT(result, WithinAbs(35.0, 1e-15));
}

TEST_CASE("dtw_runtime with SakoeChibaBand and SquaredL2",
          "[dtw_api][dtw_runtime][SquaredL2]")
{
  const std::vector<double> x{1, 2, 3};
  const std::vector<double> y{3, 4, 5, 6, 7};
  const int band = 100;

  DTWOptions opts;
  opts.constraint = ConstraintType::SakoeChibaBand;
  opts.band = band;
  opts.metric = MetricType::SquaredL2;

  const double result = dtw_runtime(x.data(), x.size(),
                                    y.data(), y.size(), opts);
  const double ref = dtwc::dtwBanded<double>(x, y, band, -1.0, MetricType::SquaredL2);

  REQUIRE_THAT(result, WithinAbs(ref, 1e-15));
}

// ----- Symmetry checks --------------------------------------------------

TEST_CASE("dtw_distance is symmetric",
          "[dtw_api][dtw_distance]")
{
  const std::vector<double> x{1, 2, 3};
  const std::vector<double> y{3, 4, 5, 6, 7};

  REQUIRE_THAT(dtw_distance(x, y),
               WithinAbs(dtw_distance(y, x), 1e-15));
}

// ----- Zero distance for identical series --------------------------------

TEST_CASE("dtw_distance returns 0 for identical series",
          "[dtw_api][dtw_distance]")
{
  const std::vector<double> x{1, 2, 3, 4, 5};

  REQUIRE_THAT(dtw_distance(x, x), WithinAbs(0.0, 1e-15));
}

// ===========================================================================
//  Task 0.6 regressions
// ===========================================================================

// ----- (a) SoftDTW dispatch must NOT fall through to Standard-L1 ------------
//
// BUG (dtw.cpp): DTWVariant::SoftDTW used `[[fallthrough]]` to the Standard
// case, so dtw_runtime silently returned a Standard-L1 distance for a SoftDTW
// request (a wrong number), and the gamma>0 precondition was skipped.
//
// Why the UNFIXED code fails these tests:
//   * it returns the Standard-L1 distance, which is strictly larger than the
//     real Soft-DTW value (softmin < min at every interior cell for gamma>0),
//     so it fails BOTH `WithinAbs(ref_soft, ...)` and `!= standard_l1`;
//   * it never reaches soft_dtw()'s gamma guard, so it does NOT throw for
//     gamma<=0, failing REQUIRE_THROWS.
TEST_CASE("dtw_runtime SoftDTW computes Soft-DTW, not Standard-L1",
          "[dtw_api][dtw_runtime][softdtw]")
{
  const std::vector<double> x{1, 2, 3};
  const std::vector<double> y{3, 4, 5, 6, 7};
  const double gamma = 1.0;

  DTWOptions opts;
  opts.constraint = ConstraintType::None;
  opts.variant_params.variant = DTWVariant::SoftDTW;
  opts.variant_params.sdtw_gamma = gamma;

  const double result =
    dtw_runtime(x.data(), x.size(), y.data(), y.size(), opts);

  // Oracle: the dedicated, independently cross-validated soft_dtw() kernel.
  const double ref_soft = dtwc::soft_dtw<double>(x, y, gamma);
  // The buggy fallthrough returned this Standard-L1 distance instead (== 13.0).
  const double standard_l1 = dtwc::dtwFull_L<double>(x, y);

  REQUIRE_THAT(result, WithinAbs(ref_soft, 1e-12));  // must equal real Soft-DTW
  REQUIRE(result != standard_l1);                    // must NOT be Standard-L1
}

TEST_CASE("dtw_runtime SoftDTW throws on gamma <= 0",
          "[dtw_api][dtw_runtime][softdtw]")
{
  const std::vector<double> x{1, 2, 3};
  const std::vector<double> y{3, 4, 5, 6, 7};

  DTWOptions opts;
  opts.variant_params.variant = DTWVariant::SoftDTW;

  opts.variant_params.sdtw_gamma = 0.0;
  REQUIRE_THROWS(dtw_runtime(x.data(), x.size(), y.data(), y.size(), opts));

  opts.variant_params.sdtw_gamma = -1.0;
  REQUIRE_THROWS(dtw_runtime(x.data(), x.size(), y.data(), y.size(), opts));
}

// ----- (b) Multivariate L2 must be Euclidean, not L1 -----------------------
//
// LIVE PATH: this drives dtwc::dtwBanded_mv — the *exact* public entry the
// library dispatches to for multivariate DTW (dtw_dispatch.cpp:113/132 ->
// dtwBanded_mv -> warping.hpp detail::dispatch_mv_metric). The prior wave's
// test validated the now-deleted core::dispatch_mv_metric — a DEAD function
// with zero library call sites — so it never noticed that the LIVE dispatcher
// still aliased MetricType::L2 to the multivariate L1 (Manhattan) functor.
//
// BUG (warping.hpp detail::dispatch_mv_metric): `case L2:` fell through to
// MVL1Dist, so a multivariate "L2" request computed sum-of-|diff| instead of a
// Euclidean per-step cost.
//
// Hand-computed 2-channel example; values REGISTERED BEFORE the run:
//   x = 4 timesteps of (0, 0)   (interleaved: 8 zeros)
//   y = 4 timesteps of (3, 4)   (interleaved: {3,4,3,4,3,4,3,4})
//   band = 1, ndim = 2
//   Per-step pointwise cost is CONSTANT across every (i, j) cell:
//     L1        = |0-3| + |0-4|     = 7
//     L2 (true) = sqrt(3^2 + 4^2)   = 5
//     SquaredL2 = 3^2 + 4^2         = 25
//   Equal-length-4 series with constant cell cost c => the banded (band=1) DP
//   walks the diagonal, so DTW = 4 * c:
//     L1        DTW = 4 * 7  = 28
//     L2 (true) DTW = 4 * 5  = 20   <-- fix target
//     SquaredL2 DTW = 4 * 25 = 100
// On the UNFIXED dispatcher the L2 branch returns 28 (L1), failing the 20.0
// assertion and the `!=` check below.
TEST_CASE("dtwBanded_mv L2 is Euclidean, distinct from L1 (live path)",
          "[dtw_api][warping][mv][L2]")
{
  // interleaved layout: x[t * ndim + d]
  const std::vector<double> x(8, 0.0);                     // 4 steps of (0, 0)
  const std::vector<double> y{3, 4, 3, 4, 3, 4, 3, 4};     // 4 steps of (3, 4)
  const std::size_t ndim = 2;
  const int band = 1;

  const auto mv = [&](MetricType m) {
    return dtwc::dtwBanded_mv<double>(
      x.data(), 4, y.data(), 4, ndim, band, -1.0, m);
  };

  const double l1 = mv(MetricType::L1);
  const double l2 = mv(MetricType::L2);
  const double sq = mv(MetricType::SquaredL2);

  REQUIRE_THAT(l1, WithinAbs(28.0, 1e-12));
  REQUIRE_THAT(sq, WithinAbs(100.0, 1e-12));
  REQUIRE_THAT(l2, WithinAbs(20.0, 1e-12));   // Euclidean, was 28.0 (L1) pre-fix
  REQUIRE(l2 != l1);                          // L2 must NOT alias L1

  // Same fix must be visible through the unbanded live path (dtwFull_L_mv);
  // constant-cost equal-length DTW is 4 * c there too.
  const double l2_full =
    dtwc::dtwFull_L_mv<double>(x.data(), 4, y.data(), 4, ndim, -1.0, MetricType::L2);
  REQUIRE_THAT(l2_full, WithinAbs(20.0, 1e-12));

  // Direct check on the functor the live dispatcher now selects for L2
  // (migrated from the deleted core::MVL2Dist). One step (0,0) vs (3,4):
  //   sqrt(3^2 + 4^2) = 5.
  const double a[2] = {0.0, 0.0};
  const double b[2] = {3.0, 4.0};
  REQUIRE_THAT(dtwc::detail::MVL2Dist{}(a, b, std::size_t{2}), WithinAbs(5.0, 1e-12));
}
