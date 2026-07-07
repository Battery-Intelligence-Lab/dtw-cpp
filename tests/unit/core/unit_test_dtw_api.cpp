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
// BUG (dtw_cost.hpp): core::dispatch_mv_metric mapped MetricType::L2 to the
// multivariate L1 functor, so a multivariate "L2" request computed a Manhattan
// (sum-of-|diff|) cost. Hand-computed 2-point / 3-channel example; expected
// values REGISTERED BEFORE the fix:
//   a = (0,0,0), b = (1,2,2)
//   L1        = |0-1| + |0-2| + |0-2|      = 5
//   SquaredL2 = 1 + 4 + 4                   = 9
//   L2 (true) = sqrt(1 + 4 + 4) = sqrt(9)   = 3   <-- fix target
// On the UNFIXED dispatcher the L2 branch returns 5 (L1), failing the 3.0
// assertion and the `!=` check below.
TEST_CASE("core::dispatch_mv_metric L2 is Euclidean, distinct from L1",
          "[dtw_api][dtw_cost][mv][L2]")
{
  const double a[3] = {0.0, 0.0, 0.0};
  const double b[3] = {1.0, 2.0, 2.0};

  const auto eval = [&](MetricType m) {
    return dispatch_mv_metric(
      m, [&](auto dist) { return dist(a, b, std::size_t{3}); });
  };

  REQUIRE_THAT(eval(MetricType::L1), WithinAbs(5.0, 1e-12));
  REQUIRE_THAT(eval(MetricType::SquaredL2), WithinAbs(9.0, 1e-12));
  REQUIRE_THAT(eval(MetricType::L2), WithinAbs(3.0, 1e-12));  // was 5.0 (L1) pre-fix
  REQUIRE(eval(MetricType::L2) != eval(MetricType::L1));      // L2 must not alias L1

  // Direct functor check on the same hand-computed example.
  REQUIRE_THAT(MVL2Dist{}(a, b, std::size_t{3}), WithinAbs(3.0, 1e-12));
}
