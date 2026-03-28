/**
 * @file unit_test_dtw_api.cpp
 * @brief Unit tests for the unified DTW API (dtwc::core).
 *
 * @details Verifies that the new dtw_distance / dtw_runtime wrappers
 *          produce identical results to the underlying warping.hpp
 *          functions they delegate to.
 *
 * @date 28 Mar 2026
 */

#include <core/dtw.hpp>
#include <core/lower_bounds.hpp>
#include <warping.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc::core;

// ----- Compile-time lower-bound compatibility checks --------------------

static_assert(lb_keogh_valid<L1Metric>,
              "LB_Keogh must be valid for L1Metric");
static_assert(!lb_keogh_valid<L2Metric>,
              "LB_Keogh must NOT be valid for L2Metric");
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
  opts.band_width = band;

  const double result = dtw_runtime(x.data(), x.size(),
                                    y.data(), y.size(), opts);
  const double ref = dtwc::dtwBanded<double>(x, y, band);

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
