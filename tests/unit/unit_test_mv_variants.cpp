/**
 * @file unit_test_mv_variants.cpp
 * @brief Unit tests for multivariate WDTW, ADTW, and DDTW functions.
 * @author Volkan Kumtepeli
 *
 * Tests cover:
 *   - wdtwFull_mv / wdtwBanded_mv: ndim=1 parity, ndim=2 identity/symmetry/non-negativity
 *   - adtwFull_L_mv / adtwBanded_mv: ndim=1 parity, ndim=2 identity/symmetry/penalty effect
 *   - Problem integration: WDTW, ADTW, DDTW with ndim=2 data
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>
#include <cmath>
#include <limits>

using Catch::Matchers::WithinAbs;

// =========================================================================
//  WDTW MV — wdtwFull_mv
// =========================================================================

TEST_CASE("WDTW MV: ndim=1 matches scalar wdtwFull", "[mv][wdtw]")
{
  std::vector<double> x = {1, 3, 4, 2, 5};
  std::vector<double> y = {2, 4, 3, 5, 1};
  const int max_dev = static_cast<int>(std::max(x.size(), y.size())) - 1;
  auto w = dtwc::wdtw_weights<double>(max_dev, 0.05);
  double d_scalar = dtwc::wdtwFull(x, y, w);
  double d_mv = dtwc::wdtwFull_mv(x.data(), 5, y.data(), 5, 1, 0.05);
  REQUIRE_THAT(d_mv, WithinAbs(d_scalar, 1e-10));
}

TEST_CASE("WDTW MV: ndim=2 same pointer returns 0", "[mv][wdtw]")
{
  double x[] = {1, 2, 3, 4, 5, 6};  // 3 timesteps, 2 features
  REQUIRE(dtwc::wdtwFull_mv(x, 3, x, 3, 2) == 0.0);
}

TEST_CASE("WDTW MV: ndim=2 identical arrays = 0", "[mv][wdtw]")
{
  double x[] = {1, 2, 3, 4, 5, 6};
  double y[] = {1, 2, 3, 4, 5, 6};
  // Different pointers, same content — not 0 via pointer check, but distance should be 0
  // (all pointwise distances are 0, so weighted sum is 0)
  REQUIRE_THAT(dtwc::wdtwFull_mv(x, 3, y, 3, 2), WithinAbs(0.0, 1e-10));
}

TEST_CASE("WDTW MV: ndim=2 symmetry", "[mv][wdtw]")
{
  double x[] = {1, 2, 3, 4, 5, 6};
  double y[] = {6, 5, 4, 3, 2, 1};
  double d1 = dtwc::wdtwFull_mv(x, 3, y, 3, 2);
  double d2 = dtwc::wdtwFull_mv(y, 3, x, 3, 2);
  REQUIRE_THAT(d1, WithinAbs(d2, 1e-10));
}

TEST_CASE("WDTW MV: ndim=2 non-negative distance", "[mv][wdtw]")
{
  double x[] = {1, 2, 3, 4};
  double y[] = {5, 6, 7, 8};
  REQUIRE(dtwc::wdtwFull_mv(x, 2, y, 2, 2) >= 0.0);
}

TEST_CASE("WDTW MV: empty series returns max", "[mv][wdtw]")
{
  double x[] = {1, 2, 3, 4};
  constexpr double maxVal = std::numeric_limits<double>::max();
  REQUIRE(dtwc::wdtwFull_mv(x, 2, static_cast<double *>(nullptr), 0, 2) == maxVal);
  REQUIRE(dtwc::wdtwFull_mv(static_cast<double *>(nullptr), 0, x, 2, 2) == maxVal);
}

TEST_CASE("WDTW MV: different lengths (ndim=2)", "[mv][wdtw]")
{
  double x[] = {0, 0, 1, 1, 2, 2};  // 3 timesteps
  double y[] = {0, 0, 2, 2};         // 2 timesteps
  double d = dtwc::wdtwFull_mv(x, 3, y, 2, 2);
  REQUIRE(d >= 0.0);
  REQUIRE(d < std::numeric_limits<double>::max());
}

// =========================================================================
//  WDTW MV — wdtwBanded_mv
// =========================================================================

TEST_CASE("WDTW MV banded: negative band delegates to full", "[mv][wdtw]")
{
  double x[] = {1, 2, 3, 4, 5, 6};
  double y[] = {6, 5, 4, 3, 2, 1};
  double d_full = dtwc::wdtwFull_mv(x, 3, y, 3, 2);
  double d_band = dtwc::wdtwBanded_mv(x, 3, y, 3, 2, -1);
  REQUIRE_THAT(d_full, WithinAbs(d_band, 1e-10));
}

TEST_CASE("WDTW MV banded: ndim=1 matches scalar wdtwBanded", "[mv][wdtw]")
{
  std::vector<double> x = {1, 3, 4, 2, 5};
  std::vector<double> y = {2, 4, 3, 5, 1};
  double d_scalar = dtwc::wdtwBanded(x, y, 2, 0.05);
  double d_mv = dtwc::wdtwBanded_mv(x.data(), 5, y.data(), 5, 1, 2, 0.05);
  REQUIRE_THAT(d_mv, WithinAbs(d_scalar, 1e-10));
}

// =========================================================================
//  ADTW MV — adtwFull_L_mv
// =========================================================================

TEST_CASE("ADTW MV: ndim=1 matches scalar adtwFull_L", "[mv][adtw]")
{
  std::vector<double> x = {1, 3, 4, 2, 5};
  std::vector<double> y = {2, 4, 3, 5, 1};
  double d_scalar = dtwc::adtwFull_L(x, y, 1.0);
  double d_mv = dtwc::adtwFull_L_mv(x.data(), 5, y.data(), 5, 1, 1.0);
  REQUIRE_THAT(d_mv, WithinAbs(d_scalar, 1e-10));
}

TEST_CASE("ADTW MV: ndim=2 same pointer returns 0", "[mv][adtw]")
{
  double x[] = {1, 2, 3, 4, 5, 6};
  REQUIRE(dtwc::adtwFull_L_mv(x, 3, x, 3, 2) == 0.0);
}

TEST_CASE("ADTW MV: ndim=2 identical arrays = 0", "[mv][adtw]")
{
  double x[] = {1, 2, 3, 4, 5, 6};
  double y[] = {1, 2, 3, 4, 5, 6};
  // All pointwise distances are 0, so total = 0 regardless of penalty
  REQUIRE_THAT(dtwc::adtwFull_L_mv(x, 3, y, 3, 2, 1.0), WithinAbs(0.0, 1e-10));
}

TEST_CASE("ADTW MV: ndim=2 symmetry", "[mv][adtw]")
{
  double x[] = {1, 2, 3, 4, 5, 6};
  double y[] = {6, 5, 4, 3, 2, 1};
  double d1 = dtwc::adtwFull_L_mv(x, 3, y, 3, 2);
  double d2 = dtwc::adtwFull_L_mv(y, 3, x, 3, 2);
  REQUIRE_THAT(d1, WithinAbs(d2, 1e-10));
}

TEST_CASE("ADTW MV: ndim=2 non-negative distance", "[mv][adtw]")
{
  double x[] = {1, 2, 3, 4};
  double y[] = {5, 6, 7, 8};
  REQUIRE(dtwc::adtwFull_L_mv(x, 2, y, 2, 2) >= 0.0);
}

TEST_CASE("ADTW MV: penalty increases distance for unequal-length series", "[mv][adtw]")
{
  // Unequal lengths force non-diagonal steps, so penalty has a direct effect
  double x[] = {0, 0, 1, 1, 2, 2};  // 3 timesteps
  double y[] = {0, 0, 2, 2};         // 2 timesteps
  double d_low = dtwc::adtwFull_L_mv(x, 3, y, 2, 2, 0.0);
  double d_high = dtwc::adtwFull_L_mv(x, 3, y, 2, 2, 10.0);
  REQUIRE(d_high >= d_low);
}

TEST_CASE("ADTW MV: empty series returns max", "[mv][adtw]")
{
  double x[] = {1, 2, 3, 4};
  constexpr double maxVal = std::numeric_limits<double>::max();
  REQUIRE(dtwc::adtwFull_L_mv(x, 2, static_cast<double *>(nullptr), 0, 2) == maxVal);
  REQUIRE(dtwc::adtwFull_L_mv(static_cast<double *>(nullptr), 0, x, 2, 2) == maxVal);
}

// =========================================================================
//  ADTW MV — adtwBanded_mv
// =========================================================================

TEST_CASE("ADTW MV banded: negative band delegates to full", "[mv][adtw]")
{
  double x[] = {1, 2, 3, 4, 5, 6};
  double y[] = {6, 5, 4, 3, 2, 1};
  double d_full = dtwc::adtwFull_L_mv(x, 3, y, 3, 2);
  double d_band = dtwc::adtwBanded_mv(x, 3, y, 3, 2, -1);
  REQUIRE_THAT(d_full, WithinAbs(d_band, 1e-10));
}

TEST_CASE("ADTW MV banded: ndim=1 matches scalar adtwBanded", "[mv][adtw]")
{
  std::vector<double> x = {1, 3, 4, 2, 5};
  std::vector<double> y = {2, 4, 3, 5, 1};
  double d_scalar = dtwc::adtwBanded(x, y, 2, 1.0);
  double d_mv = dtwc::adtwBanded_mv(x.data(), 5, y.data(), 5, 1, 2, 1.0);
  REQUIRE_THAT(d_mv, WithinAbs(d_scalar, 1e-10));
}

// =========================================================================
//  Problem integration — WDTW MV
// =========================================================================

TEST_CASE("Problem: WDTW MV via set_variant", "[mv][wdtw][problem]")
{
  dtwc::Data data;
  data.ndim = 2;
  data.p_vec = {
    {0, 0, 1, 1, 2, 2},
    {10, 10, 11, 11, 12, 12}
  };
  data.p_names = {"a", "b"};

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.set_variant(dtwc::core::DTWVariant::WDTW);
  prob.verbose = false;
  prob.fillDistanceMatrix();
  REQUIRE(prob.distByInd(0, 1) > 0.0);
}

TEST_CASE("Problem: WDTW MV identical series = 0", "[mv][wdtw][problem]")
{
  dtwc::Data data;
  data.ndim = 2;
  data.p_vec = {
    {0, 0, 1, 1, 2, 2},
    {0, 0, 1, 1, 2, 2}
  };
  data.p_names = {"a", "b"};

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.set_variant(dtwc::core::DTWVariant::WDTW);
  prob.verbose = false;
  prob.fillDistanceMatrix();
  REQUIRE_THAT(prob.distByInd(0, 1), WithinAbs(0.0, 1e-10));
}

// =========================================================================
//  Problem integration — ADTW MV
// =========================================================================

TEST_CASE("Problem: ADTW MV via set_variant", "[mv][adtw][problem]")
{
  dtwc::Data data;
  data.ndim = 2;
  data.p_vec = {
    {0, 0, 1, 1, 2, 2},
    {10, 10, 11, 11, 12, 12}
  };
  data.p_names = {"a", "b"};

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.set_variant(dtwc::core::DTWVariant::ADTW);
  prob.verbose = false;
  prob.fillDistanceMatrix();
  REQUIRE(prob.distByInd(0, 1) > 0.0);
}

TEST_CASE("Problem: ADTW MV identical series = 0", "[mv][adtw][problem]")
{
  dtwc::Data data;
  data.ndim = 2;
  data.p_vec = {
    {1, 2, 3, 4, 5, 6},
    {1, 2, 3, 4, 5, 6}
  };
  data.p_names = {"a", "b"};

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.set_variant(dtwc::core::DTWVariant::ADTW);
  prob.verbose = false;
  prob.fillDistanceMatrix();
  REQUIRE_THAT(prob.distByInd(0, 1), WithinAbs(0.0, 1e-10));
}

// =========================================================================
//  Problem integration — DDTW MV
// =========================================================================

TEST_CASE("Problem: DDTW MV via set_variant", "[mv][ddtw][problem]")
{
  dtwc::Data data;
  data.ndim = 2;
  data.p_vec = {
    {0, 0, 1, 1, 2, 2, 3, 3},   // linear ramp in both channels
    {0, 0, 1, 1, 2, 2, 3, 3},   // identical
    {10, 10, 11, 11, 12, 12, 13, 13}  // same shape, different offset
  };
  data.p_names = {"a", "b", "c"};

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.set_variant(dtwc::core::DTWVariant::DDTW);
  prob.verbose = false;
  prob.fillDistanceMatrix();

  // Identical series must yield 0
  REQUIRE_THAT(prob.distByInd(0, 1), WithinAbs(0.0, 1e-10));
  // Different offset but same shape: derivative is identical => DDTW distance is 0
  REQUIRE_THAT(prob.distByInd(0, 2), WithinAbs(0.0, 1e-10));
}

TEST_CASE("Problem: DDTW MV different shape > 0", "[mv][ddtw][problem]")
{
  dtwc::Data data;
  data.ndim = 2;
  data.p_vec = {
    {0, 0, 1, 1, 2, 2, 3, 3},      // linear ramp
    {0, 0, 2, 2, 0, 0, 2, 2}       // oscillating — different derivative
  };
  data.p_names = {"a", "b"};

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.set_variant(dtwc::core::DTWVariant::DDTW);
  prob.verbose = false;
  prob.fillDistanceMatrix();

  REQUIRE(prob.distByInd(0, 1) > 0.0);
}

// =========================================================================
//  Phase 1 unified-kernel regressions: banded MV is now a first-class path.
//  Previously adtwBanded_mv / wdtwBanded_mv silently fell back to the
//  unbanded MV variant (documented TODO). With the unified kernel, band
//  restrictions are honoured and must produce different results from the
//  unbanded MV when the band is tight enough to constrain the warping path.
// =========================================================================

TEST_CASE("ADTW MV banded actually restricts the warping path", "[mv][adtw][banded]")
{
  // Shifted-peak pattern: x has a peak at step 2, y has the same peak at step 8.
  // Unbanded DTW warps heavily to align the peaks; band=1 forbids that warp.
  // Previously adtwBanded_mv fell back to the unbanded path (documented TODO),
  // so the two results were identical. With the unified kernel, the banded
  // result is genuinely higher.
  std::vector<double> x = {0,0, 0,0, 9,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0};
  std::vector<double> y = {0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 9,0, 0,0};
  const double penalty = 0.5;

  const double d_unbanded = dtwc::adtwFull_L_mv<double>(x.data(), 10, y.data(), 10, 2, penalty);
  const double d_banded1  = dtwc::adtwBanded_mv<double>(x.data(), 10, y.data(), 10, 2, 1, penalty);

  INFO("d_unbanded=" << d_unbanded << " d_banded1=" << d_banded1);
  REQUIRE(d_banded1 > d_unbanded); // previously equal (fallback); now distinct and strictly greater.
}

TEST_CASE("WDTW MV banded actually restricts the warping path", "[mv][wdtw][banded]")
{
  std::vector<double> x = {0,0, 0,0, 9,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0};
  std::vector<double> y = {0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 0,0, 9,0, 0,0};
  const double g = 0.05;

  const double d_unbanded = dtwc::wdtwFull_mv<double>(x.data(), 10, y.data(), 10, 2, g);
  const double d_banded1  = dtwc::wdtwBanded_mv<double>(x.data(), 10, y.data(), 10, 2, 1, g);

  INFO("d_unbanded=" << d_unbanded << " d_banded1=" << d_banded1);
  REQUIRE(d_banded1 > d_unbanded);
}

// =========================================================================
//  dtw_runtime() variant dispatch regression — previously silently dropped
//  opts.variant_params.variant. ADTW with a non-zero penalty must differ
//  from Standard DTW.
// =========================================================================

TEST_CASE("dtw_runtime honours DTWVariant::ADTW", "[dtw_runtime][adtw]")
{
  std::vector<double> x = {1, 3, 4, 2, 5, 4, 3, 2, 1};
  std::vector<double> y = {2, 4, 3, 5, 1, 2, 4, 3, 2};

  dtwc::core::DTWOptions opts_std;
  opts_std.variant_params.variant = dtwc::core::DTWVariant::Standard;
  const double d_std = dtwc::core::dtw_runtime(x.data(), x.size(), y.data(), y.size(), opts_std);

  dtwc::core::DTWOptions opts_adtw;
  opts_adtw.variant_params.variant = dtwc::core::DTWVariant::ADTW;
  opts_adtw.variant_params.adtw_penalty = 2.0;
  const double d_adtw = dtwc::core::dtw_runtime(x.data(), x.size(), y.data(), y.size(), opts_adtw);

  INFO("d_std=" << d_std << " d_adtw=" << d_adtw);
  REQUIRE(d_adtw > d_std); // ADTW with penalty > 0 must be >= Standard DTW.
}

TEST_CASE("dtw_runtime honours DTWVariant::WDTW", "[dtw_runtime][wdtw]")
{
  std::vector<double> x = {1, 3, 4, 2, 5, 4, 3, 2, 1};
  std::vector<double> y = {2, 4, 3, 5, 1, 2, 4, 3, 2};

  dtwc::core::DTWOptions opts_std;
  const double d_std = dtwc::core::dtw_runtime(x.data(), x.size(), y.data(), y.size(), opts_std);

  dtwc::core::DTWOptions opts_wdtw;
  opts_wdtw.variant_params.variant = dtwc::core::DTWVariant::WDTW;
  opts_wdtw.variant_params.wdtw_g = 0.05;
  const double d_wdtw = dtwc::core::dtw_runtime(x.data(), x.size(), y.data(), y.size(), opts_wdtw);

  INFO("d_std=" << d_std << " d_wdtw=" << d_wdtw);
  REQUIRE(d_wdtw != d_std); // WDTW applies per-cell weights -> different result.
}
