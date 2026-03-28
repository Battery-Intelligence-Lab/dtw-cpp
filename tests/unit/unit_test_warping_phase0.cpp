/**
 * @file unit_test_warping_phase0.cpp
 * @brief Phase 0 bug-detection tests for DTW warping functions.
 *
 * @details These tests verify known bugs and properties of the DTW functions.
 * Some tests are EXPECTED TO FAIL on unmodified code -- they prove bugs exist.
 *
 * EXPECTED FAILURES on unmodified code:
 *   - "[Phase0] dtwBanded float default" / "deduced template matches double"
 *     dtwBanded has `template <typename data_t = float>`, so passing
 *     `std::vector<double>` deduces `data_t = double`, but the default
 *     template parameter is float. This test checks that explicit and
 *     deduced calls agree, and that banded matches full DTW for wide bands.
 *
 * @date 28 Mar 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

// ---------------------------------------------------------------------------
// Test 2: dtwBanded float default template bug
// ---------------------------------------------------------------------------
TEST_CASE("[Phase0] dtwBanded float default", "[Phase0][warping][dtwBanded]")
{
  using data_t = double;
  std::vector<data_t> x{ 1.0, 2.0, 3.0 };
  std::vector<data_t> y{ 3.0, 4.0, 5.0, 6.0, 7.0 };
  constexpr int wide_band = 100;

  // Ground truth from dtwFull (known correct for these inputs).
  const double full_result = dtwFull<data_t>(x, y);

  SECTION("banded with wide band matches dtwFull")
  {
    // With a band wider than both series, dtwBanded must equal dtwFull.
    const double banded_result = dtwBanded<data_t>(x, y, wide_band);
    REQUIRE_THAT(banded_result, WithinAbs(full_result, 1e-15));
  }

  SECTION("deduced template matches double")
  {
    // When we pass std::vector<double>, the compiler deduces data_t = double.
    // The explicit call dtwBanded<double>(...) should give the same result.
    // If the default template param were float and somehow affected deduction,
    // this would catch it.
    const double deduced_result = dtwBanded(x, y, wide_band);
    const double explicit_result = dtwBanded<double>(x, y, wide_band);
    REQUIRE_THAT(deduced_result, WithinAbs(explicit_result, 1e-15));
    // Both should also match dtwFull.
    REQUIRE_THAT(deduced_result, WithinAbs(full_result, 1e-15));
  }
}

// ---------------------------------------------------------------------------
// Test 3: DTW metric properties (symmetry, identity, non-negativity)
// ---------------------------------------------------------------------------
TEST_CASE("[Phase0] DTW property tests", "[Phase0][warping][properties]")
{
  using data_t = double;
  std::vector<data_t> x{ 1.0, 2.0, 3.0 };
  std::vector<data_t> y{ 3.0, 4.0, 5.0, 6.0, 7.0 };

  SECTION("dtwFull properties")
  {
    const double dxy = dtwFull<data_t>(x, y);
    const double dyx = dtwFull<data_t>(y, x);
    const double dxx = dtwFull<data_t>(x, x);
    const double dyy = dtwFull<data_t>(y, y);

    // Symmetry: d(x,y) == d(y,x)
    REQUIRE_THAT(dxy, WithinAbs(dyx, 1e-15));

    // Identity: d(x,x) == 0
    REQUIRE_THAT(dxx, WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(dyy, WithinAbs(0.0, 1e-15));

    // Non-negativity: d(x,y) >= 0
    REQUIRE(dxy >= 0.0);
  }

  SECTION("dtwFull_L properties")
  {
    const double dxy = dtwFull_L<data_t>(x, y);
    const double dyx = dtwFull_L<data_t>(y, x);
    const double dxx = dtwFull_L<data_t>(x, x);
    const double dyy = dtwFull_L<data_t>(y, y);

    // Symmetry
    REQUIRE_THAT(dxy, WithinAbs(dyx, 1e-15));

    // Identity
    REQUIRE_THAT(dxx, WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(dyy, WithinAbs(0.0, 1e-15));

    // Non-negativity
    REQUIRE(dxy >= 0.0);
  }

  SECTION("dtwBanded properties")
  {
    constexpr int wide_band = 100;
    const double dxy = dtwBanded<data_t>(x, y, wide_band);
    const double dyx = dtwBanded<data_t>(y, x, wide_band);
    const double dxx = dtwBanded<data_t>(x, x, wide_band);
    const double dyy = dtwBanded<data_t>(y, y, wide_band);

    // Symmetry
    REQUIRE_THAT(dxy, WithinAbs(dyx, 1e-15));

    // Identity
    REQUIRE_THAT(dxx, WithinAbs(0.0, 1e-15));
    REQUIRE_THAT(dyy, WithinAbs(0.0, 1e-15));

    // Non-negativity
    REQUIRE(dxy >= 0.0);
  }
}
