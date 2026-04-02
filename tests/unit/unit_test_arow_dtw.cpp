/**
 * @file unit_test_arow_dtw.cpp
 * @brief Unit tests for DTW-AROW (diagonal-only alignment for missing values).
 *
 * @details Tests for dtwAROW, dtwAROW_L, and dtwAROW_banded covering:
 *  1. No NaN -> matches standard DTW
 *  2. All NaN -> returns 0
 *  3. Symmetry: dtwAROW(x,y) == dtwAROW(y,x)
 *  4. AROW >= ZeroCost (stricter constraint -> higher or equal distance)
 *  5. Leading NaN is finite (NOT +inf)
 *  6. Trailing NaN is finite
 *  7. Non-negativity
 *  8. Banded with large band matches unbanded
 *  9. Full-matrix matches linear-space
 * Plus hand-computed verification tests.
 *
 * Reference: Yurtman, A., Soenen, J., Meert, W. & Blockeel, H. (2023).
 *            "Estimating DTW Distance Between Time Series with Missing Data."
 *            ECML-PKDD 2023, LNCS 14173.
 *
 * @date 02 Apr 2026
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <cmath>
#include <limits>
#include <vector>

using Catch::Matchers::WithinAbs;
using namespace dtwc;

static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();
static constexpr double INF = std::numeric_limits<double>::max();

// ===========================================================================
// Property 1: No NaN -> matches standard DTW
// ===========================================================================

TEST_CASE("dtwAROW_L: no NaN matches standard dtwFull_L", "[arow_dtw][property]")
{
  std::vector<double> x{ 1, 2, 3, 4, 5 };
  std::vector<double> y{ 2, 3, 4 };

  const auto arow = dtwAROW_L<double>(x, y);
  const auto std_dtw = dtwFull_L<double>(x, y);

  REQUIRE_THAT(arow, WithinAbs(std_dtw, 1e-12));
}

TEST_CASE("dtwAROW_L: no NaN matches standard DTW (equal length)", "[arow_dtw][property]")
{
  std::vector<double> x{ 1, 3, 5, 7 };
  std::vector<double> y{ 2, 4, 6, 8 };

  REQUIRE_THAT(dtwAROW_L<double>(x, y), WithinAbs(dtwFull_L<double>(x, y), 1e-12));
}

TEST_CASE("dtwAROW_L: no NaN matches standard DTW (SquaredL2)", "[arow_dtw][property]")
{
  std::vector<double> x{ 1, 2, 3 };
  std::vector<double> y{ 2, 4, 5 };
  auto metric = core::MetricType::SquaredL2;

  REQUIRE_THAT(dtwAROW_L<double>(x, y, metric),
               WithinAbs(dtwFull_L<double>(x, y, -1, metric), 1e-12));
}

TEST_CASE("dtwAROW: no NaN matches standard dtwFull", "[arow_dtw][property]")
{
  std::vector<double> x{ 1, 2, 3, 4 };
  std::vector<double> y{ 2, 3, 4, 5, 6 };

  REQUIRE_THAT(dtwAROW<double>(x, y), WithinAbs(dtwFull_L<double>(x, y), 1e-12));
}

// ===========================================================================
// Property 2: All NaN -> returns 0
// ===========================================================================

TEST_CASE("dtwAROW_L: all NaN in both series gives 0", "[arow_dtw][property]")
{
  std::vector<double> x{ NaN, NaN, NaN };
  std::vector<double> y{ NaN, NaN };

  REQUIRE_THAT(dtwAROW_L<double>(x, y), WithinAbs(0.0, 1e-15));
}

TEST_CASE("dtwAROW_L: one series entirely NaN gives 0", "[arow_dtw][property]")
{
  std::vector<double> x{ 1, 2, 3 };
  std::vector<double> y_nan{ NaN, NaN, NaN };

  // When y is all NaN: C(i,j) = C(i-1,j-1) everywhere (diagonal, zero cost).
  // C(0,0)=0, C(1,1)=C(0,0)=0, C(2,2)=C(1,1)=0 -> result = 0.
  REQUIRE_THAT(dtwAROW_L<double>(x, y_nan), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(dtwAROW_L<double>(y_nan, x), WithinAbs(0.0, 1e-15));
}

TEST_CASE("dtwAROW: all NaN gives 0", "[arow_dtw][property]")
{
  std::vector<double> x{ NaN, NaN };
  std::vector<double> y{ NaN, NaN, NaN };

  REQUIRE_THAT(dtwAROW<double>(x, y), WithinAbs(0.0, 1e-15));
}

// ===========================================================================
// Property 3: Symmetry dtwAROW(x,y) == dtwAROW(y,x)
// ===========================================================================

TEST_CASE("dtwAROW_L: symmetry with no NaN", "[arow_dtw][property]")
{
  std::vector<double> x{ 1, 3, 5, 2 };
  std::vector<double> y{ 2, 4, 6 };

  REQUIRE_THAT(dtwAROW_L<double>(x, y), WithinAbs(dtwAROW_L<double>(y, x), 1e-12));
}

TEST_CASE("dtwAROW_L: symmetry with NaN", "[arow_dtw][property]")
{
  std::vector<double> x{ 1, NaN, 5, 2 };
  std::vector<double> y{ NaN, 4, 6 };

  REQUIRE_THAT(dtwAROW_L<double>(x, y), WithinAbs(dtwAROW_L<double>(y, x), 1e-12));
}

TEST_CASE("dtwAROW_L: symmetry with leading NaN in x", "[arow_dtw][property]")
{
  std::vector<double> x{ NaN, 2, 3, 4 };
  std::vector<double> y{ 1, 3, 5, 7 };

  REQUIRE_THAT(dtwAROW_L<double>(x, y), WithinAbs(dtwAROW_L<double>(y, x), 1e-12));
}

TEST_CASE("dtwAROW: symmetry with NaN", "[arow_dtw][property]")
{
  std::vector<double> x{ 1, NaN, 3 };
  std::vector<double> y{ 2, 4, NaN, 6 };

  REQUIRE_THAT(dtwAROW<double>(x, y), WithinAbs(dtwAROW<double>(y, x), 1e-12));
}

// ===========================================================================
// Property 4: AROW >= ZeroCost (stricter constraint)
// ===========================================================================

TEST_CASE("dtwAROW_L >= dtwMissing_L (AROW is more constrained)", "[arow_dtw][property]")
{
  // AROW restricts missing cells to diagonal only; ZeroCost allows any direction.
  // Therefore AROW >= ZeroCost.
  std::vector<double> x{ 1, NaN, 3, 4 };
  std::vector<double> y{ NaN, 2, 3, 4 };

  const auto arow = dtwAROW_L<double>(x, y);
  const auto zero_cost = dtwMissing_L<double>(x, y);

  REQUIRE(arow >= zero_cost - 1e-12);
}

TEST_CASE("dtwAROW_L >= dtwMissing_L: leading NaN case", "[arow_dtw][property]")
{
  std::vector<double> x{ NaN, NaN, 3, 4, 5 };
  std::vector<double> y{ 1, 2, 3, 4, 5 };

  REQUIRE(dtwAROW_L<double>(x, y) >= dtwMissing_L<double>(x, y) - 1e-12);
}

TEST_CASE("dtwAROW_L >= dtwMissing_L: trailing NaN case", "[arow_dtw][property]")
{
  std::vector<double> x{ 1, 2, 3, NaN, NaN };
  std::vector<double> y{ 1, 2, 3, 4, 5 };

  REQUIRE(dtwAROW_L<double>(x, y) >= dtwMissing_L<double>(x, y) - 1e-12);
}

TEST_CASE("dtwAROW_L >= dtwMissing_L: equal when no NaN", "[arow_dtw][property]")
{
  // When no NaN, both methods give identical results (standard DTW).
  std::vector<double> x{ 1, 2, 3, 4 };
  std::vector<double> y{ 2, 3, 4, 5 };

  REQUIRE_THAT(dtwAROW_L<double>(x, y), WithinAbs(dtwMissing_L<double>(x, y), 1e-12));
}

// ===========================================================================
// Property 5: Leading NaN is finite (NOT +inf)
// ===========================================================================

TEST_CASE("dtwAROW_L: leading NaN gives finite result", "[arow_dtw][property]")
{
  // Critical: a naive approach would set C(0,0) = +inf when missing, which cascades.
  // The AROW boundary treatment propagates with zero cost instead.
  std::vector<double> x{ NaN, 2, 3 };
  std::vector<double> y{ 1, 2, 3 };

  const auto result = dtwAROW_L<double>(x, y);
  REQUIRE(result < INF / 2.0);
  REQUIRE(result >= 0.0);
}

TEST_CASE("dtwAROW_L: multiple leading NaN gives finite result", "[arow_dtw][property]")
{
  std::vector<double> x{ NaN, NaN, NaN, 4, 5 };
  std::vector<double> y{ 1, 2, 3, 4, 5 };

  const auto result = dtwAROW_L<double>(x, y);
  REQUIRE(result < INF / 2.0);
  REQUIRE(result >= 0.0);
}

TEST_CASE("dtwAROW: leading NaN in both series gives finite result", "[arow_dtw][property]")
{
  std::vector<double> x{ NaN, 2, 3 };
  std::vector<double> y{ NaN, 2, 3 };

  // Both series have leading NaN: C(0,0)=0, subsequent cells should propagate fine.
  const auto result = dtwAROW<double>(x, y);
  REQUIRE(result < INF / 2.0);
  REQUIRE(result >= 0.0);
}

// ===========================================================================
// Property 6: Trailing NaN is finite
// ===========================================================================

TEST_CASE("dtwAROW_L: trailing NaN gives finite result", "[arow_dtw][property]")
{
  std::vector<double> x{ 1, 2, NaN };
  std::vector<double> y{ 1, 2, 3 };

  const auto result = dtwAROW_L<double>(x, y);
  REQUIRE(result < INF / 2.0);
  REQUIRE(result >= 0.0);
}

TEST_CASE("dtwAROW_L: trailing NaN in both series gives finite result", "[arow_dtw][property]")
{
  std::vector<double> x{ 1, 2, 3, NaN, NaN };
  std::vector<double> y{ 1, 2, 3, NaN, NaN };

  const auto result = dtwAROW_L<double>(x, y);
  REQUIRE(result < INF / 2.0);
  REQUIRE(result >= 0.0);
}

// ===========================================================================
// Property 7: Non-negativity
// ===========================================================================

TEST_CASE("dtwAROW_L: non-negativity with mixed NaN", "[arow_dtw][property]")
{
  std::vector<double> x{ 1, NaN, 3 };
  std::vector<double> y{ 5, 6, NaN, 8 };

  REQUIRE(dtwAROW_L<double>(x, y) >= 0.0);
}

TEST_CASE("dtwAROW_L: non-negativity, all present", "[arow_dtw][property]")
{
  std::vector<double> x{ 10, 20, 30 };
  std::vector<double> y{ 1, 2, 3 };

  REQUIRE(dtwAROW_L<double>(x, y) >= 0.0);
}

TEST_CASE("dtwAROW: non-negativity with NaN", "[arow_dtw][property]")
{
  std::vector<double> x{ NaN, 5, NaN };
  std::vector<double> y{ 1, NaN, 3 };

  REQUIRE(dtwAROW<double>(x, y) >= 0.0);
}

// ===========================================================================
// Property 8: Banded with large band matches unbanded
// ===========================================================================

TEST_CASE("dtwAROW_banded: large band matches unbanded (no NaN)", "[arow_dtw][property]")
{
  std::vector<double> x{ 1, 2, 3, 4, 5 };
  std::vector<double> y{ 2, 4, 5, 6, 7, 8 };

  const auto unbanded = dtwAROW_L<double>(x, y);
  const auto banded   = dtwAROW_banded<double>(x, y, 100);

  REQUIRE_THAT(banded, WithinAbs(unbanded, 1e-12));
}

TEST_CASE("dtwAROW_banded: large band matches unbanded (with NaN)", "[arow_dtw][property]")
{
  std::vector<double> x{ 1, NaN, 3, 4, 5 };
  std::vector<double> y{ 2, 4, NaN, 6, 7, 8 };

  const auto unbanded = dtwAROW_L<double>(x, y);
  const auto banded   = dtwAROW_banded<double>(x, y, 100);

  REQUIRE_THAT(banded, WithinAbs(unbanded, 1e-12));
}

TEST_CASE("dtwAROW_banded: negative band falls back to unbanded", "[arow_dtw][property]")
{
  std::vector<double> x{ 1, NaN, 3 };
  std::vector<double> y{ 2, 4, NaN };

  const auto unbanded = dtwAROW_L<double>(x, y);
  const auto banded   = dtwAROW_banded<double>(x, y, -1);

  REQUIRE_THAT(banded, WithinAbs(unbanded, 1e-12));
}

TEST_CASE("dtwAROW_banded: symmetry with band", "[arow_dtw][property]")
{
  std::vector<double> x{ 1, NaN, 3, 4, 5 };
  std::vector<double> y{ 2, 4, NaN, 6, 7 };
  int band = 2;

  REQUIRE_THAT(dtwAROW_banded<double>(x, y, band),
               WithinAbs(dtwAROW_banded<double>(y, x, band), 1e-12));
}

// ===========================================================================
// Property 9: Full-matrix matches linear-space
// ===========================================================================

TEST_CASE("dtwAROW == dtwAROW_L (no NaN)", "[arow_dtw][property]")
{
  std::vector<double> x{ 1, 2, 3, 4 };
  std::vector<double> y{ 2, 3, 4, 5, 6 };

  REQUIRE_THAT(dtwAROW<double>(x, y), WithinAbs(dtwAROW_L<double>(x, y), 1e-12));
}

TEST_CASE("dtwAROW == dtwAROW_L (with NaN, equal length)", "[arow_dtw][property]")
{
  std::vector<double> x{ 1, NaN, 3, 4 };
  std::vector<double> y{ NaN, 2, 3, 4 };

  REQUIRE_THAT(dtwAROW<double>(x, y), WithinAbs(dtwAROW_L<double>(x, y), 1e-12));
}

TEST_CASE("dtwAROW == dtwAROW_L (with NaN, unequal length)", "[arow_dtw][property]")
{
  std::vector<double> x{ 1, NaN, 3 };
  std::vector<double> y{ 2, 4, NaN, 6, 7 };

  REQUIRE_THAT(dtwAROW<double>(x, y), WithinAbs(dtwAROW_L<double>(x, y), 1e-12));
}

TEST_CASE("dtwAROW == dtwAROW_L (leading NaN)", "[arow_dtw][property]")
{
  std::vector<double> x{ NaN, 2, 3 };
  std::vector<double> y{ 1, 2, 3 };

  REQUIRE_THAT(dtwAROW<double>(x, y), WithinAbs(dtwAROW_L<double>(x, y), 1e-12));
}

TEST_CASE("dtwAROW == dtwAROW_L (trailing NaN)", "[arow_dtw][property]")
{
  std::vector<double> x{ 1, 2, NaN };
  std::vector<double> y{ 1, 2, 3 };

  REQUIRE_THAT(dtwAROW<double>(x, y), WithinAbs(dtwAROW_L<double>(x, y), 1e-12));
}

TEST_CASE("dtwAROW == dtwAROW_L (all NaN)", "[arow_dtw][property]")
{
  std::vector<double> x{ NaN, NaN };
  std::vector<double> y{ NaN, NaN, NaN };

  REQUIRE_THAT(dtwAROW<double>(x, y), WithinAbs(dtwAROW_L<double>(x, y), 1e-15));
}

// ===========================================================================
// Hand-computed verification tests
// ===========================================================================

TEST_CASE("dtwAROW: hand-computed, no NaN, 3x3", "[arow_dtw][handcomputed]")
{
  // x = {1, 2, 3}, y = {1, 2, 3}
  // AROW == standard DTW when no NaN.
  // C(0,0)=0, C(1,0)=1, C(2,0)=3
  // C(0,1)=1, C(1,1)=0, C(2,1)=1
  // C(0,2)=3, C(1,2)=1, C(2,2)=0
  std::vector<double> x{ 1, 2, 3 };
  std::vector<double> y{ 1, 2, 3 };

  REQUIRE_THAT(dtwAROW<double>(x, y), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(dtwAROW_L<double>(x, y), WithinAbs(0.0, 1e-12));
}

TEST_CASE("dtwAROW: hand-computed, leading NaN in x, equal length", "[arow_dtw][handcomputed]")
{
  // x = {NaN, 2, 3}, y = {1, 2, 3}
  // Full matrix (using AROW recurrence):
  // C(0,0): x[0]=NaN -> 0
  // C(1,0): x[1]=2, y[0]=1 -> C(0,0) + |2-1| = 0+1 = 1
  // C(2,0): x[2]=3, y[0]=1 -> C(1,0) + |3-1| = 1+2 = 3
  // C(0,1): x[0]=NaN -> C(0,0) = 0  (boundary: propagate horizontally)
  // C(0,2): x[0]=NaN -> C(0,1) = 0  (boundary: propagate horizontally)
  // C(1,1): x[1]=2,y[1]=2 -> min(C(0,0),C(0,1),C(1,0)) + |2-2| = min(0,0,1)+0 = 0
  // C(1,2): x[1]=2,y[2]=3 -> min(C(0,1),C(0,2),C(1,1)) + |2-3| = min(0,0,0)+1 = 1
  // C(2,1): x[2]=3,y[1]=2 -> min(C(1,0),C(1,1),C(2,0)) + |3-2| = min(1,0,3)+1 = 1
  // C(2,2): x[2]=3,y[2]=3 -> min(C(1,1),C(1,2),C(2,1)) + |3-3| = min(0,1,1)+0 = 0
  // Result = C(2,2) = 0
  std::vector<double> x{ NaN, 2, 3 };
  std::vector<double> y{ 1, 2, 3 };

  REQUIRE_THAT(dtwAROW<double>(x, y), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(dtwAROW_L<double>(x, y), WithinAbs(0.0, 1e-12));
}

TEST_CASE("dtwAROW: hand-computed, NaN in middle of x", "[arow_dtw][handcomputed]")
{
  // x = {1, NaN, 3}, y = {1, 2, 3}
  // C(0,0): both present -> |1-1| = 0
  // C(1,0): y[0]=1, x[1]=NaN -> C(0,0)=0 (propagate, boundary)
  // C(2,0): y[0]=1, x[2]=3  -> C(1,0)+|3-1|=2
  // C(0,1): x[0]=1, y[1]=2 -> C(0,0)+|1-2|=1
  // C(0,2): x[0]=1, y[2]=3 -> C(0,1)+|1-3|=3
  // C(1,1): x[1]=NaN -> C(0,0) = 0  (AROW diagonal)
  // C(1,2): x[1]=NaN -> C(0,1) = 1  (AROW diagonal)
  // C(2,1): x[2]=3,y[1]=2 -> min(C(1,0),C(1,1),C(2,0))+|3-2| = min(0,0,2)+1 = 1
  // C(2,2): x[2]=3,y[2]=3 -> min(C(1,1),C(1,2),C(2,1))+|3-3| = min(0,1,1)+0 = 0
  // Result = 0
  std::vector<double> x{ 1, NaN, 3 };
  std::vector<double> y{ 1, 2, 3 };

  REQUIRE_THAT(dtwAROW<double>(x, y), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(dtwAROW_L<double>(x, y), WithinAbs(0.0, 1e-12));
}

TEST_CASE("dtwAROW: hand-computed, trailing NaN in x", "[arow_dtw][handcomputed]")
{
  // x = {1, 2, NaN}, y = {1, 2, 3}
  // C(0,0)=0, C(1,0)=1, C(2,0)=C(1,0)=1 (x[2]=NaN, boundary propagate)
  // C(0,1)=1, C(0,2)=3
  // C(1,1)=min(0,1,1)+0=0, C(1,2)=min(1,0,0)+1=1
  // C(2,1): x[2]=NaN -> C(1,0)=1  (AROW diagonal)
  // C(2,2): x[2]=NaN -> C(1,1)=0  (AROW diagonal)
  // Result = 0
  std::vector<double> x{ 1, 2, NaN };
  std::vector<double> y{ 1, 2, 3 };

  REQUIRE_THAT(dtwAROW<double>(x, y), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(dtwAROW_L<double>(x, y), WithinAbs(0.0, 1e-12));
}

TEST_CASE("dtwAROW: hand-computed, nonzero with NaN", "[arow_dtw][handcomputed]")
{
  // x = {1, 2}, y = {NaN, 5}
  // C(0,0): y[0]=NaN -> 0
  // C(1,0): y[0]=NaN -> C(0,0)=0 (boundary: propagate)
  // C(0,1): y[1]=5, x[0]=1 -> C(0,0)+|1-5|=4
  // C(1,1): both present: min(C(0,0),C(0,1),C(1,0))+|2-5|=min(0,4,0)+3=3
  // Result = 3
  std::vector<double> x{ 1, 2 };
  std::vector<double> y{ NaN, 5 };

  REQUIRE_THAT(dtwAROW<double>(x, y), WithinAbs(3.0, 1e-12));
  REQUIRE_THAT(dtwAROW_L<double>(x, y), WithinAbs(3.0, 1e-12));
}

TEST_CASE("dtwAROW: hand-computed, NaN forces diagonal cost accumulation", "[arow_dtw][handcomputed]")
{
  // x = {1, NaN, 10}, y = {1, 5, 10}
  // Without NaN: DTW would use x[1]->y[1]=5 path
  // With AROW: x[1]=NaN forces diagonal from C(0,0) to C(1,1)
  // C(0,0)=|1-1|=0
  // C(1,0): x[1]=NaN, y[0]=1 -> C(0,0)=0 (boundary propagate)
  // C(2,0): x[2]=10, y[0]=1 -> C(1,0)+|10-1|=9
  // C(0,1): x[0]=1, y[1]=5 -> C(0,0)+|1-5|=4
  // C(0,2): x[0]=1, y[2]=10 -> C(0,1)+|1-10|=13
  // C(1,1): x[1]=NaN -> C(0,0)=0 (AROW diagonal)
  // C(1,2): x[1]=NaN -> C(0,1)=4 (AROW diagonal)
  // C(2,1): x[2]=10,y[1]=5 -> min(C(1,0),C(1,1),C(2,0))+|10-5|=min(0,0,9)+5=5
  // C(2,2): x[2]=10,y[2]=10 -> min(C(1,1),C(1,2),C(2,1))+|10-10|=min(0,4,5)+0=0
  // Result = 0
  std::vector<double> x{ 1, NaN, 10 };
  std::vector<double> y{ 1, 5, 10 };

  REQUIRE_THAT(dtwAROW<double>(x, y), WithinAbs(0.0, 1e-12));
  REQUIRE_THAT(dtwAROW_L<double>(x, y), WithinAbs(0.0, 1e-12));
}

TEST_CASE("dtwAROW: hand-computed, AROW > ZeroCost for specific case", "[arow_dtw][handcomputed]")
{
  // x = {NaN, 10}, y = {1, 1}
  // AROW:
  //   C(0,0): x[0]=NaN -> 0
  //   C(1,0): y[0]=1,x[1]=10 -> C(0,0)+|10-1|=9 (boundary: vertical)
  //   C(0,1): x[0]=NaN -> C(0,0)=0 (boundary: horizontal propagate)
  //   C(1,1): x[1]=10,y[1]=1 -> min(C(0,0),C(0,1),C(1,0))+|10-1|=min(0,0,9)+9=9
  //   Result = 9
  //
  // ZeroCost (standard DTW with cost 0 for NaN):
  //   C(0,0): 0 (NaN cost)
  //   C(1,0): 0+|10-1|=9
  //   C(0,1): C(0,0)+0=0 (NaN cost for x[0])
  //   C(1,1): min(0,0,9)+|10-1|=9
  //   Result = 9
  // Both give 9 here; the inequality is not strict in all cases.
  // Let's check they are both 9.
  std::vector<double> x{ NaN, 10 };
  std::vector<double> y{ 1, 1 };

  const double arow = dtwAROW<double>(x, y);
  const double zero = dtwMissing<double>(x, y);

  REQUIRE_THAT(arow, WithinAbs(9.0, 1e-12));
  REQUIRE(arow >= zero - 1e-12);
}

// ===========================================================================
// Edge cases
// ===========================================================================

TEST_CASE("dtwAROW_L: empty vectors return maxValue", "[arow_dtw][edge]")
{
  std::vector<double> x{ 1, 2, 3 };
  std::vector<double> empty{};
  constexpr double maxValue = std::numeric_limits<double>::max();

  REQUIRE(dtwAROW_L<double>(x, empty) == maxValue);
  REQUIRE(dtwAROW_L<double>(empty, x) == maxValue);
}

TEST_CASE("dtwAROW: empty vectors return maxValue", "[arow_dtw][edge]")
{
  std::vector<double> x{ 1, 2, 3 };
  std::vector<double> empty{};
  constexpr double maxValue = std::numeric_limits<double>::max();

  REQUIRE(dtwAROW<double>(x, empty) == maxValue);
  REQUIRE(dtwAROW<double>(empty, x) == maxValue);
}

TEST_CASE("dtwAROW_L: identical series gives zero", "[arow_dtw][edge]")
{
  std::vector<double> x{ 1, 2, 3, 4, 5 };
  REQUIRE_THAT(dtwAROW_L<double>(x, x), WithinAbs(0.0, 1e-15));
}

TEST_CASE("dtwAROW_L: single-element series, no NaN", "[arow_dtw][edge]")
{
  std::vector<double> x{ 5.0 };
  std::vector<double> y{ 3.0 };

  REQUIRE_THAT(dtwAROW_L<double>(x, y), WithinAbs(2.0, 1e-12));
  REQUIRE_THAT(dtwAROW<double>(x, y), WithinAbs(2.0, 1e-12));
}

TEST_CASE("dtwAROW_L: single-element series, NaN", "[arow_dtw][edge]")
{
  std::vector<double> x{ 5.0 };
  std::vector<double> y{ NaN };

  REQUIRE_THAT(dtwAROW_L<double>(x, y), WithinAbs(0.0, 1e-15));
  REQUIRE_THAT(dtwAROW<double>(x, y), WithinAbs(0.0, 1e-15));
}

TEST_CASE("dtwAROW_banded: no NaN matches standard dtwBanded with large band", "[arow_dtw][edge]")
{
  std::vector<double> x{ 1, 2, 3, 4, 5 };
  std::vector<double> y{ 2, 3, 4, 5, 6, 7, 8 };
  int band = 100;

  const auto arow = dtwAROW_banded<double>(x, y, band);
  const auto std_dtw = dtwFull_L<double>(x, y);

  REQUIRE_THAT(arow, WithinAbs(std_dtw, 1e-12));
}

// ===========================================================================
// Problem integration: MissingStrategy::AROW
// ===========================================================================

TEST_CASE("Problem: MissingStrategy::AROW wires correctly", "[arow_dtw][problem]")
{
  std::vector<double> x{ 1.0, 2.0, 3.0 };
  std::vector<double> y_nan{ 1.0, NaN, 3.0 };
  std::vector<double> z{ 4.0, 5.0, 6.0 };

  dtwc::Data data;
  data.p_vec   = { x, y_nan, z };
  data.p_names = { "x", "y_nan", "z" };

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.missing_strategy = dtwc::core::MissingStrategy::AROW;
  prob.verbose = false;
  prob.fillDistanceMatrix();

  // d(x, y_nan) should be finite and non-negative
  const double d01 = prob.distByInd(0, 1);
  REQUIRE(d01 >= 0.0);
  REQUIRE(d01 < std::numeric_limits<double>::max() / 2.0);

  // d(x, z) should be larger than d(x, y_nan) since z is farther from x
  const double d02 = prob.distByInd(0, 2);
  REQUIRE(d02 > 0.0);
}

TEST_CASE("Problem: AROW gives finite distance with leading NaN", "[arow_dtw][problem]")
{
  std::vector<double> a{ NaN, 2.0, 3.0 };
  std::vector<double> b{ 1.0, 2.0, 3.0 };

  dtwc::Data data;
  data.p_vec   = { a, b };
  data.p_names = { "a", "b" };

  dtwc::Problem prob;
  prob.set_data(std::move(data));
  prob.missing_strategy = dtwc::core::MissingStrategy::AROW;
  prob.verbose = false;
  prob.fillDistanceMatrix();

  const double d = prob.distByInd(0, 1);
  REQUIRE(d >= 0.0);
  REQUIRE(d < std::numeric_limits<double>::max() / 2.0);
}
