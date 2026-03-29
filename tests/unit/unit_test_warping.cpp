/**
 * @file unit_test_warping.cpp
 * @brief Unit test file for time warping functions
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 03 Dec 2023
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinAbs;

using namespace dtwc;

TEST_CASE("dtwFull_test", "[dtwFull]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 }, z{ 1, 2, 3 }, empty{};
  constexpr double ground_truth = 13;

  // Zero distance between same vectors:
  REQUIRE_THAT(dtwFull<data_t>(x, x), WithinAbs(0, 1e-15));
  REQUIRE_THAT(dtwFull<data_t>(x, z), WithinAbs(0, 1e-15));
  REQUIRE_THAT(dtwFull<data_t>(z, x), WithinAbs(0, 1e-15));

  // Some distance between others: 13
  REQUIRE_THAT(dtwFull<data_t>(x, y), WithinAbs(ground_truth, 1e-15));
  REQUIRE_THAT(dtwFull<data_t>(y, x), WithinAbs(ground_truth, 1e-15));

  // Empty vector should give infinite cost.
  REQUIRE(dtwFull<data_t>(x, empty) > 1e10);
  REQUIRE(dtwFull<data_t>(empty, x) > 1e10);
}

TEST_CASE("dtwFull_L_test", "[dtwFull_L]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 }, z{ 1, 2, 3 }, empty{};
  constexpr double ground_truth = 13;

  // Zero distance between same vectors:
  REQUIRE_THAT(dtwFull_L<data_t>(x, x), WithinAbs(0, 1e-15));
  REQUIRE_THAT(dtwFull_L<data_t>(x, z), WithinAbs(0, 1e-15));
  REQUIRE_THAT(dtwFull_L<data_t>(z, x), WithinAbs(0, 1e-15));

  // Some distance between others: 13
  REQUIRE_THAT(dtwFull_L<data_t>(x, y), WithinAbs(ground_truth, 1e-15));
  REQUIRE_THAT(dtwFull_L<data_t>(y, x), WithinAbs(ground_truth, 1e-15));

  // Empty vector should give infinite cost.
  REQUIRE(dtwFull_L<data_t>(x, empty) > 1e10);
  REQUIRE(dtwFull_L<data_t>(empty, x) > 1e10);
}

TEST_CASE("[Phase2] dtwFull_L early abandon returns maxValue when threshold exceeded", "[dtwFull_L][early_abandon]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 100, 200, 300 };

  // The true DTW distance is large; set a tiny threshold to trigger early abandon
  constexpr data_t threshold = 1.0;
  auto result = dtwFull_L<data_t>(x, y, threshold);
  REQUIRE(result > 1e10); // Should return maxValue
}

TEST_CASE("[Phase2] dtwFull_L early abandon returns normal result when threshold is large", "[dtwFull_L][early_abandon]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 };
  constexpr double ground_truth = 13;

  // Large threshold should not trigger early abandon
  constexpr data_t threshold = 1000.0;
  REQUIRE_THAT(dtwFull_L<data_t>(x, y, threshold), WithinAbs(ground_truth, 1e-15));
}

TEST_CASE("[Phase2] dtwFull_L early abandon negative threshold disables feature", "[dtwFull_L][early_abandon]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 };
  constexpr double ground_truth = 13;

  // Negative threshold (default) means no early abandoning
  REQUIRE_THAT(dtwFull_L<data_t>(x, y, static_cast<data_t>(-1)), WithinAbs(ground_truth, 1e-15));
}

TEST_CASE("[Phase2] dtwFull_L early abandon exact threshold boundary", "[dtwFull_L][early_abandon]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 };
  constexpr double ground_truth = 13;

  // Threshold equal to the true DTW distance should return the distance (not abandoned)
  REQUIRE_THAT(dtwFull_L<data_t>(x, y, static_cast<data_t>(ground_truth)), WithinAbs(ground_truth, 1e-15));
}

TEST_CASE("[Phase2] dtwFull_L default parameter backward compatibility", "[dtwFull_L][early_abandon]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 };
  constexpr double ground_truth = 13;

  // Calling without the early_abandon parameter should work exactly as before
  REQUIRE_THAT(dtwFull_L<data_t>(x, y), WithinAbs(ground_truth, 1e-15));
}

// =========================================================================
// SquaredL2 metric tests
// =========================================================================

TEST_CASE("dtwFull SquaredL2 basic", "[dtwFull][SquaredL2]")
{
  using data_t = double;
  namespace core = dtwc::core;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 }, z{ 1, 2, 3 }, empty{};
  constexpr double ground_truth = 35.0; // hand-computed squared-L2 DTW

  REQUIRE_THAT(dtwFull<data_t>(x, x, core::MetricType::SquaredL2), WithinAbs(0, 1e-15));
  REQUIRE_THAT(dtwFull<data_t>(x, z, core::MetricType::SquaredL2), WithinAbs(0, 1e-15));
  REQUIRE_THAT(dtwFull<data_t>(x, y, core::MetricType::SquaredL2), WithinAbs(ground_truth, 1e-15));
  REQUIRE_THAT(dtwFull<data_t>(y, x, core::MetricType::SquaredL2), WithinAbs(ground_truth, 1e-15));
  REQUIRE(dtwFull<data_t>(x, empty, core::MetricType::SquaredL2) > 1e10);
}

TEST_CASE("dtwFull_L SquaredL2 basic", "[dtwFull_L][SquaredL2]")
{
  using data_t = double;
  namespace core = dtwc::core;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 }, z{ 1, 2, 3 }, empty{};
  constexpr double ground_truth = 35.0;

  REQUIRE_THAT(dtwFull_L<data_t>(x, x, -1, core::MetricType::SquaredL2), WithinAbs(0, 1e-15));
  REQUIRE_THAT(dtwFull_L<data_t>(x, z, -1, core::MetricType::SquaredL2), WithinAbs(0, 1e-15));
  REQUIRE_THAT(dtwFull_L<data_t>(x, y, -1, core::MetricType::SquaredL2), WithinAbs(ground_truth, 1e-15));
  REQUIRE_THAT(dtwFull_L<data_t>(y, x, -1, core::MetricType::SquaredL2), WithinAbs(ground_truth, 1e-15));
  REQUIRE(dtwFull_L<data_t>(x, empty, -1, core::MetricType::SquaredL2) > 1e10);
}

TEST_CASE("dtwFull_L SquaredL2 matches dtwFull SquaredL2", "[dtwFull_L][SquaredL2]")
{
  using data_t = double;
  namespace core = dtwc::core;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 };

  const auto full = dtwFull<data_t>(x, y, core::MetricType::SquaredL2);
  const auto light = dtwFull_L<data_t>(x, y, -1, core::MetricType::SquaredL2);
  REQUIRE_THAT(light, WithinAbs(full, 1e-15));
}

TEST_CASE("dtwFull_L SquaredL2 early abandon works", "[dtwFull_L][SquaredL2][early_abandon]")
{
  using data_t = double;
  namespace core = dtwc::core;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 100, 200, 300 };

  // True distance is huge; tiny threshold should trigger early abandon
  auto result = dtwFull_L<data_t>(x, y, 1.0, core::MetricType::SquaredL2);
  REQUIRE(result > 1e10);

  // Large threshold should not trigger early abandon
  auto result2 = dtwFull_L<data_t>(x, y, 1e12, core::MetricType::SquaredL2);
  auto ref = dtwFull<data_t>(x, y, core::MetricType::SquaredL2);
  REQUIRE_THAT(result2, WithinAbs(ref, 1e-10));
}

TEST_CASE("dtwFull_L SquaredL2 gives different result than L1", "[dtwFull_L][SquaredL2]")
{
  using data_t = double;
  namespace core = dtwc::core;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 };

  const auto l1 = dtwFull_L<data_t>(x, y);
  const auto sq = dtwFull_L<data_t>(x, y, -1, core::MetricType::SquaredL2);

  // L1 = 13, SquaredL2 = 35 — they must differ
  REQUIRE(l1 != sq);
  REQUIRE_THAT(l1, WithinAbs(13.0, 1e-15));
  REQUIRE_THAT(sq, WithinAbs(35.0, 1e-15));
}

TEST_CASE("dtwBanded_test", "[dtwBanded]")
{
  using data_t = double;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 }, z{ 1, 2, 3 }, empty{};
  constexpr double ground_truth = 13;

  // Zero distance between same vectors:
  REQUIRE_THAT(dtwBanded<data_t>(x, x), WithinAbs(0, 1e-15));
  REQUIRE_THAT(dtwBanded<data_t>(x, z), WithinAbs(0, 1e-15));
  REQUIRE_THAT(dtwBanded<data_t>(z, x), WithinAbs(0, 1e-15));

  // Some distance between others with too large band, should be same as unbanded.
  int band = 100;
  REQUIRE_THAT(dtwBanded<data_t>(x, y, band), WithinAbs(ground_truth, 1e-15));
  REQUIRE_THAT(dtwBanded<data_t>(y, x, band), WithinAbs(ground_truth, 1e-15));

  // Banded distance:
  band = 2;
  REQUIRE_THAT(dtwBanded<data_t>(x, y, band), WithinAbs(ground_truth, 1e-15));
  REQUIRE_THAT(dtwBanded<data_t>(y, x, band), WithinAbs(ground_truth, 1e-15));

  // Empty vector should give infinite cost.
  REQUIRE(dtwBanded<data_t>(x, empty) > 1e10);
  REQUIRE(dtwBanded<data_t>(empty, x) > 1e10);
}

TEST_CASE("dtwBanded SquaredL2 matches dtwFull_L SquaredL2", "[dtwBanded][SquaredL2]")
{
  using data_t = double;
  namespace core = dtwc::core;
  std::vector<data_t> x{ 1, 2, 3 }, y{ 3, 4, 5, 6, 7 };
  constexpr double ground_truth = 35.0;

  // band=-1 falls back to dtwFull_L
  REQUIRE_THAT(dtwBanded<data_t>(x, y, -1, -1, core::MetricType::SquaredL2),
               WithinAbs(ground_truth, 1e-15));

  // Large band should give same result as unbanded
  REQUIRE_THAT(dtwBanded<data_t>(x, y, 100, -1, core::MetricType::SquaredL2),
               WithinAbs(ground_truth, 1e-15));

  // Tight band may give higher cost (restricted warping path)
  const auto banded_sq = dtwBanded<data_t>(x, y, 2, -1, core::MetricType::SquaredL2);
  REQUIRE(banded_sq >= ground_truth - 1e-15); // banded >= unbanded

  // Symmetry under banding
  REQUIRE_THAT(dtwBanded<data_t>(y, x, 2, -1, core::MetricType::SquaredL2),
               WithinAbs(banded_sq, 1e-15));

  // Zero for identical series
  REQUIRE_THAT(dtwBanded<data_t>(x, x, 2, -1, core::MetricType::SquaredL2),
               WithinAbs(0, 1e-15));
}