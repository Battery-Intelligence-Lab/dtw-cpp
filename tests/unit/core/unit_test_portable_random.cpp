/**
 * @file unit_test_portable_random.cpp
 * @brief Cross-library regression tests for portable seeded sampling.
 */

#include <core/portable_random.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

TEST_CASE("portable random mapping is fixed across standard libraries",
          "[random][determinism]")
{
  STATIC_REQUIRE(dtwc::core::detail::multiply_high(0, UINT64_MAX) == 0);
  STATIC_REQUIRE(dtwc::core::detail::multiply_high(UINT64_MAX, UINT64_MAX)
                 == UINT64_MAX - 1);
  STATIC_REQUIRE(dtwc::core::detail::multiply_high(
                   0xfedcba9876543210ULL, 0xdeadbeefcafebabeULL)
                 == 15974376802349750943ULL);

  std::mt19937_64 engine(0);
  std::vector<std::uint64_t> bounded;
  for (int i = 0; i < 5; ++i)
    bounded.push_back(dtwc::core::portable_bounded(engine, 10));

  REQUIRE(bounded == std::vector<std::uint64_t>{ 1, 9, 0, 5, 5 });
  REQUIRE(dtwc::core::portable_unit_interval(engine)
          == 0.057159791465323573);

  std::vector<int> values(8);
  std::iota(values.begin(), values.end(), 0);
  dtwc::core::portable_shuffle(values.begin(), values.end(), engine);
  REQUIRE(values == std::vector<int>{ 0, 2, 5, 3, 6, 1, 4, 7 });
  REQUIRE(engine() == 6352792256529822470ULL);

  std::mt19937_64 rejection_engine(0);
  constexpr auto rejection_heavy_bound = (std::uint64_t{ 1 } << 63) + 1;
  REQUIRE(dtwc::core::portable_bounded(
            rejection_engine, rejection_heavy_bound)
          == 364959846503117916ULL);
  // This literal pins the rejection path's draw consumption as well as output.
  REQUIRE(rejection_engine() == 11021831128136023278ULL);

  std::mt19937_64 short_shuffle_engine(0);
  std::array<int, 3> short_values{ 0, 1, 2 };
  dtwc::core::portable_shuffle(
    short_values.begin(), short_values.end(), short_shuffle_engine);
  REQUIRE(short_values == std::array<int, 3>{ 1, 0, 2 });
}

#if defined(__SIZEOF_INT128__)
TEST_CASE("portable multiply-high matches the native 128-bit arbiter",
          "[random][determinism][differential]")
{
  std::mt19937_64 lhs_engine(123456789);
  std::mt19937_64 rhs_engine(987654321);
  for (int iteration = 0; iteration < 1'000'000; ++iteration) {
    const std::uint64_t lhs = lhs_engine();
    const std::uint64_t rhs = rhs_engine();
    const auto reference = static_cast<std::uint64_t>(
      (static_cast<unsigned __int128>(lhs) * rhs) >> 64);
    const auto actual = dtwc::core::detail::multiply_high(lhs, rhs);
    if (actual != reference) {
      INFO("iteration=" << iteration << " lhs=" << lhs << " rhs=" << rhs);
      FAIL("portable multiply-high disagrees with unsigned __int128");
    }
  }
  SUCCEED();
}
#endif

TEST_CASE("portable bounded sampling rejects an empty domain",
          "[random][errors]")
{
  std::mt19937_64 engine(0);
  REQUIRE(dtwc::core::portable_bounded(engine, 1) == 0);
  REQUIRE(engine() == 2947667278772165694ULL);
  REQUIRE_THROWS_AS(dtwc::core::portable_bounded(engine, 0),
                    std::invalid_argument);

  std::mt19937_64 real_engine(0);
  const double value = dtwc::core::portable_real_below(real_engine, 3.0);
  REQUIRE(value == 0.47938009011138238);
  REQUIRE(value >= 0.0);
  REQUIRE(value < 3.0);
  REQUIRE_THROWS_AS(dtwc::core::portable_real_below(real_engine, 0.0),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(dtwc::core::portable_real_below(
                      real_engine, std::numeric_limits<double>::infinity()),
                    std::invalid_argument);
}

TEST_CASE("portable weighted and selection sampling have fixed boundaries",
          "[random][determinism][sampling]")
{
  std::mt19937_64 weighted_engine(42);
  const std::array<double, 5> weights{ 0.0, 1.0, 0.0, 3.0, 0.0 };
  REQUIRE(dtwc::core::portable_weighted_index(
            weights.begin(), weights.end(), 4.0, weighted_engine)
          == 3);
  REQUIRE(weighted_engine() == 11788048577503494824ULL);

  std::mt19937_64 all_zero_engine(42);
  const auto untouched = all_zero_engine;
  const std::array<double, 3> all_zero{ 0.0, 0.0, 0.0 };
  REQUIRE_THROWS_AS(dtwc::core::portable_weighted_index(
                      all_zero.begin(), all_zero.end(), 0.0, all_zero_engine),
                    std::invalid_argument);
  REQUIRE(all_zero_engine == untouched);

  std::mt19937_64 sample_engine(42);
  REQUIRE(dtwc::core::portable_sample_indices<int>(10, 4, sample_engine)
          == std::vector<int>{ 3, 5, 7, 8 });
  // The literal pins the raw engine state after the complete scan; its final
  // singleton bounded call consumes no engine word.
  REQUIRE(sample_engine() == 7199227068870524257ULL);

  std::mt19937_64 int_engine(42);
  std::mt19937_64 int64_engine(42);
  const auto first_int = dtwc::core::portable_sample_indices<int>(
    10, 4, int_engine);
  const auto first_int64 = dtwc::core::portable_sample_indices<std::int64_t>(
    10, 4, int64_engine);
  const auto second_int = dtwc::core::portable_sample_indices<int>(
    10, 4, int_engine);
  const auto second_int64 = dtwc::core::portable_sample_indices<std::int64_t>(
    10, 4, int64_engine);
  REQUIRE(first_int == std::vector<int>{ 3, 5, 7, 8 });
  REQUIRE(first_int64 == std::vector<std::int64_t>{ 3, 5, 7, 8 });
  REQUIRE(second_int == std::vector<int>{ 0, 1, 8, 9 });
  REQUIRE(second_int64 == std::vector<std::int64_t>{ 0, 1, 8, 9 });
  REQUIRE(int_engine == int64_engine);
  REQUIRE(int_engine() == 863363284242328609ULL);
  REQUIRE(int64_engine() == 863363284242328609ULL);

  std::mt19937_64 empty_sample_engine(42);
  const auto empty_untouched = empty_sample_engine;
  REQUIRE(dtwc::core::portable_sample_indices<int>(10, 0, empty_sample_engine)
            .empty());
  REQUIRE(empty_sample_engine == empty_untouched);
  REQUIRE_THROWS_AS(dtwc::core::portable_sample_indices<int>(
                      3, 4, empty_sample_engine),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(dtwc::core::portable_sample_indices<int>(
                      -1, 0, empty_sample_engine),
                    std::invalid_argument);
  REQUIRE_THROWS_AS(dtwc::core::portable_sample_indices<int>(
                      3, -1, empty_sample_engine),
                    std::invalid_argument);

  std::mt19937_64 full_sample_engine(7);
  std::mt19937_64 full_sample_reference(7);
  std::vector<int> full_expected(10);
  std::iota(full_expected.begin(), full_expected.end(), 0);
  REQUIRE(dtwc::core::portable_sample_indices<int>(
            10, 10, full_sample_engine)
          == full_expected);
  for (int i = 0; i < 9; ++i) (void)full_sample_reference();
  REQUIRE(full_sample_engine == full_sample_reference);
}
