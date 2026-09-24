/**
 * @file test_problem_move_traits.cpp
 * @brief S-13: Problem's move assignment makes no noexcept promise.
 *
 * @details Moving Problem's members — the distance-matrix variant, the bound
 * dispatchers, the WDTW weight map — may throw. A move assignment declared
 * noexcept turned such a throw into std::terminate. The trait is pinned at
 * compile time; the case below checks that moving still carries the state.
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

static_assert(std::is_move_constructible_v<dtwc::Problem>);
static_assert(std::is_move_assignable_v<dtwc::Problem>);
static_assert(!std::is_nothrow_move_assignable_v<dtwc::Problem>,
              "Problem::operator=(Problem&&) must not be noexcept (S-13)");

TEST_CASE("S-13: Problem move assignment still transfers data and distances",
          "[problem][s13][move]")
{
  dtwc::Problem source("s13_source");
  source.set_data(dtwc::Data(std::vector<std::vector<double>>{ { 0, 1, 2 }, { 0, 2, 4 } },
                             std::vector<std::string>{ "a", "b" }));
  source.fill_distance_matrix();
  const double distance = source.dist_by_ind(0, 1);

  dtwc::Problem target("s13_target");
  target = std::move(source);
  CHECK(target.name() == "s13_source");
  CHECK(target.size() == 2);
  CHECK(target.is_distance_matrix_filled());
  CHECK(target.dist_by_ind(0, 1) == distance);
  CHECK(target.dtw_function()(target.series(0), target.series(1)) == distance);
}
