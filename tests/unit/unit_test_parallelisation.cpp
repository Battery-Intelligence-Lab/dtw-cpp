/**
 * @file unit_test_parallelisation.cpp
 * @brief Unit test file for parallelisation functions
 *
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 * @date 16 Dec 2023
 */

#include <dtwc.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <vector>
#include <atomic>

TEST_CASE("Parallel Execution", "[run_std]")
{
  std::vector<int> results(100, 0);
  auto task = [&](int i) { results[i] = 1; };

  dtwc::run_std(task, results.size(), true);

  for (int res : results)
    REQUIRE(res == 1);
}

TEST_CASE("Sequential Execution", "[run_std]")
{
  std::vector<int> results(100, 0);
  auto task = [&](int i) { results[i] = 1; };

  dtwc::run_std(task, results.size(), false);

  for (int res : results) {
    REQUIRE(res == 1);
  }
}

TEST_CASE("Functionality of run", "[run]")
{
  std::vector<int> results(100, 0);
  auto task = [&](int i) { results[i] = 1; };

  // Test with parallel execution
  dtwc::run(task, results.size(), 32);
  for (int res : results) {
    REQUIRE(res == 1);
  }

  // Reset and test with sequential execution
  std::fill(results.begin(), results.end(), 0);
  dtwc::run(task, results.size(), 1);
  for (int res : results) {
    REQUIRE(res == 1);
  }
}

TEST_CASE("Correct Number of Iterations", "[run_std]")
{
  std::atomic<int> count = 0;
  auto task = [&](int) { count++; };

  dtwc::run_std(task, 50, true);
  REQUIRE(count == 50);
}

TEST_CASE("Boundary Conditions", "[run_std]")
{
  int count = 0;
  auto task = [&](int) { count++; };

  dtwc::run_std(task, 0, true);
  REQUIRE(count == 0);
}