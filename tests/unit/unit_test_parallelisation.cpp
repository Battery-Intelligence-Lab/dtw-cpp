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
#include <chrono>
#include <stdexcept>
#include <thread>

#ifdef _OPENMP
#include <omp.h>
#endif

TEST_CASE("Parallel Execution", "[run_openmp]")
{
  std::vector<int> results(100, 0);
  auto task = [&](size_t i) { results[i] = 1; };

  dtwc::run_openmp(task, results.size(), true);

  for (int res : results)
    REQUIRE(res == 1);
}

TEST_CASE("Sequential Execution", "[run_openmp]")
{
  std::vector<int> results(100, 0);
  auto task = [&](size_t i) { results[i] = 1; };

  dtwc::run_openmp(task, results.size(), false);

  for (int res : results) {
    REQUIRE(res == 1);
  }
}

TEST_CASE("Functionality of run", "[run]")
{
  std::vector<int> results(100, 0);
  auto task = [&](size_t i) { results[i] = 1; };

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

TEST_CASE("numMaxParallelWorkers controls thread count", "[run]")
{
  std::vector<int> results(100, 0);
  auto task = [&](size_t i) { results[i] = 1; };

  // Test with different worker counts
  dtwc::run(task, results.size(), 2);
  for (int res : results) {
    REQUIRE(res == 1);
  }

  // Reset and test with 4 workers
  std::fill(results.begin(), results.end(), 0);
  dtwc::run(task, results.size(), 4);
  for (int res : results) {
    REQUIRE(res == 1);
  }
}

TEST_CASE("Correct Number of Iterations", "[run_openmp]")
{
  std::atomic<int> count = 0;
  auto task = [&](size_t) { count++; };

  dtwc::run_openmp(task, 50, true);
  REQUIRE(count == 50);
}

TEST_CASE("Boundary Conditions", "[run_openmp]")
{
  int count = 0;
  auto task = [&](size_t) { count++; };

  dtwc::run_openmp(task, 0, true);
  REQUIRE(count == 0);
}

TEST_CASE("OpenMP task failures rethrow the lowest-index typed exception",
          "[run_openmp][m40]")
{
  auto run_failures = [] {
    std::atomic<bool> row_seven_failed{false};
    const auto deadline = std::chrono::steady_clock::now()
                        + std::chrono::seconds(5);
    auto task = [&](size_t i) {
      if (i == 2) {
#ifdef _OPENMP
        // Force the higher row to fail first in wall-clock order. Correctness
        // must still select row 2 by canonical loop index.
        while (!row_seven_failed.load(std::memory_order_acquire))
        {
          if (std::chrono::steady_clock::now() >= deadline)
            throw std::runtime_error("test coordination timeout");
          std::this_thread::yield();
        }
#endif
        throw std::invalid_argument("failure at row 2");
      }
      if (i == 7) {
        row_seven_failed.store(true, std::memory_order_release);
        throw std::runtime_error("failure at row 7");
      }
    };
    // 64 chunks/thread makes chunk=1 for this 64-row fixture, so rows 2 and 7
    // cannot be trapped sequentially inside one worker's chunk.
    dtwc::run_openmp(task, 64, true, 64);
  };

#ifdef _OPENMP
  const int previous_threads = omp_get_max_threads();
  const int previous_dynamic = omp_get_dynamic();
  struct RestoreOpenMP {
    int threads;
    int dynamic;
    ~RestoreOpenMP() { omp_set_dynamic(dynamic); omp_set_num_threads(threads); }
  } restore{previous_threads, previous_dynamic};
  omp_set_dynamic(0);
  omp_set_num_threads(2);
  int actual_threads = 1;
#pragma omp parallel
  {
#pragma omp single
    actual_threads = omp_get_num_threads();
  }
  if (actual_threads < 2) SKIP("two OpenMP workers unavailable");
#endif

  REQUIRE_THROWS_AS(run_failures(), std::invalid_argument);
  REQUIRE_THROWS_WITH(run_failures(), "failure at row 2");

  REQUIRE_THROWS_WITH(dtwc::omp_chunk_size(64, 0),
                      "omp_chunk_size: chunks_per_thread must be positive");
  REQUIRE_THROWS_WITH(dtwc::omp_chunk_size(64, -1),
                      "omp_chunk_size: chunks_per_thread must be positive");
  auto no_op = [](size_t) {};
  REQUIRE_THROWS_WITH(dtwc::run_openmp(no_op, 64, true, 0),
                      "run_openmp: chunks_per_thread must be positive");
}
