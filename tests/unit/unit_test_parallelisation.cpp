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

TEST_CASE("run_openmp rethrows a task's failure on the caller", "[run_openmp]")
{
  auto fails_at_37 = [](size_t i) {
    if (i == 37) throw std::invalid_argument("failure at row 37");
  };
  REQUIRE_THROWS_AS(dtwc::run_openmp(fails_at_37, 64, false), std::invalid_argument);
  REQUIRE_THROWS_WITH(dtwc::run_openmp(fails_at_37, 64, false), "failure at row 37");

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

  // 64 chunks per thread make chunk = 1 for 64 rows, so both threads take rows.
  REQUIRE_THROWS_AS(dtwc::run_openmp(fails_at_37, 64, true, 64), std::invalid_argument);
  REQUIRE_THROWS_WITH(dtwc::run_openmp(fails_at_37, 64, true, 64), "failure at row 37");

  // A thread runs no row after its first failure: with every row failing, each of
  // the two threads runs at most one.
  std::atomic<int> calls{ 0 };
  auto every_row_fails = [&](size_t) {
    ++calls;
    throw std::runtime_error("every row fails");
  };
  REQUIRE_THROWS_WITH(dtwc::run_openmp(every_row_fails, 64, true, 64), "every row fails");
  CHECK(calls >= 1);
  CHECK(calls <= 2);

#ifdef _OPENMP
  // A failure stored by the worker thread, not the caller's thread 0, is found:
  // thread 0 waits in its row until thread 1 has failed.
  std::atomic<bool> worker_failed{ false };
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  auto worker_fails = [&](size_t) {
    if (omp_get_thread_num() == 1) {
      worker_failed = true;
      throw std::invalid_argument("worker failed");
    }
    while (!worker_failed && std::chrono::steady_clock::now() < deadline) std::this_thread::yield();
  };
  REQUIRE_THROWS_WITH(dtwc::run_openmp(worker_fails, 64, true, 64), "worker failed");
#endif
}
