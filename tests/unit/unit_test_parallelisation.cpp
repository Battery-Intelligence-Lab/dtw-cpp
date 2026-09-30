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
#include <string>
#include <thread>

#ifdef _OPENMP
#include <omp.h>

namespace {

/// Asks the runtime for `threads` workers until the end of the scope; `actual`
/// is the team size it gave.
struct RequestWorkers {
  int previous_threads = omp_get_max_threads();
  int previous_dynamic = omp_get_dynamic();
  int actual = 1;

  explicit RequestWorkers(int threads)
  {
    omp_set_dynamic(0);
    omp_set_num_threads(threads);
#pragma omp parallel
    {
#pragma omp single
      actual = omp_get_num_threads();
    }
  }
  ~RequestWorkers()
  {
    omp_set_dynamic(previous_dynamic);
    omp_set_num_threads(previous_threads);
  }
};

} // namespace
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
  const RequestWorkers workers(2);
  if (workers.actual < 2) SKIP("two OpenMP workers unavailable");
#endif

  // 64 chunks per thread make chunk = 1 for 64 rows, so both threads take rows.
  REQUIRE_THROWS_AS(dtwc::run_openmp(fails_at_37, 64, true, 64), std::invalid_argument);
  REQUIRE_THROWS_WITH(dtwc::run_openmp(fails_at_37, 64, true, 64), "failure at row 37");

#ifdef _OPENMP
  // A thread runs no row above one of its own failures, so with every row failing
  // the rows it runs strictly decrease. It may still run a lower row: the schedule
  // is nonmonotonic, and skipping that row could lose the lowest failure.
  std::vector<std::vector<size_t>> ran(2);
  auto every_row_fails = [&](size_t i) {
    ran[static_cast<size_t>(omp_get_thread_num())].push_back(i);
    throw std::runtime_error("every row fails");
  };
  REQUIRE_THROWS_WITH(dtwc::run_openmp(every_row_fails, 64, true, 64), "every row fails");
  for (const auto &rows : ran)
    for (size_t k = 1; k < rows.size(); ++k) CHECK(rows[k] < rows[k - 1]);

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

TEST_CASE("run_openmp rethrows the failure with the lowest index", "[run_openmp]")
{
#ifdef _OPENMP
  const RequestWorkers workers(4);
  if (workers.actual < 2) SKIP("two OpenMP workers unavailable");

  // Every row from 40 on fails. Row 40 is the lowest, and its thread waits there
  // until another thread has failed, so a higher failure is stored before row
  // 40's: picking the first failure to arrive, or the first slot by thread
  // number, would not return row 40. It lies in the upper half of the rows so
  // that thread 0, which libomp starts on the lowest rows, does not hold it.
  constexpr size_t lowest = 40;
  for (int repeat = 0; repeat < 100; ++repeat) {
    std::atomic<int> failed_elsewhere{ 0 };
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
    auto rows_from_40_fail = [&](size_t i) {
      if (i < lowest) return;
      if (i == lowest) {
        while (failed_elsewhere == 0 && std::chrono::steady_clock::now() < deadline)
          std::this_thread::yield();
      } else {
        ++failed_elsewhere;
      }
      throw std::invalid_argument("row " + std::to_string(i));
    };
    REQUIRE_THROWS_WITH(dtwc::run_openmp(rows_from_40_fail, 64, true, 16), "row 40");
  }
#else
  SKIP("built without OpenMP");
#endif
}

TEST_CASE("run_openmp runs the rows below a thread's own failure", "[run_openmp]")
{
#ifdef _OPENMP
  const RequestWorkers workers(2);
  if (workers.actual < 2) SKIP("two OpenMP workers unavailable");

  // libomp's dynamic schedule is nonmonotonic: a worker that has run its own half
  // of the rows steals rows below them from the other. Rows from 31 on fail and
  // the rows below take 1 ms, so the worker that failed at row 32 steals row 31
  // while the other is still near the bottom; skipping that row would report 32.
  auto slow_below_31 = [](size_t i) {
    if (i < 31) {
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
      return;
    }
    throw std::invalid_argument("row " + std::to_string(i));
  };
  REQUIRE_THROWS_WITH(dtwc::run_openmp(slow_below_31, 64, true, 64), "row 31");
#else
  SKIP("built without OpenMP");
#endif
}
