/**
 * @file unit_test_run_thread_scope.cpp
 * @brief dtwc::run() must scope its worker limit to its own parallel region.
 *
 * @details run() used to call omp_set_num_threads(), which mutates PROCESS-WIDE
 * OpenMP state that is never restored: one k-means++ init at 2 workers pinned
 * every later distance fill to 2 threads, and pruned_distance_matrix.cpp derives
 * its block count from get_max_threads(), so PruningStats became call-order
 * dependent. The limit belongs in a num_threads(...) clause on the pragma.
 */

#include <parallelisation.hpp>

#include <catch2/catch_test_macros.hpp>

#include <atomic>
#include <cstddef>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

TEST_CASE("run() does not mutate process-wide OpenMP thread state",
          "[run][openmp][regression]")
{
#ifdef _OPENMP
  const int before = omp_get_max_threads();
  if (before < 2) {
    SUCCEED("machine reports a single OpenMP thread; nothing to constrain");
    return;
  }

  std::vector<int> results(64, 0);
  auto task = [&](std::size_t i) { results[i] = 1; };

  dtwc::run(task, results.size(), 2);   // ask for at most 2 workers
  REQUIRE(omp_get_max_threads() == before);

  dtwc::run(task, results.size(), 1);   // serial request
  REQUIRE(omp_get_max_threads() == before);

  for (const int r : results) REQUIRE(r == 1);

  // The limit must still be honoured inside the region it was requested for.
  std::atomic<int> observed_max{ 0 };
  auto probe = [&](std::size_t) {
    const int n = omp_get_num_threads();
    int seen = observed_max.load(std::memory_order_relaxed);
    while (n > seen
           && !observed_max.compare_exchange_weak(seen, n,
                                                  std::memory_order_relaxed,
                                                  std::memory_order_relaxed)) {}
  };
  dtwc::run(probe, 256, 2);
  REQUIRE(observed_max.load() <= 2);
  REQUIRE(omp_get_max_threads() == before);
#else
  SUCCEED("OpenMP disabled; run() is serial and mutates no global state");
#endif
}
