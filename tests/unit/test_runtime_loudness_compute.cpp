/**
 * @file test_runtime_loudness_compute.cpp
 * @brief Runtime loudness — single-thread warning on the COMPUTE entry points
 *        (Task 3.6, adversarial-review finding H1).
 *
 * @details Task 3.2 wired the single-thread warning into the dtwc::Env
 * constructor, but the compute hot paths — Problem::fill_distance_matrix, the
 * Python distance-matrix bindings, direct C++ Problem use — never construct
 * dtwc::env(). So on a build with OpenMP present but only 1 usable thread
 * (OMP_NUM_THREADS=1, common on SLURM/containers) the computation ran SILENTLY
 * single-threaded — a no-silent-fallback violation that survived Phase 3's own
 * gate. The fix routes every compute path through dtwc::get_max_threads(), which
 * now calls the shared process-once emitter dtwc::warn_if_single_threaded().
 *
 * This test drives a REAL compute path (fill_distance_matrix on 3 tiny series)
 * with NO dtwc::Env ever constructed, and asserts the loud warning still fires —
 * exactly once, and never again on a second compute call.
 *
 * Separate executable on purpose: the warning is process-once (a single
 * std::call_once guard SHARED with the Env constructor, so the CLI — which both
 * builds env() and computes — warns at most once). A dedicated test process gives
 * this test a fresh guard. The Env-constructor half of the guarantee is pinned in
 * test_runtime_loudness_env.cpp.
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#include <dtwc.hpp>
#include <env.hpp>
#include <parallelisation.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <functional>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <thread>
#include <vector>

#ifdef DTWC_HAS_OPENMP
#include <omp.h>
#endif

using namespace dtwc;

// ===========================================================================
// Registered expected values — transcribed VERBATIM from dtwc/env.cpp
// (detail::sequential_warning_text). Declared BEFORE any run so every assertion
// is judged against a pre-declared band, never a post-hoc one.
// ===========================================================================

static const std::string kMsgRuntimeSingleThread =
  "[DTWC++ WARNING] OpenMP is available but only 1 thread is usable — DTWC++ is running SINGLE-THREADED.\n"
  "  Distance-matrix computation will be extremely slow for large datasets.\n"
  "  Raise the thread count (unset OMP_NUM_THREADS, or set OMP_NUM_THREADS>1) to use all CPU cores.\n";

static const std::string kMsgSequentialBuild =
  "[DTWC++ WARNING] This build was compiled WITHOUT OpenMP (-DDTWC_ALLOW_SEQUENTIAL=ON) — DTWC++ is running SINGLE-THREADED.\n"
  "  Distance-matrix computation will be extremely slow for large datasets.\n"
  "  Rebuild without -DDTWC_ALLOW_SEQUENTIAL=ON (with OpenMP available) for parallel execution.\n";

// Run `fn` with std::cerr redirected into a buffer; return everything it wrote.
static std::string capture_cerr(const std::function<void()> &fn)
{
  std::ostringstream oss;
  std::streambuf *old = std::cerr.rdbuf(oss.rdbuf());
  try {
    fn();
  } catch (...) {
    std::cerr.rdbuf(old);
    throw;
  }
  std::cerr.rdbuf(old);
  return oss.str();
}

static std::size_t count_occurrences(const std::string &hay, const std::string &needle)
{
  if (needle.empty()) return 0;
  std::size_t n = 0, pos = 0;
  while ((pos = hay.find(needle, pos)) != std::string::npos) {
    ++n;
    pos += needle.size();
  }
  return n;
}

// Build a tiny Problem and fill its distance matrix — a real compute path that
// funnels through omp_chunk_size() -> get_max_threads(). Constructs NO Env.
static void run_tiny_fill()
{
  std::vector<std::vector<data_t>> vecs{
    { 0.0, 1.0, 2.0, 3.0 },
    { 1.0, 1.0, 1.0, 1.0 },
    { 3.0, 2.0, 1.0, 0.0 }
  };
  std::vector<std::string> names{ "a", "b", "c" };
  Data data(std::move(vecs), std::move(names));
  Problem prob("loudness_compute");
  prob.set_data(std::move(data));
  prob.fill_distance_matrix();
}

// ===========================================================================
// A compute path with only 1 usable thread warns loudly — WITHOUT any Env.
// ===========================================================================

TEST_CASE("compute path warns when single-threaded, without constructing Env", "[loudness][compute]")
{
#if defined(DTWC_SEQUENTIAL_BUILD)
  // No OpenMP (-DDTWC_ALLOW_SEQUENTIAL=ON): the compute loops run serially and do
  // not reach omp_chunk_size, but the shared get_max_threads() chokepoint still
  // emits the SequentialBuild warning on first use — exercise it directly.
  const std::string captured = capture_cerr([] { (void)dtwc::get_max_threads(); });
  REQUIRE(count_occurrences(captured, kMsgSequentialBuild) == 1);

  // Second use: the process-once guard is consumed — no repeat.
  const std::string again = capture_cerr([] { (void)dtwc::get_max_threads(); });
  REQUIRE(again.find(kMsgSequentialBuild) == std::string::npos);

#elif defined(DTWC_HAS_OPENMP)
  const int saved = omp_get_max_threads();
  omp_set_num_threads(1); // force effective threads == 1 (equivalent to OMP_NUM_THREADS=1)

  const std::string captured = capture_cerr([] { run_tiny_fill(); });
  // Second compute call must NOT repeat the warning (process-once guard consumed).
  const std::string again = capture_cerr([] { (void)dtwc::get_max_threads(); });

  omp_set_num_threads(saved); // restore for any sibling code in this process

  if (std::thread::hardware_concurrency() > 1) {
    REQUIRE(count_occurrences(captured, kMsgRuntimeSingleThread) == 1);
    REQUIRE(again.find(kMsgRuntimeSingleThread) == std::string::npos);
  } else {
    SUCCEED("single-core host: the runtime single-thread warning is intentionally suppressed");
  }

#else
  // No OpenMP and no sequential-build macro (unsanctioned post-Task-3.1): threads()
  // == 1 always; still warn via the shared chokepoint on a multicore host.
  const std::string captured = capture_cerr([] { (void)dtwc::get_max_threads(); });
  if (std::thread::hardware_concurrency() > 1)
    REQUIRE(count_occurrences(captured, kMsgRuntimeSingleThread) == 1);
  else
    SUCCEED("single-core host: the runtime single-thread warning is intentionally suppressed");
#endif
}
