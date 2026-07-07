/**
 * @file test_runtime_loudness_env.cpp
 * @brief Runtime loudness — Env single-thread warning (Task 3.2, wave A).
 *
 * @details Drives the LIVE public entry point `dtwc::Env` constructor (env.hpp /
 * env.cpp). When DTWC++ would run single-threaded — either forced to 1 usable
 * OpenMP thread on a multicore host, or compiled without OpenMP via
 * -DDTWC_ALLOW_SEQUENTIAL=ON (DTWC_SEQUENTIAL_BUILD) — the constructor emits ONE
 * loud warning to stderr (no-silent-fallback global constraint).
 *
 * Two layers of coverage:
 *   1. The PURE message/predicate SSOT `dtwc::detail::sequential_cause` +
 *      `sequential_warning_text` — asserts the exact strings byte-for-byte and
 *      every predicate branch, deterministically and build-config-independently.
 *   2. The LIVE constructor: `omp_set_num_threads(1)` forces the effective thread
 *      count to 1, then a `dtwc::Env` is constructed with std::cerr redirected and
 *      the captured text is asserted against the same registered string.
 *
 * Mechanism note (documented per task): the reliable forcing knob here is
 * `omp_set_num_threads(1)` in-process. This test executable constructs NO other
 * `dtwc::Env` (and never calls the `dtwc::env()` singleton), so the process-wide
 * `std::call_once` guard in the constructor is guaranteed fresh when the live test
 * runs — the warning fires deterministically. (`OMP_NUM_THREADS=1` in a subprocess
 * is the equivalent env-var mechanism; omp_set_num_threads is simpler and needs no
 * child process, so it is preferred here.)
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#include <env.hpp>

#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <thread>

#ifdef DTWC_HAS_OPENMP
#include <omp.h>
#endif

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

// ===========================================================================
// 1. PURE predicate + message SSOT (backs the LIVE dtwc::Env constructor).
//    Byte-for-byte string pins + every branch of detail::sequential_cause.
// ===========================================================================

TEST_CASE("Env sequential warning: message text + predicate SSOT", "[env][loudness]")
{
  using dtwc::detail::SeqCause;
  using dtwc::detail::sequential_cause;
  using dtwc::detail::sequential_warning_text;

  // Byte-for-byte message pins.
  REQUIRE(sequential_warning_text(SeqCause::RuntimeSingleThread) == kMsgRuntimeSingleThread);
  REQUIRE(sequential_warning_text(SeqCause::SequentialBuild) == kMsgSequentialBuild);
  REQUIRE(sequential_warning_text(SeqCause::None).empty());

  // Predicate branches (effective_max_threads, hw_concurrency, sequential_build).
  REQUIRE(sequential_cause(1, 8, false) == SeqCause::RuntimeSingleThread); // 1 thread on an 8-core host
  REQUIRE(sequential_cause(8, 8, false) == SeqCause::None);                // genuinely parallel → silent
  REQUIRE(sequential_cause(1, 1, false) == SeqCause::None);                // real single-core → no crying wolf
  REQUIRE(sequential_cause(1, 0, false) == SeqCause::None);                // unknown concurrency → silent
  REQUIRE(sequential_cause(8, 8, true) == SeqCause::SequentialBuild);      // sequential build dominates
  REQUIRE(sequential_cause(1, 8, true) == SeqCause::SequentialBuild);
}

// ===========================================================================
// 2. LIVE constructor: forcing 1 usable thread makes dtwc::Env() emit the exact
//    warning to stderr. This is the ONLY dtwc::Env constructed in this process,
//    so the constructor's process-once guard is fresh (see file header).
// ===========================================================================

TEST_CASE("dtwc::Env() emits the sequential warning under forced single-thread", "[env][loudness]")
{
#if defined(DTWC_SEQUENTIAL_BUILD)
  // Compiled without OpenMP via the escape hatch: the warning fires unconditionally.
  const std::string captured = capture_cerr([] { dtwc::Env e; });
  REQUIRE(captured == kMsgSequentialBuild);
#elif defined(DTWC_HAS_OPENMP)
  const int saved = omp_get_max_threads();
  omp_set_num_threads(1); // force effective threads == 1 (equivalent to OMP_NUM_THREADS=1)
  const std::string captured = capture_cerr([] { dtwc::Env e; });
  omp_set_num_threads(saved); // restore so later tests in any sibling target are unaffected

  if (std::thread::hardware_concurrency() > 1)
    REQUIRE(captured == kMsgRuntimeSingleThread);
  else
    SUCCEED("single-core host: the runtime single-thread warning is intentionally suppressed");
#else
  // No OpenMP and no sequential-build macro: threads() == 1 always.
  const std::string captured = capture_cerr([] { dtwc::Env e; });
  if (std::thread::hardware_concurrency() > 1)
    REQUIRE(captured == kMsgRuntimeSingleThread);
  else
    SUCCEED("single-core host: the runtime single-thread warning is intentionally suppressed");
#endif
}
