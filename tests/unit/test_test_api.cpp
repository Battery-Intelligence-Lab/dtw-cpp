/**
 * @file test_test_api.cpp
 * @brief Unit tests for the header-only `dtwc::test` introspection API (Task 3.3).
 *
 * @details Every test drives the LIVE public entry points
 * `dtwc::test::parallelisation()` and `dtwc::test::gpu()` (dtwc/test_api.hpp) —
 * the exact functions the Python (`dtwcpp.test.*`) and MATLAB
 * (`dtwc_mex('test_parallelisation'|'test_gpu')`) surfaces bind to.
 *
 * Registered expectations (written BEFORE the run, judged after):
 *   - Baseline build here is OpenMP-ON + CUDA-OFF (build/baseline-2026-07-06):
 *       parallelisation() -> available, pass, threads_engaged >= 2 (24-core host);
 *       gpu()             -> unavailable branch, reason non-empty, never throws.
 *   - The SAME test is correct under a sequential build (DTWC_SEQUENTIAL_BUILD)
 *     and under Task 3.5's CUDA-ON build: it branches on the reported `available`
 *     flag so both the available and unavailable paths are asserted honestly.
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#include <test_api.hpp>

#include <catch2/catch_test_macros.hpp>

// ===========================================================================
// parallelisation() — proof-of-engagement, not a flag read.
// ===========================================================================

TEST_CASE("dtwc::test::parallelisation reports real thread engagement", "[test_api][parallel]")
{
  const auto r = dtwc::test::parallelisation();

  INFO("available=" << r.available << " max_threads=" << r.max_threads
       << " threads_engaged=" << r.threads_engaged << " pass=" << r.pass
       << " reason=" << r.reason);

  // Never throws; the report is always self-consistent.
  REQUIRE(r.max_threads >= 1);
  REQUIRE(r.threads_engaged >= 1);

#if defined(DTWC_SEQUENTIAL_BUILD)
  // Sequential build: honest loud "unavailable" answer, never a fake pass.
  REQUIRE_FALSE(r.available);
  REQUIRE_FALSE(r.reason.empty());
  REQUIRE_FALSE(r.pass);
#else
  // Baseline registered band (OpenMP build, 24-core host).
  REQUIRE(r.available);
  REQUIRE(r.reason.empty());
  if (r.max_threads >= 2) {
    // The load-bearing assertion: a REAL parallel region engaged >= 2 threads.
    REQUIRE(r.threads_engaged >= 2);
    REQUIRE(r.threads_engaged <= r.max_threads);
    REQUIRE(r.pass);
  } else {
    // Genuine single-thread host: engaging the one available thread passes.
    REQUIRE(r.pass);
  }
#endif
}

// ===========================================================================
// gpu() — execute-and-validate, or a loud reason. Never throws, never silent.
// ===========================================================================

TEST_CASE("dtwc::test::gpu validates against a CPU oracle or names what is missing",
          "[test_api][gpu]")
{
  const auto r = dtwc::test::gpu();

  INFO("available=" << r.available << " backend=" << r.backend
       << " device_name=" << r.device_name << " validated=" << r.validated
       << " pass=" << r.pass << " reason=" << r.reason);

  if (r.available) {
    // GPU backend compiled AND a device present (Task 3.5's CUDA build): the
    // tiny kernel MUST match the CPU oracle within tolerance.
    REQUIRE_FALSE(r.backend.empty());
    REQUIRE_FALSE(r.device_name.empty());
    REQUIRE(r.validated);
    REQUIRE(r.pass);
  } else {
    // Unavailable branch (baseline CUDA-OFF here): no throw, no silent degrade —
    // just a non-empty reason naming exactly what is missing.
    REQUIRE_FALSE(r.reason.empty());
    REQUIRE_FALSE(r.validated);
    REQUIRE_FALSE(r.pass);
  }
}
