/**
 * @file test_build_parallelism.cpp
 * @brief Build-time parallelism guarantee (Task 3.1).
 *
 * @details Pins the no-silent-serial build contract enforced by the OpenMP gate
 *          in dtwc/CMakeLists.txt. A successfully-built dtwc++ must be in EXACTLY
 *          one of two states, both PUBLIC compile definitions on the dtwc++
 *          target (and therefore visible to any consumer that links it):
 *            - DTWC_HAS_OPENMP        : OpenMP found, parallel build.
 *            - DTWC_SEQUENTIAL_BUILD  : explicit -DDTWC_ALLOW_SEQUENTIAL=ON opt-out.
 *          Neither-defined means a silent serial build slipped through (forbidden);
 *          both-defined is a contradiction. The configure ABORTS (FATAL_ERROR)
 *          before any library is produced when OpenMP is missing and the opt-out
 *          is not given, so in any built library exactly one holds.
 *
 *          This test is a CONSUMER of dtwc++ (it links it, exactly like the
 *          nanobind module / dtwc_cl do). It therefore also verifies the hole-5
 *          fix: the OpenMP compile flag is now a PUBLIC usage requirement of
 *          dtwc++, so when DTWC_HAS_OPENMP is set this TU must itself be compiled
 *          with the OpenMP flag (`_OPENMP` defined) — the property that
 *          dtwc::get_max_threads() (parallelisation.hpp, which keys off `_OPENMP`)
 *          relies on to actually run parallel in every consumer.
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#include <catch2/catch_test_macros.hpp>

#include <dtwc.hpp>

// ── Compile-time contract enforcement ────────────────────────────────────────
#if defined(DTWC_HAS_OPENMP) && defined(DTWC_SEQUENTIAL_BUILD)
#  error "DTWC_HAS_OPENMP and DTWC_SEQUENTIAL_BUILD are mutually exclusive — a build cannot be both parallel and sequential."
#endif

#if !defined(DTWC_HAS_OPENMP) && !defined(DTWC_SEQUENTIAL_BUILD)
#  error "Neither DTWC_HAS_OPENMP nor DTWC_SEQUENTIAL_BUILD is defined — this is a silent serial build, forbidden by the no-silent-fallback contract. Missing OpenMP must FATAL at configure unless -DDTWC_ALLOW_SEQUENTIAL=ON is given (which defines DTWC_SEQUENTIAL_BUILD)."
#endif

// Hole-5 propagation guard: dtwc++ carries the OpenMP flag as a PUBLIC usage
// requirement, so a consumer that links dtwc++ (this TU) must be compiled with
// the OpenMP flag whenever the library reports OpenMP.
#if defined(DTWC_HAS_OPENMP) && !defined(_OPENMP)
#  error "dtwc++ reports DTWC_HAS_OPENMP but this consumer TU was compiled without the OpenMP flag (_OPENMP undefined) — the PUBLIC OpenMP usage requirement is not propagating to consumers (hole-5 regression)."
#endif

TEST_CASE("Build-time parallelism contract: exactly one mode active", "[parallelisation][build]")
{
  constexpr bool has_openmp =
#ifdef DTWC_HAS_OPENMP
    true;
#else
    false;
#endif
  constexpr bool sequential =
#ifdef DTWC_SEQUENTIAL_BUILD
    true;
#else
    false;
#endif

  // Exactly one of the two build modes must hold: never neither (silent serial),
  // never both (contradiction).
  REQUIRE(has_openmp != sequential);
}

#ifdef DTWC_HAS_OPENMP
TEST_CASE("OpenMP flag propagates to dtwc++ consumers (_OPENMP defined here)", "[parallelisation][build]")
{
  // If this compiled at all, the #error guard above already passed; assert at
  // runtime too so ctest records a positive check of the hole-5 fix.
  const bool openmp_flag_in_consumer =
#ifdef _OPENMP
    true;
#else
    false;
#endif
  REQUIRE(openmp_flag_in_consumer);
}
#endif
