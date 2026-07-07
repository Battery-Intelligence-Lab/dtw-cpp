/**
 * @file test_runtime_loudness_gpu.cpp
 * @brief Runtime loudness — un-gated GPU->CPU fallback warnings (Task 3.2, wave A).
 *
 * @details Drives the LIVE public entry point `dtwc::Problem::fill_distance_matrix()`
 * (Problem.cpp). Requesting a GPU distance-matrix strategy on a build with no GPU
 * backend compiled in must ALWAYS warn to stderr before falling back to the CPU —
 * the warning is NOT gated by `verbose` (no-silent-fallback global constraint).
 *
 * The baseline build has neither DTWC_HAS_CUDA nor DTWC_HAS_METAL, so:
 *   - strategy=Metal  -> emits exactly the "Metal not compiled in ..." line, then
 *                        falls back to CPU brute-force.
 *   - strategy=CUDA   -> the CUDA case emits its line, then falls through the
 *                        (also absent) Metal case, so BOTH un-compiled lines print.
 * Both cases run with verbose=false to prove the gate does not suppress the warning.
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#include <Data.hpp>
#include <Problem.hpp>

#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <iostream>
#include <sstream>
#include <streambuf>
#include <string>
#include <vector>

// ===========================================================================
// Registered expected values — transcribed VERBATIM from dtwc/Problem.cpp
// (fill_distance_matrix, the CUDA/Metal `#else` branches). Declared BEFORE any
// run so the assertions are judged against a pre-declared band.
// ===========================================================================

static const std::string kMsgCudaNotCompiled =
  "CUDA not compiled in, falling back to CPU brute-force.\n";
static const std::string kMsgMetalNotCompiled =
  "Metal not compiled in, falling back to CPU brute-force.\n";

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

// Small in-memory Problem (no DataLoader, no repo-relative path; no NaN so the
// default missing_strategy=Error pre-scan passes and CPU brute-force succeeds).
static dtwc::Problem make_tiny_problem()
{
  std::vector<std::vector<dtwc::data_t>> vecs;
  std::vector<std::string> names;
  for (int i = 0; i < 4; ++i) {
    vecs.push_back({ double(i), double(i + 1), double(i + 2), double(i + 3) });
    names.push_back("s" + std::to_string(i));
  }
  dtwc::Problem prob("loudness_gpu");
  prob.set_data(dtwc::Data(std::move(vecs), std::move(names)));
  return prob;
}

// ===========================================================================
// Metal strategy on a non-Metal build: warns to stderr even with verbose=false.
// ===========================================================================

TEST_CASE("fill_distance_matrix(strategy=Metal) warns on a non-GPU build (verbose=false)", "[loudness][gpu]")
{
  auto prob = make_tiny_problem();
  prob.distance_strategy = dtwc::DistanceMatrixStrategy::Metal;
  prob.verbose = false; // the gate must NOT suppress the fallback warning

  const std::string captured = capture_cerr([&] { prob.fill_distance_matrix(); });

#if defined(DTWC_HAS_METAL)
  SUCCEED("Metal compiled in — real GPU dispatch path, no un-compiled fallback warning expected");
#else
  REQUIRE(captured == kMsgMetalNotCompiled);
#endif
  REQUIRE(prob.is_distance_matrix_filled()); // fell back to CPU and actually computed
}

// ===========================================================================
// CUDA strategy on a non-CUDA build: warns to stderr even with verbose=false.
// In the baseline build (no CUDA AND no Metal) the CUDA `#else` line prints and
// then control falls through the absent Metal case, printing that line too.
// ===========================================================================

TEST_CASE("fill_distance_matrix(strategy=CUDA) warns on a non-GPU build (verbose=false)", "[loudness][gpu]")
{
  auto prob = make_tiny_problem();
  prob.distance_strategy = dtwc::DistanceMatrixStrategy::CUDA;
  prob.verbose = false;

  const std::string captured = capture_cerr([&] { prob.fill_distance_matrix(); });

#if defined(DTWC_HAS_CUDA) || defined(DTWC_HAS_METAL)
  SUCCEED("GPU backend compiled in — real dispatch path exercised; message content is build-specific");
#else
  REQUIRE(captured == kMsgCudaNotCompiled + kMsgMetalNotCompiled);
#endif
  REQUIRE(prob.is_distance_matrix_filled());
}
