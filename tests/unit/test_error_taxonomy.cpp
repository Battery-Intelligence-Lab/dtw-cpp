/**
 * @file test_error_taxonomy.cpp
 * @brief Unit tests for the dtwc::Error exception taxonomy (dtwc/base/error.hpp).
 *
 * @details Covers, for Error and every derived type (InvalidInput, SolverError,
 * DeviceError, IOError):
 *   - constructibility from a message and what() preservation,
 *   - catchability as dtwc::Error and as std::runtime_error/std::exception, and
 *   - that sibling types are distinct (an InvalidInput is not a SolverError).
 *
 * LIVE code-path coverage: the taxonomy types are thrown by the sites they were migrated to. The
 * migrated site exercised here is the public, core (no HiGHS/Gurobi/CUDA needed)
 * entry point dtwc::soft_dtw_gradient() in dtwc/soft_dtw.hpp. Its former
 * `assert(mx > 0 && my > 0)` was a no-op under NDEBUG that let an empty span
 * fall through to an out-of-bounds read of x[0]/y[0]; it is now
 * `throw dtwc::InvalidInput(...)`. The final TEST_CASE drives that function with
 * an empty series and asserts the new type + message, pinning the real throw
 * rather than a constructed one. (The MIP status checks — mip_Highs.cpp /
 * mip_Gurobi.cpp — also throw dtwc::SolverError, but they require an optional
 * solver dependency, so the core path is exercised here instead.)
 *
 * One more table (the typed-throw sweep): for every file whose bare
 * `throw std::` sites became typed errors, a live, user-reachable site in it must
 * raise the type the error contract names. Arrow/Parquet rows run only in a
 * build with those readers; CUDA is pinned in test_cuda_correctness.cpp.
 *
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#include <DataLoader.hpp>
#include <Problem.hpp>
#include <algorithms/fast_pam.hpp>
#include <algorithms/hierarchical.hpp>
#include <algorithms/tadpole.hpp>
#include <base/error.hpp>
#include <base/missing_utils.hpp>
#include <checkpoint.hpp>
#include <core/distance_sampling_weights.hpp>
#include <core/matrix_io.hpp>
#include <initialisation.hpp>
#include <io/read_data.hpp>
#include <scores.hpp>
#include <soft_dtw.hpp>

#include "../support/scratch_directory.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// ===========================================================================
// Constructibility + what() preservation
// ===========================================================================

TEST_CASE("Error taxonomy: constructible from message, what() preserved", "[error]")
{
  REQUIRE(std::string(dtwc::Error("base failure").what()) == "base failure");
  REQUIRE(std::string(dtwc::InvalidInput("bad input").what()) == "bad input");
  REQUIRE(std::string(dtwc::SolverError("solver failed").what()) == "solver failed");
  REQUIRE(std::string(dtwc::DeviceError("device failed").what()) == "device failed");
  REQUIRE(std::string(dtwc::IOError("io failed").what()) == "io failed");
}

// ===========================================================================
// Catchability: every derived type is-a dtwc::Error, std::runtime_error, exc.
// ===========================================================================

TEST_CASE("Error taxonomy: InvalidInput catchable up the hierarchy", "[error]")
{
  REQUIRE_THROWS_AS(throw dtwc::InvalidInput("x"), dtwc::InvalidInput);
  REQUIRE_THROWS_AS(throw dtwc::InvalidInput("x"), dtwc::Error);
  REQUIRE_THROWS_AS(throw dtwc::InvalidInput("x"), std::runtime_error);
  REQUIRE_THROWS_AS(throw dtwc::InvalidInput("x"), std::exception);
}

TEST_CASE("Error taxonomy: SolverError catchable up the hierarchy", "[error]")
{
  REQUIRE_THROWS_AS(throw dtwc::SolverError("x"), dtwc::SolverError);
  REQUIRE_THROWS_AS(throw dtwc::SolverError("x"), dtwc::Error);
  REQUIRE_THROWS_AS(throw dtwc::SolverError("x"), std::runtime_error);
  REQUIRE_THROWS_AS(throw dtwc::SolverError("x"), std::exception);
}

TEST_CASE("Error taxonomy: DeviceError catchable up the hierarchy", "[error]")
{
  REQUIRE_THROWS_AS(throw dtwc::DeviceError("x"), dtwc::DeviceError);
  REQUIRE_THROWS_AS(throw dtwc::DeviceError("x"), dtwc::Error);
  REQUIRE_THROWS_AS(throw dtwc::DeviceError("x"), std::runtime_error);
}

TEST_CASE("Error taxonomy: IOError catchable up the hierarchy", "[error]")
{
  REQUIRE_THROWS_AS(throw dtwc::IOError("x"), dtwc::IOError);
  REQUIRE_THROWS_AS(throw dtwc::IOError("x"), dtwc::Error);
  REQUIRE_THROWS_AS(throw dtwc::IOError("x"), std::runtime_error);
}

// ===========================================================================
// Sibling types are distinct — catching one does not catch another.
// ===========================================================================

TEST_CASE("Error taxonomy: siblings are distinct types", "[error]")
{
  bool caught_as_invalid_input = false;
  try {
    throw dtwc::SolverError("not an input error");
  } catch (const dtwc::InvalidInput &) {
    caught_as_invalid_input = true;
  } catch (const dtwc::Error &) {
    // Correct: a SolverError is a dtwc::Error but NOT a dtwc::InvalidInput.
  }
  REQUIRE_FALSE(caught_as_invalid_input);
}

// ===========================================================================
// LIVE migrated site: dtwc::soft_dtw_gradient (dtwc/soft_dtw.hpp), a public
// core entry point. Empty input formerly hit assert(mx>0 && my>0) (UB under
// NDEBUG); it now throws dtwc::InvalidInput. This pins the real throw.
// ===========================================================================

TEST_CASE("soft_dtw_gradient: empty series throws dtwc::InvalidInput (live path)", "[error][soft_dtw]")
{
  const std::vector<double> empty{};
  const std::vector<double> y{ 1.0, 2.0, 3.0 };

  // Catchable as the new type, as the base, and as std::runtime_error.
  REQUIRE_THROWS_AS(dtwc::soft_dtw_gradient<double>(empty, y), dtwc::InvalidInput);
  REQUIRE_THROWS_AS(dtwc::soft_dtw_gradient<double>(empty, y), dtwc::Error);
  REQUIRE_THROWS_AS(dtwc::soft_dtw_gradient<double>(y, empty), std::runtime_error);

  // Message content: names the function and the reason (non-empty required).
  try {
    (void)dtwc::soft_dtw_gradient<double>(empty, y);
    FAIL("soft_dtw_gradient did not throw on empty input");
  } catch (const dtwc::InvalidInput &e) {
    const std::string what = e.what();
    REQUIRE(what.find("soft_dtw_gradient") != std::string::npos);
    REQUIRE(what.find("non-empty") != std::string::npos);
  }
}

// ===========================================================================
// One live site per converted file raises its contract type. The oracle
// is the error contract — bad input or configuration is InvalidInput, a
// file, stream or filesystem failure is IOError — not the old bare std:: type.
// ===========================================================================

namespace {

namespace fs = std::filesystem;

/// Name the most specific contract type `call` raised; anything else is named
/// with its message, so a failing row says what came out instead.
std::string raised(const std::function<void()> &call)
{
  try {
    call();
  } catch (const dtwc::InvalidInput &) {
    return "InvalidInput";
  } catch (const dtwc::IOError &) {
    return "IOError";
  } catch (const dtwc::DeviceError &) {
    return "DeviceError";
  } catch (const dtwc::SolverError &) {
    return "SolverError";
  } catch (const std::exception &e) {
    return std::string("untyped: ") + e.what();
  }
  return "no exception";
}

dtwc::Problem three_series()
{
  dtwc::Problem prob("gt4_typed_throws");
  prob.set_data(dtwc::Data(std::vector<std::vector<double>>{ { 0, 1, 2 }, { 1, 2, 3 }, { 5, 6, 7 } },
                           std::vector<std::string>{ "a", "b", "c" }));
  prob.set_verbose(false);
  return prob;
}

fs::path write_file(const fs::path &directory, const std::string &name, const std::string &text)
{
  std::ofstream(directory / name, std::ios::binary) << text;
  return directory / name;
}

} // namespace

TEST_CASE("GT-4: each converted file raises its contract type from a live site", "[error][gt4]")
{
  const dtwc::test_support::ScratchDirectory dir{ "gt4_typed_throws" };
  const fs::path bad_csv = write_file(dir.path, "bad.csv", "1,2,3\n4,abc,6\n");
  const fs::path garbage = write_file(dir.path, "garbage.bin", "not a cache");
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double inf = std::numeric_limits<double>::infinity();

  struct Row
  {
    const char *site;
    const char *expected;
    std::function<void()> call;
  };
  const std::vector<Row> rows{
    { "Data.hpp: series and names differ in count", "InvalidInput",
      [] { (void)dtwc::Data(std::vector<std::vector<double>>{ { 1.0 } }, std::vector<std::string>{}); } },
    { "Problem.cpp: set_clusters with the wrong count", "InvalidInput",
      [] {
        auto prob = three_series();
        prob.set_n_clusters(2);
        std::vector<dtwc::index_t> one{ 0 };
        prob.set_clusters(one);
      } },
    { "hierarchical.cpp: dendrogram over another point count", "InvalidInput",
      [] {
        auto prob = three_series();
        (void)dtwc::algorithms::cut_dendrogram(dtwc::algorithms::Dendrogram{}, prob, 1);
      } },
    { "tadpole.cpp: zero clusters", "InvalidInput",
      [] {
        auto prob = three_series();
        (void)dtwc::algorithms::tadpole(prob, 0, 1.0);
      } },
    { "missing_utils.hpp: interpolate an all-NaN series", "InvalidInput",
      [nan] {
        std::vector<double> buffer;
        (void)dtwc::interpolate_linear_into(std::vector<double>{ nan, nan }, buffer);
      } },
    { "distance_sampling_weights.hpp: a non-finite distance", "InvalidInput",
      [inf] { (void)dtwc::core::distance_sampling_weights(std::vector<double>{ inf, 1.0 }, {}, "gt4"); } },
    { "initialisation.cpp: more clusters than series", "InvalidInput",
      [] {
        dtwc::Problem empty("gt4_empty");
        dtwc::init::random(empty);
      } },
    { "scores.cpp: label vectors of different length", "InvalidInput",
      [] { (void)dtwc::scores::adjusted_rand({ 0, 1 }, { 0 }); } },
    { "checkpoint.cpp: a checkpoint of no series", "InvalidInput",
      [&dir] { dtwc::save_checkpoint(dtwc::Problem("gt4_empty"), (dir.path / "ckpt").string()); } },
    { "matrix_io.hpp: a distance matrix that does not exist", "IOError",
      [&dir] {
        dtwc::core::DistanceMatrix matrix;
        dtwc::io::read_csv(matrix, dir.path / "missing.csv");
      } },
    { "fileOperations.hpp: a non-numeric field", "IOError",
      [&bad_csv] { (void)dtwc::DataLoader(bad_csv).load(); } },
    // Without llfio map() is IOError too.
    { "distance_matrix.cpp: a file that is not a .dtwm matrix", "IOError",
      [&garbage] { (void)dtwc::core::DistanceMatrix::map(garbage, 0, {}); } },
#ifdef DTWC_HAS_MMAP
    { "Problem_IO.cpp: a CSV read into a mapped cache", "InvalidInput",
      [&dir] {
        auto prob = three_series();
        prob.use_mmap_distance_matrix(dir.path / "cache.dtwcm");
        prob.read_distance_matrix(dir.path / "matrix.csv");
      } },
#endif
#ifdef DTWC_HAS_ARROW
    { "read_data.cpp: an Arrow IPC file that does not exist", "IOError",
      [&dir] { (void)dtwc::read_data(dir.path / "missing.arrow"); } },
#endif
#ifdef DTWC_HAS_PARQUET
    { "read_data.cpp: a Parquet file that does not exist", "IOError",
      [&dir] { (void)dtwc::read_data(dir.path / "missing.parquet"); } },
#endif
  };

  for (const auto &row : rows) {
    INFO(row.site);
    CHECK(raised(row.call) == row.expected);
  }
}
