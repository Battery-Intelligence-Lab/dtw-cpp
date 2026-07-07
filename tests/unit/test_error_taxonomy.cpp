/**
 * @file test_error_taxonomy.cpp
 * @brief Unit tests for the dtwc::Error exception taxonomy (dtwc/error.hpp).
 *
 * @details Covers, for Error and every derived type (InvalidInput, SolverError,
 * DeviceError, IOError):
 *   - constructibility from a message and what() preservation,
 *   - catchability as dtwc::Error and as std::runtime_error/std::exception, and
 *   - that sibling types are distinct (an InvalidInput is not a SolverError).
 *
 * LIVE code-path coverage (see .claude/LESSONS.md "Tests must pin the LIVE code
 * path"): the taxonomy types are thrown by the sites migrated in Task 1.2. The
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
 * @author Volkan Kumtepeli
 * @date 07 Jul 2026
 */

#include <error.hpp>
#include <soft_dtw.hpp>

#include <catch2/catch_test_macros.hpp>

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
