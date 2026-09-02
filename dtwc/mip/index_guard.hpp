/**
 * @file index_guard.hpp
 * @brief Backend-neutral model-dimension guard shared by every MIP entry point.
 *
 * @details The p-median models are O(N²) in columns, rows and nonzeros, and
 * every supported solver API indexes those with a 32-bit integer (`HighsInt` in
 * a default HiGHS build; the `int` count of `GRBModel::addVars`; the `int`
 * triplet fields this repo builds the HiGHS matrix from). Beyond N = 46340 that
 * truncates into a wrong-sized model, so reject it loudly instead.
 *
 * Pulls in no solver header, so it is usable from the Gurobi translation unit
 * and from a build with neither optional solver enabled.
 *
 * @author Volkan Kumtepeli
 * @date 02 Sep 2026
 */

#pragma once

#include "../error.hpp"

#include <cstddef>
#include <string>
#include <string_view>

namespace dtwc::mip {

/// Throw SolverError when a model dimension does not fit the backend's index type.
inline void require_index_range(
  std::size_t value,
  std::size_t max_value,
  std::string_view what,
  std::string_view backend)
{
  if (value > max_value)
    throw SolverError(std::string(backend) + ": " + std::string(what) + " = "
      + std::to_string(value) + " exceeds this backend's index limit ("
      + std::to_string(max_value)
      + "); the model cannot be passed without silent truncation.");
}

} // namespace dtwc::mip
