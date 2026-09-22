/**
 * @file settings.hpp
 * @brief This file contains settings and configurations for DTWC++ library.
 *
 * @details It includes settings for data types, random number generation, filesystem paths,
 * debugging options, and default algorithmic settings.
 *
 * @date 21 Jan 2022
 * @author Volkan Kumtepeli
 * @author Becky Perriment
 */

#pragma once

#include "enums/enums.hpp"

#include <cstdint>
// <string> is unused here; kept because 24 TUs (incl. examples/ and
// benchmarks/, which are OFF in every configured gate) use std::string
// without including it, so removal cannot be build-proven from here.
#include <string>
#include <filesystem>
// <iostream> removed (C-21a). The five translation units that were relying on it
// transitively -- benchmarks/UCR_dtwc.cpp, examples/cpp/example_project/main.cpp,
// dtwc/mip/mip_Gurobi.cpp, tests/unit/core/unit_test_pruned_distance_matrix.cpp
// and tests/unit/test_storage_policy.cpp -- now include it themselves, and the
// removal was proven by a build with DTWC_BUILD_BENCHMARK=ON and
// DTWC_BUILD_EXAMPLES=ON, since benchmarks and examples are off in the
// canonical gate and a green build without them would prove nothing.
// <random> left with dtwc::randGenerator for dtwc/random_engine.hpp (X-12).

namespace dtwc {
// Data type settings:

namespace settings {
/// @brief Default scalar type for public templated APIs.
/// @details This controls default template arguments such as
///          `template <typename T = settings::default_data_t>`.
using default_data_t = double;
} // namespace settings

/// @brief Alias for the core storage / internal precision type.
/// @note As of DTWC++ 2.0 this coincides in type with `settings::default_data_t`
///       (both `double`); the names stay separate by role — `data_t` is the
///       internal storage/accumulation type, `settings::default_data_t` the
///       default template argument on public distance helpers.
using data_t = double;

// Random number settings: dtwc::randGenerator moved to dtwc/random_engine.hpp
// (X-12), which dtwc/dtwc.hpp includes, so the public name is unaffected.
} // namespace dtwc


namespace dtwc::settings {
// Filesystem settings:

/// @brief Namespace alias for std::filesystem.
namespace fs = std::filesystem;

/// @brief Runtime-configurable paths for data and results directories.
/// @details Use these paths throughout the library. They can be set at runtime
///          via the setter functions or by direct assignment, enabling Python/MATLAB
///          bindings to configure paths before use.
namespace paths {

/// @brief Path to the data directory.
/// @details Defaults to "./data" relative to current working directory.
///          Can be set at runtime via set_data_path() or direct assignment.
inline fs::path data = fs::path{ "." } / "data";

/// @brief Path to the results/output directory.
/// @details Defaults to "./results/" relative to current working directory.
///          Can be set at runtime via set_results_path() or direct assignment.
inline fs::path results = fs::path{ "." } / "results/";

/// @brief Set the data directory path.
/// @param path New path (as fs::path).
inline void set_data_path(const fs::path &path) { data = path; }

/// @brief Set the data directory path from C-string.
/// @param path New path (as C-string).
inline void set_data_path(const char *path) { data = fs::path(path); }

/// @brief Set the results directory path.
/// @param path New path (as fs::path).
inline void set_results_path(const fs::path &path) { results = path; }

/// @brief Set the results directory path from C-string.
/// @param path New path (as C-string).
inline void set_results_path(const char *path) { results = fs::path(path); }

/// @brief Deprecated 1.x alias for set_data_path().
[[deprecated("use set_data_path")]]
inline void setDataPath(const fs::path &path) { set_data_path(path); }

/// @brief Deprecated 1.x C-string alias for set_data_path().
[[deprecated("use set_data_path")]]
inline void setDataPath(const char *path) { set_data_path(path); }

/// @brief Deprecated 1.x alias for set_results_path().
[[deprecated("use set_results_path")]]
inline void setResultsPath(const fs::path &path) { set_results_path(path); }

/// @brief Deprecated 1.x C-string alias for set_results_path().
[[deprecated("use set_results_path")]]
inline void setResultsPath(const char *path) { set_results_path(path); }

} // namespace paths

/// @brief Flag for debug mode for developers.
/// @details When set to true, the program may output additional debug information.
constexpr bool isDebug = false;

/// Invocation-local default seed for deterministic Tier-1 sampling algorithms.
///
/// The mutable `dtwc::randGenerator` (now in random_engine.hpp) is a legacy
/// Tier-2 facility whose
/// seed remains 29 for source/behaviour compatibility.  Tier-1 entry points do
/// not read, reseed, or otherwise consume that process-global engine.
inline constexpr std::uint64_t DEFAULT_RANDOM_SEED = 42;

/// @brief Default band length.
/// @details If no band is required, this value should be set to -1.
constexpr int DEFAULT_BAND = -1;

// Default settings:

/// @brief Default mixed-integer programming solver.
/// @note Please do not modify here, you can modify the relevant Problem class member to use a different MIP solver.
constexpr dtwc::Solver DEFAULT_MIP_SOLVER = dtwc::Solver::HiGHS;

/// @brief Default method for clustering.
/// @note Please do not modify here; use Problem::set_method() to select a different clustering method.
constexpr dtwc::Method DEFAULT_CLUSTERING_METHOD = dtwc::Method::Kmedoids;

/// @brief Default maximum number of iterations.
/// @details Used in iterative algorithms where a limit on iterations is necessary.
constexpr int DEFAULT_MAX_ITER = 100;
} // namespace dtwc::settings
