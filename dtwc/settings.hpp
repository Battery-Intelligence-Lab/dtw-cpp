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

#include <string>
#include <filesystem>
#include <iostream>
#include <random>

namespace dtwc {
// Data type settings:

namespace settings {
/// @brief Default scalar type for public templated APIs.
/// @details This controls default template arguments such as
///          `template <typename T = settings::default_data_t>`.
using default_data_t = float;
} // namespace settings

/// @brief Alias for the core storage / internal precision type.
/// @note This is still distinct from `settings::default_data_t`, which controls
///       default template arguments on public distance helpers.
using data_t = double;

// Random number settings:

/// @brief Mersenne Twister random number generator.
/// @details This random number generator is used for all random number generation in the code.
///          The seed value is fixed to 29 for reproducibility.
///          To use a non-deterministic seed, replace '29' with 'std::random_device{}()'.
inline std::mt19937 randGenerator(29);
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
///          Can be set at runtime via setDataPath() or direct assignment.
inline fs::path data = fs::path{ "." } / "data";

/// @brief Path to the results/output directory.
/// @details Defaults to "./results/" relative to current working directory.
///          Can be set at runtime via setResultsPath() or direct assignment.
inline fs::path results = fs::path{ "." } / "results/";

/// @brief Set the data directory path.
/// @param path New path (as fs::path).
inline void setDataPath(const fs::path &path) { data = path; }

/// @brief Set the data directory path from C-string.
/// @param path New path (as C-string).
inline void setDataPath(const char *path) { data = fs::path(path); }

/// @brief Set the results directory path.
/// @param path New path (as fs::path).
inline void setResultsPath(const fs::path &path) { results = path; }

/// @brief Set the results directory path from C-string.
/// @param path New path (as C-string).
inline void setResultsPath(const char *path) { results = fs::path(path); }

} // namespace paths

/// @brief Flag for debug mode for developers.
/// @details When set to true, the program may output additional debug information.
constexpr bool isDebug = false;

/// @brief Default band length.
/// @details If no band is required, this value should be set to -1.
constexpr int DEFAULT_BAND = -1;

// Default settings:

/// @brief Default mixed-integer programming solver.
/// @note Please do not modify here, you can modify the relevant Problem class member to use a different MIP solver.
constexpr dtwc::Solver DEFAULT_MIP_SOLVER = dtwc::Solver::HiGHS;

/// @brief Default method for clustering.
/// @note Please do not modify here, you can modify the relevant Problem class member to use a different clustering method.
constexpr dtwc::Method DEFAULT_CLUSTERING_METHOD = dtwc::Method::Kmedoids;

/// @brief Default maximum number of iterations.
/// @details Used in iterative algorithms where a limit on iterations is necessary.
constexpr int DEFAULT_MAX_ITER = 100;
} // namespace dtwc::settings
