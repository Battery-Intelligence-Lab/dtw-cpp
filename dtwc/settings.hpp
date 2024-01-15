/**
 * @file settings.hpp
 * @brief This file contains settings and configurations for DTWC++ library.
 *
 * It includes settings for data types, random number generation, filesystem paths,
 * debugging options, and default algorithmic settings.
 *
 * @date 21 Jan 2022
 * @author Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include "enums/enums.hpp"

#include <string>
#include <filesystem>
#include <iostream>
#include <random>

namespace dtwc {
// Data type settings:

/// @brief Alias for the default data type used throughout the code.
/// @note The default data type can be either double or float, depending on precision requirements.
using data_t = double;

// Random number settings:

/// @brief Mersenne Twister random number generator.
/// @details This random number generator is used for all random number generation in the code.
///          The seed value is fixed to 29 for reproducibility.
///          To use a non-deterministic seed, replace '29' with 'std::random_device{}()'.
static std::mt19937 randGenerator(29);
} // namespace dtwc


namespace dtwc::settings {
// Filesystem settings:

/// @brief Namespace alias for std::filesystem.
namespace fs = std::filesystem;

/// @brief Path to the root folder.
/// @details This path is used to define subdirectories for data and results.
///          The root folder path is obtained from the DTWC_ROOT_FOLDER macro
///          which leads to the path of the top-level CMakeLists.txt.
const static fs::path root_folder(DTWC_ROOT_FOLDER);

/// @brief Path to the results directory.
/// @details Concatenates the root folder path with a subdirectory named "results".
const auto resultsPath = root_folder / "results/";

/// @brief Path to the data directory.
/// @details Concatenates the root folder path with a subdirectory named "data".
const auto dataPath = root_folder / "data";

/// @brief Flag for debug mode for developers.
/// @details When set to true, the program may output additional debug information.
constexpr bool isDebug = false;

/// @brief Default band length.
/// @details If no band is required, this value should be set to -1.
constexpr int DEFAULT_BAND_LENGTH = -1;

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