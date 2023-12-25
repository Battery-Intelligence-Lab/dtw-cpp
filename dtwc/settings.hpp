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

using data_t = double; // Default data type double or float.
// Random number settings:
static std::mt19937 randGenerator(29); // std::mt19937{ std::random_device{}() }
} // namespace dtwc


namespace dtwc::settings {

// Filesystem settings:
namespace fs = std::filesystem; //!< std::filesystem alias.

const static fs::path root_folder(DTWC_ROOT_FOLDER); //!< Root folder for defining data and results folders.

const auto resultsPath = root_folder / "results/";
const auto dataPath = root_folder / "data";

constexpr bool isDebug = false;

constexpr int band = 0; // Size of band to use (if no band put 0)


// Default settings:
constexpr dtwc::Solver DEFAULT_MIP_SOLVER = dtwc::Solver::HiGHS;
constexpr dtwc::Method DEFAULT_CLUSTERING_METHOD = dtwc::Method::Kmedoids;
constexpr int DEFAULT_MAX_ITER = 100;
} // namespace dtwc::settings