/*
 * settings.hpp
 *
 * This file contains time warping functions
 *
 * Created on: 21 Jan 2022
 * Author(s): Volkan Kumtepeli, Becky Perriment
 */

#pragma once

#include <string>
#include <filesystem>
#include <iostream>
#include <random>

namespace dtwc {
// Data type settings:
using data_t = double; // Data type double or float.
// Random number settings:
static std::mt19937 randGenerator(5); // std::mt19937{ std::random_device{}() }
} // namespace dtwc


namespace dtwc::settings {

// Filesystem settings:
namespace fs = std::filesystem;

const static fs::path root_folder = DTWC_ROOT_FOLDER;

const auto resultsPath = root_folder / "results/";
const auto dataPath = root_folder / "data";

constexpr bool writeAsFileNames = true; // If true it writes the file names as report. Otherwise numbers.

constexpr bool isDebug = false;

constexpr int band = 0; // Size of band to use (if no band put 0)

} // namespace dtwc::settings