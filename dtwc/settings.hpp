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
#include <vector>
#include <array>
#include <random>

#define USE_STD_PAR_ALGORITMHS FALSE // On some computer it does not create thread so we are bound to manual thread creating.

namespace dtwc {
// Data type settings:
using data_t = double;      // Data type double or float.
using ind_t = unsigned int; // Data type for indices.
// Random number settings:
static std::mt19937 randGenerator(5); // std::mt19937{ std::random_device{}() }
// Future settings
// namespace stdr = std::ranges;
// namespace stdv = std::views;
} // namespace dtwc


namespace dtwc::settings {

// Filesystem settings:
namespace fs = std::filesystem;

const static fs::path root_folder = "../";

const auto resultsPath = root_folder / "results/";
const auto dataPath = root_folder / "data";

constexpr bool isParallel = true;
constexpr int numMaxParallelWorkers = 1024; // Change accordingly more cores than your computer has. It is limited to the maximum physical cores.
constexpr bool writeAsFileNames = true;     // If true it writes the file names as report. Otherwise numbers.

constexpr bool isDebug = false;

// MIP solution settings. Default = false, false;
constexpr bool is_relaxed = false;

constexpr int band = 0; // Size of band to use (if no band put 0)


constexpr bool debug_Simplex = false;


} // namespace dtwc::settings