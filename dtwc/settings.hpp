// Vk: 2022.01.21

#pragma once

#include <string>
#include <filesystem>

#define USE_STD_PAR_ALGORITMHS FALSE


using data_t = double;      // Data type double or float.
using ind_t = unsigned int; // Data type for indices.

inline std::filesystem::path operator+(const std::filesystem::path &lhs, const std::string &rhs)
{ //!< To make path type compatible with strings.
  const std::filesystem::path temp{ rhs };
  return lhs / temp;
}

namespace settings {

namespace fs = std::filesystem;

const static fs::path root_folder = "../../";

const auto resultsPath = root_folder + "results/";
constexpr bool isParallel = true;

constexpr int numMaxParallelWorkers = 1024; // Change accordingly more cores than your computer has. It is limited to the maximum physical cores.
constexpr bool writeAsFileNames = false;

constexpr bool isDebug = false;


const auto path = root_folder + "data/dummy";
constexpr int band = 0; // Size of band to use (if no band put 0)
} // namespace settings