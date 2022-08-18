// Vk: 2022.01.21

#pragma once

#include <string>
#include <filesystem>


using Tdata = float; // Data type double or float.


inline std::filesystem::path operator+(const std::filesystem::path &lhs, const std::string &rhs)
{ //!< To make path type compatible with strings.
  const std::filesystem::path temp{ rhs };
  return lhs / temp;
}


namespace settings {

namespace fs = std::filesystem;

const static fs::path root_folder = ROOT_FOLDER;

const auto resultsPath = root_folder + "results/";
constexpr bool isParallel = true;

constexpr int numMaxParallelWorkers = 32;
constexpr bool writeAsFileNames = true;

constexpr bool isDebug = false;


const auto path = root_folder + "data/dummy";
constexpr int Ndata_max = 50; // Maximum number of files loaded.
constexpr int band = 0;       // Size of band to use (if no band put 0)
} // namespace settings