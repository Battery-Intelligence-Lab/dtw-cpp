# Changelog {#changelog}

[TOC]

This changelog contains a non-exhaustive list of new features and notable bug-fixes (not all bug-fixes will be listed).


<br/><br/>
# Unreleased

## New features
* Added `rapidcsv` library for robust multi-column CSV parsing.
* New functions in fileOperations.hpp:
  - `readCSV()` - Read multi-column CSV files into 2D vectors
  - `readTimeSeriesCSV()` - Read CSV files where each row is a time series
  - `readCSVColumn()` - Read a single column from a CSV file

## Notable Bug-fixes
* **CRITICAL**: Fixed `readFile()` in fileOperations.hpp where `std::runtime_error` was constructed but not thrown, causing silent failures when files could not be opened.
* Fixed `load_batch_file()` to throw proper `std::runtime_error` instead of throwing an integer (`throw 2`).
* Fixed signed/unsigned mismatch in parallelisation.hpp where `int` loop variable was compared against `size_t` bound.
* Fixed `numMaxParallelWorkers` parameter in `run()` function which was previously ignored; now properly sets OpenMP thread count.

## Bug-fixes (CMake)
* Fixed `option()` syntax in root CMakeLists.txt — added missing description strings.
* Changed `DTWC_ENABLE_GUROBI` and `DTWC_ENABLE_HIGHS` from `set()` to `option()` so they can be overridden by parent projects.
* Removed unconditional `enable_testing()` calls; `include(CTest)` handles this internally.
* Wrapped standalone executables (`dtwc_main`, `dtwc_cl`) in `if(PROJECT_IS_TOP_LEVEL)` guard so the project can be consumed as a subdirectory without building executables.
* Fixed misleading build-type status message (said "RelWithDebInfo" but actually set "Release").
* Changed runtime output directories from `CMAKE_CURRENT_SOURCE_DIR` to `CMAKE_BINARY_DIR` to prevent source-tree pollution.
* Updated Armadillo dependency from 12.6.x to 14.2.x branch, and added `if(NOT TARGET armadillo)` guard.
* Added `CONFIGURE_DEPENDS` to test source file glob for automatic re-globbing on file changes.

## Improvements
* OpenMP is now properly optional with `#ifdef _OPENMP` guards and serial fallback.
* Added bounds checking for loop indices to prevent undefined behavior with very large iteration counts.
* Added unit tests for exception throwing on missing files.
* Removed compile-time path dependencies (`DTWC_ROOT_FOLDER`, `CURRENT_ROOT_FOLDER`). Paths now default to current working directory and can be configured at runtime via `Problem::output_folder` or explicit `DataLoader` paths.

## API changes
* **New**: Added `settings::paths` namespace with runtime-configurable paths:
  - `settings::paths::dataPath` - Data directory (default: `./data`)
  - `settings::paths::resultsPath` - Results directory (default: `./results/`)
  - `settings::paths::setDataPath(path)` - Set data path at runtime
  - `settings::paths::setResultsPath(path)` - Set results path at runtime
* **Deprecated**: `settings::resultsPath`, `settings::dataPath`, `settings::dtwc_dataPath` are now deprecated aliases pointing to the new `settings::paths::` variables.

## Dependency updates
* Added `rapidcsv` v8.84 (BSD 3-Clause license) for CSV parsing.

## Developer updates
* Added cpp-style.md documenting C++ coding conventions.
* Added python-style.md documenting Python coding conventions.
* Added TODO.md for development task tracking.
* Removed `DTWC_ROOT_FOLDER` and `CURRENT_ROOT_FOLDER` CMake cache variables and compile definitions.

<br/><br/>
# DTWC v1.0.0

## New features
* HiGHS solver is added for open-source alternative to Gurobi (which is now not necessary for compilation and can be enabled by necessary flags). 
* Command line interface is added. 
* Documentation is improved (Doxygen website).

## Notable Bug-fixes
* Sakoe-Chiba band implementation is now more accurate. 

## API changes
* Replaced `VecMatrix<data_t>` class with `arma::Mat<data_t>`. 

## Dependency updates:
* Required C++ standard is reduced from C++20 to C++17 as it was causing `call to consteval function 'std::chrono::hh_mm_ss::_S_fractional_width' is not a constant expression` error for clang versions older than clang-15.
* `OpenMP` for parallelisation is adopted as `Apple-clang` does not support `std::execution`. 

## Developer updates: 
* The software is now being tested via Catch2 library. 
* Dependabot is added. 
* `CURRENT_ROOT_FOLDER` and `DTWC_ROOT_FOLDER` are seperated as DTW-C++ library can be included by other libraries. 

<br/><br/>
# DTWC v0.3.0

## New features
* UCR_test_2018 data integration for benchmarking. 

## Notable Bug-fixes
* N/A

## API changes
* DataLoader class is added for data reading. 
* `settings::resultsPath` is changed with `out_folder` member variable to have more flexibility. 
* `get_name` function added to remove `settings::writeAsFileNames` repetition)
* `std::filesystem::path operator+` was unnecessary and removed. 

<br/><br/>
# DTWC v0.2.0

A user interface is created for other people's use. 

## New features / updates
- Scores file with silhouette score is added. 
- `dtwFull_L` (L = light) is added for reducing memory requirements substantially.  

## API changes
- Problem class for a better interface. 
- `mip.hpp` and `mip.cpp` files are created to contain MIP functions.

## Notable Bug-fixes
* Gurobi better path finding in macOS. 
* TBB could not be used in macOS so it is now option with alternative thread-based parallelisation. 
* Time was showing wrong on macOS with std::clock. Therefore, moved to chrono library.

## Formatting: 
- Include a clang-format file. 

## Dependency updates
  * Required C++ standard is upgraded from C++17 to C++20. 

<br/><br/>
# DTWC v0.1.0

This is the initial release of DTWC. 

## Features
- Iterative algortihms for K-means and K-medoids 
- Mixed-integer programming solution support via YALMIP/MATLAB. 
- Support for `*.csv` files generated by Pandas.  

## Dependencies
  * A compiler with C++17 support. 
  * We require at least CMake 3.16. 