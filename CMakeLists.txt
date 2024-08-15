cmake_minimum_required(VERSION 3.21)

# This CMake file is adapted from https://github.com/cpp-best-practices/cmake_template

# This template attempts to be "fetch_content"-able
# so that it works well with tools like CPM or other
# manual dependency management

project(DTWC++ VERSION "1.0.0"
  DESCRIPTION "A C++ library for fast Dynamic Time Wrapping Clustering"
  HOMEPAGE_URL https://battery-intelligence-lab.github.io/dtw-cpp/
  LANGUAGES CXX C)

set(DTWC_ROOT_FOLDER "${CMAKE_SOURCE_DIR}" CACHE STRING "Root folder for DTWC project") 

include(cmake/StandardProjectSettings.cmake)

option(DTWC_BUILD_EXAMPLES OFF)
option(DTWC_BUILD_TESTING OFF)
option(DTWC_BUILD_BENCHMARK OFF)

set(DTWC_ENABLE_GUROBI ON)
set(DTWC_ENABLE_HIGHS ON)
include(cmake/Dependencies.cmake) # Include external projects
include(cmake/FindGUROBI.cmake)

include(cmake/PreventInSourceBuilds.cmake)
dtwc_setup_dependencies()

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_17)

enable_testing()

add_subdirectory(dtwc bin)


add_executable(dtwc_main
  dtwc/main.cpp
)

target_link_libraries(dtwc_main
  PRIVATE
  dtwc++
  project_warnings
  project_options
  armadillo
)

add_executable(dtwc_cl
  dtwc/dtwc_cl.cpp
)

target_link_libraries(dtwc_cl
  PRIVATE
  dtwc++
  CLI11::CLI11
  project_warnings
  project_options
)

# Adding the tests:
include(CTest)
enable_testing()
if(DTWC_BUILD_TESTING)
  include(cmake/Coverage.cmake)
  add_subdirectory(tests)
endif()

if(DTWC_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(DTWC_BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()

cpm_licenses_create_disclaimer_target(write-licenses "${CMAKE_CURRENT_SOURCE_DIR}/bin/third_party.txt" "${CPM_PACKAGES}")