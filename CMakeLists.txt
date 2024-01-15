cmake_minimum_required(VERSION 3.21)

# This CMake file is adapted from https://github.com/cpp-best-practices/cmake_template

# This template attempts to be "fetch_content"-able
# so that it works well with tools like CPM or other
# manual dependency management

project(DTWC++ VERSION "0.0.4"
  DESCRIPTION "A C++ library for fast Dynamic Time Wrapping Clustering"
  HOMEPAGE_URL https://battery-intelligence-lab.github.io/dtw-cpp/
  LANGUAGES CXX C)

# Define the root path
add_definitions(-DDTWC_ROOT_FOLDER="${CMAKE_SOURCE_DIR}")
include(cmake/StandardProjectSettings.cmake)

option(BUILD_EXAMPLES OFF)
option(BUILD_TESTING OFF)
option(BUILD_BENCHMARK OFF)

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
if(BUILD_TESTING)
  #message(AUTHOR_WARNING "Building Tests. Be sure to check out test/constexpr_tests.cpp for constexpr testing")
  include(cmake/Coverage.cmake)
  add_subdirectory(tests)
endif()


if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

if(BUILD_BENCHMARK)
  add_subdirectory(benchmark)
endif()