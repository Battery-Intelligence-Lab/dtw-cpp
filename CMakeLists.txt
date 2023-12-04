cmake_minimum_required(VERSION 3.21)

# This CMake file is adapted from https://github.com/cpp-best-practices/cmake_template

# This template attempts to be "fetch_content"-able
# so that it works well with tools like CPM or other
# manual dependency management

# Only set the cxx_standard if it is not set by someone else
if (NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 20)
endif()

# strongly encouraged to enable this globally to avoid conflicts between
# -Wpedantic being enabled and -std=c++20 and -std=gnu++20 for example
# when compiling with PCH enabled
set(CMAKE_CXX_EXTENSIONS OFF)

project(DTWC++ VERSION "0.0.4"
  DESCRIPTION "A C++ library for fast Dynamic Time Wrapping Clustering"
  HOMEPAGE_URL https://battery-intelligence-lab.github.io/dtw-cpp/
  LANGUAGES CXX C)

include(cmake/ProjectOptions.cmake)
include(cmake/PreventInSourceBuilds.cmake)

dtwc_setup_options()

dtwc_global_options()
include(cmake/Dependencies.cmake) # Include external projects
dtwc_setup_dependencies()

dtwc_local_options()

# don't know if this should be set globally from here or not... (I also don't know)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)

target_compile_features(dtwc_options INTERFACE cxx_std_${CMAKE_CXX_STANDARD})

add_library(dtwc::dtwc_options ALIAS dtwc_options)
add_library(dtwc::dtwc_warnings ALIAS dtwc_warnings)

add_subdirectory(dtwc)

# Don't even look at tests if we're not top level
if(NOT PROJECT_IS_TOP_LEVEL)
  return()
endif()

# # Adding the tests:
# include(CTest)

# if(BUILD_TESTING)
#   #message(AUTHOR_WARNING "Building Tests. Be sure to check out test/constexpr_tests.cpp for constexpr testing")
#   include(cmake/Coverage.cmake)
#   add_subdirectory(tests)
# endif()


# If MSVC is being used, and ASAN is enabled, we need to set the debugger environment
# so that it behaves well with MSVC's debugger, and we can run the target from visual studio
if(MSVC)
  get_all_installable_targets(all_targets)
  message("all_targets=${all_targets}")
  set_target_properties(${all_targets} PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=$(VC_ExecutablePath_x64);%PATH%")
endif()

# set the startup project for the "play" button in MSVC
set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT dtwc_main)
