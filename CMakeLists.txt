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

set(DTWC_ENABLE_GUROBI ON)
include(cmake/Dependencies.cmake) # Include external projects
include(cmake/FindGUROBI.cmake)
# include(cmake/ProjectOptions.cmake)

include(cmake/PreventInSourceBuilds.cmake)
dtwc_setup_dependencies()


# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_20)

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
)

# dtwc_setup_options()

# dtwc_global_options()
# include(cmake/Dependencies.cmake) # Include external projects
# dtwc_setup_dependencies()

# dtwc_local_options()

# # don't know if this should be set globally from here or not... (I also don't know)
# set(CMAKE_CXX_VISIBILITY_PRESET hidden)

# target_compile_features(dtwc_options INTERFACE cxx_std_${CMAKE_CXX_STANDARD})

# add_library(dtwc::dtwc_options ALIAS dtwc_options)
# add_library(dtwc::dtwc_warnings ALIAS dtwc_warnings)

# add_subdirectory(dtwc)

# # Don't even look at tests if we're not top level
# if(NOT PROJECT_IS_TOP_LEVEL)
#   return()
# endif()

# Adding the tests:
include(CTest)
enable_testing()
if(BUILD_TESTING)
  #message(AUTHOR_WARNING "Building Tests. Be sure to check out test/constexpr_tests.cpp for constexpr testing")
  include(cmake/Coverage.cmake)
  add_subdirectory(tests)
endif()


# # If MSVC is being used, and ASAN is enabled, we need to set the debugger environment
# # so that it behaves well with MSVC's debugger, and we can run the target from visual studio
# if(MSVC)
#   get_all_installable_targets(all_targets)
#   message("all_targets=${all_targets}")
#   set_target_properties(${all_targets} PROPERTIES VS_DEBUGGER_ENVIRONMENT "PATH=$(VC_ExecutablePath_x64);%PATH%")
# endif()

# # set the startup project for the "play" button in MSVC
# set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT dtwc_main)
