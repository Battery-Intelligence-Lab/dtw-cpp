cmake_minimum_required(VERSION 3.16)

project(DTWC++ VERSION "0.0.4"
  DESCRIPTION "A C++ library for fast Dynamic Time Wrapping Clustering"
  HOMEPAGE_URL https://battery-intelligence-lab.github.io/dtw-cpp/
  LANGUAGES C CXX)

include(cmake/StandardProjectSettings.cmake)
include(cmake/PreventInSourceBuilds.cmake)
  
option(CXX "enable C++ compilation" ON)

if(CXX)
  enable_language(CXX)
endif()

set(DTWC_ENABLE_GUROBI ON)
set(DTWC_ENABLE_IWYU OFF)

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_20)

if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  option(ENABLE_BUILD_WITH_TIME_TRACE "Enable -ftime-trace to generate time tracing .json files on clang" OFF)

  if(ENABLE_BUILD_WITH_TIME_TRACE)
    add_compile_definitions(project_options INTERFACE -ftime-trace)
  endif()
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# # enable cache system
# include(cmake/Cache.cmake)
include(cmake/FindGUROBI.cmake)

# standard compiler warnings
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

include(cmake/Dependencies.cmake) # Include external projects
add_subdirectory(dtwc/solver)


add_executable(dtwc++
  dtwc/main.cpp
  dtwc/Problem.cpp
  dtwc/mip.cpp
  dtwc/mip_Gurobi.cpp
  dtwc/parallelisation.hpp
)

if(DTWC_ENABLE_IWYU)
  find_program(iwyu_path NAMES include-what-you-use iwyu REQUIRED)
  set_property(TARGET dtwc++ PROPERTY CXX_INCLUDE_WHAT_YOU_USE ${iwyu_path})
endif()

target_link_libraries(dtwc++
  PRIVATE
  project_warnings
  project_options
  solvers
  eigen
  range-v3
  fmt
  highs::highs
)
if(DTWC_ENABLE_GUROBI AND Gurobi_FOUND)
  message("Gurobi is found!")
  add_definitions(-DDTWC_ENABLE_GUROBI) 
  target_link_libraries(dtwc++ PRIVATE Gurobi::GurobiCXX)
endif()


target_include_directories(dtwc++ PRIVATE ${HIGHS_INCLUDE_DIRS} ${range-v3_INCLUDE_DIRS})
target_compile_definitions(dtwc++ PRIVATE ROOT_FOLDER="${PROJECT_SOURCE_DIR}")
