cmake_minimum_required(VERSION 3.21)

add_library(dtwc++ STATIC)
target_compile_definitions(dtwc++ PUBLIC DTWC_ROOT_FOLDER="${DTWC_ROOT_FOLDER}")

add_subdirectory(mip)

target_sources(dtwc++
  PRIVATE
  Problem.cpp
  Problem_IO.cpp
  initialisation.cpp
  scores.cpp
  PUBLIC
  Data.hpp
  DataLoader.hpp
  dtwc.hpp
  initialisation.hpp
  parallelisation.hpp
  Problem.hpp
  scores.hpp
  settings.hpp
  timing.hpp
  utility.hpp
  warping.hpp
  fileOperations.hpp
  DataLoader.hpp
)

target_include_directories(dtwc++ PUBLIC . ${range-v3_INCLUDE_DIRS})

target_link_libraries(dtwc++
  PRIVATE
  mip-solvers
  project_warnings
  project_options
  PUBLIC 
  armadillo
)

