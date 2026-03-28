cmake_minimum_required(VERSION 3.21)

add_library(dtwc++ STATIC)

add_subdirectory(mip)

target_sources(dtwc++
  PRIVATE
  Problem.cpp
  Problem_IO.cpp
  initialisation.cpp
  scores.cpp
  core/dtw.cpp
  algorithms/fast_pam.cpp
  core/pruned_distance_matrix.cpp
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
  warping_ddtw.hpp
  warping_wdtw.hpp
  warping_adtw.hpp
  soft_dtw.hpp
  fileOperations.hpp
  DataLoader.hpp
  core/lower_bounds.hpp
  core/pruned_distance_matrix.hpp
)

target_include_directories(dtwc++ PUBLIC . ${range-v3_INCLUDE_DIRS})

target_link_libraries(dtwc++
  PRIVATE
  mip-solvers
  project_warnings
  project_options
  PUBLIC
  rapidcsv::rapidcsv
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message("OpenMP is found!")
  target_link_libraries(dtwc++ PUBLIC OpenMP::OpenMP_CXX)
endif()
