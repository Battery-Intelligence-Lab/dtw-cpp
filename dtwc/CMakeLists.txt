cmake_minimum_required(VERSION 3.21)

add_library(dtwc++ STATIC)

add_subdirectory(mip)

target_sources(dtwc++
  PRIVATE
  Problem.cpp
  initialisation.cpp
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
)

target_include_directories(dtwc++ PUBLIC . ${range-v3_INCLUDE_DIRS})

target_link_libraries(dtwc++
  PRIVATE
  mip-solvers
  project_warnings
  project_options
  PUBLIC 
  eigen
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(dtwc++ PUBLIC OpenMP::OpenMP_CXX)
endif()

#set_property(TARGET dtwc++ PROPERTY CXX_STANDARD 17)