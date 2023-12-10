cmake_minimum_required(VERSION 3.21)

add_library(dtwc++ STATIC)

target_sources(dtwc++
  PRIVATE
  Problem.cpp
  mip.cpp
  mip_Gurobi.cpp
  parallelisation.hpp
  PUBLIC
  Data.hpp
  DataLoader.hpp
  dtwc.hpp
  initialisation.hpp
  mip.hpp
  parallelisation.hpp
  Problem.hpp
  scores.hpp
  settings.hpp
  timing.hpp
  utility.hpp
  warping.hpp
)

target_include_directories(dtwc++ PUBLIC .)


if(DTWC_ENABLE_GUROBI)
  message(STATUS "Gurobi is enabled!")
  if(Gurobi_FOUND)
    message(STATUS "Gurobi is found!")
    add_definitions(-DDTWC_ENABLE_GUROBI) 
    target_link_libraries(dtwc++ PRIVATE Gurobi::GurobiCXX)
  endif()
endif()


target_include_directories(dtwc++ PRIVATE ${range-v3_INCLUDE_DIRS} )

target_link_libraries(dtwc++
  PRIVATE
  fmt
  highs::highs
  project_warnings
  project_options
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  target_link_libraries(dtwc++ PUBLIC OpenMP::OpenMP_CXX)
endif()

set_property(TARGET dtwc++ PROPERTY CXX_STANDARD 20)