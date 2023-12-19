cmake_minimum_required(VERSION 3.21)
add_library(mip-solvers OBJECT)

target_sources(mip-solvers
  PRIVATE
    mip_Gurobi.cpp
    mip_Highs.cpp
  PUBLIC
    mip.hpp
  )

target_include_directories(mip-solvers PUBLIC .)
target_link_libraries(mip-solvers PRIVATE eigen)

if(DTWC_ENABLE_GUROBI)
  message(STATUS "Gurobi is enabled!")
  if(Gurobi_FOUND)
    message(STATUS "Gurobi is found!")
    target_compile_definitions(mip-solvers PUBLIC DTWC_ENABLE_GUROBI)
    target_link_libraries(mip-solvers PRIVATE Gurobi::GurobiCXX)
  endif()
endif()

if(DTWC_ENABLE_HIGHS)
  target_compile_definitions(mip-solvers PUBLIC DTWC_ENABLE_HIGHS)
  target_link_libraries(mip-solvers PRIVATE highs::highs)
endif()