cmake_minimum_required(VERSION 3.16)
add_library(solvers OBJECT)

target_sources(solvers
  PRIVATE
    Simplex.cpp
    SparseSimplex.cpp
  PUBLIC
    solver_util.hpp 
    Simplex.hpp
    SparseSimplex.hpp
  )

target_include_directories(solvers PUBLIC .)
target_link_libraries(solvers PRIVATE eigen fmt)