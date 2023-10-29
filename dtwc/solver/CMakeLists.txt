cmake_minimum_required(VERSION 3.16)
add_library(solvers OBJECT)

target_sources(solvers
  PRIVATE
    Simplex.cpp
    SparseSimplex.cpp
    SimplexTable.cpp
  PUBLIC
    solver_util.hpp 
    Simplex.hpp
    EqualityConstraints.hpp
    SparseSimplex.hpp
    EqualityConstraints.hpp
    SimplexTable.hpp
  )

target_include_directories(solvers PUBLIC .)
target_link_libraries(solvers PRIVATE eigen fmt)