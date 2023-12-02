cmake_minimum_required(VERSION 3.16)
add_library(solvers OBJECT)

target_sources(solvers
  PRIVATE
    Simplex.cpp
    SparseSimplex.cpp
    SimplexTable.cpp
    SimplexRowTable.cpp
    SimplexFlatRowTable.cpp
    SimplexFlatTable.cpp

  PUBLIC
    solver_util.hpp 
    Simplex.hpp
    EqualityConstraints.hpp
    SparseSimplex.hpp
    EqualityConstraints.hpp
    SimplexTable.hpp
    SimplexRowTable.hpp
    SimplexFlatRowTable.hpp
    SimplexFlatTable.hpp

  )

target_include_directories(solvers PUBLIC .)
target_link_libraries(solvers PRIVATE eigen fmt range-v3 highs::highs)