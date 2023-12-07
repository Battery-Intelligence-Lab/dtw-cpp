target_sources(dtwc++
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

target_include_directories(dtwc++ PUBLIC .)