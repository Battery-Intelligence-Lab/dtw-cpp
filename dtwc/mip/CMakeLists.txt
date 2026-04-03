cmake_minimum_required(VERSION 3.21)
add_library(mip-solvers OBJECT)
set_target_properties(mip-solvers PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_sources(mip-solvers
  PRIVATE
    mip_Gurobi.cpp
    mip_Highs.cpp
    benders.cpp
  PUBLIC
    mip.hpp
    benders.hpp
  )

target_include_directories(mip-solvers PUBLIC .)
target_link_libraries(mip-solvers
  PRIVATE
  rapidcsv::rapidcsv
  Eigen3::Eigen
  project_warnings
  project_options
)

if(DTWC_ENABLE_GUROBI AND TARGET Gurobi::GurobiCXX)
  message(STATUS "Gurobi found and enabled")
  target_compile_definitions(mip-solvers PUBLIC DTWC_ENABLE_GUROBI)
  target_link_libraries(mip-solvers PRIVATE Gurobi::GurobiCXX)
endif()

if(DTWC_ENABLE_HIGHS AND TARGET highs::highs)
  target_compile_definitions(mip-solvers PUBLIC DTWC_ENABLE_HIGHS)
  target_link_libraries(mip-solvers PRIVATE highs::highs)
endif()
