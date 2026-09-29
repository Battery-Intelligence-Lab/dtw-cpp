cmake_minimum_required(VERSION 3.26)
add_library(mip-solvers OBJECT)
set_target_properties(mip-solvers PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_sources(mip-solvers
  PRIVATE
    mip_Gurobi.cpp
    mip_Highs.cpp
    solution_transaction.cpp
    warm_start.cpp
    benders.cpp
    lagrangian_root.cpp
    reduced_cost_fixing.cpp
    pdlp_lp.cpp
  PUBLIC
    mip.hpp
    highs_support.hpp
    index_guard.hpp
    nearest_medoid.hpp
    solver_types.hpp
    solution_transaction.hpp
    warm_start.hpp
    benders.hpp
    lagrangian_root.hpp
    reduced_cost_fixing.hpp
    pdlp_lp.hpp
  )

target_include_directories(mip-solvers PUBLIC .)
target_link_libraries(mip-solvers
  PRIVATE
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
  # PDLP GPU backend compiled into HiGHS (CUPDLP_GPU). Drives pdlp_lp_bound's
  # honest gpu_used reporting: without this define, use_gpu raises DeviceError.
  if(DTWC_HIGHS_GPU)
    target_compile_definitions(mip-solvers PUBLIC DTWC_HIGHS_GPU)
  endif()
endif()
