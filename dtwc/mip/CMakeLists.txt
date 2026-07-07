cmake_minimum_required(VERSION 3.26)
add_library(mip-solvers OBJECT)
set_target_properties(mip-solvers PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_sources(mip-solvers
  PRIVATE
    mip_Gurobi.cpp
    mip_Highs.cpp
    benders.cpp
    lagrangian_root.cpp
    reduced_cost_fixing.cpp
  PUBLIC
    mip.hpp
    benders.hpp
    lagrangian_root.hpp
    reduced_cost_fixing.hpp
  )

target_include_directories(mip-solvers PUBLIC .)
target_link_libraries(mip-solvers
  PRIVATE
  rapidcsv::rapidcsv
  Eigen3::Eigen
  project_warnings
  project_options
)

# llfio (OPTIONAL) — memory-mapped distance matrices.
# Task R3 (optional-deps rule): llfio_hl used to be linked UNCONDITIONALLY into
# mip-solvers, so a no-optional-deps configuration (-DDTWC_ENABLE_LLFIO=OFF, or
# llfio simply absent) referenced a target that does not exist. Guard the link
# on the target, exactly as dtwc/CMakeLists.txt does for dtwc++.
#
# Why mip-solvers needs llfio at all: it does NOT use llfio directly (grep of
# mip_Gurobi.cpp / mip_Highs.cpp / benders.cpp finds no llfio symbol). It needs
# it TRANSITIVELY — every mip source includes ../Problem.hpp, which includes
# core/mmap_distance_matrix.hpp (pulls <llfio/v2.0/llfio.hpp>) and declares
# `distMat_t = std::variant<DenseDistanceMatrix, MmapDistanceMatrix>` as a member
# of Problem (Problem.hpp:88). So mip-solvers needs BOTH the llfio include path
# AND the same DTWC_HAS_MMAP definition as dtwc++ — otherwise Problem's layout
# would differ across translation units (ODR violation). Mirror dtwc++ exactly.
#
# Behaviour when OFF/absent: the link and the DTWC_HAS_MMAP define are dropped
# together, so mip-solvers builds against the same non-mmap Problem view as the
# rest of the core (no silent divergence). The user-facing mmap error path lives
# in Problem::use_mmap_distance_matrix (dtwc/Problem.cpp), not in mip/.
if(TARGET llfio_hl)
  target_link_libraries(mip-solvers PRIVATE llfio_hl)
  target_compile_definitions(mip-solvers PUBLIC DTWC_HAS_MMAP)
endif()

if(DTWC_ENABLE_GUROBI AND TARGET Gurobi::GurobiCXX)
  message(STATUS "Gurobi found and enabled")
  target_compile_definitions(mip-solvers PUBLIC DTWC_ENABLE_GUROBI)
  target_link_libraries(mip-solvers PRIVATE Gurobi::GurobiCXX)
endif()

if(DTWC_ENABLE_HIGHS AND TARGET highs::highs)
  target_compile_definitions(mip-solvers PUBLIC DTWC_ENABLE_HIGHS)
  target_link_libraries(mip-solvers PRIVATE highs::highs)
endif()
