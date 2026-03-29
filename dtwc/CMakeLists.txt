cmake_minimum_required(VERSION 3.21)

add_library(dtwc++ STATIC)

# Required for linking into the Python shared library (.so) on Linux
set_target_properties(dtwc++ PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_subdirectory(mip)

target_sources(dtwc++
  PRIVATE
  Problem.cpp
  Problem_IO.cpp
  checkpoint.cpp
  initialisation.cpp
  scores.cpp
  core/dtw.cpp
  algorithms/fast_pam.cpp
  algorithms/fast_clara.cpp
  core/pruned_distance_matrix.cpp
  PUBLIC
  # Top-level headers
  Data.hpp
  DataLoader.hpp
  checkpoint.hpp
  dtwc.hpp
  fileOperations.hpp
  initialisation.hpp
  parallelisation.hpp
  Problem.hpp
  scores.hpp
  settings.hpp
  timing.hpp
  utility.hpp
  # DTW implementations
  warping.hpp
  warping_ddtw.hpp
  warping_wdtw.hpp
  warping_adtw.hpp
  soft_dtw.hpp
  # Core types
  core/clustering_result.hpp
  core/distance_matrix.hpp
  core/distance_metric.hpp
  core/dtw.hpp
  core/dtw_options.hpp
  core/lower_bounds.hpp
  core/lower_bound_impl.hpp
  core/pruned_distance_matrix.hpp
  core/scratch_matrix.hpp
  core/time_series.hpp
  core/z_normalize.hpp
  # Algorithm headers
  algorithms/fast_pam.hpp
  algorithms/fast_clara.hpp
)

target_include_directories(dtwc++ PUBLIC .)

target_link_libraries(dtwc++
  PRIVATE
  mip-solvers
  project_warnings
  project_options
  PUBLIC
  rapidcsv::rapidcsv
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found")
  target_link_libraries(dtwc++ PUBLIC OpenMP::OpenMP_CXX)
  # MSVC needs /openmp:experimental for #pragma omp simd (OpenMP 4.0+)
  if(MSVC)
    target_compile_options(dtwc++ PUBLIC /openmp:experimental)
  endif()
endif()

# Google Highway SIMD (optional)
if(DTWC_ENABLE_SIMD AND TARGET hwy::hwy)
  target_sources(dtwc++ PRIVATE
    simd/lb_keogh_simd.cpp
    simd/z_normalize_simd.cpp
    simd/multi_pair_dtw.cpp
  )
  target_link_libraries(dtwc++ PRIVATE hwy::hwy)
  target_compile_definitions(dtwc++ PUBLIC DTWC_HAS_HIGHWAY)
  message(STATUS "SIMD enabled via Google Highway")
endif()
