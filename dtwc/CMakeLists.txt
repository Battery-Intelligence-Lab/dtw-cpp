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
  algorithms/hierarchical.cpp
  algorithms/clarans.cpp
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
  algorithms/hierarchical.hpp
  algorithms/clarans.hpp
  algorithms/detail/medoid_utils.hpp
)

target_include_directories(dtwc++ PUBLIC .)

target_link_libraries(dtwc++
  PRIVATE
  mip-solvers
  project_warnings
  project_options
  PUBLIC
  rapidcsv::rapidcsv
  Eigen3::Eigen
)

# Only request the CXX component — if CUDA is enabled as a language, a bare
# find_package(OpenMP) would also search for OpenMP_CUDA which fails on MSVC.
find_package(OpenMP COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found")
  target_link_libraries(dtwc++ PUBLIC OpenMP::OpenMP_CXX)
  # MSVC needs /openmp:experimental for #pragma omp simd (OpenMP 4.0+)
  # Guard with COMPILE_LANGUAGE to avoid passing MSVC flags to nvcc.
  if(MSVC)
    target_compile_options(dtwc++ PUBLIC $<$<COMPILE_LANGUAGE:C,CXX>:/openmp:experimental>)
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

# MPI (optional) — check for the imported target rather than the variable,
# because find_package(MPI) runs inside dtwc_setup_dependencies() (a function)
# and MPI_CXX_FOUND doesn't propagate to this scope.
if(DTWC_ENABLE_MPI AND TARGET MPI::MPI_CXX)
  target_sources(dtwc++ PRIVATE mpi/mpi_distance_matrix.cpp)
  target_link_libraries(dtwc++ PUBLIC MPI::MPI_CXX)
  target_compile_definitions(dtwc++ PUBLIC DTWC_HAS_MPI)
  message(STATUS "MPI enabled for distributed distance matrix computation")
endif()

# CUDA (optional)
if(DTWC_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  target_sources(dtwc++ PRIVATE cuda/cuda_dtw.cu)
  target_link_libraries(dtwc++ PUBLIC CUDA::cudart)
  target_compile_definitions(dtwc++ PUBLIC DTWC_HAS_CUDA)
  set_source_files_properties(cuda/cuda_dtw.cu PROPERTIES LANGUAGE CUDA)
  message(STATUS "CUDA enabled for GPU DTW acceleration")
endif()
