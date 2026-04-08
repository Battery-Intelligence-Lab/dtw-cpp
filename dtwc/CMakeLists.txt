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
  core/mmap_distance_matrix.hpp
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

# ── OpenMP detection ──────────────────────────────────────────────────────────
# Only request the CXX component — if CUDA is enabled as a language, a bare
# find_package(OpenMP) would also search for OpenMP_CUDA which fails on MSVC.
if(MSVC)
  set(OpenMP_RUNTIME_MSVC experimental)
endif()

# Clang targeting MSVC (clang++ -target x86_64-pc-windows-msvc) ships libomp
# but CMake's FindOpenMP cannot auto-detect the flags. Probe the compiler's
# LLVM install directory for libomp and set the hints if found.
if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" AND WIN32 AND NOT OpenMP_CXX_FOUND)
  if(NOT DEFINED OpenMP_CXX_FLAGS)
    # Resolve the LLVM root from the compiler path: .../bin/clang++.exe → .../
    get_filename_component(_llvm_bin "${CMAKE_CXX_COMPILER}" DIRECTORY)
    get_filename_component(_llvm_root "${_llvm_bin}" DIRECTORY)
    find_library(_llvm_libomp omp PATHS "${_llvm_root}/lib" NO_DEFAULT_PATH)
    if(_llvm_libomp)
      set(OpenMP_CXX_FLAGS "-fopenmp" CACHE STRING "OpenMP flags for Clang/Windows (auto-detected)")
      set(OpenMP_CXX_LIB_NAMES "omp" CACHE STRING "OpenMP lib name for Clang/Windows (auto-detected)")
      set(OpenMP_omp_LIBRARY "${_llvm_libomp}" CACHE FILEPATH "Path to libomp (auto-detected)")
      message(STATUS "Auto-detected LLVM libomp at ${_llvm_libomp}")
    endif()
    unset(_llvm_bin)
    unset(_llvm_root)
    unset(_llvm_libomp)
  endif()
endif()

find_package(OpenMP COMPONENTS CXX)
if(OpenMP_CXX_FOUND)
  message(STATUS "OpenMP ${OpenMP_CXX_VERSION} found — parallel execution enabled")
  target_compile_definitions(dtwc++ PUBLIC DTWC_HAS_OPENMP)
  if(MSVC)
    target_compile_options(dtwc_options INTERFACE $<$<COMPILE_LANGUAGE:C,CXX>:/openmp:experimental>)
  else()
    target_link_libraries(dtwc++ PUBLIC OpenMP::OpenMP_CXX)
  endif()
else()
  message(WARNING
    "OpenMP NOT found — DTWC++ will run single-threaded.\n"
    "  This will be extremely slow for large datasets.\n"
    "  To fix: install libomp (Clang), enable OpenMP in your compiler,\n"
    "  or pass -DOpenMP_CXX_FLAGS / -DOpenMP_CXX_LIB_NAMES manually.")
endif()

# Guard against parent projects that use -ffast-math (which includes -ffinite-math-only).
# This is target-scoped so it applies even when consumed as a subproject.
if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(dtwc++ PRIVATE -fno-finite-math-only)
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

# llfio (required) — memory-mapped distance matrices
if(TARGET llfio_hl)
  target_link_libraries(dtwc++ PUBLIC llfio_hl)
  target_compile_definitions(dtwc++ PUBLIC DTWC_HAS_MMAP)
  message(STATUS "llfio linked — memory-mapped distance matrix support enabled")
endif()

# CUDA (optional)
if(DTWC_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
  target_sources(dtwc++ PRIVATE cuda/cuda_dtw.cu)
  target_link_libraries(dtwc++ PUBLIC CUDA::cudart)
  target_compile_definitions(dtwc++ PUBLIC DTWC_HAS_CUDA)
  set_source_files_properties(cuda/cuda_dtw.cu PROPERTIES LANGUAGE CUDA)
  message(STATUS "CUDA enabled for GPU DTW acceleration")
endif()
