cmake_minimum_required(VERSION 3.26)

add_library(dtwc++ STATIC)

# Required for linking into the Python shared library (.so) on Linux
set_target_properties(dtwc++ PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Each folder lists its own files, and the settings that belong to them, on dtwc++.
add_subdirectory(mip)
add_subdirectory(algorithms)
add_subdirectory(base)
add_subdirectory(cli)
add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(extern/nanoarrow)

target_sources(dtwc++
  PRIVATE
  Problem.cpp
  Problem_IO.cpp
  api.cpp
  checkpoint.cpp
  initialisation.cpp
  scores.cpp
  PUBLIC
  # Top-level headers
  Data.hpp
  DataLoader.hpp
  api.hpp
  checkpoint.hpp
  config.hpp
  dtwc.hpp
  distance.hpp
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
)

target_include_directories(dtwc++ PUBLIC .)
# The vendored nanoarrow and fast_float are PRIVATE: only io/arrow_c_data.cpp and
# io/parse_number.cpp include them, behind headers that name none of their types,
# so they never reach a consumer.
target_include_directories(dtwc++ PRIVATE extern)

target_link_libraries(dtwc++
  PRIVATE
  mip-solvers
  project_warnings
  project_options
  PUBLIC
  # Header-only, included by cli/config.cpp alone. PUBLIC because the public
  # cli::bind(CLI::App &, Config &) takes the caller's App.
  CLI11::CLI11
)

# cli/config.cpp reads --config files through cli/config_file.hpp, which takes
# YAML only with fkYAML. The definition is PUBLIC, as every capability is, so a
# consumer (and the tests) can tell whether a YAML config file will be read.
if(TARGET fkYAML::fkYAML)
  target_link_libraries(dtwc++ PRIVATE fkYAML::fkYAML)
  target_compile_definitions(dtwc++ PUBLIC DTWC_HAS_YAML)
endif()

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
    # PUBLIC on dtwc++, so a consumer that links dtwc++ but not project_options
    # (the Python module) still compiles its `#pragma omp` loops in parallel.
    # project_options carries it too, for mip-solvers, which links that alone.
    target_compile_options(dtwc++ PUBLIC $<$<COMPILE_LANGUAGE:C,CXX>:/openmp:experimental>)
    target_compile_options(dtwc_options INTERFACE $<$<COMPILE_LANGUAGE:C,CXX>:/openmp:experimental>)
  else()
    # OpenMP::OpenMP_CXX carries -fopenmp and the libomp link, PUBLIC on dtwc++.
    target_link_libraries(dtwc++ PUBLIC OpenMP::OpenMP_CXX)
  endif()
elseif(DTWC_ALLOW_SEQUENTIAL)
  # Explicit opt-out of the parallelism requirement (keeps the "core builds
  # without optional deps" contract as a DELIBERATE choice). Loud warning + the
  # DTWC_SEQUENTIAL_BUILD compile definition consumed by the runtime thread
  # warning (base/env.cpp, warn_if_single_threaded) — DO NOT rename this macro.
  message(WARNING
    "OpenMP NOT found and -DDTWC_ALLOW_SEQUENTIAL=ON given — building SEQUENTIAL.\n"
    "  DTWC++ will run SINGLE-THREADED; this is EXTREMELY slow for large datasets.\n"
    "  DTWC_SEQUENTIAL_BUILD is defined on the dtwc targets; the runtime will warn.")
  target_compile_definitions(dtwc++ PUBLIC DTWC_SEQUENTIAL_BUILD)
else()
  # No silent serial builds (global no-silent-fallback constraint). OpenMP is
  # required for shipped artifacts unless sequential is EXPLICITLY allowed.
  message(FATAL_ERROR
    "OpenMP NOT found — DTWC++ requires OpenMP for parallel execution.\n"
    "  Silent single-threaded builds are disabled by design (no-silent-fallback).\n"
    "  To build anyway (SINGLE-THREADED, EXTREMELY slow for large datasets),\n"
    "  re-run configure with -DDTWC_ALLOW_SEQUENTIAL=ON.\n"
    "  To fix properly: install libomp (Clang), enable OpenMP in your compiler,\n"
    "  or pass -DOpenMP_CXX_FLAGS / -DOpenMP_CXX_LIB_NAMES manually.")
endif()

# Guard against parent projects that use -ffast-math (which includes -ffinite-math-only).
# This is target-scoped so it applies even when consumed as a subproject.
if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(dtwc++ PRIVATE -fno-finite-math-only)
endif()

# A switch over an enum names every enumerator, with no default: adding an
# enumerator is then a compile error at each switch that misses it, instead of
# a runtime throw. Values reach users only as names, which the name tables check.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  target_compile_options(dtwc++ PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/we4062>)
elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(dtwc++ PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror=switch>)
endif()

# llfio (optional) — memory-mapped distance matrices. core/distance_matrix.cpp
# is the one file that includes it, so the link is PRIVATE; DTWC_HAS_MMAP is
# PUBLIC, as every capability is, so a consumer (and the tests) can tell whether
# use_mmap_distance_matrix can map a file.
if(TARGET llfio_hl)
  target_link_libraries(dtwc++ PRIVATE llfio_hl)
  target_compile_definitions(dtwc++ PUBLIC DTWC_HAS_MMAP)
  message(STATUS "llfio linked — memory-mapped distance matrix support enabled")
endif()

# Arrow + Parquet (optional) — Arrow IPC and Parquet reading. Arrow without Parquet is a
# legitimate IPC-only build; Arrow found with no target to link is an error, not a build that links nothing.
if(DTWC_ENABLE_ARROW AND Arrow_FOUND)
  if(DTWC_ARROW_FROM_CPM)
    # CPM-built static Arrow + Parquet
    target_link_libraries(dtwc++ PUBLIC arrow_static parquet_static)
    target_compile_definitions(dtwc++ PUBLIC DTWC_HAS_ARROW DTWC_HAS_PARQUET)
    message(STATUS "Arrow + Parquet linked — IPC and Parquet reading enabled")
  elseif(TARGET Arrow::arrow_shared)
    # System-installed Arrow (conda, apt, brew)
    target_link_libraries(dtwc++ PUBLIC Arrow::arrow_shared)
    target_compile_definitions(dtwc++ PUBLIC DTWC_HAS_ARROW)
    if(Parquet_FOUND)
      target_link_libraries(dtwc++ PUBLIC Parquet::parquet_shared)
      target_compile_definitions(dtwc++ PUBLIC DTWC_HAS_PARQUET)
      message(STATUS "Arrow + Parquet linked — IPC and Parquet reading enabled")
    else()
      message(STATUS "Arrow linked, IPC only: Parquet not found — reading a Parquet file raises")
    endif()
  else()
    message(FATAL_ERROR
      "DTWC_ENABLE_ARROW=ON: Arrow was found but it defines no Arrow::arrow_shared target to link.\n"
      "  Install a shared Arrow (conda install -c conda-forge arrow-cpp), or pass -DDTWC_ENABLE_ARROW=OFF.")
  endif()
endif()

# GPU back ends (optional); each folder checks its own switch.
add_subdirectory(cuda)
add_subdirectory(metal)
