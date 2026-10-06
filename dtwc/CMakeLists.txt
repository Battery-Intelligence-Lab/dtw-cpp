cmake_minimum_required(VERSION 3.26)

# Three libraries and one name for consumers:
#   dtwc_core  the algorithms, the session, the text reader and the result
#              writers: everything the Python module and the MEX link;
#   dtwc_io    the readers that need Arrow (Arrow IPC, Parquet, and FastCLARA
#              streaming from Parquet), in a build with Arrow: the one library
#              Arrow links;
#   dtwc_cli   the command line: cli::bind and its config files (CLI11, fkYAML),
#              dtwc::run and the Tier-1 facade (api.cpp) that runs it.
# dtwc++ links them all, so tests, examples, benchmarks and a parent project keep
# their link line.
add_library(dtwc_core STATIC)
add_library(dtwc_cli STATIC)
add_library(dtwc++ INTERFACE)
set(_dtwc_libs dtwc_core dtwc_cli)
if(DTWC_ENABLE_ARROW AND Arrow_FOUND)
  add_library(dtwc_io STATIC)
  target_link_libraries(dtwc_io PUBLIC dtwc_core)
  target_link_libraries(dtwc_cli PUBLIC dtwc_io)
  list(APPEND _dtwc_libs dtwc_io)
endif()
target_link_libraries(dtwc_cli PUBLIC dtwc_core)
target_link_libraries(dtwc++ INTERFACE ${_dtwc_libs})

# Required for linking into the Python shared library (.so) on Linux
set_target_properties(${_dtwc_libs} PROPERTIES POSITION_INDEPENDENT_CODE ON)

# The headers: each folder lists its own in its library's set. The base, dtwc/
# (never the repo root), is the include directory of the library and of
# everything that links it: <core/...>, <mip/...>, "Problem.hpp".
foreach(_dtwc_lib IN LISTS _dtwc_libs)
  target_sources(${_dtwc_lib} PUBLIC FILE_SET HEADERS BASE_DIRS .)
endforeach()
# The umbrella header includes the libraries' headers.
target_sources(dtwc++ INTERFACE FILE_SET HEADERS BASE_DIRS . FILES dtwc.hpp)

# Each folder lists its own files, and the settings that belong to them, on its library.
add_subdirectory(mip)
add_subdirectory(algorithms)
add_subdirectory(base)
add_subdirectory(cli)
add_subdirectory(core)
add_subdirectory(io)
add_subdirectory(extern/nanoarrow)

target_sources(dtwc_core
  PRIVATE
  Problem.cpp
  Problem_IO.cpp
  checkpoint.cpp
  initialisation.cpp
  scores.cpp
  PUBLIC FILE_SET HEADERS FILES
  # Top-level headers
  Data.hpp
  DataLoader.hpp
  checkpoint.hpp
  config.hpp
  distance.hpp
  fileOperations.hpp
  initialisation.hpp
  parallelisation.hpp
  Problem.hpp
  scores.hpp
  settings.hpp
  test_api.hpp
  timing.hpp
  utility.hpp
  # DTW implementations
  warping.hpp
  warping_ddtw.hpp
  warping_wdtw.hpp
  warping_adtw.hpp
  warping_missing.hpp
  warping_missing_arow.hpp
  soft_dtw.hpp
  # The folders that hold headers only
  detail/decode_pair.hpp
  enums/Method.hpp
  enums/Solver.hpp
  enums/enums.hpp
  types/Index.hpp
  types/Range.hpp
  types/types.hpp
)
# The facade the command line runs.
target_sources(dtwc_cli PRIVATE api.cpp PUBLIC FILE_SET HEADERS FILES api.hpp)

# The vendored nanoarrow and fast_float are PRIVATE: only io/arrow_c_data.cpp and
# io/parse_number.cpp include them, behind headers that name none of their types,
# so they never reach a consumer.
target_include_directories(dtwc_core PRIVATE extern)

target_link_libraries(dtwc_core PRIVATE mip-solvers)
# Header-only, included by cli/config.cpp alone. PUBLIC because the public
# cli::bind(CLI::App &, Config &) takes the caller's App.
target_link_libraries(dtwc_cli PUBLIC CLI11::CLI11)

# cli/config.cpp reads --config files through cli/config_file.hpp, which takes
# YAML only with fkYAML. The definition is PUBLIC, as every capability is, so a
# consumer (and the tests) can tell whether a YAML config file will be read.
if(TARGET fkYAML::fkYAML)
  target_link_libraries(dtwc_cli PRIVATE fkYAML::fkYAML)
  target_compile_definitions(dtwc_cli PUBLIC DTWC_HAS_YAML)
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
  target_compile_definitions(dtwc_core PUBLIC DTWC_HAS_OPENMP)
  # mip-solvers is an object library compiled apart from dtwc_core, so it takes the definition
  # and the OpenMP target itself, PRIVATE (its objects join dtwc_core, which carries both to
  # consumers). Without them lagrangian_root.cpp's `#pragma omp` loops compile out and the
  # Lagrangian dual runs serial.
  target_compile_definitions(mip-solvers PRIVATE DTWC_HAS_OPENMP)
  if(MSVC)
    # PUBLIC on dtwc_core, so a consumer that links dtwc_core but not project_options
    # (the Python module) still compiles its `#pragma omp` loops in parallel.
    # project_options carries it too, for mip-solvers, which links that alone.
    target_compile_options(dtwc_core PUBLIC $<$<COMPILE_LANGUAGE:C,CXX>:/openmp:experimental>)
    target_compile_options(dtwc_options INTERFACE $<$<COMPILE_LANGUAGE:C,CXX>:/openmp:experimental>)
  else()
    # OpenMP::OpenMP_CXX carries -fopenmp and the libomp link, PUBLIC on dtwc_core.
    target_link_libraries(dtwc_core PUBLIC OpenMP::OpenMP_CXX)
    target_link_libraries(mip-solvers PRIVATE OpenMP::OpenMP_CXX)
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
  target_compile_definitions(dtwc_core PUBLIC DTWC_SEQUENTIAL_BUILD)
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

# What each library compiles with and passes to nothing that links it:
# - the project's options and warnings;
# - -fno-finite-math-only, against a parent project's -ffast-math (which includes
#   -ffinite-math-only), target-scoped so it applies even in a subproject;
# - a switch over an enum names every enumerator, with no default: adding an
#   enumerator is then a compile error at each switch that misses it, instead of
#   a runtime throw. Values reach users only as names, which the name tables check.
foreach(_dtwc_lib IN LISTS _dtwc_libs)
  target_link_libraries(${_dtwc_lib} PRIVATE project_warnings project_options)
  if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${_dtwc_lib} PRIVATE -fno-finite-math-only)
  endif()
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    target_compile_options(${_dtwc_lib} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/we4062>)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(${_dtwc_lib} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror=switch>)
  endif()
endforeach()

# llfio (optional) — memory-mapped distance matrices. core/distance_matrix.cpp
# is the one file that includes it, so the link is PRIVATE; DTWC_HAS_MMAP is
# PUBLIC, as every capability is, so a consumer (and the tests) can tell whether
# use_mmap_distance_matrix can map a file.
if(TARGET llfio_hl)
  target_link_libraries(dtwc_core PRIVATE llfio_hl)
  target_compile_definitions(dtwc_core PUBLIC DTWC_HAS_MMAP)
  message(STATUS "llfio linked — memory-mapped distance matrix support enabled")
endif()

# Arrow + Parquet (optional) — Arrow IPC and Parquet reading, on dtwc_io alone. Arrow without Parquet is a
# legitimate IPC-only build; Arrow found with no target to link is an error, not a build that links nothing.
if(TARGET dtwc_io)
  if(DTWC_ARROW_FROM_CPM)
    # CPM-built static Arrow + Parquet
    target_link_libraries(dtwc_io PUBLIC arrow_static parquet_static)
    target_compile_definitions(dtwc_io PUBLIC DTWC_HAS_ARROW DTWC_HAS_PARQUET)
    message(STATUS "Arrow + Parquet linked — IPC and Parquet reading enabled")
  elseif(TARGET Arrow::arrow_shared)
    # System-installed Arrow (conda, apt, brew)
    target_link_libraries(dtwc_io PUBLIC Arrow::arrow_shared)
    target_compile_definitions(dtwc_io PUBLIC DTWC_HAS_ARROW)
    if(Parquet_FOUND)
      target_link_libraries(dtwc_io PUBLIC Parquet::parquet_shared)
      target_compile_definitions(dtwc_io PUBLIC DTWC_HAS_PARQUET)
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
