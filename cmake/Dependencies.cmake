# This cmake file is to add external dependency projects.
# Adapted from https://github.com/cpp-best-practices/cmake_tecomplate/tree/main
include(cmake/CPM.cmake)

# Done as a function so that updates to variables like
# CMAKE_CXX_FLAGS don't propagate out to other
# targets

function(dtwc_setup_dependencies)
  # For each dependency, see if it's
  # already been provided to us by a parent project
  CPMAddPackage(
    NAME CPMLicenses.cmake
    GITHUB_REPOSITORY cpm-cmake/CPMLicenses.cmake
    VERSION 0.0.7
  )

  if(NOT TARGET Catch2::Catch2WithMain) # Catch2 library:
    CPMAddPackage(
      NAME Catch2
      URL "https://github.com/catchorg/Catch2/archive/refs/tags/v3.13.0.tar.gz"
      OPTIONS 
      "CATCH_INSTALL_DOCS OFF" "CATCH_INSTALL_EXTRAS OFF" "CATCH_BUILD_TESTING OFF"
    )
  endif()

  # HiGHS library:
  if(NOT TARGET highs::highs AND DTWC_ENABLE_HIGHS)# HiGHS library:
  CPMAddPackage(
    NAME highs
    URL "https://github.com/ERGO-Code/HiGHS/archive/refs/tags/v1.13.1.tar.gz"
    SYSTEM
    EXCLUDE_FROM_ALL
    OPTIONS
    "CI OFF" "ZLIB OFF" "BUILD_EXAMPLES OFF" "BUILD_TESTING OFF" "FAST_BUILD ON"
    )
    # HiGHS v1.13.1 has debug assertions (ub_consistent) that fire on valid warm-start
    # MIP solves due to numerical noise. Suppress by defining NDEBUG on HiGHS targets.
    if(TARGET highs)
      target_compile_definitions(highs PRIVATE NDEBUG)
    endif()
  endif()

  if (NOT TARGET CLI11::CLI11)
  CPMAddPackage(
    NAME CLI11
    URL "https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.6.2.tar.gz"
    DOWNLOAD_ONLY YES 
  )

   add_library(CLI11::CLI11 INTERFACE IMPORTED)
  set_target_properties(CLI11::CLI11 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CLI11_SOURCE_DIR}/include")
  endif()

  # nanobind — Python bindings (BSD-3, by Wenzel Jakob)
  # find_package(Python) and nanobind discovery are handled in python/CMakeLists.txt
  # to ensure scikit-build-core has configured paths first.

  # RapidCSV - header-only CSV parser (BSD 3-Clause license)
  if (NOT TARGET rapidcsv::rapidcsv)
    CPMAddPackage(
      NAME rapidcsv
      GITHUB_REPOSITORY d99kris/rapidcsv
      VERSION 8.92
      DOWNLOAD_ONLY YES
    )

    add_library(rapidcsv::rapidcsv INTERFACE IMPORTED)
    set_target_properties(rapidcsv::rapidcsv PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${rapidcsv_SOURCE_DIR}/src")
  endif()

  # Eigen3 — header-only linear algebra (MPL2 / Apache-2.0 / BSD-3)
  # Used for scratch matrices (aligned SIMD-ready allocation, zero-copy Map),
  # replacing custom ScratchMatrix and DenseDistanceMatrix internals.
  if(NOT TARGET Eigen3::Eigen)
    CPMAddPackage(
      NAME Eigen
      URL "https://gitlab.com/libeigen/eigen/-/archive/5.0.1/eigen-5.0.1.tar.bz2"
      DOWNLOAD_ONLY YES
    )
    add_library(Eigen3::Eigen INTERFACE IMPORTED)
    set_target_properties(Eigen3::Eigen PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${Eigen_SOURCE_DIR}"
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${Eigen_SOURCE_DIR}")
  endif()

  if(DTWC_BUILD_BENCHMARK)
    if(NOT TARGET benchmark::benchmark)
      CPMAddPackage(
        NAME benchmark
        GITHUB_REPOSITORY google/benchmark
        VERSION 1.9.5
        OPTIONS
          "BENCHMARK_ENABLE_TESTING OFF"
          "BENCHMARK_ENABLE_GTEST_TESTS OFF"
          "BENCHMARK_ENABLE_WERROR OFF"
      )
    endif()
  endif()

  # yaml-cpp — YAML configuration file support (MIT license, optional)
  if(DTWC_ENABLE_YAML AND NOT TARGET yaml-cpp)
    CPMAddPackage(
      NAME yaml-cpp
      URL "https://github.com/jbeder/yaml-cpp/archive/refs/tags/0.9.0.tar.gz"
      SYSTEM
      EXCLUDE_FROM_ALL
      OPTIONS "YAML_CPP_BUILD_TESTS OFF" "YAML_CPP_BUILD_TOOLS OFF"
    )
    # Ensure namespaced alias exists (CPM subdirectory may not create it)
    if(TARGET yaml-cpp AND NOT TARGET yaml-cpp::yaml-cpp)
      add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)
    endif()
    if(NOT TARGET yaml-cpp)
      message(WARNING "yaml-cpp not found -- YAML config support disabled")
      set(DTWC_ENABLE_YAML OFF PARENT_SCOPE)
    endif()
  endif()

  # llfio — memory-mapped I/O for large distance matrices (required)
  if(NOT TARGET llfio_hl)
    CPMAddPackage(
      NAME llfio
      GITHUB_REPOSITORY ned14/llfio
      GIT_TAG develop
      DOWNLOAD_ONLY YES
    )
    if(llfio_ADDED)
      add_subdirectory(${llfio_SOURCE_DIR} ${llfio_BINARY_DIR} EXCLUDE_FROM_ALL)
    endif()
  endif()

  if(TARGET llfio_hl)
    message(STATUS "  llfio:    YES (memory-mapped distance matrix enabled)")
    set(DTWC_HAS_MMAP TRUE)
  else()
    message(FATAL_ERROR "  llfio:    NOT FOUND — llfio is a required dependency")
  endif()

  # Apache Arrow + Parquet (optional) — zero-copy IPC and Parquet reading.
  # No Python required. No external package manager required.
  #
  # Detection order:
  #   1. find_package (conda, system install — fastest, no build)
  #   2. CPM: download and build minimal Arrow from source (~5 min first time, cached after)
  #
  if(DTWC_ENABLE_ARROW)
    # 1. Try system-installed Arrow first (conda, apt, brew, module load)
    find_package(Arrow QUIET CONFIG)
    find_package(Parquet QUIET CONFIG)

    if(Arrow_FOUND)
      message(STATUS "  Arrow:    YES (v${Arrow_VERSION}) — system install")
      set(Arrow_FOUND TRUE PARENT_SCOPE)
      if(Parquet_FOUND)
        message(STATUS "  Parquet:  YES (v${Parquet_VERSION}) — system install")
        set(DTWC_HAS_PARQUET_LIB TRUE PARENT_SCOPE)
      endif()
    else()
      # 2. Build minimal Arrow+Parquet from source via CPM
      message(STATUS "  Arrow:    not found — building from source via CPM (first build takes ~5 min)")

      # Workaround: Arrow's ExternalProject passes CMAKE_CXX_FLAGS_* to sub-builds
      # (snappy, zstd, thrift). On Windows+Clang, flags like "-Xclang --dependent-lib=msvcrt"
      # and "-D_DLL -D_MT" contain spaces that break CMake argument parsing in ExternalProject.
      # Strip these from ALL flag variables before Arrow configure, restore after.
      # Flag stripping not needed — Windows+Clang skips CPM build (see guard above)
      # NOTE: On Windows+Clang, Arrow's ExternalProject sub-builds fail because
      # CMake's platform module sets "-Xclang --dependent-lib=msvcrt" in default
      # flags, and the space breaks ExternalProject's semicolon-separated command.
      # This is an Arrow upstream issue. Workarounds:
      #   - Use MSVC compiler instead of Clang on Windows
      #   - Use conda: conda install -c conda-forge arrow-cpp (find_package path)
      #   - Use Linux (SLURM) where this issue doesn't exist
      if(WIN32 AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(WARNING "Arrow CPM build is not supported with Windows+Clang due to "
          "ExternalProject flag quoting issues. Use one of:\n"
          "  1. conda install -c conda-forge arrow-cpp  (then find_package works)\n"
          "  2. Build with MSVC instead of Clang\n"
          "  3. Use Linux (SLURM) where CPM build works\n"
          "  4. Use dtwc-convert (Python) as a workaround")
        set(DTWC_ENABLE_ARROW OFF PARENT_SCOPE)
      else()

      CPMAddPackage(
        NAME Arrow
        VERSION 19.0.1
        URL "https://github.com/apache/arrow/archive/refs/tags/apache-arrow-19.0.1.tar.gz"
        SOURCE_SUBDIR cpp
        SYSTEM
        EXCLUDE_FROM_ALL
        OPTIONS
          "ARROW_BUILD_STATIC ON"
          "ARROW_BUILD_SHARED OFF"
          "ARROW_PARQUET ON"
          "ARROW_IPC ON"
          "ARROW_FILESYSTEM ON"
          "ARROW_COMPUTE OFF"
          "ARROW_CSV OFF"
          "ARROW_DATASET OFF"
          "ARROW_JSON OFF"
          "ARROW_FLIGHT OFF"
          "ARROW_GANDIVA OFF"
          "ARROW_ORC OFF"
          "ARROW_PLASMA OFF"
          "ARROW_PYTHON OFF"
          "ARROW_S3 OFF"
          "ARROW_HDFS OFF"
          "ARROW_JEMALLOC OFF"
          "ARROW_MIMALLOC OFF"
          "ARROW_WITH_SNAPPY OFF"
          "ARROW_WITH_ZSTD OFF"
          "ARROW_WITH_LZ4 OFF"
          "ARROW_WITH_BROTLI OFF"
          "ARROW_WITH_BZ2 OFF"
          "ARROW_WITH_ZLIB OFF"
          "ARROW_DEPENDENCY_SOURCE BUNDLED"
          "ARROW_SIMD_LEVEL NONE"
          "ARROW_USE_XSIMD OFF"
          "ARROW_RUNTIME_SIMD_LEVEL NONE"
          "ARROW_BUILD_TESTS OFF"
          "ARROW_BUILD_BENCHMARKS OFF"
          "ARROW_BUILD_EXAMPLES OFF"
          "ARROW_BUILD_UTILITIES OFF"
          "PARQUET_BUILD_EXECUTABLES OFF"
          "PARQUET_BUILD_EXAMPLES OFF"
          "PARQUET_REQUIRE_ENCRYPTION OFF"
      )

      if(TARGET arrow_static)
        set(Arrow_FOUND TRUE PARENT_SCOPE)
        set(DTWC_ARROW_FROM_CPM TRUE PARENT_SCOPE)
        set(DTWC_HAS_PARQUET_LIB TRUE PARENT_SCOPE)
        message(STATUS "  Arrow:    built from source (static, IPC + Parquet)")
      else()
        message(WARNING "Arrow CPM build failed.\n"
          "  Install: conda install -c conda-forge arrow-cpp")
        set(DTWC_ENABLE_ARROW OFF PARENT_SCOPE)
      endif()

      endif() # Windows+Clang guard
    endif()
  endif()

  # MPI (optional) — distributed distance matrix computation
  if(DTWC_ENABLE_MPI)
    # On Windows, help CMake find MS-MPI by setting hints from well-known
    # environment variables and registry locations.
    if(WIN32 AND NOT MPI_CXX_FOUND)
      # MS-MPI SDK sets MSMPI_INC and MSMPI_LIB64 / MSMPI_LIB32.
      # If the SDK env-vars are not set, try the default install path.
      if(NOT DEFINED ENV{MSMPI_INC})
        set(_msmpi_sdk_dir "C:/Program Files (x86)/Microsoft SDKs/MPI")
        if(EXISTS "${_msmpi_sdk_dir}/Include/mpi.h")
          set(ENV{MSMPI_INC} "${_msmpi_sdk_dir}/Include")
          if(CMAKE_SIZEOF_VOID_P EQUAL 8)
            set(ENV{MSMPI_LIB64} "${_msmpi_sdk_dir}/Lib/x64")
          else()
            set(ENV{MSMPI_LIB32} "${_msmpi_sdk_dir}/Lib/x86")
          endif()
          message(STATUS "MS-MPI SDK found at ${_msmpi_sdk_dir}")
        endif()
      endif()
    endif()

    find_package(MPI COMPONENTS CXX)
    if(MPI_CXX_FOUND)
      message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
    else()
      message(WARNING "MPI requested but not found — disabling.\n"
        "  On Windows, install the MS-MPI SDK from:\n"
        "  https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi\n"
        "  (both the runtime MSMpiSetup.exe AND the SDK msmpisdk.msi are required)")
      set(DTWC_ENABLE_MPI OFF PARENT_SCOPE)
    endif()
  endif()

  # CUDA detection is done in root CMakeLists.txt (enable_language requires
  # directory scope). CUDAToolkit is already found there.

endfunction()
