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
    "CI OFF" "ZLIB OFF" "BUILD_EXAMPLES OFF" "BUILD_TESTING OFF"
    )
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
