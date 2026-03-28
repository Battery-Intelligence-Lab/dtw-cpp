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
      URL "https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.0.tar.gz"
      OPTIONS 
      "CATCH_INSTALL_DOCS OFF" "CATCH_INSTALL_EXTRAS OFF" "CATCH_BUILD_TESTING OFF"
    )
  endif()

  # HiGHS library:
  if(NOT TARGET highs::highs AND DTWC_ENABLE_HIGHS)# HiGHS library:
  CPMAddPackage(
    NAME highs
    URL "https://github.com/ERGO-Code/HiGHS/archive/refs/tags/v1.7.2.tar.gz"
    SYSTEM
    EXCLUDE_FROM_ALL
    OPTIONS
    "CI OFF" "ZLIB OFF" "BUILD_EXAMPLES OFF" "BUILD_TESTING OFF"
    )
  endif()

  if (NOT TARGET CLI11::CLI11)
  CPMAddPackage(
    NAME CLI11
    URL "https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.3.2.tar.gz"
    DOWNLOAD_ONLY YES 
  )

   add_library(CLI11::CLI11 INTERFACE IMPORTED)
  set_target_properties(CLI11::CLI11 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${CLI11_SOURCE_DIR}/include")
  endif()

  if(NOT TARGET armadillo)
    CPMAddPackage(
      NAME armadillo
      URL "https://gitlab.com/conradsnicta/armadillo-code/-/archive/14.2.x/armadillo-code-14.2.x.tar.gz"
      OPTIONS
      "BUILD_SMOKE_TEST OFF"
    )
  endif()

  # RapidCSV - header-only CSV parser (BSD 3-Clause license)
  if (NOT TARGET rapidcsv::rapidcsv)
    CPMAddPackage(
      NAME rapidcsv
      GITHUB_REPOSITORY d99kris/rapidcsv
      VERSION 8.84
      DOWNLOAD_ONLY YES
    )

    add_library(rapidcsv::rapidcsv INTERFACE IMPORTED)
    set_target_properties(rapidcsv::rapidcsv PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${rapidcsv_SOURCE_DIR}/src")
  endif()

  if(DTWC_BUILD_BENCHMARK)
    if(NOT TARGET benchmark::benchmark)
      CPMAddPackage(
        NAME benchmark
        GITHUB_REPOSITORY google/benchmark
        VERSION 1.9.1
        OPTIONS
          "BENCHMARK_ENABLE_TESTING OFF"
          "BENCHMARK_ENABLE_GTEST_TESTS OFF"
      )
    endif()
  endif()

endfunction()
