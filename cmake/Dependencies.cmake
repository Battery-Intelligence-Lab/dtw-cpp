# This cmake file is to add external dependency projects.
# Adapted from https://github.com/cpp-best-practices/cmake_tecomplate/tree/main
include(cmake/CPM.cmake)

# Done as a function so that updates to variables like
# CMAKE_CXX_FLAGS don't propagate out to other
# targets

function(dtwc_setup_dependencies)
  # For each dependency, see if it's
  # already been provided to us by a parent project

  if(NOT TARGET Catch2::Catch2WithMain) # Catch2 library:
    CPMAddPackage(
      NAME Catch2
      URL "https://github.com/catchorg/Catch2/archive/refs/tags/v3.3.2.tar.gz"
      OPTIONS 
      "CATCH_INSTALL_DOCS OFF" "CATCH_INSTALL_EXTRAS OFF" "CATCH_BUILD_TESTING OFF"
    )
  endif()

  # if(NOT TARGET fmt) # fmt library:
  # CPMAddPackage(
  #   NAME fmt
  #   URL "https://github.com/fmtlib/fmt/archive/refs/tags/10.1.1.tar.gz"
  # )
  # endif()

  # HiGHS library:
  if(NOT TARGET highs::highs AND DTWC_ENABLE_HIGHS)# HiGHS library:
  CPMAddPackage(
    NAME highs
    URL "https://github.com/ERGO-Code/HiGHS/archive/refs/tags/v1.6.0.tar.gz"
    SYSTEM
    EXCLUDE_FROM_ALL
    OPTIONS
    "ZLIB OFF" "FAST_BUILD ON" "BUILD_TESTING OFF" "BUILD_EXAMPLES OFF" "BUILD_SHARED_LIBS OFF")
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

  CPMAddPackage(
    NAME armadillo
    URL "https://gitlab.com/conradsnicta/armadillo-code/-/archive/12.6.x/armadillo-code-12.6.x.tar.gz"
    OPTIONS
    "BUILD_SMOKE_TEST OFF"
  )
endfunction()
