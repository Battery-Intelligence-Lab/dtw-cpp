# This cmake file is to add external dependency projects.
# Adapted from https://github.com/cpp-best-practices/cmake_template/tree/main
include(cmake/CPM.cmake)

# Done as a function so that updates to variables like
# CMAKE_CXX_FLAGS don't propagate out to other
# targets

function(dtwc_setup_dependencies)
  # For each dependency, see if it's
  # already been provided to us by a parent project

  if(NOT TARGET range-v3)  # Range-v3 library:
  CPMAddPackage(
    NAME range-v3
    URL "https://github.com/ericniebler/range-v3/archive/refs/tags/0.12.0.tar.gz"
    DOWNLOAD_ONLY YES 
  )

  if(range-v3_ADDED) 
  add_library(range-v3 INTERFACE IMPORTED)
  target_include_directories(range-v3 INTERFACE ${range-v3_SOURCE_DIR}/include)
  endif()
  endif()


  if(NOT TARGET eigen) # Eigen library:
  CPMAddPackage(
    NAME eigen
    URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"
    DOWNLOAD_ONLY YES
  )

  add_library(eigen INTERFACE IMPORTED)
  target_include_directories(eigen SYSTEM INTERFACE ${eigen_SOURCE_DIR})
  endif()

  if(NOT TARGET Catch2::Catch2WithMain) # Catch2 library:
    CPMAddPackage(
      NAME Catch2
      URL "https://github.com/catchorg/Catch2/archive/refs/tags/v3.4.0.tar.gz"
    )
  endif()


  if(NOT TARGET fmt) # fmt library:
  CPMAddPackage(
    NAME fmt
    URL "https://github.com/fmtlib/fmt/archive/refs/tags/10.1.1.tar.gz"
  )
  endif()

  if(NOT TARGET highs::highs)# HiGHS library:
  include(FetchContent)

  FetchContent_Declare(
      highs
      GIT_REPOSITORY "https://github.com/ERGO-Code/HiGHS.git"
      GIT_TAG        "bazel"
  )
  set(FAST_BUILD ON CACHE INTERNAL "Fast Build")
  
  FetchContent_MakeAvailable(highs)

  endif()
endfunction()
