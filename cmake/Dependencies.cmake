# This cmake file is to add external dependency projects.
# Adapted from https://github.com/cpp-best-practices/cmake_template/tree/main

include(cmake/CPM.cmake)

# Range-v3 library:
CPMAddPackage(
  NAME range-v3
  URL "https://github.com/ericniebler/range-v3/archive/refs/tags/0.12.0.tar.gz"
  DOWNLOAD_ONLY YES 
)

if(range-v3_ADDED) 
add_library(range-v3 INTERFACE IMPORTED)
target_include_directories(range-v3 INTERFACE ${range-v3_SOURCE_DIR}/include)
endif()

# Eigen library:
CPMAddPackage(
  NAME eigen
  URL "https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz"
  DOWNLOAD_ONLY YES
)

add_library(eigen INTERFACE IMPORTED)
target_include_directories(eigen SYSTEM INTERFACE ${eigen_SOURCE_DIR})

# Catch2 library:
if(NOT TARGET Catch2::Catch2WithMain)
  CPMAddPackage(
    NAME Catch2
    URL "https://github.com/catchorg/Catch2/archive/refs/tags/v3.4.0.tar.gz"
  )
endif()

# CPMAddPackage("gh:fmtlib/fmt#10.1.1") #fmt library
# fmt library:
CPMAddPackage(
  NAME fmt
  URL "https://github.com/fmtlib/fmt/archive/refs/tags/10.1.1.tar.gz"
)

# HiGHS library:
CPMAddPackage(
    NAME highs
    URL "https://github.com/ERGO-Code/HiGHS/archive/refs/tags/v1.6.0.tar.gz"
    SYSTEM
    OPTIONS 
        "HIGHS_BUILD_SHARED NO"
        "HIGHS_BUILD_EXAMPLES NO"
        "HIGHS_BUILD_TESTS NO"
)
set(FAST_BUILD ON CACHE INTERNAL "Fast Build")