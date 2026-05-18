cmake_minimum_required(VERSION 3.26)

# Python discovery — scikit-build-core pre-configures the correct paths
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

# nanobind discovery — try pip-installed first, fall back to CPM
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE NB_DIR OUTPUT_STRIP_TRAILING_WHITESPACE
  RESULT_VARIABLE NB_RESULT ERROR_QUIET
)
if(NB_RESULT EQUAL 0 AND EXISTS "${NB_DIR}")
  list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")
endif()
find_package(nanobind CONFIG QUIET)
if(NOT nanobind_FOUND)
  include(../cmake/CPM.cmake)
  CPMAddPackage(
    NAME nanobind
    GITHUB_REPOSITORY wjakob/nanobind
    VERSION 2.4.0
    OPTIONS "NB_TEST OFF"
  )
endif()

# STABLE_ABI requires Python >= 3.12
if(Python_VERSION VERSION_GREATER_EQUAL "3.12")
  nanobind_add_module(_dtwcpp_core STABLE_ABI src/_dtwcpp_core.cpp)
else()
  nanobind_add_module(_dtwcpp_core src/_dtwcpp_core.cpp)
endif()

target_link_libraries(_dtwcpp_core PRIVATE dtwc++ project_options)
# `project_options` propagates the MSVC `/openmp:experimental` compile flag
# (set in dtwc/CMakeLists.txt). Without it, the python module's TU contains
# `#pragma omp parallel for` (e.g. in `compute_distance_matrix`) but compiles
# without an OpenMP runtime, leaving every parallel loop running serially and
# `_OPENMP` undefined — so `dtwcpp.OPENMP_AVAILABLE` reports False even when
# the underlying dtwc++ static library was built with OpenMP.

install(TARGETS _dtwcpp_core DESTINATION dtwcpp)
