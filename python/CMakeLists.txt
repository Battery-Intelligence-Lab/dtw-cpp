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
    GIT_TAG 0f9ce749b257fdfe701edb3cf6f7027ba029434a # v2.4.0
    OPTIONS "NB_TEST OFF"
  )
endif()

# STABLE_ABI requires Python >= 3.12. NOMINSIZE: nanobind's default -Os also compiles the kernel templates this
# file instantiates, and under LTO that copy can be the one the whole module runs.
if(Python_VERSION VERSION_GREATER_EQUAL "3.12")
  nanobind_add_module(_dtwcpp_core STABLE_ABI NOMINSIZE src/_dtwcpp_core.cpp)
else()
  nanobind_add_module(_dtwcpp_core NOMINSIZE src/_dtwcpp_core.cpp)
endif()

# The CLI library, which brings the core: _run is dtwc::run, whose run report prints the command line's key table
# (CLI11) and the machine's facts (cpuinfo). Never Arrow C++: the module reads Parquet through pyarrow (and Arrow IPC
# with the core's reader), so a build with DTWC_ENABLE_ARROW=ON, where dtwc_cli links dtwc_io, stops at configure
# (CMakeLists.txt); the wheel builds with DTWC_ENABLE_YAML=OFF, so no fkYAML (pyproject.toml).
target_link_libraries(_dtwcpp_core PRIVATE dtwc_cli project_options)
# The MSVC `/openmp:experimental` flag (and `-fopenmp`/libomp on Clang/GCC) is
# attached DIRECTLY to dtwc_core as a PUBLIC usage requirement (dtwc/CMakeLists.txt),
# so linking `dtwc_core` (through dtwc_cli) is enough for this module to compile
# against the OpenMP runtime (`_OPENMP`, `omp_get_max_threads`). `project_options`
# is kept in the link for its other usage requirements; the OpenMP behaviour does
# not depend on it. Without the flag on this module `_OPENMP` is undefined, so
# `dtwcpp.OPENMP_AVAILABLE` would report False even where the static library was
# built with OpenMP.

install(TARGETS _dtwcpp_core DESTINATION dtwcpp)
