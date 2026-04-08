cmake_minimum_required(VERSION 3.21)

# This CMake file is adapted from https://github.com/cpp-best-practices/cmake_template

# This template attempts to be "fetch_content"-able
# so that it works well with tools like CPM or other
# manual dependency management

# Read version from single source of truth
file(STRINGS "${CMAKE_CURRENT_SOURCE_DIR}/VERSION" DTWC_VERSION_STRING)

project(DTWC++ VERSION "${DTWC_VERSION_STRING}"
  DESCRIPTION "A C++ library for fast Dynamic Time Warping Clustering"
  HOMEPAGE_URL https://battery-intelligence-lab.github.io/dtw-cpp/
  LANGUAGES CXX C)

include(cmake/StandardProjectSettings.cmake)
include(CMakeDependentOption)

option(DTWC_BUILD_EXAMPLES "Build example programs" OFF)
option(DTWC_BUILD_TESTING "Build unit tests" OFF)
option(DTWC_BUILD_BENCHMARK "Build benchmarks" OFF)
option(DTWC_BUILD_PYTHON "Build Python bindings (nanobind)" OFF)
option(DTWC_BUILD_MATLAB "Build MATLAB MEX bindings" OFF)
option(DTWC_DEV_MODE "Enable developer-only warnings, analyzers, and sanitizer options" OFF)

option(DTWC_ENABLE_GUROBI "Enable Gurobi MIP solver" ON)
option(DTWC_ENABLE_HIGHS "Enable HiGHS MIP solver" ON)
option(DTWC_ENABLE_MPI "Enable MPI distributed computing" OFF)
option(DTWC_ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)
option(DTWC_ENABLE_YAML "Enable YAML configuration file support via yaml-cpp" OFF)
option(DTWC_ENABLE_ARROW "Enable Apache Arrow IPC for zero-copy data loading" OFF)

# CUDA must be detected at directory scope (enable_language cannot be called
# inside a function). Do it here before dtwc_setup_dependencies().
if(DTWC_ENABLE_CUDA)
  if(APPLE)
    message(WARNING
      "CUDA is not supported on macOS. Apple dropped NVIDIA driver support "
      "in macOS 10.14 (Mojave). Disabling CUDA.\n"
      "  For GPU acceleration, use Linux or Windows with an NVIDIA GPU.")
    set(DTWC_ENABLE_CUDA OFF)
  elseif(WIN32)
  # ── MSVC / Visual Studio generator CUDA fix ──────────────────────────────
  # MSBuild picks the latest CUDA .props/.targets in BuildCustomizations.
  # Each version expects a CUDA_PATH_Vxx_y env-var (e.g. CUDA_PATH_V13_2).
  # If a newer CUDA was partially installed (VS integration only, no env-var),
  # MSBuild gets an empty CudaToolkitDir and the build fails.
  #
  # Fix: scan the CUDA install root for all installed versions and set any
  # missing CUDA_PATH_Vxx_y env-vars so MSBuild can resolve any version.
    # Determine the CUDA base directory from CUDA_PATH or well-known location.
    if(DEFINED ENV{CUDA_PATH})
      file(TO_CMAKE_PATH "$ENV{CUDA_PATH}" _cuda_active)
      get_filename_component(_cuda_base "${_cuda_active}" DIRECTORY)
    else()
      set(_cuda_base "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
    endif()

    if(EXISTS "${_cuda_base}")
      file(GLOB _cuda_versions LIST_DIRECTORIES true "${_cuda_base}/v*")
      set(_cuda_best_ver "0.0")
      set(_cuda_best_path "")
      foreach(_vdir ${_cuda_versions})
        if(IS_DIRECTORY "${_vdir}" AND EXISTS "${_vdir}/bin/nvcc.exe")
          get_filename_component(_vname "${_vdir}" NAME)  # e.g. "v12.2"
          string(REGEX MATCH "([0-9]+)\\.([0-9]+)" _vm "${_vname}")
          if(_vm)
            set(_major "${CMAKE_MATCH_1}")
            set(_minor "${CMAKE_MATCH_2}")
            # Set the versioned env-var MSBuild expects (e.g. CUDA_PATH_V12_2)
            set(_env_name "CUDA_PATH_V${_major}_${_minor}")
            if(NOT DEFINED ENV{${_env_name}})
              file(TO_CMAKE_PATH "${_vdir}" _vdir_cmake)
              set(ENV{${_env_name}} "${_vdir_cmake}")
              message(STATUS "Set missing ${_env_name} = ${_vdir_cmake}")
            endif()
            # Track the best (newest) version that the driver can run.
            # nvidia-smi reports the max supported toolkit version in the
            # driver; building with a newer toolkit causes runtime failures.
            # Prefer CUDA_PATH's version if set, otherwise pick the newest.
            if("${_major}.${_minor}" VERSION_GREATER "${_cuda_best_ver}")
              set(_cuda_best_ver "${_major}.${_minor}")
              set(_cuda_best_path "${_vdir}")
            endif()
          endif()
        endif()
      endforeach()

      # If CUDA_PATH is not set, point it at the newest version found.
      if(NOT DEFINED ENV{CUDA_PATH} AND NOT "${_cuda_best_path}" STREQUAL "")
        file(TO_CMAKE_PATH "${_cuda_best_path}" _cuda_best_cmake)
        set(ENV{CUDA_PATH} "${_cuda_best_cmake}")
        message(STATUS "Set CUDA_PATH = ${_cuda_best_cmake} (auto-detected)")
      endif()
    endif()
  endif()

  # ── Linux CUDA hints ──────────────────────────────────────────────────
  # Standard CUDA Toolkit installs land in /usr/local/cuda (a symlink).
  # If nvcc is not on PATH, help check_language() find it.
  if(UNIX AND NOT APPLE AND NOT DEFINED CMAKE_CUDA_COMPILER)
    if(DEFINED ENV{CUDA_PATH})
      set(_nvcc_hint "$ENV{CUDA_PATH}/bin/nvcc")
    else()
      set(_nvcc_hint "/usr/local/cuda/bin/nvcc")
    endif()
    if(EXISTS "${_nvcc_hint}")
      set(CMAKE_CUDA_COMPILER "${_nvcc_hint}" CACHE FILEPATH "CUDA compiler")
      message(STATUS "Auto-detected nvcc at ${_nvcc_hint}")
    endif()
  endif()

  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    # Set default CUDA arch list before enable_language(CUDA) — CMake reads this at language-enable time.
    # Covers the common HPC/cloud GPU fleet: V100(70), A100(80), RTX Ampere(86), L40s/Ada(89), H100(90).
    # P100(60) is omitted: EOL at most HPC centers, roughly doubles compile time.
    # Override with -DCMAKE_CUDA_ARCHITECTURES=... or -DDTWC_CUDA_ARCH_LIST=... on the command line.
    set(DTWC_CUDA_ARCH_LIST "70;80;86;89;90" CACHE STRING
        "CUDA architectures to compile for (default covers V100 through H100)")
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES OR CMAKE_CUDA_ARCHITECTURES STREQUAL "")
      set(CMAKE_CUDA_ARCHITECTURES "${DTWC_CUDA_ARCH_LIST}" CACHE STRING
          "CUDA architectures (set from DTWC_CUDA_ARCH_LIST)" FORCE)
    endif()

    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    # Set CUDA C++ standard to match the host (C++20)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)

    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER_VERSION}")
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

    # ── Persist toolkit path for MSBuild at build time ─────────────────────
    # set(ENV{...}) only affects the CMake process; the MSBuild invocation
    # from "cmake --build" is a separate process that won't see it.
    # Write a Directory.Build.props so MSBuild can resolve CudaToolkitDir
    # even when the versioned env-var (CUDA_PATH_Vxx_y) is absent.
    #
    # IMPORTANT: The VS generator always picks the newest CUDA .targets in
    # BuildCustomizations. If that version is newer than the installed GPU
    # driver supports, builds succeed but binaries crash at runtime.
    # We pin CudaToolkitCustomDir to CUDA_PATH (the user's chosen version)
    # when it differs from what the VS generator selected.
    if(WIN32 AND CMAKE_GENERATOR MATCHES "Visual Studio")
      # Determine the toolkit dir from CUDA_PATH if set, otherwise from
      # the compiler path that check_language found.
      if(DEFINED ENV{CUDA_PATH})
        file(TO_CMAKE_PATH "$ENV{CUDA_PATH}" _cuda_toolkit_dir)
      else()
        get_filename_component(_cuda_toolkit_dir "${CMAKE_CUDA_COMPILER}" DIRECTORY)  # .../bin
        get_filename_component(_cuda_toolkit_dir "${_cuda_toolkit_dir}" DIRECTORY)     # .../vX.Y
      endif()
      file(TO_NATIVE_PATH "${_cuda_toolkit_dir}\\" _cuda_toolkit_native)
      set(_dbp "${CMAKE_BINARY_DIR}/Directory.Build.props")
      if(NOT EXISTS "${_dbp}")
        file(WRITE "${_dbp}"
"<Project>\n\
  <PropertyGroup>\n\
    <CudaToolkitCustomDir>${_cuda_toolkit_native}</CudaToolkitCustomDir>\n\
  </PropertyGroup>\n\
</Project>\n")
        message(STATUS "Generated ${_dbp} — pinned CudaToolkitCustomDir to ${_cuda_toolkit_native}")
      endif()
    endif()
  else()
    message(WARNING "CUDA requested but nvcc not found — disabling.\n"
      "  Hints: set CUDA_PATH env-var, or pass -DCMAKE_CUDA_COMPILER=<path-to-nvcc>")
    set(DTWC_ENABLE_CUDA OFF)
  endif()
endif()

include(cmake/Dependencies.cmake) # Include external projects
include(cmake/FindGUROBI.cmake)
include(cmake/ProjectOptions.cmake)

include(cmake/PreventInSourceBuilds.cmake)
dtwc_setup_dependencies()
dtwc_setup_options()
dtwc_global_options()
dtwc_local_options()

add_library(project_warnings ALIAS dtwc_warnings)
add_library(project_options ALIAS dtwc_options)

add_subdirectory(dtwc bin)

if(PROJECT_IS_TOP_LEVEL)
  add_executable(dtwc_main
    dtwc/main.cpp
  )

  target_link_libraries(dtwc_main
    PRIVATE
    dtwc++
    project_warnings
    project_options
  )

  add_executable(dtwc_cl
    dtwc/dtwc_cl.cpp
  )

  target_link_libraries(dtwc_cl
    PRIVATE
    dtwc++
    CLI11::CLI11
    project_warnings
    project_options
  )

  # Optional YAML config support for CLI tool
  if(DTWC_ENABLE_YAML AND TARGET yaml-cpp::yaml-cpp)
    target_link_libraries(dtwc_cl PRIVATE yaml-cpp::yaml-cpp)
    target_compile_definitions(dtwc_cl PRIVATE DTWC_HAS_YAML)
  endif()
endif()

# Adding the tests:
include(CTest)
if(DTWC_BUILD_TESTING)
  include(cmake/Coverage.cmake)
  add_subdirectory(tests)
endif()

if(DTWC_BUILD_EXAMPLES)
  add_subdirectory(examples/cpp)
endif()

if(DTWC_BUILD_BENCHMARK)
  add_subdirectory(benchmarks)
endif()

if(DTWC_BUILD_PYTHON)
  add_subdirectory(python)
endif()

if(DTWC_BUILD_MATLAB)
  add_subdirectory(bindings/matlab)
endif()

cpm_licenses_create_disclaimer_target(write-licenses "${CMAKE_CURRENT_SOURCE_DIR}/bin/third_party.txt" "${CPM_PACKAGES}")

# ── Configuration summary ──────────────────────────────────────────────────
if(PROJECT_IS_TOP_LEVEL)
  # Determine what was actually found
  set(_omp_status "OFF")
  # OpenMP is found inside dtwc/CMakeLists.txt — check the compile definition
  # or the imported target, which are visible in parent scope.
  get_target_property(_dtwc_defs dtwc++ INTERFACE_COMPILE_DEFINITIONS)
  if(_dtwc_defs MATCHES "DTWC_HAS_OPENMP")
    set(_omp_status "ON")
    # Try to extract version from cache (set by FindOpenMP)
    if(DEFINED CACHE{OpenMP_CXX_SPEC_DATE})
      set(_omp_status "ON (spec ${OpenMP_CXX_SPEC_DATE})")
    endif()
  endif()

  set(_gurobi_status "OFF")
  if(TARGET Gurobi::GurobiCXX)
    set(_gurobi_status "ON (${GUROBI_HOME})")
  elseif(DTWC_ENABLE_GUROBI)
    set(_gurobi_status "REQUESTED BUT NOT FOUND")
  endif()

  set(_highs_status "OFF")
  if(TARGET highs::highs)
    set(_highs_status "ON")
  endif()

  set(_cuda_status "OFF")
  if(DTWC_ENABLE_CUDA AND CMAKE_CUDA_COMPILER)
    set(_cuda_status "ON (${CMAKE_CUDA_COMPILER_VERSION})")
  elseif(DTWC_ENABLE_CUDA)
    set(_cuda_status "REQUESTED BUT NOT FOUND")
  endif()

  set(_mpi_status "OFF")
  if(DTWC_ENABLE_MPI AND TARGET MPI::MPI_CXX)
    set(_mpi_status "ON")
  elseif(DTWC_ENABLE_MPI)
    set(_mpi_status "REQUESTED BUT NOT FOUND")
  endif()

  message(STATUS "")
  message(STATUS "╔══════════════════════════════════════════════════╗")
  message(STATUS "║  DTWC++ ${DTWC_VERSION_STRING} Configuration Summary")
  message(STATUS "╠══════════════════════════════════════════════════╣")
  message(STATUS "║  Compiler:  ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
  message(STATUS "║  OpenMP:    ${_omp_status}")
  message(STATUS "║  CUDA:      ${_cuda_status}")
  message(STATUS "║  MPI:       ${_mpi_status}")
  message(STATUS "║  Gurobi:    ${_gurobi_status}")
  message(STATUS "║  HiGHS:     ${_highs_status}")
  message(STATUS "║  Testing:   ${DTWC_BUILD_TESTING}")
  message(STATUS "╚══════════════════════════════════════════════════╝")
  message(STATUS "")

  # Warn about missing optional deps that were requested
  if(DTWC_ENABLE_GUROBI AND NOT TARGET Gurobi::GurobiCXX)
    message(WARNING
      "Gurobi requested (DTWC_ENABLE_GUROBI=ON) but not found.\n"
      "  MIP-based clustering will fall back to HiGHS (if available).\n"
      "  To fix: set GUROBI_HOME env-var to your Gurobi installation,\n"
      "  e.g., GUROBI_HOME=C:/gurobi1301/win64 or /opt/gurobi1301/linux64\n"
      "  Or pass -DDTWC_ENABLE_GUROBI=OFF to silence this warning.")
  endif()

  if(NOT TARGET Gurobi::GurobiCXX AND NOT TARGET highs::highs)
    message(WARNING
      "No MIP solver available — MIP-based clustering (Method::MIP) will fail at runtime.\n"
      "  Enable at least one: -DDTWC_ENABLE_GUROBI=ON or -DDTWC_ENABLE_HIGHS=ON")
  endif()
endif()
