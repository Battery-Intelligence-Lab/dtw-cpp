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

option(DTWC_BUILD_EXAMPLES "Build example programs" OFF)
option(DTWC_BUILD_TESTING "Build unit tests" OFF)
option(DTWC_BUILD_BENCHMARK "Build benchmarks" OFF)
option(DTWC_BUILD_PYTHON "Build Python bindings (nanobind)" OFF)
option(DTWC_BUILD_MATLAB "Build MATLAB MEX bindings" OFF)

option(DTWC_ENABLE_GUROBI "Enable Gurobi MIP solver" ON)
option(DTWC_ENABLE_HIGHS "Enable HiGHS MIP solver" ON)
option(DTWC_ENABLE_SIMD "Enable SIMD acceleration via Google Highway" OFF)
option(DTWC_ENABLE_MPI "Enable MPI distributed computing" OFF)
option(DTWC_ENABLE_CUDA "Enable CUDA GPU acceleration" OFF)

# CUDA must be detected at directory scope (enable_language cannot be called
# inside a function). Do it here before dtwc_setup_dependencies().
if(DTWC_ENABLE_CUDA)
  # ── MSVC / Visual Studio generator CUDA fix ──────────────────────────────
  # MSBuild picks the latest CUDA .props/.targets in BuildCustomizations.
  # Each version expects a CUDA_PATH_Vxx_y env-var (e.g. CUDA_PATH_V13_2).
  # If a newer CUDA was partially installed (VS integration only, no env-var),
  # MSBuild gets an empty CudaToolkitDir and the build fails.
  #
  # Fix: scan the CUDA install root for all installed versions and set any
  # missing CUDA_PATH_Vxx_y env-vars so MSBuild can resolve any version.
  if(WIN32)
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

  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER_VERSION}")

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

# include(cmake/PreventInSourceBuilds.cmake)
dtwc_setup_dependencies()

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_17)

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
endif()

# Adding the tests:
include(CTest)
if(DTWC_BUILD_TESTING)
  include(cmake/Coverage.cmake)
  add_subdirectory(tests)
endif()

if(DTWC_BUILD_EXAMPLES)
  add_subdirectory(examples)
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