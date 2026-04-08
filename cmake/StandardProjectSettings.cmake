# Set a default build type if none was specified
set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS "Setting build type to 'Release' as none was specified.")
  set(CMAKE_BUILD_TYPE
      Release
      CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui, ccmake
  set_property(
    CACHE CMAKE_BUILD_TYPE
    PROPERTY STRINGS
             "Debug"
             "Release"
             "MinSizeRel"
             "RelWithDebInfo")
endif()

# Generate compile_commands.json to make it easier to work with clang based tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# strongly encouraged to enable this globally to avoid conflicts between
# -Wpedantic being enabled and -std=c++20 and -std=gnu++20 for example
# when compiling with PCH enabled
set(CMAKE_CXX_EXTENSIONS OFF)
# Set C++ standard globally — needed for OBJECT libraries (mip-solvers) that
# don't transitively inherit target_compile_features from the main target.
set(CMAKE_CXX_STANDARD 20)
#set(CMAKE_VERBOSE_MAKEFILE ON) -> activate if compilation command for every file is needed.
 


# Enhance error reporting and compiler messages
if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  # Guard with COMPILE_LANGUAGE to prevent leaking to nvcc (which doesn't understand these flags)
  add_compile_options($<$<COMPILE_LANGUAGE:C>:-fcolor-diagnostics> $<$<COMPILE_LANGUAGE:CXX>:-fcolor-diagnostics>)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  add_compile_options($<$<COMPILE_LANGUAGE:C>:-fdiagnostics-color=always>
                      $<$<COMPILE_LANGUAGE:CXX>:-fdiagnostics-color=always>)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC" AND MSVC_VERSION GREATER 1900)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:/diagnostics:column>)
else()
  message(STATUS "No colored compiler diagnostic set for '${CMAKE_CXX_COMPILER_ID}' compiler.")
endif()


# Fast floating-point for Release builds — enables FMA fusion, reordering.
# Explicit sub-flags instead of -ffast-math: preserves std::isnan() by omitting
# -ffinite-math-only, while retaining all other fast-math optimizations.
# See missing_utils.hpp for NaN handling design notes.
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  # /fp:precise preserves std::isnan() semantics (required by missing_utils.hpp).
  # /fp:contract enables FMA contraction for performance.
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:Release>:/fp:precise>>)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:Release>:/fp:contract>>)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:RelWithDebInfo>:/fp:precise>>)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:RelWithDebInfo>:/fp:contract>>)
  # /Gy: function-level linking — lets the linker eliminate/fold unused COMDATs.
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:Release>:/Gy>>)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:RelWithDebInfo>:/Gy>>)
elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # Full safe fast-math subset: all components of -ffast-math except -ffinite-math-only.
  # -ffinite-math-only is deliberately omitted — it breaks std::isnan() under GCC/Clang.
  # -fno-rounding-math: assume default round-to-nearest (code never calls fesetround).
  # -fno-signaling-nans: treat SNaNs as quiet NaNs (only quiet NaN is used in this project).
  foreach(_flag
      -fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math
      -fno-signed-zeros -fno-rounding-math -fno-signaling-nans)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:Release>:${_flag}>>)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:RelWithDebInfo>:${_flag}>>)
  endforeach()
endif()

# Architecture tuning — unlocks AVX2/AVX-512/NEON auto-vectorization.
# Disabled for Python wheels (DTWC_BUILD_PYTHON) to keep wheel binaries portable.
# Disabled when consumed as a sub-project (PROJECT_IS_TOP_LEVEL=OFF) so the
# parent project controls its own arch flags.
#
# DTWC_ARCH_LEVEL overrides -march=native with a specific x86-64 microarchitecture
# level for HPC cross-compile scenarios where the build node differs from compute nodes:
#   ""   (default) — -march=native / /arch:AVX2 as before
#   "v3" — AVX2 + FMA baseline; safe for ALL modern HPC CPUs (Broadwell, Haswell,
#           Cascade Lake, Sapphire/Emerald Rapids, Rome, Genoa, Turin).
#   "v4" — AVX-512 baseline; Cascade Lake Xeon, Sapphire/Emerald Rapids, Genoa, Turin.
#           NOT safe for Broadwell, Haswell, or Rome nodes.
# When DTWC_ENABLE_SIMD=ON, Highway provides runtime dispatch across all ISAs regardless
# of this flag — DTWC_ARCH_LEVEL only affects compiler auto-vectorization paths.
option(DTWC_ENABLE_NATIVE_ARCH "Tune for the host CPU architecture (-march=native / /arch:AVX2)" ON)
set(DTWC_ARCH_LEVEL "" CACHE STRING
    "Override native arch with x86-64 microarchitecture level for HPC: '' (native), 'v3' (AVX2+FMA), 'v4' (AVX-512)")
set_property(CACHE DTWC_ARCH_LEVEL PROPERTY STRINGS "" "v3" "v4")

if(DTWC_ENABLE_NATIVE_ARCH AND PROJECT_IS_TOP_LEVEL AND NOT DTWC_BUILD_PYTHON)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    if(DTWC_ARCH_LEVEL STREQUAL "v4")
      set(_arch_flag /arch:AVX512)
    else()
      set(_arch_flag /arch:AVX2)  # v3 and native both map to AVX2 on MSVC
    endif()
  elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    if(DTWC_ARCH_LEVEL STREQUAL "v3" OR DTWC_ARCH_LEVEL STREQUAL "v4")
      set(_arch_flag -march=x86-64-${DTWC_ARCH_LEVEL})
    else()
      set(_arch_flag -march=native)
    endif()
  endif()
  if(DEFINED _arch_flag)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:Release>:${_arch_flag}>>)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:RelWithDebInfo>:${_arch_flag}>>)
    message(STATUS "Architecture tuning: ${_arch_flag}")
  endif()
else()
  message(STATUS "Architecture tuning disabled — portable binary mode")
endif()

# run vcvarsall when msvc is used
include("${CMAKE_CURRENT_LIST_DIR}/VCEnvironment.cmake")
run_vcvarsall()


message(STATUS "Host system: ${CMAKE_HOST_SYSTEM}")
message(STATUS "Target architecture: ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL ${CMAKE_BINARY_DIR}/bin)
