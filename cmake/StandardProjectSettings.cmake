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


# Selected floating-point relaxations for optimised builds (X-15).
#
# These are computed here but NOT applied here. `add_compile_options()` is
# directory-scope and is inherited by every subdirectory added afterwards —
# including the ones CPM creates for fetched dependencies. That is ledger row
# S-03, and it was real: before this change 146 dependency translation units
# were compiled with -fassociative-math, 31 of them HiGHS, whose simplex and
# interior-point code is exactly where reassociating a floating-point sum can
# change a pivot or a tolerance comparison. Catch2 was affected too, so the
# framework's own float matchers were built relaxed.
#
# The flags now ride on the dtwc_options INTERFACE target, which only our own
# targets link (cmake/ProjectOptions.cmake). dtwc_options is created *after*
# dtwc_setup_dependencies() has added the fetched projects, so a dependency
# cannot pick these up even by accident.
#
# DTWC_FP_MODEL selects the policy. It is a CACHE variable so that it appears in
# CMakeCache.txt and is harvested into machine records by scripts/machine_facts.py
# (X-26) — a plain set() would be invisible to them.
set(DTWC_FP_MODEL "fast" CACHE STRING
    "Floating-point policy for optimised builds: 'fast' (selected relaxations) or 'strict' (none)")
set_property(CACHE DTWC_FP_MODEL PROPERTY STRINGS "fast" "strict")
if(NOT DTWC_FP_MODEL STREQUAL "fast" AND NOT DTWC_FP_MODEL STREQUAL "strict")
  message(FATAL_ERROR
    "DTWC_FP_MODEL must be 'fast' or 'strict', not '${DTWC_FP_MODEL}'. "
    "'fast' applies the selected relaxations documented in "
    "cmake/StandardProjectSettings.cmake; 'strict' applies none of them and is "
    "what conformance output is pinned against.")
endif()

# DTWC_FP_FLAGS is consumed by dtwc_local_options() in cmake/ProjectOptions.cmake.
set(DTWC_FP_FLAGS "")
if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
  if(DTWC_FP_MODEL STREQUAL "fast")
    # /fp:precise preserves std::isnan() semantics (required by missing_utils.hpp).
    # /fp:contract enables FMA contraction for performance.
    list(APPEND DTWC_FP_FLAGS /fp:precise /fp:contract)
  else()
    # /fp:strict also disables FMA contraction, so the two are not independent.
    list(APPEND DTWC_FP_FLAGS /fp:strict)
  endif()
  # /Gy: function-level linking — lets the linker eliminate/fold unused COMDATs.
  # Not a floating-point flag; unaffected by DTWC_FP_MODEL.
  list(APPEND DTWC_FP_FLAGS /Gy)
elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  if(DTWC_FP_MODEL STREQUAL "fast")
    # The selected relaxations cover no errno/trapping, reciprocal and associative
    # transformations, no signed-zero distinction, a fixed rounding mode, and no
    # signalling NaNs. GCC/Clang use explicit flags rather than -ffast-math.
    # `-ffinite-math-only` is deliberately omitted — it breaks std::isnan() under GCC/Clang.
    # See missing_utils.hpp for NaN handling design notes.
    # -fno-rounding-math: assume default round-to-nearest (code never calls fesetround).
    # -fno-signaling-nans: treat SNaNs as quiet NaNs (only quiet NaN is used in this project).
    list(APPEND DTWC_FP_FLAGS
      -fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math
      -fno-signed-zeros -fno-rounding-math -fno-signaling-nans)
  endif()
endif()
message(STATUS "Floating-point model: ${DTWC_FP_MODEL}")

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

# Reproducible-build flags (opt-in).
#
# Enables bit-identical binaries across build hosts when the following are also
# ensured externally:
#   - SOURCE_DATE_EPOCH is exported (gcc/clang read this env var automatically
#     and use it to stabilise __DATE__ / __TIME__ macros — no CMake plumbing
#     needed).
#   - Build directory is deterministic (e.g. /build rather than
#     /home/user/project/build).
#
# -ffile-prefix-map rewrites absolute source paths in debug info AND in
# __FILE__ macros, so the binaries don't embed the developer's home directory.
# Applied to Clang and GCC; MSVC has no equivalent single flag (use /d1 flags
# if ever required).
#
# This is opt-in because the flag slows compilation slightly and alters debug
# info paths, which can surprise local IDE debuggers.
option(DTWC_REPRODUCIBLE_BUILD
  "Strip absolute source paths from debug info and __FILE__ (for Debian-style packaging)"
  OFF)

if(DTWC_REPRODUCIBLE_BUILD)
  if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(
      $<$<COMPILE_LANGUAGE:C,CXX>:-ffile-prefix-map=${CMAKE_SOURCE_DIR}=.>
      $<$<COMPILE_LANGUAGE:C,CXX>:-ffile-prefix-map=${CMAKE_BINARY_DIR}=./build>)
    message(STATUS "Reproducible build: -ffile-prefix-map enabled")
  else()
    message(STATUS "Reproducible build: not supported on ${CMAKE_CXX_COMPILER_ID} (no-op)")
  endif()
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
