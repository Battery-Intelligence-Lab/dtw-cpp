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
set(CMAKE_CXX_STANDARD 17)
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
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:Release>:/fp:fast>>)
  add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:RelWithDebInfo>:/fp:fast>>)
elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  # -fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros
  # give ~95% of -ffast-math (FMA, reassociation, reciprocal) without -ffinite-math-only.
  foreach(_flag -fno-math-errno -fno-trapping-math -freciprocal-math -fassociative-math -fno-signed-zeros)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:Release>:${_flag}>>)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:RelWithDebInfo>:${_flag}>>)
  endforeach()
endif()

# Native architecture tuning — unlocks AVX2/AVX-512/NEON auto-vectorization.
# Disabled for Python wheels (DTWC_BUILD_PYTHON) to keep wheel binaries portable.
# Disabled when consumed as a sub-project (PROJECT_IS_TOP_LEVEL=OFF) so the
# parent project controls its own arch flags.
option(DTWC_ENABLE_NATIVE_ARCH "Tune for the host CPU architecture (-march=native / /arch:AVX2)" ON)
if(DTWC_ENABLE_NATIVE_ARCH AND PROJECT_IS_TOP_LEVEL AND NOT DTWC_BUILD_PYTHON)
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:Release>:/arch:AVX2>>)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:RelWithDebInfo>:/arch:AVX2>>)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:Release>:-march=native>>)
    add_compile_options($<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:RelWithDebInfo>:-march=native>>)
  endif()
  message(STATUS "Native arch tuning enabled (DTWC_ENABLE_NATIVE_ARCH=ON)")
else()
  message(STATUS "Native arch tuning disabled — portable binary mode")
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
