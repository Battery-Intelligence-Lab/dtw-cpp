include(CheckCXXCompilerFlag)


macro(dtwc_supports_sanitizers)
  if((CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" OR CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*") AND NOT WIN32)
    set(SUPPORTS_UBSAN ON)
  else()
    set(SUPPORTS_UBSAN OFF)
  endif()

  if((CMAKE_CXX_COMPILER_ID MATCHES ".*Clang.*" OR CMAKE_CXX_COMPILER_ID MATCHES ".*GNU.*") AND WIN32)
    set(SUPPORTS_ASAN OFF)
  else()
    set(SUPPORTS_ASAN ON)
  endif()
endmacro()

# B-14 (adopted 2026-09-07): the maintainer options carry the same `DTWC_` prefix
# as every other cache variable in the project. The lowercase `dtwc_` spellings
# are still accepted for one release.
#
# A legacy value seeds the new option, and the stale cache entry is then removed,
# so the notice appears once for each place that actually sets one rather than on
# every configure for the rest of the build directory's life. The warning is
# raised only when the legacy value differs from the new default: an existing
# build tree carries all thirteen legacy entries left by `option()` itself, and
# warning about those would be noise about something the user never chose.
macro(_dtwc_maintainer_option name docstring default)
  set(_dtwc_opt_default "${default}")
  if(DEFINED dtwc_${name})
    if(NOT "${dtwc_${name}}" STREQUAL "${default}")
      message(WARNING
        "dtwc_${name} is deprecated and will be removed after the next release; "
        "use DTWC_${name}. Honouring dtwc_${name}=${dtwc_${name}} for now.")
      set(_dtwc_opt_default "${dtwc_${name}}")
    endif()
    unset(dtwc_${name} CACHE)
  endif()
  option(DTWC_${name} "${docstring}" ${_dtwc_opt_default})
  unset(_dtwc_opt_default)
endmacro()

macro(dtwc_setup_options)
  dtwc_supports_sanitizers()

  _dtwc_maintainer_option(ENABLE_IPO "Enable IPO/LTO for dtwc targets" ${PROJECT_IS_TOP_LEVEL})
  _dtwc_maintainer_option(ENABLE_COMPILER_WARNINGS "Enable maintainer warning set for dtwc targets" ${DTWC_DEV_MODE})
  _dtwc_maintainer_option(WARNINGS_AS_ERRORS "Treat maintainer warnings as errors" ${DTWC_DEV_MODE})
  _dtwc_maintainer_option(ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
  _dtwc_maintainer_option(ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
  _dtwc_maintainer_option(ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" OFF)
  _dtwc_maintainer_option(ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
  _dtwc_maintainer_option(ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
  _dtwc_maintainer_option(ENABLE_UNITY_BUILD "Enable unity builds" OFF)
  _dtwc_maintainer_option(ENABLE_CLANG_TIDY "Enable clang-tidy analysis" ${DTWC_DEV_MODE})
  _dtwc_maintainer_option(ENABLE_CPPCHECK "Enable cppcheck analysis" ${DTWC_DEV_MODE})
  _dtwc_maintainer_option(ENABLE_PCH "Enable precompiled headers" OFF)
  _dtwc_maintainer_option(ENABLE_CACHE "Enable ccache" ${DTWC_DEV_MODE})

  if(NOT PROJECT_IS_TOP_LEVEL OR NOT DTWC_DEV_MODE)
    mark_as_advanced(
      DTWC_ENABLE_IPO
      DTWC_ENABLE_COMPILER_WARNINGS
      DTWC_WARNINGS_AS_ERRORS
      DTWC_ENABLE_SANITIZER_ADDRESS
      DTWC_ENABLE_SANITIZER_LEAK
      DTWC_ENABLE_SANITIZER_UNDEFINED
      DTWC_ENABLE_SANITIZER_THREAD
      DTWC_ENABLE_SANITIZER_MEMORY
      DTWC_ENABLE_UNITY_BUILD
      DTWC_ENABLE_CLANG_TIDY
      DTWC_ENABLE_CPPCHECK
      DTWC_ENABLE_PCH
      DTWC_ENABLE_CACHE)
  endif()
endmacro()

macro(dtwc_global_options)
  if(DTWC_ENABLE_IPO)
    include(cmake/InterproceduralOptimization.cmake)
    dtwc_enable_ipo()
  endif()
endmacro()

macro(dtwc_local_options)
  add_library(dtwc_warnings INTERFACE)
  add_library(dtwc_options INTERFACE)
  target_compile_features(dtwc_options INTERFACE cxx_std_20)

  # Floating-point policy (X-15, closes S-03). DTWC_FP_FLAGS is computed in
  # cmake/StandardProjectSettings.cmake from DTWC_FP_MODEL. Applying it here
  # rather than with a directory-scope add_compile_options() is the whole point:
  # this target is reached only by things that link it, so the flags cannot leak
  # into fetched dependencies — HiGHS, Catch2 and llfio were all picking them up.
  # Optimised configurations only, matching the previous CONFIG generator
  # expressions; Debug is unaffected.
  foreach(_dtwc_fp_flag IN LISTS DTWC_FP_FLAGS)
    target_compile_options(dtwc_options INTERFACE
      $<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:Release>:${_dtwc_fp_flag}>>
      $<$<COMPILE_LANGUAGE:C,CXX>:$<$<CONFIG:RelWithDebInfo>:${_dtwc_fp_flag}>>)
  endforeach()
  unset(_dtwc_fp_flag)

  if(DTWC_ENABLE_COMPILER_WARNINGS)
    include(cmake/CompilerWarnings.cmake)
    dtwc_set_project_warnings(
      dtwc_warnings
      ${DTWC_WARNINGS_AS_ERRORS}
      ""
      ""
      ""
      "")
  endif()

  include(cmake/Sanitizers.cmake)
  dtwc_enable_sanitizers(
    dtwc_options
    ${DTWC_ENABLE_SANITIZER_ADDRESS}
    ${DTWC_ENABLE_SANITIZER_LEAK}
    ${DTWC_ENABLE_SANITIZER_UNDEFINED}
    ${DTWC_ENABLE_SANITIZER_THREAD}
    ${DTWC_ENABLE_SANITIZER_MEMORY})

  set_target_properties(dtwc_options PROPERTIES UNITY_BUILD ${DTWC_ENABLE_UNITY_BUILD})

  if(DTWC_ENABLE_PCH)
    target_precompile_headers(
      dtwc_options
      INTERFACE
      <vector>
      <string>
      <utility>)
  endif()

  if(DTWC_ENABLE_CACHE)
    include(cmake/Cache.cmake)
    dtwc_enable_cache()
  endif()

  include(cmake/StaticAnalyzers.cmake)
  if(DTWC_ENABLE_CLANG_TIDY)
    dtwc_enable_clang_tidy(dtwc_options ${DTWC_WARNINGS_AS_ERRORS})
  endif()

  if(DTWC_ENABLE_CPPCHECK)
    dtwc_enable_cppcheck(${DTWC_WARNINGS_AS_ERRORS} "" # override cppcheck options
    )
  endif()

  if(DTWC_WARNINGS_AS_ERRORS)
    check_cxx_compiler_flag("-Wl,--fatal-warnings" LINKER_FATAL_WARNINGS)
    if(LINKER_FATAL_WARNINGS)
      # This is not working consistently, so disabling for now
      # target_link_options(dtwc_options INTERFACE -Wl,--fatal-warnings)
    endif()
  endif()
endmacro()
