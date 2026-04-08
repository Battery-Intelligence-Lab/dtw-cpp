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

macro(dtwc_setup_options)
  dtwc_supports_sanitizers()

  option(dtwc_ENABLE_IPO "Enable IPO/LTO for dtwc targets" ${PROJECT_IS_TOP_LEVEL})
  option(dtwc_ENABLE_COMPILER_WARNINGS "Enable maintainer warning set for dtwc targets" ${DTWC_DEV_MODE})
  option(dtwc_WARNINGS_AS_ERRORS "Treat maintainer warnings as errors" ${DTWC_DEV_MODE})
  option(dtwc_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
  option(dtwc_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
  option(dtwc_ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" OFF)
  option(dtwc_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
  option(dtwc_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
  option(dtwc_ENABLE_UNITY_BUILD "Enable unity builds" OFF)
  option(dtwc_ENABLE_CLANG_TIDY "Enable clang-tidy analysis" ${DTWC_DEV_MODE})
  option(dtwc_ENABLE_CPPCHECK "Enable cppcheck analysis" ${DTWC_DEV_MODE})
  option(dtwc_ENABLE_PCH "Enable precompiled headers" OFF)
  option(dtwc_ENABLE_CACHE "Enable ccache" ${DTWC_DEV_MODE})

  if(NOT PROJECT_IS_TOP_LEVEL OR NOT DTWC_DEV_MODE)
    mark_as_advanced(
      dtwc_ENABLE_IPO
      dtwc_ENABLE_COMPILER_WARNINGS
      dtwc_WARNINGS_AS_ERRORS
      dtwc_ENABLE_SANITIZER_ADDRESS
      dtwc_ENABLE_SANITIZER_LEAK
      dtwc_ENABLE_SANITIZER_UNDEFINED
      dtwc_ENABLE_SANITIZER_THREAD
      dtwc_ENABLE_SANITIZER_MEMORY
      dtwc_ENABLE_UNITY_BUILD
      dtwc_ENABLE_CLANG_TIDY
      dtwc_ENABLE_CPPCHECK
      dtwc_ENABLE_PCH
      dtwc_ENABLE_CACHE)
  endif()
endmacro()

macro(dtwc_global_options)
  if(dtwc_ENABLE_IPO)
    include(cmake/InterproceduralOptimization.cmake)
    dtwc_enable_ipo()
  endif()
endmacro()

macro(dtwc_local_options)
  add_library(dtwc_warnings INTERFACE)
  add_library(dtwc_options INTERFACE)
  target_compile_features(dtwc_options INTERFACE cxx_std_20)

  if(dtwc_ENABLE_COMPILER_WARNINGS)
    include(cmake/CompilerWarnings.cmake)
    dtwc_set_project_warnings(
      dtwc_warnings
      ${dtwc_WARNINGS_AS_ERRORS}
      ""
      ""
      ""
      "")
  endif()

  include(cmake/Sanitizers.cmake)
  dtwc_enable_sanitizers(
    dtwc_options
    ${dtwc_ENABLE_SANITIZER_ADDRESS}
    ${dtwc_ENABLE_SANITIZER_LEAK}
    ${dtwc_ENABLE_SANITIZER_UNDEFINED}
    ${dtwc_ENABLE_SANITIZER_THREAD}
    ${dtwc_ENABLE_SANITIZER_MEMORY})

  set_target_properties(dtwc_options PROPERTIES UNITY_BUILD ${dtwc_ENABLE_UNITY_BUILD})

  if(dtwc_ENABLE_PCH)
    target_precompile_headers(
      dtwc_options
      INTERFACE
      <vector>
      <string>
      <utility>)
  endif()

  if(dtwc_ENABLE_CACHE)
    include(cmake/Cache.cmake)
    dtwc_enable_cache()
  endif()

  include(cmake/StaticAnalyzers.cmake)
  if(dtwc_ENABLE_CLANG_TIDY)
    dtwc_enable_clang_tidy(dtwc_options ${dtwc_WARNINGS_AS_ERRORS})
  endif()

  if(dtwc_ENABLE_CPPCHECK)
    dtwc_enable_cppcheck(${dtwc_WARNINGS_AS_ERRORS} "" # override cppcheck options
    )
  endif()

  if(dtwc_WARNINGS_AS_ERRORS)
    check_cxx_compiler_flag("-Wl,--fatal-warnings" LINKER_FATAL_WARNINGS)
    if(LINKER_FATAL_WARNINGS)
      # This is not working consistently, so disabling for now
      # target_link_options(dtwc_options INTERFACE -Wl,--fatal-warnings)
    endif()
  endif()
endmacro()
