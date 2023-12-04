# include(SystemLink.cmake)
# include(LibFuzzer.cmake)
include(CMakeDependentOption)
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
  option(dtwc_ENABLE_HARDENING "Enable hardening" ON)
  option(dtwc_ENABLE_COVERAGE "Enable coverage reporting" OFF)
  cmake_dependent_option(
    dtwc_ENABLE_GLOBAL_HARDENING
    "Attempt to push hardening options to built dependencies"
    ON
    dtwc_ENABLE_HARDENING
    OFF)

  dtwc_supports_sanitizers()

  if(NOT PROJECT_IS_TOP_LEVEL OR dtwc_PACKAGING_MAINTAINER_MODE)
    option(dtwc_ENABLE_IPO "Enable IPO/LTO" OFF)
    option(dtwc_WARNINGS_AS_ERRORS "Treat Warnings As Errors" OFF)
    option(dtwc_ENABLE_USER_LINKER "Enable user-selected linker" OFF)
    option(dtwc_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" OFF)
    option(dtwc_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
    option(dtwc_ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" OFF)
    option(dtwc_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
    option(dtwc_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
    option(dtwc_ENABLE_UNITY_BUILD "Enable unity builds" OFF)
    option(dtwc_ENABLE_CLANG_TIDY "Enable clang-tidy" OFF)
    option(dtwc_ENABLE_CPPCHECK "Enable cpp-check analysis" OFF)
    option(dtwc_ENABLE_PCH "Enable precompiled headers" OFF)
    option(dtwc_ENABLE_CACHE "Enable ccache" OFF)
  else()
    option(dtwc_ENABLE_IPO "Enable IPO/LTO" ON)
    option(dtwc_WARNINGS_AS_ERRORS "Treat Warnings As Errors" ON)
    option(dtwc_ENABLE_USER_LINKER "Enable user-selected linker" OFF)
    option(dtwc_ENABLE_SANITIZER_ADDRESS "Enable address sanitizer" ${SUPPORTS_ASAN})
    option(dtwc_ENABLE_SANITIZER_LEAK "Enable leak sanitizer" OFF)
    option(dtwc_ENABLE_SANITIZER_UNDEFINED "Enable undefined sanitizer" ${SUPPORTS_UBSAN})
    option(dtwc_ENABLE_SANITIZER_THREAD "Enable thread sanitizer" OFF)
    option(dtwc_ENABLE_SANITIZER_MEMORY "Enable memory sanitizer" OFF)
    option(dtwc_ENABLE_UNITY_BUILD "Enable unity builds" OFF)
    option(dtwc_ENABLE_CLANG_TIDY "Enable clang-tidy" ON)
    option(dtwc_ENABLE_CPPCHECK "Enable cpp-check analysis" ON)
    option(dtwc_ENABLE_PCH "Enable precompiled headers" OFF)
    option(dtwc_ENABLE_CACHE "Enable ccache" ON)
  endif()

  if(NOT PROJECT_IS_TOP_LEVEL)
    mark_as_advanced(
      dtwc_ENABLE_IPO
      dtwc_WARNINGS_AS_ERRORS
      dtwc_ENABLE_USER_LINKER
      dtwc_ENABLE_SANITIZER_ADDRESS
      dtwc_ENABLE_SANITIZER_LEAK
      dtwc_ENABLE_SANITIZER_UNDEFINED
      dtwc_ENABLE_SANITIZER_THREAD
      dtwc_ENABLE_SANITIZER_MEMORY
      dtwc_ENABLE_UNITY_BUILD
      dtwc_ENABLE_CLANG_TIDY
      dtwc_ENABLE_CPPCHECK
      dtwc_ENABLE_COVERAGE
      dtwc_ENABLE_PCH
      dtwc_ENABLE_CACHE)
  endif()

  # dtwc_check_libfuzzer_support(LIBFUZZER_SUPPORTED)
  # if(LIBFUZZER_SUPPORTED AND (dtwc_ENABLE_SANITIZER_ADDRESS OR dtwc_ENABLE_SANITIZER_THREAD OR dtwc_ENABLE_SANITIZER_UNDEFINED))
  #   set(DEFAULT_FUZZER ON)
  # else()
  #   set(DEFAULT_FUZZER OFF)
  # endif()

  # option(dtwc_BUILD_FUZZ_TESTS "Enable fuzz testing executable" ${DEFAULT_FUZZER})

endmacro()

macro(dtwc_global_options)
  if(dtwc_ENABLE_IPO)
    include(cmake/InterproceduralOptimization.cmake)
    dtwc_enable_ipo()
  endif()

  dtwc_supports_sanitizers()

  # if(dtwc_ENABLE_HARDENING AND dtwc_ENABLE_GLOBAL_HARDENING)
  #   include(Hardening.cmake)
  #   if(NOT SUPPORTS_UBSAN 
  #      OR dtwc_ENABLE_SANITIZER_UNDEFINED
  #      OR dtwc_ENABLE_SANITIZER_ADDRESS
  #      OR dtwc_ENABLE_SANITIZER_THREAD
  #      OR dtwc_ENABLE_SANITIZER_LEAK)
  #     set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
  #   else()
  #     set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
  #   endif()
  #   message("${dtwc_ENABLE_HARDENING} ${ENABLE_UBSAN_MINIMAL_RUNTIME} ${dtwc_ENABLE_SANITIZER_UNDEFINED}")
  #   dtwc_enable_hardening(dtwc_options ON ${ENABLE_UBSAN_MINIMAL_RUNTIME})
  # endif()
endmacro()

macro(dtwc_local_options)
  if(PROJECT_IS_TOP_LEVEL)
    include(cmake/StandardProjectSettings.cmake)
  endif()

  add_library(dtwc_warnings INTERFACE)
  add_library(dtwc_options INTERFACE)

  include(cmake/CompilerWarnings.cmake)
  dtwc_set_project_warnings(
    dtwc_warnings
    ${dtwc_WARNINGS_AS_ERRORS}
    ""
    ""
    ""
    "")

  if(dtwc_ENABLE_USER_LINKER)
    include(cmake/Linker.cmake)
    configure_linker(dtwc_options)
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

  if(dtwc_ENABLE_COVERAGE)
    include(cmake/Coverage.cmake)
    dtwc_enable_coverage(dtwc_options)
  endif()

  if(dtwc_WARNINGS_AS_ERRORS)
    check_cxx_compiler_flag("-Wl,--fatal-warnings" LINKER_FATAL_WARNINGS)
    if(LINKER_FATAL_WARNINGS)
      # This is not working consistently, so disabling for now
      # target_link_options(dtwc_options INTERFACE -Wl,--fatal-warnings)
    endif()
  endif()

  if(dtwc_ENABLE_HARDENING AND NOT dtwc_ENABLE_GLOBAL_HARDENING)
    include(cmake/Hardening.cmake)
    if(NOT SUPPORTS_UBSAN 
       OR dtwc_ENABLE_SANITIZER_UNDEFINED
       OR dtwc_ENABLE_SANITIZER_ADDRESS
       OR dtwc_ENABLE_SANITIZER_THREAD
       OR dtwc_ENABLE_SANITIZER_LEAK)
      set(ENABLE_UBSAN_MINIMAL_RUNTIME FALSE)
    else()
      set(ENABLE_UBSAN_MINIMAL_RUNTIME TRUE)
    endif()
    dtwc_enable_hardening(dtwc_options OFF ${ENABLE_UBSAN_MINIMAL_RUNTIME})
  endif()

endmacro()