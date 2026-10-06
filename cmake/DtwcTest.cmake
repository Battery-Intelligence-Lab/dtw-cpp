include_guard(GLOBAL)

option(DTWC_ENABLE_COVERAGE "Enable coverage reporting for GCC or Clang" OFF)

# The one skip regex for the tree: anchored at a
# line start, matches SKIP / SKIPPED / SKIPPING as a word, never the marker
# substring "skips=0".
set(DTWC_TEST_SKIP_REGEX
    "(^|[\r\n])[ \t]*[Ss][Kk][Ii][Pp]([Pp][Ee][Dd]|[Pp][Ii][Nn][Gg])?([ :]|$)")

# dtwc_public_definitions(<var>): the definitions dtwc++ publishes, those of the
# libraries it links (each publishes its own capabilities: DTWC_HAS_MMAP, ...).
function(dtwc_public_definitions out)
  set(_defs "")
  foreach(_lib IN ITEMS dtwc_core dtwc_io dtwc_cli)
    if(TARGET ${_lib})
      get_target_property(_lib_defs ${_lib} INTERFACE_COMPILE_DEFINITIONS)
      if(_lib_defs)
        list(APPEND _defs ${_lib_defs})
      endif()
    endif()
  endforeach()
  set(${out} "${_defs}" PARENT_SCOPE)
endfunction()

# dtwc_add_test(NAME <target> SOURCE <file>
#   [MARKER <regex>] [MAY_SKIP] [MARKER_ONLY]
#   [REQUIRES <compile-definition>...] [LAUNCHER <command>...]
#   [ENVIRONMENT <k=v>...] [SERIAL] [PROCESSORS <n>] [TIMEOUT <s>] [LABELS <l>...]
#   [FIXTURE_ROOT <absolute-path> FIXTURE_DEFINE <MACRO>]
#   [COMPILE_DEFINITIONS <d>...])
#
# One Catch2 test executable + one CTest entry. A test passes when it ran at
# least one case, failed none and did not skip:
#   * the binary must print Catch2's summary "All tests passed (A assertions in
#     C test cases)" with A >= 1 and C >= 1. Catch2 prints that line only when
#     no case failed or skipped, so a skipped case simply removes the PASS match;
#     zero cases print "No tests ran" and zero assertions print the column
#     summary instead, so neither passes. (DTWC_TEST_SKIP_REGEX is line-anchored
#     and does NOT match Catch2's own mid-line "<file>(145): SKIPPED:"; it earns
#     its keep on script-test output, which has no summary to withhold.)
#   * MAY_SKIP opts into Catch2's skip, whose SKIP("<reason>") prints why: a run
#     in which every case skipped exits 4 and is reported Skipped, never Passed;
#     a partial skip passes only when a case passed and an assertion ran;
#   * MARKER, when given, must precede the summary (subject ran, not just
#     "some assertion ran");
#   * MARKER_ONLY: the subject owns main() and is not a Catch2 binary, so no
#     Catch2 summary exists in ANY configuration. MARKER alone is the gate then,
#     and it is mandatory: the subject must still print its own execution
#     evidence, so this is never a green stub;
#   * REQUIRES <def>: register only when dtwc++ publishes that definition;
#     otherwise the subject is absent by construction and nothing is added.
function(dtwc_add_test)
  set(options MAY_SKIP MARKER_ONLY SERIAL)
  set(one NAME SOURCE MARKER PROCESSORS TIMEOUT FIXTURE_ROOT FIXTURE_DEFINE)
  set(multi REQUIRES LAUNCHER ENVIRONMENT LABELS COMPILE_DEFINITIONS)
  cmake_parse_arguments(ARG "${options}" "${one}" "${multi}" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "dtwc_add_test(${ARG_NAME}): unknown arguments ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT ARG_NAME OR NOT ARG_SOURCE)
    message(FATAL_ERROR "dtwc_add_test: NAME and SOURCE are required")
  endif()

  # Record every name handled here, including ones REQUIRES declines below, so a
  # caller's catch-all pass cannot silently re-register them with the defaults.
  list(APPEND DTWC_TEST_REGISTERED ${ARG_NAME})
  set(DTWC_TEST_REGISTERED "${DTWC_TEST_REGISTERED}" PARENT_SCOPE)

  dtwc_public_definitions(_public_defs)
  foreach(req IN LISTS ARG_REQUIRES)
    if(NOT "${req}" IN_LIST _public_defs)
      message(STATUS "dtwc_add_test: ${ARG_NAME} not registered (dtwc++ does not publish ${req})")
      return()
    endif()
  endforeach()

  add_executable(${ARG_NAME} ${ARG_SOURCE})
  target_link_libraries(${ARG_NAME} PRIVATE dtwc++ Catch2::Catch2WithMain project_options)
  target_compile_definitions(${ARG_NAME} PRIVATE DTWC_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data" ${ARG_COMPILE_DEFINITIONS})
  if(DTWC_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      target_compile_options(${ARG_NAME} PUBLIC --coverage -O0)
      target_link_libraries(${ARG_NAME} PUBLIC --coverage)
    else()
      message(FATAL_ERROR "GCC or Clang required with DTWC_ENABLE_COVERAGE: found ${CMAKE_CXX_COMPILER_ID}")
    endif()
  endif()

  set(_env ${ARG_ENVIRONMENT})
  if(ARG_FIXTURE_ROOT)
    if(NOT IS_ABSOLUTE "${ARG_FIXTURE_ROOT}")
      message(FATAL_ERROR "dtwc_add_test(${ARG_NAME}): FIXTURE_ROOT must be absolute: ${ARG_FIXTURE_ROOT}")
    endif()
    if(NOT ARG_FIXTURE_DEFINE)
      message(FATAL_ERROR "dtwc_add_test(${ARG_NAME}): FIXTURE_ROOT needs FIXTURE_DEFINE")
    endif()
    target_compile_definitions(${ARG_NAME} PRIVATE ${ARG_FIXTURE_DEFINE}="${ARG_FIXTURE_ROOT}")
    # The redirect below points the subject's -- and any subprocess's -- temp
    # directory at this root, so it must exist on a fresh build tree.
    file(MAKE_DIRECTORY "${ARG_FIXTURE_ROOT}")
    list(APPEND _env "TMP=${ARG_FIXTURE_ROOT}" "TEMP=${ARG_FIXTURE_ROOT}" "TMPDIR=${ARG_FIXTURE_ROOT}")
  endif()

  get_target_property(_emulator ${ARG_NAME} CROSSCOMPILING_EMULATOR)
  if(NOT _emulator)
    set(_emulator "")
  endif()
  add_test(NAME ${ARG_NAME}
           COMMAND ${ARG_LAUNCHER} ${_emulator} $<TARGET_FILE:${ARG_NAME}>
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  if(ARG_MARKER_ONLY)
    if(NOT ARG_MARKER)
      message(FATAL_ERROR "dtwc_add_test(${ARG_NAME}): MARKER_ONLY requires MARKER")
    endif()
    if(ARG_MAY_SKIP)
      message(FATAL_ERROR
        "dtwc_add_test(${ARG_NAME}): MARKER_ONLY with MAY_SKIP has no Catch2 summary to "
        "recognise a skip in, and would drop the failure guard; gate the subject's own "
        "absence report with MARKER instead")
    endif()
    set(_pass "${ARG_MARKER}")
  else()
    set(_summary "All tests passed \\([1-9][0-9]* assertions? in [1-9][0-9]* test cases?\\)")
    if(ARG_MARKER)
      set(_pass "${ARG_MARKER}(.|[\r\n])*${_summary}")
    else()
      set(_pass "${_summary}")
    endif()
  endif()
  if(ARG_MAY_SKIP)
    # Catch2 v3 exits 4 only when every case skipped ("test cases: N | N
    # skipped"); a failure exits 42. A partial skip exits 0 and prints
    # "test cases: N | P passed | K skipped" then "assertions: A | A passed".
    #
    # PASS_REGULAR_EXPRESSION makes CTest ignore the exit code entirely, so the
    # FAIL regex is what turns a MIXED summary such as "test cases: 3 | 1 passed
    # | 1 failed | 1 skipped" into a failure. Match Catch2's summary column
    # ("| <n> failed") rather than a bare "[0-9]+ failed", which would trip on a
    # subject printing its own "0 failed" prose.
    set_tests_properties(${ARG_NAME} PROPERTIES
      SKIP_RETURN_CODE 4
      FAIL_REGULAR_EXPRESSION "\\| *[1-9][0-9]* failed"
      PASS_REGULAR_EXPRESSION
        "${_pass}|test cases: *[0-9]+ \\| *[1-9][0-9]* passed \\| *[1-9][0-9]* skipped[\r\n]+assertions: *[1-9]")
  else()
    set_tests_properties(${ARG_NAME} PROPERTIES
      FAIL_REGULAR_EXPRESSION "${DTWC_TEST_SKIP_REGEX}"
      PASS_REGULAR_EXPRESSION "${_pass}")
  endif()

  if(_env)
    set_property(TEST ${ARG_NAME} PROPERTY ENVIRONMENT "${_env}")
  endif()
  if(ARG_SERIAL)
    set_property(TEST ${ARG_NAME} PROPERTY RUN_SERIAL TRUE)
  endif()
  if(ARG_PROCESSORS)
    set_property(TEST ${ARG_NAME} PROPERTY PROCESSORS ${ARG_PROCESSORS})
  endif()
  if(ARG_TIMEOUT)
    set_property(TEST ${ARG_NAME} PROPERTY TIMEOUT ${ARG_TIMEOUT})
  endif()
  if(ARG_LABELS)
    set_property(TEST ${ARG_NAME} PROPERTY LABELS "${ARG_LABELS}")
  endif()
endfunction()

# dtwc_add_script_test(NAME <name> SCRIPT <file.cmake> [ARGS <-Dk=v>...]
#   MARKER <regex> [SERIAL] [TIMEOUT <s>] [LABELS <l>...] [WORKING_DIRECTORY <d>])
# A `cmake -P` integration script driving real binaries. Always strict: a skip
# line fails, the exact marker (with computed counters) must be printed.
function(dtwc_add_script_test)
  cmake_parse_arguments(ARG "SERIAL" "NAME;SCRIPT;MARKER;TIMEOUT;WORKING_DIRECTORY" "ARGS;LABELS" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "dtwc_add_script_test(${ARG_NAME}): unknown arguments ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT ARG_NAME OR NOT ARG_SCRIPT OR NOT ARG_MARKER)
    message(FATAL_ERROR "dtwc_add_script_test: NAME, SCRIPT and MARKER are required")
  endif()
  if(NOT ARG_WORKING_DIRECTORY)
    set(ARG_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  endif()
  add_test(NAME ${ARG_NAME}
           COMMAND ${CMAKE_COMMAND} ${ARG_ARGS} -P "${ARG_SCRIPT}"
           WORKING_DIRECTORY "${ARG_WORKING_DIRECTORY}")
  set_tests_properties(${ARG_NAME} PROPERTIES
    FAIL_REGULAR_EXPRESSION "${DTWC_TEST_SKIP_REGEX}"
    PASS_REGULAR_EXPRESSION "${ARG_MARKER}")
  if(ARG_SERIAL)
    set_property(TEST ${ARG_NAME} PROPERTY RUN_SERIAL TRUE)
  endif()
  if(ARG_TIMEOUT)
    set_property(TEST ${ARG_NAME} PROPERTY TIMEOUT ${ARG_TIMEOUT})
  endif()
  if(ARG_LABELS)
    set_property(TEST ${ARG_NAME} PROPERTY LABELS "${ARG_LABELS}")
  endif()
endfunction()
