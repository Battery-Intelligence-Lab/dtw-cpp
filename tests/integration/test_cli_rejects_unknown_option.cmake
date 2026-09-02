cmake_minimum_required(VERSION 3.26)

# The hand-rolled --yaml-config loader was removed (TOML via --config is the one
# CLI config mechanism). PASS_REGULAR_EXPRESSION made CTest ignore the exit code
# entirely and pinned CLI11's exact phrasing; this script asserts the two things
# that actually matter: the run FAILS, and the offending flag is named back to
# the user. It does not pin the parser's wording.
foreach(required_var IN ITEMS CLI OPTION)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        message(FATAL_ERROR "missing required -D${required_var}=...")
    endif()
endforeach()

cmake_path(ABSOLUTE_PATH CLI NORMALIZE OUTPUT_VARIABLE cli_path)
if(NOT EXISTS "${cli_path}")
    message(FATAL_ERROR "CLI does not exist: ${cli_path}")
endif()

execute_process(
    COMMAND "${cli_path}" "${OPTION}" "unknown-option-probe.yaml"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE stdout
    ERROR_VARIABLE stderr
    ENCODING UTF-8)

set(combined "${stdout}\n${stderr}")

if("${result}" STREQUAL "0")
    message(FATAL_ERROR
        "CLI accepted the removed option ${OPTION} (exit=0)\n"
        "stdout:\n${stdout}\nstderr:\n${stderr}")
endif()
# A crash/timeout string is not a rejection either.
if(NOT "${result}" MATCHES "^[0-9]+$")
    message(FATAL_ERROR
        "CLI did not exit normally for ${OPTION}: result=${result}\n"
        "stdout:\n${stdout}\nstderr:\n${stderr}")
endif()

string(FIND "${combined}" "${OPTION}" option_offset)
if(option_offset EQUAL -1)
    message(FATAL_ERROR
        "CLI rejected ${OPTION} without naming it (exit=${result})\n"
        "stdout:\n${stdout}\nstderr:\n${stderr}")
endif()

string(
    REGEX MATCH "[Ss][Kk][Ii][Pp]([Pp]|[ :])" skip_match "${combined}")
if(skip_match)
    message(FATAL_ERROR "CLI emitted skip text:\n${combined}")
endif()

message(STATUS
    "CLI_REJECTS_UNKNOWN_OPTION subject=real_dtwc_cl option=${OPTION} "
    "exit=${result} nonzero=1 named=1 skips=0")
