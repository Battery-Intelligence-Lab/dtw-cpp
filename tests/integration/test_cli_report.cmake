cmake_minimum_required(VERSION 3.26)

# dtwc_cl's run report, <name>_report.txt, on the real binary: a file of its own, written after the four result files,
# which are the same bytes with it and without it (--no-report); the run names it on stdout, and --no-report writes
# and names none. The report's text holds times and the machine's facts: only its first lines are read here.
# @author Volkan Kumtepeli
# @author Claude Opus 5.5 (RR1b)
foreach(required_var IN ITEMS CLI INPUT WORK_ROOT)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        message(FATAL_ERROR "missing required -D${required_var}=...")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_ROOT}")
foreach(mode IN ITEMS report no_report)
    set(args -i "${INPUT}" -k 3 -m pam --name rep -o "${WORK_ROOT}/${mode}" --ram-limit 1G)
    if(mode STREQUAL "no_report")
        list(APPEND args --no-report)
    endif()
    execute_process(COMMAND "${CLI}" ${args}
        RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr ENCODING UTF-8)
    if(NOT "${result}" STREQUAL "0")
        message(FATAL_ERROR "dtwc_cl (${mode}): exit=${result}\nstdout:\n${stdout}\nstderr:\n${stderr}")
    endif()
    string(REPLACE "\r" "" stdout_${mode} "${stdout}")
endforeach()

# The report: a completed run's, named on stdout; none, and no line, under --no-report.
set(report "${WORK_ROOT}/report/rep_report.txt")
if(NOT EXISTS "${report}")
    file(GLOB written "${WORK_ROOT}/report/*")
    message(FATAL_ERROR "no ${report}; the run wrote: ${written}")
endif()
file(READ "${report}" text)
string(REPLACE "\r" "" text "${text}")
if(NOT text MATCHES "^DTWC\\+\\+ run report" OR NOT text MATCHES "\n   status         completed\n")
    message(FATAL_ERROR "${report} is no completed run's report:\n${text}")
endif()
if(NOT stdout_report MATCHES "\n  Report: +[^\n]*rep_report\\.txt\n")
    message(FATAL_ERROR "the run did not name its report on stdout:\n${stdout_report}")
endif()
file(GLOB quiet "${WORK_ROOT}/no_report/*_report.txt")
if(quiet OR stdout_no_report MATCHES "Report:")
    message(FATAL_ERROR "--no-report wrote or named a report (${quiet}):\n${stdout_no_report}")
endif()

# The four result files: the same bytes with the report and without it.
set(equal 0)
foreach(result IN ITEMS labels medoids distance_matrix silhouettes)
    file(SHA256 "${WORK_ROOT}/report/rep_${result}.csv" with)
    file(SHA256 "${WORK_ROOT}/no_report/rep_${result}.csv" without)
    if(NOT with STREQUAL without)
        message(FATAL_ERROR "rep_${result}.csv differs with the report (${with}) and without it (${without})")
    endif()
    math(EXPR equal "${equal} + 1")
endforeach()
message(STATUS "CLI_REPORT subject=real_dtwc_cl report=1 no_report=0 results_equal=${equal}/4")
