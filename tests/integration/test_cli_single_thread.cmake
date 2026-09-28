cmake_minimum_required(VERSION 3.26)

# dtwc_cl run with OMP_NUM_THREADS=1 on a multicore host must say, once, that
# it runs single-threaded and how to fix it; with the variable unset it must stay
# quiet. A build without OpenMP always says so, with its own remedy.
foreach(required_var IN ITEMS CLI INPUT WORK_ROOT SEQUENTIAL)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        message(FATAL_ERROR "missing required -D${required_var}=...")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_ROOT}")
file(MAKE_DIRECTORY "${WORK_ROOT}")
cmake_host_system_information(RESULT cores QUERY NUMBER_OF_LOGICAL_CORES)

set(runtime_warning "OpenMP is available but only 1 thread is usable")
set(sequential_warning "compiled WITHOUT OpenMP")

function(run_cli case env_arg out_var)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E env ${env_arg}
                "${CLI}" -i "${INPUT}" -k 2 -o "${WORK_ROOT}/${case}"
        RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr ENCODING UTF-8)
    if(NOT "${result}" STREQUAL "0")
        message(FATAL_ERROR "${case}: exit=${result}\nstdout:\n${stdout}\nstderr:\n${stderr}")
    endif()
    set(${out_var} "${stderr}" PARENT_SCOPE)
endfunction()

function(count_matches text needle out_var)
    string(REGEX MATCHALL "${needle}" hits "${text}")
    list(LENGTH hits n)
    set(${out_var} ${n} PARENT_SCOPE)
endfunction()

run_cli(one_thread "OMP_NUM_THREADS=1" one_err)
run_cli(default "--unset=OMP_NUM_THREADS" default_err)

if(SEQUENTIAL)
    set(expected_one 1)
    set(expected_default 1)
    set(needle "${sequential_warning}")
elseif(cores GREATER 1)
    set(expected_one 1)
    set(expected_default 0)
    set(needle "${runtime_warning}")
else() # a single-core host is serial by nature: no warning
    set(expected_one 0)
    set(expected_default 0)
    set(needle "${runtime_warning}")
endif()
count_matches("${one_err}" "${needle}" one)
count_matches("${default_err}" "${needle}" default)
if(NOT one EQUAL expected_one OR NOT default EQUAL expected_default)
    message(FATAL_ERROR "single-thread warning: one_thread=${one} (want ${expected_one}), "
                        "default=${default} (want ${expected_default})\n"
                        "one_thread stderr:\n${one_err}\ndefault stderr:\n${default_err}")
endif()
if(one EQUAL 1 AND NOT SEQUENTIAL AND NOT "${one_err}" MATCHES "OMP_NUM_THREADS>1")
    message(FATAL_ERROR "the warning does not name its remedy:\n${one_err}")
endif()
message(STATUS "CLI_SINGLE_THREAD subject=real_dtwc_cl one_thread=${one} default=${default} skips=0")
