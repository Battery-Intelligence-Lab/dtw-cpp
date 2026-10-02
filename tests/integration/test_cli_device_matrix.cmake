cmake_minimum_required(VERSION 3.26)

# IF-2 S3: dtwc::run's method x device table through the real dtwc_cl, and
# --print-config against the golden files. Every cell asserts its outcome:
#   cpu    every method runs; `auto` is pam at N = 27 (the summary's Method:)
#   gpu    the matrix methods run (the GPU fills the matrix), and clara with a
#          sample smaller than N (its samples and, on CUDA, its assignment on the
#          GPU), or the backend refuses on a machine without the device;
#          onebatch and tadpole are refused; a variant the kernels lack and
#          (Metal) a GPU index are refused before the input is read, which does
#          not exist; `cuda` is `gpu`. A build without a GPU refuses every gpu
#          cell with the api-contract-2.0.md §6.1 message.
#   hpc    refused for every method (D-10); an unknown device is refused
#   --print-config  the golden file reads back to itself (45 keys, each at a
#          value that is not its default) and a bare --print-config prints
#          config_defaults.toml: every default and spelling, pinned on the binary
foreach(required_var IN ITEMS CLI INPUT GOLDEN DEFAULTS GPU HAS_HIGHS WORK_ROOT)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        message(FATAL_ERROR "missing required -D${required_var}=...")
    endif()
endforeach()

cmake_path(ABSOLUTE_PATH CLI NORMALIZE OUTPUT_VARIABLE cli)
if(NOT EXISTS "${cli}")
    message(FATAL_ERROR "CLI does not exist: ${cli}")
endif()
file(REMOVE_RECURSE "${WORK_ROOT}")
file(MAKE_DIRECTORY "${WORK_ROOT}")
set(missing_input "${WORK_ROOT}/never_created.csv")

set(cells 0)
set(all_output "")

macro(run_cli case)
    execute_process(
        COMMAND "${cli}" ${ARGN}
        RESULT_VARIABLE exit_code
        OUTPUT_VARIABLE out
        ERROR_VARIABLE err
        ENCODING UTF-8)
    string(REPLACE "\r\n" "\n" out "${out}")
    string(REPLACE "\r\n" "\n" err "${err}")
    string(APPEND all_output "${out}\n${err}\n")
    if(NOT "${exit_code}" MATCHES "^[0-9]+$")
        message(FATAL_ERROR "${case}: dtwc_cl did not exit normally (result=${exit_code})\n"
                            "stdout:\n${out}\nstderr:\n${err}")
    endif()
endmacro()

# A cell that must run: exit 0, and the summary names the method that ran.
macro(expect_ran case method)
    run_cli(${case} -i "${INPUT}" -k 3 -o "${WORK_ROOT}/${case}" --name "${case}" ${ARGN})
    if(NOT "${exit_code}" STREQUAL "0" OR NOT "${out}" MATCHES "\n  Method:     ${method}\n")
        message(FATAL_ERROR "${case}: expected a ${method} run, exit=${exit_code}\n"
                            "stdout:\n${out}\nstderr:\n${err}")
    endif()
    math(EXPR cells "${cells} + 1")
endmacro()

# A cell that must be refused: a non-zero exit and one error naming the reason.
macro(expect_refused case needle)
    run_cli(${case} -k 3 -o "${WORK_ROOT}/${case}" --name "${case}" ${ARGN})
    string(FIND "${err}" "${needle}" needle_at)
    if("${exit_code}" STREQUAL "0" OR needle_at EQUAL -1 OR NOT "${err}" MATCHES "^Error: ")
        message(FATAL_ERROR "${case}: expected a refusal naming '${needle}', exit=${exit_code}\n"
                            "stdout:\n${out}\nstderr:\n${err}")
    endif()
    if(EXISTS "${WORK_ROOT}/${case}/${case}_labels.csv")
        message(FATAL_ERROR "${case}: a refused run wrote labels")
    endif()
    math(EXPR cells "${cells} + 1")
endmacro()

# ---- --print-config: the golden file, and the defaults ----
function(expected_text path output)
    file(READ "${path}" text)
    string(REPLACE "\r\n" "\n" text "${text}")
    string(REGEX REPLACE "^(#[^\n]*\n)+" "" text "${text}") # the leading comment block
    set(${output} "${text}" PARENT_SCOPE)
endfunction()
expected_text("${GOLDEN}" golden)
expected_text("${DEFAULTS}" defaults)
run_cli(print_config_golden --config "${GOLDEN}" --print-config)
if(NOT "${exit_code}" STREQUAL "0" OR NOT "${out}" STREQUAL "${golden}")
    message(FATAL_ERROR "print_config_golden: exit=${exit_code}\nstdout:\n${out}\nexpected:\n${golden}\nstderr:\n${err}")
endif()
run_cli(print_config_defaults --print-config)
if(NOT "${exit_code}" STREQUAL "0" OR NOT "${out}" STREQUAL "${defaults}")
    message(FATAL_ERROR "print_config_defaults: exit=${exit_code}\nstdout:\n${out}\nexpected:\n${defaults}\nstderr:\n${err}")
endif()
string(REGEX MATCHALL "\n" golden_lines "${golden}")
list(LENGTH golden_lines golden_keys)

# ---- cpu: every method runs ----
set(methods pam onebatch clara kmedoids lrcore tadpole hierarchical)
if(HAS_HIGHS)
    list(APPEND methods mip)
endif()
expect_ran(cpu_auto pam --device cpu --method auto)
foreach(method IN LISTS methods)
    expect_ran(cpu_${method} ${method} --device cpu --method ${method})
endforeach()

# ---- gpu ----
set(as_it_goes "computes its distances on the CPU as it goes, so device 'gpu' would sit idle")
set(gpu_methods pam clara kmedoids lrcore hierarchical)
if(HAS_HIGHS)
    list(APPEND gpu_methods mip)
endif()
if(GPU STREQUAL "none")
    set(device "none")
    set(not_built "this build has no GPU backend compiled in")
    foreach(method IN LISTS gpu_methods ITEMS auto onebatch tadpole)
        expect_refused(gpu_${method} "${not_built}" -i "${INPUT}" --device gpu --method ${method})
    endforeach()
    expect_refused(gpu_cuda_alias "${not_built}" -i "${INPUT}" --device cuda --method pam)
    expect_refused(gpu_wdtw "${not_built}" -i "${missing_input}" --device gpu --variant wdtw)
else()
    # The first run tells whether this machine has the device; a GPU build
    # without one must refuse, never compute on the CPU.
    run_cli(gpu_probe -i "${INPUT}" -k 3 -o "${WORK_ROOT}/gpu_probe" --name gpu_probe --device gpu --method pam)
    if("${exit_code}" STREQUAL "0")
        set(device "present")
    elseif("${err}" MATCHES "GPU was detected")
        set(device "absent")
    else()
        message(FATAL_ERROR "gpu_probe: exit=${exit_code}\nstdout:\n${out}\nstderr:\n${err}")
    endif()
    foreach(method IN LISTS gpu_methods ITEMS auto)
        set(ran ${method})
        if(method STREQUAL "auto")
            set(ran pam) # at any N on a GPU
        endif()
        if(device STREQUAL "present")
            expect_ran(gpu_${method} ${ran} --device gpu --method ${method})
        else()
            expect_refused(gpu_${method} "GPU was detected" -i "${INPUT}" --device gpu --method ${method})
        endif()
    endforeach()
    if(device STREQUAL "present")
        expect_ran(gpu_cuda_alias pam --device cuda --method pam)
    else()
        expect_refused(gpu_cuda_alias "GPU was detected" -i "${INPUT}" --device cuda --method pam)
    endif()
    foreach(method IN ITEMS onebatch tadpole)
        expect_refused(gpu_${method} "${as_it_goes}" -i "${INPUT}" --device gpu --method ${method})
    endforeach()
    if(device STREQUAL "present")
        expect_ran(gpu_clara_sample clara --device gpu --method clara --sample-size 5)
    else()
        expect_refused(gpu_clara_sample "GPU was detected" -i "${INPUT}" --device gpu --method clara --sample-size 5)
    endif()
    expect_refused(gpu_wdtw "variant = WDTW" -i "${missing_input}" --device gpu --variant wdtw)
    if(GPU STREQUAL "metal")
        expect_refused(gpu_index_1 "GPU index = 1" -i "${missing_input}" --device gpu:1)
    endif()
endif()

# ---- hpc, and a name no grammar reads ----
foreach(method IN ITEMS auto pam)
    expect_refused(hpc_${method} "device 'hpc' submits a whole run to a SLURM cluster" -i "${INPUT}"
                   --device hpc --method ${method})
endforeach()
expect_refused(device_tpu "unknown device 'tpu'" -i "${INPUT}" --device tpu)

# ---- every cell ran ----
list(LENGTH methods cpu_cells)
math(EXPR expected "${cpu_cells} + 1 + 3")
list(LENGTH gpu_methods gpu_cells)
if(GPU STREQUAL "none")
    math(EXPR expected "${expected} + ${gpu_cells} + 5")
elseif(GPU STREQUAL "metal")
    math(EXPR expected "${expected} + ${gpu_cells} + 7")
else()
    math(EXPR expected "${expected} + ${gpu_cells} + 6")
endif()
if(NOT cells EQUAL expected)
    message(FATAL_ERROR "ran ${cells} cells, expected ${expected}")
endif()
# DTWC_TEST_SKIP_REGEX's pattern: a line that starts with "skip" (not skip-rows).
string(REGEX MATCH "(^|\n)[ \t]*[Ss][Kk][Ii][Pp]([Pp][Ee][Dd]|[Pp][Ii][Nn][Gg])?([ :]|\n|$)" skip_match "${all_output}")
if(skip_match)
    message(FATAL_ERROR "dtwc_cl emitted skip text:\n${all_output}")
endif()
message(STATUS "CLI_DEVICE_MATRIX subject=real_dtwc_cl gpu=${GPU} device=${device} "
               "cells=${cells}/${expected} print_config=2/2 keys=${golden_keys} skips=0")
