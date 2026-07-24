cmake_minimum_required(VERSION 3.26)

foreach(required_var IN ITEMS
        CLI FIXTURE_WRITER INPUT CONFIG BINARY_ROOT WORK_ROOT)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        message(FATAL_ERROR "F17 missing required -D${required_var}=...")
    endif()
endforeach()

cmake_path(ABSOLUTE_PATH CLI NORMALIZE OUTPUT_VARIABLE cli_path)
cmake_path(
    ABSOLUTE_PATH FIXTURE_WRITER NORMALIZE
    OUTPUT_VARIABLE fixture_writer_path)
cmake_path(ABSOLUTE_PATH INPUT NORMALIZE OUTPUT_VARIABLE input_path)
cmake_path(ABSOLUTE_PATH CONFIG NORMALIZE OUTPUT_VARIABLE config_path)
cmake_path(
    ABSOLUTE_PATH BINARY_ROOT NORMALIZE
    OUTPUT_VARIABLE binary_root)
cmake_path(ABSOLUTE_PATH WORK_ROOT NORMALIZE OUTPUT_VARIABLE work_root)

foreach(executable IN ITEMS cli_path fixture_writer_path)
    if(NOT EXISTS "${${executable}}")
        message(FATAL_ERROR "F17 executable does not exist: ${${executable}}")
    endif()
endforeach()
foreach(fixture IN ITEMS input_path config_path)
    if(NOT EXISTS "${${fixture}}")
        message(FATAL_ERROR "F17 tracked fixture does not exist: ${${fixture}}")
    endif()
endforeach()
if(NOT IS_DIRECTORY "${binary_root}")
    message(FATAL_ERROR "F17 binary root is not a directory: ${binary_root}")
endif()

# This script may recursively clean only its registered exact child of the
# configured tests binary directory.
cmake_path(
    APPEND binary_root "f17-cli-resume"
    OUTPUT_VARIABLE expected_work_root)
if(NOT work_root STREQUAL expected_work_root)
    message(FATAL_ERROR
        "F17 unsafe work root '${work_root}'; expected '${expected_work_root}'")
endif()
if(IS_SYMLINK "${work_root}")
    message(FATAL_ERROR "F17 work root must not be a symlink: ${work_root}")
endif()
file(REAL_PATH "${binary_root}" binary_root_real)
cmake_path(
    APPEND binary_root_real "f17-cli-resume"
    OUTPUT_VARIABLE expected_work_real)
if(EXISTS "${work_root}")
    file(REAL_PATH "${work_root}" existing_work_real)
    if(NOT existing_work_real STREQUAL expected_work_real)
        message(FATAL_ERROR
            "F17 resolved work root escaped: '${existing_work_real}'")
    endif()
endif()
file(REMOVE_RECURSE "${work_root}")
file(MAKE_DIRECTORY "${work_root}")

file(SHA256 "${input_path}" input_sha_before)
file(SHA256 "${config_path}" config_sha_before)
string(TOUPPER "${input_sha_before}" input_sha_before)
string(TOUPPER "${config_sha_before}" config_sha_before)
if(NOT input_sha_before STREQUAL
        "6230423623F0564D3CA0F8C27D78D5D7CEB863DCC3332BBAE9FFE2B78B869F3F")
    message(FATAL_ERROR "F17 conformance input hash drift: ${input_sha_before}")
endif()
if(NOT config_sha_before STREQUAL
        "7F36DB93AB26BCB7DC8D393671221BA1FD2190B567A790E011B3FC0B16DDCD55")
    message(FATAL_ERROR "F17 conformance config hash drift: ${config_sha_before}")
endif()

set(expected_replay_labels [=[name,cluster
1,0
2,0
3,0
4,0
5,0
6,0
7,0
8,0
9,0
10,1
11,1
12,1
13,1
14,1
15,1
16,1
17,1
18,1
19,2
20,2
21,2
22,2
23,2
24,2
25,2
26,2
27,2
]=])
set(expected_replay_medoids [=[cluster,medoid_index,medoid_name
0,0,1
1,9,10
2,18,19
]=])
set(expected_fresh_labels [=[name,cluster
1,1
2,1
3,1
4,1
5,1
6,1
7,1
8,1
9,1
10,2
11,2
12,2
13,2
14,2
15,2
16,2
17,2
18,2
19,0
20,0
21,0
22,0
23,0
24,0
25,0
26,0
27,0
]=])
set(expected_fresh_medoids [=[cluster,medoid_index,medoid_name
0,20,21
1,4,5
2,15,16
]=])

function(normalized_file path output)
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "F17 expected output does not exist: ${path}")
    endif()
    file(READ "${path}" contents)
    string(REPLACE "\r\n" "\n" contents "${contents}")
    string(REPLACE "\r" "\n" contents "${contents}")
    set(${output} "${contents}" PARENT_SCOPE)
endfunction()

function(silhouette_values path output)
    file(STRINGS "${path}" rows)
    list(LENGTH rows row_count)
    if(NOT row_count EQUAL 28)
        message(FATAL_ERROR
            "F17 silhouette rows=${row_count}, expected=28: ${path}")
    endif()
    list(POP_FRONT rows header)
    if(NOT header STREQUAL "name,cluster,silhouette")
        message(FATAL_ERROR "F17 silhouette header mismatch: ${header}")
    endif()
    set(values)
    foreach(row IN LISTS rows)
        if(NOT row MATCHES "^[^,]+,[^,]+,([^,]+)$")
            message(FATAL_ERROR "F17 malformed silhouette row: ${row}")
        endif()
        list(APPEND values "${CMAKE_MATCH_1}")
    endforeach()
    set(${output} "${values}" PARENT_SCOPE)
endfunction()

function(require_occurrences haystack needle expected context)
    string(LENGTH "${haystack}" before_length)
    string(LENGTH "${needle}" needle_length)
    if(needle_length EQUAL 0)
        message(FATAL_ERROR "F17 internal empty marker for ${context}")
    endif()
    string(REPLACE "${needle}" "" stripped "${haystack}")
    string(LENGTH "${stripped}" after_length)
    math(EXPR removed_length "${before_length} - ${after_length}")
    math(EXPR occurrence_count "${removed_length} / ${needle_length}")
    if(NOT occurrence_count EQUAL expected)
        message(FATAL_ERROR
            "F17 ${context} count=${occurrence_count}, expected=${expected}\n"
            "marker=${needle}\noutput:\n${haystack}")
    endif()
endfunction()

function(require_no_skip context stdout stderr)
    string(
        REGEX MATCH "[Ss][Kk][Ii][Pp]([Pp]|[ :])"
        skip_match "${stdout}\n${stderr}")
    if(skip_match)
        message(FATAL_ERROR
            "F17 ${context} emitted skip text\n"
            "stdout:\n${stdout}\nstderr:\n${stderr}")
    endif()
endfunction()

function(write_fixture output_dir mode)
    file(MAKE_DIRECTORY "${output_dir}")
    set(checkpoint "${output_dir}/conformance_checkpoint.bin")
    execute_process(
        COMMAND "${fixture_writer_path}" "${checkpoint}" "${mode}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr
        ENCODING UTF-8)
    if(NOT "${result}" STREQUAL "0")
        message(FATAL_ERROR
            "F17 fixture ${mode} exit=${result}\n"
            "stdout:\n${stdout}\nstderr:\n${stderr}")
    endif()
    require_no_skip("fixture ${mode}" "${stdout}" "${stderr}")
    require_occurrences(
        "${stdout}"
        "F17_CHECKPOINT_FIXTURE writer=production_serializer mode=${mode}"
        1 "fixture ${mode} marker")
    if(mode STREQUAL "valid")
        file(SIZE "${checkpoint}" checkpoint_size)
        if(NOT checkpoint_size EQUAL 152)
            message(FATAL_ERROR
                "F17 valid checkpoint bytes=${checkpoint_size}, expected=152")
        endif()
    endif()
endfunction()

function(capture_checkpoint path sha_var time_var)
    if(NOT EXISTS "${path}")
        message(FATAL_ERROR "F17 checkpoint does not exist: ${path}")
    endif()
    file(SHA256 "${path}" checkpoint_sha)
    file(TIMESTAMP "${path}" checkpoint_time UTC)
    set(${sha_var} "${checkpoint_sha}" PARENT_SCOPE)
    set(${time_var} "${checkpoint_time}" PARENT_SCOPE)
endfunction()

function(require_checkpoint_unchanged path expected_sha expected_time context)
    capture_checkpoint("${path}" actual_sha actual_time)
    if(NOT actual_sha STREQUAL expected_sha)
        message(FATAL_ERROR
            "F17 ${context} changed checkpoint SHA-256: "
            "${expected_sha} -> ${actual_sha}")
    endif()
    if(NOT actual_time STREQUAL expected_time)
        message(FATAL_ERROR
            "F17 ${context} changed checkpoint timestamp: "
            "${expected_time} -> ${actual_time}")
    endif()
endfunction()

function(cli_command output_dir)
    set(command
        "${cli_path}"
        --config "${config_path}"
        --input "${input_path}"
        --output "${output_dir}"
        --name conformance
        --max-iter 1
        --n-init 1
        --seed 42
        --mmap-threshold 1000000
        --device cpu)
    set(f17_cli_command "${command}" PARENT_SCOPE)
endfunction()

function(run_valid_replay route verbose requested_method imported_matrix)
    set(output_dir "${work_root}/${route}")
    write_fixture("${output_dir}" valid)
    set(checkpoint "${output_dir}/conformance_checkpoint.bin")
    capture_checkpoint("${checkpoint}" checkpoint_sha checkpoint_time)
    cli_command("${output_dir}")
    list(APPEND f17_cli_command --method "${requested_method}")
    if(NOT imported_matrix STREQUAL "")
        list(APPEND f17_cli_command --dist-matrix "${imported_matrix}")
    endif()
    if(verbose)
        list(APPEND f17_cli_command --verbose)
    endif()
    list(APPEND f17_cli_command --resume)

    execute_process(
        COMMAND ${f17_cli_command}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr
        ENCODING UTF-8)
    if(NOT "${result}" STREQUAL "0")
        message(FATAL_ERROR
            "F17 ${route} replay exit=${result}\n"
            "stdout:\n${stdout}\nstderr:\n${stderr}")
    endif()
    require_no_skip("${route} replay" "${stdout}" "${stderr}")
    require_occurrences(
        "${stdout}"
        "Replaying completed result checkpoint: N=27, k=3, iterations=41, converged=no"
        1 "${route} unconditional replay marker")
    require_occurrences(
        "${stdout}" "requested method=${requested_method}" 1
        "${route} requested-method marker")
    require_occurrences(
        "${stdout}" "  Method:     checkpoint replay" 1
        "${route} summary source")
    require_occurrences(
        "${stdout}" "  Requested:  ${requested_method}" 1
        "${route} requested-method summary")
    require_occurrences(
        "${stdout}" "  Total cost: 1650" 1 "${route} total cost")
    require_occurrences(
        "${stdout}" "  Converged:  no" 1 "${route} convergence")
    require_occurrences(
        "${stdout}" "  Iterations: 41" 1 "${route} iterations")
    if(verbose)
        require_occurrences(
            "${stdout}" "Running FastPAM" 0 "${route} clustering exclusion")
    endif()

    normalized_file(
        "${output_dir}/conformance_labels.csv" replay_labels)
    normalized_file(
        "${output_dir}/conformance_medoids.csv" replay_medoids)
    if(NOT replay_labels STREQUAL expected_replay_labels)
        message(FATAL_ERROR
            "F17 ${route} replay labels mismatch:\n${replay_labels}")
    endif()
    if(NOT replay_medoids STREQUAL expected_replay_medoids)
        message(FATAL_ERROR
            "F17 ${route} replay medoids mismatch:\n${replay_medoids}")
    endif()
    if(imported_matrix STREQUAL "")
        file(GLOB output_files
            RELATIVE "${output_dir}" "${output_dir}/*")
        list(SORT output_files)
        set(expected_files
            conformance_checkpoint.bin
            conformance_labels.csv
            conformance_medoids.csv)
        if(NOT output_files STREQUAL expected_files)
            message(FATAL_ERROR
                "F17 ${route} file inventory='${output_files}', "
                "expected='${expected_files}'")
        endif()
    else()
        foreach(expected IN ITEMS
                conformance_distance_matrix.csv
                conformance_silhouettes.csv)
            if(NOT EXISTS "${output_dir}/${expected}")
                message(FATAL_ERROR
                    "F17 ${route} rehydration did not create ${expected}")
            endif()
        endforeach()
        if(EXISTS "${output_dir}/conformance_distmat.cache")
            message(FATAL_ERROR
                "F17 ${route} unexpectedly created an mmap cache")
        endif()
        silhouette_values(
            "${output_dir}/conformance_silhouettes.csv" replay_silhouette_values)
        if(NOT replay_silhouette_values STREQUAL fresh_silhouette_values)
            message(FATAL_ERROR
                "F17 ${route} silhouette values differ from fresh partition\n"
                "fresh=${fresh_silhouette_values}\n"
                "replay=${replay_silhouette_values}")
        endif()
    endif()
    require_checkpoint_unchanged(
        "${checkpoint}" "${checkpoint_sha}" "${checkpoint_time}" "${route}")

    set(f17_${route}_labels "${replay_labels}" PARENT_SCOPE)
    set(f17_${route}_medoids "${replay_medoids}" PARENT_SCOPE)
endfunction()

set(fresh_dir "${work_root}/fresh")
file(MAKE_DIRECTORY "${fresh_dir}")
cli_command("${fresh_dir}")
list(APPEND f17_cli_command --verbose)
execute_process(
    COMMAND ${f17_cli_command}
    RESULT_VARIABLE fresh_result
    OUTPUT_VARIABLE fresh_stdout
    ERROR_VARIABLE fresh_stderr
    ENCODING UTF-8)
if(NOT "${fresh_result}" STREQUAL "0")
    message(FATAL_ERROR
        "F17 fresh control exit=${fresh_result}\n"
        "stdout:\n${fresh_stdout}\nstderr:\n${fresh_stderr}")
endif()
require_no_skip("fresh control" "${fresh_stdout}" "${fresh_stderr}")
require_occurrences(
    "${fresh_stdout}" "Running FastPAM (k=3)" 1 "fresh algorithm route")
require_occurrences(
    "${fresh_stdout}" "  Total cost: 976" 1 "fresh total cost")
require_occurrences(
    "${fresh_stdout}" "  Converged:  no" 1 "fresh convergence")
require_occurrences(
    "${fresh_stdout}" "  Iterations: 1" 1 "fresh iterations")
normalized_file("${fresh_dir}/conformance_labels.csv" fresh_labels)
normalized_file("${fresh_dir}/conformance_medoids.csv" fresh_medoids)
silhouette_values(
    "${fresh_dir}/conformance_silhouettes.csv" fresh_silhouette_values)
if(NOT fresh_labels STREQUAL expected_fresh_labels)
    message(FATAL_ERROR "F17 fresh labels mismatch:\n${fresh_labels}")
endif()
if(NOT fresh_medoids STREQUAL expected_fresh_medoids)
    message(FATAL_ERROR "F17 fresh medoids mismatch:\n${fresh_medoids}")
endif()

run_valid_replay(valid_verbose TRUE pam "")
run_valid_replay(
    valid_quiet FALSE kmedoids
    "${fresh_dir}/conformance_distance_matrix.csv")
if(f17_valid_verbose_labels STREQUAL fresh_labels)
    message(FATAL_ERROR "F17 replay labels equal the fresh control")
endif()
if(f17_valid_verbose_medoids STREQUAL fresh_medoids)
    message(FATAL_ERROR "F17 replay medoids equal the fresh control")
endif()
if(NOT f17_valid_verbose_labels STREQUAL f17_valid_quiet_labels
        OR NOT f17_valid_verbose_medoids STREQUAL f17_valid_quiet_medoids)
    message(FATAL_ERROR "F17 verbose/quiet replay payloads differ")
endif()

function(run_rejection case mode expected_error)
    set(output_dir "${work_root}/reject-${case}")
    file(MAKE_DIRECTORY "${output_dir}")
    set(checkpoint "${output_dir}/conformance_checkpoint.bin")
    if(NOT mode STREQUAL "missing")
        write_fixture("${output_dir}" "${mode}")
        capture_checkpoint("${checkpoint}" checkpoint_sha checkpoint_time)
    endif()

    cli_command("${output_dir}")
    list(APPEND f17_cli_command --resume --verbose)
    execute_process(
        COMMAND ${f17_cli_command}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr
        ENCODING UTF-8)
    if("${result}" STREQUAL "0")
        message(FATAL_ERROR
            "F17 ${case} unexpectedly succeeded\n"
            "stdout:\n${stdout}\nstderr:\n${stderr}")
    endif()
    require_no_skip("${case} rejection" "${stdout}" "${stderr}")
    require_occurrences(
        "${stdout}\n${stderr}" "Error:" 1 "${case} typed error")
    require_occurrences(
        "${stdout}\n${stderr}" "${expected_error}" 1 "${case} diagnostic")
    require_occurrences(
        "${stdout}" "Running FastPAM" 0 "${case} clustering exclusion")
    foreach(unexpected IN ITEMS
            conformance_labels.csv
            conformance_medoids.csv
            conformance_distance_matrix.csv
            conformance_silhouettes.csv
            conformance_distmat.cache)
        if(EXISTS "${output_dir}/${unexpected}")
            message(FATAL_ERROR
                "F17 ${case} rejection created ${unexpected}")
        endif()
    endforeach()
    if(mode STREQUAL "missing")
        if(EXISTS "${checkpoint}")
            message(FATAL_ERROR
                "F17 missing rejection created a binary checkpoint")
        endif()
    else()
        require_checkpoint_unchanged(
            "${checkpoint}" "${checkpoint_sha}" "${checkpoint_time}" "${case}")
    endif()
endfunction()

run_rejection(
    missing missing
    "--resume requires a readable binary result checkpoint")
run_rejection(
    malformed malformed
    "--resume requires a readable binary result checkpoint")
run_rejection(
    wrong_n wrong-n
    "current input has 27 series")
run_rejection(
    wrong_k wrong-k
    "--n-clusters requests 3")
run_rejection(
    bad_label bad-label
    "checkpoint label[0]=3 is outside [0,3)")
run_rejection(
    bad_medoid bad-medoid
    "checkpoint medoid[0]=27 is outside [0,27)")
run_rejection(
    duplicate_medoid duplicate-medoid
    "checkpoint medoid index 0 is duplicated")
run_rejection(
    negative_iterations negative-iterations
    "checkpoint iteration count -1 is negative")
run_rejection(
    nonfinite_cost nonfinite-cost
    "checkpoint total cost is not finite")

file(SHA256 "${input_path}" input_sha_after)
file(SHA256 "${config_path}" config_sha_after)
string(TOUPPER "${input_sha_after}" input_sha_after)
string(TOUPPER "${config_sha_after}" config_sha_after)
if(NOT input_sha_after STREQUAL input_sha_before)
    message(FATAL_ERROR
        "F17 CLI modified tracked input: ${input_sha_before} -> ${input_sha_after}")
endif()
if(NOT config_sha_after STREQUAL config_sha_before)
    message(FATAL_ERROR
        "F17 CLI modified tracked config: ${config_sha_before} -> ${config_sha_after}")
endif()

message(
    "F17_CLI_RESUME subject=real_dtwc_cl writer=production_serializer "
    "runs=12/12 replay_fields=10/10 markers=2/2 algorithm_skipped=1/1 "
    "fresh_discriminator=4/4 checkpoint_preserved=10/10 "
    "rejection_cases=9/9 sources_preserved=2/2 skips=0")
