cmake_minimum_required(VERSION 3.26)

foreach(required_var IN ITEMS
        CLI FIXTURE_WRITER FIXTURE BINARY_ROOT WORK_ROOT BYTE_ORDER)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        message(FATAL_ERROR "F13 missing required -D${required_var}=...")
    endif()
endforeach()

cmake_path(ABSOLUTE_PATH CLI NORMALIZE OUTPUT_VARIABLE cli_path)
cmake_path(
    ABSOLUTE_PATH FIXTURE_WRITER NORMALIZE
    OUTPUT_VARIABLE fixture_writer_path)
cmake_path(ABSOLUTE_PATH FIXTURE NORMALIZE OUTPUT_VARIABLE fixture_path)
cmake_path(ABSOLUTE_PATH BINARY_ROOT NORMALIZE OUTPUT_VARIABLE binary_root)
cmake_path(ABSOLUTE_PATH WORK_ROOT NORMALIZE OUTPUT_VARIABLE work_root)

if(NOT EXISTS "${cli_path}")
    message(FATAL_ERROR "F13 real CLI does not exist: ${cli_path}")
endif()
if(NOT EXISTS "${fixture_writer_path}")
    message(FATAL_ERROR
        "F13 poison fixture writer does not exist: ${fixture_writer_path}")
endif()
if(NOT EXISTS "${fixture_path}")
    message(FATAL_ERROR "F13 tracked fixture does not exist: ${fixture_path}")
endif()

# The test owns only this exact child of the configured test binary directory.
# Refuse a broad or escaped path before the recursive cleanup.
string(FIND "${work_root}" "${binary_root}/" work_prefix)
if(NOT work_prefix EQUAL 0 OR work_root STREQUAL binary_root)
    message(FATAL_ERROR
        "F13 unsafe work root '${work_root}' outside '${binary_root}'")
endif()
file(REMOVE_RECURSE "${work_root}")
file(MAKE_DIRECTORY "${work_root}")

foreach(precision IN ITEMS f64 f32)
    set(poison_fixture "${work_root}/poison-${precision}.parquet")
    execute_process(
        COMMAND "${fixture_writer_path}" "${poison_fixture}" "${precision}"
        RESULT_VARIABLE writer_result
        OUTPUT_VARIABLE writer_stdout
        ERROR_VARIABLE writer_stderr
        ENCODING UTF-8)
    if(NOT "${writer_result}" STREQUAL "0"
        OR NOT EXISTS "${poison_fixture}")
        message(FATAL_ERROR
            "F13 ${precision} poison fixture generation failed "
            "with exit=${writer_result}\n"
            "stdout:\n${writer_stdout}\nstderr:\n${writer_stderr}")
    endif()
    string(FIND
        "${writer_stdout}"
        "F13_POISON_PARQUET precision=${precision} rows=129 row_groups=2"
        writer_marker)
    if(writer_marker EQUAL -1)
        message(FATAL_ERROR
            "F13 ${precision} poison fixture writer marker missing\n"
            "stdout:\n${writer_stdout}\nstderr:\n${writer_stderr}")
    endif()
endforeach()

set(expected_fixture_sha
    "2F259F418A6BB9C62CA0004CB334C05E8309213F5A76DC890F83C15D5BDA3CA8")
file(SIZE "${fixture_path}" fixture_size)
if(NOT fixture_size EQUAL 1451)
    message(FATAL_ERROR
        "F13 fixture size mismatch: expected=1451 actual=${fixture_size}")
endif()
file(SHA256 "${fixture_path}" fixture_sha)
string(TOUPPER "${fixture_sha}" fixture_sha)
if(NOT fixture_sha STREQUAL expected_fixture_sha)
    message(FATAL_ERROR
        "F13 fixture SHA mismatch: expected=${expected_fixture_sha} actual=${fixture_sha}")
endif()

set(expected_labels [=[name,cluster
series_0,0
series_1,0
series_2,0
series_3,1
series_4,1
series_5,1
series_6,1
series_7,1
]=])
set(expected_medoids [=[cluster,medoid_index,medoid_name
0,0,series_0
1,3,series_3
]=])
set(expected_files
    assignment_checkpoint.bin
    assignment_labels.csv
    assignment_medoids.csv)

set_property(GLOBAL PROPERTY f13_runs)
set_property(GLOBAL PROPERTY f13_route_checks)
set_property(GLOBAL PROPERTY f13_pairs)
set_property(GLOBAL PROPERTY f13_payloads)
set_property(GLOBAL PROPERTY f13_objectives)
set_property(GLOBAL PROPERTY f13_stream_rejections)

function(run_f13_cli precision mode output_dir dtype expected_cost_hex_le
    expected_cost_hex_be)
    file(MAKE_DIRECTORY "${output_dir}")
    set(args
        --input "${fixture_path}"
        --output "${output_dir}"
        --column series
        --method clara
        --n-clusters 2
        --sample-size 2
        --n-samples 1
        --seed 40
        --device cpu
        --name assignment
        --verbose
        --dtype "${dtype}"
        --variant standard)
    if(mode STREQUAL "stream")
        list(APPEND args --ram-limit 900)
    endif()

    execute_process(
        COMMAND "${cli_path}" ${args}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr
        ENCODING UTF-8)
    if(NOT "${result}" STREQUAL "0")
        message(FATAL_ERROR
            "F13 ${precision}/${mode} exit=${result}\n"
            "stdout:\n${stdout}\nstderr:\n${stderr}")
    endif()

    string(REGEX MATCH "[Ss][Kk][Ii][Pp]" skip_match "${stdout}\n${stderr}")
    if(skip_match)
        message(FATAL_ERROR
            "F13 ${precision}/${mode} emitted skip text\n"
            "stdout:\n${stdout}\nstderr:\n${stderr}")
    endif()

    set(eager_marker "Data loaded from Parquet: 8 series")
    set(stream_marker "Parquet metadata selected streaming: 8 series")
    string(FIND "${stdout}" "${eager_marker}" eager_pos)
    string(FIND "${stdout}" "${stream_marker}" stream_pos)
    if(mode STREQUAL "resident")
        if(eager_pos EQUAL -1 OR NOT stream_pos EQUAL -1)
            message(FATAL_ERROR
                "F13 ${precision}/resident did not prove the exclusive eager route\n${stdout}")
        endif()
    else()
        if(stream_pos EQUAL -1 OR NOT eager_pos EQUAL -1)
            message(FATAL_ERROR
                "F13 ${precision}/stream did not prove the exclusive streaming route\n${stdout}")
        endif()
    endif()
    set_property(
        GLOBAL APPEND PROPERTY f13_route_checks
        "${precision}/${mode}:required"
        "${precision}/${mode}:forbidden")

    foreach(execution_marker IN ITEMS
        "Running FastCLARA (k=2)"
        "FastCLARA finished"
        "Variant:  standard"
        "Dtype:    ${dtype}")
        string(FIND "${stdout}" "${execution_marker}" marker_pos)
        if(marker_pos EQUAL -1)
            message(FATAL_ERROR
                "F13 ${precision}/${mode} missing '${execution_marker}'\n${stdout}")
        endif()
    endforeach()

    file(GLOB output_files RELATIVE "${output_dir}" "${output_dir}/*")
    list(SORT output_files)
    if(NOT "${output_files}" STREQUAL "${expected_files}")
        message(FATAL_ERROR
            "F13 ${precision}/${mode} artifacts=${output_files}; "
            "expected=${expected_files}")
    endif()

    file(READ "${output_dir}/assignment_labels.csv" labels)
    file(READ "${output_dir}/assignment_medoids.csv" medoids)
    string(REPLACE "\r\n" "\n" labels "${labels}")
    string(REPLACE "\r\n" "\n" medoids "${medoids}")
    if(NOT labels STREQUAL expected_labels)
        message(FATAL_ERROR
            "F13 ${precision}/${mode} labels payload mismatch:\n${labels}")
    endif()
    if(NOT medoids STREQUAL expected_medoids)
        message(FATAL_ERROR
            "F13 ${precision}/${mode} medoids payload mismatch:\n${medoids}")
    endif()
    set_property(GLOBAL APPEND PROPERTY f13_payloads "${precision}/${mode}")

    file(SIZE "${output_dir}/assignment_checkpoint.bin" checkpoint_size)
    if(NOT checkpoint_size EQUAL 72)
        message(FATAL_ERROR
            "F13 ${precision}/${mode} checkpoint size=${checkpoint_size}, expected=72")
    endif()
    if(BYTE_ORDER STREQUAL "LITTLE_ENDIAN")
        set(expected_cost_hex "${expected_cost_hex_le}")
    elseif(BYTE_ORDER STREQUAL "BIG_ENDIAN")
        set(expected_cost_hex "${expected_cost_hex_be}")
    else()
        message(FATAL_ERROR "F13 unknown C++ byte order '${BYTE_ORDER}'")
    endif()
    file(READ
        "${output_dir}/assignment_checkpoint.bin"
        cost_hex
        OFFSET 24
        LIMIT 8
        HEX)
    string(TOUPPER "${cost_hex}" cost_hex)
    if(NOT cost_hex STREQUAL expected_cost_hex)
        message(FATAL_ERROR
            "F13 ${precision}/${mode} total_cost bytes=${cost_hex}, "
            "expected=${expected_cost_hex}")
    endif()
    set_property(GLOBAL APPEND PROPERTY f13_objectives "${precision}/${mode}")
    set_property(GLOBAL APPEND PROPERTY f13_runs "${precision}/${mode}")
endfunction()

function(require_f13_stream_rejection precision dtype)
    set(input_path "${work_root}/poison-${precision}.parquet")
    set(output_dir "${work_root}/${precision}-stream-rejection")
    file(MAKE_DIRECTORY "${output_dir}")
    execute_process(
        COMMAND
            "${cli_path}"
            --input "${input_path}"
            --output "${output_dir}"
            --column series
            --method clara
            --n-clusters 1
            --sample-size 2
            --n-samples 1
            --seed 0
            --device cpu
            --name assignment
            --verbose
            --dtype "${dtype}"
            --variant standard
            --ram-limit 5000
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr
        ENCODING UTF-8)
    if("${result}" STREQUAL "0")
        message(FATAL_ERROR
            "F13 ${precision} poison stream unexpectedly succeeded\n"
            "stdout:\n${stdout}\nstderr:\n${stderr}")
    endif()

    set(all_output "${stdout}\n${stderr}")
    string(REGEX MATCH "[Ss][Kk][Ii][Pp]" skip_match "${all_output}")
    if(skip_match)
        message(FATAL_ERROR
            "F13 ${precision} poison stream emitted skip text\n"
            "stdout:\n${stdout}\nstderr:\n${stderr}")
    endif()

    foreach(required_marker IN ITEMS
        "Parquet metadata selected streaming: 129 series"
        "Running FastCLARA (k=1)"
        "FastCLARA: streaming from Parquet (129 rows, 2 row groups")
        string(FIND "${all_output}" "${required_marker}" marker_pos)
        if(marker_pos EQUAL -1)
            message(FATAL_ERROR
                "F13 ${precision} poison stream missing '${required_marker}'\n"
                "stdout:\n${stdout}\nstderr:\n${stderr}")
        endif()
    endforeach()
    string(FIND
        "${all_output}" "Data loaded from Parquet: 129 series"
        eager_marker)
    if(NOT eager_marker EQUAL -1)
        message(FATAL_ERROR
            "F13 ${precision} poison stream entered the eager route\n"
            "stdout:\n${stdout}\nstderr:\n${stderr}")
    endif()

    string(REPLACE "\r\n" "\n" normalized_stderr "${stderr}")
    string(REGEX REPLACE "\n+$" "" normalized_stderr "${normalized_stderr}")
    set(expected_stderr
        "Error: fast_clara: non-finite nearest-medoid distance at point 0, medoid slot 0 (index 65).")
    if(NOT normalized_stderr STREQUAL expected_stderr)
        message(FATAL_ERROR
            "F13 ${precision} poison stream stderr mismatch\n"
            "expected:\n${expected_stderr}\nactual:\n${normalized_stderr}")
    endif()
    set_property(
        GLOBAL APPEND PROPERTY f13_stream_rejections "${precision}")
endfunction()

function(check_f13_precision precision dtype expected_cost_hex_le
    expected_cost_hex_be)
    set(resident_dir "${work_root}/${precision}-resident")
    set(stream_dir "${work_root}/${precision}-stream")

    run_f13_cli(
        "${precision}" resident "${resident_dir}" "${dtype}"
        "${expected_cost_hex_le}" "${expected_cost_hex_be}")
    run_f13_cli(
        "${precision}" stream "${stream_dir}" "${dtype}"
        "${expected_cost_hex_le}" "${expected_cost_hex_be}")

    foreach(artifact IN LISTS expected_files)
        execute_process(
            COMMAND
                "${CMAKE_COMMAND}" -E compare_files
                "${resident_dir}/${artifact}"
                "${stream_dir}/${artifact}"
            RESULT_VARIABLE compare_result)
        if(NOT "${compare_result}" STREQUAL "0")
            message(FATAL_ERROR
                "F13 ${precision}/${artifact} differs byte-for-byte")
        endif()
        file(SHA256 "${resident_dir}/${artifact}" resident_sha)
        file(SHA256 "${stream_dir}/${artifact}" stream_sha)
        string(TOUPPER "${resident_sha}" resident_sha)
        string(TOUPPER "${stream_sha}" stream_sha)
        if(NOT resident_sha STREQUAL stream_sha)
            message(FATAL_ERROR
                "F13 ${precision}/${artifact} resident=${resident_sha} "
                "stream=${stream_sha}")
        endif()
        set_property(GLOBAL APPEND PROPERTY f13_pairs "${precision}/${artifact}")
    endforeach()
endfunction()

check_f13_precision(
    f64 float64
    9A99999999896340 406389999999999A)
check_f13_precision(
    f32 float32
    0000809E99896340 406389999E800000)
require_f13_stream_rejection(f64 float64)
require_f13_stream_rejection(f32 float32)

get_property(f13_runs GLOBAL PROPERTY f13_runs)
get_property(f13_route_checks GLOBAL PROPERTY f13_route_checks)
get_property(f13_pairs GLOBAL PROPERTY f13_pairs)
get_property(f13_payloads GLOBAL PROPERTY f13_payloads)
get_property(f13_objectives GLOBAL PROPERTY f13_objectives)
get_property(
    f13_stream_rejections GLOBAL PROPERTY f13_stream_rejections)
list(LENGTH f13_runs run_count)
list(LENGTH f13_route_checks route_check_count)
list(LENGTH f13_pairs pair_count)
list(LENGTH f13_payloads payload_count)
list(LENGTH f13_objectives objective_count)
list(LENGTH f13_stream_rejections stream_rejection_count)
if(NOT run_count EQUAL 4
    OR NOT route_check_count EQUAL 8
    OR NOT pair_count EQUAL 6
    OR NOT payload_count EQUAL 4
    OR NOT objective_count EQUAL 4
    OR NOT stream_rejection_count EQUAL 2)
    message(FATAL_ERROR
        "F13 counter mismatch: runs=${run_count} routes=${route_check_count} "
        "pairs=${pair_count} payloads=${payload_count} "
        "objectives=${objective_count} "
        "stream_rejections=${stream_rejection_count}")
endif()

message(STATUS
    "F13_ASSIGNMENT_CONTRACT subject=real_dtwc_cl runs=${run_count}/4 "
    "route_markers=${route_check_count}/8 artifact_pairs=${pair_count}/6 "
    "payloads=${payload_count}/4 objective_bytes=${objective_count}/4 "
    "stream_rejections=${stream_rejection_count}/2 "
    "skips=0 fixture_sha256=${fixture_sha}")
