cmake_minimum_required(VERSION 3.26)

foreach(required_var IN ITEMS CLI FIXTURE BINARY_ROOT WORK_ROOT BYTE_ORDER)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        message(FATAL_ERROR "F8 missing required -D${required_var}=...")
    endif()
endforeach()

cmake_path(ABSOLUTE_PATH CLI NORMALIZE OUTPUT_VARIABLE cli_path)
cmake_path(ABSOLUTE_PATH FIXTURE NORMALIZE OUTPUT_VARIABLE fixture_path)
cmake_path(ABSOLUTE_PATH BINARY_ROOT NORMALIZE OUTPUT_VARIABLE binary_root)
cmake_path(ABSOLUTE_PATH WORK_ROOT NORMALIZE OUTPUT_VARIABLE work_root)

if(NOT EXISTS "${cli_path}")
    message(FATAL_ERROR "F8 real CLI does not exist: ${cli_path}")
endif()
if(NOT EXISTS "${fixture_path}")
    message(FATAL_ERROR "F8 tracked fixture does not exist: ${fixture_path}")
endif()

# The test owns only this exact child of the configured test binary directory.
# Refuse a broad or escaped path before the recursive cleanup.
string(FIND "${work_root}" "${binary_root}/" work_prefix)
if(NOT work_prefix EQUAL 0 OR work_root STREQUAL binary_root)
    message(FATAL_ERROR
        "F8 unsafe work root '${work_root}' outside '${binary_root}'")
endif()
file(REMOVE_RECURSE "${work_root}")
file(MAKE_DIRECTORY "${work_root}")

set(expected_fixture_sha
    "2F259F418A6BB9C62CA0004CB334C05E8309213F5A76DC890F83C15D5BDA3CA8")
file(SIZE "${fixture_path}" fixture_size)
if(NOT fixture_size EQUAL 1451)
    message(FATAL_ERROR
        "F8 fixture size mismatch: expected=1451 actual=${fixture_size}")
endif()
file(SHA256 "${fixture_path}" fixture_sha)
string(TOUPPER "${fixture_sha}" fixture_sha)
if(NOT fixture_sha STREQUAL expected_fixture_sha)
    message(FATAL_ERROR
        "F8 fixture SHA mismatch: expected=${expected_fixture_sha} actual=${fixture_sha}")
endif()

set(expected_labels [=[name,cluster
series_0,0
series_1,0
series_2,0
series_3,0
series_4,1
series_5,1
series_6,1
series_7,1
]=])
set(expected_medoids [=[cluster,medoid_index,medoid_name
0,1,series_1
1,7,series_7
]=])
set(expected_labels_sha
    "39CCD0E9D5F520678899193B7A7903D6331217B0F99C96417F24BF943A7268EB")
set(expected_medoids_sha
    "2CEF6784BE8808E00B6F78D3EF2DE8E55C6D4E7021F9C81092EB6A0279E3D22F")

set_property(GLOBAL PROPERTY f8_runs)
set_property(GLOBAL PROPERTY f8_route_checks)
set_property(GLOBAL PROPERTY f8_pairs)
set_property(GLOBAL PROPERTY f8_configs)

function(run_f8_cli config_id mode output_dir dtype variant gamma expected_cost)
    file(MAKE_DIRECTORY "${output_dir}")
    set(args
        --input "${fixture_path}"
        --output "${output_dir}"
        --column series
        --method clara
        --n-clusters 2
        --sample-size 4
        --n-samples 2
        --seed 42
        --device cpu
        --name parity
        --verbose
        --dtype "${dtype}"
        --variant "${variant}")
    if(NOT gamma STREQUAL "none")
        list(APPEND args --sdtw-gamma "${gamma}")
    endif()
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
            "F8 ${config_id}/${mode} exit=${result}\nstdout:\n${stdout}\nstderr:\n${stderr}")
    endif()

    set(eager_marker "Data loaded from Parquet: 8 series")
    set(stream_marker "Parquet metadata selected streaming: 8 series")
    string(FIND "${stdout}" "${eager_marker}" eager_pos)
    string(FIND "${stdout}" "${stream_marker}" stream_pos)
    if(mode STREQUAL "resident")
        if(eager_pos EQUAL -1 OR NOT stream_pos EQUAL -1)
            message(FATAL_ERROR
                "F8 ${config_id}/resident did not prove the eager route\n${stdout}")
        endif()
    else()
        if(stream_pos EQUAL -1 OR NOT eager_pos EQUAL -1)
            message(FATAL_ERROR
                "F8 ${config_id}/stream did not prove the streaming route\n${stdout}")
        endif()
    endif()
    set_property(
        GLOBAL APPEND PROPERTY f8_route_checks
        "${config_id}/${mode}:required"
        "${config_id}/${mode}:forbidden")

    foreach(execution_marker IN ITEMS
        "Running FastCLARA (k=2)"
        "FastCLARA finished"
        "Variant:  ${variant}"
        "Dtype:    ${dtype}")
        string(FIND "${stdout}" "${execution_marker}" marker_pos)
        if(marker_pos EQUAL -1)
            message(FATAL_ERROR
                "F8 ${config_id}/${mode} missing '${execution_marker}'\n${stdout}")
        endif()
    endforeach()

    string(FIND "${stdout}" "Total cost: ${expected_cost}" cost_pos)
    if(cost_pos EQUAL -1)
        message(FATAL_ERROR
            "F8 ${config_id}/${mode} missing cost ${expected_cost}\n${stdout}")
    endif()

    set_property(GLOBAL APPEND PROPERTY f8_runs "${config_id}/${mode}")
    set("${config_id}_${mode}_stdout" "${stdout}" PARENT_SCOPE)
endfunction()

function(check_f8_config
    config_id dtype variant gamma expected_cost
    expected_cost_hex_le expected_cost_hex_be expected_checkpoint_sha)
    set(resident_dir "${work_root}/${config_id}-resident")
    set(stream_dir "${work_root}/${config_id}-stream")

    run_f8_cli(
        "${config_id}" resident "${resident_dir}"
        "${dtype}" "${variant}" "${gamma}" "${expected_cost}")
    run_f8_cli(
        "${config_id}" stream "${stream_dir}"
        "${dtype}" "${variant}" "${gamma}" "${expected_cost}")

    set(expected_files
        parity_checkpoint.bin
        parity_labels.csv
        parity_medoids.csv)
    foreach(mode IN ITEMS resident stream)
        set(output_dir "${${mode}_dir}")
        file(GLOB output_files RELATIVE "${output_dir}" "${output_dir}/*")
        list(SORT output_files)
        if(NOT "${output_files}" STREQUAL "${expected_files}")
            message(FATAL_ERROR
                "F8 ${config_id}/${mode} artifacts=${output_files}; expected=${expected_files}")
        endif()
    endforeach()

    foreach(artifact IN LISTS expected_files)
        execute_process(
            COMMAND
                "${CMAKE_COMMAND}" -E compare_files
                "${resident_dir}/${artifact}"
                "${stream_dir}/${artifact}"
            RESULT_VARIABLE compare_result)
        if(NOT "${compare_result}" STREQUAL "0")
            message(FATAL_ERROR
                "F8 ${config_id}/${artifact} differs byte-for-byte")
        endif()
        file(SHA256 "${resident_dir}/${artifact}" resident_sha)
        file(SHA256 "${stream_dir}/${artifact}" stream_sha)
        string(TOUPPER "${resident_sha}" resident_sha)
        string(TOUPPER "${stream_sha}" stream_sha)
        if(NOT resident_sha STREQUAL stream_sha)
            message(FATAL_ERROR
                "F8 ${config_id}/${artifact} resident=${resident_sha} stream=${stream_sha}")
        endif()
        set_property(
            GLOBAL APPEND PROPERTY f8_pairs "${config_id}/${artifact}")
    endforeach()

    file(READ "${resident_dir}/parity_labels.csv" labels)
    file(READ "${resident_dir}/parity_medoids.csv" medoids)
    string(REPLACE "\r\n" "\n" labels "${labels}")
    string(REPLACE "\r\n" "\n" medoids "${medoids}")
    if(NOT labels STREQUAL expected_labels)
        message(FATAL_ERROR "F8 ${config_id} labels payload mismatch:\n${labels}")
    endif()
    if(NOT medoids STREQUAL expected_medoids)
        message(FATAL_ERROR "F8 ${config_id} medoids payload mismatch:\n${medoids}")
    endif()

    file(SIZE "${resident_dir}/parity_checkpoint.bin" checkpoint_size)
    if(NOT checkpoint_size EQUAL 72)
        message(FATAL_ERROR
            "F8 ${config_id} checkpoint size=${checkpoint_size}, expected=72")
    endif()
    file(SHA256 "${resident_dir}/parity_checkpoint.bin" checkpoint_sha)
    string(TOUPPER "${checkpoint_sha}" checkpoint_sha)

    if(BYTE_ORDER STREQUAL "LITTLE_ENDIAN")
        set(expected_cost_hex "${expected_cost_hex_le}")
    elseif(BYTE_ORDER STREQUAL "BIG_ENDIAN")
        set(expected_cost_hex "${expected_cost_hex_be}")
    else()
        message(FATAL_ERROR "F8 unknown C++ byte order '${BYTE_ORDER}'")
    endif()
    foreach(mode IN ITEMS resident stream)
        file(READ
            "${${mode}_dir}/parity_checkpoint.bin"
            cost_hex
            OFFSET 24
            LIMIT 8
            HEX)
        string(TOUPPER "${cost_hex}" cost_hex)
        if(NOT cost_hex STREQUAL expected_cost_hex)
            message(FATAL_ERROR
                "F8 ${config_id}/${mode} total_cost bytes=${cost_hex}, "
                "expected=${expected_cost_hex}")
        endif()
    endforeach()

    if(WIN32)
        file(SHA256 "${resident_dir}/parity_labels.csv" labels_sha)
        file(SHA256 "${resident_dir}/parity_medoids.csv" medoids_sha)
        string(TOUPPER "${labels_sha}" labels_sha)
        string(TOUPPER "${medoids_sha}" medoids_sha)
        if(NOT labels_sha STREQUAL expected_labels_sha)
            message(FATAL_ERROR
                "F8 ${config_id} labels SHA=${labels_sha}, expected=${expected_labels_sha}")
        endif()
        if(NOT medoids_sha STREQUAL expected_medoids_sha)
            message(FATAL_ERROR
                "F8 ${config_id} medoids SHA=${medoids_sha}, expected=${expected_medoids_sha}")
        endif()
        if(NOT checkpoint_sha STREQUAL expected_checkpoint_sha)
            message(FATAL_ERROR
                "F8 ${config_id} checkpoint SHA=${checkpoint_sha}, expected=${expected_checkpoint_sha}")
        endif()
    endif()

    set_property(GLOBAL APPEND PROPERTY f8_configs "${config_id}")
    set("${config_id}_checkpoint_sha" "${checkpoint_sha}" PARENT_SCOPE)
endfunction()

check_f8_config(
    f64_standard float64 standard none 4.4
    9899999999991140 4011999999999998
    B666C1108C42A6B0705741B357F2527E319FEEBE3380BE6623E1494DAFD1A02C)
check_f8_config(
    f32_standard float32 standard none 4.4
    000000EC99991140 40119999EC000000
    EEA65070341F900BA212909242AD54FE7A48F1E4A94704864CBF77C89FC28FC1)
check_f8_config(
    f64_softdtw float64 softdtw 0.7 -10.344
    CC31540B20B024C0 C024B0200B5431CC
    67D818D58370CB6E17E4E8922175A343ADE2D53C7873A62522FE9D3C14734B79)

if(f64_standard_checkpoint_sha STREQUAL f32_standard_checkpoint_sha
    OR f64_standard_checkpoint_sha STREQUAL f64_softdtw_checkpoint_sha
    OR f32_standard_checkpoint_sha STREQUAL f64_softdtw_checkpoint_sha)
    message(FATAL_ERROR
        "F8 configuration checkpoints are not 3/3 distinct: "
        "${f64_standard_checkpoint_sha};${f32_standard_checkpoint_sha};"
        "${f64_softdtw_checkpoint_sha}")
endif()

get_property(f8_runs GLOBAL PROPERTY f8_runs)
get_property(f8_route_checks GLOBAL PROPERTY f8_route_checks)
get_property(f8_pairs GLOBAL PROPERTY f8_pairs)
get_property(f8_configs GLOBAL PROPERTY f8_configs)
list(LENGTH f8_runs run_count)
list(LENGTH f8_route_checks route_check_count)
list(LENGTH f8_pairs pair_count)
list(LENGTH f8_configs config_count)
if(NOT run_count EQUAL 6
    OR NOT route_check_count EQUAL 12
    OR NOT pair_count EQUAL 9
    OR NOT config_count EQUAL 3)
    message(FATAL_ERROR
        "F8 counter mismatch: runs=${run_count} routes=${route_check_count} "
        "pairs=${pair_count} configs=${config_count}")
endif()

message(STATUS
    "F8_PARITY subject=real_dtwc_cl runs=${run_count} "
    "route_markers=${route_check_count}/12 parity=${pair_count}/9 "
    "configs=${config_count}/3_distinct fixture_sha256=${fixture_sha}")
