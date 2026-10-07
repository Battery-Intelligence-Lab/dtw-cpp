cmake_minimum_required(VERSION 3.26)

foreach(required_var IN ITEMS CLI FIXTURE BINARY_ROOT WORK_ROOT)
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

function(run_f8_cli config_id mode output_dir dtype variant gamma accepted_costs)
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

    # dtwc_cl prints the total cost in shortest round-trip form, so this text
    # is the exact double.
    if(NOT stdout MATCHES "Total cost: ([^\r\n]+)")
        message(FATAL_ERROR
            "F8 ${config_id}/${mode} missing 'Total cost:'\n${stdout}")
    endif()
    set(cost "${CMAKE_MATCH_1}")
    list(FIND accepted_costs "${cost}" accepted_index)
    if(accepted_index EQUAL -1)
        message(FATAL_ERROR
            "F8 ${config_id}/${mode} total cost=${cost}, accepted=${accepted_costs}")
    endif()

    set_property(GLOBAL APPEND PROPERTY f8_runs "${config_id}/${mode}")
    set("${config_id}_${mode}_cost" "${cost}" PARENT_SCOPE)
endfunction()

function(check_f8_config config_id dtype variant gamma accepted_costs)
    set(resident_dir "${work_root}/${config_id}-resident")
    set(stream_dir "${work_root}/${config_id}-stream")

    run_f8_cli(
        "${config_id}" resident "${resident_dir}"
        "${dtype}" "${variant}" "${gamma}" "${accepted_costs}")
    run_f8_cli(
        "${config_id}" stream "${stream_dir}"
        "${dtype}" "${variant}" "${gamma}" "${accepted_costs}")

    set(expected_files
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

    # Resident and stream must agree on the exact total cost, not only on the
    # accepted set.
    if(NOT "${${config_id}_resident_cost}" STREQUAL "${${config_id}_stream_cost}")
        message(FATAL_ERROR
            "F8 ${config_id} resident/stream total cost split: "
            "${${config_id}_resident_cost} vs ${${config_id}_stream_cost}")
    endif()
    set_property(GLOBAL APPEND PROPERTY f8_pairs "${config_id}/total_cost")

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
    endif()

    set_property(GLOBAL APPEND PROPERTY f8_configs "${config_id}")
    set("${config_id}_cost" "${${config_id}_resident_cost}" PARENT_SCOPE)
endfunction()

# Accepted total costs are the shortest round-trip text of these IEEE-754
# values (big-endian hex):
# 4011999999999998, 40119999EC000000, and for Soft-DTW C024B0200B5431CC
# (the MSVC and Apple Clang baseline of 2026-07-23) or C024B0200B5431CE (GCC 14 +
# -fassociative-math, 2 ULP; Arrhenius LastTest.log 2026-09-01).
# Labels/medoids stay exact. Resident and stream stay byte-identical.
check_f8_config(f64_standard float64 standard none "4.399999999999999")
check_f8_config(f32_standard float32 standard none "4.400001227855682")
check_f8_config(f64_softdtw float64 softdtw 0.7
    "-10.343994478252078;-10.343994478252082")

# The labels and medoids are the same in all three configurations; the exact
# cost is what tells them apart.
if(f64_standard_cost STREQUAL f32_standard_cost
    OR f64_standard_cost STREQUAL f64_softdtw_cost
    OR f32_standard_cost STREQUAL f64_softdtw_cost)
    message(FATAL_ERROR
        "F8 configuration costs are not 3/3 distinct: "
        "${f64_standard_cost};${f32_standard_cost};${f64_softdtw_cost}")
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
    "configs=${config_count}/3_distinct fixture_sha256=${fixture_sha} "
    "softdtw_cost=${f64_softdtw_cost}")
