# @file test_fast_clara_parquet_parity.cmake
# @brief CLI integration test verifying streamed FastCLARA parity against in-RAM Parquet runs.
# @author Volkan Kumtepeli
# @date 23 Jul 2026

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

# The test owns only this exact child of the configured test binary directory.
# Refuse a broad or escaped path before the recursive cleanup.
string(FIND "${work_root}" "${binary_root}/" work_prefix)
if(NOT work_prefix EQUAL 0 OR work_root STREQUAL binary_root)
    message(FATAL_ERROR
        "F8 unsafe work root '${work_root}' outside '${binary_root}'")
endif()
file(REMOVE_RECURSE "${work_root}")
file(MAKE_DIRECTORY "${work_root}")

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

# A total cost agrees with its expected value to 12 significant digits: the costs here lie in [1, 10), so the two
# may differ by at most 5e-12, half a unit in the 12th digit. Compilers round the last bits differently (FMA
# contraction, reassociation); that moves no clustering. CMake has no floating point, so the text (plain decimal) is
# read as a whole count of 1e-13: its digit, then its first 13 decimals.
function(f8_cost_units text out_var)
    if(NOT text MATCHES "^([1-9])(\\.([0-9]*))?$")
        message(FATAL_ERROR "F8 total cost '${text}' is not a plain decimal in [1, 10)")
    endif()
    string(SUBSTRING "${CMAKE_MATCH_3}0000000000000" 0 13 decimals)
    set("${out_var}" "${CMAKE_MATCH_1}${decimals}" PARENT_SCOPE)
endfunction()

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
    # The limit holds the stream (the medoids, two samples' labels and nearest
    # distances, one decode of the 8 rows, one row group) below the in-RAM
    # estimate with CLARA's pass, so the run streams. Both prices count a
    # std::string a row, 32 bytes in libstdc++ and MSVC and 24 in libc++: float64
    # streams in [4550, 4766) and [4502, 4702) under libc++, float32 in
    # [4442, 4622) and [4394, 4558). That is the data cap: the limit is it beside
    # the reserve for the threads' working set, which -v's memory plan names.
    if(mode STREQUAL "stream")
        if(dtype STREQUAL "float32")
            set(data_cap 4500)
        else()
            set(data_cap 4600)
        endif()
        set(probe_args ${args})
        list(TRANSFORM probe_args REPLACE "^${output_dir}$" "${output_dir}-probe")
        execute_process(
            COMMAND "${cli_path}" ${probe_args} --ram-limit ${data_cap}
            RESULT_VARIABLE probe_result
            OUTPUT_VARIABLE probe_stdout
            ERROR_VARIABLE probe_stderr
            ENCODING UTF-8)
        if(NOT probe_stdout MATCHES "Memory plan: budget [0-9]+ bytes, reserve ([0-9]+) bytes")
            message(FATAL_ERROR "F8 ${config_id}/stream: no memory plan at --ram-limit ${data_cap}\n"
                                "stdout:\n${probe_stdout}\nstderr:\n${probe_stderr}")
        endif()
        math(EXPR limit "${CMAKE_MATCH_1} + ${data_cap}")
        list(APPEND args --ram-limit ${limit})
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
    f8_cost_units("${cost}" cost_units)
    f8_cost_units("${expected_cost}" expected_units)
    math(EXPR difference "${cost_units} - ${expected_units}")
    if(difference GREATER 50 OR difference LESS -50)
        message(FATAL_ERROR
            "F8 ${config_id}/${mode} total cost=${cost}, expected ${expected_cost} to 12 significant digits")
    endif()

    set_property(GLOBAL APPEND PROPERTY f8_runs "${config_id}/${mode}")
    set("${config_id}_${mode}_cost" "${cost}" PARENT_SCOPE)
endfunction()

function(check_f8_config config_id dtype variant gamma expected_cost)
    set(resident_dir "${work_root}/${config_id}-resident")
    set(stream_dir "${work_root}/${config_id}-stream")

    run_f8_cli(
        "${config_id}" resident "${resident_dir}"
        "${dtype}" "${variant}" "${gamma}" "${expected_cost}")
    run_f8_cli(
        "${config_id}" stream "${stream_dir}"
        "${dtype}" "${variant}" "${gamma}" "${expected_cost}")

    set(expected_files
        parity_labels.csv
        parity_medoids.csv)
    foreach(mode IN ITEMS resident stream)
        set(output_dir "${${mode}_dir}")
        file(GLOB output_files RELATIVE "${output_dir}" "${output_dir}/*")
        # The run report is the run's own account (times, the machine): not an artefact the routes share.
        list(FILTER output_files EXCLUDE REGEX "_report\\.txt$")
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

    # The payload check above folds CRLF to LF; these hashes pin the files' own bytes (CRLF rows on Windows).
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

# Each total cost to 12 significant digits (f8_cost_units); labels/medoids stay exact, and resident and stream
# byte-identical.
check_f8_config(f64_standard float64 standard none "4.399999999999999")
check_f8_config(f32_standard float32 standard none "4.400001227855682")
check_f8_config(f64_softdtw float64 softdtw 0.7 "4.604996263691627")

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
    "configs=${config_count}/3_distinct softdtw_cost=${f64_softdtw_cost}")
