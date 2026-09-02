cmake_minimum_required(VERSION 3.26)

foreach(required_var IN ITEMS
        CLI RESULT_WRITER INPUT CONFIG BINARY_ROOT WORK_ROOT HAS_MMAP)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        message(FATAL_ERROR "F14 missing required -D${required_var}=...")
    endif()
endforeach()

cmake_path(ABSOLUTE_PATH CLI NORMALIZE OUTPUT_VARIABLE cli_path)
cmake_path(
    ABSOLUTE_PATH RESULT_WRITER NORMALIZE
    OUTPUT_VARIABLE result_writer_path)
cmake_path(ABSOLUTE_PATH INPUT NORMALIZE OUTPUT_VARIABLE input_path)
cmake_path(ABSOLUTE_PATH CONFIG NORMALIZE OUTPUT_VARIABLE config_path)
cmake_path(
    ABSOLUTE_PATH BINARY_ROOT NORMALIZE OUTPUT_VARIABLE binary_root)
cmake_path(ABSOLUTE_PATH WORK_ROOT NORMALIZE OUTPUT_VARIABLE work_root)

foreach(executable IN ITEMS cli_path result_writer_path)
    if(NOT EXISTS "${${executable}}")
        message(FATAL_ERROR "F14 executable does not exist: ${${executable}}")
    endif()
endforeach()
foreach(fixture IN ITEMS input_path config_path)
    if(NOT EXISTS "${${fixture}}")
        message(FATAL_ERROR "F14 tracked fixture does not exist: ${${fixture}}")
    endif()
endforeach()
if(NOT IS_DIRECTORY "${binary_root}")
    message(FATAL_ERROR "F14 binary root is not a directory: ${binary_root}")
endif()

string(TOUPPER "${HAS_MMAP}" has_mmap)
if(NOT has_mmap STREQUAL "ON" AND NOT has_mmap STREQUAL "OFF")
    message(FATAL_ERROR "F14 HAS_MMAP must be exactly ON or OFF: ${HAS_MMAP}")
endif()

# This gate may recursively clean only its registered exact child of the
# configured tests binary directory. Reject symlinks and resolved-path escapes
# before touching an existing directory.
cmake_path(
    APPEND binary_root "f14-distance-matrix-csv-public"
    OUTPUT_VARIABLE expected_work_root)
if(NOT work_root STREQUAL expected_work_root)
    message(FATAL_ERROR
        "F14 unsafe work root '${work_root}'; expected exact child "
        "'${expected_work_root}'")
endif()
if(IS_SYMLINK "${work_root}")
    message(FATAL_ERROR "F14 work root must not be a symlink: ${work_root}")
endif()
file(REAL_PATH "${binary_root}" binary_root_real)
cmake_path(
    APPEND binary_root_real "f14-distance-matrix-csv-public"
    OUTPUT_VARIABLE expected_work_real)
if(EXISTS "${work_root}")
    file(REAL_PATH "${work_root}" existing_work_real)
    if(NOT existing_work_real STREQUAL expected_work_real)
        message(FATAL_ERROR
            "F14 resolved work root escaped: '${existing_work_real}'")
    endif()
endif()
file(REMOVE_RECURSE "${work_root}")
file(MAKE_DIRECTORY "${work_root}")

# The tracked conformance inputs are read-only. Their hashes are captured before
# every run and must remain identical afterward.
file(SHA256 "${input_path}" input_sha_before)
file(SHA256 "${config_path}" config_sha_before)
file(SIZE "${input_path}" input_size_before)
file(SIZE "${config_path}" config_size_before)
file(STRINGS "${input_path}" input_rows)
list(LENGTH input_rows input_row_count)
if(NOT input_row_count EQUAL 27)
    message(FATAL_ERROR
        "F14 conformance input rows=${input_row_count}, expected=27")
endif()

function(require_config_line regex description)
    file(STRINGS "${config_path}" matches REGEX "${regex}")
    list(LENGTH matches match_count)
    if(NOT match_count EQUAL 1)
        message(FATAL_ERROR
            "F14 conformance TOML must contain one ${description}; "
            "found=${match_count}")
    endif()
endfunction()

require_config_line(
    "^n-clusters[ \t]*=[ \t]*3[ \t]*$" "n-clusters=3 setting")
require_config_line(
    "^method[ \t]*=[ \t]*\"pam\"[ \t]*$" "method=pam setting")
require_config_line(
    "^band[ \t]*=[ \t]*3[ \t]*$" "band=3 setting")
require_config_line(
    "^metric[ \t]*=[ \t]*\"l1\"[ \t]*$" "metric=l1 setting")
require_config_line(
    "^variant[ \t]*=[ \t]*\"standard\"[ \t]*$"
    "variant=standard setting")
require_config_line(
    "^max-iter[ \t]*=[ \t]*100[ \t]*$" "max-iter=100 setting")
require_config_line(
    "^name[ \t]*=[ \t]*\"conformance\"[ \t]*$"
    "name=conformance setting")

set_property(GLOBAL PROPERTY f14_runs)
set_property(GLOBAL PROPERTY f14_route_markers)
set_property(GLOBAL PROPERTY f14_matrix_pairs)

function(require_occurrences haystack needle expected context)
    string(LENGTH "${haystack}" before_length)
    string(LENGTH "${needle}" needle_length)
    if(needle_length EQUAL 0)
        message(FATAL_ERROR "F14 internal empty marker for ${context}")
    endif()
    string(REPLACE "${needle}" "" stripped "${haystack}")
    string(LENGTH "${stripped}" after_length)
    math(EXPR removed_length "${before_length} - ${after_length}")
    math(EXPR occurrence_count "${removed_length} / ${needle_length}")
    if(NOT occurrence_count EQUAL expected)
        message(FATAL_ERROR
            "F14 ${context} marker count=${occurrence_count}, "
            "expected=${expected}\nmarker=${needle}\noutput:\n${haystack}")
    endif()
endfunction()

function(require_no_skip context stdout stderr)
    string(
        REGEX MATCH "[Ss][Kk][Ii][Pp]([Pp]|[ :])"
        skip_match "${stdout}\n${stderr}")
    if(skip_match)
        message(FATAL_ERROR
            "F14 ${context} emitted skip text\n"
            "stdout:\n${stdout}\nstderr:\n${stderr}")
    endif()
endfunction()

function(run_cli_route route output_dir threshold expect_mmap)
    execute_process(
        COMMAND
            "${cli_path}"
            --config "${config_path}"
            --input "${input_path}"
            --output "${output_dir}"
            --mmap-threshold "${threshold}"
            --device cpu
            --verbose
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr
        ENCODING UTF-8)
    if(NOT "${result}" STREQUAL "0")
        message(FATAL_ERROR
            "F14 ${route} CLI exit=${result}\n"
            "stdout:\n${stdout}\nstderr:\n${stderr}")
    endif()
    require_no_skip("${route} CLI" "${stdout}" "${stderr}")

    # std::filesystem::path streams through std::quoted, so a Windows
    # separator arrives as an escaped pair. Collapse the escaped form
    # first so both platforms normalize to a single "/" and the exact
    # path markers below stay platform-neutral.
    string(REPLACE "\\\\" "/" normalized_stdout "${stdout}")
    string(REPLACE "\\" "/" normalized_stdout "${normalized_stdout}")
    set(matrix_path "${output_dir}/conformance_distance_matrix.csv")
    set(cache_path "${output_dir}/conformance_distmat.cache")
    set(matrix_marker "Distance matrix written to")
    require_occurrences(
        "${normalized_stdout}" "${matrix_marker}" 1
        "${route} distance-matrix output")
    require_occurrences(
        "${normalized_stdout}"
        "\"${matrix_path}\"" 1
        "${route} exact distance-matrix path")
    foreach(execution_marker IN ITEMS
            "DTWC++ Clustering"
            "Data loaded: 27 series"
            "Running FastPAM (k=3)"
            "FastPAM converged")
        require_occurrences(
            "${normalized_stdout}" "${execution_marker}" 1
            "${route} ${execution_marker}")
    endforeach()

    set(generic_mmap_marker "Using memory-mapped distance matrix:")
    if(expect_mmap)
        require_occurrences(
            "${normalized_stdout}" "${generic_mmap_marker}" 1
            "${route} mmap route")
        require_occurrences(
            "${normalized_stdout}"
            "\"${cache_path}\"" 1
            "${route} exact mmap cache")
        if(NOT EXISTS "${cache_path}")
            message(FATAL_ERROR
                "F14 ${route} mmap marker has no cache: ${cache_path}")
        endif()
        file(SIZE "${cache_path}" cache_size)
        file(SHA256 "${cache_path}" cache_sha)
        string(TOUPPER "${cache_sha}" cache_sha)
        if(cache_size LESS_EQUAL 0)
            message(FATAL_ERROR "F14 ${route} mmap cache is empty")
        endif()
        message(STATUS
            "F14_CSV_CACHE route=${route} bytes=${cache_size} "
            "sha256=${cache_sha} marker=1")
    else()
        require_occurrences(
            "${normalized_stdout}" "${generic_mmap_marker}" 0
            "${route} resident exclusion")
        if(EXISTS "${cache_path}")
            message(FATAL_ERROR
                "F14 ${route} unexpectedly created mmap cache: ${cache_path}")
        endif()
    endif()

    set_property(GLOBAL APPEND PROPERTY f14_runs "${route}")
    set_property(GLOBAL APPEND PROPERTY f14_route_markers "${route}")
endfunction()

function(run_result_route output_dir)
    execute_process(
        COMMAND "${result_writer_path}" "${input_path}" "${output_dir}"
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr
        ENCODING UTF-8)
    if(NOT "${result}" STREQUAL "0")
        message(FATAL_ERROR
            "F14 native Result::save exit=${result}\n"
            "stdout:\n${stdout}\nstderr:\n${stderr}")
    endif()
    require_no_skip("native Result::save" "${stdout}" "${stderr}")
    require_occurrences(
        "${stdout}"
        "F14_RESULT_SAVE subject=native_result labels=27 medoids=3 device=cpu"
        1
        "native Result::save")
    if(EXISTS "${output_dir}/conformance_distmat.cache")
        message(FATAL_ERROR "F14 native Result::save unexpectedly used mmap")
    endif()
    set_property(GLOBAL APPEND PROPERTY f14_runs "result")
endfunction()

function(inspect_matrix route matrix_path)
    if(NOT EXISTS "${matrix_path}")
        message(FATAL_ERROR "F14 ${route} matrix missing: ${matrix_path}")
    endif()
    file(SIZE "${matrix_path}" matrix_size)
    file(SHA256 "${matrix_path}" matrix_sha)
    string(TOUPPER "${matrix_sha}" matrix_sha)
    file(READ "${matrix_path}" matrix_hex HEX)
    string(TOLOWER "${matrix_hex}" matrix_hex)

    if(matrix_size LESS_EQUAL 0)
        message(FATAL_ERROR "F14 ${route} matrix is empty")
    endif()
    if(matrix_hex MATCHES "^efbbbf")
        message(FATAL_ERROR "F14 ${route} matrix contains a UTF-8 BOM")
    endif()

    string(REGEX MATCHALL "0a" lf_matches "${matrix_hex}")
    string(REGEX MATCHALL "0d" cr_matches "${matrix_hex}")
    list(LENGTH lf_matches lf_count)
    list(LENGTH cr_matches cr_count)
    if(NOT lf_count EQUAL 27 OR NOT cr_count EQUAL 0)
        message(FATAL_ERROR
            "F14 ${route} raw terminators lf=${lf_count} cr=${cr_count}; "
            "expected lf=27 cr=0")
    endif()

    string(LENGTH "${matrix_hex}" hex_length)
    math(EXPR final_byte_offset "${hex_length} - 2")
    string(SUBSTRING "${matrix_hex}" ${final_byte_offset} 2 final_byte)
    if(NOT final_byte STREQUAL "0a")
        message(FATAL_ERROR "F14 ${route} matrix has no final LF byte")
    endif()
    string(FIND "${matrix_hex}" "0a0a" blank_line_offset)
    if(NOT blank_line_offset EQUAL -1)
        message(FATAL_ERROR
            "F14 ${route} matrix contains a blank row at raw hex offset "
            "${blank_line_offset}")
    endif()

    file(STRINGS "${matrix_path}" matrix_rows ENCODING UTF-8)
    list(LENGTH matrix_rows row_count)
    if(NOT row_count EQUAL 27)
        message(FATAL_ERROR
            "F14 ${route} matrix rows=${row_count}, expected=27")
    endif()
    set(row_index 0)
    foreach(row IN LISTS matrix_rows)
        string(REGEX MATCHALL "," commas "${row}")
        list(LENGTH commas comma_count)
        if(NOT comma_count EQUAL 26)
            message(FATAL_ERROR
                "F14 ${route} row=${row_index} commas=${comma_count}, "
                "expected=26")
        endif()
        math(EXPR row_index "${row_index} + 1")
    endforeach()

    message(STATUS
        "F14_CSV_BYTES route=${route} bytes=${matrix_size} "
        "sha256=${matrix_sha} rows=${row_count} lf=${lf_count} cr=${cr_count} "
        "final_lf=1 blank_tail=0")
    set("${route}_sha" "${matrix_sha}" PARENT_SCOPE)
    set("${route}_rows" "${row_count}" PARENT_SCOPE)
    set("${route}_lf" "${lf_count}" PARENT_SCOPE)
    set("${route}_cr" "${cr_count}" PARENT_SCOPE)
endfunction()

function(require_matrix_pair lhs_name lhs_path lhs_sha rhs_name rhs_path rhs_sha)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E compare_files "${lhs_path}" "${rhs_path}"
        RESULT_VARIABLE compare_result)
    if(NOT "${compare_result}" STREQUAL "0" OR NOT lhs_sha STREQUAL rhs_sha)
        message(FATAL_ERROR
            "F14 matrix pair differs: ${lhs_name}=${lhs_sha} "
            "${rhs_name}=${rhs_sha}")
    endif()
    set_property(
        GLOBAL APPEND PROPERTY f14_matrix_pairs "${lhs_name}/${rhs_name}")
endfunction()

set(resident_dir "${work_root}/resident")
set(result_dir "${work_root}/result")
set(mmap_dir "${work_root}/mmap")
set(resident_matrix "${resident_dir}/conformance_distance_matrix.csv")
set(result_matrix "${result_dir}/conformance_distance_matrix.csv")
set(mmap_matrix "${mmap_dir}/conformance_distance_matrix.csv")

run_cli_route("resident" "${resident_dir}" 1000000 FALSE)
run_result_route("${result_dir}")
inspect_matrix("resident" "${resident_matrix}")
inspect_matrix("result" "${result_matrix}")
require_matrix_pair(
    "resident" "${resident_matrix}" "${resident_sha}"
    "result" "${result_matrix}" "${result_sha}")

if(has_mmap STREQUAL "ON")
    run_cli_route("mmap" "${mmap_dir}" 0 TRUE)
    inspect_matrix("mmap" "${mmap_matrix}")
    require_matrix_pair(
        "resident" "${resident_matrix}" "${resident_sha}"
        "mmap" "${mmap_matrix}" "${mmap_sha}")
    require_matrix_pair(
        "result" "${result_matrix}" "${result_sha}"
        "mmap" "${mmap_matrix}" "${mmap_sha}")
endif()

file(SHA256 "${input_path}" input_sha_after)
file(SHA256 "${config_path}" config_sha_after)
file(SIZE "${input_path}" input_size_after)
file(SIZE "${config_path}" config_size_after)
if(NOT input_sha_after STREQUAL input_sha_before
    OR NOT config_sha_after STREQUAL config_sha_before
    OR NOT input_size_after EQUAL input_size_before
    OR NOT config_size_after EQUAL config_size_before)
    message(FATAL_ERROR
        "F14 tracked conformance input/config changed during the gate")
endif()

get_property(f14_runs GLOBAL PROPERTY f14_runs)
get_property(f14_route_markers GLOBAL PROPERTY f14_route_markers)
get_property(f14_matrix_pairs GLOBAL PROPERTY f14_matrix_pairs)
list(LENGTH f14_runs run_count)
list(LENGTH f14_route_markers route_marker_count)
list(LENGTH f14_matrix_pairs matrix_pair_count)

if(has_mmap STREQUAL "ON")
    if(NOT run_count EQUAL 3
        OR NOT route_marker_count EQUAL 2
        OR NOT matrix_pair_count EQUAL 3)
        message(FATAL_ERROR
            "F14 mmap counter mismatch: runs=${run_count} "
            "routes=${route_marker_count} pairs=${matrix_pair_count}")
    endif()
    message(STATUS
        "F14_CSV_PUBLIC subject=real_dtwc_cl+native_result "
        "runs=${run_count}/3 route_markers=${route_marker_count}/2 "
        "matrix_pairs=${matrix_pair_count}/3 "
        "rows=${resident_rows}/${result_rows}/${mmap_rows} "
        "lf=${resident_lf}/${result_lf}/${mmap_lf} "
        "cr=${resident_cr}/${result_cr}/${mmap_cr} "
        "final_lf=3/3 blank_tail=0 mmap=ran skips=0")
else()
    if(NOT run_count EQUAL 2
        OR NOT route_marker_count EQUAL 1
        OR NOT matrix_pair_count EQUAL 1)
        message(FATAL_ERROR
            "F14 dense counter mismatch: runs=${run_count} "
            "routes=${route_marker_count} pairs=${matrix_pair_count}")
    endif()
    message(STATUS
        "F14_CSV_PUBLIC subject=real_dtwc_cl+native_result "
        "runs=${run_count}/2 route_markers=${route_marker_count}/1 "
        "matrix_pairs=${matrix_pair_count}/1 "
        "rows=${resident_rows}/${result_rows} "
        "lf=${resident_lf}/${result_lf} "
        "cr=${resident_cr}/${result_cr} "
        "final_lf=2/2 blank_tail=0 mmap=unavailable skips=0")
endif()
