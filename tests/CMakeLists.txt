file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp") # Get list of all test source files recursively

# Iterate over each test source file and add an executable for each
foreach(TEST_SOURCE ${TEST_SOURCES})
    add_executable_with_coverage_and_test(${TEST_SOURCE})    # Add executable with coverage
endforeach()

# F21 is a public-header compile and state-equivalence gate, never a capability
# probe. It uses the existing DataLoader target so CTest and tracked-CMake
# inventories stay fixed. Require both the exact route marker and Catch2's
# independent floor; clear the repository-wide skip return code.
if(TARGET unit_test_DataLoader)
    set_property(
        TEST unit_test_DataLoader
        PROPERTY SKIP_RETURN_CODE)
    set_tests_properties(
        unit_test_DataLoader
        PROPERTIES
            FAIL_REGULAR_EXPRESSION "[Ss][Kk][Ii][Pp]([Pp]|[ :])"
            PASS_REGULAR_EXPRESSION
                "F21_CPP_NAMES canonical=4/4 legacy=4/4 overloads=12/12 loader_state=22/22 path_state=16/16 cstring_copy=4/4 skips=0 verdict=PASS(.|[\r\n])*All tests passed \\((79|[89][0-9]|[1-9][0-9][0-9]+) assertions in ([2-9]|[1-9][0-9]+) test cases\\)")
endif()

# F13 is an executed-contract gate, not a capability probe. Require Catch2's
# own summary to prove the registered assertion/case floors and force the
# 65-point resident failure fixture through a multi-worker OpenMP region.
if(TEST unit_test_nearest_medoid_assignment)
    set_property(
        TEST unit_test_nearest_medoid_assignment
        PROPERTY SKIP_RETURN_CODE)
    set_tests_properties(
        unit_test_nearest_medoid_assignment
        PROPERTIES
            ENVIRONMENT "OMP_NUM_THREADS=4"
            FAIL_REGULAR_EXPRESSION "[Ss][Kk][Ii][Pp]"
            PASS_REGULAR_EXPRESSION
                "All tests passed \\(([5-9][0-9]|[0-9][0-9][0-9][0-9]*) assertions in ([89]|[1-9][0-9][0-9]*) test cases\\)"
            PROCESSORS 4)
endif()

# F14 freezes native distance-matrix CSV bytes before R4 may consolidate the
# formatter loops. The focused target always executes dense coverage; mmap is
# an additional required subject only in a build that publishes that
# capability. One regex pins both the route marker and Catch2's own floor,
# while allowing Catch2 to randomize which passing case prints the marker.
if(TARGET unit_test_distance_matrix_csv)
    target_compile_definitions(
        unit_test_distance_matrix_csv
        PRIVATE
            DTWC_F14_TEST_ROOT="${CMAKE_CURRENT_BINARY_DIR}/f14-distance-matrix-csv-unit")
    set_property(
        TEST unit_test_distance_matrix_csv
        PROPERTY SKIP_RETURN_CODE)
    get_target_property(
        _dtwc_f14_compile_defs
        dtwc++
        INTERFACE_COMPILE_DEFINITIONS)
    if("DTWC_HAS_MMAP" IN_LIST _dtwc_f14_compile_defs)
        set(_f14_focused_regex
            "F14_CSV_CONTRACT dense=ran mmap=ran skips=0(.|[\r\n])*All tests passed \\(([8-9][0-9]|[1-9][0-9][0-9]+) assertions in (1[2-9]|[2-9][0-9]|[1-9][0-9][0-9]+) test cases\\)")
    else()
        set(_f14_focused_regex
            "F14_CSV_CONTRACT dense=ran mmap=unavailable skips=0(.|[\r\n])*All tests passed \\(([5-9][0-9]|[1-9][0-9][0-9]+) assertions in ([89]|[1-9][0-9]+) test cases\\)")
    endif()
    set_tests_properties(
        unit_test_distance_matrix_csv
        PROPERTIES
            FAIL_REGULAR_EXPRESSION "[Ss][Kk][Ii][Pp]([Pp]|[ :])"
            PASS_REGULAR_EXPRESSION "${_f14_focused_regex}")
endif()

# F19 is an executed public-API gate, not a capability probe. The configured
# output root keeps Lloyd's required artifacts inside the build tree. Require
# both its exact subject marker and Catch2's independent assertion/case floor.
if(TARGET unit_test_problem_encapsulation)
    target_compile_definitions(
        unit_test_problem_encapsulation
        PRIVATE
            DTWC_F19_TEST_ROOT="${CMAKE_CURRENT_BINARY_DIR}/f19-problem-api")
    set_property(
        TEST unit_test_problem_encapsulation
        PROPERTY SKIP_RETURN_CODE)
    set_tests_properties(
        unit_test_problem_encapsulation
        PROPERTIES
            FAIL_REGULAR_EXPRESSION "[Ss][Kk][Ii][Pp]([Pp]|[ :])"
            PASS_REGULAR_EXPRESSION
                "F19_PROBLEM_API getters=10/10 setters=9/9 lloyd=ran skips=0(.|[\r\n])*All tests passed \\(([3-9][0-9]|[1-9][0-9][0-9]+) assertions in ([3-9]|[1-9][0-9]+) test cases\\)")
endif()

# F20 is a live Problem series-storage gate, not a DataLoader-only capability
# probe. Keep its direct temporary mmap artifact under the build tree, require
# the exact LLFIO-specific execution marker, and reject Catch2 skip semantics.
if(TARGET unit_test_problem_storage_policy)
    set(_f20_test_root
        "${CMAKE_CURRENT_BINARY_DIR}/f20-problem-storage")
    target_compile_definitions(
        unit_test_problem_storage_policy
        PRIVATE
            DTWC_F20_TEST_ROOT="${_f20_test_root}")
    set_property(
        TEST unit_test_problem_storage_policy
        PROPERTY SKIP_RETURN_CODE)
    get_target_property(
        _dtwc_f20_compile_defs
        dtwc++
        INTERFACE_COMPILE_DEFINITIONS)
    if("DTWC_HAS_MMAP" IN_LIST _dtwc_f20_compile_defs)
        set(_f20_focused_regex
            "F20_PROBLEM_STORAGE_POLICY build=llfio-on footprint=288 heap=owning mmap=view values=72/72 names=12/12 ndim_routes=2/2 ordered_pairs=72/72 artifact=pass lifetime=pass loader_auto=mmap view_override=pass subject_skips=0 verdict=PASS(.|[\r\n])*All tests passed \\((1[89][0-9]|[2-9][0-9][0-9]|[1-9][0-9][0-9][0-9]+) assertions in ([4-9]|[1-9][0-9]+) test cases\\)")
    else()
        set(_f20_focused_regex
            "F20_PROBLEM_STORAGE_POLICY build=llfio-off footprint=288 heap=owning mmap=rejected values=36/36 names=6/6 ndim_routes=1/1 ordered_pairs=36/36 transaction=pass loader_auto=heap-warning view_override=pass subject_skips=0 verdict=PASS(.|[\r\n])*All tests passed \\(([9][0-9]|[1-9][0-9][0-9]+) assertions in ([4-9]|[1-9][0-9]+) test cases\\)")
    endif()
    set_tests_properties(
        unit_test_problem_storage_policy
        PROPERTIES
            ENVIRONMENT
                "TMP=${_f20_test_root};TEMP=${_f20_test_root};TMPDIR=${_f20_test_root}"
            FAIL_REGULAR_EXPRESSION "[Ss][Kk][Ii][Pp]([Pp]|[ :])"
            PASS_REGULAR_EXPRESSION "${_f20_focused_regex}"
            RUN_SERIAL TRUE)
endif()

# F15 freezes the two shared deterministic generator schedules and the dense
# symmetric CPU-reference traversal. This target is mandatory in every build.
if(TARGET unit_test_deterministic_series)
    target_compile_definitions(
        unit_test_deterministic_series
        PRIVATE
            DTWC_F15_SOURCE_ROOT="${CMAKE_SOURCE_DIR}")
    set_property(
        TEST unit_test_deterministic_series
        PROPERTY SKIP_RETURN_CODE)
    set(_f15_focused_regex
        "F15_TEST_SUPPORT profile=(relaxed|precise|libstdcxx) scalar=ran row_seeded=ran continuous=ran dense=ran source_audit=ran skips=0(.|[\r\n])*All tests passed \\((3[5-9]|[4-9][0-9]|[1-9][0-9][0-9]+) assertions in ([5-9]|[1-9][0-9]+) test cases\\)")
    set_tests_properties(
        unit_test_deterministic_series
        PROPERTIES
            FAIL_REGULAR_EXPRESSION "[Ss][Kk][Ii][Pp]([Pp]|[ :])"
            PASS_REGULAR_EXPRESSION "${_f15_focused_regex}")
endif()

# F16 keeps the tracked presets truthful and host-specific. CMake's JSON
# implementation is the configure-time parser arbiter; the native test below
# independently pins the same fixed scalar metadata and prints the runtime
# execution marker required by CTest.
set(_f16_presets_file "${CMAKE_SOURCE_DIR}/CMakePresets.json")
file(READ "${_f16_presets_file}" _f16_presets_json)
set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${_f16_presets_file}"
        "${CMAKE_SOURCE_DIR}/pyproject.toml")

string(JSON _f16_schema GET "${_f16_presets_json}" version)
string(JSON _f16_floor_major GET
    "${_f16_presets_json}" cmakeMinimumRequired major)
string(JSON _f16_floor_minor GET
    "${_f16_presets_json}" cmakeMinimumRequired minor)
string(JSON _f16_floor_patch GET
    "${_f16_presets_json}" cmakeMinimumRequired patch)
set(_f16_preset_floor
    "${_f16_floor_major}.${_f16_floor_minor}.${_f16_floor_patch}")
file(STRINGS "${CMAKE_SOURCE_DIR}/CMakeLists.txt"
    _f16_root_first_line
    LIMIT_COUNT 1)
if(NOT _f16_root_first_line STREQUAL
        "cmake_minimum_required(VERSION 3.26)")
    message(FATAL_ERROR
        "F16 root first-command drift: ${_f16_root_first_line}")
endif()
set(_f16_root_floor "3.26.0")

if(NOT _f16_schema EQUAL 6)
    message(FATAL_ERROR
        "F16 preset schema drift: expected 6, found ${_f16_schema}")
endif()
if(NOT _f16_preset_floor VERSION_EQUAL "3.26.0"
        OR NOT _f16_root_floor VERSION_EQUAL "3.26.0"
        OR NOT _f16_preset_floor VERSION_EQUAL "${_f16_root_floor}")
    message(FATAL_ERROR
        "F16 CMake floor drift: preset=${_f16_preset_floor}, "
        "root=${_f16_root_floor}, expected=3.26.0")
endif()

function(_f16_find_preset out_index collection expected_name)
    string(JSON _count LENGTH "${_f16_presets_json}" "${collection}")
    math(EXPR _last "${_count} - 1")
    set(_found -1)
    foreach(_index RANGE 0 ${_last})
        string(JSON _name GET
            "${_f16_presets_json}" "${collection}" ${_index} name)
        if(_name STREQUAL expected_name)
            if(NOT _found EQUAL -1)
                message(FATAL_ERROR
                    "F16 duplicate ${collection} preset: ${expected_name}")
            endif()
            set(_found ${_index})
        endif()
    endforeach()
    if(_found EQUAL -1)
        message(FATAL_ERROR
            "F16 missing ${collection} preset: ${expected_name}")
    endif()
    set(${out_index} ${_found} PARENT_SCOPE)
endfunction()

function(_f16_require_condition collection index expected_host)
    string(JSON _type GET
        "${_f16_presets_json}" "${collection}" ${index} condition type)
    string(JSON _lhs GET
        "${_f16_presets_json}" "${collection}" ${index} condition lhs)
    string(JSON _rhs GET
        "${_f16_presets_json}" "${collection}" ${index} condition rhs)
    if(NOT _type STREQUAL "equals"
            OR NOT _lhs STREQUAL "\${hostSystemName}"
            OR NOT _rhs STREQUAL "${expected_host}")
        message(FATAL_ERROR
            "F16 ${collection}[${index}] condition drift: "
            "type=${_type}, lhs=${_lhs}, rhs=${_rhs}, "
            "expected=${expected_host}")
    endif()
endfunction()

string(JSON _f16_configure_count LENGTH
    "${_f16_presets_json}" configurePresets)
string(JSON _f16_build_count LENGTH
    "${_f16_presets_json}" buildPresets)
string(JSON _f16_test_count LENGTH
    "${_f16_presets_json}" testPresets)
if(NOT _f16_configure_count EQUAL 6
        OR NOT _f16_build_count EQUAL 3
        OR NOT _f16_test_count EQUAL 2)
    message(FATAL_ERROR
        "F16 preset inventory drift: "
        "configure=${_f16_configure_count}/6, "
        "build=${_f16_build_count}/3, test=${_f16_test_count}/2")
endif()

_f16_find_preset(_f16_default configurePresets default)
_f16_find_preset(_f16_clang_win configurePresets clang-win)
_f16_find_preset(_f16_clang_win_debug configurePresets clang-win-debug)
_f16_find_preset(_f16_msvc configurePresets msvc)
_f16_find_preset(_f16_gcc_linux configurePresets gcc-linux)
_f16_find_preset(_f16_clang_macos configurePresets clang-macos)

_f16_require_condition(configurePresets ${_f16_clang_win} Windows)
_f16_require_condition(configurePresets ${_f16_clang_win_debug} Windows)
_f16_require_condition(configurePresets ${_f16_msvc} Windows)
_f16_require_condition(configurePresets ${_f16_gcc_linux} Linux)
_f16_require_condition(configurePresets ${_f16_clang_macos} Darwin)

string(JSON _f16_default_binary GET
    "${_f16_presets_json}" configurePresets ${_f16_default} binaryDir)
string(JSON _f16_default_hidden GET
    "${_f16_presets_json}" configurePresets ${_f16_default} hidden)
string(JSON _f16_default_compiler ERROR_VARIABLE _f16_default_compiler_error
    GET "${_f16_presets_json}" configurePresets ${_f16_default}
    cacheVariables CMAKE_CXX_COMPILER)
string(JSON _f16_default_c_compiler
    ERROR_VARIABLE _f16_default_c_compiler_error
    GET "${_f16_presets_json}" configurePresets ${_f16_default}
    cacheVariables CMAKE_C_COMPILER)
string(JSON _f16_default_toolchain ERROR_VARIABLE _f16_default_toolchain_error
    GET "${_f16_presets_json}" configurePresets ${_f16_default}
    cacheVariables CMAKE_TOOLCHAIN_FILE)
if(NOT _f16_default_hidden
        OR NOT _f16_default_binary STREQUAL "\${sourceDir}/build"
        OR _f16_default_compiler_error STREQUAL "NOTFOUND")
    message(FATAL_ERROR
        "F16 hidden default must keep the shared build directory and no compiler")
endif()
if(_f16_default_c_compiler_error STREQUAL "NOTFOUND"
        OR _f16_default_toolchain_error STREQUAL "NOTFOUND")
    message(FATAL_ERROR
        "F16 hidden default must not encode a C compiler or toolchain")
endif()
string(REGEX MATCH "[A-Za-z]:/" _f16_windows_absolute_path
    "${_f16_presets_json}")
if(_f16_windows_absolute_path)
    message(FATAL_ERROR
        "F16 tracked presets must not encode a Windows drive path")
endif()

string(JSON _f16_win_compiler GET
    "${_f16_presets_json}" configurePresets ${_f16_clang_win}
    cacheVariables CMAKE_CXX_COMPILER)
if(NOT _f16_win_compiler STREQUAL "clang++")
    message(FATAL_ERROR
        "F16 clang-win must use clang++ from PATH on Windows only")
endif()

string(JSON _f16_win_parent GET
    "${_f16_presets_json}" configurePresets ${_f16_clang_win} inherits)
string(JSON _f16_debug_parent GET
    "${_f16_presets_json}" configurePresets ${_f16_clang_win_debug} inherits)
string(JSON _f16_msvc_parent GET
    "${_f16_presets_json}" configurePresets ${_f16_msvc} inherits)
string(JSON _f16_gcc_parent GET
    "${_f16_presets_json}" configurePresets ${_f16_gcc_linux} inherits)
string(JSON _f16_gcc_compiler GET
    "${_f16_presets_json}" configurePresets ${_f16_gcc_linux}
    cacheVariables CMAKE_CXX_COMPILER)
string(JSON _f16_mac_parent GET
    "${_f16_presets_json}" configurePresets ${_f16_clang_macos} inherits)
string(JSON _f16_mac_c GET
    "${_f16_presets_json}" configurePresets ${_f16_clang_macos}
    cacheVariables CMAKE_C_COMPILER)
string(JSON _f16_mac_cxx GET
    "${_f16_presets_json}" configurePresets ${_f16_clang_macos}
    cacheVariables CMAKE_CXX_COMPILER)
if(NOT _f16_win_parent STREQUAL "default"
        OR NOT _f16_debug_parent STREQUAL "clang-win"
        OR NOT _f16_msvc_parent STREQUAL "default"
        OR NOT _f16_gcc_parent STREQUAL "default"
        OR NOT _f16_gcc_compiler STREQUAL "g++"
        OR NOT _f16_mac_parent STREQUAL "default"
        OR NOT _f16_mac_c STREQUAL "/usr/bin/clang"
        OR NOT _f16_mac_cxx STREQUAL "/usr/bin/clang++")
    message(FATAL_ERROR
        "F16 platform preset inheritance/compiler/condition drift")
endif()

foreach(_f16_collection IN ITEMS buildPresets testPresets)
    _f16_find_preset(_f16_win_entry "${_f16_collection}" clang-win)
    _f16_find_preset(_f16_mac_entry "${_f16_collection}" clang-macos)
    string(JSON _f16_win_reference GET
        "${_f16_presets_json}" "${_f16_collection}" ${_f16_win_entry}
        configurePreset)
    string(JSON _f16_mac_reference GET
        "${_f16_presets_json}" "${_f16_collection}" ${_f16_mac_entry}
        configurePreset)
    _f16_require_condition(
        "${_f16_collection}" ${_f16_win_entry} Windows)
    _f16_require_condition(
        "${_f16_collection}" ${_f16_mac_entry} Darwin)
    if(NOT _f16_win_reference STREQUAL "clang-win"
            OR NOT _f16_mac_reference STREQUAL "clang-macos")
        message(FATAL_ERROR
            "F16 ${_f16_collection} reference/condition drift")
    endif()
endforeach()
_f16_find_preset(_f16_debug_build buildPresets clang-win-debug)
string(JSON _f16_debug_build_reference GET
    "${_f16_presets_json}" buildPresets ${_f16_debug_build} configurePreset)
_f16_require_condition(buildPresets ${_f16_debug_build} Windows)
if(NOT _f16_debug_build_reference STREQUAL "clang-win-debug")
    message(FATAL_ERROR "F16 clang-win-debug build preset drift")
endif()

file(READ "${CMAKE_SOURCE_DIR}/pyproject.toml" _f16_pyproject)
string(REGEX MATCHALL
    "(^|[\r\n])[ \t]*cmake[.]version[ \t]*=[ \t]*\\\">=3[.]26\\\""
    _f16_pyproject_floor_matches
    "${_f16_pyproject}")
list(LENGTH _f16_pyproject_floor_matches _f16_pyproject_floor_count)
if(NOT _f16_pyproject_floor_count EQUAL 1)
    message(FATAL_ERROR
        "F16 pyproject CMake floor drift: expected one >=3.26 scalar")
endif()

if(TARGET test_supply_chain_pinning)
    set_property(
        TEST test_supply_chain_pinning
        PROPERTY SKIP_RETURN_CODE)
    set(_f16_focused_regex
        "F16_CMAKE_PRESETS floor=3[.]26[.]0 compiler=clang[+][+] host_conditions=ran metadata_guard=ran skips=0(.|[\r\n])*All tests passed \\((3[5-9]|[4-9][0-9]|[1-9][0-9][0-9]+) assertions in ([4-9]|[1-9][0-9]+) test cases\\)")
    set_tests_properties(
        test_supply_chain_pinning
        PROPERTIES
            FAIL_REGULAR_EXPRESSION "[Ss][Kk][Ii][Pp]([Pp]|[ :])"
            PASS_REGULAR_EXPRESSION "${_f16_focused_regex}")
endif()

# The Arrow C Data ingest test synthesises ArrowSchema/ArrowArray structs with the
# vendored nanoarrow builder (the symbols are linked in via dtwc++). nanoarrow is
# a PRIVATE include of dtwc++, so grant this one test target the include path.
if(TARGET test_arrow_c_data)
    target_include_directories(test_arrow_c_data PRIVATE ${CMAKE_SOURCE_DIR}/dtwc/extern)
endif()

# This adversarial target contains a wall-clock parity assertion comparing the
# scalar and ndim=1 multivariate kernels.  Running it beside other compute-heavy
# tests can deschedule only one side of that comparison and create a false
# performance regression, so keep its measurement uncontended under CTest.
if(TEST test_multivariate_adversarial)
    set_tests_properties(test_multivariate_adversarial PROPERTIES RUN_SERIAL TRUE)
endif()

# Public F14 reachability: compare a real native Result::save with resident CLI
# output in every build, and with forced-mmap CLI output when llfio is linked.
# The .cc suffix deliberately keeps the helper out of the automatic *.cpp CTest
# glob above.
if(TARGET dtwc_cl)
    add_executable(
        f14_result_save_writer
        integration/f14_result_save_writer.cc)
    target_link_libraries(
        f14_result_save_writer
        PRIVATE
            dtwc++
            project_options)

    get_target_property(
        _dtwc_f14_public_compile_defs
        dtwc++
        INTERFACE_COMPILE_DEFINITIONS)
    if("DTWC_HAS_MMAP" IN_LIST _dtwc_f14_public_compile_defs)
        set(_f14_has_mmap ON)
        set(_f14_public_regex
            "F14_CSV_PUBLIC subject=real_dtwc_cl[+]native_result runs=3/3 route_markers=2/2 matrix_pairs=3/3 rows=27/27/27 lf=27/27/27 cr=0/0/0 final_lf=3/3 blank_tail=0 mmap=ran skips=0")
    else()
        set(_f14_has_mmap OFF)
        set(_f14_public_regex
            "F14_CSV_PUBLIC subject=real_dtwc_cl[+]native_result runs=2/2 route_markers=1/1 matrix_pairs=1/1 rows=27/27 lf=27/27 cr=0/0 final_lf=2/2 blank_tail=0 mmap=unavailable skips=0")
    endif()

    add_test(
        NAME test_distance_matrix_csv_contract
        COMMAND
            ${CMAKE_COMMAND}
            "-DCLI=$<TARGET_FILE:dtwc_cl>"
            "-DRESULT_WRITER=$<TARGET_FILE:f14_result_save_writer>"
            "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/conformance/data/conformance_series.csv"
            "-DCONFIG=${CMAKE_CURRENT_SOURCE_DIR}/conformance/conformance.toml"
            "-DBINARY_ROOT=${CMAKE_CURRENT_BINARY_DIR}"
            "-DWORK_ROOT=${CMAKE_CURRENT_BINARY_DIR}/f14-distance-matrix-csv-public"
            "-DHAS_MMAP=${_f14_has_mmap}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_distance_matrix_csv_contract.cmake"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    set_tests_properties(
        test_distance_matrix_csv_contract
        PROPERTIES
            FAIL_REGULAR_EXPRESSION "[Ss][Kk][Ii][Pp]([Pp]|[ :])"
            LABELS "integration;f14"
            PASS_REGULAR_EXPRESSION "${_f14_public_regex}"
            RUN_SERIAL TRUE
            TIMEOUT 120)
endif()

# F17 must drive the real CLI from a binary result written by the production
# serializer. The .cc suffix keeps the fixture writer out of the recursive
# Catch2 *.cpp test-target glob.
if(TARGET dtwc_cl)
    add_executable(
        f17_checkpoint_writer
        integration/f17_checkpoint_writer.cc)
    target_link_libraries(
        f17_checkpoint_writer
        PRIVATE
            dtwc++
            project_options)

    add_test(
        NAME test_cli_resume_state
        COMMAND
            ${CMAKE_COMMAND}
            "-DCLI=$<TARGET_FILE:dtwc_cl>"
            "-DFIXTURE_WRITER=$<TARGET_FILE:f17_checkpoint_writer>"
            "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/conformance/data/conformance_series.csv"
            "-DCONFIG=${CMAKE_CURRENT_SOURCE_DIR}/conformance/conformance.toml"
            "-DBINARY_ROOT=${CMAKE_CURRENT_BINARY_DIR}"
            "-DWORK_ROOT=${CMAKE_CURRENT_BINARY_DIR}/f17-cli-resume"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_cli_resume_state.cmake"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    set_tests_properties(
        test_cli_resume_state
        PROPERTIES
            FAIL_REGULAR_EXPRESSION
                "(^|[\r\n])[ \t]*[Ss][Kk][Ii][Pp]([Pp][Ee][Dd])?([ :]|$)"
            LABELS "integration;f17"
            PASS_REGULAR_EXPRESSION
                "F17_CLI_RESUME subject=real_dtwc_cl writer=production_serializer runs=12/12 replay_fields=10/10 markers=2/2 algorithm_skipped=1/1 fresh_discriminator=4/4 checkpoint_preserved=10/10 rejection_cases=9/9 sources_preserved=2/2 skips=0"
            RUN_SERIAL TRUE
            TIMEOUT 120)
endif()

# F8 is a real-CLI Parquet integration contract, not an Arrow-OFF capability
# probe. Add no CTest target at all unless the linked library actually publishes
# Parquet support; a passing stub or SKIP_RETURN_CODE would hide the subject.
get_target_property(_dtwc_test_compile_defs dtwc++ INTERFACE_COMPILE_DEFINITIONS)
if(TARGET dtwc_cl AND "DTWC_HAS_PARQUET" IN_LIST _dtwc_test_compile_defs)
    add_executable(
        f13_poison_parquet_writer
        integration/f13_poison_parquet_writer.cc)
    target_link_libraries(
        f13_poison_parquet_writer
        PRIVATE
            Arrow::arrow_shared
            Parquet::parquet_shared)

    add_test(
        NAME test_fast_clara_parquet_parity
        COMMAND
            ${CMAKE_COMMAND}
            "-DCLI=$<TARGET_FILE:dtwc_cl>"
            "-DFIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fast_clara_streaming_8x4.parquet"
            "-DBINARY_ROOT=${CMAKE_CURRENT_BINARY_DIR}"
            "-DWORK_ROOT=${CMAKE_CURRENT_BINARY_DIR}/f8-fast-clara-parity"
            "-DBYTE_ORDER=${CMAKE_CXX_BYTE_ORDER}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_fast_clara_parquet_parity.cmake"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    set_tests_properties(
        test_fast_clara_parquet_parity
        PROPERTIES
            LABELS "integration;arrow;f8"
            RUN_SERIAL TRUE
            TIMEOUT 120)

    add_test(
        NAME test_fast_clara_assignment_contract
        COMMAND
            ${CMAKE_COMMAND}
            "-DCLI=$<TARGET_FILE:dtwc_cl>"
            "-DFIXTURE_WRITER=$<TARGET_FILE:f13_poison_parquet_writer>"
            "-DFIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fast_clara_streaming_8x4.parquet"
            "-DBINARY_ROOT=${CMAKE_CURRENT_BINARY_DIR}"
            "-DWORK_ROOT=${CMAKE_CURRENT_BINARY_DIR}/f13-fast-clara-assignment"
            "-DBYTE_ORDER=${CMAKE_CXX_BYTE_ORDER}"
            -P "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_fast_clara_assignment_contract.cmake"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    set_tests_properties(
        test_fast_clara_assignment_contract
        PROPERTIES
            LABELS "integration;arrow;f13"
            RUN_SERIAL TRUE
            TIMEOUT 120)

    # Imported Arrow runtimes are not installed beside the test executable.
    # Make an ordinary Windows CTest invocation self-contained, including
    # PyArrow's companion DLL directory and clang's libomp runtime.
    if(WIN32)
        get_filename_component(_arrow_compiler_runtime_dir
            "${CMAKE_CXX_COMPILER}" DIRECTORY)
        set(_arrow_runtime_dirs "${_arrow_compiler_runtime_dir}")
        foreach(_arrow_target IN ITEMS
            Arrow::arrow_shared
            Parquet::parquet_shared)
            if(TARGET "${_arrow_target}")
                get_target_property(
                    _arrow_runtime
                    "${_arrow_target}"
                    IMPORTED_LOCATION)
                if(_arrow_runtime AND NOT _arrow_runtime MATCHES "-NOTFOUND$")
                    get_filename_component(_arrow_runtime_dir
                        "${_arrow_runtime}" DIRECTORY)
                    list(APPEND _arrow_runtime_dirs "${_arrow_runtime_dir}")
                    get_filename_component(_arrow_site_packages
                        "${_arrow_runtime_dir}" DIRECTORY)
                    if(IS_DIRECTORY "${_arrow_site_packages}/pyarrow.libs")
                        list(APPEND _arrow_runtime_dirs
                            "${_arrow_site_packages}/pyarrow.libs")
                    endif()
                endif()
            endif()
        endforeach()
        list(REMOVE_DUPLICATES _arrow_runtime_dirs)
        get_property(_arrow_runtime_tests DIRECTORY PROPERTY TESTS)
        foreach(_arrow_runtime_test IN LISTS _arrow_runtime_tests)
            foreach(_arrow_runtime_dir IN LISTS _arrow_runtime_dirs)
                set_property(
                    TEST "${_arrow_runtime_test}"
                    APPEND PROPERTY ENVIRONMENT_MODIFICATION
                    "PATH=path_list_prepend:${_arrow_runtime_dir}")
            endforeach()
        endforeach()
    endif()
endif()
