file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp") # Get list of all test source files recursively

# Iterate over each test source file and add an executable for each
foreach(TEST_SOURCE ${TEST_SOURCES})
    add_executable_with_coverage_and_test(${TEST_SOURCE})    # Add executable with coverage
endforeach()

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
