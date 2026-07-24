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
        foreach(_arrow_cli_test IN ITEMS
            test_fast_clara_parquet_parity
            test_fast_clara_assignment_contract)
            foreach(_arrow_runtime_dir IN LISTS _arrow_runtime_dirs)
                set_property(
                    TEST "${_arrow_cli_test}"
                    APPEND PROPERTY ENVIRONMENT_MODIFICATION
                    "PATH=path_list_prepend:${_arrow_runtime_dir}")
            endforeach()
        endforeach()
    endif()
endif()
