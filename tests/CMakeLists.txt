file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp") # Get list of all test source files recursively

# Iterate over each test source file and add an executable for each
foreach(TEST_SOURCE ${TEST_SOURCES})
    add_executable_with_coverage_and_test(${TEST_SOURCE})    # Add executable with coverage
endforeach()

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

    # Imported Arrow runtimes are not installed beside the test executable.
    # Make an ordinary Windows CTest invocation self-contained, including
    # PyArrow's companion DLL directory and clang's libomp runtime.
    if(WIN32)
        get_filename_component(_f8_compiler_runtime_dir
            "${CMAKE_CXX_COMPILER}" DIRECTORY)
        set(_f8_runtime_dirs "${_f8_compiler_runtime_dir}")
        foreach(_f8_arrow_target IN ITEMS
            Arrow::arrow_shared
            Parquet::parquet_shared)
            if(TARGET "${_f8_arrow_target}")
                get_target_property(
                    _f8_runtime
                    "${_f8_arrow_target}"
                    IMPORTED_LOCATION)
                if(_f8_runtime AND NOT _f8_runtime MATCHES "-NOTFOUND$")
                    get_filename_component(_f8_runtime_dir
                        "${_f8_runtime}" DIRECTORY)
                    list(APPEND _f8_runtime_dirs "${_f8_runtime_dir}")
                    get_filename_component(_f8_site_packages
                        "${_f8_runtime_dir}" DIRECTORY)
                    if(IS_DIRECTORY "${_f8_site_packages}/pyarrow.libs")
                        list(APPEND _f8_runtime_dirs
                            "${_f8_site_packages}/pyarrow.libs")
                    endif()
                endif()
            endif()
        endforeach()
        list(REMOVE_DUPLICATES _f8_runtime_dirs)
        foreach(_f8_runtime_dir IN LISTS _f8_runtime_dirs)
            set_property(
                TEST test_fast_clara_parquet_parity
                APPEND PROPERTY ENVIRONMENT_MODIFICATION
                "PATH=path_list_prepend:${_f8_runtime_dir}")
        endforeach()
    endif()
endif()
