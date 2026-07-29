option (DTWC_ENABLE_COVERAGE "Enable coverage reporting for GCC or Clang" OFF)
# Setup macro for coverage testing for GCC or Clang
macro(add_executable_with_coverage_and_test TARGET_PATH)
    get_filename_component(TARGET_NAME ${TARGET_PATH} NAME_WE)
    add_executable(${TARGET_NAME} ${TARGET_PATH})
    target_link_libraries(${TARGET_NAME} PRIVATE dtwc++ Catch2::Catch2WithMain project_options)
    # Pass the project source directory to tests for finding test data
    target_compile_definitions(${TARGET_NAME} PRIVATE DTWC_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
    if(${ARGC} GREATER 1)
        get_target_property(
            _dtwc_test_emulator
            ${TARGET_NAME}
            CROSSCOMPILING_EMULATOR)
        if(_dtwc_test_emulator)
            add_test(
                NAME ${TARGET_NAME}
                COMMAND
                    ${ARGN}
                    ${_dtwc_test_emulator}
                    $<TARGET_FILE:${TARGET_NAME}>
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        else()
            add_test(
                NAME ${TARGET_NAME}
                COMMAND
                    ${ARGN}
                    $<TARGET_FILE:${TARGET_NAME}>
                WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
        endif()
    else()
        add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME} WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
    endif()
    set_tests_properties(${TARGET_NAME} PROPERTIES SKIP_RETURN_CODE 4)
    if (DTWC_ENABLE_COVERAGE)
        if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU" OR ${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
            message (STATUS "Configuring with coverage")
            target_compile_options(${TARGET_NAME} PUBLIC --coverage -O0)
            target_link_libraries(${TARGET_NAME} PUBLIC --coverage)
        else ()
            message (FATAL_ERROR "GCC or Clang required with DTWC_ENABLE_COVERAGE: found ${CMAKE_CXX_COMPILER_ID}")
        endif ()
    endif ()
endmacro()
