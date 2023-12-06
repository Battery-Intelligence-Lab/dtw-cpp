option (DTWC_ENABLE_COVERAGE "Enable coverage reporting for GCC or Clang" OFF)
# Setup macro for coverage testing for GCC or Clang
macro(add_executable_with_coverage_and_test TARGET_NAME)
    add_executable(${TARGET_NAME} "${TARGET_NAME}.cpp")
    target_link_libraries(${TARGET_NAME} PRIVATE dtwc++ Catch2::Catch2WithMain)
    add_test(NAME ${TARGET_NAME} COMMAND ${TARGET_NAME})
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