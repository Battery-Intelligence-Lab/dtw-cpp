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
