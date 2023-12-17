file(GLOB_RECURSE TEST_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp") # Get list of all test source files recursively

# Iterate over each test source file and add an executable for each
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE) # Extract the base name without the extension and parent folders.
    add_executable_with_coverage_and_test(${TEST_SOURCE})    # Add executable with coverage
endforeach()