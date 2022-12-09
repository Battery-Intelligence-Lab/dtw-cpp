# File reading unit tests:
add_executable(unit_test_file_io unit_test_file_io.cpp)
target_link_libraries(unit_test_file_io PRIVATE src)
add_test(NAME unit_test_file_io COMMAND unit_test_file_io WORKING_DIRECTORY bin)