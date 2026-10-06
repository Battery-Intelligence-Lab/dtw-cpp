cmake_minimum_required(VERSION 3.26)

add_executable(example_Kmedoids_single Kmedoids_single.cpp)
target_link_libraries(example_Kmedoids_single PRIVATE dtwc++)

add_executable(example_MIP_single MIP_single.cpp)
target_link_libraries(example_MIP_single PRIVATE dtwc++)

add_executable(example_MIP_multiple MIP_multiple.cpp)
target_link_libraries(example_MIP_multiple PRIVATE dtwc++)

add_executable(example_quickstart quickstart.cpp)
target_link_libraries(example_quickstart PRIVATE dtwc++)

add_executable(example_tier1 tier1.cpp)
target_link_libraries(example_tier1 PRIVATE dtwc++)

# ctest runs the Tier-1 example: its last line is printed only after it checked the clustering.
if(DTWC_BUILD_TESTING)
  add_test(NAME example_tier1 COMMAND example_tier1)
  set_tests_properties(example_tier1 PROPERTIES PASS_REGULAR_EXPRESSION "example_tier1: two groups recovered")
endif()
