cmake_minimum_required(VERSION 3.21)

add_executable(example_Kmedoids_single Kmedoids_single.cpp)
target_link_libraries(example_Kmedoids_single PRIVATE dtwc++)

add_executable(example_MIP_single MIP_single.cpp)
target_link_libraries(example_MIP_single PRIVATE dtwc++)

add_executable(example_MIP_multiple MIP_multiple.cpp)
target_link_libraries(example_MIP_multiple PRIVATE dtwc++)