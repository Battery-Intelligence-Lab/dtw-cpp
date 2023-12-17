cmake_minimum_required(VERSION 3.21)

add_executable(example_Kmeans_single Kmeans_single.cpp)
target_link_libraries(example_Kmeans_single PRIVATE dtwc++)

add_executable(example_MIP_single MIP_single.cpp)
target_link_libraries(example_MIP_single PRIVATE dtwc++)

add_executable(example_MIP_multiple MIP_multiple.cpp)
target_link_libraries(example_MIP_multiple PRIVATE dtwc++)