cmake_minimum_required(VERSION 3.21)

add_executable(benchmark_UCR UCR_dtwc.cpp)
target_link_libraries(benchmark_UCR PRIVATE dtwc++)