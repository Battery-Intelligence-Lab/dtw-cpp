cmake_minimum_required(VERSION 3.21)

project(MY_AWESOME_PROJECT VERSION "0.0.1"
        LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 17)

include(CPM.cmake)

CPMAddPackage(
    NAME dtw-cpp
    URL "https://github.com/Battery-Intelligence-Lab/dtw-cpp/archive/refs/heads/documentation_update.zip"
)

add_executable(MY_AWESOME_PROJECT 
              main.cpp)

target_link_libraries(MY_AWESOME_PROJECT dtwc++)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})