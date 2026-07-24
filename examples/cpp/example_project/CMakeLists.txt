cmake_minimum_required(VERSION 3.26)

project(MY_AWESOME_PROJECT VERSION "0.0.1"
        LANGUAGES CXX C)

set(CMAKE_CXX_STANDARD 20)

include(CPM.cmake)

CPMAddPackage(
    NAME dtw-cpp
    URL "https://github.com/Battery-Intelligence-Lab/dtw-cpp/archive/eda1b92bc89ee51568b052a6af86f615d336de3c.zip"
    URL_HASH SHA256=d9e991dc05804f5eedebdf3981eb400da229e485f4693ac3225ca17ae4a10696
)

add_executable(MY_AWESOME_PROJECT 
              main.cpp)

target_link_libraries(MY_AWESOME_PROJECT dtwc++)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
