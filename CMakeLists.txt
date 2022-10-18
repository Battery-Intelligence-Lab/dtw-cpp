cmake_minimum_required(VERSION 3.17)

include(cmake/StandardProjectSettings.cmake)
include(cmake/PreventInSourceBuilds.cmake)

project(DTWC++ VERSION "2.0.0"
  DESCRIPTION "A C++ library for fast Dynamic Time Wrapping Clustering"
  HOMEPAGE_URL https://battery-intelligence-lab.github.io/dtw-cpp/
  LANGUAGES CXX)

option(CXX "enable C++ compilation" ON)

if(CXX)
  enable_language(CXX)
endif()

# Link this 'library' to use the warnings specified in CompilerWarnings.cmake
add_library(project_warnings INTERFACE)

# Link this 'library' to set the c++ standard / compile-time options requested
add_library(project_options INTERFACE)
target_compile_features(project_options INTERFACE cxx_std_20)

target_compile_definitions(dtwc++ PRIVATE ROOT_FOLDER="${PROJECT_SOURCE_DIR}")

if(CMAKE_CXX_COMPILER_ID MATCHES ".*Clang")
  option(ENABLE_BUILD_WITH_TIME_TRACE "Enable -ftime-trace to generate time tracing .json files on clang" OFF)

  if(ENABLE_BUILD_WITH_TIME_TRACE)
    add_compile_definitions(project_options INTERFACE -ftime-trace)
  endif()
endif()

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# # enable cache system
# include(cmake/Cache.cmake)
# include(cmake/FindGUROBI.cmake)
find_package(GUROBI REQUIRED)
include_directories(${GUROBI_INCLUDE_DIRS})

# standard compiler warnings
include(cmake/CompilerWarnings.cmake)
set_project_warnings(project_warnings)

# # sanitizer options if supported by compiler
# include(cmake/Sanitizers.cmake)
# enable_sanitizers(project_options)

# enable doxygen
# include(cmake/Doxygen.cmake)
# enable_doxygen(src doxygen)

# # allow for static analysis options
# include(cmake/StaticAnalyzers.cmake)
message(STATUS "GUROBI CXX LIBRARY START")
message(STATUS ${GUROBI_CXX_LIBRARY})
message(STATUS "GUROBI CXX LIBRARY END")

message(STATUS "GUROBI CXX DEBUG LIBRARY START")
message(STATUS ${GUROBI_CXX_DEBUG_LIBRARY})
message(STATUS "GUROBI CXX DEBUG LIBRARY END")

message(STATUS "GUROBI LIBRARY START")
message(STATUS ${GUROBI_LIBRARY})
message(STATUS "GUROBI LIBRARY END")

file(GLOB SRC_FILES
  $ENV{GUROBI_HOME}/src/cpp/*.h
  $ENV{GUROBI_HOME}/src/cpp/*.cpp
)

foreach(header ${HEADERS_FILES})
  # here replace / for linux and \ for windows
  string(REPLACE "/" "_" header_path header)
  string(TOUPPER header_path final_header_path)
  file(APPEND headers_defines_file define
    PATH_${final_header_path} ${CMAKE_CURRENT_SOURCE_DIR}/header)
endforeach(header)

add_executable(dtwc++
  dtwc/main.cpp
  ${SRC_FILES}
)

target_link_libraries(dtwc++
  PRIVATE
  project_warnings
  project_options
  PUBLIC
  ${GUROBI_CXX_LIBRARY}
  ${GUROBI_LIBRARY}
)
