cmake_minimum_required(VERSION 3.21)

# STABLE_ABI requires Python >= 3.12
if(Python_VERSION VERSION_GREATER_EQUAL "3.12")
  nanobind_add_module(_dtwcpp_core STABLE_ABI src/_dtwcpp_core.cpp)
else()
  nanobind_add_module(_dtwcpp_core src/_dtwcpp_core.cpp)
endif()

target_link_libraries(_dtwcpp_core PRIVATE dtwc++)

install(TARGETS _dtwcpp_core DESTINATION dtwcpp)
