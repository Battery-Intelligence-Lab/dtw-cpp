cmake_minimum_required(VERSION 3.21)

nanobind_add_module(_dtwcpp_core
  STABLE_ABI
  src/_dtwcpp_core.cpp
)

target_link_libraries(_dtwcpp_core PRIVATE dtwc++)

install(TARGETS _dtwcpp_core DESTINATION dtwcpp)
