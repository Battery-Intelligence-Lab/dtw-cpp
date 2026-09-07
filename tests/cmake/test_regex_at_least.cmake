# cmake -P self-test of _dtwc_regex_at_least. Prints
# CMAKE_REGEX_AT_LEAST cases=<n> failures=<m>; non-zero exit on any failure.
cmake_minimum_required(VERSION 3.26)
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/DtwcRegex.cmake")

set(cases 0)
set(failures 0)
# "<n>;<value>;<expected 1=match 0=no match>"
foreach(row IN ITEMS
    "1;1;1" "1;0;0" "1;10;1"
    "9;8;0" "9;9;1" "9;10;1"
    "79;7;0" "79;78;0" "79;79;1" "79;80;1" "79;99;1" "79;100;1" "79;1000;1"
    "99;98;0" "99;99;1" "99;100;1"
    "100;99;0" "100;100;1" "100;101;1" "100;110;1" "100;999;1" "100;1000;1"
    "270;269;0" "270;270;1" "270;271;1" "270;300;1" "270;2700;1"
    "177;176;0" "177;177;1" "177;180;1" "177;199;1" "177;200;1")
  list(GET row 0 n)
  list(GET row 1 value)
  list(GET row 2 expected)
  _dtwc_regex_at_least(${n} rx)
  if("${value}" MATCHES "^${rx}$")
    set(got 1)
  else()
    set(got 0)
  endif()
  math(EXPR cases "${cases} + 1")
  if(NOT got EQUAL expected)
    math(EXPR failures "${failures} + 1")
    message(SEND_ERROR "n=${n} value=${value} expected=${expected} got=${got} regex=${rx}")
  endif()
endforeach()
message(STATUS "CMAKE_REGEX_AT_LEAST cases=${cases} failures=${failures}")
if(failures GREATER 0)
  message(FATAL_ERROR "regex generator self-test failed")
endif()
