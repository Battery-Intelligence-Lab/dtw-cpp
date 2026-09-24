cmake_minimum_required(VERSION 3.26)

# Rebuilds the deprecated_shims_probe object (tests/CMakeLists.txt) with the
# configured compiler and passes only when the compile succeeds AND prints a
# deprecation diagnostic naming every entry of SHIMS. An up-to-date object
# compiles nothing and prints nothing, so the object is deleted first; a wrong
# OBJECT path therefore fails closed (no diagnostics), it cannot pass.
foreach(required_var IN ITEMS BUILD_DIR TARGET OBJECT FAMILY SHIMS)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        message(FATAL_ERROR "missing required -D${required_var}=...")
    endif()
endforeach()

file(REMOVE "${OBJECT}")
set(build_command "${CMAKE_COMMAND}" --build "${BUILD_DIR}" --target "${TARGET}")
if(NOT "${CONFIG}" STREQUAL "")
    list(APPEND build_command --config "${CONFIG}")
endif()
execute_process(
    COMMAND ${build_command}
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE output)

# Colour codes would split "'name' is deprecated" apart.
string(ASCII 27 escape)
string(REGEX REPLACE "${escape}\\[[0-9;]*[mK]" "" output "${output}")

string(REPLACE "," ";" shims "${SHIMS}")
list(LENGTH shims total)
set(warned 0)
set(silent "")
foreach(name IN LISTS shims)
    if(FAMILY STREQUAL "msvc")
        set(diagnostic "C4996: '[^'\n]*${name}[^'\n]*'")
    else()
        set(diagnostic "'[^'\n]*${name}[^'\n]*' is deprecated")
    endif()
    if(output MATCHES "${diagnostic}")
        math(EXPR warned "${warned} + 1")
    else()
        list(APPEND silent "${name}")
    endif()
endforeach()

if(NOT "${result}" STREQUAL "0" OR NOT warned EQUAL total)
    message("${output}")
    message(FATAL_ERROR
        "DEPRECATED_SHIMS family=${FAMILY} exit=${result} warned=${warned}/${total} "
        "silent=${silent}: every 1.x shim must compile and warn")
endif()
message(STATUS "DEPRECATED_SHIMS family=${FAMILY} exit=${result} warned=${warned}/${total}")
