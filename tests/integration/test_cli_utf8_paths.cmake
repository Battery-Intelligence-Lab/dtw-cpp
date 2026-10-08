cmake_minimum_required(VERSION 3.26)

# dtwc_cl must read the file it was named and write where it was told when the
# path has a character outside the Windows ANSI code page. On Windows argv used to
# arrive in that code page, so `δ.csv` became `d.csv` and the CLI quietly clustered
# a different file. The folder holds both files, with different series: 4 in δ.csv
# (two tight pairs), 3 flat ones in d.csv, so the series count and the pairing
# say which file was read. This script is UTF-8.
foreach(required_var IN ITEMS CLI WORK_ROOT)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        message(FATAL_ERROR "missing required -D${required_var}=...")
    endif()
endforeach()

# string(LENGTH) counts bytes: the Greek delta below is 2 only if this file kept its UTF-8 encoding.
set(delta "δ")
string(LENGTH "${delta}" delta_bytes)
if(NOT delta_bytes EQUAL 2)
    message(FATAL_ERROR "this script lost its UTF-8 encoding: δ is ${delta_bytes} byte(s), want 2")
endif()

file(REMOVE_RECURSE "${WORK_ROOT}")
file(MAKE_DIRECTORY "${WORK_ROOT}/in")
file(WRITE "${WORK_ROOT}/in/${delta}.csv" "0,1,2,3\n0,1,2,3.5\n5,6,7,8\n5,6,7,8.5\n")
file(WRITE "${WORK_ROOT}/in/d.csv" "1,1,1,1\n2,2,2,2\n3,3,3,3\n")
if(NOT EXISTS "${WORK_ROOT}/in/${delta}.csv")
    message(FATAL_ERROR "cannot create ${delta}.csv under ${WORK_ROOT}: the test cannot run here")
endif()

execute_process(
    COMMAND "${CLI}" -i "${WORK_ROOT}/in/${delta}.csv" -k 2 -o "${WORK_ROOT}/out_${delta}" -v
    RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr ENCODING UTF-8)
if(NOT "${result}" STREQUAL "0")
    message(FATAL_ERROR "dtwc_cl: exit=${result}\nstdout:\n${stdout}\nstderr:\n${stderr}")
endif()
if(NOT "${stdout}" MATCHES "Data loaded: 4 series")
    message(FATAL_ERROR "dtwc_cl did not read ${delta}.csv (4 series):\n${stdout}")
endif()

# The output directory and the default name (the input's stem) both keep the delta.
set(labels "${WORK_ROOT}/out_${delta}/${delta}_labels.csv")
if(NOT EXISTS "${labels}")
    file(GLOB_RECURSE written LIST_DIRECTORIES false "${WORK_ROOT}/out*/*")
    message(FATAL_ERROR "missing ${labels}; the CLI wrote:\n${written}")
endif()
file(STRINGS "${labels}" rows)
list(LENGTH rows row_count)
list(GET rows 1 row1)
list(GET rows 2 row2)
list(GET rows 3 row3)
list(GET rows 4 row4)
string(REGEX REPLACE "^[^,]*," "" cluster1 "${row1}")
string(REGEX REPLACE "^[^,]*," "" cluster2 "${row2}")
string(REGEX REPLACE "^[^,]*," "" cluster3 "${row3}")
string(REGEX REPLACE "^[^,]*," "" cluster4 "${row4}")
if(NOT row_count EQUAL 5 OR NOT cluster1 STREQUAL cluster2 OR NOT cluster3 STREQUAL cluster4
   OR cluster1 STREQUAL cluster3)
    message(FATAL_ERROR "labels of the 4 series {0,1,2,3.x} {5,6,7,8.x} are not two pairs:\n${rows}")
endif()
message(STATUS "CLI_UTF8_PATHS subject=real_dtwc_cl series=4/4 pairs=2/2 output_path=utf8 skips=0")
