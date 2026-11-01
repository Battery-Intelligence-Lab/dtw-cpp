# @author Volkan Kumtepeli
#
# The real dtwc_cl on Arrow IPC files streamed a record batch at a time under --ram-limit: FastCLARA's labels,
# medoids and total cost are the in-RAM run's and the CSV twin's, and the route is the one the build prices. Fixtures:
# tests/data/reader/make_ipc_fixtures.py (256 series in eight record batches; uncompressed, LZ4, ZSTD, Float32 LZ4).
# A --ram-limit of 24000 bytes holds the stream (the names and labels after it, about 19,000 bytes, the largest need)
# and not the series in RAM (from 31,000 bytes), but for an uncompressed file mapped with llfio, whose values are the
# page cache's: it is read into RAM (about 20,000 bytes).
cmake_minimum_required(VERSION 3.26)

foreach(required_var IN ITEMS CLI DATA HAS_MMAP WORK_ROOT)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        message(FATAL_ERROR "CLI_IPC_STREAM missing required -D${required_var}=...")
    endif()
endforeach()
if(NOT EXISTS "${CLI}")
    message(FATAL_ERROR "CLI_IPC_STREAM real CLI does not exist: ${CLI}")
endif()
file(REMOVE_RECURSE "${WORK_ROOT}")
file(MAKE_DIRECTORY "${WORK_ROOT}")

set(runs 0)
set(streamed 0)
set(parity 0)

# run_cli(<id> <input> <ram-limit> <dtype>): dtwc_cl's FastCLARA; sets <id>_route, <id>_labels (the cluster column),
# <id>_medoids (the index column) and <id>_cost.
function(run_cli id input limit dtype)
    set(out "${WORK_ROOT}/${id}")
    execute_process(
        COMMAND "${CLI}" --input "${DATA}/${input}" --output "${out}" --name t --method clara --n-clusters 3
                --sample-size 8 --n-samples 2 --seed 42 --device cpu --dtype ${dtype} --ram-limit ${limit} --verbose
        RESULT_VARIABLE result OUTPUT_VARIABLE stdout ERROR_VARIABLE stderr ENCODING UTF-8)
    if(NOT result EQUAL 0)
        message(FATAL_ERROR "CLI_IPC_STREAM ${id} exit=${result}\nstdout:\n${stdout}\nstderr:\n${stderr}")
    endif()
    string(FIND "${stdout}" "Arrow IPC walk selected streaming: 256 series" stream_at)
    string(FIND "${stdout}" "Data loaded" loaded_at)
    if(NOT stream_at EQUAL -1 AND loaded_at EQUAL -1)
        set(route stream)
    elseif(stream_at EQUAL -1 AND NOT loaded_at EQUAL -1)
        set(route ram)
    else()
        message(FATAL_ERROR "CLI_IPC_STREAM ${id}: no single route in\n${stdout}")
    endif()
    if(NOT stdout MATCHES "Total cost: ([^\r\n]+)")
        message(FATAL_ERROR "CLI_IPC_STREAM ${id}: no 'Total cost:' in\n${stdout}")
    endif()
    set(cost "${CMAKE_MATCH_1}")
    foreach(table IN ITEMS labels medoids)
        file(STRINGS "${out}/t_${table}.csv" lines)
        list(POP_FRONT lines) # the header
        set(column "")
        foreach(line IN LISTS lines)
            string(REPLACE "," ";" fields "${line}")
            list(GET fields 1 field) # labels: name,cluster; medoids: cluster,medoid_index,medoid_name
            list(APPEND column "${field}")
        endforeach()
        list(LENGTH column count)
        if(count EQUAL 0)
            message(FATAL_ERROR "CLI_IPC_STREAM ${id}: ${table} empty")
        endif()
        set(${id}_${table} "${column}" PARENT_SCOPE)
    endforeach()
    set(${id}_route ${route} PARENT_SCOPE)
    set(${id}_cost "${cost}" PARENT_SCOPE)
    math(EXPR runs "${runs} + 1")
    set(runs ${runs} PARENT_SCOPE)
endfunction()

# same(<a> <b>): the labels, the medoids and the cost of runs a and b equal.
function(same a b)
    foreach(what IN ITEMS labels medoids cost)
        if(NOT "${${a}_${what}}" STREQUAL "${${b}_${what}}")
            message(FATAL_ERROR "CLI_IPC_STREAM ${a} and ${b}: ${what} differ:\n${${a}_${what}}\n${${b}_${what}}")
        endif()
        math(EXPR parity "${parity} + 1")
    endforeach()
    set(parity ${parity} PARENT_SCOPE)
endfunction()

run_cli(csv clara_256.csv 0 float64)
foreach(config IN ITEMS plain:clara_256.arrow:float64 lz4:clara_256_lz4.arrow:float64
                        zstd:clara_256_zstd.arrow:float64 f32:clara_256_f32.arrow:float64
                        lz4_as_f32:clara_256_lz4.arrow:float32)
    string(REPLACE ":" ";" parts "${config}")
    list(GET parts 0 id)
    list(GET parts 1 input)
    list(GET parts 2 dtype)
    run_cli(${id}_ram ${input} 0 ${dtype})
    run_cli(${id}_limited ${input} 24000 ${dtype})
    if(NOT ${id}_ram_route STREQUAL "ram")
        message(FATAL_ERROR "CLI_IPC_STREAM ${id}: without --ram-limit the route is ${${id}_ram_route}")
    endif()
    # A mapped uncompressed file read as stored holds only its views: it stays in RAM. Every other one streams.
    if(id STREQUAL "plain" AND HAS_MMAP)
        set(expected ram)
    else()
        set(expected stream)
    endif()
    if(NOT ${id}_limited_route STREQUAL expected)
        message(FATAL_ERROR "CLI_IPC_STREAM ${id}: at --ram-limit 24000 the route is ${${id}_limited_route}, "
                            "expected ${expected}")
    endif()
    if(expected STREQUAL "stream")
        math(EXPR streamed "${streamed} + 1")
    endif()
    same(${id}_limited ${id}_ram)
    same(${id}_ram csv)
endforeach()

if(HAS_MMAP)
    set(mmap 1)
else()
    set(mmap 0)
endif()
message(STATUS "CLI_IPC_STREAM subject=real_dtwc_cl runs=${runs} streamed=${streamed} parity=${parity}/30 "
               "mmap=${mmap} skips=0")
