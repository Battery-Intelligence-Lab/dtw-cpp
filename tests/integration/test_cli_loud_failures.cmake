cmake_minimum_required(VERSION 3.26)

# FX-3 (S-04, B-05, S-10): a failure must not exit 0. Drives the real dtwc_cl
# and asserts, per case, a non-zero exit AND an error that names what failed
# (the option and/or the file), so the user can act on it. Control runs on the
# same fixtures must succeed, so each failure is caused by its case alone.
#
#   checkpoint_save      --checkpoint names a regular file: rejected before the
#                        data is read, so no clustering work is lost
#   checkpoint_late      --checkpoint is a directory whose generations/ is a
#                        file, so only the save itself fails: the results must
#                        already be on disk
#   checkpoint_late_replay  the same, replaying the result with --resume
#   results_blocked_checkpoint_kept  a result write fails (a directory occupies
#                        <name>_labels.csv): the checkpoint, saved before the
#                        results, must be on disk
#   results_blocked_checkpoint_kept_replay  the same in a --resume replay
#   dist_matrix_bad      --dist-matrix names a non-square CSV
#   dist_matrix_missing  --dist-matrix names a file that does not exist
#   dist_matrix_n_mismatch  --dist-matrix is 4 x 4 for six series: it used to be
#                        discarded at the first lookup and recomputed, exit 0
#   dist_matrix_empty    --dist-matrix is an empty file: it loaded nothing, exit 0
#   dist_matrix_known_pairs  (must succeed) a complete --dist-matrix under a band
#                        no warping path fits: nothing is computed, so it runs
#                        and reports the file's cost
#   max_iter_zero        --max-iter 0 ran no iteration and reported the result
#   solver_gurobi        --solver gurobi on a build without Gurobi solved with
#                        HiGHS and exited 0 (a Gurobi build must honour it)
#   silhouettes_blocked  a directory occupies <name>_silhouettes.csv
#   labels_efbig         (POSIX) a file-size limit makes writes fail with EFBIG
#                        after <name>_labels.csv was opened: the stream must be
#                        checked after closing, not only after opening
#   clusters_zero        --clusters 0: the error names -k/--n-clusters
foreach(required_var IN ITEMS CLI WORK_ROOT)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        message(FATAL_ERROR "missing required -D${required_var}=...")
    endif()
endforeach()

cmake_path(ABSOLUTE_PATH CLI NORMALIZE OUTPUT_VARIABLE cli)
if(NOT EXISTS "${cli}")
    message(FATAL_ERROR "CLI does not exist: ${cli}")
endif()

file(REMOVE_RECURSE "${WORK_ROOT}")
file(MAKE_DIRECTORY "${WORK_ROOT}")

# Integer-valued series, one CSV per series, header row + index column.
function(write_series directory count name_prefix)
    file(MAKE_DIRECTORY "${directory}")
    math(EXPR last "${count} - 1")
    foreach(s RANGE ${last})
        set(text "t,value\n")
        foreach(t RANGE 7)
            math(EXPR value "(${t} * 7 + ${s} * 13) % 11")
            string(APPEND text "${t},${value}\n")
        endforeach()
        file(WRITE "${directory}/${name_prefix}${s}.csv" "${text}")
    endforeach()
endfunction()

write_series("${WORK_ROOT}/series" 6 "s")
set(common -i "${WORK_ROOT}/series" -k 2 --skip-rows 1 --skip-cols 1
    --max-iter 2 --n-init 1 --method pam --name loud)

set(cases 0)
set(all_output "")
macro(expect_loud_failure case_name)
    cmake_parse_arguments(LOUD "" "" "NAMES;COMMAND" ${ARGN})
    execute_process(
        COMMAND ${LOUD_COMMAND}
        RESULT_VARIABLE loud_result
        OUTPUT_VARIABLE loud_stdout
        ERROR_VARIABLE loud_stderr
        ENCODING UTF-8)
    string(APPEND all_output "${loud_stdout}\n${loud_stderr}\n")
    if("${loud_result}" STREQUAL "0")
        message(FATAL_ERROR
            "${case_name}: dtwc_cl exited 0 on a failure it must report\n"
            "stdout:\n${loud_stdout}\nstderr:\n${loud_stderr}")
    endif()
    if(NOT "${loud_result}" MATCHES "^[0-9]+$")
        message(FATAL_ERROR
            "${case_name}: dtwc_cl did not exit normally (result=${loud_result})\n"
            "stdout:\n${loud_stdout}\nstderr:\n${loud_stderr}")
    endif()
    foreach(needle IN LISTS LOUD_NAMES)
        string(FIND "${loud_stderr}" "${needle}" loud_at)
        if(loud_at EQUAL -1)
            message(FATAL_ERROR
                "${case_name}: exit=${loud_result} but the error does not name "
                "'${needle}'\nstderr:\n${loud_stderr}")
        endif()
    endforeach()
    math(EXPR cases "${cases} + 1")
    message(STATUS "${case_name}: exit=${loud_result} named=${LOUD_NAMES}")
endmacro()

# Control: the fixture and flags are valid.
execute_process(
    COMMAND "${cli}" ${common} -o "${WORK_ROOT}/out_control"
    RESULT_VARIABLE control_result
    OUTPUT_VARIABLE control_stdout
    ERROR_VARIABLE control_stderr
    ENCODING UTF-8)
if(NOT "${control_result}" STREQUAL "0"
        OR NOT EXISTS "${WORK_ROOT}/out_control/loud_silhouettes.csv")
    message(FATAL_ERROR
        "control run failed (exit=${control_result})\n"
        "stdout:\n${control_stdout}\nstderr:\n${control_stderr}")
endif()

# S-04: a checkpoint that cannot be saved. A path that is not a directory is
# refused before the data is read: nothing read, nothing clustered or written.
file(WRITE "${WORK_ROOT}/checkpoint_is_a_file" "not a directory\n")
expect_loud_failure(checkpoint_save
    NAMES "--checkpoint" "checkpoint_is_a_file"
    COMMAND "${cli}" ${common} -o "${WORK_ROOT}/out_checkpoint"
            --checkpoint "${WORK_ROOT}/checkpoint_is_a_file")
file(GLOB early_outputs "${WORK_ROOT}/out_checkpoint/*")
if(early_outputs OR "${loud_stdout}" MATCHES "time-series data are read|=== Results")
    message(FATAL_ERROR
        "checkpoint_save: the bad --checkpoint was found only after the run "
        "(outputs: ${early_outputs})\nstdout:\n${loud_stdout}")
endif()

# A save that fails only at the end (here generations/ is a file) must leave
# every result artefact on disk, in a fresh run and in a --resume replay.
set(result_files loud_labels.csv loud_medoids.csv loud_distance_matrix.csv
    loud_silhouettes.csv)
file(MAKE_DIRECTORY "${WORK_ROOT}/checkpoint_late")
file(WRITE "${WORK_ROOT}/checkpoint_late/generations" "not a directory\n")
expect_loud_failure(checkpoint_late
    NAMES "--checkpoint" "checkpoint_late"
    COMMAND "${cli}" ${common} -o "${WORK_ROOT}/out_late"
            --checkpoint "${WORK_ROOT}/checkpoint_late")
foreach(result_file IN LISTS result_files)
    if(NOT EXISTS "${WORK_ROOT}/out_late/${result_file}")
        message(FATAL_ERROR
            "checkpoint_late: ${result_file} was not written before the failing "
            "checkpoint save")
    endif()
endforeach()
file(REMOVE "${WORK_ROOT}/out_late/loud_labels.csv"
    "${WORK_ROOT}/out_late/loud_medoids.csv")
expect_loud_failure(checkpoint_late_replay
    NAMES "--checkpoint" "checkpoint_late"
    COMMAND "${cli}" ${common} -o "${WORK_ROOT}/out_late" --resume
            --checkpoint "${WORK_ROOT}/checkpoint_late")
if(NOT "${loud_stdout}" MATCHES "Replaying completed result checkpoint"
        OR NOT EXISTS "${WORK_ROOT}/out_late/loud_labels.csv"
        OR NOT EXISTS "${WORK_ROOT}/out_late/loud_medoids.csv")
    message(FATAL_ERROR
        "checkpoint_late_replay: the replayed labels/medoids were not written "
        "before the failing checkpoint save\nstdout:\n${loud_stdout}")
endif()

# The converse: a result write that fails (a directory occupies the labels
# file) must not lose the distance checkpoint, which is saved before the results.
file(MAKE_DIRECTORY "${WORK_ROOT}/out_blocked/loud_labels.csv")
expect_loud_failure(results_blocked_checkpoint_kept
    NAMES "loud_labels.csv"
    COMMAND "${cli}" ${common} -o "${WORK_ROOT}/out_blocked"
            --checkpoint "${WORK_ROOT}/checkpoint_kept")
file(GLOB kept_generation "${WORK_ROOT}/checkpoint_kept/generations/*/distances.csv")
if(NOT EXISTS "${WORK_ROOT}/checkpoint_kept/CURRENT" OR NOT kept_generation)
    message(FATAL_ERROR
        "results_blocked_checkpoint_kept: the failing result write lost the "
        "distance checkpoint\nstderr:\n${loud_stderr}")
endif()
# The same in a --resume replay of a completed run.
execute_process(
    COMMAND "${cli}" ${common} -o "${WORK_ROOT}/out_blocked_replay"
    RESULT_VARIABLE replay_seed_result
    OUTPUT_QUIET ERROR_VARIABLE replay_seed_stderr ENCODING UTF-8)
if(NOT "${replay_seed_result}" STREQUAL "0")
    message(FATAL_ERROR "replay seed run failed:\n${replay_seed_stderr}")
endif()
file(REMOVE "${WORK_ROOT}/out_blocked_replay/loud_labels.csv")
file(MAKE_DIRECTORY "${WORK_ROOT}/out_blocked_replay/loud_labels.csv")
expect_loud_failure(results_blocked_checkpoint_kept_replay
    NAMES "loud_labels.csv"
    COMMAND "${cli}" ${common} -o "${WORK_ROOT}/out_blocked_replay" --resume
            --checkpoint "${WORK_ROOT}/checkpoint_kept_replay")
file(GLOB kept_replay_generation
    "${WORK_ROOT}/checkpoint_kept_replay/generations/*/distances.csv")
if(NOT "${loud_stdout}" MATCHES "Replaying completed result checkpoint"
        OR NOT EXISTS "${WORK_ROOT}/checkpoint_kept_replay/CURRENT"
        OR NOT kept_replay_generation)
    message(FATAL_ERROR
        "results_blocked_checkpoint_kept_replay: the replay's failing result "
        "write lost the distance checkpoint\nstdout:\n${loud_stdout}")
endif()

# S-04: a precomputed matrix that cannot be loaded.
file(WRITE "${WORK_ROOT}/not_square.csv" "0,1\n1,0,5\n")
expect_loud_failure(dist_matrix_bad
    NAMES "--dist-matrix" "not_square.csv"
    COMMAND "${cli}" ${common} -o "${WORK_ROOT}/out_dm_bad"
            --dist-matrix "${WORK_ROOT}/not_square.csv")
expect_loud_failure(dist_matrix_missing
    NAMES "--dist-matrix" "no_such_matrix.csv"
    COMMAND "${cli}" ${common} -o "${WORK_ROOT}/out_dm_missing"
            --dist-matrix "${WORK_ROOT}/no_such_matrix.csv")

# FX-3: a matrix of another size describes other series, and an empty file
# holds none; both used to be accepted and every distance recomputed.
file(WRITE "${WORK_ROOT}/wrong_n.csv" "0,1,2,3\n1,0,1,2\n2,1,0,1\n3,2,1,0\n")
expect_loud_failure(dist_matrix_n_mismatch
    NAMES "--dist-matrix" "wrong_n.csv" "has 4 rows" "holds 6 series"
    COMMAND "${cli}" ${common} -o "${WORK_ROOT}/out_dm_n"
            --dist-matrix "${WORK_ROOT}/wrong_n.csv")
file(WRITE "${WORK_ROOT}/empty_matrix.csv" "")
expect_loud_failure(dist_matrix_empty
    NAMES "--dist-matrix" "empty_matrix.csv" "has 0 rows"
    COMMAND "${cli}" ${common} -o "${WORK_ROOT}/out_dm_empty"
            --dist-matrix "${WORK_ROOT}/empty_matrix.csv")

# The converse: a matrix that holds every pair computes nothing, so a band no
# warping path fits (lengths 4, 10, 5, 10 under --band 2) is no reason to fail.
# The file's distances give cost 1 + 2 = 3 for k = 2; DTW on these series would not.
file(MAKE_DIRECTORY "${WORK_ROOT}/series_known")
foreach(spec IN ITEMS "a:4" "b:10" "c:5" "d:10")
    string(REPLACE ":" ";" spec "${spec}")
    list(GET spec 0 known_name)
    list(GET spec 1 known_length)
    math(EXPR known_last "${known_length} - 1")
    set(text "t,value\n")
    foreach(t RANGE ${known_last})
        math(EXPR value "(${t} * 5 + ${known_length}) % 7")
        string(APPEND text "${t},${value}\n")
    endforeach()
    file(WRITE "${WORK_ROOT}/series_known/${known_name}.csv" "${text}")
endforeach()
file(WRITE "${WORK_ROOT}/known.csv" "0,1,9,9\n1,0,9,9\n9,9,0,2\n9,9,2,0\n")
execute_process(
    COMMAND "${cli}" -i "${WORK_ROOT}/series_known" -k 2 --skip-rows 1
            --skip-cols 1 --band 2 --max-iter 5 --n-init 1 --method pam
            --name known -o "${WORK_ROOT}/out_known"
            --dist-matrix "${WORK_ROOT}/known.csv"
    RESULT_VARIABLE known_result
    OUTPUT_VARIABLE known_stdout
    ERROR_VARIABLE known_stderr
    ENCODING UTF-8)
string(APPEND all_output "${known_stdout}\n${known_stderr}\n")
if(NOT "${known_result}" STREQUAL "0"
        OR NOT "${known_stdout}" MATCHES "Total cost: 3[\r\n]")
    message(FATAL_ERROR
        "dist_matrix_known_pairs: exit=${known_result}; a complete --dist-matrix "
        "must be used under an infeasible band, with the file's cost 3\n"
        "stdout:\n${known_stdout}\nstderr:\n${known_stderr}")
endif()
math(EXPR cases "${cases} + 1")
message(STATUS "dist_matrix_known_pairs: exit=0 cost=3 (from the file)")

# O-06: no iteration is no clustering. `common` already holds --max-iter.
expect_loud_failure(max_iter_zero
    NAMES "--max-iter"
    COMMAND "${cli}" -i "${WORK_ROOT}/series" -k 2 --skip-rows 1 --skip-cols 1
            --max-iter 0 --n-init 1 --method pam --name loud
            -o "${WORK_ROOT}/out_max_iter")

# FX-3: --solver gurobi must not solve with HiGHS. A build without Gurobi fails
# naming the flag and the fix; a build with it runs, which is accepted only when
# no fallback notice was printed.
execute_process(
    COMMAND "${cli}" ${common} -o "${WORK_ROOT}/out_solver" --solver gurobi
    RESULT_VARIABLE solver_result
    OUTPUT_VARIABLE solver_stdout
    ERROR_VARIABLE solver_stderr
    ENCODING UTF-8)
string(APPEND all_output "${solver_stdout}\n${solver_stderr}\n")
if("${solver_result}" STREQUAL "0")
    if("${solver_stdout}${solver_stderr}" MATCHES "Gurobi is not available")
        message(FATAL_ERROR
            "solver_gurobi: exited 0 after replacing Gurobi with HiGHS\n"
            "stdout:\n${solver_stdout}\nstderr:\n${solver_stderr}")
    endif()
    set(solver_case "gurobi_built")
else()
    if(NOT "${solver_result}" MATCHES "^[0-9]+$")
        message(FATAL_ERROR
            "solver_gurobi: dtwc_cl did not exit normally (result=${solver_result})\n"
            "stderr:\n${solver_stderr}")
    endif()
    foreach(needle IN ITEMS "--solver gurobi" "without Gurobi" "--solver highs")
        string(FIND "${solver_stderr}" "${needle}" solver_at)
        if(solver_at EQUAL -1)
            message(FATAL_ERROR
                "solver_gurobi: exit=${solver_result} but the error does not name "
                "'${needle}'\nstderr:\n${solver_stderr}")
        endif()
    endforeach()
    set(solver_case "loud")
endif()
math(EXPR cases "${cases} + 1")
message(STATUS "solver_gurobi: exit=${solver_result} (${solver_case})")

# B-05: an output artefact that cannot be written is an error, not a warning.
file(MAKE_DIRECTORY "${WORK_ROOT}/out_silhouettes/loud_silhouettes.csv")
expect_loud_failure(silhouettes_blocked
    NAMES "loud_silhouettes.csv"
    COMMAND "${cli}" ${common} -o "${WORK_ROOT}/out_silhouettes")

# B-05: a write that fails after the file was opened (full disk, quota). Under
# `ulimit -f 1` a write past one block (512 bytes under dash, 1024 under macOS
# /bin/sh) fails with EFBIG; SIGXFSZ is ignored so the write returns an error
# instead of killing the process. Every artefact written before the labels must
# fit in 512 bytes and the labels must exceed 1024, checked on a control run.
# The shell script uses newlines, not semicolons: a CMake list would split it.
set(efbig "unavailable")
if(NOT WIN32)
    string(REPEAT "x" 140 pad)
    write_series("${WORK_ROOT}/series_long" 10 "${pad}_")
    set(common_long -i "${WORK_ROOT}/series_long" -k 2 --skip-rows 1 --skip-cols 1
        --max-iter 2 --n-init 1 --method pam --name loud)
    execute_process(
        COMMAND "${cli}" ${common_long} -o "${WORK_ROOT}/out_long_control"
        RESULT_VARIABLE long_result
        OUTPUT_VARIABLE long_stdout
        ERROR_VARIABLE long_stderr
        ENCODING UTF-8)
    if(NOT "${long_result}" STREQUAL "0")
        message(FATAL_ERROR
            "long-name control run failed (exit=${long_result})\n"
            "stdout:\n${long_stdout}\nstderr:\n${long_stderr}")
    endif()
    file(SIZE "${WORK_ROOT}/out_long_control/loud_checkpoint.bin" before_labels)
    file(SIZE "${WORK_ROOT}/out_long_control/loud_labels.csv" labels_size)
    if(before_labels GREATER 512 OR NOT labels_size GREATER 1024)
        message(FATAL_ERROR
            "EFBIG fixture drifted: checkpoint.bin=${before_labels} bytes (must be "
            "<= 512), labels.csv=${labels_size} bytes (must be > 1024)")
    endif()
    file(MAKE_DIRECTORY "${WORK_ROOT}/out_efbig")
    expect_loud_failure(labels_efbig
        NAMES "loud_labels.csv"
        COMMAND /bin/sh -c "trap '' XFSZ\nulimit -f 1\nexec \"$0\" \"$@\""
                "${cli}" ${common_long} -o "${WORK_ROOT}/out_efbig")
    set(efbig "ran")
endif()

# S-10: the post-parse -k check names the canonical flag, not the deprecated one.
execute_process(
    COMMAND "${cli}" -i "${WORK_ROOT}/series" --clusters 0 --skip-rows 1
            --skip-cols 1 -o "${WORK_ROOT}/out_k" --name loud
    RESULT_VARIABLE k_result
    OUTPUT_VARIABLE k_stdout
    ERROR_VARIABLE k_stderr
    ENCODING UTF-8)
if("${k_result}" STREQUAL "0" OR NOT "${k_stderr}" MATCHES "Error: [^\n]*--n-clusters")
    message(FATAL_ERROR
        "clusters_zero: exit=${k_result}; the error line must name --n-clusters\n"
        "stderr:\n${k_stderr}")
endif()
math(EXPR cases "${cases} + 1")

string(REGEX MATCH "[Ss][Kk][Ii][Pp]([Pp]|[ :])" skip_match
    "${all_output}${k_stdout}${k_stderr}")
if(skip_match)
    message(FATAL_ERROR "dtwc_cl emitted skip text:\n${all_output}")
endif()

set(expected 14)
if(efbig STREQUAL "ran")
    set(expected 15)
endif()
message(STATUS
    "CLI_LOUD_FAILURES subject=real_dtwc_cl cases=${cases}/${expected} "
    "efbig=${efbig} skips=0")
