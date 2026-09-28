file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp") # Get list of all test source files recursively

include("${CMAKE_SOURCE_DIR}/cmake/DtwcTest.cmake")

# B-19: the 1.x [[deprecated]] shims that docs/api-contract-2.0.md §3 keeps until
# 3.0 must still compile, and must warn (test_problem_api_2_0 proves they forward).
# One use of each shim kind stands in for all 29. The probe is an EXCLUDE_FROM_ALL
# object built by the configured compiler with dtwc++'s usage requirements; the
# script deletes the object, rebuilds it, and passes only on exit 0 plus one
# deprecation diagnostic naming each entry of _dtwc_shims. Flags differ per
# compiler family: GCC, Clang and clang-cl print "'<name>' is deprecated", cl
# prints C4996; -Wno-error keeps a -Werror in CMAKE_CXX_FLAGS from turning the
# warning under test into a failed compile. SERIAL: it builds in this tree.
set(_dtwc_shims set_numberOfClusters distByInd maxIter daviesBouldinIndex
    adjustedRandIndex startColumn fillDistanceMatrix)
add_library(deprecated_shims_probe OBJECT EXCLUDE_FROM_ALL integration/deprecated_shims_probe.cc)
target_link_libraries(deprecated_shims_probe PRIVATE dtwc++ project_options)
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_dtwc_shim_family msvc)
    target_compile_options(deprecated_shims_probe PRIVATE /w14996)
elseif(MSVC)
    set(_dtwc_shim_family gnu)
    target_compile_options(deprecated_shims_probe PRIVATE
        /clang:-Wdeprecated-declarations /clang:-Wno-error=deprecated-declarations)
else()
    set(_dtwc_shim_family gnu)
    target_compile_options(deprecated_shims_probe PRIVATE
        -Wdeprecated-declarations -Wno-error=deprecated-declarations)
endif()
list(LENGTH _dtwc_shims _dtwc_shim_count)
list(JOIN _dtwc_shims "," _dtwc_shim_csv)
dtwc_add_script_test(NAME test_deprecated_shims_warn
    SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_deprecated_shims_warn.cmake"
    ARGS "-DBUILD_DIR=${CMAKE_BINARY_DIR}" "-DCONFIG=$<CONFIG>"
         "-DTARGET=deprecated_shims_probe" "-DOBJECT=$<TARGET_OBJECTS:deprecated_shims_probe>"
         "-DFAMILY=${_dtwc_shim_family}" "-DSHIMS=${_dtwc_shim_csv}"
    MARKER "DEPRECATED_SHIMS family=${_dtwc_shim_family} exit=0 warned=${_dtwc_shim_count}/${_dtwc_shim_count}"
    LABELS compat b19 SERIAL TIMEOUT 300)

# Capability flags, queried once (they select markers and skip tolerance).
get_target_property(_dtwc_public_defs dtwc++ INTERFACE_COMPILE_DEFINITIONS)
if("DTWC_HAS_MMAP" IN_LIST _dtwc_public_defs)
    set(_dtwc_has_mmap ON)
    set(_dtwc_mmap_skip "")
else()
    set(_dtwc_has_mmap OFF)
    set(_dtwc_mmap_skip MAY_SKIP)      # SKIP("mmap support not compiled in ...")
endif()
if(DTWC_ENABLE_HIGHS AND TARGET highs::highs)
    set(_dtwc_highs_skip "")
else()
    set(_dtwc_highs_skip MAY_SKIP)     # SKIP("HiGHS is not compiled into this build.")
endif()
if("DTWC_SEQUENTIAL_BUILD" IN_LIST _dtwc_public_defs)
    set(_dtwc_seq_skip MAY_SKIP)       # SKIP("two OpenMP workers unavailable")
else()
    set(_dtwc_seq_skip "")
endif()

# Pass 1: name -> source for every Catch2 file under tests/.
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(_dtwc_test_name ${TEST_SOURCE} NAME_WE)
    set(_dtwc_test_source_${_dtwc_test_name} ${TEST_SOURCE})
endforeach()

# Explicit registrations: subject markers, fixture roots, capability gating.
# Everything not listed here is registered by pass 2 with the defaults (strict).

# F22: every retained C++ compatibility surface forwards to its canonical name
# (test_deprecated_shims_warn above proves the shims still warn).
dtwc_add_test(NAME test_problem_api_2_0 SOURCE ${_dtwc_test_source_test_problem_api_2_0}
    FIXTURE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/f22-cpp-compat" FIXTURE_DEFINE DTWC_F22_TEST_ROOT
    SERIAL)

# F51: binary-v1 wire-format gate, fixture writes confined to the build tree.
dtwc_add_test(NAME unit_test_checkpoint_binary SOURCE ${_dtwc_test_source_unit_test_checkpoint_binary}
    FIXTURE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/f51-binary-checkpoint" FIXTURE_DEFINE DTWC_F51_TEST_ROOT
    SERIAL)

# F13: nearest-medoid assignment through a multi-worker OpenMP region.
dtwc_add_test(NAME unit_test_nearest_medoid_assignment SOURCE ${_dtwc_test_source_unit_test_nearest_medoid_assignment}
    ENVIRONMENT "OMP_NUM_THREADS=4" PROCESSORS 4)

# F14 (focused): native distance-matrix CSV bytes; mmap is a required subject
# only in a build that publishes it.
if(_dtwc_has_mmap)
    set(_f14_marker "F14_CSV_CONTRACT dense=ran mmap=ran skips=0")
else()
    set(_f14_marker "F14_CSV_CONTRACT dense=ran mmap=unavailable skips=0")
endif()
dtwc_add_test(NAME unit_test_distance_matrix_csv SOURCE ${_dtwc_test_source_unit_test_distance_matrix_csv}
    FIXTURE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/f14-distance-matrix-csv-unit" FIXTURE_DEFINE DTWC_F14_TEST_ROOT
    MARKER "${_f14_marker}")

# F19: executed public-API gate; Lloyd's artifacts stay inside the build tree.
dtwc_add_test(NAME unit_test_problem_encapsulation SOURCE ${_dtwc_test_source_unit_test_problem_encapsulation}
    FIXTURE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/f19-problem-api" FIXTURE_DEFINE DTWC_F19_TEST_ROOT)

# F20: live Problem series-storage gate, per build flavour.
if(_dtwc_has_mmap)
    set(_f20_marker "F20_PROBLEM_STORAGE_POLICY build=llfio-on footprint=288 heap=owning mmap=view values=72/72 names=12/12 ndim_routes=2/2 ordered_pairs=72/72 artifact=pass lifetime=pass loader_auto=mmap view_override=pass subject_skips=0 verdict=PASS")
else()
    set(_f20_marker "F20_PROBLEM_STORAGE_POLICY build=llfio-off footprint=288 heap=owning mmap=rejected values=36/36 names=6/6 ndim_routes=1/1 ordered_pairs=36/36 transaction=pass loader_auto=heap-warning view_override=pass subject_skips=0 verdict=PASS")
endif()
dtwc_add_test(NAME unit_test_problem_storage_policy SOURCE ${_dtwc_test_source_unit_test_problem_storage_policy}
    FIXTURE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/f20-problem-storage" FIXTURE_DEFINE DTWC_F20_TEST_ROOT
    MARKER "${_f20_marker}" SERIAL)

# F15: the three portable deterministic generator schedules.
dtwc_add_test(NAME unit_test_deterministic_series SOURCE ${_dtwc_test_source_unit_test_deterministic_series}
    COMPILE_DEFINITIONS DTWC_F15_SOURCE_ROOT="${CMAKE_SOURCE_DIR}")

# F6: the Arrow/Parquet readers must RUN when the build publishes Arrow, and
# are not registered at all when it does not (no green stub).
dtwc_add_test(NAME test_io_readers SOURCE ${_dtwc_test_source_test_io_readers}
    REQUIRES DTWC_HAS_ARROW)

# Capability-dependent tests: skip tolerance only in the build that lacks the
# capability; strict everywhere else.
foreach(_dtwc_mmap_test IN ITEMS
        unit_test_mmap_data_store unit_test_mmap_distance_matrix
        unit_test_problem_semantic_transactions
        test_storage_policy unit_test_cli_args unit_test_dtw_function_semantics
        unit_test_variant_precision unit_test_variant_distmat)
    dtwc_add_test(NAME ${_dtwc_mmap_test} SOURCE ${_dtwc_test_source_${_dtwc_mmap_test}} ${_dtwc_mmap_skip})
endforeach()
foreach(_dtwc_highs_test IN ITEMS
        test_lagrangian_root test_mip_backend_guards test_pdlp_lp
        unit_test_benders unit_test_mip)
    dtwc_add_test(NAME ${_dtwc_highs_test} SOURCE ${_dtwc_test_source_${_dtwc_highs_test}} ${_dtwc_highs_skip})
endforeach()
dtwc_add_test(NAME unit_test_parallelisation SOURCE ${_dtwc_test_source_unit_test_parallelisation} ${_dtwc_seq_skip})

# Device tests: no CUDA/Metal device in any CI matrix, so each may skip; its
# SKIP message names the missing device.
foreach(_dtwc_device_test IN ITEMS
        test_cuda_correctness test_cuda_kernel_override test_cuda_launch_guards
        test_metal_correctness test_metal_mmap)
    dtwc_add_test(NAME ${_dtwc_device_test} SOURCE ${_dtwc_test_source_${_dtwc_device_test}} MAY_SKIP)
endforeach()

# The scalar-vs-ndim=1 wall-clock parity assertion must not share the machine.
dtwc_add_test(NAME test_multivariate_adversarial SOURCE ${_dtwc_test_source_test_multivariate_adversarial} SERIAL)

# Pass 2: everything else, strict. A name already handled above stays handled
# even when REQUIRES declined to register it.
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(_dtwc_test_name ${TEST_SOURCE} NAME_WE)
    if(NOT "${_dtwc_test_name}" IN_LIST DTWC_TEST_REGISTERED)
        dtwc_add_test(NAME ${_dtwc_test_name} SOURCE ${TEST_SOURCE})
    endif()
endforeach()

# The Arrow C Data ingest test synthesises ArrowSchema/ArrowArray structs with the
# vendored nanoarrow builder (the symbols are linked in via dtwc++). nanoarrow is
# a PRIVATE include of dtwc++, so grant this one test target the include path.
if(TARGET test_arrow_c_data)
    target_include_directories(test_arrow_c_data PRIVATE ${CMAKE_SOURCE_DIR}/dtwc/extern)
endif()

# Public F14 reachability: compare a real native Result::save with resident CLI
# output in every build, and with forced-mmap CLI output when llfio is linked.
# The .cc suffix deliberately keeps the helper out of the automatic *.cpp CTest
# glob above.
if(TARGET dtwc_cl)
    add_executable(
        f14_result_save_writer
        integration/f14_result_save_writer.cc)
    target_link_libraries(
        f14_result_save_writer
        PRIVATE
            dtwc++
            project_options)

    if(_dtwc_has_mmap)
        set(_f14_public_regex
            "F14_CSV_PUBLIC subject=real_dtwc_cl[+]native_result runs=3/3 route_markers=2/2 matrix_pairs=3/3 rows=27/27/27 lf=27/27/27 cr=0/0/0 final_lf=3/3 blank_tail=0 mmap=ran skips=0")
    else()
        set(_f14_public_regex
            "F14_CSV_PUBLIC subject=real_dtwc_cl[+]native_result runs=2/2 route_markers=1/1 matrix_pairs=1/1 rows=27/27 lf=27/27 cr=0/0 final_lf=2/2 blank_tail=0 mmap=unavailable skips=0")
    endif()

    dtwc_add_script_test(NAME test_distance_matrix_csv_contract
        SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_distance_matrix_csv_contract.cmake"
        ARGS "-DCLI=$<TARGET_FILE:dtwc_cl>" "-DRESULT_WRITER=$<TARGET_FILE:f14_result_save_writer>"
             "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/conformance/data/conformance_series.csv"
             "-DCONFIG=${CMAKE_CURRENT_SOURCE_DIR}/conformance/conformance.toml"
             "-DBINARY_ROOT=${CMAKE_CURRENT_BINARY_DIR}"
             "-DWORK_ROOT=${CMAKE_CURRENT_BINARY_DIR}/f14-distance-matrix-csv-public"
             "-DHAS_MMAP=${_dtwc_has_mmap}"
        MARKER "${_f14_public_regex}" LABELS integration f14 SERIAL TIMEOUT 120)
endif()

# F17 must drive the real CLI from a binary result written by the production
# serializer. The .cc suffix keeps the fixture writer out of the recursive
# Catch2 *.cpp test-target glob.
if(TARGET dtwc_cl)
    add_executable(
        f17_checkpoint_writer
        integration/f17_checkpoint_writer.cc)
    target_link_libraries(
        f17_checkpoint_writer
        PRIVATE
            dtwc++
            project_options)

    dtwc_add_script_test(NAME test_cli_resume_state
        SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_cli_resume_state.cmake"
        ARGS "-DCLI=$<TARGET_FILE:dtwc_cl>" "-DFIXTURE_WRITER=$<TARGET_FILE:f17_checkpoint_writer>"
             "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/conformance/data/conformance_series.csv"
             "-DCONFIG=${CMAKE_CURRENT_SOURCE_DIR}/conformance/conformance.toml"
             "-DBINARY_ROOT=${CMAKE_CURRENT_BINARY_DIR}" "-DWORK_ROOT=${CMAKE_CURRENT_BINARY_DIR}/f17-cli-resume"
        MARKER "F17_CLI_RESUME subject=real_dtwc_cl writer=production_serializer runs=12/12 replay_fields=10/10 markers=2/2 algorithm_skipped=1/1 fresh_discriminator=4/4 checkpoint_preserved=10/10 rejection_cases=9/9 sources_preserved=2/2 skips=0"
        LABELS integration f17 SERIAL TIMEOUT 120)
endif()

# The hand-rolled --yaml-config loader was removed (TOML via --config is the one
# CLI config mechanism). Pin the real binary rejecting the flag: a stale build or
# a re-added option would make this rejection disappear. The script asserts a
# non-zero exit AND that the flag is named back, instead of a
# PASS_REGULAR_EXPRESSION (which ignores the exit code and pinned CLI11 wording).
if(TARGET dtwc_cl)
    dtwc_add_script_test(NAME test_cli_rejects_yaml_config
        SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_cli_rejects_unknown_option.cmake"
        ARGS "-DCLI=$<TARGET_FILE:dtwc_cl>" "-DOPTION=--yaml-config"
        MARKER "CLI_REJECTS_UNKNOWN_OPTION subject=real_dtwc_cl option=--yaml-config exit=[0-9]+ nonzero=1 named=1 skips=0"
        LABELS integration cli TIMEOUT 60)
endif()

# --config takes TOML or YAML through CLI11's own config interface. Registered in
# BOTH build flavours: with YAML off the same script asserts the typed refusal,
# so the guard is exercised outside its own #ifdef. The expected check count is
# pinned per flavour so a silently short-circuited script cannot pass.
if(TARGET dtwc_cl)
    get_target_property(_dtwc_cl_defs dtwc_cl COMPILE_DEFINITIONS)
    if("DTWC_HAS_YAML" IN_LIST _dtwc_cl_defs)
        set(_dtwc_cl_yaml 1)
        set(_dtwc_cl_yaml_checks 23)
    else()
        set(_dtwc_cl_yaml 0)
        set(_dtwc_cl_yaml_checks 7)
    endif()

    dtwc_add_script_test(NAME test_cli_config_formats
        SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_cli_config_formats.cmake"
        ARGS "-DCLI=$<TARGET_FILE:dtwc_cl>"
             "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/conformance/data/conformance_series.csv"
             "-DTOML=${CMAKE_CURRENT_SOURCE_DIR}/conformance/conformance.toml"
             "-DHAS_YAML=${_dtwc_cl_yaml}" "-DWORK_ROOT=${CMAKE_CURRENT_BINARY_DIR}/cli-config-formats"
        MARKER "CLI_CONFIG_FORMATS subject=real_dtwc_cl yaml=${_dtwc_cl_yaml} checks=${_dtwc_cl_yaml_checks} skips=0"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}" LABELS integration cli SERIAL TIMEOUT 180)
endif()

# FX-3: a failed checkpoint save, --dist-matrix load or output write exits
# non-zero and names what failed. The EFBIG case needs a POSIX shell's ulimit,
# so the expected case count is pinned per platform.
if(TARGET dtwc_cl)
    if(WIN32)
        set(_dtwc_cl_loud_marker "cases=14/14 efbig=unavailable")
    else()
        set(_dtwc_cl_loud_marker "cases=15/15 efbig=ran")
    endif()
    dtwc_add_script_test(NAME test_cli_loud_failures
        SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_cli_loud_failures.cmake"
        ARGS "-DCLI=$<TARGET_FILE:dtwc_cl>" "-DWORK_ROOT=${CMAKE_CURRENT_BINARY_DIR}/cli-loud-failures"
        MARKER "CLI_LOUD_FAILURES subject=real_dtwc_cl ${_dtwc_cl_loud_marker} skips=0"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}" LABELS integration cli SERIAL TIMEOUT 180)
endif()

# IF-2 S3: dtwc::run's method x device table on the real binary, per build
# flavour, and --print-config against the golden files (49 keys: every value
# and every default). A GPU build on a machine without the device must refuse.
if(TARGET dtwc_cl)
    if("DTWC_HAS_CUDA" IN_LIST _dtwc_public_defs)
        set(_dtwc_cl_gpu cuda)
    elseif("DTWC_HAS_METAL" IN_LIST _dtwc_public_defs)
        set(_dtwc_cl_gpu metal)
    else()
        set(_dtwc_cl_gpu none)
    endif()
    if(DTWC_ENABLE_HIGHS AND TARGET highs::highs)
        set(_dtwc_cl_highs 1)
    else()
        set(_dtwc_cl_highs 0)
    endif()
    dtwc_add_script_test(NAME test_cli_device_matrix
        SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_cli_device_matrix.cmake"
        ARGS "-DCLI=$<TARGET_FILE:dtwc_cl>"
             "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/conformance/data/conformance_series.csv"
             "-DGOLDEN=${CMAKE_CURRENT_SOURCE_DIR}/conformance/config_all_fields.toml"
             "-DDEFAULTS=${CMAKE_CURRENT_SOURCE_DIR}/conformance/config_defaults.toml"
             "-DGPU=${_dtwc_cl_gpu}" "-DHAS_HIGHS=${_dtwc_cl_highs}"
             "-DWORK_ROOT=${CMAKE_CURRENT_BINARY_DIR}/cli-device-matrix"
        MARKER "CLI_DEVICE_MATRIX subject=real_dtwc_cl gpu=${_dtwc_cl_gpu} device=(present|absent|none) cells=[0-9]+/[0-9]+ print_config=2/2 keys=49 skips=0"
        LABELS integration cli SERIAL TIMEOUT 300)
endif()

# F8 is a real-CLI Parquet integration contract, not an Arrow-OFF capability
# probe. Add no CTest target at all unless the linked library actually publishes
# Parquet support; a passing stub or SKIP_RETURN_CODE would hide the subject.
if(TARGET dtwc_cl AND "DTWC_HAS_PARQUET" IN_LIST _dtwc_public_defs)
    add_executable(
        f13_poison_parquet_writer
        integration/f13_poison_parquet_writer.cc)
    target_link_libraries(
        f13_poison_parquet_writer
        PRIVATE
            Arrow::arrow_shared
            Parquet::parquet_shared
            project_options)

    dtwc_add_script_test(NAME test_fast_clara_parquet_parity
        SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_fast_clara_parquet_parity.cmake"
        ARGS "-DCLI=$<TARGET_FILE:dtwc_cl>" "-DFIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fast_clara_streaming_8x4.parquet"
             "-DBINARY_ROOT=${CMAKE_CURRENT_BINARY_DIR}" "-DWORK_ROOT=${CMAKE_CURRENT_BINARY_DIR}/f8-fast-clara-parity"
             "-DBYTE_ORDER=${CMAKE_CXX_BYTE_ORDER}"
        MARKER "F8_PARITY subject=real_dtwc_cl runs=6 route_markers=12/12 parity=9/9 configs=3/3_distinct"
        LABELS integration arrow f8 SERIAL TIMEOUT 120)

    dtwc_add_script_test(NAME test_fast_clara_assignment_contract
        SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_fast_clara_assignment_contract.cmake"
        ARGS "-DCLI=$<TARGET_FILE:dtwc_cl>" "-DFIXTURE_WRITER=$<TARGET_FILE:f13_poison_parquet_writer>"
             "-DFIXTURE=${CMAKE_CURRENT_SOURCE_DIR}/fixtures/fast_clara_streaming_8x4.parquet"
             "-DBINARY_ROOT=${CMAKE_CURRENT_BINARY_DIR}" "-DWORK_ROOT=${CMAKE_CURRENT_BINARY_DIR}/f13-fast-clara-assignment"
             "-DBYTE_ORDER=${CMAKE_CXX_BYTE_ORDER}"
        MARKER "F13_ASSIGNMENT_CONTRACT subject=real_dtwc_cl runs=4/4 route_markers=8/8 artifact_pairs=6/6 payloads=4/4 objective_bytes=4/4 stream_rejections=2/2 skips=0"
        LABELS integration arrow f13 SERIAL TIMEOUT 120)

    # Imported Arrow runtimes are not installed beside the test executable.
    # Make an ordinary Windows CTest invocation self-contained, including
    # PyArrow's companion DLL directory and clang's libomp runtime.
    if(WIN32)
        get_filename_component(_arrow_compiler_runtime_dir
            "${CMAKE_CXX_COMPILER}" DIRECTORY)
        set(_arrow_runtime_dirs "${_arrow_compiler_runtime_dir}")
        foreach(_arrow_target IN ITEMS
            Arrow::arrow_shared
            Parquet::parquet_shared)
            if(TARGET "${_arrow_target}")
                get_target_property(
                    _arrow_runtime
                    "${_arrow_target}"
                    IMPORTED_LOCATION)
                if(_arrow_runtime AND NOT _arrow_runtime MATCHES "-NOTFOUND$")
                    get_filename_component(_arrow_runtime_dir
                        "${_arrow_runtime}" DIRECTORY)
                    list(APPEND _arrow_runtime_dirs "${_arrow_runtime_dir}")
                    get_filename_component(_arrow_site_packages
                        "${_arrow_runtime_dir}" DIRECTORY)
                    if(IS_DIRECTORY "${_arrow_site_packages}/pyarrow.libs")
                        list(APPEND _arrow_runtime_dirs
                            "${_arrow_site_packages}/pyarrow.libs")
                    endif()
                endif()
            endif()
        endforeach()
        list(REMOVE_DUPLICATES _arrow_runtime_dirs)
        get_property(_arrow_runtime_tests DIRECTORY PROPERTY TESTS)
        foreach(_arrow_runtime_test IN LISTS _arrow_runtime_tests)
            foreach(_arrow_runtime_dir IN LISTS _arrow_runtime_dirs)
                set_property(
                    TEST "${_arrow_runtime_test}"
                    APPEND PROPERTY ENVIRONMENT_MODIFICATION
                    "PATH=path_list_prepend:${_arrow_runtime_dir}")
            endforeach()
        endforeach()
    endif()
endif()

# ── MATLAB .m suites ───────────────────────────────────────────────────────
# One CTest entry drives tests/matlab through the freshly built MEX and passes as
# a dtwc_add_test does (GT-1): at least one case passed, none failed and none was
# Incomplete (MATLAB's "filtered by assumption", green in its own summary). The
# registered skip is the parallelisation pair, which prints its reason: an OpenMP
# MEX filters the serial case, a sequential MEX the runtime case.

# find_program(matlab) is PATH-only, so a normal install (MATLAB not on PATH)
# made this test vanish with no diagnostic. find_package caches the same
# variables bindings/matlab uses, and MAIN_PROGRAM yields the executable itself.
if(DTWC_BUILD_MATLAB)
    find_package(Matlab COMPONENTS MX_LIBRARY MAIN_PROGRAM)
    # This guard is what makes the $<TARGET_FILE_DIR:dtwc_mex> genex below legal:
    # bindings/matlab is added AFTER tests, but it creates dtwc_mex under exactly
    # this MX_LIBRARY condition, and a genex is resolved at generate time.
    if(Matlab_FOUND AND Matlab_MAIN_PROGRAM)
        add_test(
            NAME matlab_suite
            COMMAND "${Matlab_MAIN_PROGRAM}" -batch
                "addpath('${CMAKE_SOURCE_DIR}/bindings/matlab'); addpath('$<TARGET_FILE_DIR:dtwc_mex>'); r = runtests('${CMAKE_CURRENT_SOURCE_DIR}/matlab'); allowed = {'test_test_api/test_parallelisation_runtime_engages', 'test_test_api/test_parallelisation_serial_is_honest'}; bad = setdiff({r([r.Incomplete]).Name}, allowed); fprintf('matlab_suite: %d run, %d passed, %d failed, %d incomplete%s', numel(r), sum([r.Passed]), sum([r.Failed]), sum([r.Incomplete]), newline); assert(sum([r.Passed]) >= 1 && sum([r.Failed]) == 0 && isempty(bad), 'matlab_suite gate failed');"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
        set_tests_properties(matlab_suite PROPERTIES
            LABELS "matlab" RUN_SERIAL TRUE TIMEOUT 1800)
    else()
        message(WARNING
            "DTWC_BUILD_MATLAB=ON but MATLAB not found: matlab_suite not registered")
    endif()
endif()
