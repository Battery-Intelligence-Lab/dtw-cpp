file(GLOB_RECURSE TEST_SOURCES CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.cpp") # Get list of all test source files recursively

# F22's existing CTest entry also drives a real-compiler diagnostic gate.
# This tests-only requirement does not add Python to core/test-off builds.
#
# A repo .venv is the project convention (`uv venv`), so prefer it — but only
# after checking that it runs and meets the floor. Setting Python3_EXECUTABLE is
# *exclusive*: FindPython3 then uses exactly that interpreter and does not search
# on when it turns out to be unusable. Verified by pointing it at a non-executable
# — find_package fails rather than falling back. So an abandoned .venv (a dangling
# interpreter after a Python upgrade) or one built on an older Python would fail
# the whole test configure on a machine with a perfectly good system interpreter,
# and the error would name FindPackageHandleStandardArgs rather than the .venv.
if(NOT DEFINED Python3_EXECUTABLE)
    if(WIN32)
        set(_dtwc_repo_python
            "${CMAKE_SOURCE_DIR}/.venv/Scripts/python.exe")
    else()
        set(_dtwc_repo_python
            "${CMAKE_SOURCE_DIR}/.venv/bin/python")
    endif()
    if(EXISTS "${_dtwc_repo_python}")
        execute_process(
            COMMAND "${_dtwc_repo_python}" -c
                    "import sys; sys.exit(0 if sys.version_info[:2] >= (3, 9) else 1)"
            RESULT_VARIABLE _dtwc_repo_python_usable
            OUTPUT_QUIET ERROR_QUIET)
        if(_dtwc_repo_python_usable EQUAL 0)
            # A normal variable is a FindPython3 hint, not a forced cache choice.
            set(Python3_EXECUTABLE "${_dtwc_repo_python}")
        else()
            message(STATUS
                "Tests: ignoring ${_dtwc_repo_python} (does not run, or is older "
                "than 3.9) and searching for a system interpreter instead")
        endif()
        unset(_dtwc_repo_python_usable)
    endif()
    unset(_dtwc_repo_python)
endif()

find_package(Python3 3.9 COMPONENTS Interpreter QUIET)
if(NOT Python3_Interpreter_FOUND)
    message(FATAL_ERROR
        "DTWC_BUILD_TESTING=ON requires a Python 3.9 or newer interpreter. It is "
        "used by exactly one gate — test_problem_api_2_0, which drives a real "
        "compiler through scripts/test_f22_cpp_deprecations.py — but that gate is "
        "not allowed to disappear quietly, so a missing interpreter stops the "
        "configure rather than silently dropping it. Create one with `uv venv` in "
        "the repo root, point at an existing one with "
        "-DPython3_EXECUTABLE=/path/to/python, or configure with "
        "-DDTWC_BUILD_TESTING=OFF.")
endif()
message(STATUS "Tests: Python ${Python3_VERSION} at ${Python3_EXECUTABLE}")

# Multi-config generators do not provide compile_commands.json. These
# EXCLUDE_FROM_ALL object probes reuse the configured target usage requirements
# without adding a CTest entry. Unique pragma markers prove each requested
# compile actually ran; build-local wrappers keep the tracked fixtures single
# source of truth.
set(_f22_probe_root "${CMAKE_CURRENT_BINARY_DIR}/f22-cpp-probes")
file(MAKE_DIRECTORY "${_f22_probe_root}")
set(_f22_legacy_fixture
    "${CMAKE_SOURCE_DIR}/tests/fixtures/f22_cpp_legacy_diagnostics.inc")
set(_f22_canonical_fixture
    "${CMAKE_SOURCE_DIR}/tests/fixtures/f22_cpp_canonical_diagnostics.inc")
set(_f22_legacy_werror_source
    "${_f22_probe_root}/f22_cpp_legacy_werror.cpp")
set(_f22_legacy_suppressed_source
    "${_f22_probe_root}/f22_cpp_legacy_suppressed.cpp")
set(_f22_canonical_werror_source
    "${_f22_probe_root}/f22_cpp_canonical_werror.cpp")
file(WRITE "${_f22_legacy_werror_source}"
    "#pragma message(\"F22_PROBE_LEGACY_WERROR_RAN\")\n"
    "#include \"${_f22_legacy_fixture}\"\n")
file(WRITE "${_f22_legacy_suppressed_source}"
    "#pragma message(\"F22_PROBE_LEGACY_SUPPRESSED_RAN\")\n"
    "#include \"${_f22_legacy_fixture}\"\n")
file(WRITE "${_f22_canonical_werror_source}"
    "#pragma message(\"F22_PROBE_CANONICAL_WERROR_RAN\")\n"
    "#include \"${_f22_canonical_fixture}\"\n")

add_library(
    f22_cpp_legacy_werror
    OBJECT EXCLUDE_FROM_ALL
    "${_f22_legacy_werror_source}")
add_library(
    f22_cpp_legacy_suppressed
    OBJECT EXCLUDE_FROM_ALL
    "${_f22_legacy_suppressed_source}")
add_library(
    f22_cpp_canonical_werror
    OBJECT EXCLUDE_FROM_ALL
    "${_f22_canonical_werror_source}")

if(MSVC)
    # MSVC's STL deprecates std::codecvt<char16_t, char8_t> (LWG-3767, STL4047)
    # and llfio's path_view.ipp instantiates it; without this define /we4996
    # dies on a third-party deprecation before a single dtwc diagnostic.
    set(_f22_stl_silence _SILENCE_ALL_CXX20_DEPRECATION_WARNINGS)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(_f22_deprecation_error "/clang:-Werror=deprecated-declarations")
        set(_f22_deprecation_suppression "/clang:-Wno-deprecated-declarations")
        set(_f22_pragma_nonerror "/clang:-Wno-error=#pragma-messages")
        set(_f22_diagnostic_limit "/clang:-ferror-limit=0")
    else()
        set(_f22_deprecation_error "/we4996")
        set(_f22_deprecation_suppression "/wd4996")
        set(_f22_pragma_nonerror "")
        set(_f22_diagnostic_limit "")
    endif()
else()
    set(_f22_stl_silence "")
    set(_f22_deprecation_error "-Werror=deprecated-declarations")
    set(_f22_deprecation_suppression "-Wno-deprecated-declarations")
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
        set(_f22_pragma_nonerror "-Wno-error=#pragma-messages")
        set(_f22_diagnostic_limit "-ferror-limit=0")
    else()
        set(_f22_pragma_nonerror "")
        set(_f22_diagnostic_limit "-fmax-errors=0")
    endif()
endif()
set(_f22_options_f22_cpp_legacy_werror
    ${_f22_deprecation_error} ${_f22_pragma_nonerror} ${_f22_diagnostic_limit})
set(_f22_options_f22_cpp_legacy_suppressed
    ${_f22_deprecation_suppression} ${_f22_pragma_nonerror})
set(_f22_options_f22_cpp_canonical_werror
    ${_f22_deprecation_error} ${_f22_pragma_nonerror})
foreach(_f22_probe_target IN ITEMS
        f22_cpp_legacy_werror f22_cpp_legacy_suppressed f22_cpp_canonical_werror)
    target_link_libraries(${_f22_probe_target} PRIVATE dtwc++ project_options)
    target_compile_features(${_f22_probe_target} PRIVATE cxx_std_20)
    target_compile_definitions(${_f22_probe_target} PRIVATE ${_f22_stl_silence})
    target_compile_options(${_f22_probe_target} PRIVATE ${_f22_options_${_f22_probe_target}})
endforeach()

include("${CMAKE_SOURCE_DIR}/cmake/DtwcTest.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/floors.cmake")

# The floor-regex generator gates every other entry in this file, so it gets a
# CTest entry of its own, registered through the same helper as the rest.
dtwc_add_script_test(NAME cmake_regex_at_least
    SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_regex_at_least.cmake"
    MARKER "CMAKE_REGEX_AT_LEAST cases=32 failures=0" LABELS tooling)

# X-16: the count of bare `throw std::` in dtwc/ may only fall. Pure Python over
# the source tree, so unlike the codegen report it runs on every toolchain and is
# a real gate rather than a report.
add_test(NAME check_typed_throws
         COMMAND "${Python3_EXECUTABLE}"
                 "${CMAKE_SOURCE_DIR}/scripts/check_typed_throws.py"
         WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
set_tests_properties(check_typed_throws PROPERTIES
    PASS_REGULAR_EXPRESSION
        "TYPED_THROW_RATCHET files=[1-9][0-9]* untyped=[0-9]+ ceiling=[0-9]+ verdict=PASS"
    LABELS "tooling;x16")

# A-06 / core-map Q10 evidence: does IPO inline Problem::dist_by_ind into the
# FastPAM SWAP kernels? Report-only — the marker is the evidence and W4 decides.
# Needs an ELF/Mach-O binary with symbols and an objdump, so it is not registered
# on Windows. DTWC_ENABLE_IPO is the project switch; CMAKE_INTERPROCEDURAL_OPTIMIZATION
# is what dtwc_enable_ipo() actually sets once check_ipo_supported() agrees, so
# the label follows the latter and cannot claim IPO that was silently unavailable.
if(NOT WIN32 AND TARGET dtwc_cl)
    if(CMAKE_INTERPROCEDURAL_OPTIMIZATION)
        set(_dtwc_ipo_state on)
    else()
        set(_dtwc_ipo_state off)
    endif()
    add_test(NAME check_ipo_inlining
             COMMAND "${Python3_EXECUTABLE}"
                     "${CMAKE_SOURCE_DIR}/scripts/check_ipo_inlining.py"
                     "$<TARGET_FILE:dtwc_cl>" --ipo ${_dtwc_ipo_state}
             WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    set_tests_properties(check_ipo_inlining PROPERTIES
        PASS_REGULAR_EXPRESSION
            "IPO_INLINING binary=dtwc_cl tool=(llvm-objdump|objdump) ipo=${_dtwc_ipo_state} callers=[1-9][0-9]* dist_by_ind_calls=[0-9]+ symbol_present=[01]"
        LABELS "tooling;a06")
    unset(_dtwc_ipo_state)
endif()

# Capability flags, queried once (they select markers, floors and skip tolerance).
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

# Explicit registrations: subject markers, author-chosen floors, fixture roots,
# capability gating. Everything not listed here is registered by pass 2 with
# the defaults (strict, floored from tests/floors.cmake).

# F22 executes every retained C++ compatibility surface in exact diagnostic
# order through the real-compiler probe launcher.
dtwc_add_test(NAME test_problem_api_2_0 SOURCE ${_dtwc_test_source_test_problem_api_2_0}
    LAUNCHER "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/test_f22_cpp_deprecations.py"
        --build-dir "${CMAKE_BINARY_DIR}" --source-dir "${CMAKE_SOURCE_DIR}"
        --cmake-command "${CMAKE_COMMAND}" --cmake-config "$<CONFIG>"
        --probe-root "${_f22_probe_root}" --launch
    FIXTURE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/f22-cpp-compat" FIXTURE_DEFINE DTWC_F22_TEST_ROOT
    MARKER "F22_CPP_DIAGNOSTICS inventory=33/33 legacy=33/33 canonical_silent=33/33 overloads=31/31 fields=2/2 skips=0 verdict=PASS(.|[\r\n])*F22_CPP_COMPAT inventory=33/33 behavior=33/33 field_routes=4/4 io_routes=7/7 file_identity=6/6 stdout_identity=2/2 skips=0 verdict=PASS"
    ASSERT_FLOOR 65 CASE_FLOOR 5 SERIAL)

# F51: binary-v1 wire-format gate, fixture writes confined to the build tree.
dtwc_add_test(NAME unit_test_checkpoint_binary SOURCE ${_dtwc_test_source_unit_test_checkpoint_binary}
    FIXTURE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/f51-binary-checkpoint" FIXTURE_DEFINE DTWC_F51_TEST_ROOT
    MARKER "F51_BINARY_CHECKPOINT corpus=85 rejected=85 throws=0 unchanged=85/85 size_preflight=1/1 valid_bytes=72/72 fields=5/5 resave=72/72 semantic_compat=7/7 skips=0 verdict=PASS"
    ASSERT_FLOOR 270 CASE_FLOOR 2 SERIAL)

# D2: exhaustive LB_Keogh proof; serial OpenMP for deterministic pair order.
dtwc_add_test(NAME test_lb_keogh_derivation SOURCE ${_dtwc_test_source_test_lb_keogh_derivation}
    ENVIRONMENT "OMP_NUM_THREADS=1"
    MARKER "D2_LB_KEOGH_GATE envelope_cases=2004 equal_cases=28602 unequal_cases=17712 call_sites=2/2 skips=0 verdict=PASS"
    ASSERT_FLOOR 65 CASE_FLOOR 1 SERIAL)

# D3: LB_Enhanced / LB_Webb proof plus two live pruning routes.
dtwc_add_test(NAME test_lb_enhanced_webb_derivation SOURCE ${_dtwc_test_source_test_lb_enhanced_webb_derivation}
    ENVIRONMENT "OMP_NUM_THREADS=1"
    MARKER "D3_LB_ENHANCED_WEBB_GATE envelope_cases=2004 path_cases=35982 full_cover_cases=7380 enhanced_cases=68787 enhanced_v5=4/4 webb_cases=35982 webb_branches=4/4 webb_strict=2/2 tail_cases=35982 tail_strict=2/2 metric_cases=140 order_witnesses=2/2 cascade_routes=2/2 skips=0 verdict=PASS"
    ASSERT_FLOOR 40 CASE_FLOOR 1 SERIAL TIMEOUT 60)

# F57: the valid INT_MAX CPU-window defect, isolated from D3.
dtwc_add_test(NAME test_lb_webb_intmax SOURCE ${_dtwc_test_source_test_lb_webb_intmax}
    MARKER "F57_LB_WEBB_INTMAX l1=4/4 squared=8/8 global_parity=2/2 admissible=2/2 skips=0 verdict=PASS"
    ASSERT_FLOOR 12 CASE_FLOOR 1 SERIAL TIMEOUT 30)

# F21: public-header compile and state-equivalence gate.
dtwc_add_test(NAME unit_test_DataLoader SOURCE ${_dtwc_test_source_unit_test_DataLoader}
    MARKER "F21_CPP_NAMES canonical=4/4 legacy=4/4 overloads=12/12 loader_state=22/22 path_state=16/16 cstring_copy=4/4 skips=0 verdict=PASS"
    ASSERT_FLOOR 79 CASE_FLOOR 2)

# F13: nearest-medoid assignment through a multi-worker OpenMP region.
dtwc_add_test(NAME unit_test_nearest_medoid_assignment SOURCE ${_dtwc_test_source_unit_test_nearest_medoid_assignment}
    ENVIRONMENT "OMP_NUM_THREADS=4" PROCESSORS 4
    ASSERT_FLOOR 50 CASE_FLOOR 8)

# F14 (focused): native distance-matrix CSV bytes; mmap is a required subject
# only in a build that publishes it.
if(_dtwc_has_mmap)
    set(_f14_marker "F14_CSV_CONTRACT dense=ran mmap=ran skips=0")
    set(_f14_floor ASSERT_FLOOR 80 CASE_FLOOR 12)
else()
    set(_f14_marker "F14_CSV_CONTRACT dense=ran mmap=unavailable skips=0")
    set(_f14_floor ASSERT_FLOOR 50 CASE_FLOOR 8)
endif()
dtwc_add_test(NAME unit_test_distance_matrix_csv SOURCE ${_dtwc_test_source_unit_test_distance_matrix_csv}
    FIXTURE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/f14-distance-matrix-csv-unit" FIXTURE_DEFINE DTWC_F14_TEST_ROOT
    MARKER "${_f14_marker}" ${_f14_floor})

# F19: executed public-API gate; Lloyd's artifacts stay inside the build tree.
dtwc_add_test(NAME unit_test_problem_encapsulation SOURCE ${_dtwc_test_source_unit_test_problem_encapsulation}
    FIXTURE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/f19-problem-api" FIXTURE_DEFINE DTWC_F19_TEST_ROOT
    MARKER "F19_PROBLEM_API getters=10/10 setters=9/9 lloyd=ran skips=0"
    ASSERT_FLOOR 30 CASE_FLOOR 3)

# F20: live Problem series-storage gate, per build flavour.
if(_dtwc_has_mmap)
    set(_f20_marker "F20_PROBLEM_STORAGE_POLICY build=llfio-on footprint=288 heap=owning mmap=view values=72/72 names=12/12 ndim_routes=2/2 ordered_pairs=72/72 artifact=pass lifetime=pass loader_auto=mmap view_override=pass subject_skips=0 verdict=PASS")
    set(_f20_floor ASSERT_FLOOR 180 CASE_FLOOR 4)
else()
    set(_f20_marker "F20_PROBLEM_STORAGE_POLICY build=llfio-off footprint=288 heap=owning mmap=rejected values=36/36 names=6/6 ndim_routes=1/1 ordered_pairs=36/36 transaction=pass loader_auto=heap-warning view_override=pass subject_skips=0 verdict=PASS")
    set(_f20_floor ASSERT_FLOOR 90 CASE_FLOOR 4)
endif()
dtwc_add_test(NAME unit_test_problem_storage_policy SOURCE ${_dtwc_test_source_unit_test_problem_storage_policy}
    FIXTURE_ROOT "${CMAKE_CURRENT_BINARY_DIR}/f20-problem-storage" FIXTURE_DEFINE DTWC_F20_TEST_ROOT
    MARKER "${_f20_marker}" ${_f20_floor} SERIAL)

# F15: the three portable deterministic generator schedules.
dtwc_add_test(NAME unit_test_deterministic_series SOURCE ${_dtwc_test_source_unit_test_deterministic_series}
    COMPILE_DEFINITIONS DTWC_F15_SOURCE_ROOT="${CMAKE_SOURCE_DIR}"
    MARKER "F15_TEST_SUPPORT generator=portable scalar=ran row_seeded=ran continuous=ran dense=ran source_audit=ran skips=0"
    ASSERT_FLOOR 177 CASE_FLOOR 6)

# F16: tracked presets stay truthful (configure-time guard below, runtime marker here).
dtwc_add_test(NAME test_supply_chain_pinning SOURCE ${_dtwc_test_source_test_supply_chain_pinning}
    MARKER "F16_CMAKE_PRESETS floor=3[.]26[.]0 compiler=clang[+][+] host_conditions=ran metadata_guard=ran skips=0"
    ASSERT_FLOOR 35 CASE_FLOOR 4)

# F6: the Arrow/Parquet readers must RUN when the build publishes Arrow, and
# are not registered at all when it does not (no green stub).
dtwc_add_test(NAME test_io_readers SOURCE ${_dtwc_test_source_test_io_readers}
    REQUIRES DTWC_HAS_ARROW ASSERT_FLOOR 300 CASE_FLOOR 1)

# unit_test_mpi owns main() (MPI_Init/MPI_Finalize cannot bracket Catch2's main),
# so it prints no Catch2 summary in ANY configuration and cannot be floored; its
# own line is the gate instead. The two branches of its #ifdef DTWC_HAS_MPI print
# different lines, so the marker follows the same definition the source does.
# Only the OFF branch is measured (every W0 matrix is DTWC_ENABLE_MPI=OFF); the
# ON marker is transcribed from unit_test_mpi.cpp:200-203 and unverified here.
set(_dtwc_mpi_marker "MPI not enabled \\(DTWC_ENABLE_MPI=OFF\\)[.] Skipping MPI tests[.]")
if("DTWC_HAS_MPI" IN_LIST _dtwc_public_defs)
    set(_dtwc_mpi_marker "=== MPI Tests: [0-9]+/[0-9]+ passed, 0 failed ===")
endif()
dtwc_add_test(NAME unit_test_mpi SOURCE ${_dtwc_test_source_unit_test_mpi}
    MARKER_ONLY MARKER "${_dtwc_mpi_marker}")

# Capability-dependent tests: skip tolerance only in the build that lacks the
# capability; strict and floored everywhere else.
foreach(_dtwc_mmap_test IN ITEMS
        unit_test_mmap_data_store unit_test_mmap_distance_matrix
        unit_test_problem_semantic_transactions unit_test_pruned_distance_matrix
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

# Device tests: no CUDA/Metal device in any CI matrix, so no measured floor
# exists yet; W2b pins real floors on the local RTX 4000 Ada / macOS leg.
foreach(_dtwc_device_test IN ITEMS
        test_cuda_correctness test_cuda_kernel_override test_cuda_launch_guards
        test_cuda_lb_keogh test_metal_correctness test_metal_lb_keogh test_metal_mmap)
    dtwc_add_test(NAME ${_dtwc_device_test} SOURCE ${_dtwc_test_source_${_dtwc_device_test}}
        MAY_SKIP ASSERT_FLOOR 1 CASE_FLOOR 1)
endforeach()

# The scalar-vs-ndim=1 wall-clock parity assertion must not share the machine.
dtwc_add_test(NAME test_multivariate_adversarial SOURCE ${_dtwc_test_source_test_multivariate_adversarial} SERIAL)

# Pass 2: everything else, strict and floored from tests/floors.cmake. A name
# already handled above stays handled even when REQUIRES declined to register it.
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(_dtwc_test_name ${TEST_SOURCE} NAME_WE)
    if(NOT "${_dtwc_test_name}" IN_LIST DTWC_TEST_REGISTERED)
        dtwc_add_test(NAME ${_dtwc_test_name} SOURCE ${TEST_SOURCE})
    endif()
endforeach()

# F16 keeps the tracked presets truthful and host-specific. CMake's JSON
# implementation is the configure-time parser arbiter; the native test below
# independently pins the same fixed scalar metadata and prints the runtime
# execution marker required by CTest.
set(_f16_presets_file "${CMAKE_SOURCE_DIR}/CMakePresets.json")
file(READ "${_f16_presets_file}" _f16_presets_json)
set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS
        "${_f16_presets_file}"
        "${CMAKE_SOURCE_DIR}/pyproject.toml")

string(JSON _f16_schema GET "${_f16_presets_json}" version)
string(JSON _f16_floor_major GET
    "${_f16_presets_json}" cmakeMinimumRequired major)
string(JSON _f16_floor_minor GET
    "${_f16_presets_json}" cmakeMinimumRequired minor)
string(JSON _f16_floor_patch GET
    "${_f16_presets_json}" cmakeMinimumRequired patch)
set(_f16_preset_floor
    "${_f16_floor_major}.${_f16_floor_minor}.${_f16_floor_patch}")
file(STRINGS "${CMAKE_SOURCE_DIR}/CMakeLists.txt"
    _f16_root_first_line
    LIMIT_COUNT 1)
if(NOT _f16_root_first_line STREQUAL
        "cmake_minimum_required(VERSION 3.26)")
    message(FATAL_ERROR
        "F16 root first-command drift: ${_f16_root_first_line}")
endif()
set(_f16_root_floor "3.26.0")

if(NOT _f16_schema EQUAL 6)
    message(FATAL_ERROR
        "F16 preset schema drift: expected 6, found ${_f16_schema}")
endif()
if(NOT _f16_preset_floor VERSION_EQUAL "3.26.0"
        OR NOT _f16_root_floor VERSION_EQUAL "3.26.0"
        OR NOT _f16_preset_floor VERSION_EQUAL "${_f16_root_floor}")
    message(FATAL_ERROR
        "F16 CMake floor drift: preset=${_f16_preset_floor}, "
        "root=${_f16_root_floor}, expected=3.26.0")
endif()

function(_f16_find_preset out_index collection expected_name)
    string(JSON _count LENGTH "${_f16_presets_json}" "${collection}")
    math(EXPR _last "${_count} - 1")
    set(_found -1)
    foreach(_index RANGE 0 ${_last})
        string(JSON _name GET
            "${_f16_presets_json}" "${collection}" ${_index} name)
        if(_name STREQUAL expected_name)
            if(NOT _found EQUAL -1)
                message(FATAL_ERROR
                    "F16 duplicate ${collection} preset: ${expected_name}")
            endif()
            set(_found ${_index})
        endif()
    endforeach()
    if(_found EQUAL -1)
        message(FATAL_ERROR
            "F16 missing ${collection} preset: ${expected_name}")
    endif()
    set(${out_index} ${_found} PARENT_SCOPE)
endfunction()

function(_f16_require_condition collection index expected_host)
    string(JSON _type GET
        "${_f16_presets_json}" "${collection}" ${index} condition type)
    string(JSON _lhs GET
        "${_f16_presets_json}" "${collection}" ${index} condition lhs)
    string(JSON _rhs GET
        "${_f16_presets_json}" "${collection}" ${index} condition rhs)
    if(NOT _type STREQUAL "equals"
            OR NOT _lhs STREQUAL "\${hostSystemName}"
            OR NOT _rhs STREQUAL "${expected_host}")
        message(FATAL_ERROR
            "F16 ${collection}[${index}] condition drift: "
            "type=${_type}, lhs=${_lhs}, rhs=${_rhs}, "
            "expected=${expected_host}")
    endif()
endfunction()

string(JSON _f16_configure_count LENGTH
    "${_f16_presets_json}" configurePresets)
string(JSON _f16_build_count LENGTH
    "${_f16_presets_json}" buildPresets)
string(JSON _f16_test_count LENGTH
    "${_f16_presets_json}" testPresets)
if(NOT _f16_configure_count EQUAL 6
        OR NOT _f16_build_count EQUAL 3
        OR NOT _f16_test_count EQUAL 2)
    message(FATAL_ERROR
        "F16 preset inventory drift: "
        "configure=${_f16_configure_count}/6, "
        "build=${_f16_build_count}/3, test=${_f16_test_count}/2")
endif()

_f16_find_preset(_f16_default configurePresets default)
_f16_find_preset(_f16_clang_win configurePresets clang-win)
_f16_find_preset(_f16_clang_win_debug configurePresets clang-win-debug)
_f16_find_preset(_f16_msvc configurePresets msvc)
_f16_find_preset(_f16_gcc_linux configurePresets gcc-linux)
_f16_find_preset(_f16_clang_macos configurePresets clang-macos)

_f16_require_condition(configurePresets ${_f16_clang_win} Windows)
_f16_require_condition(configurePresets ${_f16_clang_win_debug} Windows)
_f16_require_condition(configurePresets ${_f16_msvc} Windows)
_f16_require_condition(configurePresets ${_f16_gcc_linux} Linux)
_f16_require_condition(configurePresets ${_f16_clang_macos} Darwin)

string(JSON _f16_default_binary GET
    "${_f16_presets_json}" configurePresets ${_f16_default} binaryDir)
string(JSON _f16_default_hidden GET
    "${_f16_presets_json}" configurePresets ${_f16_default} hidden)
string(JSON _f16_default_compiler ERROR_VARIABLE _f16_default_compiler_error
    GET "${_f16_presets_json}" configurePresets ${_f16_default}
    cacheVariables CMAKE_CXX_COMPILER)
string(JSON _f16_default_c_compiler
    ERROR_VARIABLE _f16_default_c_compiler_error
    GET "${_f16_presets_json}" configurePresets ${_f16_default}
    cacheVariables CMAKE_C_COMPILER)
string(JSON _f16_default_toolchain ERROR_VARIABLE _f16_default_toolchain_error
    GET "${_f16_presets_json}" configurePresets ${_f16_default}
    cacheVariables CMAKE_TOOLCHAIN_FILE)
if(NOT _f16_default_hidden
        OR NOT _f16_default_binary STREQUAL "\${sourceDir}/build"
        OR _f16_default_compiler_error STREQUAL "NOTFOUND")
    message(FATAL_ERROR
        "F16 hidden default must keep the shared build directory and no compiler")
endif()
if(_f16_default_c_compiler_error STREQUAL "NOTFOUND"
        OR _f16_default_toolchain_error STREQUAL "NOTFOUND")
    message(FATAL_ERROR
        "F16 hidden default must not encode a C compiler or toolchain")
endif()
string(REGEX MATCH "[A-Za-z]:/" _f16_windows_absolute_path
    "${_f16_presets_json}")
if(_f16_windows_absolute_path)
    message(FATAL_ERROR
        "F16 tracked presets must not encode a Windows drive path")
endif()

string(JSON _f16_win_compiler GET
    "${_f16_presets_json}" configurePresets ${_f16_clang_win}
    cacheVariables CMAKE_CXX_COMPILER)
if(NOT _f16_win_compiler STREQUAL "clang++")
    message(FATAL_ERROR
        "F16 clang-win must use clang++ from PATH on Windows only")
endif()

string(JSON _f16_win_parent GET
    "${_f16_presets_json}" configurePresets ${_f16_clang_win} inherits)
string(JSON _f16_debug_parent GET
    "${_f16_presets_json}" configurePresets ${_f16_clang_win_debug} inherits)
string(JSON _f16_msvc_parent GET
    "${_f16_presets_json}" configurePresets ${_f16_msvc} inherits)
string(JSON _f16_gcc_parent GET
    "${_f16_presets_json}" configurePresets ${_f16_gcc_linux} inherits)
string(JSON _f16_gcc_compiler GET
    "${_f16_presets_json}" configurePresets ${_f16_gcc_linux}
    cacheVariables CMAKE_CXX_COMPILER)
string(JSON _f16_mac_parent GET
    "${_f16_presets_json}" configurePresets ${_f16_clang_macos} inherits)
string(JSON _f16_mac_c GET
    "${_f16_presets_json}" configurePresets ${_f16_clang_macos}
    cacheVariables CMAKE_C_COMPILER)
string(JSON _f16_mac_cxx GET
    "${_f16_presets_json}" configurePresets ${_f16_clang_macos}
    cacheVariables CMAKE_CXX_COMPILER)
if(NOT _f16_win_parent STREQUAL "default"
        OR NOT _f16_debug_parent STREQUAL "clang-win"
        OR NOT _f16_msvc_parent STREQUAL "default"
        OR NOT _f16_gcc_parent STREQUAL "default"
        OR NOT _f16_gcc_compiler STREQUAL "g++"
        OR NOT _f16_mac_parent STREQUAL "default"
        OR NOT _f16_mac_c STREQUAL "/usr/bin/clang"
        OR NOT _f16_mac_cxx STREQUAL "/usr/bin/clang++")
    message(FATAL_ERROR
        "F16 platform preset inheritance/compiler/condition drift")
endif()

foreach(_f16_collection IN ITEMS buildPresets testPresets)
    _f16_find_preset(_f16_win_entry "${_f16_collection}" clang-win)
    _f16_find_preset(_f16_mac_entry "${_f16_collection}" clang-macos)
    string(JSON _f16_win_reference GET
        "${_f16_presets_json}" "${_f16_collection}" ${_f16_win_entry}
        configurePreset)
    string(JSON _f16_mac_reference GET
        "${_f16_presets_json}" "${_f16_collection}" ${_f16_mac_entry}
        configurePreset)
    _f16_require_condition(
        "${_f16_collection}" ${_f16_win_entry} Windows)
    _f16_require_condition(
        "${_f16_collection}" ${_f16_mac_entry} Darwin)
    if(NOT _f16_win_reference STREQUAL "clang-win"
            OR NOT _f16_mac_reference STREQUAL "clang-macos")
        message(FATAL_ERROR
            "F16 ${_f16_collection} reference/condition drift")
    endif()
endforeach()
_f16_find_preset(_f16_debug_build buildPresets clang-win-debug)
string(JSON _f16_debug_build_reference GET
    "${_f16_presets_json}" buildPresets ${_f16_debug_build} configurePreset)
_f16_require_condition(buildPresets ${_f16_debug_build} Windows)
if(NOT _f16_debug_build_reference STREQUAL "clang-win-debug")
    message(FATAL_ERROR "F16 clang-win-debug build preset drift")
endif()

file(READ "${CMAKE_SOURCE_DIR}/pyproject.toml" _f16_pyproject)
string(REGEX MATCHALL
    "(^|[\r\n])[ \t]*cmake[.]version[ \t]*=[ \t]*\\\">=3[.]26\\\""
    _f16_pyproject_floor_matches
    "${_f16_pyproject}")
list(LENGTH _f16_pyproject_floor_matches _f16_pyproject_floor_count)
if(NOT _f16_pyproject_floor_count EQUAL 1)
    message(FATAL_ERROR
        "F16 pyproject CMake floor drift: expected one >=3.26 scalar")
endif()

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
# One CTest entry drives tests/matlab through the freshly built MEX. Incomplete
# means "filtered by assumption" — a silent skip that reads as green in MATLAB's
# summary — so it fails the gate. The sole exclusion is the parallelisation
# capability PAIR: BOTH names are allow-listed, so at most these two may be
# Incomplete (an OpenMP-ON build filters the serial case, an OpenMP-OFF build the
# runtime case; the allow-list does not itself pin which, nor that only one is).
# The passed-count floor proves the suite actually ran rather than being filtered
# out. DTWC_MATLAB_SUITE_MIN_PASSED is pinned two below the measured run (123
# passed on R2024b and R2025b, 2026-09-02); re-pin it and the AGENTS.md MATLAB
# inventory whenever the suite grows.
set(DTWC_MATLAB_SUITE_MIN_PASSED 121 CACHE STRING
    "Minimum MATLAB test cases that must PASS for matlab_suite to be green")

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
                "addpath('${CMAKE_SOURCE_DIR}/bindings/matlab'); addpath('$<TARGET_FILE_DIR:dtwc_mex>'); r = runtests('${CMAKE_CURRENT_SOURCE_DIR}/matlab'); allowed = {'test_test_api/test_parallelisation_runtime_engages', 'test_test_api/test_parallelisation_serial_is_honest'}; bad = setdiff({r([r.Incomplete]).Name}, allowed); fprintf('matlab_suite: %d run, %d passed, %d failed, %d incomplete%s', numel(r), sum([r.Passed]), sum([r.Failed]), sum([r.Incomplete]), newline); assert(sum([r.Failed]) == 0 && isempty(bad) && sum([r.Passed]) >= ${DTWC_MATLAB_SUITE_MIN_PASSED}, 'matlab_suite gate failed');"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
        set_tests_properties(matlab_suite PROPERTIES
            LABELS "matlab" RUN_SERIAL TRUE TIMEOUT 1800)
    else()
        message(WARNING
            "DTWC_BUILD_MATLAB=ON but MATLAB not found: matlab_suite not registered")
    endif()
endif()
