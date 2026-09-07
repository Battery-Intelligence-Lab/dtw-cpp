# W0 Baseline + Tooling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Put the gates in place before anything they protect changes: every CTest entry floored with skip tolerance opt-in, the conformance oracle read-only and hash-pinned, build/CI hygiene, a layer-manifest report, a shared allocation guard, two registered benchmarks and an IPO-inlining report, all recorded as the campaign baseline.

**Architecture:** One CMake registration helper (`dtwc_add_test` / `dtwc_add_script_test`) replaces the blanket `SKIP_RETURN_CODE 4` and 13 hand-written gate blocks; a generated floors table feeds it. The conformance test loses its self-healing branch and gains a SHA-256 pin; regeneration becomes a separate tool. Build hygiene is one slice over `CMakeLists.txt`, `cmake/*.cmake`, presets and workflows. Evidence (floors, benchmark numbers, layer edges, IPO symbols) lands in one run-log.

**Tech Stack:** CMake 3.26 (no `{m,n}` in its regex dialect), Catch2 v3.13, Google Benchmark, Python 3 (stdlib only, run through `uv`), GitHub Actions, GCC/Clang/MSVC.

**Spec:** `.claude/specs/2026-09-07-design-2.0-campaign.md` (Part IV.2 row "W0", III.7, III.8, II.6) and ledger rows T-15, T-16, B-12, B-13, B-15, B-17, C-08, C-24, A-06 in `.claude/specs/2026-09-07-diff-ledger.md`.

## Global Constraints

- Branch `design-2.0` from `Claude` at `a31956e`; one ledger row = one conventional commit (spec IV.1). Never `git push`, tag, or delete `build*/` directories (AGENTS.md).
- Optional dependencies stay optional (OpenMP, HiGHS, Gurobi, CUDA, Metal, MPI, llfio, Arrow, YAML); every guard that must fire on a build WITHOUT a dependency lives OUTSIDE that dependency's `#ifdef`.
- No silent fallbacks; no runtime dependence on repo-relative paths.
- C++20 floor: GCC 11/12, Clang 14–17, Apple Clang, MSVC 19.3x+ (spec II.10). No new third-party library in this wave.
- Python only through `uv` (`uv run --no-project python <script>`).
- Tier-2 `Problem` signatures are frozen; W0 touches no public C++ signature.
- Every user-visible change gets one `CHANGELOG.md` line under `## [Unreleased]`; lessons go to `.claude/LESSONS.md`; floors are updated in `AGENTS.md` in the same session.
- Machine-independent counters beside every wall-clock number (spec II.6).
- At most 4 concurrent agents.
- Records: run-log `.claude/baselines/2026-09-07-design-2.0-W0.md`, ledger `status` column, handoff `.claude/summaries/handoff-2026-09-07-design-2.0.md`.
- Build directories (AGENTS.md "Build & gate recipes"): `build/highs-1151` (clang + Ninja + Release, HiGHS ON, llfio ON, Arrow OFF, YAML ON — canonical), `build/nollfio` (same with `-DDTWC_ENABLE_LLFIO=OFF`), `build/arrow-pyarrow-23` (Arrow ON via PyArrow 23), `build/msvc-debug` (Visual Studio generator, Debug). Full test runs are SERIAL evidence (`ctest -j1`): concurrent runs collide on source-root-relative test artifacts.

---

## File map

| File | Responsibility |
| --- | --- |
| `cmake/DtwcRegex.cmake` (new) | `_dtwc_regex_at_least(n out)`: CMake-dialect regex for "integer ≥ n" |
| `cmake/DtwcTest.cmake` (new; replaces `cmake/Coverage.cmake`) | `DTWC_TEST_SKIP_REGEX`, `dtwc_add_test`, `dtwc_add_script_test`, coverage option |
| `cmake/DtwcOptional.cmake` (new) | `dtwc_disable_optional(NAME reason)` — the one degradation path |
| `cmake/DtwcLayers.cmake` (new) | layer manifest + `dtwc_check_layers` scanner (report-only in W0) |
| `tests/floors.cmake` (generated) | `DTWC_TEST_FLOOR_<test> = "<assertions>;<cases>"` |
| `tests/cmake/test_regex_at_least.cmake`, `tests/cmake/test_layer_check.cmake` (new) | `cmake -P` self-tests of the two CMake tools |
| `tests/CMakeLists.txt` | registrations through the helper; table-driven F16 guard |
| `tests/conformance/conformance_pipeline.hpp` (new) | the conformance pipeline shared by the test and the regen tool |
| `tests/conformance/cpp_conformance.cpp` | read-only, hash-pinned conformance test |
| `tests/conformance/regen_conformance_reference.cc` (new) | explicit regeneration tool (`.cc`: outside the test glob) |
| `tests/integration/test_conformance_reference_read_only.cmake` (new) | proves a missing reference fails instead of healing |
| `tests/support/allocation_guard.hpp` (new) | global `operator new` counter + `AllocationCount` scope |
| `tests/unit/unit_test_allocation_guard.cpp` (new) | contract test of the guard |
| `tests/unit/test_fast_math_consumer.cpp` (new) | AROW/NaN contract under `-ffast-math` (C-08) |
| `scripts/measure_test_floors.py`, `scripts/check_ipo_inlining.py`, `scripts/check_warning_ratchet.py` (new) | floors table generator; disassembly report; CI warning ratchet |
| `benchmarks/CMakeLists.txt`, `benchmarks/bench_fast_pam_swap.cpp`, `benchmarks/bench_matrix_set_contention.cpp` | `dtwc_add_benchmark`, guarded backend benchmarks, the two W0 benchmarks |
| `CMakeLists.txt`, `cmake/Dependencies.cmake`, `cmake/FindGUROBI.cmake`, `cmake/StaticAnalyzers.cmake`, `cmake/InterproceduralOptimization.cmake`, `dtwc/CMakeLists.txt`, `dtwc/missing_utils.hpp`, `CMakePresets.json`, `cmake/CPM.cmake`, `THIRD_PARTY_LICENSES.md` | hygiene slice, header file set, fast-math contract, presets, vendored CPM |
| `.github/workflows/ubuntu-unit.yml`, `.github/workflows/lint.yml` (new) | sanitizer options, bare-core leg, dev-warnings leg, lint job |
| `.claude/baselines/2026-09-07-design-2.0-W0.md` (new), `AGENTS.md`, `CHANGELOG.md`, `.claude/LESSONS.md`, ledger, handoff | records |

---

### Task 0: Branch, baseline logs, run-log

**Files:**

- Create: `.claude/baselines/2026-09-07-design-2.0-W0.md`
- Commit (untracked today): `.claude/specs/*.md`, `.claude/reports/2026-09-07-*`, `.claude/summaries/handoff-2026-09-07-design-2.0.md`

**Interfaces:**

- Produces: four `ctest -V` logs (`build/<dir>/w0-baseline-ctest.log`) consumed by Task 1b; the run-log every later task appends to.

- [ ] **Step 1: Create the branch**

```bash
git checkout -b design-2.0 a31956e
git status --short   # only the untracked .claude/ files listed
```

- [ ] **Step 2: Commit the campaign inputs**

```bash
git add .claude/specs .claude/reports/2026-09-07-asis-algorithms-mip.md .claude/reports/2026-09-07-asis-backends.md .claude/reports/2026-09-07-asis-core.md .claude/reports/2026-09-07-asis-io-cli-build.md .claude/reports/2026-09-07-asis-orchestration.md .claude/reports/2026-09-07-dupscan.txt .claude/reports/2026-09-07-include-graph.tsv .claude/reports/2026-09-07-tests-taxonomy-adversarial-integration.md .claude/reports/2026-09-07-tests-taxonomy-unit-flat.md .claude/reports/2026-09-07-tests-taxonomy-unit-subdirs.md .claude/summaries/handoff-2026-09-07-design-2.0.md
git commit -m "docs: design 2.0 spec, diff ledger, as-is maps and handoff"
```

- [ ] **Step 3: Rebuild the four matrices on the untouched tree**

For each `<dir>` in `build/highs-1151`, `build/nollfio`, `build/arrow-pyarrow-23`:

```bash
cmake -S . -B <dir>            # re-run configure on the existing cache
cmake --build <dir> --parallel
```

For `build/msvc-debug`:

```bash
cmake -S . -B build/msvc-debug
cmake --build build/msvc-debug --config Debug --parallel
```

If a directory is missing, configure it first: `build/highs-1151` with `-G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DDTWC_BUILD_TESTING=ON -DDTWC_ENABLE_HIGHS=ON -DDTWC_ENABLE_LLFIO=ON -DDTWC_ENABLE_ARROW=OFF -DDTWC_ENABLE_YAML=ON`; `build/nollfio` the same plus `-DDTWC_ENABLE_LLFIO=OFF`; `build/msvc-debug` with `-G "Visual Studio 18 2026" -DDTWC_BUILD_TESTING=ON`. `build/arrow-pyarrow-23` needs PyArrow 23 (see AGENTS.md); if it cannot be configured, record that and use the three others.
Expected: every build ends with exit 0.

- [ ] **Step 4: Record the baseline test logs (serial, verbose)**

```bash
ctest --test-dir build/highs-1151 -j1 -V > build/highs-1151/w0-baseline-ctest.log 2>&1; echo "exit=$?"
ctest --test-dir build/nollfio -j1 -V > build/nollfio/w0-baseline-ctest.log 2>&1; echo "exit=$?"
ctest --test-dir build/arrow-pyarrow-23 -j1 -V > build/arrow-pyarrow-23/w0-baseline-ctest.log 2>&1; echo "exit=$?"
ctest --test-dir build/msvc-debug -C Debug -j1 -V > build/msvc-debug/w0-baseline-ctest.log 2>&1; echo "exit=$?"
grep -E "tests passed|tests failed" build/*/w0-baseline-ctest.log
```

Expected (AGENTS.md floors): highs-1151 131/131 with 6 skips; nollfio 131/131 with 9 skips; arrow-pyarrow-23 133/133 with 8 skips; msvc-debug 131/131 with 6 skips. Any failure stops the wave: the baseline must be green before a gate is changed.

- [ ] **Step 5: Start the run-log**

Create `.claude/baselines/2026-09-07-design-2.0-W0.md`:

```markdown
# 2026-09-07 — design 2.0 campaign, W0 baseline + tooling run-log

Base commit `a31956e` (branch `design-2.0`, created from `Claude`). Environment:
Windows 11, clang 21 + Ninja (`build/highs-1151` canonical, `build/nollfio`,
`build/arrow-pyarrow-23`), MSVC 19.50 Debug (`build/msvc-debug`), Python via `uv`.
Spec: `.claude/specs/2026-09-07-design-2.0-campaign.md` Part IV.2 row W0.

## Baseline (before any change)
| Matrix | Result | Skips | Log |
| --- | --- | --- | --- |
| highs-1151 | <n>/<n> | <k> | `build/highs-1151/w0-baseline-ctest.log` |
| nollfio | <n>/<n> | <k> | `build/nollfio/w0-baseline-ctest.log` |
| arrow-pyarrow-23 | <n>/<n> | <k> | `build/arrow-pyarrow-23/w0-baseline-ctest.log` |
| msvc-debug | <n>/<n> | <k> | `build/msvc-debug/w0-baseline-ctest.log` |

## Tasks (append one section per task: question, command, criterion, result, decision)
```

Fill the table from Step 4 (the `<n>`/`<k>` are the numbers the logs printed).

- [ ] **Step 6: Commit**

```bash
git add .claude/baselines/2026-09-07-design-2.0-W0.md
git commit -m "docs: W0 run-log with the pre-change baseline"
```

---

### Task 1a: `_dtwc_regex_at_least` and the registration helper

**Files:**

- Create: `cmake/DtwcRegex.cmake`, `cmake/DtwcTest.cmake`, `tests/cmake/test_regex_at_least.cmake`
- Delete: `cmake/Coverage.cmake` (its macro moves into `DtwcTest.cmake`; the `DTWC_ENABLE_COVERAGE` option survives)
- Modify: `CMakeLists.txt` (the `include(cmake/Coverage.cmake)` line — find it with `grep -n Coverage CMakeLists.txt`)

**Interfaces:**

- Produces: `_dtwc_regex_at_least(<n> <out-var>)`; `DTWC_TEST_SKIP_REGEX`; `dtwc_add_test(NAME SOURCE [MARKER] [ASSERT_FLOOR] [CASE_FLOOR] [MAY_SKIP] [REQUIRES ...] [LAUNCHER ...] [ENVIRONMENT ...] [SERIAL] [PROCESSORS n] [TIMEOUT s] [LABELS ...] [FIXTURE_ROOT p FIXTURE_DEFINE M] [COMPILE_DEFINITIONS ...])`; `dtwc_add_script_test(NAME SCRIPT ARGS ... MARKER [SERIAL] [TIMEOUT s] [LABELS ...] [WORKING_DIRECTORY d])`.
- Consumes: `tests/floors.cmake` variables `DTWC_TEST_FLOOR_<name>` (Task 1b).

- [ ] **Step 1: Write the regex generator**

`cmake/DtwcRegex.cmake`:

```cmake
include_guard(GLOBAL)

# _dtwc_regex_at_least(<n> <out-var>)
# Regex matching every decimal integer >= n (n >= 1, no leading zeros), in the
# CMake regex dialect, which has no {m,n} repetition. Used to floor Catch2's
# "All tests passed (A assertions in C test cases)" summary without pinning an
# upper bound (adding a test must never fail the gate).
function(_dtwc_regex_at_least n out_var)
  if(NOT n MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "_dtwc_regex_at_least: n must be a positive integer, got '${n}'")
  endif()
  string(LENGTH "${n}" digits)
  # Same digit count: n itself, or n's prefix up to position p, a larger digit
  # at p, then free digits.
  set(alternatives "${n}")
  math(EXPR last "${digits} - 1")
  foreach(p RANGE 0 ${last})
    string(SUBSTRING "${n}" 0 ${p} prefix)
    string(SUBSTRING "${n}" ${p} 1 d)
    if(d LESS 9)
      math(EXPR d1 "${d} + 1")
      math(EXPR free "${digits} - ${p} - 1")
      string(REPEAT "[0-9]" ${free} tail)
      list(APPEND alternatives "${prefix}[${d1}-9]${tail}")
    endif()
  endforeach()
  # More digits than n.
  string(REPEAT "[0-9]" ${digits} same)
  list(APPEND alternatives "[1-9]${same}[0-9]*")
  list(JOIN alternatives "|" joined)
  set(${out_var} "(${joined})" PARENT_SCOPE)
endfunction()
```

- [ ] **Step 2: Write its self-test (failing first: the file does not exist yet if you run this before Step 1 — run it once now to see it pass, and once with a deliberately wrong expectation to see it fail)**

`tests/cmake/test_regex_at_least.cmake`:

```cmake
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
```

- [ ] **Step 3: Run the self-test**

Run: `cmake -P tests/cmake/test_regex_at_least.cmake`
Expected: `-- CMAKE_REGEX_AT_LEAST cases=32 failures=0`, exit 0. Then temporarily change `"79;78;0"` to `"79;78;1"`, rerun, expect one `SEND_ERROR` line and exit 1; revert.

- [ ] **Step 4: Write the registration helper**

`cmake/DtwcTest.cmake`:

```cmake
include_guard(GLOBAL)
include("${CMAKE_CURRENT_LIST_DIR}/DtwcRegex.cmake")

option(DTWC_ENABLE_COVERAGE "Enable coverage reporting for GCC or Clang" OFF)

# The one skip regex for the tree (tests taxonomy §2.3 variant B): anchored at a
# line start, matches SKIP / SKIPPED / SKIPPING as a word, never the marker
# substring "skips=0".
set(DTWC_TEST_SKIP_REGEX
    "(^|[\r\n])[ \t]*[Ss][Kk][Ii][Pp]([Pp][Ee][Dd]|[Pp][Ii][Nn][Gg])?([ :]|$)")

# _dtwc_test_summary_regex(<assert-floor> <case-floor> <out-var>)
# Catch2's own summary line, floored on both dimensions.
function(_dtwc_test_summary_regex assert_floor case_floor out_var)
  _dtwc_regex_at_least(${assert_floor} a)
  _dtwc_regex_at_least(${case_floor} c)
  set(${out_var} "All tests passed \\(${a} assertions? in ${c} test cases?\\)" PARENT_SCOPE)
endfunction()

# dtwc_add_test(NAME <target> SOURCE <file>
#   [MARKER <regex>] [ASSERT_FLOOR <n>] [CASE_FLOOR <n>] [MAY_SKIP]
#   [REQUIRES <compile-definition>...] [LAUNCHER <command>...]
#   [ENVIRONMENT <k=v>...] [SERIAL] [PROCESSORS <n>] [TIMEOUT <s>] [LABELS <l>...]
#   [FIXTURE_ROOT <absolute-path> FIXTURE_DEFINE <MACRO>]
#   [COMPILE_DEFINITIONS <d>...])
#
# One Catch2 test executable + one CTest entry. Defaults are the strict ones:
#   * a skip line fails the test; MAY_SKIP opts into Catch2's exit-4 skip
#     (the all-skipped summary "N skipped" is then accepted);
#   * the binary must print Catch2's summary with at least ASSERT_FLOOR
#     assertions in at least CASE_FLOOR cases. Floors default to the entry in
#     tests/floors.cmake (DTWC_TEST_FLOOR_<target>); a test with no floor at all
#     is a configure error, never a green stub;
#   * MARKER, when given, must precede the summary (subject ran, not just
#     "many assertions ran");
#   * REQUIRES <def>: register only when dtwc++ publishes that definition;
#     otherwise the subject is absent by construction and nothing is added.
function(dtwc_add_test)
  set(options MAY_SKIP SERIAL)
  set(one NAME SOURCE MARKER ASSERT_FLOOR CASE_FLOOR PROCESSORS TIMEOUT FIXTURE_ROOT FIXTURE_DEFINE)
  set(multi REQUIRES LAUNCHER ENVIRONMENT LABELS COMPILE_DEFINITIONS)
  cmake_parse_arguments(ARG "${options}" "${one}" "${multi}" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "dtwc_add_test(${ARG_NAME}): unknown arguments ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT ARG_NAME OR NOT ARG_SOURCE)
    message(FATAL_ERROR "dtwc_add_test: NAME and SOURCE are required")
  endif()

  get_target_property(_public_defs dtwc++ INTERFACE_COMPILE_DEFINITIONS)
  foreach(req IN LISTS ARG_REQUIRES)
    if(NOT "${req}" IN_LIST _public_defs)
      message(STATUS "dtwc_add_test: ${ARG_NAME} not registered (dtwc++ does not publish ${req})")
      return()
    endif()
  endforeach()

  add_executable(${ARG_NAME} ${ARG_SOURCE})
  target_link_libraries(${ARG_NAME} PRIVATE dtwc++ Catch2::Catch2WithMain project_options)
  target_compile_definitions(${ARG_NAME} PRIVATE DTWC_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data" ${ARG_COMPILE_DEFINITIONS})
  if(DTWC_ENABLE_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
      target_compile_options(${ARG_NAME} PUBLIC --coverage -O0)
      target_link_libraries(${ARG_NAME} PUBLIC --coverage)
    else()
      message(FATAL_ERROR "GCC or Clang required with DTWC_ENABLE_COVERAGE: found ${CMAKE_CXX_COMPILER_ID}")
    endif()
  endif()

  set(_env ${ARG_ENVIRONMENT})
  if(ARG_FIXTURE_ROOT)
    if(NOT IS_ABSOLUTE "${ARG_FIXTURE_ROOT}")
      message(FATAL_ERROR "dtwc_add_test(${ARG_NAME}): FIXTURE_ROOT must be absolute: ${ARG_FIXTURE_ROOT}")
    endif()
    if(NOT ARG_FIXTURE_DEFINE)
      message(FATAL_ERROR "dtwc_add_test(${ARG_NAME}): FIXTURE_ROOT needs FIXTURE_DEFINE")
    endif()
    target_compile_definitions(${ARG_NAME} PRIVATE ${ARG_FIXTURE_DEFINE}="${ARG_FIXTURE_ROOT}")
    list(APPEND _env "TMP=${ARG_FIXTURE_ROOT}" "TEMP=${ARG_FIXTURE_ROOT}" "TMPDIR=${ARG_FIXTURE_ROOT}")
  endif()

  get_target_property(_emulator ${ARG_NAME} CROSSCOMPILING_EMULATOR)
  if(NOT _emulator)
    set(_emulator "")
  endif()
  add_test(NAME ${ARG_NAME}
           COMMAND ${ARG_LAUNCHER} ${_emulator} $<TARGET_FILE:${ARG_NAME}>
           WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

  if(NOT ARG_ASSERT_FLOOR AND DEFINED DTWC_TEST_FLOOR_${ARG_NAME})
    list(GET DTWC_TEST_FLOOR_${ARG_NAME} 0 ARG_ASSERT_FLOOR)
    list(GET DTWC_TEST_FLOOR_${ARG_NAME} 1 ARG_CASE_FLOOR)
  endif()
  if(NOT ARG_ASSERT_FLOOR OR NOT ARG_CASE_FLOOR)
    message(FATAL_ERROR
      "dtwc_add_test(${ARG_NAME}): no floor. Pass ASSERT_FLOOR/CASE_FLOOR or regenerate "
      "tests/floors.cmake with scripts/measure_test_floors.py.")
  endif()
  _dtwc_test_summary_regex(${ARG_ASSERT_FLOOR} ${ARG_CASE_FLOOR} _summary)
  if(ARG_MARKER)
    set(_pass "${ARG_MARKER}(.|[\r\n])*${_summary}")
  else()
    set(_pass "${_summary}")
  endif()
  if(ARG_MAY_SKIP)
    # Catch2 exits 4 and prints "test cases: N | N skipped" when everything
    # skipped; a partial skip prints "... | K skipped" with exit 0.
    set_tests_properties(${ARG_NAME} PROPERTIES
      SKIP_RETURN_CODE 4
      PASS_REGULAR_EXPRESSION "${_pass}|test cases: *[0-9]+ \\|.*[0-9]+ skipped")
  else()
    set_tests_properties(${ARG_NAME} PROPERTIES
      FAIL_REGULAR_EXPRESSION "${DTWC_TEST_SKIP_REGEX}"
      PASS_REGULAR_EXPRESSION "${_pass}")
  endif()

  if(_env)
    set_property(TEST ${ARG_NAME} PROPERTY ENVIRONMENT "${_env}")
  endif()
  if(ARG_SERIAL)
    set_property(TEST ${ARG_NAME} PROPERTY RUN_SERIAL TRUE)
  endif()
  if(ARG_PROCESSORS)
    set_property(TEST ${ARG_NAME} PROPERTY PROCESSORS ${ARG_PROCESSORS})
  endif()
  if(ARG_TIMEOUT)
    set_property(TEST ${ARG_NAME} PROPERTY TIMEOUT ${ARG_TIMEOUT})
  endif()
  if(ARG_LABELS)
    set_property(TEST ${ARG_NAME} PROPERTY LABELS "${ARG_LABELS}")
  endif()
endfunction()

# dtwc_add_script_test(NAME <name> SCRIPT <file.cmake> [ARGS <-Dk=v>...]
#   MARKER <regex> [SERIAL] [TIMEOUT <s>] [LABELS <l>...] [WORKING_DIRECTORY <d>])
# A `cmake -P` integration script driving real binaries. Always strict: a skip
# line fails, the exact marker (with computed counters) must be printed.
function(dtwc_add_script_test)
  cmake_parse_arguments(ARG "SERIAL" "NAME;SCRIPT;MARKER;TIMEOUT;WORKING_DIRECTORY" "ARGS;LABELS" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "dtwc_add_script_test(${ARG_NAME}): unknown arguments ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  if(NOT ARG_NAME OR NOT ARG_SCRIPT OR NOT ARG_MARKER)
    message(FATAL_ERROR "dtwc_add_script_test: NAME, SCRIPT and MARKER are required")
  endif()
  if(NOT ARG_WORKING_DIRECTORY)
    set(ARG_WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
  endif()
  add_test(NAME ${ARG_NAME}
           COMMAND ${CMAKE_COMMAND} ${ARG_ARGS} -P "${ARG_SCRIPT}"
           WORKING_DIRECTORY "${ARG_WORKING_DIRECTORY}")
  set_tests_properties(${ARG_NAME} PROPERTIES
    FAIL_REGULAR_EXPRESSION "${DTWC_TEST_SKIP_REGEX}"
    PASS_REGULAR_EXPRESSION "${ARG_MARKER}")
  if(ARG_SERIAL)
    set_property(TEST ${ARG_NAME} PROPERTY RUN_SERIAL TRUE)
  endif()
  if(ARG_TIMEOUT)
    set_property(TEST ${ARG_NAME} PROPERTY TIMEOUT ${ARG_TIMEOUT})
  endif()
  if(ARG_LABELS)
    set_property(TEST ${ARG_NAME} PROPERTY LABELS "${ARG_LABELS}")
  endif()
endfunction()
```

- [ ] **Step 5: Replace `Coverage.cmake`**

```bash
git rm cmake/Coverage.cmake
grep -n "Coverage.cmake" CMakeLists.txt cmake/*.cmake tests/CMakeLists.txt
```

Change the `include(cmake/Coverage.cmake)` line in `CMakeLists.txt` to `include(cmake/DtwcTest.cmake)`. Leave `tests/CMakeLists.txt` alone in this task (its calls to `add_executable_with_coverage_and_test` are rewritten in Task 1c; configure will fail until then, which is why Tasks 1a–1c share one gate).

- [ ] **Step 6: The CTest registration of this self-test is part of Task 1c Step 2 (`cmake_regex_at_least`); nothing to add here**

- [ ] **Step 7: Commit (Tasks 1a–1c land together after Task 1c's gate; stage now)**

```bash
git add cmake/DtwcRegex.cmake cmake/DtwcTest.cmake tests/cmake/test_regex_at_least.cmake CMakeLists.txt
```

---

### Task 1b: Floors table

**Files:**

- Create: `scripts/measure_test_floors.py`, `tests/floors.cmake`

**Interfaces:**

- Consumes: the four `w0-baseline-ctest.log` files from Task 0.
- Produces: `tests/floors.cmake` with `set(DTWC_TEST_FLOOR_<name> "<assertions>;<cases>")` per test that printed a Catch2 summary in at least one log (minimum over logs).

- [ ] **Step 1: Write the script**

`scripts/measure_test_floors.py`:

```python
#!/usr/bin/env python3
"""Derive tests/floors.cmake from one or more `ctest -V` logs.

A ctest -V log prefixes every test output line with "<n>: " and announces each
test with "Start <n>: <name>". Catch2 prints "All tests passed (A assertions in
C test cases)" only when nothing skipped or failed. The floor of a test is the
MINIMUM (A, C) over every log in which it printed that summary: configurations
compile different subsets of assertions (mmap, HiGHS, Arrow, OpenMP), and the
floor must hold in all of them. Tests that never printed the summary get no
entry, and dtwc_add_test() refuses to register them without an explicit floor.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

START = re.compile(r"^\s*Start\s+(\d+): (\S+)")
SUMMARY = re.compile(
    r"^(\d+): All tests passed \((\d+) assertions? in (\d+) test cases?\)")


def parse(log: Path) -> dict[str, tuple[int, int]]:
    names: dict[str, str] = {}
    floors: dict[str, tuple[int, int]] = {}
    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
        started = START.match(line)
        if started:
            names[started.group(1)] = started.group(2)
            continue
        summary = SUMMARY.match(line)
        if summary and summary.group(1) in names:
            floors[names[summary.group(1)]] = (int(summary.group(2)),
                                               int(summary.group(3)))
    return floors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("logs", nargs="+", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    merged: dict[str, tuple[int, int]] = {}
    for log in args.logs:
        for name, (assertions, cases) in parse(log).items():
            previous = merged.get(name)
            merged[name] = ((assertions, cases) if previous is None
                            else (min(previous[0], assertions),
                                  min(previous[1], cases)))
    if not merged:
        print("no Catch2 summaries found in the given logs", file=sys.stderr)
        return 1

    lines = [
        "# Generated by scripts/measure_test_floors.py -- do not edit by hand.",
        "# DTWC_TEST_FLOOR_<test> = \"<min assertions>;<min test cases>\" over the",
        "# ctest -V logs recorded in .claude/baselines/2026-09-07-design-2.0-W0.md.",
        "# Regenerate after a deliberate test change:",
        "#   uv run --no-project python scripts/measure_test_floors.py <logs...> --out tests/floors.cmake",
        "",
    ]
    for name in sorted(merged):
        assertions, cases = merged[name]
        lines.append(f'set(DTWC_TEST_FLOOR_{name} "{assertions};{cases}")')
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"TEST_FLOORS tests={len(merged)} logs={len(args.logs)} out={args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Generate the table**

```bash
uv run --no-project python scripts/measure_test_floors.py \
  build/highs-1151/w0-baseline-ctest.log build/nollfio/w0-baseline-ctest.log \
  build/arrow-pyarrow-23/w0-baseline-ctest.log build/msvc-debug/w0-baseline-ctest.log \
  --out tests/floors.cmake
wc -l tests/floors.cmake
```

Expected: `TEST_FLOORS tests=<about 115> logs=4 out=tests/floors.cmake`. Tests that appear in no log with a summary (expected: `test_cuda_correctness`, `test_cuda_kernel_override`, `test_cuda_launch_guards`, `test_cuda_lb_keogh`, `test_metal_correctness`, `test_metal_lb_keogh`, `test_metal_mmap`; `test_io_readers` appears only in the Arrow log) get explicit floors in Task 1c. Any OTHER test missing from the table is a finding: open its log section, record why it never printed the summary (partial skip in every configuration, or a failure) in the run-log, and give it an explicit `ASSERT_FLOOR`/`CASE_FLOOR` with `MAY_SKIP` in Task 1c.

- [ ] **Step 3: Sanity-check three rows against the logs by hand**

```bash
grep -n "unit_test_barycenter\b" build/highs-1151/w0-baseline-ctest.log | head -3
grep -n "DTWC_TEST_FLOOR_unit_test_barycenter " tests/floors.cmake
```

Expected: the assertion/case numbers match the log's summary line for that test (and are the minimum across the four logs).

- [ ] **Step 4: Stage**

```bash
git add scripts/measure_test_floors.py tests/floors.cmake
```

---

### Task 1c: Register every test through the helper

**Files:**

- Modify: `tests/CMakeLists.txt` (lines 1–141 registration + F22 loops; 143–397 gate blocks; 619–669 F16/F6/serial blocks; 671–909 script tests)

**Interfaces:**

- Consumes: `dtwc_add_test`, `dtwc_add_script_test`, `DTWC_TEST_FLOOR_*`.
- Produces: the same CTest inventory (names unchanged) with every entry floored; `_dtwc_public_defs`, `_dtwc_mmap_skip`, `_dtwc_highs_skip`, `_dtwc_seq_skip` variables reused by later tasks in this file.

- [ ] **Step 1: Replace the F22 probe setup (lines 59–122) with one loop**

Keep lines 1–58 (Python lookup, probe sources, the three `add_library(... OBJECT EXCLUDE_FROM_ALL ...)`). Replace lines 59–122 with:

```cmake
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
```

- [ ] **Step 2: Replace the registration loop and all 13 gate blocks (lines 124–397 and 619–669) with the two-pass registration**

```cmake
include("${CMAKE_SOURCE_DIR}/cmake/DtwcTest.cmake")
include("${CMAKE_CURRENT_SOURCE_DIR}/floors.cmake")

add_test(NAME cmake_regex_at_least
         COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_regex_at_least.cmake")
set_tests_properties(cmake_regex_at_least PROPERTIES
  PASS_REGULAR_EXPRESSION "CMAKE_REGEX_AT_LEAST cases=32 failures=0"
  LABELS "tooling")

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
        test_lagrangian_root test_mip_backend_guards unit_test_benders unit_test_mip)
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

# Pass 2: everything else, strict and floored from tests/floors.cmake.
foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(_dtwc_test_name ${TEST_SOURCE} NAME_WE)
    if(NOT TARGET ${_dtwc_test_name})
        dtwc_add_test(NAME ${_dtwc_test_name} SOURCE ${TEST_SOURCE})
    endif()
endforeach()
```

Keep the `test_arrow_c_data` include-path block (656–661): it is not a gate. Delete the `test_multivariate_adversarial` `RUN_SERIAL` block (663–669) and the F6 block (632–654), both covered above. Keep the F16 configure-time guard (399–617) untouched in this task (Task 4 rewrites it).

- [ ] **Step 3: Migrate the script tests (lines 671–909)**

Replace each `add_test(... -P ...)` + `set_tests_properties` pair with `dtwc_add_script_test`, keeping the executables and the surrounding `if(TARGET dtwc_cl)` / Parquet guards:

```cmake
    dtwc_add_script_test(NAME test_distance_matrix_csv_contract
        SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_distance_matrix_csv_contract.cmake"
        ARGS "-DCLI=$<TARGET_FILE:dtwc_cl>" "-DRESULT_WRITER=$<TARGET_FILE:f14_result_save_writer>"
             "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/conformance/data/conformance_series.csv"
             "-DCONFIG=${CMAKE_CURRENT_SOURCE_DIR}/conformance/conformance.toml"
             "-DBINARY_ROOT=${CMAKE_CURRENT_BINARY_DIR}"
             "-DWORK_ROOT=${CMAKE_CURRENT_BINARY_DIR}/f14-distance-matrix-csv-public"
             "-DHAS_MMAP=${_dtwc_has_mmap}"
        MARKER "${_f14_public_regex}" LABELS integration f14 SERIAL TIMEOUT 120)
```

with `_f14_public_regex` chosen from `_dtwc_has_mmap` exactly as today (lines 689–697; `_f14_has_mmap` becomes `_dtwc_has_mmap`). Likewise:

```cmake
    dtwc_add_script_test(NAME test_cli_resume_state
        SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_cli_resume_state.cmake"
        ARGS "-DCLI=$<TARGET_FILE:dtwc_cl>" "-DFIXTURE_WRITER=$<TARGET_FILE:f17_checkpoint_writer>"
             "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/conformance/data/conformance_series.csv"
             "-DCONFIG=${CMAKE_CURRENT_SOURCE_DIR}/conformance/conformance.toml"
             "-DBINARY_ROOT=${CMAKE_CURRENT_BINARY_DIR}" "-DWORK_ROOT=${CMAKE_CURRENT_BINARY_DIR}/f17-cli-resume"
        MARKER "F17_CLI_RESUME subject=real_dtwc_cl writer=production_serializer runs=12/12 replay_fields=10/10 markers=2/2 algorithm_skipped=1/1 fresh_discriminator=4/4 checkpoint_preserved=10/10 rejection_cases=9/9 sources_preserved=2/2 skips=0"
        LABELS integration f17 SERIAL TIMEOUT 120)

    dtwc_add_script_test(NAME test_cli_rejects_yaml_config
        SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_cli_rejects_unknown_option.cmake"
        ARGS "-DCLI=$<TARGET_FILE:dtwc_cl>" "-DOPTION=--yaml-config"
        MARKER "CLI_REJECTS_UNKNOWN_OPTION subject=real_dtwc_cl option=--yaml-config exit=[0-9]+ nonzero=1 named=1 skips=0"
        LABELS integration cli TIMEOUT 60)

    dtwc_add_script_test(NAME test_cli_config_formats
        SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_cli_config_formats.cmake"
        ARGS "-DCLI=$<TARGET_FILE:dtwc_cl>"
             "-DINPUT=${CMAKE_CURRENT_SOURCE_DIR}/conformance/data/conformance_series.csv"
             "-DTOML=${CMAKE_CURRENT_SOURCE_DIR}/conformance/conformance.toml"
             "-DHAS_YAML=${_dtwc_cl_yaml}" "-DWORK_ROOT=${CMAKE_CURRENT_BINARY_DIR}/cli-config-formats"
        MARKER "CLI_CONFIG_FORMATS subject=real_dtwc_cl yaml=${_dtwc_cl_yaml} checks=${_dtwc_cl_yaml_checks} skips=0"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}" LABELS integration cli SERIAL TIMEOUT 180)
```

Inside the Parquet block (F8 / F13), which today have NO pass/fail regex (taxonomy §2.6), use the markers the scripts already print:

```cmake
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
        MARKER "F13_ASSIGNMENT_CONTRACT subject=real_dtwc_cl runs=4/4"
        LABELS integration arrow f13 SERIAL TIMEOUT 120)
```

Read `tests/integration/test_fast_clara_assignment_contract.cmake:371` and copy the full marker text it prints (it starts with `F13_ASSIGNMENT_CONTRACT subject=real_dtwc_cl runs=${run_count}/4`; the plan pins the prefix, the executor pins the whole line with its computed counters). Also link `project_options` into `f13_poison_parquet_writer` (taxonomy §2.4 item 6). Replace the `_dtwc_test_compile_defs` query at line 822 with `_dtwc_public_defs`. Keep the Windows Arrow runtime `PATH` block as the last statement of the Parquet block.

- [ ] **Step 4: Configure the canonical build and inspect the registered properties**

```bash
cmake -S . -B build/highs-1151 2>&1 | tail -20
ctest --test-dir build/highs-1151 --show-only=json-v1 > build/highs-1151/w0-tests.json
uv run --no-project python - <<'EOF'
import json
d = json.load(open("build/highs-1151/w0-tests.json"))
tests = d["tests"]
def prop(t, name):
    return next((p["value"] for p in t.get("properties", []) if p["name"] == name), None)
no_pass = [t["name"] for t in tests if not prop(t, "PASS_REGULAR_EXPRESSION")]
skip_ok = [t["name"] for t in tests if prop(t, "SKIP_RETURN_CODE") == 4]
print("registered", len(tests)); print("without PASS regex", no_pass); print("skip-tolerant", skip_ok)
EOF
```

Expected: `registered` = the baseline count (131) minus 1 (`test_io_readers` is no longer registered in the Arrow-OFF build) plus 1 (`cmake_regex_at_least`) = 131; `without PASS regex` = `[]` except `matlab_suite` when MATLAB is enabled; `skip-tolerant` = exactly the seven device tests (llfio and HiGHS are ON here). A test named in `no_pass` is a bug in this task.

- [ ] **Step 5: Run the full canonical matrix serially**

```bash
ctest --test-dir build/highs-1151 -j1 --output-on-failure 2>&1 | tail -15
```

Expected: `100% tests passed`, 5 skipped (cuda×2 and metal×3 exit 4 as before; the two partially-running device tests now report Passed through the skip alternative; `test_io_readers` is absent), same wall time as the baseline within noise. Any new failure is a floor or marker transcription error: compare the failing test's summary line in `w0-baseline-ctest.log` with its `PASS_REGULAR_EXPRESSION`.

- [ ] **Step 6: Run `build/nollfio` and `build/msvc-debug` the same way**

```bash
cmake -S . -B build/nollfio && cmake --build build/nollfio --parallel && ctest --test-dir build/nollfio -j1 --output-on-failure 2>&1 | tail -8
cmake -S . -B build/msvc-debug && cmake --build build/msvc-debug --config Debug --parallel && ctest --test-dir build/msvc-debug -C Debug -j1 --output-on-failure 2>&1 | tail -8
```

Expected: nollfio 100% passed (9 skips at baseline become: cuda×2, metal×3, plus the mmap-only binaries that skip entirely — count them from the output and record); msvc-debug 100% passed, 5 skipped.

- [ ] **Step 7: Prove the floor bites (negative control)**

Temporarily edit `tests/floors.cmake`: raise `DTWC_TEST_FLOOR_unit_test_barycenter` assertions by 1000; reconfigure; run `ctest --test-dir build/highs-1151 -R '^unit_test_barycenter$'`. Expected: `***Failed  Required regular expression not found`. Revert the edit and reconfigure. Record the observation in the run-log.

- [ ] **Step 8: Commit**

```bash
git add tests/CMakeLists.txt
git commit -m "build(tests): register every test through dtwc_add_test; floors from tests/floors.cmake (T-16, B-13)"
```

The staged Task 1a/1b files ride in this commit.

---

### Task 2: Conformance reference is read-only and hash-pinned (T-15)

**Files:**

- Create: `tests/conformance/conformance_pipeline.hpp`, `tests/conformance/regen_conformance_reference.cc`, `tests/integration/test_conformance_reference_read_only.cmake`
- Modify: `tests/conformance/cpp_conformance.cpp`, `tests/conformance/conformance_reference.txt` (header comment only), `tests/CMakeLists.txt`

**Interfaces:**

- Produces: `dtwc::conformance::{run_pipeline, read_reference, write_reference, sha256_hex, conformance_dir}` in the header; the `regen_conformance_reference <output-path>` tool; env override `DTWC_CONFORMANCE_DIR` (absolute path to a directory holding `data/conformance_series.csv` and `conformance_reference.txt`) used only by the negative test.

- [ ] **Step 1: Move the pipeline into the shared header**

`tests/conformance/conformance_pipeline.hpp` — the anonymous-namespace code of `cpp_conformance.cpp` lines 57–200 verbatim inside `namespace dtwc::conformance` (no Catch2 include: the regen tool is a plain program), with these changes:

```cpp
#pragma once
#include <dtwc.hpp>
#include <core/sha256.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef DTWC_TEST_DATA_DIR
#error "DTWC_TEST_DATA_DIR must be defined by the build (dtwc_add_test / the regen target)"
#endif

namespace fs = std::filesystem;

namespace dtwc::conformance {

constexpr int kNClusters = 3;
constexpr int kBand = 3;
constexpr int kMaxIter = 100;
constexpr unsigned kSeed = 29;

/// Repo-root/tests/conformance, located from the compile-time DTWC_TEST_DATA_DIR
/// (never a runtime-relative path). DTWC_CONFORMANCE_DIR overrides it with an
/// absolute directory: the read-only integration test points it at a scratch
/// copy that lacks the reference.
inline fs::path conformance_dir()
{
  if (const char* override_dir = std::getenv("DTWC_CONFORMANCE_DIR"))
    return fs::path{ override_dir };
  return fs::path{ DTWC_TEST_DATA_DIR }.parent_path() / "tests" / "conformance";
}
inline fs::path data_csv() { return conformance_dir() / "data" / "conformance_series.csv"; }
inline fs::path reference_file() { return conformance_dir() / "conformance_reference.txt"; }

/// Lower-case hex SHA-256 of a file's bytes (binary read: the file is LF-only
/// by .gitattributes, and the pin must not depend on the platform's text mode).
inline std::string sha256_hex(const fs::path& path)
{
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path.string());
  dtwc::core::detail::Sha256 hasher;
  char buffer[4096];
  while (in.read(buffer, sizeof buffer) || in.gcount() > 0)
    hasher.update(buffer, static_cast<std::size_t>(in.gcount()));
  std::ostringstream hex;
  hex << std::hex << std::setfill('0');
  for (const auto byte : hasher.digest()) hex << std::setw(2) << static_cast<unsigned>(byte);
  return hex.str();
}

// fmt17, CanonicalResult, canonicalise, run_pipeline: unchanged.

/// Write a reference file. Binary mode: LF line endings on every platform, so
/// the bytes the tool writes are the bytes git tracks and the pin hashes.
inline void write_reference(const CanonicalResult& r, const fs::path& out_path)
{
  std::ofstream out(out_path, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write reference: " + out_path.string());
  out << "# DTWC++ 2.0 cross-language conformance reference (Phase 2 Task 2.4).\n"
      << "# Recorded by tests/conformance/regen_conformance_reference.cc, which runs\n"
      << "# the LIVE pipeline: DataLoader -> set_band(3) -> fill_distance_matrix\n"
      << "# -> fast_pam(k=3, seed=29) -> silhouette/davies_bouldin/dunn. The C++,\n"
      << "# Python, MATLAB and CLI routes assert digit-identical labels/medoids and\n"
      << "# scores within 1e-12 rel against THIS file; the C++ route also pins its\n"
      << "# SHA-256 (kReferenceSha256 in cpp_conformance.cpp). No test writes here.\n"
      << "# REGENERATE (only when the pipeline contract changes): build the target\n"
      << "# regen_conformance_reference, run it with this file's path, then update\n"
      << "# the pin it prints. Values are canonical: medoids = sorted series indices;\n"
      << "# labels[i] = rank of series i's medoid; silhouette = mean per-point.\n";
  // (labels / medoids / silhouette / davies_bouldin / dunn lines unchanged)
}

// read_reference: unchanged (takes no argument; reads reference_file()).

} // namespace dtwc::conformance
```

- [ ] **Step 2: Rewrite the test (the failing part first: the hash pin with a placeholder value fails until the reference is regenerated in Step 4)**

`tests/conformance/cpp_conformance.cpp`:

```cpp
/**
 * @file cpp_conformance.cpp
 * @brief C++ route of the cross-language conformance fixture.
 *
 * The permanent parity gate for docs/api-contract-2.0.md §9: the live Tier-2
 * pipeline (conformance_pipeline.hpp) must reproduce the TRACKED reference
 * digit-for-digit. The reference is read-only here and pinned by SHA-256;
 * regenerating it is a separate, explicit tool (regen_conformance_reference)
 * followed by a code change to the pin, so a drifting pipeline can never heal
 * its own oracle.
 */
#include "conformance_pipeline.hpp"

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

using Catch::Matchers::WithinRel;
using namespace dtwc::conformance;

namespace {
/// SHA-256 of tests/conformance/conformance_reference.txt (LF bytes). Update it
/// only together with a regenerated reference and a CHANGELOG line.
constexpr std::string_view kReferenceSha256 =
  "0000000000000000000000000000000000000000000000000000000000000000";
} // namespace

TEST_CASE("Cross-language conformance: reference is tracked, read-only and hash-pinned",
          "[conformance]")
{
  REQUIRE(fs::exists(reference_file()));
  REQUIRE(sha256_hex(reference_file()) == kReferenceSha256);
}

TEST_CASE("Cross-language conformance: C++ route matches recorded reference",
          "[Phase2][conformance]")
{
  REQUIRE(fs::exists(data_csv()));
  const std::string bytes_before = sha256_hex(reference_file());

  const CanonicalResult live = run_pipeline();
  const CanonicalResult ref = read_reference();

  REQUIRE(live.labels == ref.labels);
  REQUIRE(live.medoids == ref.medoids);
  REQUIRE(static_cast<int>(live.medoids.size()) == kNClusters);
  CHECK_THAT(live.silhouette, WithinRel(ref.silhouette, 1e-12));
  CHECK_THAT(live.davies_bouldin, WithinRel(ref.davies_bouldin, 1e-12));
  CHECK_THAT(live.dunn, WithinRel(ref.dunn, 1e-12));

  // The pipeline is a reader: the tracked bytes are untouched after it ran.
  REQUIRE(sha256_hex(reference_file()) == bytes_before);
  REQUIRE(bytes_before == kReferenceSha256);
}
```

- [ ] **Step 3: Write the regeneration tool**

`tests/conformance/regen_conformance_reference.cc`:

```cpp
/**
 * @file regen_conformance_reference.cc
 * @brief The ONLY writer of tests/conformance/conformance_reference.txt.
 *
 * usage: regen_conformance_reference <output-path>
 * Runs the live C++ pipeline and writes the canonical reference to
 * <output-path> (binary, LF). Prints the SHA-256 to paste into kReferenceSha256
 * in cpp_conformance.cpp. Never run by CTest.
 */
#include "conformance_pipeline.hpp"

#include <iostream>

int main(int argc, char** argv)
{
  if (argc != 2) {
    std::cerr << "usage: regen_conformance_reference <output-path>\n";
    return 2;
  }
  const std::filesystem::path out_path{ argv[1] };
  const auto live = dtwc::conformance::run_pipeline();
  dtwc::conformance::write_reference(live, out_path);
  std::cout << "wrote " << out_path.string() << "\n"
            << "sha256 " << dtwc::conformance::sha256_hex(out_path) << "\n"
            << "update kReferenceSha256 in tests/conformance/cpp_conformance.cpp\n";
  return 0;
}
```

Register it in `tests/CMakeLists.txt` (near the `f14_result_save_writer` helper):

```cmake
# The only writer of the conformance reference; built on demand, never a test.
add_executable(regen_conformance_reference EXCLUDE_FROM_ALL conformance/regen_conformance_reference.cc)
target_link_libraries(regen_conformance_reference PRIVATE dtwc++ project_options)
target_compile_definitions(regen_conformance_reference PRIVATE DTWC_TEST_DATA_DIR="${CMAKE_SOURCE_DIR}/data")
```

- [ ] **Step 4: Regenerate the reference once (header comment changed), pin the hash**

```bash
cmake --build build/highs-1151 --target regen_conformance_reference
build/highs-1151/bin/regen_conformance_reference build/highs-1151/conformance_reference.new.txt
diff <(grep -v '^#' tests/conformance/conformance_reference.txt) <(grep -v '^#' build/highs-1151/conformance_reference.new.txt) && echo VALUES_IDENTICAL
cp build/highs-1151/conformance_reference.new.txt tests/conformance/conformance_reference.txt
sha256sum tests/conformance/conformance_reference.txt
file tests/conformance/conformance_reference.txt   # must not say CRLF
```

Expected: `VALUES_IDENTICAL` (the five value lines are byte-identical to the tracked ones: labels `0×9,1×9,2×9`, medoids `4,13,22`, silhouette `0.96894972764334841`, davies_bouldin `0.038333333333333337`, dunn `11.5`); the printed sha256 equals what the tool printed. Put that hex into `kReferenceSha256`. If the values differ, STOP: the pipeline drifted from the tracked oracle on this machine and that is a finding for the run-log, not something to overwrite.

- [ ] **Step 5: Build and run the test**

```bash
cmake --build build/highs-1151 --target cpp_conformance && ctest --test-dir build/highs-1151 -R '^cpp_conformance$' --output-on-failure
```

Expected: passes with `All tests passed (10 assertions in 2 test cases)` (2 + 8). Then update `tests/floors.cmake`'s `DTWC_TEST_FLOOR_cpp_conformance` to `"10;2"` by hand (this task adds a test case; note the edit in the run-log — regeneration from logs happens at the exit gate).

- [ ] **Step 6: Write the negative script test**

`tests/integration/test_conformance_reference_read_only.cmake`:

```cmake
# Proves the conformance route cannot heal a missing reference: pointed at a
# scratch copy that has the data but no reference, the binary must fail and
# must not create one; pointed at a complete copy, it must pass.
# -DTEST_BINARY=<cpp_conformance exe> -DSOURCE_DIR=<tests/conformance> -DWORK_ROOT=<abs dir>
cmake_minimum_required(VERSION 3.26)
foreach(v TEST_BINARY SOURCE_DIR WORK_ROOT)
  if(NOT DEFINED ${v})
    message(FATAL_ERROR "missing -D${v}")
  endif()
endforeach()
if(NOT IS_ABSOLUTE "${WORK_ROOT}")
  message(FATAL_ERROR "WORK_ROOT must be absolute: ${WORK_ROOT}")
endif()
file(REMOVE_RECURSE "${WORK_ROOT}")
file(MAKE_DIRECTORY "${WORK_ROOT}/missing/data" "${WORK_ROOT}/complete/data")
file(COPY "${SOURCE_DIR}/data/conformance_series.csv" DESTINATION "${WORK_ROOT}/missing/data")
file(COPY "${SOURCE_DIR}/data/conformance_series.csv" DESTINATION "${WORK_ROOT}/complete/data")
file(COPY "${SOURCE_DIR}/conformance_reference.txt" DESTINATION "${WORK_ROOT}/complete")

set(ENV{DTWC_CONFORMANCE_DIR} "${WORK_ROOT}/missing")
execute_process(COMMAND "${TEST_BINARY}" RESULT_VARIABLE missing_rc OUTPUT_VARIABLE missing_out ERROR_VARIABLE missing_err)
if(missing_rc EQUAL 0)
  message(FATAL_ERROR "conformance passed WITHOUT a reference file:\n${missing_out}${missing_err}")
endif()
if(EXISTS "${WORK_ROOT}/missing/conformance_reference.txt")
  message(FATAL_ERROR "conformance route regenerated the missing reference")
endif()

set(ENV{DTWC_CONFORMANCE_DIR} "${WORK_ROOT}/complete")
execute_process(COMMAND "${TEST_BINARY}" RESULT_VARIABLE complete_rc OUTPUT_VARIABLE complete_out ERROR_VARIABLE complete_err)
if(NOT complete_rc EQUAL 0)
  message(FATAL_ERROR "conformance failed on a complete copy:\n${complete_out}${complete_err}")
endif()
file(SHA256 "${SOURCE_DIR}/conformance_reference.txt" tracked_sha)
file(SHA256 "${WORK_ROOT}/complete/conformance_reference.txt" copy_sha)
if(NOT tracked_sha STREQUAL copy_sha)
  message(FATAL_ERROR "conformance route modified the reference copy")
endif()

message(STATUS "CONFORMANCE_READ_ONLY subject=cpp_conformance runs=2/2 missing_exit=${missing_rc} nonzero=1 created=0 complete_exit=0 unchanged=1 skips=0")
```

Register it in `tests/CMakeLists.txt` after the conformance target exists (pass 2 creates `cpp_conformance`; place this after the pass-2 loop):

```cmake
dtwc_add_script_test(NAME test_conformance_reference_read_only
    SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/integration/test_conformance_reference_read_only.cmake"
    ARGS "-DTEST_BINARY=$<TARGET_FILE:cpp_conformance>"
         "-DSOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR}/conformance"
         "-DWORK_ROOT=${CMAKE_CURRENT_BINARY_DIR}/conformance-read-only"
    MARKER "CONFORMANCE_READ_ONLY subject=cpp_conformance runs=2/2 missing_exit=[0-9]+ nonzero=1 created=0 complete_exit=0 unchanged=1 skips=0"
    LABELS integration conformance SERIAL TIMEOUT 120)
```

- [ ] **Step 7: Run both, then the F17 test (it pins the CSV/config hashes, not the reference — must stay green)**

```bash
cmake -S . -B build/highs-1151 && cmake --build build/highs-1151 --parallel
ctest --test-dir build/highs-1151 -R 'conformance|test_cli_resume_state' --output-on-failure
```

Expected: 3/3 passed; the read-only marker printed with `nonzero=1 created=0`.

- [ ] **Step 8: Python and MATLAB routes still read the same file**

```bash
uv run --no-project python -c "import pathlib; t=pathlib.Path('tests/conformance/conformance_reference.txt').read_text(); print([l for l in t.splitlines() if not l.startswith('#')])"
```

Expected: the five value lines, unchanged. (`tests/conformance/test_conformance.py:32` and `test_conformance.m:35` skip `#` lines.)

- [ ] **Step 9: CHANGELOG + LESSONS, commit**

CHANGELOG (Unreleased → Changed): `- Conformance reference is read-only and SHA-256 pinned in the C++ route; regeneration moved to the explicit tool regen_conformance_reference (T-15).`
LESSONS: `- A test that regenerates its oracle when the oracle is missing proves nothing; the reference must be tracked, read-only in every test route, and hash-pinned in code so replacing it is a reviewed change (cpp_conformance, 2026-09-07).`

```bash
git add tests/conformance tests/integration/test_conformance_reference_read_only.cmake tests/CMakeLists.txt tests/floors.cmake CHANGELOG.md .claude/LESSONS.md
git commit -m "test(conformance): reference is tracked, read-only and hash-pinned; explicit regen tool (T-15)"
```

---

### Task 3: Build hygiene slice (B-12, part of B-13)

**Files:**

- Create: `cmake/DtwcOptional.cmake`
- Modify: `CMakeLists.txt` (options block ~29–50; `include(cmake/FindGUROBI.cmake)` at ~229; CUDA "nvcc not found" at ~208–211), `cmake/Dependencies.cmake` (152, 299, 328, 346–353, 405–411, 442–447), `cmake/FindGUROBI.cmake` (168–169), `cmake/StaticAnalyzers.cmake` (101–108), `cmake/InterproceduralOptimization.cmake`, `benchmarks/CMakeLists.txt`

**Interfaces:**

- Produces: `dtwc_disable_optional(<NAME> <reason>)` macro; `dtwc_add_benchmark(NAME <t> [OWN_MAIN] [PLAIN] [REQUIRES <var>...])` (benchmarks-local function, used again in Task 11); `DTWC_ENABLE_LLFIO` declared in the root option block.

- [ ] **Step 1: The degradation macro**

`cmake/DtwcOptional.cmake`:

```cmake
include_guard(GLOBAL)

# dtwc_disable_optional(<NAME> <reason>)
# The one way a requested optional dependency that cannot be delivered is
# turned off: one WARNING shape, the option forced OFF in the calling scope and
# (from inside a function) in the parent scope that owns it. Never silent.
macro(dtwc_disable_optional _name _reason)
  message(WARNING "  ${_name}: ${_reason}\n  DTWC_ENABLE_${_name} forced OFF for this configure.")
  set(DTWC_ENABLE_${_name} OFF)
  if(CMAKE_CURRENT_FUNCTION)
    set(DTWC_ENABLE_${_name} OFF PARENT_SCOPE)
  endif()
endmacro()
```

Include it in `CMakeLists.txt` right after `include(cmake/StandardProjectSettings.cmake)`.

- [ ] **Step 2: Apply the macro at the four degradation sites**

In `cmake/Dependencies.cmake`:
- lines 346–353 (Arrow, Windows+Clang): replace the `message(WARNING ...)` + `set(DTWC_ENABLE_ARROW OFF PARENT_SCOPE)` with `dtwc_disable_optional(ARROW "Arrow CPM build is not supported with Windows+Clang (ExternalProject flag quoting). Use conda arrow-cpp (find_package), MSVC, or Linux.")`
- lines 408–411 (Arrow CPM build failed): `dtwc_disable_optional(ARROW "Arrow CPM build failed. Install: conda install -c conda-forge arrow-cpp")`
- lines 442–447 (MPI not found): `dtwc_disable_optional(MPI "MPI requested but not found. On Windows install the MS-MPI runtime AND SDK (https://learn.microsoft.com/en-us/message-passing-interface/microsoft-mpi)")`
- lines 301–303 (llfio requested but not found): `dtwc_disable_optional(LLFIO "requested but NOT FOUND; memory-mapped distance matrices disabled (in-memory store only)")`

In `CMakeLists.txt` ~208–211 (CUDA): `dtwc_disable_optional(CUDA "requested but nvcc not found. Set CUDA_PATH or pass -DCMAKE_CUDA_COMPILER=<path-to-nvcc>")`.

- [ ] **Step 3: Option placement and dead variables**

- Move `option(DTWC_ENABLE_LLFIO "Enable llfio memory-mapped distance matrices" ON)` from `cmake/Dependencies.cmake:152` to `CMakeLists.txt`, directly after the `DTWC_ENABLE_YAML` option; keep the explanatory comment above it, trimmed to: `# llfio: memory-mapped distance matrices. OFF or absent => DTWC_HAS_MMAP is never defined and stores stay in memory.`
- Delete `set(DTWC_HAS_MMAP TRUE)` (`Dependencies.cmake:299`) and both `set(DTWC_HAS_PARQUET_LIB TRUE PARENT_SCOPE)` lines (328, 405). Verify: `grep -rn "DTWC_HAS_PARQUET_LIB\|DTWC_HAS_MMAP TRUE" cmake CMakeLists.txt dtwc tests python bindings` → no hits.
- Delete `macro(dtwc_enable_include_what_you_use) ... endmacro()` from `cmake/StaticAnalyzers.cmake` (zero callers: `grep -rn include_what_you_use --include=*.cmake --include=CMakeLists.txt .` → none after the edit).
- Delete the two "legacy support" lines in `cmake/FindGUROBI.cmake` (`GUROBI_INCLUDE_DIRS`, `GUROBI_LIBRARIES`; zero readers).

- [ ] **Step 4: Probe Gurobi only when enabled; name the IPO escape hatch**

`CMakeLists.txt` ~229:

```cmake
if(DTWC_ENABLE_GUROBI)
  include(cmake/FindGUROBI.cmake)   # probes GUROBI_HOME and C:/gurobi*/win64; only worth it when asked for
endif()
```

`dtwc/mip/CMakeLists.txt:60` already guards on `DTWC_ENABLE_GUROBI AND TARGET Gurobi::GurobiCXX`, so nothing else changes.

`cmake/InterproceduralOptimization.cmake`:

```cmake
macro(dtwc_enable_ipo)
  include(CheckIPOSupported)
  check_ipo_supported(RESULT result OUTPUT output)
  if(result)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)
  else()
    message(SEND_ERROR
      "IPO/LTO is not supported by this toolchain: ${output}\n"
      "  Configure with -Ddtwc_ENABLE_IPO=OFF to build without it.")
  endif()
endmacro()
```

- [ ] **Step 5: Benchmark registration helper and backend guards**

`benchmarks/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.26)

# dtwc_add_benchmark(NAME <target> [OWN_MAIN] [PLAIN] [REQUIRES <cmake-var>...])
#   Source is <target>.cpp. OWN_MAIN: the source defines main() (links
#   benchmark::benchmark only). PLAIN: no Google Benchmark at all. REQUIRES:
#   every named variable must be true, else the target is not added -- a backend
#   benchmark never exists on a build without its backend.
function(dtwc_add_benchmark)
  cmake_parse_arguments(ARG "OWN_MAIN;PLAIN" "NAME" "REQUIRES" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS OR NOT ARG_NAME)
    message(FATAL_ERROR "dtwc_add_benchmark: bad arguments ${ARGN}")
  endif()
  foreach(req IN LISTS ARG_REQUIRES)
    if(NOT ${req})
      message(STATUS "benchmark ${ARG_NAME} not built (${req} is off)")
      return()
    endif()
  endforeach()
  add_executable(${ARG_NAME} ${ARG_NAME}.cpp)
  target_link_libraries(${ARG_NAME} PRIVATE dtwc++ project_options)
  if(NOT ARG_PLAIN)
    target_link_libraries(${ARG_NAME} PRIVATE benchmark::benchmark)
    if(NOT ARG_OWN_MAIN)
      target_link_libraries(${ARG_NAME} PRIVATE benchmark::benchmark_main)
    endif()
  endif()
endfunction()

dtwc_add_benchmark(NAME bench_dtw_baseline)
dtwc_add_benchmark(NAME bench_mmap_access)
dtwc_add_benchmark(NAME bench_f32_vs_f64)
dtwc_add_benchmark(NAME bench_openmp_schedule OWN_MAIN)
dtwc_add_benchmark(NAME bench_cuda_dtw OWN_MAIN REQUIRES DTWC_ENABLE_CUDA CMAKE_CUDA_COMPILER)
dtwc_add_benchmark(NAME bench_metal_dtw OWN_MAIN REQUIRES DTWC_HAS_METAL_DETECTED)
dtwc_add_benchmark(NAME bench_mpi_dtw PLAIN REQUIRES DTWC_ENABLE_MPI)

# Legacy UCR driver (source name differs from the target name).
add_executable(benchmark_UCR UCR_dtwc.cpp)
target_link_libraries(benchmark_UCR PRIVATE dtwc++ project_options)
```

(`project_options` was not linked before; it carries the C++20 feature and the project FP flags, so benchmarks now measure the library's own flag model. CHANGELOG line below.)

- [ ] **Step 6: Configure the four matrices plus a benchmark-ON configure**

```bash
for d in build/highs-1151 build/nollfio build/arrow-pyarrow-23 build/msvc-debug; do cmake -S . -B $d 2>&1 | grep -E "Warning|Error|forced OFF|not built" ; echo "$d exit=${PIPESTATUS[0]}"; done
cmake -S . -B build/w0-bench -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release -DDTWC_BUILD_BENCHMARK=ON -DDTWC_ENABLE_GUROBI=OFF 2>&1 | grep -E "benchmark .* not built|Error"; echo "exit=${PIPESTATUS[0]}"
cmake --build build/w0-bench --parallel
cmake -S . -B build/w0-nogurobi -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DDTWC_ENABLE_GUROBI=OFF --trace-expand 2>&1 | grep -c "FindGUROBI.cmake("
```

Expected: all four configures exit 0 with no new warnings; the benchmark configure prints `benchmark bench_cuda_dtw not built (DTWC_ENABLE_CUDA is off)`, the same for `bench_metal_dtw` and `bench_mpi_dtw`, and builds; the trace count is 0 (no command inside `FindGUROBI.cmake` executed when Gurobi is disabled; the root `include(...)` line itself does not match this pattern). Remove `build/w0-nogurobi` afterwards (you created it): `rm -rf build/w0-nogurobi`.

- [ ] **Step 7: Commit**

CHANGELOG (Unreleased → Changed): `- CMake: DTWC_ENABLE_LLFIO is declared with the other options in the root CMakeLists; Gurobi is probed only when DTWC_ENABLE_GUROBI=ON; backend benchmarks (CUDA/Metal/MPI) are added only when their backend is enabled; benchmarks link project_options; dead DTWC_HAS_MMAP/DTWC_HAS_PARQUET_LIB writes, the IWYU macro and legacy Gurobi variables removed; IPO failure names -Ddtwc_ENABLE_IPO=OFF (B-12).`

```bash
git add cmake/DtwcOptional.cmake CMakeLists.txt cmake/Dependencies.cmake cmake/FindGUROBI.cmake cmake/StaticAnalyzers.cmake cmake/InterproceduralOptimization.cmake benchmarks/CMakeLists.txt CHANGELOG.md
git commit -m "build: hygiene slice -- options beside siblings, one degradation path, guarded probes and benchmarks (B-12)"
```

---

### Task 3b: One CMake option prefix (B-14, approved 2026-09-07)

**Files:**

- Modify: `cmake/ProjectOptions.cmake` (definitions 21–33, `mark_as_advanced` 37–49, readers 54, 65, 69, 79–83, 85, 87, 96, 102–103, 106–107, 111), `cmake/InterproceduralOptimization.cmake` (the escape-hatch message written in Task 3), `README.md` (line 66, the sanitizer example)

**Interfaces:**

- Produces: options `DTWC_ENABLE_IPO`, `DTWC_ENABLE_COMPILER_WARNINGS`, `DTWC_WARNINGS_AS_ERRORS`, `DTWC_ENABLE_SANITIZER_ADDRESS`, `DTWC_ENABLE_SANITIZER_LEAK`, `DTWC_ENABLE_SANITIZER_UNDEFINED`, `DTWC_ENABLE_SANITIZER_THREAD`, `DTWC_ENABLE_SANITIZER_MEMORY`, `DTWC_ENABLE_UNITY_BUILD`, `DTWC_ENABLE_CLANG_TIDY`, `DTWC_ENABLE_CPPCHECK`, `DTWC_ENABLE_PCH`, `DTWC_ENABLE_CACHE`. The old `dtwc_*` spellings are honoured with a `DEPRECATION` message for the 2.0.x releases. Tasks 8 and 12 use the new names.

- [ ] **Step 1: Observe the defect (new name silently ignored today)**

```bash
cmake -S . -B build/w0-b14 -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DDTWC_BUILD_TESTING=OFF -DDTWC_ENABLE_HIGHS=OFF -DDTWC_ENABLE_GUROBI=OFF -DDTWC_ENABLE_LLFIO=OFF -DDTWC_ENABLE_IPO=OFF 2>&1 | grep -A2 "not used by the project"
cmake -LA -N build/w0-b14 | grep -i "ENABLE_IPO"
```

Expected today: CMake warns `Manually-specified variables were not used by the project: DTWC_ENABLE_IPO`, and the cache shows `dtwc_ENABLE_IPO:BOOL=ON` — the user's request was dropped.

- [ ] **Step 2: Rename with a legacy shim**

In `cmake/ProjectOptions.cmake`, above `macro(dtwc_setup_options)`:

```cmake
# B-14: one option prefix (DTWC_). The pre-2.0 dtwc_* spellings are honoured
# with a deprecation message for the 2.0.x releases: on the first configure
# their value seeds the DTWC_* cache entry, which then owns the setting.
function(dtwc_option name doc default)
  string(REGEX REPLACE "^DTWC_" "dtwc_" legacy "${name}")
  if(DEFINED ${legacy} AND NOT DEFINED CACHE{${name}})
    message(DEPRECATION
      "${legacy} is deprecated; use -D${name}=${${legacy}} (honoured for 2.0.x, removed after)")
    set(default "${${legacy}}")
  endif()
  option(${name} "${doc}" "${default}")
endfunction()
```

Then, inside `dtwc_setup_options`, replace the 13 `option(dtwc_… )` lines with `dtwc_option(DTWC_… "<same doc>" <same default>)` and rename every `dtwc_ENABLE_*` / `dtwc_WARNINGS_AS_ERRORS` reference in the file (the `mark_as_advanced` list and the readers at 54, 65, 69, 79–83, 85, 87, 96, 102–103, 106–107, 111) to the `DTWC_` spelling. Function/macro names (`dtwc_setup_options`, `dtwc_enable_sanitizers`, `dtwc_enable_clang_tidy`, `dtwc_enable_cppcheck`, `dtwc_enable_cache`, `dtwc_options`, `dtwc_warnings`) are not options and stay.

`cmake/InterproceduralOptimization.cmake`: the message reads `Configure with -DDTWC_ENABLE_IPO=OFF to build without it.`

`README.md:66`: the example becomes `-DDTWC_ENABLE_SANITIZER_ADDRESS=ON` (keep the sentence otherwise).

- [ ] **Step 3: Verify both spellings**

```bash
rm -rf build/w0-b14
cmake -S . -B build/w0-b14 -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DDTWC_BUILD_TESTING=OFF -DDTWC_ENABLE_HIGHS=OFF -DDTWC_ENABLE_GUROBI=OFF -DDTWC_ENABLE_LLFIO=OFF -DDTWC_ENABLE_IPO=OFF 2>&1 | grep -E "not used|DEPRECATION"; cmake -LA -N build/w0-b14 | grep "ENABLE_IPO"
rm -rf build/w0-b14
cmake -S . -B build/w0-b14 -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DDTWC_BUILD_TESTING=OFF -DDTWC_ENABLE_HIGHS=OFF -DDTWC_ENABLE_GUROBI=OFF -DDTWC_ENABLE_LLFIO=OFF -Ddtwc_ENABLE_IPO=OFF 2>&1 | grep -E "DEPRECATION|dtwc_ENABLE_IPO"; cmake -LA -N build/w0-b14 | grep "ENABLE_IPO"
rm -rf build/w0-b14
git grep -n -E "dtwc_(ENABLE_[A-Z_]+|WARNINGS_AS_ERRORS)" -- ':!.claude/baselines'
```

Expected: first configure — no "not used" warning, cache `DTWC_ENABLE_IPO:BOOL=OFF`; second configure — one `DEPRECATION` message naming `dtwc_ENABLE_IPO`, cache `DTWC_ENABLE_IPO:BOOL=OFF`; the grep prints only the `dtwc_option` shim lines in `ProjectOptions.cmake`. Reconfigure the four matrices once (`cmake -S . -B <dir>`): each prints the deprecation message for the `dtwc_*` entries already in its cache and carries on; a second reconfigure is silent.

- [ ] **Step 4: Commit**

CHANGELOG (Unreleased → Changed): `- CMake: maintainer options use the DTWC_ prefix (DTWC_ENABLE_IPO, DTWC_ENABLE_SANITIZER_*, DTWC_WARNINGS_AS_ERRORS, DTWC_ENABLE_CLANG_TIDY, …); the old dtwc_* spellings are honoured with a deprecation message for 2.0.x (B-14).`

```bash
git add cmake/ProjectOptions.cmake cmake/InterproceduralOptimization.cmake README.md CHANGELOG.md
git commit -m "build: one option prefix DTWC_; dtwc_* spellings honoured with a deprecation message (B-14)"
```

---

### Task 4: Preset twins and a table-driven F16 guard (B-12 O15, B-13 D15)

**Files:**

- Modify: `CMakePresets.json`, `tests/CMakeLists.txt` (F16 block, today lines 399–617), `tests/unit/test_supply_chain_pinning.cpp` (lines 200–400)

**Interfaces:**

- Produces: build presets `msvc`, `gcc-linux`; test presets `clang-win-debug`, `msvc`, `gcc-linux`; F16 inventory 6/5/5.

- [ ] **Step 1: Extend the C++ pins first (they fail until the presets change)**

In `tests/unit/test_supply_chain_pinning.cpp`:
- `"lhs"\s*:\s*"\$\{hostSystemName\}"` count `== 10` → `== 15`; `"type"\s*:\s*"equals"` `== 10` → `== 15`; `"rhs"\s*:\s*"Windows"` `== 6` → `== 9`; `"rhs"\s*:\s*"Linux"` `== 1` → `== 3`; Darwin stays `== 3`.
- After the existing `build_block` checks for `clang-win`, `clang-win-debug`, `clang-macos`, add the same `regex_count(build_block, ... "configurePreset"\s*:\s*"msvc"\s*[,}] ...) == 1` and `... "gcc-linux" ... == 1`.
- After the `test_block` checks for `clang-win` and `clang-macos`, add `clang-win-debug`, `msvc`, `gcc-linux` each `== 1`.
- Where the test locates `build_win_pos`/`test_win_pos` blocks and asserts host conditions (lines ~340–370), add positions and `has_host_condition` checks for the new entries: `build_msvc` → Windows, `build_gcc` → Linux, `test_debug` → Windows, `test_msvc` → Windows, `test_gcc` → Linux, following the existing `find`/`substr`/`REQUIRE(... != npos)` pattern for each.

Run: `cmake --build build/highs-1151 --target test_supply_chain_pinning && build/highs-1151/bin/test_supply_chain_pinning`
Expected: FAILED (the new counts do not match yet).

- [ ] **Step 2: Add the presets**

In `CMakePresets.json`, append to `buildPresets` (after `clang-macos`):

```json
    {
      "name": "msvc",
      "configurePreset": "msvc",
      "condition": { "type": "equals", "lhs": "${hostSystemName}", "rhs": "Windows" },
      "configuration": "Release"
    },
    {
      "name": "gcc-linux",
      "configurePreset": "gcc-linux",
      "condition": { "type": "equals", "lhs": "${hostSystemName}", "rhs": "Linux" },
      "configuration": "Release"
    }
```

and to `testPresets` (after `clang-macos`):

```json
    {
      "name": "clang-win-debug",
      "configurePreset": "clang-win-debug",
      "configuration": "Debug",
      "condition": { "type": "equals", "lhs": "${hostSystemName}", "rhs": "Windows" },
      "output": { "outputOnFailure": true },
      "execution": { "jobs": 8 }
    },
    {
      "name": "msvc",
      "configurePreset": "msvc",
      "configuration": "Release",
      "condition": { "type": "equals", "lhs": "${hostSystemName}", "rhs": "Windows" },
      "output": { "outputOnFailure": true }
    },
    {
      "name": "gcc-linux",
      "configurePreset": "gcc-linux",
      "configuration": "Release",
      "condition": { "type": "equals", "lhs": "${hostSystemName}", "rhs": "Linux" },
      "output": { "outputOnFailure": true }
    }
```

Keep the existing multi-line `condition` formatting style if the C++ regexes require it (they allow `\s*`, so the compact form above is accepted; match the file's existing style anyway for consistency: expand each `condition` to the four-line form used elsewhere in the file).

- [ ] **Step 3: Rewrite the F16 configure-time guard as a table**

Replace `tests/CMakeLists.txt` lines 399–617 (from `set(_f16_presets_file ...)` to the pyproject floor check) with:

```cmake
# F16: the tracked presets stay truthful and host-specific. CMake's JSON parser
# is the configure-time arbiter; test_supply_chain_pinning re-pins the same
# scalars at runtime and prints the CTest marker.
set(_f16_presets_file "${CMAKE_SOURCE_DIR}/CMakePresets.json")
file(READ "${_f16_presets_file}" _f16_presets_json)
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${_f16_presets_file}" "${CMAKE_SOURCE_DIR}/pyproject.toml")

function(_f16_fail what)
    message(FATAL_ERROR "F16 preset drift: ${what}")
endfunction()

function(_f16_find_preset out_index collection expected_name)
    string(JSON _count LENGTH "${_f16_presets_json}" "${collection}")
    math(EXPR _last "${_count} - 1")
    set(_found -1)
    foreach(_index RANGE 0 ${_last})
        string(JSON _name GET "${_f16_presets_json}" "${collection}" ${_index} name)
        if(_name STREQUAL expected_name)
            if(NOT _found EQUAL -1)
                _f16_fail("duplicate ${collection} preset ${expected_name}")
            endif()
            set(_found ${_index})
        endif()
    endforeach()
    if(_found EQUAL -1)
        _f16_fail("missing ${collection} preset ${expected_name}")
    endif()
    set(${out_index} ${_found} PARENT_SCOPE)
endfunction()

# Scalars.
string(JSON _f16_schema GET "${_f16_presets_json}" version)
string(JSON _f16_major GET "${_f16_presets_json}" cmakeMinimumRequired major)
string(JSON _f16_minor GET "${_f16_presets_json}" cmakeMinimumRequired minor)
string(JSON _f16_patch GET "${_f16_presets_json}" cmakeMinimumRequired patch)
file(STRINGS "${CMAKE_SOURCE_DIR}/CMakeLists.txt" _f16_root_first_line LIMIT_COUNT 1)
if(NOT _f16_schema EQUAL 6)
    _f16_fail("schema ${_f16_schema}, expected 6")
endif()
if(NOT "${_f16_major}.${_f16_minor}.${_f16_patch}" VERSION_EQUAL "3.26.0"
        OR NOT _f16_root_first_line STREQUAL "cmake_minimum_required(VERSION 3.26)")
    _f16_fail("CMake floor: presets ${_f16_major}.${_f16_minor}.${_f16_patch}, root '${_f16_root_first_line}', expected 3.26.0")
endif()
string(REGEX MATCH "[A-Za-z]:/" _f16_drive_path "${_f16_presets_json}")
if(_f16_drive_path)
    _f16_fail("a Windows drive path is encoded in the tracked presets")
endif()
file(READ "${CMAKE_SOURCE_DIR}/pyproject.toml" _f16_pyproject)
string(REGEX MATCHALL "(^|[\r\n])[ \t]*cmake[.]version[ \t]*=[ \t]*\\\">=3[.]26\\\"" _f16_pyproject_floor "${_f16_pyproject}")
list(LENGTH _f16_pyproject_floor _f16_pyproject_floor_count)
if(NOT _f16_pyproject_floor_count EQUAL 1)
    _f16_fail("pyproject cmake.version floor: expected exactly one >=3.26 scalar")
endif()

# Inventory: exactly these presets, nothing more.
foreach(_row IN ITEMS "configurePresets;6" "buildPresets;5" "testPresets;5")
    list(GET _row 0 _collection)
    list(GET _row 1 _expected)
    string(JSON _n LENGTH "${_f16_presets_json}" "${_collection}")
    if(NOT _n EQUAL _expected)
        _f16_fail("${_collection} count ${_n}, expected ${_expected}")
    endif()
endforeach()

# The hidden default: shared build dir, no compiler, no toolchain.
_f16_find_preset(_f16_default configurePresets default)
string(JSON _f16_default_hidden GET "${_f16_presets_json}" configurePresets ${_f16_default} hidden)
string(JSON _f16_default_binary GET "${_f16_presets_json}" configurePresets ${_f16_default} binaryDir)
# string(JSON ... ERROR_VARIABLE e GET ...) sets e to the literal "NOTFOUND" on
# SUCCESS and to an error message when the key is absent.
foreach(_key IN ITEMS CMAKE_CXX_COMPILER CMAKE_C_COMPILER CMAKE_TOOLCHAIN_FILE)
    string(JSON _v ERROR_VARIABLE _err GET "${_f16_presets_json}" configurePresets ${_f16_default} cacheVariables ${_key})
    if(_err STREQUAL "NOTFOUND")
        _f16_fail("hidden default must not set ${_key}")
    endif()
endforeach()
if(NOT _f16_default_hidden OR NOT _f16_default_binary STREQUAL "\${sourceDir}/build")
    _f16_fail("hidden default must be hidden and keep \${sourceDir}/build")
endif()

# Every visible preset: host condition, parent (inherits / configurePreset),
# and the compilers the platform presets are allowed to name.
#   name,collection,host,parent,cxx-compiler-or-'-',c-compiler-or-'-'
# (comma-separated: a ';' inside a set() argument would split the row into
# separate list elements.)
set(_f16_expected
    "clang-win,configurePresets,Windows,default,clang++,-"
    "clang-win-debug,configurePresets,Windows,clang-win,-,-"
    "msvc,configurePresets,Windows,default,-,-"
    "gcc-linux,configurePresets,Linux,default,g++,-"
    "clang-macos,configurePresets,Darwin,default,/usr/bin/clang++,/usr/bin/clang"
    "clang-win,buildPresets,Windows,clang-win,-,-"
    "clang-win-debug,buildPresets,Windows,clang-win-debug,-,-"
    "msvc,buildPresets,Windows,msvc,-,-"
    "gcc-linux,buildPresets,Linux,gcc-linux,-,-"
    "clang-macos,buildPresets,Darwin,clang-macos,-,-"
    "clang-win,testPresets,Windows,clang-win,-,-"
    "clang-win-debug,testPresets,Windows,clang-win-debug,-,-"
    "msvc,testPresets,Windows,msvc,-,-"
    "gcc-linux,testPresets,Linux,gcc-linux,-,-"
    "clang-macos,testPresets,Darwin,clang-macos,-,-")
foreach(_row IN LISTS _f16_expected)
    string(REPLACE "," ";" _row "${_row}")
    list(GET _row 0 _name)
    list(GET _row 1 _collection)
    list(GET _row 2 _host)
    list(GET _row 3 _parent)
    list(GET _row 4 _cxx)
    list(GET _row 5 _c)
    _f16_find_preset(_i "${_collection}" "${_name}")
    string(JSON _type GET "${_f16_presets_json}" "${_collection}" ${_i} condition type)
    string(JSON _lhs GET "${_f16_presets_json}" "${_collection}" ${_i} condition lhs)
    string(JSON _rhs GET "${_f16_presets_json}" "${_collection}" ${_i} condition rhs)
    if(NOT _type STREQUAL "equals" OR NOT _lhs STREQUAL "\${hostSystemName}" OR NOT _rhs STREQUAL _host)
        _f16_fail("${_collection}/${_name} condition ${_type} ${_lhs} ${_rhs}, expected equals hostSystemName ${_host}")
    endif()
    if(_collection STREQUAL "configurePresets")
        set(_parent_key inherits)
    else()
        set(_parent_key configurePreset)
    endif()
    string(JSON _got_parent GET "${_f16_presets_json}" "${_collection}" ${_i} ${_parent_key})
    if(NOT _got_parent STREQUAL _parent)
        _f16_fail("${_collection}/${_name} ${_parent_key} ${_got_parent}, expected ${_parent}")
    endif()
    foreach(_pair IN ITEMS "CMAKE_CXX_COMPILER,${_cxx}" "CMAKE_C_COMPILER,${_c}")
        string(REPLACE "," ";" _pair "${_pair}")
        list(GET _pair 0 _key)
        list(GET _pair 1 _want)
        string(JSON _got ERROR_VARIABLE _err GET "${_f16_presets_json}" "${_collection}" ${_i} cacheVariables ${_key})
        if(_want STREQUAL "-")
            if(_err STREQUAL "NOTFOUND")   # GET succeeded: the key exists
                _f16_fail("${_collection}/${_name} must not set ${_key}")
            endif()
        elseif(NOT _got STREQUAL _want)
            _f16_fail("${_collection}/${_name} ${_key} '${_got}', expected '${_want}'")
        endif()
    endforeach()
endforeach()
```

Then rerun the test binary from Step 1 and the configure:

```bash
cmake -S . -B build/highs-1151 2>&1 | grep -E "F16|Error" ; echo "configure exit=${PIPESTATUS[0]}"
cmake --build build/highs-1151 --target test_supply_chain_pinning && ctest --test-dir build/highs-1151 -R '^test_supply_chain_pinning$' --output-on-failure
```

Expected: configure exit 0 with no F16 message; the test passes with its marker. Negative control: temporarily change `"msvc,buildPresets,Windows,msvc,-,-"` to `"msvc,buildPresets,Linux,msvc,-,-"`, reconfigure, expect `F16 preset drift: buildPresets/msvc condition equals ${hostSystemName} Windows, expected equals hostSystemName Linux`; revert.

- [ ] **Step 4: Use the new presets once each where the host allows**

```bash
cmake --preset msvc -DDTWC_BUILD_TESTING=OFF && cmake --build --preset msvc --target dtwc_cl
```

Expected: exit 0 (the `msvc` configure preset uses the `Visual Studio 17 2022` generator: if this machine only has VS 18, record the generator mismatch in the run-log as an existing preset defect, DECIDE for Volkan, and do not change the generator in this task).

- [ ] **Step 5: Commit**

CHANGELOG (Unreleased → Added): `- CMake presets: build/test twins for msvc and gcc-linux and a test preset for clang-win-debug; the F16 preset guard is table-driven.`

```bash
git add CMakePresets.json tests/CMakeLists.txt tests/unit/test_supply_chain_pinning.cpp CHANGELOG.md
git commit -m "build: build/test preset twins for msvc and gcc-linux; table-driven F16 guard (B-12 O15, B-13 D15)"
```

The floor for `test_supply_chain_pinning` is the explicit 35/4 (unchanged; the added CHECKs only raise the count).

---

### Task 5: Generated public header file set (C-24)

**Files:**

- Modify: `dtwc/CMakeLists.txt` lines 10–82 (`target_sources`)

- [ ] **Step 1: Replace the hand-written PUBLIC list**

```cmake
target_sources(dtwc++
  PRIVATE
  Problem.cpp
  Problem_IO.cpp
  api.cpp
  checkpoint.cpp
  env.cpp
  initialisation.cpp
  scores.cpp
  system_memory.cpp
  core/dtw.cpp
  core/dtw_dispatch.cpp
  algorithms/fast_pam.cpp
  algorithms/fast_clara.cpp
  algorithms/one_batch_pam.cpp
  algorithms/barycenter.cpp
  algorithms/hierarchical.cpp
  algorithms/clarans.cpp
  algorithms/tadpole.cpp
  core/pruned_distance_matrix.cpp
  io/arrow_c_data.cpp
  extern/nanoarrow/nanoarrow.c
)

# Public headers as a generated file set: the include directory carries the
# contract, this inventory exists for IDEs and install() and cannot go stale.
# extern/ (vendored nanoarrow) stays PRIVATE.
file(GLOB_RECURSE _dtwc_public_headers CONFIGURE_DEPENDS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
     "${CMAKE_CURRENT_SOURCE_DIR}/*.hpp" "${CMAKE_CURRENT_SOURCE_DIR}/*.cuh")
list(FILTER _dtwc_public_headers EXCLUDE REGEX "^extern/")
list(LENGTH _dtwc_public_headers _dtwc_public_header_count)
if(_dtwc_public_header_count LESS 100)
  message(FATAL_ERROR "dtwc++ public header glob found only ${_dtwc_public_header_count} files; the glob root is wrong")
endif()
target_sources(dtwc++ PUBLIC FILE_SET HEADERS BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}" FILES ${_dtwc_public_headers})
message(STATUS "dtwc++ public headers: ${_dtwc_public_header_count}")
```

Keep `target_include_directories(dtwc++ PUBLIC .)` (the documented contract) and the PRIVATE `extern` include.

- [ ] **Step 2: Configure and build everything on clang and MSVC**

```bash
cmake -S . -B build/highs-1151 2>&1 | grep "public headers"
cmake --build build/highs-1151 --parallel && cmake --build build/msvc-debug --config Debug --parallel
ctest --test-dir build/highs-1151 -R 'unit_test_problem_encapsulation|unit_test_DataLoader|cpp_conformance' --output-on-failure | tail -4
```

Expected: `dtwc++ public headers: <n>` with n ≥ 100 (the tree holds about 115 `.hpp` plus `cuda/cuda_dtw.cuh`); both builds succeed; the three tests pass. If the Python binding is configured anywhere (`python/CMakeLists.txt` links `dtwc++`), it keeps compiling: run `cmake --build build/<python-dir>` if such a dir exists, else record "not exercised".

- [ ] **Step 3: Commit**

```bash
git add dtwc/CMakeLists.txt
git commit -m "build(core): public headers are a generated file set, not a hand-written list (C-24)"
```

---

### Task 6: The NaN contract survives a `-ffast-math` consumer (C-08)

**Files:**

- Modify: `dtwc/CMakeLists.txt` (lines 196–200), `dtwc/missing_utils.hpp` (after `#pragma once`), `tests/CMakeLists.txt` (explicit registration)
- Create: `tests/unit/test_fast_math_consumer.cpp`

**Interfaces:**

- Produces: `-fno-finite-math-only` as a PUBLIC usage requirement of `dtwc++` on GNU-frontend GCC/Clang; a compile-time refusal in `missing_utils.hpp` when `__FINITE_MATH_ONLY__` is set.

- [ ] **Step 1: Write the consumer test (fails to link/build until the flag is PUBLIC — it currently fails at compile time once Step 3's `#error` exists, and before that it compiles but `is_missing` may fold to false)**

`tests/unit/test_fast_math_consumer.cpp`:

```cpp
/**
 * @file test_fast_math_consumer.cpp
 * @brief Compiled with -ffast-math on purpose (tests/CMakeLists.txt): a consumer
 * that links dtwc++ must still see NaN as "missing". dtwc++ publishes
 * -fno-finite-math-only after the consumer's own options, and
 * missing_utils.hpp refuses to compile when finite-math-only is still on, so
 * this file compiling AND these assertions passing is the contract (C-08).
 */
#include <catch2/catch_test_macros.hpp>

#include "missing_utils.hpp"
#include "warping_missing_arow.hpp"

#include <cmath>
#include <limits>
#include <vector>

// The header above already refuses a finite-math-only TU; this is the same
// check stated from the consumer's side so the failure names this file.
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "this TU still has finite-math-only on: dtwc++ did not publish -fno-finite-math-only"
#endif

namespace {
/// A NaN the optimiser cannot constant-fold away.
double runtime_nan()
{
  volatile double zero = 0.0;
  return zero / zero;
}
} // namespace

TEST_CASE("a -ffast-math consumer still sees NaN as missing", "[fast_math][missing][c08]")
{
  const double nan = runtime_nan();
  REQUIRE(dtwc::is_missing(nan));
  REQUIRE(std::isnan(nan));
  REQUIRE_FALSE(dtwc::is_missing(1.0));

  std::vector<double> with_gap{ 0.0, 1.0, nan, 3.0 };
  std::vector<double> complete{ 0.0, 1.0, 2.0, 3.0 };
  // AROW: a missing step is restricted to the diagonal with zero local cost, so
  // the gapped series is at distance 0 from its complete twin; a series that
  // ignored the NaN would compare 2.0 against NaN and return NaN or garbage.
  const double d = dtwc::dtwAROW(with_gap, complete);
  REQUIRE(std::isfinite(d));
  REQUIRE(d == 0.0);
  REQUIRE(dtwc::dtwAROW(complete, complete) == 0.0);
}
```

Confirm the AROW expectation against the existing oracle before relying on it: `grep -n "matches standard DTW\|All NaN\|== 0" tests/unit/unit_test_arow_dtw.cpp | head` (property 1: no NaN ⇒ standard DTW ⇒ `dtwAROW(complete, complete) == 0`; for the gapped series run the existing binary's own value: if `dtwAROW({0,1,NaN,3},{0,1,2,3})` is not 0.0 under the standard build, replace `REQUIRE(d == 0.0)` with `REQUIRE(d == <that value>)` computed by a five-line probe program compiled WITHOUT fast-math, and record the value in the run-log).

- [ ] **Step 2: Register it under GNU-frontend compilers only**

In `tests/CMakeLists.txt`, before pass 2:

```cmake
# C-08: a consumer built with -ffast-math must still get honest NaN handling.
# Only GNU-frontend compilers take these flags; MSVC and clang-cl are excluded.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang" AND NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
    dtwc_add_test(NAME test_fast_math_consumer SOURCE ${_dtwc_test_source_test_fast_math_consumer}
        ASSERT_FLOOR 6 CASE_FLOOR 1 LABELS c08)
    target_compile_options(test_fast_math_consumer PRIVATE -ffast-math)
endif()
```

and make pass 2 skip it everywhere: add `if(_dtwc_test_name STREQUAL "test_fast_math_consumer") continue() endif()` at the top of the pass-2 loop body (it is registered above or intentionally absent).

- [ ] **Step 3: Refuse finite-math-only in the header**

`dtwc/missing_utils.hpp`, after `#pragma once`:

```cpp
// "NaN means missing" rests on std::isnan telling the truth. -ffinite-math-only
// (part of -ffast-math) lets the compiler assume no NaN exists and silently
// breaks the AROW/ZeroCost paths. dtwc++ publishes -fno-finite-math-only to its
// consumers (dtwc/CMakeLists.txt); a TU that still has it on cannot be correct,
// so refuse here instead of returning wrong distances.
#if defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__
#error "DTWC++ requires -fno-finite-math-only: do not compile dtwc headers with -ffast-math / -ffinite-math-only"
#endif
```

Update the file's `@details` block: replace the two "Build flags ..." bullets with `- dtwc++ publishes -fno-finite-math-only (PUBLIC) and this header refuses finite-math-only TUs, so std::isnan() is honest in the library and in every consumer.`

- [ ] **Step 4: Publish the flag**

`dtwc/CMakeLists.txt` lines 196–200 become:

```cmake
# NaN is the missing-value sentinel and is_missing() lives in headers, so every
# TU that includes dtwc headers needs honest std::isnan. PUBLIC: CMake appends a
# dependency's interface options after the consumer's own, so a consumer built
# with -ffast-math is corrected; missing_utils.hpp refuses the TU if it is not.
# GNU-frontend compilers only (MSVC and clang-cl do not take the flag).
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU" AND NOT CMAKE_CXX_COMPILER_FRONTEND_VARIANT STREQUAL "MSVC")
  target_compile_options(dtwc++ PUBLIC $<$<COMPILE_LANGUAGE:CXX>:-fno-finite-math-only>)
endif()
```

- [ ] **Step 5: Verify the compile line order and run**

```bash
cmake -S . -B build/highs-1151 && cmake --build build/highs-1151 --target test_fast_math_consumer -v 2>&1 | grep -o -- "-ffast-math.*-fno-finite-math-only" | head -1
ctest --test-dir build/highs-1151 -R '^test_fast_math_consumer$' --output-on-failure
```

Expected: the grep prints a fragment showing `-ffast-math` BEFORE `-fno-finite-math-only` on the same command line (later flag wins in GCC/Clang); the test passes with `All tests passed (6 assertions in 1 test case)`. Negative control: comment out the `target_compile_options(dtwc++ PUBLIC ...)` line, rebuild the test target, expect the `#error` from `missing_utils.hpp`; restore.

- [ ] **Step 6: Full MSVC build still configures (flag is not applied there)**

```bash
cmake -S . -B build/msvc-debug && cmake --build build/msvc-debug --config Debug --target unit_test_arow_dtw && ctest --test-dir build/msvc-debug -C Debug -R 'unit_test_arow_dtw' | tail -3
```

Expected: builds and passes; `test_fast_math_consumer` is not registered there (`ctest --test-dir build/msvc-debug -C Debug -N | grep fast_math` prints nothing).

- [ ] **Step 7: Commit**

CHANGELOG (Unreleased → Changed): `- dtwc++ publishes -fno-finite-math-only to consumers on GCC/Clang and missing_utils.hpp refuses to compile with finite-math-only on; a -ffast-math consumer test pins the AROW/NaN contract (C-08).`
LESSONS: `- A PRIVATE compile flag protects only the library's own TUs; when the semantics live in headers (is_missing/std::isnan) the flag must be a PUBLIC usage requirement AND the header must refuse the wrong mode, because consumers do not read CMake comments (C-08, 2026-09-07).`

```bash
git add dtwc/CMakeLists.txt dtwc/missing_utils.hpp tests/unit/test_fast_math_consumer.cpp tests/CMakeLists.txt CHANGELOG.md .claude/LESSONS.md
git commit -m "build(core): publish -fno-finite-math-only, refuse finite-math TUs, pin the -ffast-math consumer contract (C-08)"
```

---

### Task 7: Vendor `CPM.cmake` 0.42.1 (B-17)

**Files:**

- Modify: `cmake/CPM.cmake` (the 24-line downloader becomes the vendored file), `THIRD_PARTY_LICENSES.md`, `cmake/readme.md`

- [ ] **Step 1: Obtain the exact upstream file and verify the pin already in the tree**

```bash
curl -sSL -o build/CPM-0.42.1.cmake https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.42.1/CPM.cmake \
  || cp build/highs-1151/cmake/CPM_0.42.1.cmake build/CPM-0.42.1.cmake
sha256sum build/CPM-0.42.1.cmake
```

Expected: `f3a6dcc6a04ce9e7f51a127307fa4f699fb2bade357a8eb4c5b45df76e1dc6a5` (the hash the downloader pins today at `cmake/CPM.cmake:6`). Any other value: stop, do not vendor.

- [ ] **Step 2: Replace the downloader**

```bash
cp build/CPM-0.42.1.cmake cmake/CPM.cmake
head -5 cmake/CPM.cmake      # upstream header: "CPM.cmake - CMake's missing package manager" + MIT notice
grep -n "CPM_DOWNLOAD\|file(DOWNLOAD" cmake/CPM.cmake | head   # none: no configure-time download of CPM itself
```

- [ ] **Step 3: Record the licence and provenance**

Append to `THIRD_PARTY_LICENSES.md` after the HiGHS block:

```markdown
The build system vendors CPM.cmake 0.42.1 (`cmake/CPM.cmake`,
SHA-256 `f3a6dcc6a04ce9e7f51a127307fa4f699fb2bade357a8eb4c5b45df76e1dc6a5`,
https://github.com/cpm-cmake/CPM.cmake), distributed under the MIT License
reproduced inside that file. It is a build-time tool and is not part of any
binary distribution.
```

Replace `cmake/readme.md`'s body with:

```markdown
CMake modules. `CPM.cmake` is the vendored upstream release 0.42.1 (MIT; see
THIRD_PARTY_LICENSES.md) — replace it wholesale when upgrading, never edit it.
The remaining files are adapted from https://github.com/lefticus/cpp_weekly/tree/master/cmake
and https://github.com/cpp-best-practices/cmake_template; each is included by
CMakeLists.txt or cmake/Dependencies.cmake (grep for its name before deleting).
```

- [ ] **Step 4: Fresh configure with an empty CPM cache proves no download of CPM itself**

```bash
CPM_SOURCE_CACHE= cmake -S . -B build/w0-cpm -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DDTWC_BUILD_TESTING=OFF -DDTWC_ENABLE_LLFIO=OFF -DDTWC_ENABLE_HIGHS=OFF -DDTWC_ENABLE_GUROBI=OFF 2>&1 | grep -i "CPM:" | head -5; echo "exit=${PIPESTATUS[0]}"
ls build/w0-cpm/cmake 2>/dev/null   # no CPM_0.42.1.cmake download artifact
rm -rf build/w0-cpm
uv run --no-project python scripts/check_supply_chain_pins.py; echo "pins exit=$?"
```

Expected: configure exit 0; no `build/w0-cpm/cmake/CPM_*.cmake`; the supply-chain script still passes. Record in the run-log that the llfio superbuild (`git clone` of quickcpplib + patch inside llfio's own CMake) is DEFERRED (B-17 "next"; ledger status stays `doing` with that note).

- [ ] **Step 5: Commit**

CHANGELOG (Unreleased → Changed): `- CPM.cmake 0.42.1 is vendored (SHA-256 pinned, MIT) instead of downloaded at configure time (B-17).`

```bash
git add cmake/CPM.cmake THIRD_PARTY_LICENSES.md cmake/readme.md CHANGELOG.md
git commit -m "build: vendor CPM.cmake 0.42.1 (B-17)"
```

---

### Task 8: CI legs — sanitizers through the project options, bare core, dev warnings, lint (B-15)

**Files:**

- Modify: `.github/workflows/ubuntu-unit.yml`
- Create: `.github/workflows/lint.yml`, `scripts/check_warning_ratchet.py`

- [ ] **Step 1: Sanitizer matrix entry through the project's own options (P27)**

In `ubuntu-unit.yml` replace the configure step's sanitizer branch:

```yaml
      - name: cmake configure
        run: |
          if [ "${{ matrix.sanitizer }}" = "true" ]; then
            cmake .. -DCMAKE_BUILD_TYPE=Debug -DDTWC_BUILD_TESTING=ON \
              -DDTWC_ENABLE_SANITIZER_ADDRESS=ON -DDTWC_ENABLE_SANITIZER_UNDEFINED=ON \
              -DCMAKE_CXX_FLAGS="-fno-omit-frame-pointer"
          else
            cmake .. -DCMAKE_BUILD_TYPE=Debug -DDTWC_BUILD_TESTING=ON
          fi
```

- [ ] **Step 2: The bare-core leg (all optional deps OFF, no OpenMP, YAML OFF)**

Append a job to `ubuntu-unit.yml`:

```yaml
  # Non-negotiable #3: the core builds and its tests pass with EVERY optional
  # dependency off, OpenMP included (explicit DTWC_ALLOW_SEQUENTIAL opt-out).
  # This is also the YAML-OFF leg: test_cli_config_formats asserts the typed
  # refusal (7 checks) instead of the 23-check TOML+YAML matrix.
  bare-core:
    name: Ubuntu unit bare core (all optional deps OFF, sequential)
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@df4cb1c069e1874edd31b4311f1884172cec0e10 # v6
      - name: update
        run: sudo apt update
      - name: install compiler
        run: sudo apt install gcc-12 g++-12
      - name: cmake configure
        run: |
          cmake -S . -B build_dir -DCMAKE_BUILD_TYPE=Debug -DDTWC_BUILD_TESTING=ON \
            -DDTWC_ENABLE_HIGHS=OFF -DDTWC_ENABLE_GUROBI=OFF -DDTWC_ENABLE_LLFIO=OFF \
            -DDTWC_ENABLE_ARROW=OFF -DDTWC_ENABLE_YAML=OFF -DDTWC_ENABLE_METAL=OFF \
            -DDTWC_ENABLE_MPI=OFF -DDTWC_ENABLE_CUDA=OFF \
            -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON -DDTWC_ALLOW_SEQUENTIAL=ON \
            2>&1 | tee configure.log
        env:
          CC: gcc-12
          CXX: g++-12
      - name: assert the sequential build was chosen explicitly (never silently)
        run: grep -q "building SEQUENTIAL" configure.log
      - name: cmake build
        run: cmake --build build_dir --parallel 2
      - name: cmake test
        run: ctest --test-dir build_dir -j2 -C Debug --output-on-failure --no-tests=error
```

- [ ] **Step 3: The dev-warnings leg with a ratchet**

`scripts/check_warning_ratchet.py`:

```python
#!/usr/bin/env python3
"""Count compiler warnings in a build log and fail above a ceiling.

Counts lines containing "warning:" (GCC/Clang) or ": warning C" (MSVC) whose
path does not point into a dependency (build_dir/_deps, extern/). Prints
WARNING_RATCHET count=<n> max=<m> and exits 1 when n > m. Start with a large
--max to measure, then pin the measured count; W1/W2 drive it to zero.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

WARNING = re.compile(r"(warning:|: warning C\d+)")
THIRD_PARTY = re.compile(r"(/_deps/|[\\/]extern[\\/])")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path)
    parser.add_argument("--max", type=int, required=True)
    args = parser.parse_args()
    lines = args.log.read_text(encoding="utf-8", errors="replace").splitlines()
    hits = [line for line in lines if WARNING.search(line) and not THIRD_PARTY.search(line)]
    unique = sorted(set(hits))
    print(f"WARNING_RATCHET count={len(unique)} max={args.max}")
    for line in unique[:50]:
        print(line)
    return 1 if len(unique) > args.max else 0


if __name__ == "__main__":
    sys.exit(main())
```

Job in `ubuntu-unit.yml`:

```yaml
  # P24: outside DTWC_DEV_MODE the maintainer warning set is an empty
  # INTERFACE library, so -Wconversion/-Wshadow never ran on CI. This leg runs
  # them (as warnings) and ratchets the count: the ceiling is pinned to the
  # first measured value and may only go down.
  dev-warnings:
    name: Ubuntu maintainer warnings (DTWC_DEV_MODE) ratchet
    runs-on: ubuntu-latest
    env:
      WARNING_MAX: 100000   # measure on the first run, then pin (see run-log W0)
    steps:
      - uses: actions/checkout@df4cb1c069e1874edd31b4311f1884172cec0e10 # v6
      - name: update
        run: sudo apt update
      - name: install compiler
        run: sudo apt install clang-17 libomp-17-dev
      - name: cmake configure
        run: |
          cmake -S . -B build_dir -DCMAKE_BUILD_TYPE=Debug -DDTWC_BUILD_TESTING=ON \
            -DDTWC_DEV_MODE=ON -DDTWC_WARNINGS_AS_ERRORS=OFF \
            -DDTWC_ENABLE_CLANG_TIDY=OFF -DDTWC_ENABLE_CPPCHECK=OFF -DDTWC_ENABLE_CACHE=OFF
        env:
          CC: clang-17
          CXX: clang++-17
      - name: cmake build (log kept)
        run: cmake --build build_dir --parallel 2 2>&1 | tee build.log
      - name: warning ratchet
        run: python3 scripts/check_warning_ratchet.py build.log --max "$WARNING_MAX"
      - name: cmake test
        run: ctest --test-dir build_dir -j2 -C Debug --output-on-failure
```

- [ ] **Step 4: The lint job**

`.github/workflows/lint.yml`:

```yaml
name: Lint
on:
  push:
    branches: [develop, Claude, design-2.0]
  pull_request:
    branches: [main]

jobs:
  hygiene:
    name: Repository and record hygiene
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@df4cb1c069e1874edd31b4311f1884172cec0e10 # v6
        with:
          fetch-depth: 0
      - name: repository hygiene invariants
        run: python3 scripts/check_repo_hygiene.py
      - name: record hygiene (stale claims in research/lessons records)
        run: python3 scripts/check_record_hygiene.py
      - name: clang-format on changed lines
        run: |
          set -euo pipefail
          sudo apt update && sudo apt install -y clang-format-17
          base=$(git merge-base origin/main HEAD)
          out=$(git clang-format-17 --diff --extensions cpp,hpp,cc,cu,cuh,mm "$base" HEAD -- dtwc tests benchmarks examples bindings python 2>&1 || true)
          if grep -q '^diff\|^---' <<<"$out"; then
            echo "$out"
            echo "::error::changed lines are not clang-format clean (run git clang-format $base)"
            exit 1
          fi
          echo "clang-format: changed lines clean"
```

- [ ] **Step 5: Validate the YAML and the scripts locally**

```bash
uv run --no-project python -c "import yaml,sys; [yaml.safe_load(open(f)) for f in ['.github/workflows/ubuntu-unit.yml','.github/workflows/lint.yml']]; print('yaml ok')" 2>/dev/null || uv run --with pyyaml --no-project python -c "import yaml; [yaml.safe_load(open(f)) for f in ['.github/workflows/ubuntu-unit.yml','.github/workflows/lint.yml']]; print('yaml ok')"
cmake --build build/highs-1151 --parallel 2>&1 | tee build/highs-1151/w0-warnings.log >/dev/null; uv run --no-project python scripts/check_warning_ratchet.py build/highs-1151/w0-warnings.log --max 100000
git clang-format --diff --extensions cpp,hpp,cc,cu,cuh,mm a31956e HEAD -- dtwc tests benchmarks | head -20
```

Expected: `yaml ok`; the ratchet prints `WARNING_RATCHET count=<n> max=100000` (n is the local clang-21 non-dev-mode count, informational); the clang-format diff of this branch's changes is empty (format the touched files with `git clang-format a31956e` if it is not, and amend the corresponding commit).

Then prove the bare-core leg locally before it ever runs on CI (it is the gate for non-negotiable #3):

```bash
cmake -S . -B build/w0-bare -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Debug -DDTWC_BUILD_TESTING=ON \
  -DDTWC_ENABLE_HIGHS=OFF -DDTWC_ENABLE_GUROBI=OFF -DDTWC_ENABLE_LLFIO=OFF -DDTWC_ENABLE_ARROW=OFF \
  -DDTWC_ENABLE_YAML=OFF -DDTWC_ENABLE_METAL=OFF -DDTWC_ENABLE_MPI=OFF -DDTWC_ENABLE_CUDA=OFF \
  -DCMAKE_DISABLE_FIND_PACKAGE_OpenMP=ON -DDTWC_ALLOW_SEQUENTIAL=ON 2>&1 | grep -E "building SEQUENTIAL|Error"
cmake --build build/w0-bare --parallel && ctest --test-dir build/w0-bare -j1 --output-on-failure 2>&1 | tail -12
```

Expected: the configure prints the `building SEQUENTIAL` warning; the build succeeds; `100% tests passed` with the mmap, HiGHS, YAML (7-check variant of `test_cli_config_formats`) and `unit_test_parallelisation` skips. A build error or a failing test here is a W0 finding about the bare-core contract, fixed minimally in this task and recorded in the run-log — never marked MAY_SKIP to make the leg green. Keep `build/w0-bare` (it is the local twin of the CI leg).

Record in the run-log: "dev-warnings ceiling: measure on the first CI run of `design-2.0`, then pin `WARNING_MAX` in a follow-up commit" (the local clang 21 count is not the CI clang-17 count).

- [ ] **Step 6: Commit**

CHANGELOG (Unreleased → Added): `- CI: bare-core leg (all optional deps OFF, sequential, YAML OFF), maintainer-warnings ratchet leg, lint workflow (repo/record hygiene + clang-format on changed lines); the ASan/UBSan leg uses the project's own sanitizer options (B-15).`

```bash
git add .github/workflows/ubuntu-unit.yml .github/workflows/lint.yml scripts/check_warning_ratchet.py CHANGELOG.md
git commit -m "ci: bare-core, dev-warnings ratchet and lint legs; sanitizers via project options (B-15)"
```

---

### Task 9: Layer manifest report (spec II.2 / III.1, report-only)

**Files:**

- Create: `cmake/DtwcLayers.cmake`, `tests/cmake/test_layer_check.cmake`, `tests/cmake/layer_fixture/base/a.hpp`, `tests/cmake/layer_fixture/core/b.hpp`, `tests/cmake/layer_fixture/session/c.hpp`
- Modify: `dtwc/CMakeLists.txt` (after the omp-critical scanner), `tests/CMakeLists.txt` (register the self-test)

**Interfaces:**

- Produces: `dtwc_check_layers(<source-dir> MANIFEST <list-var> [STRICT <rank>...])`, sets `DTWC_LAYER_UPWARD` (count) and `DTWC_LAYER_EDGES` (list of `from -> to` strings) in the caller's scope; `DTWC_LAYER_MANIFEST` for the dtwc tree.

- [ ] **Step 1: Write the scanner and the dtwc manifest**

`cmake/DtwcLayers.cmake`:

```cmake
include_guard(GLOBAL)

# Layer manifest for dtwc/ (spec II.2, III.1). Entries are "<regex>@<rank>"
# ('@' because a ';' inside a set() argument would split the entry); first
# match wins. Ranks: 0 base, 1 core (+ backends, which may include base+core
# only), 2 algorithms, 3 mip, 4 session, 5 io, 6 surface. A file may include
# only files of rank <= its own. The flat foundation headers are ranked one by
# one until C-11 moves them into dtwc/base/.
set(DTWC_LAYER_MANIFEST
    "^(error|missing_utils|settings|timing|utility|parallelisation|env|fileOperations)\\.(hpp|cpp)$@0"
    "^(types|enums)/@0"
    "^core/@1"
    "^(warping[a-z_]*|soft_dtw|distance)\\.hpp$@1"
    "^detail/decode_pair\\.hpp$@1"
    "^(cuda|metal|mpi)/@1"
    "^algorithms/@2"
    "^initialisation\\.(hpp|cpp)$@2"
    "^mip/@3"
    "^(Problem|Problem_IO|Data|checkpoint|scores|system_memory)\\.(hpp|cpp)$@4"
    "^detail/tier1_method_resolution\\.hpp$@4"
    "^io/@5"
    "^DataLoader\\.hpp$@5"
    "^cli/@5"
    "^(api|dtwc|test_api)\\.(hpp|cpp)$@6"
    "^(dtwc_cl|main)\\.cpp$@6")

function(_dtwc_layer_rank manifest_var path out_var)
  set(rank -1)
  foreach(entry IN LISTS ${manifest_var})
    if(NOT entry MATCHES "^(.*)@([0-9]+)$")
      message(FATAL_ERROR "layer manifest entry without '@<rank>': ${entry}")
    endif()
    set(pattern "${CMAKE_MATCH_1}")
    set(entry_rank "${CMAKE_MATCH_2}")
    if(path MATCHES "${pattern}")
      set(rank ${entry_rank})
      break()
    endif()
  endforeach()
  set(${out_var} ${rank} PARENT_SCOPE)
endfunction()

# dtwc_check_layers(<source-dir> MANIFEST <list-var> [STRICT <rank>...])
# Scans .hpp/.cpp/.cu/.cuh/.mm under <source-dir> (extern/ excluded), resolves
# every quoted #include relative to the including file, then to <source-dir>,
# and reports each edge that points to a higher rank. Edges whose INCLUDING
# file has a STRICT rank are errors; the rest are STATUS lines. Summary:
#   DTWC_LAYER_CHECK files=<n> upward=<n> strict=<ranks>
# Sets DTWC_LAYER_UPWARD and DTWC_LAYER_EDGES in the caller's scope.
function(dtwc_check_layers source_dir)
  cmake_parse_arguments(ARG "" "MANIFEST" "STRICT" ${ARGN})
  if(NOT ARG_MANIFEST)
    message(FATAL_ERROR "dtwc_check_layers: MANIFEST is required")
  endif()
  file(GLOB_RECURSE files RELATIVE "${source_dir}" CONFIGURE_DEPENDS
       "${source_dir}/*.hpp" "${source_dir}/*.cpp" "${source_dir}/*.cu"
       "${source_dir}/*.cuh" "${source_dir}/*.mm")
  list(FILTER files EXCLUDE REGEX "^extern/")
  set(upward 0)
  set(edges "")
  set(unmapped "")
  foreach(f IN LISTS files)
    _dtwc_layer_rank(${ARG_MANIFEST} "${f}" rank)
    if(rank EQUAL -1)
      list(APPEND unmapped "${f}")
      continue()
    endif()
    get_filename_component(dir "${f}" DIRECTORY)
    file(STRINGS "${source_dir}/${f}" includes REGEX "^[ \t]*#[ \t]*include[ \t]*\"")
    foreach(line IN LISTS includes)
      string(REGEX REPLACE "^[ \t]*#[ \t]*include[ \t]*\"([^\"]+)\".*" "\\1" inc "${line}")
      if(dir)
        set(candidate "${dir}/${inc}")
      else()
        set(candidate "${inc}")
      endif()
      get_filename_component(absolute "${source_dir}/${candidate}" ABSOLUTE)
      file(RELATIVE_PATH candidate "${source_dir}" "${absolute}")
      if(NOT EXISTS "${source_dir}/${candidate}")
        set(candidate "${inc}")
        if(NOT EXISTS "${source_dir}/${candidate}")
          continue()   # not in this tree (system or third-party header)
        endif()
      endif()
      _dtwc_layer_rank(${ARG_MANIFEST} "${candidate}" inc_rank)
      if(inc_rank EQUAL -1)
        continue()
      endif()
      if(inc_rank GREATER rank)
        math(EXPR upward "${upward} + 1")
        list(APPEND edges "${f} -> ${candidate}")
        if("${rank}" IN_LIST ARG_STRICT)
          message(SEND_ERROR "layer violation: ${f} (rank ${rank}) includes ${candidate} (rank ${inc_rank})")
        else()
          message(STATUS "layer report: ${f} (${rank}) -> ${candidate} (${inc_rank})")
        endif()
      endif()
    endforeach()
  endforeach()
  if(unmapped)
    message(FATAL_ERROR "layer manifest: files without a layer: ${unmapped}")
  endif()
  list(LENGTH files file_count)
  message(STATUS "DTWC_LAYER_CHECK files=${file_count} upward=${upward} strict=${ARG_STRICT}")
  set(DTWC_LAYER_UPWARD ${upward} PARENT_SCOPE)
  set(DTWC_LAYER_EDGES "${edges}" PARENT_SCOPE)
endfunction()
```

- [ ] **Step 2: Fixture and self-test (run before wiring into the real tree)**

Fixture files:
- `tests/cmake/layer_fixture/base/a.hpp`: `#pragma once`
- `tests/cmake/layer_fixture/core/b.hpp`: `#pragma once` / `#include "../base/a.hpp"` / `#include "../session/c.hpp"`
- `tests/cmake/layer_fixture/session/c.hpp`: `#pragma once` / `#include "../core/b.hpp"` / `#include <vector>`

`tests/cmake/test_layer_check.cmake`:

```cmake
# cmake -P self-test of dtwc_check_layers on a three-file fixture with exactly
# one upward edge (core/b.hpp -> session/c.hpp). Prints
# CMAKE_LAYER_CHECK fixture_upward=<n> strict_failed=<0|1>.
cmake_minimum_required(VERSION 3.26)
include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/DtwcLayers.cmake")
set(FIXTURE_MANIFEST "^base/@0" "^core/@1" "^session/@4")
dtwc_check_layers("${CMAKE_CURRENT_LIST_DIR}/layer_fixture" MANIFEST FIXTURE_MANIFEST)
if(NOT DTWC_LAYER_UPWARD EQUAL 1 OR NOT DTWC_LAYER_EDGES STREQUAL "core/b.hpp -> session/c.hpp")
  message(FATAL_ERROR "expected exactly one upward edge core/b.hpp -> session/c.hpp, got ${DTWC_LAYER_UPWARD}: ${DTWC_LAYER_EDGES}")
endif()
# Strict mode on rank 1 must turn that edge into an error: run it in a child
# process so this script can observe the failure.
execute_process(
  COMMAND "${CMAKE_COMMAND}" -DSTRICT_PROBE=1 -P "${CMAKE_CURRENT_LIST_FILE}"
  RESULT_VARIABLE strict_rc OUTPUT_QUIET ERROR_QUIET)
if(strict_rc EQUAL 0)
  message(FATAL_ERROR "strict mode did not fail on the upward edge")
endif()
message(STATUS "CMAKE_LAYER_CHECK fixture_upward=1 strict_failed=1")
```

Add at the very top of that script, before the include:

```cmake
if(DEFINED STRICT_PROBE)
  include("${CMAKE_CURRENT_LIST_DIR}/../../cmake/DtwcLayers.cmake")
  set(FIXTURE_MANIFEST "^base/@0" "^core/@1" "^session/@4")
  dtwc_check_layers("${CMAKE_CURRENT_LIST_DIR}/layer_fixture" MANIFEST FIXTURE_MANIFEST STRICT 1)
  return()
endif()
```

Run: `cmake -P tests/cmake/test_layer_check.cmake`
Expected: `-- layer report: core/b.hpp (1) -> session/c.hpp (4)`, `-- DTWC_LAYER_CHECK files=3 upward=1 strict=`, then `-- CMAKE_LAYER_CHECK fixture_upward=1 strict_failed=1`, exit 0.

Register in `tests/CMakeLists.txt` next to `cmake_regex_at_least`:

```cmake
add_test(NAME cmake_layer_check
         COMMAND ${CMAKE_COMMAND} -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/test_layer_check.cmake")
set_tests_properties(cmake_layer_check PROPERTIES
  PASS_REGULAR_EXPRESSION "CMAKE_LAYER_CHECK fixture_upward=1 strict_failed=1"
  LABELS "tooling")
```

- [ ] **Step 3: Wire the report into the library configure (report-only)**

`dtwc/CMakeLists.txt`, after the omp-critical scanner block (line ~115):

```cmake
# Layer report (spec II.2): includes may only point downward. Report-only in W0;
# W1 passes STRICT 0 once base/ exists, later waves add ranks.
include("${CMAKE_SOURCE_DIR}/cmake/DtwcLayers.cmake")
dtwc_check_layers("${CMAKE_CURRENT_SOURCE_DIR}" MANIFEST DTWC_LAYER_MANIFEST)
```

```bash
cmake -S . -B build/highs-1151 2>&1 | grep -E "layer report|DTWC_LAYER_CHECK|without a layer"
```

Expected: no "files without a layer" (if there is one, add it to the manifest with the rank its includes justify and note it in the run-log); the summary `DTWC_LAYER_CHECK files=<n> upward=<m> strict=`; and among the reported edges at least these three from the as-is core map §4a: `core/pruned_distance_matrix.hpp -> Problem.hpp`, `core/mmap_data_store.hpp -> Data.hpp`, `core/dtw_dispatch.cpp -> Problem.hpp`. Copy the full edge list into the run-log with the reconciliation: the map's nine §4a rows include six `core -> warping*/distance/soft_dtw` rows that the manifest ranks lateral (rank 1 → rank 1); the rank-based count `m` is the number the exit gate records.

- [ ] **Step 4: Commit**

```bash
git add cmake/DtwcLayers.cmake tests/cmake/test_layer_check.cmake tests/cmake/layer_fixture dtwc/CMakeLists.txt tests/CMakeLists.txt
git commit -m "build: layer manifest report at configure time, self-tested on a fixture (spec II.2, report-only)"
```

---

### Task 10: Shared allocation guard in `tests/support` (spec II.6, II.7)

**Files:**

- Create: `tests/support/allocation_guard.hpp`, `tests/unit/unit_test_allocation_guard.cpp`
- Modify: `tests/unit/unit_test_soft_dtw_hotpath.cpp` (lines 15–61), `tests/unit/algorithms/unit_test_barycenter_allocations.cpp` (lines 18–82), `tests/unit/unit_test_checkpoint_binary.cpp` (lines 38–80 and the uses at 358–364, 542)

**Interfaces:**

- Produces: `dtwc::test_support::AllocationCount` (RAII scope; `count()`, `bytes()`, `exact_size_count()`), `dtwc::test_support::allocation_guard::configure(min_bytes, exact_bytes)`, macro `REQUIRE_NO_ALLOCATIONS(statement...)`.

- [ ] **Step 1: Write the contract test first**

`tests/unit/unit_test_allocation_guard.cpp`:

```cpp
/**
 * @file unit_test_allocation_guard.cpp
 * @brief Contract of tests/support/allocation_guard.hpp: counts heap
 * allocations (all, large-only, exact-size) while a scope is alive, nothing
 * outside it.
 */
#include "../support/allocation_guard.hpp"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <vector>

using dtwc::test_support::AllocationCount;

// Snapshots are taken BEFORE any REQUIRE: Catch2's assertion machinery may
// allocate, and the counter is process-wide.
TEST_CASE("AllocationCount sees every heap allocation inside its scope", "[support][allocation]")
{
  std::vector<int> outside(10);   // before the scope: not counted
  std::size_t before = 0, after = 0, again = 0, bytes = 0;
  {
    AllocationCount probe;
    before = probe.count();
    {
      std::vector<int> inside(1000);
      auto one = std::make_unique<double>(1.0);
      after = probe.count();
      bytes = probe.bytes();
    }
    again = probe.count();          // frees are not allocations
  }
  REQUIRE(before == 0);
  REQUIRE(after == 2);
  REQUIRE(again == 2);
  REQUIRE(bytes >= 1000 * sizeof(int) + sizeof(double));
}

TEST_CASE("AllocationCount thresholds: large-only and exact-size", "[support][allocation]")
{
  std::size_t large_after_small = 0, large_after_big = 0, exact_hits = 0;
  {
    AllocationCount large{ /*min_bytes=*/500U * 1024U };
    std::vector<char> small(100);
    large_after_small = large.count();
    std::vector<char> big(600U * 1024U);
    large_after_big = large.count();
  }
  {
    AllocationCount exact{ /*min_bytes=*/0, /*exact_bytes=*/257U * sizeof(int) };
    std::vector<int> v(257);
    std::vector<int> w(258);
    exact_hits = exact.exact_size_count();
  }
  REQUIRE(large_after_small == 0);
  REQUIRE(large_after_big == 1);
  REQUIRE(exact_hits == 1);
}

TEST_CASE("REQUIRE_NO_ALLOCATIONS passes for stack-only work", "[support][allocation]")
{
  double acc = 0.0;
  REQUIRE_NO_ALLOCATIONS(for (int i = 0; i < 1000; ++i) acc += i * 0.5;);
  REQUIRE(acc > 0.0);
}
```

Register it in the pass-2 loop (default registration; its floor comes from the exit-gate regeneration — until then add `set(DTWC_TEST_FLOOR_unit_test_allocation_guard "9;3")` to `tests/floors.cmake` by hand and note it in the run-log).

- [ ] **Step 2: Write the guard**

`tests/support/allocation_guard.hpp`:

```cpp
/**
 * @file allocation_guard.hpp
 * @brief Process-wide heap allocation counter for allocation-free contracts.
 *
 * Include this header in exactly ONE translation unit per test executable: it
 * replaces the global operator new/delete, which a program may do once. Count
 * allocations made while an AllocationCount is alive (all threads), or assert
 * none with REQUIRE_NO_ALLOCATIONS(statement). Only the plain size_t overloads
 * are replaced (as the three probes it replaces did); aligned allocations go to
 * the default operator and are not counted.
 */
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdlib>
#include <new>

namespace dtwc::test_support {

namespace allocation_guard {

inline std::atomic<bool> enabled{ false };
inline std::atomic<std::size_t> min_bytes{ 0 };
inline std::atomic<std::size_t> exact_bytes{ 0 };
inline std::atomic<std::size_t> count{ 0 };
inline std::atomic<std::size_t> bytes{ 0 };
inline std::atomic<std::size_t> exact_count{ 0 };

inline void* allocate(std::size_t size)
{
  if (enabled.load(std::memory_order_relaxed)) {
    if (size >= min_bytes.load(std::memory_order_relaxed)) {
      count.fetch_add(1, std::memory_order_relaxed);
      bytes.fetch_add(size, std::memory_order_relaxed);
    }
    if (size == exact_bytes.load(std::memory_order_relaxed))
      exact_count.fetch_add(1, std::memory_order_relaxed);
  }
  if (void* memory = std::malloc(size == 0 ? 1 : size)) return memory;
  throw std::bad_alloc{};
}

} // namespace allocation_guard

/// Counts heap allocations while alive. `min_bytes` ignores smaller requests
/// (the "large allocation" probes); `exact_bytes` additionally counts requests
/// of exactly that size (the checkpoint wire-format probe). Scopes must not
/// nest: the counters are process-wide.
class AllocationCount {
public:
  explicit AllocationCount(std::size_t min_bytes = 0, std::size_t exact_bytes = 0)
  {
    namespace g = allocation_guard;
    g::min_bytes.store(min_bytes, std::memory_order_relaxed);
    g::exact_bytes.store(exact_bytes, std::memory_order_relaxed);
    g::count.store(0, std::memory_order_relaxed);
    g::bytes.store(0, std::memory_order_relaxed);
    g::exact_count.store(0, std::memory_order_relaxed);
    g::enabled.store(true, std::memory_order_relaxed);
  }
  ~AllocationCount() { allocation_guard::enabled.store(false, std::memory_order_relaxed); }
  AllocationCount(const AllocationCount&) = delete;
  AllocationCount& operator=(const AllocationCount&) = delete;

  std::size_t count() const { return allocation_guard::count.load(std::memory_order_relaxed); }
  std::size_t bytes() const { return allocation_guard::bytes.load(std::memory_order_relaxed); }
  std::size_t exact_size_count() const { return allocation_guard::exact_count.load(std::memory_order_relaxed); }
};

} // namespace dtwc::test_support

void* operator new(std::size_t size) { return dtwc::test_support::allocation_guard::allocate(size); }
void* operator new[](std::size_t size) { return dtwc::test_support::allocation_guard::allocate(size); }
void operator delete(void* memory) noexcept { std::free(memory); }
void operator delete[](void* memory) noexcept { std::free(memory); }
void operator delete(void* memory, std::size_t) noexcept { std::free(memory); }
void operator delete[](void* memory, std::size_t) noexcept { std::free(memory); }

/// Run `statement` and REQUIRE that it made no heap allocation. The count is
/// read before REQUIRE runs, so Catch2's own allocations are not counted.
#define REQUIRE_NO_ALLOCATIONS(...)                                          \
  do {                                                                       \
    std::size_t dtwc_allocations_seen_ = 0;                                  \
    {                                                                        \
      ::dtwc::test_support::AllocationCount dtwc_allocation_probe_;          \
      __VA_ARGS__                                                            \
      dtwc_allocations_seen_ = dtwc_allocation_probe_.count();               \
    }                                                                        \
    REQUIRE(dtwc_allocations_seen_ == 0);                                    \
  } while (0)
```

Build and run the contract test: `cmake -S . -B build/highs-1151 && cmake --build build/highs-1151 --target unit_test_allocation_guard && build/highs-1151/bin/unit_test_allocation_guard`
Expected: `All tests passed (9 assertions in 3 test cases)`. (The snapshots are read before any REQUIRE runs, so Catch2's own allocations cannot leak into the counts; if `after == 2` still fails, the standard library on this platform allocates more than once for `vector(1000)` — record the observed count and change the expectation to it, with the reason.)

- [ ] **Step 3: Migrate the three probes**

- `unit_test_soft_dtw_hotpath.cpp`: delete lines 15–61 (namespace `allocation_probe` and the six operator definitions); add `#include "../support/allocation_guard.hpp"`; `allocation_probe::Scope probe;` → `dtwc::test_support::AllocationCount probe;` (line 75) and `dtwc::test_support::AllocationCount probe{ 500U * 1024U };` for the large-only scope (line 103); `allocation_probe::allocations.load(...) == 0` → `probe.count() == 0` (move the REQUIRE inside the scope or keep `probe` alive: declare it before the block); `allocation_probe::large_allocations.load(...)` → `probe.count()`.
- `unit_test_barycenter_allocations.cpp`: delete lines 18–82 (probe namespace, operators, `ProbeScope`); include the header; `ProbeScope probe;` → `dtwc::test_support::AllocationCount probe{ 500U * 1024U };`; the `count` read at line 120 → `probe.count()` (declare `probe` in the enclosing scope so it is still alive).
- `unit_test_checkpoint_binary.cpp`: delete lines 38–80; include the header; at 358–364 replace the manual `exact_allocations.store(0)/enabled.store(true/false)` with `dtwc::test_support::AllocationCount probe{ 0, 257U * sizeof(int) };` in that scope; at 542 `allocation_probe::exact_allocations.load(...)` → `probe.exact_size_count()`.

```bash
cmake --build build/highs-1151 --target unit_test_soft_dtw_hotpath unit_test_barycenter_allocations unit_test_checkpoint_binary
ctest --test-dir build/highs-1151 -R 'unit_test_soft_dtw_hotpath|unit_test_barycenter_allocations|unit_test_checkpoint_binary|unit_test_allocation_guard' --output-on-failure | tail -6
grep -rn "operator new" tests --include=*.cpp --include=*.hpp | grep -v tests/support/allocation_guard.hpp
```

Expected: 4/4 passed, each with its previous assertion count (F51 keeps its marker and floor); the grep prints nothing.

- [ ] **Step 4: Commit**

```bash
git add tests/support/allocation_guard.hpp tests/unit/unit_test_allocation_guard.cpp tests/unit/unit_test_soft_dtw_hotpath.cpp tests/unit/algorithms/unit_test_barycenter_allocations.cpp tests/unit/unit_test_checkpoint_binary.cpp tests/floors.cmake
git commit -m "test(support): one allocation guard replaces three probes (spec II.6/II.7)"
```

---

### Task 11: The two registered benchmarks (A-06, C-06/C-07 evidence)

**Files:**

- Create: `benchmarks/bench_fast_pam_swap.cpp`, `benchmarks/bench_matrix_set_contention.cpp`
- Modify: `benchmarks/CMakeLists.txt` (two `dtwc_add_benchmark` lines)

**Interfaces:**

- Consumes: `dtwc::test_support::benchmark_series_set(count, length, base_seed)`, `Problem::{set_data, band, distance_strategy, fill_distance_matrix, dist_by_ind, visit_distmat, use_mmap_distance_matrix}`, `dtwc::fast_pam_swap(prob, medoids, max_iter, PAMVariant)`, `core::DenseDistanceMatrix::{resize,set}`, `core::MmapDistanceMatrix::set`.
- Produces: JSON results under `build/w0-bench/bench/` (untracked); numbers in the run-log.

- [ ] **Step 1: `bench_fast_pam_swap.cpp`**

```cpp
/**
 * @file bench_fast_pam_swap.cpp
 * @brief W0 baseline evidence for ledger A-06 / O-01: what the dist_by_ind()
 * preflight costs on a cached matrix, and whether it is visible in a FastPAM1
 * SWAP sweep that pays it N*k times per iteration.
 *
 * One N=2000, L=25, band=10 problem, matrix filled once (untimed):
 *   BM_dist_by_ind_pairs  N(N-1)/2 calls of Problem::dist_by_ind (preflight,
 *                         variant visit, packed read)
 *   BM_matrix_get_pairs   the same pairs read from the packed matrix through
 *                         visit_distmat (no preflight)
 *   BM_fastpam1_sweep     one FastPAM1 SWAP iteration from a fixed medoid set
 *   BM_fasterpam_sweep    one FasterPAM SWAP iteration from the same set
 * items_processed carries the machine-independent count (pairs, or N*k
 * candidate evaluations); wall-clock is advisory (spec II.6).
 */
#include <benchmark/benchmark.h>
#include <dtwc.hpp>

#include "../tests/support/deterministic_series.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr int kN = 2000;
constexpr int kL = 25;
constexpr int kBand = 10;
constexpr int kK = 10;

struct Fixture {
  dtwc::Problem prob{ "bench_fast_pam_swap" };
  std::vector<int> medoids;

  Fixture()
  {
    auto series = dtwc::test_support::benchmark_series_set(
      static_cast<std::size_t>(kN), static_cast<std::size_t>(kL), 100);
    std::vector<std::string> names;
    names.reserve(kN);
    for (int i = 0; i < kN; ++i) names.push_back("s" + std::to_string(i));
    prob.set_data(dtwc::Data(std::move(series), std::move(names)));
    prob.band = kBand;
    prob.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
    prob.fill_distance_matrix();
    for (int m = 0; m < kK; ++m) medoids.push_back(m * (kN / kK));   // fixed, evenly spaced start
  }
};

Fixture& fixture()
{
  static Fixture f;
  return f;
}

void BM_dist_by_ind_pairs(benchmark::State& state)
{
  auto& f = fixture();
  for (auto _ : state) {
    double sum = 0.0;
    for (int i = 1; i < kN; ++i)
      for (int j = 0; j < i; ++j) sum += f.prob.dist_by_ind(i, j);
    benchmark::DoNotOptimize(sum);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(kN) * (kN - 1) / 2);
}

void BM_matrix_get_pairs(benchmark::State& state)
{
  auto& f = fixture();
  for (auto _ : state) {
    double sum = 0.0;
    f.prob.visit_distmat([&](const auto& m) {
      for (std::size_t i = 1; i < static_cast<std::size_t>(kN); ++i)
        for (std::size_t j = 0; j < i; ++j) sum += m.get(i, j);
    });
    benchmark::DoNotOptimize(sum);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(kN) * (kN - 1) / 2);
}

void BM_fastpam1_sweep(benchmark::State& state)
{
  auto& f = fixture();
  for (auto _ : state) {
    const auto r = dtwc::fast_pam_swap(f.prob, f.medoids, 1, dtwc::PAMVariant::FastPAM1);
    benchmark::DoNotOptimize(r.total_cost);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(kN) * kK);
}

void BM_fasterpam_sweep(benchmark::State& state)
{
  auto& f = fixture();
  for (auto _ : state) {
    const auto r = dtwc::fast_pam_swap(f.prob, f.medoids, 1, dtwc::PAMVariant::FasterPAM);
    benchmark::DoNotOptimize(r.total_cost);
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(kN) * kK);
}

} // namespace

BENCHMARK(BM_dist_by_ind_pairs)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_matrix_get_pairs)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fastpam1_sweep)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_fasterpam_sweep)->Unit(benchmark::kMillisecond);

int main(int argc, char** argv)
{
  benchmark::Initialize(&argc, argv);
  benchmark::AddCustomContext("N", std::to_string(kN));
  benchmark::AddCustomContext("k", std::to_string(kK));
  benchmark::AddCustomContext("band", std::to_string(kBand));
#ifdef _OPENMP
  benchmark::AddCustomContext("omp_max_threads", std::to_string(omp_get_max_threads()));
#else
  benchmark::AddCustomContext("omp_max_threads", "1");
#endif
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
```

- [ ] **Step 2: `bench_matrix_set_contention.cpp`**

```cpp
/**
 * @file bench_matrix_set_contention.cpp
 * @brief W0 baseline evidence for ledger C-06/C-07 (spec III.3): the cost of
 * MmapDistanceMatrix::set() -- two atomic XOR digest lanes per write -- against
 * DenseDistanceMatrix::set(), serial and under OpenMP contention.
 *
 * N=2000 (2 001 000 packed cells). Each iteration writes every cell once with
 * a value that changes per iteration (so the digest early-return never fires):
 *   BM_set_dense_serial / BM_set_mmap_serial        one thread, row-major
 *   BM_set_dense_static / BM_set_mmap_static        omp parallel for, schedule(static):
 *                                                   contiguous row blocks per thread
 *   BM_set_dense_static1 / BM_set_mmap_static1      schedule(static, 1): adjacent rows
 *                                                   on different threads, so the four
 *                                                   rows sharing one digest cache line
 *                                                   are written by four threads
 * items_processed = cells written. The mmap benchmarks exist only when the
 * build publishes DTWC_HAS_MMAP.
 */
#include <benchmark/benchmark.h>
#include <dtwc.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

constexpr std::size_t kN = 2000;

dtwc::Data tiny_data()
{
  std::vector<std::vector<double>> series(kN, std::vector<double>{ 0.0, 1.0 });
  std::vector<std::string> names;
  names.reserve(kN);
  for (std::size_t i = 0; i < kN; ++i) names.push_back("s" + std::to_string(i));
  return dtwc::Data(std::move(series), std::move(names));
}

enum class Schedule { Serial, Static, Static1 };

template <class Matrix>
void write_all(Matrix& m, double base, Schedule schedule)
{
  const auto n = static_cast<std::int64_t>(kN);
  auto row = [&](std::int64_t i) {
    for (std::int64_t j = 0; j < i; ++j)
      m.set(static_cast<std::size_t>(i), static_cast<std::size_t>(j), base + static_cast<double>(i + j));
  };
  switch (schedule) {
  case Schedule::Serial:
    for (std::int64_t i = 1; i < n; ++i) row(i);
    break;
  case Schedule::Static:
#pragma omp parallel for schedule(static)
    for (std::int64_t i = 1; i < n; ++i) row(i);
    break;
  case Schedule::Static1:
#pragma omp parallel for schedule(static, 1)
    for (std::int64_t i = 1; i < n; ++i) row(i);
    break;
  }
}

void bench_dense(benchmark::State& state, Schedule schedule)
{
  dtwc::core::DenseDistanceMatrix m(kN);
  double base = 1.0;
  for (auto _ : state) {
    write_all(m, base, schedule);
    base += 1.0;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(kN) * (kN - 1) / 2);
}

void BM_set_dense_serial(benchmark::State& s) { bench_dense(s, Schedule::Serial); }
void BM_set_dense_static(benchmark::State& s) { bench_dense(s, Schedule::Static); }
void BM_set_dense_static1(benchmark::State& s) { bench_dense(s, Schedule::Static1); }
BENCHMARK(BM_set_dense_serial)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_set_dense_static)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_set_dense_static1)->Unit(benchmark::kMillisecond);

#ifdef DTWC_HAS_MMAP
void bench_mmap(benchmark::State& state, Schedule schedule)
{
  static int counter = 0;
  const auto path = std::filesystem::temp_directory_path() / "dtwc_bench"
                    / ("set_contention_" + std::to_string(counter++) + ".cache");
  std::filesystem::create_directories(path.parent_path());
  dtwc::Problem prob("bench_matrix_set_contention");
  prob.set_data(tiny_data());
  prob.distance_strategy = dtwc::DistanceMatrixStrategy::BruteForce;
  prob.use_mmap_distance_matrix(path);
  double base = 1.0;
  for (auto _ : state) {
    prob.visit_distmat([&](auto& m) {
      if constexpr (std::is_same_v<std::decay_t<decltype(m)>, dtwc::core::MmapDistanceMatrix>)
        write_all(m, base, schedule);
    });
    base += 1.0;
  }
  state.SetItemsProcessed(state.iterations() * static_cast<std::int64_t>(kN) * (kN - 1) / 2);
  std::filesystem::remove(path);
}

void BM_set_mmap_serial(benchmark::State& s) { bench_mmap(s, Schedule::Serial); }
void BM_set_mmap_static(benchmark::State& s) { bench_mmap(s, Schedule::Static); }
void BM_set_mmap_static1(benchmark::State& s) { bench_mmap(s, Schedule::Static1); }
BENCHMARK(BM_set_mmap_serial)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_set_mmap_static)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_set_mmap_static1)->Unit(benchmark::kMillisecond);
#endif

} // namespace

int main(int argc, char** argv)
{
  benchmark::Initialize(&argc, argv);
  benchmark::AddCustomContext("N", std::to_string(kN));
#ifdef _OPENMP
  benchmark::AddCustomContext("omp_max_threads", std::to_string(omp_get_max_threads()));
#else
  benchmark::AddCustomContext("omp_max_threads", "1");
#endif
#ifdef DTWC_HAS_MMAP
  benchmark::AddCustomContext("mmap", "on");
#else
  benchmark::AddCustomContext("mmap", "off");
#endif
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return 0;
}
```

Add `#include <type_traits>` to the includes. `use_mmap_distance_matrix`'s exact parameter list is at `dtwc/Problem.hpp:640`; if it takes more than the path, pass the defaults the way `benchmarks/bench_mmap_access.cpp:95` does.

- [ ] **Step 3: Register, build, run, record**

`benchmarks/CMakeLists.txt`: add `dtwc_add_benchmark(NAME bench_fast_pam_swap OWN_MAIN)` and `dtwc_add_benchmark(NAME bench_matrix_set_contention OWN_MAIN)` after `bench_openmp_schedule`.

```bash
cmake -S . -B build/w0-bench && cmake --build build/w0-bench --parallel --target bench_fast_pam_swap bench_matrix_set_contention
mkdir -p build/w0-bench/bench
build/w0-bench/benchmarks/bench_fast_pam_swap --benchmark_min_time=2s --benchmark_out=build/w0-bench/bench/fast_pam_swap.json --benchmark_out_format=json
build/w0-bench/benchmarks/bench_matrix_set_contention --benchmark_min_time=2s --benchmark_out=build/w0-bench/bench/matrix_set_contention.json --benchmark_out_format=json
```

(Executable paths: `build/w0-bench/benchmarks/` or `build/w0-bench/bin/` — check with `find build/w0-bench -name "bench_fast_pam_swap*" -type f`.)
Expected: both run to completion. Record in the run-log a table with, per benchmark, `real_time` (ms), `items_per_second`, and the context (`omp_max_threads`, `N`, `k`, `band`, `mmap`), plus the two derived numbers the ledger asks for: `ns per pair (dist_by_ind) - ns per pair (matrix get)` = preflight cost per call; `mmap_static1 / dense_static1` = digest cost under contention. No conclusions in W0: A-06 is decided in W4 on this evidence, C-06/C-07 in W2.

- [ ] **Step 4: Commit**

CHANGELOG (Unreleased → Added): `- Benchmarks: bench_fast_pam_swap (dist_by_ind preflight vs direct read; FastPAM1/FasterPAM sweep) and bench_matrix_set_contention (dense vs mmap set() serial and under OpenMP contention).`

```bash
git add benchmarks/bench_fast_pam_swap.cpp benchmarks/bench_matrix_set_contention.cpp benchmarks/CMakeLists.txt CHANGELOG.md
git commit -m "bench: FastPAM swap and matrix set() contention baselines (A-06, C-06/C-07 evidence)"
```

---

### Task 12: IPO inlining report (A-06 Q10)

**Files:**

- Create: `scripts/check_ipo_inlining.py`
- Modify: `tests/CMakeLists.txt` (registration, non-Windows only)

- [ ] **Step 1: The script**

`scripts/check_ipo_inlining.py`:

```python
#!/usr/bin/env python3
"""Report whether Problem::dist_by_ind survives as an out-of-line call inside
the FastPAM SWAP kernels of a linked binary (ledger A-06 / core map Q10).

Disassembles the binary with llvm-objdump or objdump (--disassemble
--demangle), walks the function blocks whose demangled name matches --callers,
and counts call/branch instructions whose text names dist_by_ind. Prints

  IPO_INLINING binary=<name> tool=<objdump> ipo=<on|off|unknown> callers=<n> dist_by_ind_calls=<n> symbol_present=<0|1>

Report-only: exit 0 whenever a disassembler produced function blocks; exit 2
when no disassembler is available or the binary carries no symbols (a stripped
binary cannot be inspected -- the CTest registration is Linux/macOS only).
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

FUNCTION_HEADER = re.compile(r"^[0-9a-fA-F]+ <(.+)>:$")
CALL = re.compile(r"\b(call[a-z]*|bl|blr|jmp)\b")


def disassemble(binary: Path) -> tuple[str, str]:
    for tool in ("llvm-objdump", "objdump"):
        exe = shutil.which(tool)
        if exe:
            flags = ["-d", "--no-show-raw-insn", "-C"] if tool == "objdump" else ["-d", "--no-show-raw-insn", "--demangle"]
            out = subprocess.run([exe, *flags, str(binary)], capture_output=True, text=True, errors="replace")
            if out.returncode == 0:
                return tool, out.stdout
    return "none", ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--callers", default=r"fast_pam|swap|omp_outlined|_omp_fn|nearest_and_second",
                        help="regex over demangled function names to inspect")
    parser.add_argument("--callee", default="dist_by_ind")
    parser.add_argument("--ipo", default="unknown", help="on|off|unknown, copied into the marker")
    args = parser.parse_args()

    tool, text = disassemble(args.binary)
    if tool == "none":
        print(f"IPO_INLINING binary={args.binary.name} tool=none ipo={args.ipo} callers=0 dist_by_ind_calls=0 symbol_present=0")
        return 2

    callers = re.compile(args.callers)
    current = ""
    inspected = 0
    calls = 0
    symbol_present = 0
    for line in text.splitlines():
        header = FUNCTION_HEADER.match(line)
        if header:
            current = header.group(1)
            if args.callee in current:
                symbol_present = 1
            if callers.search(current):
                inspected += 1
            continue
        if current and callers.search(current) and CALL.search(line) and args.callee in line:
            calls += 1

    print(f"IPO_INLINING binary={args.binary.name} tool={tool} ipo={args.ipo} callers={inspected} "
          f"dist_by_ind_calls={calls} symbol_present={symbol_present}")
    return 0 if inspected > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Register (Linux/macOS builds only) and run under WSL for both IPO states**

`tests/CMakeLists.txt`, after the script tests:

```cmake
# A-06 evidence: does IPO inline Problem::dist_by_ind into the SWAP kernels?
# Report-only (the marker is the evidence; W4 decides). Needs an ELF/Mach-O
# binary with symbols and an objdump: not registered on Windows.
if(NOT WIN32 AND TARGET dtwc_cl)
    if(CMAKE_INTERPROCEDURAL_OPTIMIZATION)
        set(_dtwc_ipo_state on)
    else()
        set(_dtwc_ipo_state off)
    endif()
    add_test(NAME check_ipo_inlining
             COMMAND "${Python3_EXECUTABLE}" "${CMAKE_SOURCE_DIR}/scripts/check_ipo_inlining.py"
                     "$<TARGET_FILE:dtwc_cl>" --ipo ${_dtwc_ipo_state}
             WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}")
    set_tests_properties(check_ipo_inlining PROPERTIES
        PASS_REGULAR_EXPRESSION "IPO_INLINING binary=dtwc_cl tool=(llvm-objdump|objdump) ipo=${_dtwc_ipo_state} callers=[1-9][0-9]* dist_by_ind_calls=[0-9]+ symbol_present=[01]"
        LABELS "tooling;a06")
endif()
```

Run in WSL (AGENTS.md documents `build/ubsan-wsl`, clang 18):

```bash
wsl.exe --cd /mnt/c/D/git/dtw-cpp bash -c 'cmake -S . -B build/w0-ipo-on -G Ninja -DCMAKE_BUILD_TYPE=Release -DDTWC_BUILD_TESTING=ON -DDTWC_ENABLE_IPO=ON -DDTWC_ENABLE_HIGHS=OFF -DDTWC_ENABLE_GUROBI=OFF -DDTWC_ENABLE_LLFIO=OFF && cmake --build build/w0-ipo-on --target dtwc_cl --parallel && ctest --test-dir build/w0-ipo-on -R check_ipo_inlining -V | grep IPO_INLINING'
wsl.exe --cd /mnt/c/D/git/dtw-cpp bash -c 'cmake -S . -B build/w0-ipo-off -G Ninja -DCMAKE_BUILD_TYPE=Release -DDTWC_BUILD_TESTING=ON -DDTWC_ENABLE_IPO=OFF -DDTWC_ENABLE_HIGHS=OFF -DDTWC_ENABLE_GUROBI=OFF -DDTWC_ENABLE_LLFIO=OFF && cmake --build build/w0-ipo-off --target dtwc_cl --parallel && ctest --test-dir build/w0-ipo-off -R check_ipo_inlining -V | grep IPO_INLINING'
```

Expected: two marker lines, `callers>0`; record both in the run-log (`ipo=on … dist_by_ind_calls=<a>`, `ipo=off … dist_by_ind_calls=<b>`). If `b > 0` and `a == 0`, IPO inlines the call; if `a > 0`, it does not, and A-06's "double preflight per pair" is paid at every call. State only what the numbers show. If WSL lacks `objdump`, `sudo apt install binutils` inside WSL is the fix (record it).

- [ ] **Step 3: Commit**

```bash
git add scripts/check_ipo_inlining.py tests/CMakeLists.txt
git commit -m "test: IPO inlining report for Problem::dist_by_ind in the SWAP kernels (A-06 evidence)"
```

---

### Task 13: Exit gate and records

**Files:**

- Modify: `tests/floors.cmake` (regenerated), `AGENTS.md` (floors), `.claude/baselines/2026-09-07-design-2.0-W0.md`, `.claude/specs/2026-09-07-diff-ledger.md` (status column), `.claude/summaries/handoff-2026-09-07-design-2.0.md`, `CHANGELOG.md`, `.claude/LESSONS.md`

- [ ] **Step 1: Regenerate floors from the finished tree and check they only moved where a task changed a test**

```bash
for d in build/highs-1151 build/nollfio build/arrow-pyarrow-23; do cmake -S . -B $d && cmake --build $d --parallel && ctest --test-dir $d -j1 -V > $d/w0-exit-ctest.log 2>&1; echo "$d exit=$?"; done
cmake -S . -B build/msvc-debug && cmake --build build/msvc-debug --config Debug --parallel && ctest --test-dir build/msvc-debug -C Debug -j1 -V > build/msvc-debug/w0-exit-ctest.log 2>&1; echo "msvc exit=$?"
cp tests/floors.cmake build/w0-floors-before.cmake
uv run --no-project python scripts/measure_test_floors.py build/*/w0-exit-ctest.log --out tests/floors.cmake
diff build/w0-floors-before.cmake tests/floors.cmake
```

Expected: every ctest exits 0; the diff shows only `cpp_conformance` (10;2), `unit_test_allocation_guard` (new), `test_fast_math_consumer` (new, GNU builds), `test_supply_chain_pinning` if it was table-floored (it is explicit: no row) — any other moved floor means a task changed a test's assertion count without saying so: find it in the task's diff and record it.

- [ ] **Step 2: Final serial gate on the regenerated floors**

```bash
for d in build/highs-1151 build/nollfio build/arrow-pyarrow-23; do cmake -S . -B $d && ctest --test-dir $d -j1 2>&1 | tail -3; done
cmake -S . -B build/msvc-debug && ctest --test-dir build/msvc-debug -C Debug -j1 2>&1 | tail -3
ctest --test-dir build/highs-1151 --show-only=json-v1 | uv run --no-project python -c "import json,sys; d=json.load(sys.stdin); print('registered', len(d['tests'])); print('no PASS regex:', [t['name'] for t in d['tests'] if not any(p['name']=='PASS_REGULAR_EXPRESSION' for p in t.get('properties',[]))])"
```

Expected: `100% tests passed` on all four; the JSON check prints `no PASS regex: []` (or `['matlab_suite']` when MATLAB is configured). Record the four result lines and skip counts.

- [ ] **Step 3: AGENTS.md floors**

In `AGENTS.md` "Build & gate recipes", replace the three floor sentences (canonical, llfio-OFF, Arrow-ON) with the new inventories from Step 2, e.g. `Floor (design 2.0 W0, 2026-09-07): ctest → **<n>/<n>, 0 failed**, <k> capability skips (cuda×4, metal×3; test_io_readers is not registered in Arrow-OFF builds)`, and add one bullet: `- **Every CTest entry is floored** through dtwc_add_test (cmake/DtwcTest.cmake, tests/floors.cmake): a skip is a failure unless the registration says MAY_SKIP; regenerate floors with scripts/measure_test_floors.py from ctest -V logs of all matrices after a deliberate test change.`

- [ ] **Step 4: Run-log, ledger, handoff, CHANGELOG, LESSONS**

Run-log: one section per task (question / command / criterion / result / decision) — most were appended as the tasks ran; add the "Exit gate" table and the lists the tasks promised (floors that moved, layer edges, benchmark numbers, IPO markers, dev-warnings ceiling follow-up, llfio superbuild deferral, msvc preset generator finding if any).

Ledger: set `status` to `done` for T-15, T-16, B-12, B-13, B-15, C-08, C-24; `doing` for B-17 (llfio superbuild deferred) and A-06 (evidence registered, decision W4); add the row-level notes in the last column where a task recorded a deviation.

Handoff `.claude/summaries/handoff-2026-09-07-design-2.0.md`: append a "W0 done" section — accomplishments (one line per task), the run-log path, open items (WARNING_MAX pin after first CI run; msvc preset generator; B-17 remainder), and "Resume point: W1 plan (`.claude/plans/2026-09-07-w1-base.md`, to be written with writing-plans)".

LESSONS additions (beyond the two already written): `- CMake's regex has no {m,n}: floor regexes are generated (_dtwc_regex_at_least), never hand-written.`; `- Catch2 v3 prints "All tests passed (A assertions in C test cases)" only when nothing skipped; a partially skipped binary prints "test cases: N | M passed | K skipped" — a floor regex must know which form it is reading.`; `- ctest -V logs are the cheapest floor oracle: "Start N: name" + "N: All tests passed (...)"; take the minimum over configurations.`

CHANGELOG: check every task's line is present under Unreleased (Task 2, 3, 4, 6, 7, 8, 11) and add `- Tests: every CTest entry now has an assertion/case floor and fails on a skip unless registered MAY_SKIP (T-16/B-13); cmake_regex_at_least, cmake_layer_check and check_ipo_inlining tooling tests.`

- [ ] **Step 5: Hygiene scripts and format check on the whole wave**

```bash
uv run --no-project python scripts/check_repo_hygiene.py && uv run --no-project python scripts/check_record_hygiene.py
git clang-format --diff --extensions cpp,hpp,cc,cu,cuh,mm a31956e HEAD -- dtwc tests benchmarks | head -5
git status --short
```

Expected: both `PASS`; empty format diff; no stray files (`build/` is ignored; nothing untracked under `tests/` or `benchmarks/`).

- [ ] **Step 6: Commit**

```bash
git add tests/floors.cmake AGENTS.md .claude/baselines/2026-09-07-design-2.0-W0.md .claude/specs/2026-09-07-diff-ledger.md .claude/summaries/handoff-2026-09-07-design-2.0.md CHANGELOG.md .claude/LESSONS.md
git commit -m "docs: W0 exit gate -- floors regenerated, AGENTS floors, run-log, ledger statuses, handoff"
git log --oneline a31956e..HEAD
```

Expected: 15 commits on `design-2.0` above `a31956e` (Task 0 ×2, 1c, 2, 3, 3b, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13 — Task 1a/1b ride in 1c's commit). Do not push.

---

## Self-review notes

- **Spec coverage.** W0 row items: T-15 → Task 2; T-16 → Tasks 1a–1c; B-12 → Tasks 3, 4, 5; B-13 → Tasks 1a, 3, 4; B-15 → Task 8; B-17 → Task 7 (llfio superbuild explicitly deferred, as the ledger row says); C-08 → Task 6; C-24 → Task 5; layer-manifest check → Task 9; counters + allocation guard → Task 10 (the machine-independent counters of II.6 are library-side and belong to the waves that touch the kernels; W0 delivers the test-side guard and the `items_processed` counts in both benchmarks); two benchmarks → Task 11; IPO check → Task 12; conformance snapshot → Task 2 Step 4 + Task 13 run-log. Exit gate ("three matrices + MSVC Debug at today's floors with every test floored; benchmarks recorded; layer report") → Task 13.
- **Spec wording "today's 9 upward edges":** the number comes from the nine rows of the as-is core map §4a, six of which the manifest ranks lateral (`core → warping*/distance/soft_dtw` are both rank 1). Task 9 records the rank-based count and the reconciliation; the spec's exit-gate cell should be read as "the layer report matches the as-is map".
- **B-14 (option prefix unification)** was approved on 2026-09-07 ("Go") and is Task 3b; Tasks 8 and 12 use the `DTWC_` spellings, Task 3's IPO message is rewritten by 3b.
- **Type consistency:** `dtwc_add_test` keyword names are identical in Task 1a (definition) and Task 1c/2/6/9 (calls); `dtwc_add_script_test` `MARKER` is a regex fragment everywhere; `AllocationCount(min_bytes, exact_bytes)` matches the three migrations; `dtwc_check_layers(<dir> MANIFEST <var> [STRICT ...])` matches the fixture test and the library call; `dtwc_add_benchmark` keywords match Tasks 3 and 11.
