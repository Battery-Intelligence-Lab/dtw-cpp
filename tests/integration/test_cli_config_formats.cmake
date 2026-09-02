cmake_minimum_required(VERSION 3.26)

# --config accepts TOML or YAML through CLI11's own config interface, so CLI11
# keeps precedence: an explicitly given flag must beat the file. The removed
# hand-rolled YAML loader failed exactly that, so every case here drives the real
# binary and reads its verbose echo of the settings it actually used.
foreach(required_var IN ITEMS CLI INPUT TOML WORK_ROOT HAS_YAML)
    if(NOT DEFINED ${required_var} OR "${${required_var}}" STREQUAL "")
        message(FATAL_ERROR "missing required -D${required_var}=...")
    endif()
endforeach()

cmake_path(ABSOLUTE_PATH CLI NORMALIZE OUTPUT_VARIABLE cli_path)
if(NOT EXISTS "${cli_path}")
    message(FATAL_ERROR "CLI does not exist: ${cli_path}")
endif()

file(REMOVE_RECURSE "${WORK_ROOT}")
file(MAKE_DIRECTORY "${WORK_ROOT}")

set(checks 0)

function(run_cli case)
    execute_process(
        COMMAND "${cli_path}" ${ARGN}
        RESULT_VARIABLE result
        OUTPUT_VARIABLE stdout
        ERROR_VARIABLE stderr
        ENCODING UTF-8)
    if(NOT "${result}" MATCHES "^[0-9]+$")
        message(FATAL_ERROR "${case}: CLI did not exit normally: ${result}")
    endif()
    set(exit_code "${result}" PARENT_SCOPE)
    set(combined "${stdout}\n${stderr}" PARENT_SCOPE)
endfunction()

macro(expect case pattern)
    if(NOT "${combined}" MATCHES "${pattern}")
        message(FATAL_ERROR "${case}: no match for '${pattern}'\n${combined}")
    endif()
    math(EXPR checks "${checks} + 1")
endmacro()

macro(expect_zero case)
    if(NOT "${exit_code}" STREQUAL "0")
        message(FATAL_ERROR "${case}: expected success, exit=${exit_code}\n${combined}")
    endif()
    math(EXPR checks "${checks} + 1")
endmacro()

macro(expect_nonzero case)
    if("${exit_code}" STREQUAL "0")
        message(FATAL_ERROR "${case}: expected failure, exit=0\n${combined}")
    endif()
    math(EXPR checks "${checks} + 1")
endmacro()

set(yaml "${WORK_ROOT}/settings.yaml")
file(WRITE "${yaml}"
"# canonical keys, kebab-case, identical to the TOML spelling\n"
"n-clusters: 4\n"
"method: kmedoids\n"
"band: 5\n"
"metric: l1\n"
"variant: standard\n"
"max-iter: 100\n"
"name: yamlcase\n"
"verbose: true\n")

set(deprecated_only "${WORK_ROOT}/deprecated_only.yaml")
file(WRITE "${deprecated_only}" "clusters: 6\nmethod: kmedoids\nverbose: true\n")

set(deprecated_both "${WORK_ROOT}/deprecated_both.yaml")
file(WRITE "${deprecated_both}"
    "clusters: 6\nn-clusters: 2\nmethod: kmedoids\nverbose: true\n")

set(unknown_key "${WORK_ROOT}/unknown_key.yaml")
file(WRITE "${unknown_key}" "not-a-real-option: 1\nverbose: true\n")

set(null_value "${WORK_ROOT}/null_value.yaml")
file(WRITE "${null_value}" "band: ~\nverbose: true\n")

set(malformed "${WORK_ROOT}/invalid.yaml")
file(WRITE "${malformed}" "method: [pam, kmedoids\nverbose: true\n")

if(HAS_YAML)
    # (a) a YAML file supplies the settings the run actually uses.
    run_cli(yaml_applies --config "${yaml}" -i "${INPUT}" -o "${WORK_ROOT}/a")
    expect_zero(yaml_applies)
    expect(yaml_applies "Clusters:[ \t]+4")
    expect(yaml_applies "Method:[ \t]+kmedoids")
    expect(yaml_applies "Band:[ \t]+5")

    # (b) THE regression: an explicit command-line flag beats the YAML value.
    run_cli(cli_beats_yaml
        --config "${yaml}" -i "${INPUT}" -o "${WORK_ROOT}/b" --n-clusters 2)
    expect_zero(cli_beats_yaml)
    expect(cli_beats_yaml "Clusters:[ \t]+2")
    if("${combined}" MATCHES "Clusters:[ \t]+4")
        message(FATAL_ERROR "cli_beats_yaml: YAML overrode the explicit flag\n${combined}")
    endif()
    math(EXPR checks "${checks} + 1")

    # (d) a deprecated key from YAML warns, and yields to the canonical key.
    run_cli(deprecated_only
        --config "${deprecated_only}" -i "${INPUT}" -o "${WORK_ROOT}/d1")
    expect_zero(deprecated_only)
    expect(deprecated_only "'--clusters' is deprecated, use '--n-clusters' instead")
    expect(deprecated_only "Clusters:[ \t]+6")

    run_cli(deprecated_both
        --config "${deprecated_both}" -i "${INPUT}" -o "${WORK_ROOT}/d2")
    expect_zero(deprecated_both)
    expect(deprecated_both "'--clusters' is deprecated, use '--n-clusters' instead")
    expect(deprecated_both "Clusters:[ \t]+2")

    # (e) an unknown YAML key is CLI11's error, naming the key.
    run_cli(unknown_key --config "${unknown_key}" -i "${INPUT}" -o "${WORK_ROOT}/e")
    expect_nonzero(unknown_key)
    expect(unknown_key "not-a-real-option")

    # (g) a YAML null is NOT "unset": CLI11 would take "" as a supplied value,
    #     so `band: ~` silently became band 0. It must be a loud error.
    run_cli(null_value --config "${null_value}" -i "${INPUT}" -o "${WORK_ROOT}/g")
    expect_nonzero(null_value)
    expect(null_value "band.*is null")

    # (f) malformed YAML fails and names the file.
    run_cli(malformed --config "${malformed}" -i "${INPUT}" -o "${WORK_ROOT}/f")
    expect_nonzero(malformed)
    expect(malformed "invalid\\.yaml")
else()
    # Built without YAML: the guard lives outside the #ifdef, so it must fire
    # here rather than silently parsing the file as TOML.
    run_cli(yaml_disabled --config "${yaml}" -i "${INPUT}" -o "${WORK_ROOT}/a")
    expect_nonzero(yaml_disabled)
    expect(yaml_disabled "built without YAML support; use TOML")
    expect(yaml_disabled "settings\\.yaml")
endif()

# (c) TOML still parses through the same --config option and the same formatter.
run_cli(toml_applies --config "${TOML}" -i "${INPUT}" -o "${WORK_ROOT}/c" --verbose)
expect_zero(toml_applies)
expect(toml_applies "Clusters:[ \t]+3")
expect(toml_applies "Method:[ \t]+pam")
expect(toml_applies "Band:[ \t]+3")

if("${combined}" MATCHES "[Ss][Kk][Ii][Pp]([Pp]|[ :])")
    message(FATAL_ERROR "CLI emitted skip text:\n${combined}")
endif()

message(STATUS
    "CLI_CONFIG_FORMATS subject=real_dtwc_cl yaml=${HAS_YAML} "
    "checks=${checks} skips=0")
