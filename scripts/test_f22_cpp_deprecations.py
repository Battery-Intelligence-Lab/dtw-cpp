#!/usr/bin/env python3
"""Compile the exhaustive F22 C++ compatibility diagnostic fixtures.

The driver reuses the configured compile context of test_problem_api_2_0 so
optional public-header dependencies and feature definitions match the build
under test. It deliberately expects the legacy fixture to fail compilation:
each retained name is compiled under deprecations-as-errors, while a second
suppressed pass proves there is no unrelated syntax/signature failure.

With ``--launch``, the driver remains fail-closed: it runs the same diagnostic
gate first and launches the remaining command only after the exact PASS
conditions hold. The child inherits stdout/stderr so CTest can require both the
compiler marker and the Catch2 behavior marker from one existing test.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from typing import NamedTuple


class DiagnosticGroup(NamedTuple):
    old: str
    replacement: str
    entities: tuple[str, ...]
    function_entities: int
    field_entities: int


GROUPS = (
    DiagnosticGroup(
        "set_numberOfClusters",
        "use set_n_clusters",
        ("Problem::set_numberOfClusters(int)",),
        1,
        0,
    ),
    DiagnosticGroup(
        "refreshDistanceMatrix",
        "use refresh_distance_matrix",
        ("Problem::refreshDistanceMatrix()",),
        1,
        0,
    ),
    DiagnosticGroup(
        "readDistanceMatrix",
        "use read_distance_matrix",
        ("Problem::readDistanceMatrix(const fs::path&)",),
        1,
        0,
    ),
    DiagnosticGroup(
        "maxDistance",
        "use max_distance",
        ("Problem::maxDistance() const",),
        1,
        0,
    ),
    DiagnosticGroup(
        "distByInd",
        "use dist_by_ind",
        ("Problem::distByInd(int,int)",),
        1,
        0,
    ),
    DiagnosticGroup(
        "isDistanceMatrixFilled",
        "use is_distance_matrix_filled",
        ("Problem::isDistanceMatrixFilled() const",),
        1,
        0,
    ),
    DiagnosticGroup(
        "fillDistanceMatrix",
        "use fill_distance_matrix",
        ("Problem::fillDistanceMatrix()",),
        1,
        0,
    ),
    DiagnosticGroup(
        "printDistanceMatrix",
        "use print_distance_matrix",
        ("Problem::printDistanceMatrix() const",),
        1,
        0,
    ),
    DiagnosticGroup(
        "writeDistanceMatrix",
        "use write_distance_matrix",
        (
            "Problem::writeDistanceMatrix(const std::string&) const",
            "Problem::writeDistanceMatrix() const",
        ),
        2,
        0,
    ),
    DiagnosticGroup(
        "printClusters",
        "use print_clusters",
        ("Problem::printClusters() const",),
        1,
        0,
    ),
    DiagnosticGroup(
        "writeClusters",
        "use write_clusters",
        ("Problem::writeClusters()",),
        1,
        0,
    ),
    DiagnosticGroup(
        "writeMedoidMembers",
        "use write_medoid_members",
        ("Problem::writeMedoidMembers(int,int) const",),
        1,
        0,
    ),
    DiagnosticGroup(
        "writeSilhouettes",
        "use write_silhouettes",
        ("Problem::writeSilhouettes()",),
        1,
        0,
    ),
    DiagnosticGroup(
        "findTotalCost",
        "use find_total_cost",
        ("Problem::findTotalCost()",),
        1,
        0,
    ),
    DiagnosticGroup(
        "assignClusters",
        "use assign_clusters",
        ("Problem::assignClusters()",),
        1,
        0,
    ),
    DiagnosticGroup(
        "calculateMedoids",
        "use calculate_medoids",
        ("Problem::calculateMedoids()",),
        1,
        0,
    ),
    DiagnosticGroup(
        "cluster_by_MIP",
        "use cluster_by_mip",
        ("Problem::cluster_by_MIP()",),
        1,
        0,
    ),
    DiagnosticGroup(
        "cluster_by_kMedoidsLloyd",
        "use cluster_by_kmedoids_lloyd",
        ("Problem::cluster_by_kMedoidsLloyd()",),
        1,
        0,
    ),
    DiagnosticGroup(
        "cluster_size",
        "use n_clusters",
        ("Problem::cluster_size() const",),
        1,
        0,
    ),
    DiagnosticGroup(
        "daviesBouldinIndex",
        "use scores::davies_bouldin",
        ("scores::daviesBouldinIndex(Problem&)",),
        1,
        0,
    ),
    DiagnosticGroup(
        "dunnIndex",
        "use scores::dunn",
        ("scores::dunnIndex(Problem&)",),
        1,
        0,
    ),
    DiagnosticGroup(
        "calinskiHarabaszIndex",
        "use scores::calinski_harabasz",
        ("scores::calinskiHarabaszIndex(Problem&)",),
        1,
        0,
    ),
    DiagnosticGroup(
        "adjustedRandIndex",
        "use scores::adjusted_rand",
        ("scores::adjustedRandIndex(labels,labels)",),
        1,
        0,
    ),
    DiagnosticGroup(
        "normalizedMutualInformation",
        "use scores::normalized_mutual_info",
        ("scores::normalizedMutualInformation(labels,labels)",),
        1,
        0,
    ),
    DiagnosticGroup(
        "startColumn",
        "use start_column",
        ("DataLoader::startColumn(int)",),
        1,
        0,
    ),
    DiagnosticGroup(
        "startRow",
        "use start_row",
        ("DataLoader::startRow(int)",),
        1,
        0,
    ),
    DiagnosticGroup(
        "setDataPath",
        "use set_data_path",
        (
            "settings::paths::setDataPath(const fs::path&)",
            "settings::paths::setDataPath(const char*)",
        ),
        2,
        0,
    ),
    DiagnosticGroup(
        "setResultsPath",
        "use set_results_path",
        (
            "settings::paths::setResultsPath(const fs::path&)",
            "settings::paths::setResultsPath(const char*)",
        ),
        2,
        0,
    ),
    DiagnosticGroup(
        "maxIter",
        "use set_max_iter/max_iter",
        ("Problem::maxIter",),
        0,
        1,
    ),
    DiagnosticGroup(
        "N_repetition",
        "use set_n_repetitions/n_repetitions",
        ("Problem::N_repetition",),
        0,
        1,
    ),
)

INHERITED_SILENT = (
    "Problem::readDistanceMatrix(const fs::path&)",
    "Problem::writeDistanceMatrix(const std::string&) const",
    "Problem::writeDistanceMatrix() const",
    "Problem::printClusters() const",
    "Problem::writeClusters()",
    "Problem::writeMedoidMembers(int,int) const",
    "Problem::writeSilhouettes()",
    "Problem::maxIter",
    "Problem::N_repetition",
)

ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--cmake-command", type=Path)
    parser.add_argument("--cmake-config", default="")
    parser.add_argument("--probe-root", type=Path)
    parser.add_argument(
        "--force-cmake-probes",
        action="store_true",
        help="exercise configured object probes even with compile_commands.json",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--launch",
        nargs=argparse.REMAINDER,
        help="after diagnostic PASS, run and relay this command",
    )
    return parser.parse_args()


def normal(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def load_compile_arguments(build_dir: Path, source_dir: Path) -> list[str]:
    database_path = build_dir / "compile_commands.json"
    if not database_path.is_file():
        raise RuntimeError(f"compile database is missing: {database_path}")
    database = json.loads(database_path.read_text(encoding="utf-8"))
    subject = normal(source_dir / "tests/unit/test_problem_api_2_0.cpp")
    records = [record for record in database if normal(record["file"]) == subject]
    if len(records) != 1:
        raise RuntimeError(
            "expected one test_problem_api_2_0 compile command, "
            f"observed {len(records)}"
        )
    record = records[0]
    if "arguments" in record:
        return list(record["arguments"])
    # CMake emits a Windows command string with escaped quote-bearing -D values.
    # posix=False preserves the compiler's backslashes; those quoted -D values
    # are deliberately omitted below because neither fixture consumes them.
    return shlex.split(record["command"], posix=False)


def invoke_cmake_probe(
    *,
    cmake_command: Path,
    build_dir: Path,
    config: str,
    probe_root: Path,
    source_name: str,
    target: str,
    marker: str,
) -> subprocess.CompletedProcess[str]:
    resolved_root = probe_root.resolve()
    try:
        common_root = os.path.commonpath((normal(resolved_root), normal(build_dir)))
    except ValueError as error:
        raise RuntimeError(f"invalid F22 probe root: {resolved_root}") from error
    if (
        common_root != normal(build_dir)
        or resolved_root.name != "f22-cpp-probes"
    ):
        raise RuntimeError(
            "F22 probe root must be the configured build-local "
            f"f22-cpp-probes directory: {resolved_root}"
        )

    source = (resolved_root / source_name).resolve()
    if source.parent != resolved_root or not source.is_file():
        raise RuntimeError(f"F22 probe source is missing or escaped: {source}")

    # Successful object probes may already exist from an earlier CTest run.
    # Touch the wrapper and require its pragma marker below; a coarse filesystem
    # that misses the rebuild therefore fails closed without clock skew.
    source.touch()
    command = [
        os.fspath(cmake_command),
        "--build",
        os.fspath(build_dir),
        "--target",
        target,
    ]
    if config:
        command.extend(("--config", config))
    result = subprocess.run(
        command,
        cwd=build_dir,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if marker not in ANSI_ESCAPE.sub("", result.stdout):
        raise RuntimeError(
            f"F22 probe target did not compile {source.name}: {target}"
        )
    return result


def compile_context(arguments: list[str]) -> tuple[str, list[str]]:
    if not arguments:
        raise RuntimeError("empty compile command")
    compiler = arguments[0].strip('"')
    compiler_name = Path(compiler).name.lower()
    msvc_slash_flags = compiler_name in ("cl", "cl.exe") or "clang-cl" in compiler_name
    context: list[str] = []
    index = 1
    paired_options = {
        "-isystem",
        "-idirafter",
        "-iquote",
        "--sysroot",
        "-target",
        "--target",
        # Apple Clang rejects a bare -fopenmp; CMake records
        # -Xpreprocessor -fopenmp (and sometimes -Xclang -fopenmp).
        "-Xpreprocessor",
        "-Xclang",
    }
    define_include_prefixes = ("-I", "-D", "-U")
    if msvc_slash_flags:
        define_include_prefixes += ("/I", "/D", "/U")
    while index < len(arguments):
        argument = arguments[index]
        if argument in paired_options:
            if index + 1 >= len(arguments):
                raise RuntimeError(f"compile option has no value: {argument}")
            context.extend((argument, arguments[index + 1].strip('"')))
            index += 2
            continue
        if (
            argument.startswith(define_include_prefixes)
            and '\\"' not in argument
        ):
            context.append(argument.strip('"'))
        elif argument.startswith(("-std=", "/std:", "-stdlib=")):
            context.append(argument)
        elif argument in ("-fopenmp", "/openmp", "/openmp:experimental"):
            context.append(argument)
        index += 1
    return compiler, context


def invoke(
    compiler: str,
    context: list[str],
    fixture: Path,
    *,
    diagnostics_as_errors: bool,
    suppress_diagnostics: bool,
) -> subprocess.CompletedProcess[str]:
    compiler_name = Path(compiler).name.lower()
    if compiler_name in ("cl", "cl.exe"):
        mode = ["/nologo", "/Zs", "/TP"]
        if diagnostics_as_errors:
            mode.append("/we4996")
        if suppress_diagnostics:
            mode.append("/wd4996")
    elif "clang-cl" in compiler_name:
        mode = ["/nologo", "/Zs", "/TP", "/clang:-fdiagnostics-color=never"]
        if diagnostics_as_errors:
            mode.append("/clang:-Werror=deprecated-declarations")
        if suppress_diagnostics:
            mode.append("/clang:-Wno-deprecated-declarations")
    else:
        mode = ["-x", "c++", "-fsyntax-only", "-fdiagnostics-color=never"]
        if "clang" in compiler_name:
            mode.append("-ferror-limit=0")
        else:
            mode.append("-fmax-errors=0")
        if diagnostics_as_errors:
            mode.append("-Werror=deprecated-declarations")
        if suppress_diagnostics:
            mode.append("-Wno-deprecated-declarations")
    return subprocess.run(
        [compiler, *context, *mode, os.fspath(fixture)],
        cwd=fixture.parent,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def diagnostic_lines(output: str) -> list[str]:
    clean = ANSI_ESCAPE.sub("", output)
    return [
        line
        for line in clean.splitlines()
        if "deprecated" in line.lower()
        and ("warning" in line.lower() or "error" in line.lower())
    ]


def observed_inventory(output: str) -> tuple[int, list[str], int, int]:
    lines = diagnostic_lines(output)
    total = 0
    functions = 0
    fields = 0
    silent: list[str] = []
    for group in GROUPS:
        matches = [
            line
            for line in lines
            if group.old in line and group.replacement in line
        ]
        observed = min(len(matches), len(group.entities))
        total += observed
        if group.function_entities:
            functions += observed
        if group.field_entities:
            fields += observed
        silent.extend(group.entities[observed:])
    return total, silent, functions, fields


def main() -> int:
    args = parse_args()
    source_dir = (
        args.source_dir.resolve()
        if args.source_dir
        else Path(__file__).resolve().parents[1]
    )
    build_dir = args.build_dir.resolve()
    legacy_fixture = source_dir / "tests/fixtures/f22_cpp_legacy_diagnostics.inc"
    canonical_fixture = (
        source_dir / "tests/fixtures/f22_cpp_canonical_diagnostics.inc"
    )

    try:
        if (
            (build_dir / "compile_commands.json").is_file()
            and not args.force_cmake_probes
        ):
            arguments = load_compile_arguments(build_dir, source_dir)
            compiler, context = compile_context(arguments)
            legacy = invoke(
                compiler,
                context,
                legacy_fixture,
                diagnostics_as_errors=True,
                suppress_diagnostics=False,
            )
            legacy_control = invoke(
                compiler,
                context,
                legacy_fixture,
                diagnostics_as_errors=False,
                suppress_diagnostics=True,
            )
            canonical = invoke(
                compiler,
                context,
                canonical_fixture,
                diagnostics_as_errors=True,
                suppress_diagnostics=False,
            )
        else:
            if args.cmake_command is None or args.probe_root is None:
                raise RuntimeError(
                    "compile database is missing and configured CMake probes "
                    "were not supplied"
                )
            probe_common = {
                "cmake_command": args.cmake_command.resolve(),
                "build_dir": build_dir,
                "config": args.cmake_config,
                "probe_root": args.probe_root,
            }
            legacy = invoke_cmake_probe(
                **probe_common,
                source_name="f22_cpp_legacy_werror.cpp",
                target="f22_cpp_legacy_werror",
                marker="F22_PROBE_LEGACY_WERROR_RAN",
            )
            legacy_control = invoke_cmake_probe(
                **probe_common,
                source_name="f22_cpp_legacy_suppressed.cpp",
                target="f22_cpp_legacy_suppressed",
                marker="F22_PROBE_LEGACY_SUPPRESSED_RAN",
            )
            canonical = invoke_cmake_probe(
                **probe_common,
                source_name="f22_cpp_canonical_werror.cpp",
                target="f22_cpp_canonical_werror",
                marker="F22_PROBE_CANONICAL_WERROR_RAN",
            )
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"F22_CPP_HARNESS_ERROR {error}")
        return 2

    legacy_count, silent, function_count, field_count = observed_inventory(
        legacy.stdout
    )
    canonical_deprecations = diagnostic_lines(canonical.stdout)
    signatures_ok = legacy_control.returncode == 0
    canonical_ok = canonical.returncode == 0 and not canonical_deprecations
    inventory_ok = sum(len(group.entities) for group in GROUPS) == 33
    type_inventory_ok = (
        sum(group.function_entities for group in GROUPS) == 31
        and sum(group.field_entities for group in GROUPS) == 2
    )

    silent_text = ",".join(silent) if silent else "none"
    print(
        "F22_CPP_SILENT "
        f"count={len(silent)}/33 entities={silent_text} "
        f"canonical_deprecation_lines={len(canonical_deprecations)}"
    )

    final_ok = (
        inventory_ok
        and type_inventory_ok
        and signatures_ok
        and canonical_ok
        and legacy.returncode != 0
        and legacy_count == 33
        and not silent
        and function_count == 31
        and field_count == 2
    )
    inherited_red = (
        inventory_ok
        and type_inventory_ok
        and signatures_ok
        and canonical_ok
        and legacy.returncode != 0
        and legacy_count == 24
        and tuple(silent) == INHERITED_SILENT
        and function_count == 24
        and field_count == 0
    )

    if final_ok:
        verdict = "PASS"
    elif inherited_red:
        verdict = "EXPECTED_RED"
    else:
        verdict = "FAIL"
    print(
        "F22_CPP_DIAGNOSTICS "
        f"inventory={33 if inventory_ok else 0}/33 "
        f"legacy={legacy_count}/33 "
        f"canonical_silent={33 if canonical_ok else 0}/33 "
        f"overloads={31 if signatures_ok and type_inventory_ok else 0}/31 "
        f"fields={2 if signatures_ok and type_inventory_ok else 0}/2 "
        f"skips=0 verdict={verdict}"
    )

    if args.verbose or verdict == "FAIL":
        print("F22_CPP_LEGACY_OUTPUT_BEGIN")
        print(ANSI_ESCAPE.sub("", legacy.stdout).rstrip())
        print("F22_CPP_LEGACY_OUTPUT_END")
        print("F22_CPP_SUPPRESSION_OUTPUT_BEGIN")
        print(ANSI_ESCAPE.sub("", legacy_control.stdout).rstrip())
        print("F22_CPP_SUPPRESSION_OUTPUT_END")
        print("F22_CPP_CANONICAL_OUTPUT_BEGIN")
        print(ANSI_ESCAPE.sub("", canonical.stdout).rstrip())
        print("F22_CPP_CANONICAL_OUTPUT_END")

    if not final_ok:
        return 1
    if args.launch is None:
        return 0
    if not args.launch:
        print("F22_CPP_HARNESS_ERROR --launch requires a command")
        return 2

    # Preserve diagnostic-before-behavior output ordering under CTest pipes.
    sys.stdout.flush()
    sys.stderr.flush()
    try:
        launched = subprocess.run(args.launch, check=False)
    except OSError as error:
        print(f"F22_CPP_HARNESS_ERROR launch failed: {error}")
        return 2
    return launched.returncode


if __name__ == "__main__":
    sys.exit(main())
