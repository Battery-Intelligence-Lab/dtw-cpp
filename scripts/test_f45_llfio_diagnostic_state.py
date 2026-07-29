#!/usr/bin/env python3
"""Verify that LLFIO public-header boundaries preserve Clang diagnostics."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


SENTINEL_NAME = "f45_downstream_deprecated"
SENTINEL_MESSAGE = "F45 downstream deprecation sentinel"
RAW_INCLUDE = re.compile(
    r'^\s*#\s*include\s*[<"]llfio/v2\.0/llfio\.hpp[>"]\s*$'
)
WRAPPER_INCLUDE = re.compile(
    r'^\s*#\s*include\s*"llfio_include\.hpp"\s*$'
)
ANSI_ESCAPE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
ERROR_LINE = re.compile(r"\b(?:fatal )?error(?:\s+[A-Z]+\d+)?:", re.IGNORECASE)
HEADER_SUFFIXES = {".h", ".hh", ".hpp", ".hxx", ".cuh", ".inl", ".ipp"}
FINAL_MARKER = (
    "F45_LLFIO_DIAGNOSTIC_STATE probes=4/4 diagnostics=4/4 "
    "raw_include=1/1 wrapper_routes=2/2 skips=0 verdict=PASS"
)


@dataclass(frozen=True)
class Probe:
    name: str
    includes: tuple[str, ...]
    llfio_on: bool


PROBES = (
    Probe("wrapper", ("core/llfio_include.hpp",), True),
    Probe("mmap_data_store", ("core/mmap_data_store.hpp",), True),
    Probe("mmap_distance_matrix", ("core/mmap_distance_matrix.hpp",), True),
    Probe(
        "llfio_off",
        ("core/mmap_data_store.hpp", "core/mmap_distance_matrix.hpp"),
        False,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("build/highs-1151"))
    parser.add_argument(
        "--off-build-dir", type=Path, default=Path("build/nollfio")
    )
    parser.add_argument("--source-dir", type=Path)
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def normal(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def require_inside(path: Path, root: Path, description: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"{description} escapes the repository root: {path}"
        ) from error


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
    return shlex.split(record["command"], posix=False)


def compile_context(arguments: list[str]) -> tuple[str, list[str]]:
    if not arguments:
        raise RuntimeError("empty compile command")
    compiler = arguments[0].strip('"')
    if "clang" not in Path(compiler).name.lower():
        raise RuntimeError(f"F45 requires a Clang compile context: {compiler}")

    context: list[str] = []
    paired_options = {
        "-isystem",
        "-idirafter",
        "-iquote",
        "--sysroot",
        "-target",
        "--target",
    }
    index = 1
    while index < len(arguments):
        argument = arguments[index]
        if argument in paired_options:
            if index + 1 >= len(arguments):
                raise RuntimeError(f"compile option has no value: {argument}")
            context.extend((argument, arguments[index + 1].strip('"')))
            index += 2
            continue
        if (
            argument.startswith(("-I", "/I", "-D", "/D", "-U", "/U"))
            and '\\"' not in argument
        ):
            context.append(argument.strip('"'))
        elif argument.startswith(("-std=", "/std:", "-stdlib=")):
            context.append(argument)
        elif argument in ("-fopenmp", "/openmp", "/openmp:experimental"):
            context.append(argument)
        index += 1

    return compiler, context


def is_llfio_input(argument: str) -> bool:
    lowered = argument.lower().replace("\\", "/")
    return (
        "dtwc_has_mmap" in lowered
        or "ntkernel_error_category" in lowered
        or "/_deps/llfio-" in lowered
        or "/llfio/v2.0" in lowered
    )


def validate_context(context: list[str], *, llfio_on: bool) -> None:
    has_feature = any(
        item.lower() in ("-ddtwc_has_mmap", "/ddtwc_has_mmap")
        for item in context
    )
    has_llfio_input = any(is_llfio_input(item) for item in context)
    if llfio_on and (not has_feature or not has_llfio_input):
        raise RuntimeError(
            "canonical compile context omits DTWC_HAS_MMAP or LLFIO inputs"
        )
    if not llfio_on and (has_feature or has_llfio_input):
        raise RuntimeError(
            "LLFIO-OFF compile context contains DTWC_HAS_MMAP or LLFIO inputs"
        )


def probe_source(probe: Probe) -> str:
    includes = "".join(f"#include <{header}>\n" for header in probe.includes)
    return (
        includes
        + "\n"
        + f'[[deprecated("{SENTINEL_MESSAGE}")]]\n'
        + f"inline void {SENTINEL_NAME}() {{}}\n\n"
        + f"void f45_use_sentinel() {{ {SENTINEL_NAME}(); }}\n"
    )


def invoke(
    compiler: str, context: list[str], source_path: Path
) -> subprocess.CompletedProcess[str]:
    compiler_name = Path(compiler).name.lower()
    if "clang-cl" in compiler_name:
        mode = [
            "/nologo",
            "/Zs",
            "/TP",
            "/clang:-fdiagnostics-color=never",
            "/clang:-ferror-limit=0",
            "/clang:-Werror=deprecated-declarations",
        ]
    else:
        mode = [
            "-x",
            "c++",
            "-fsyntax-only",
            "-fdiagnostics-color=never",
            "-ferror-limit=0",
            "-Werror=deprecated-declarations",
        ]
    return subprocess.run(
        [compiler, *context, *mode, os.fspath(source_path)],
        cwd=source_path.parent,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )


def error_lines(output: str) -> list[str]:
    clean = ANSI_ESCAPE.sub("", output)
    return [line for line in clean.splitlines() if ERROR_LINE.search(line)]


def audit_tracked_headers(source_dir: Path) -> tuple[bool, int, int]:
    tracked = subprocess.run(
        [
            "git",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
            "dtwc",
        ],
        cwd=source_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if tracked.returncode != 0:
        raise RuntimeError(
            "git ls-files failed: "
            + tracked.stderr.decode("utf-8", errors="replace").strip()
        )

    raw_sites: list[tuple[str, int]] = []
    wrapper_sites: list[tuple[str, int]] = []
    for encoded_path in tracked.stdout.split(b"\0"):
        if not encoded_path:
            continue
        relative = encoded_path.decode("utf-8")
        path = source_dir / relative
        if path.suffix.lower() not in HEADER_SUFFIXES:
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), start=1
        ):
            if RAW_INCLUDE.fullmatch(line):
                raw_sites.append((relative, line_number))
            if WRAPPER_INCLUDE.fullmatch(line):
                wrapper_sites.append((relative, line_number))

    expected_raw_path = "dtwc/core/llfio_include.hpp"
    expected_wrapper_paths = {
        "dtwc/core/mmap_data_store.hpp",
        "dtwc/core/mmap_distance_matrix.hpp",
    }
    wrapper_paths = {path for path, _ in wrapper_sites}
    audit_ok = (
        len(raw_sites) == 1
        and raw_sites[0][0] == expected_raw_path
        and len(wrapper_sites) == 2
        and wrapper_paths == expected_wrapper_paths
    )
    print(
        "F45_SOURCE_AUDIT "
        f"raw_include={len(raw_sites)}/1 "
        f"wrapper_routes={len(wrapper_sites)}/2 "
        f"verdict={'PASS' if audit_ok else 'FAIL'}"
    )
    return audit_ok, len(raw_sites), len(wrapper_sites)


def main() -> int:
    args = parse_args()
    source_dir = (
        args.source_dir.resolve()
        if args.source_dir
        else Path(__file__).resolve().parents[1]
    )
    build_dir = (
        args.build_dir.resolve()
        if args.build_dir.is_absolute()
        else (source_dir / args.build_dir).resolve()
    )
    off_build_dir = (
        args.off_build_dir.resolve()
        if args.off_build_dir.is_absolute()
        else (source_dir / args.off_build_dir).resolve()
    )

    try:
        require_inside(build_dir, source_dir, "build directory")
        require_inside(off_build_dir, source_dir, "LLFIO-OFF build directory")
        on_compiler, on_context = compile_context(
            load_compile_arguments(build_dir, source_dir)
        )
        off_compiler, off_context = compile_context(
            load_compile_arguments(off_build_dir, source_dir)
        )
        validate_context(on_context, llfio_on=True)
        validate_context(off_context, llfio_on=False)
        if normal(on_compiler) != normal(off_compiler):
            raise RuntimeError(
                "LLFIO-ON/OFF compile contexts use different compilers: "
                f"{on_compiler} != {off_compiler}"
            )
        probe_dir = build_dir / "f45-llfio-diagnostic-state"
        require_inside(probe_dir, source_dir, "probe directory")
        probe_dir.mkdir(parents=True, exist_ok=True)

        passed = 0
        diagnostics = 0
        failed_outputs: list[tuple[str, str]] = []
        for probe in PROBES:
            source_path = probe_dir / f"{probe.name}.cpp"
            source_path.write_text(
                probe_source(probe), encoding="utf-8", newline="\n"
            )
            result = invoke(
                on_compiler if probe.llfio_on else off_compiler,
                on_context if probe.llfio_on else off_context,
                source_path,
            )
            errors = error_lines(result.stdout)
            diagnostic_ok = (
                result.returncode != 0
                and len(errors) == 1
                and SENTINEL_NAME in errors[0]
                and SENTINEL_MESSAGE in errors[0]
                and "deprecated" in errors[0].lower()
            )
            diagnostics += int(diagnostic_ok)
            passed += int(diagnostic_ok)
            print(
                "F45_PROBE "
                f"name={probe.name} "
                f"mode={'llfio-on' if probe.llfio_on else 'llfio-off'} "
                f"diagnostic={1 if diagnostic_ok else 0}/1 "
                f"errors={len(errors)} "
                f"verdict={'PASS' if diagnostic_ok else 'FAIL'}"
            )
            if args.verbose or not diagnostic_ok:
                failed_outputs.append((probe.name, result.stdout))

        audit_ok, raw_count, wrapper_count = audit_tracked_headers(source_dir)
        final_ok = (
            passed == len(PROBES)
            and diagnostics == len(PROBES)
            and audit_ok
            and raw_count == 1
            and wrapper_count == 2
        )
        marker = (
            "F45_LLFIO_DIAGNOSTIC_STATE "
            f"probes={passed}/4 diagnostics={diagnostics}/4 "
            f"raw_include={raw_count}/1 wrapper_routes={wrapper_count}/2 "
            f"skips=0 verdict={'PASS' if final_ok else 'FAIL'}"
        )
        print(marker)

        for name, output in failed_outputs:
            print(f"F45_PROBE_OUTPUT_BEGIN name={name}")
            print(ANSI_ESCAPE.sub("", output).rstrip())
            print(f"F45_PROBE_OUTPUT_END name={name}")

        if final_ok and marker != FINAL_MARKER:
            raise RuntimeError(f"deterministic marker drift: {marker}")
        return 0 if final_ok else 1
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as error:
        print(f"F45_HARNESS_ERROR {error}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
