#!/usr/bin/env python3
"""Compile and source-audit the preregistered F19 encapsulation contract."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "scripts" / "fixtures" / "f19_problem_encapsulation.cpp"
PROBLEM_HEADER = ROOT / "dtwc" / "Problem.hpp"
MATLAB_MEX = ROOT / "bindings" / "matlab" / "dtwc_mex.cpp"
PYTHON_BINDING = ROOT / "python" / "src" / "_dtwcpp_core.cpp"
AUTHORITATIVE_WRITEBACK_FILES = {
    "fast_pam": ROOT / "dtwc" / "algorithms" / "fast_pam.cpp",
    "fast_clara": ROOT / "dtwc" / "algorithms" / "fast_clara.cpp",
    "clarans": ROOT / "dtwc" / "algorithms" / "clarans.cpp",
    "hierarchical": ROOT / "dtwc" / "algorithms" / "hierarchical.cpp",
}

PRIVATE_FIELDS = (
    "method",
    "random_seed",
    "last_iterations",
    "tadpole_dc",
    "lb_strategy",
    "storage_policy",
    "verbose",
    "output_folder",
    "name",
    "data",
)

RAW_PRIVATE_DECLARATIONS = {
    "method": r"(?m)^\s*Method\s+method\s*\{",
    "random_seed": r"(?m)^\s*std::uint64_t\s+random_seed\s*\{",
    "last_iterations": r"(?m)^\s*int\s+last_iterations\s*\{",
    "tadpole_dc": r"(?m)^\s*double\s+tadpole_dc\s*\{",
    "lb_strategy": r"(?m)^\s*LowerBoundStrategy\s+lb_strategy\s*\{",
    "storage_policy": r"(?m)^\s*core::StoragePolicy\s+storage_policy\s*\{",
    "verbose": r"(?m)^\s*bool\s+verbose\s*\{",
    "output_folder": r"(?m)^\s*path_t\s+output_folder\s*\{",
    "name": r"(?m)^\s*std::string\s+name\s*\{",
    "data": r"(?m)^\s*Data\s+data\s*;",
}

PRIVATE_BACKING_DECLARATIONS = {
    "method_": r"(?m)^\s*Method\s+method_\s*(?:\{|;)",
    "random_seed_": r"(?m)^\s*std::uint64_t\s+random_seed_\s*(?:\{|;)",
    "last_iterations_": r"(?m)^\s*int\s+last_iterations_\s*(?:\{|;)",
    "tadpole_dc_": r"(?m)^\s*double\s+tadpole_dc_\s*(?:\{|;)",
    "lb_strategy_": (
        r"(?m)^\s*LowerBoundStrategy\s+lb_strategy_\s*(?:\{|;)"
    ),
    "storage_policy_": (
        r"(?m)^\s*core::StoragePolicy\s+storage_policy_\s*(?:\{|;)"
    ),
    "verbose_": r"(?m)^\s*bool\s+verbose_\s*(?:\{|;)",
    "output_folder_": r"(?m)^\s*path_t\s+output_folder_\s*(?:\{|;)",
    "name_": r"(?m)^\s*std::string\s+name_\s*(?:\{|;)",
    "data_": r"(?m)^\s*Data\s+data_\s*(?:\{|;)",
}

MEX_REDUNDANT_WRITEBACKS = {
    "cluster_count": (
        r"(?m)^\s*prob\.set_n_clusters\s*\(\s*k\s*\)\s*;\s*$"
    ),
    "medoids": (
        r"(?m)^\s*prob\.centroids_ind\s*=\s*"
        r"result\.medoid_indices\s*;\s*$"
    ),
    "labels": (
        r"(?m)^\s*prob\.clusters_ind\s*=\s*result\.labels\s*;\s*$"
    ),
}

AUTHORITATIVE_WRITEBACKS = {
    "fast_pam": {
        "cluster_count": (
            r"(?m)^\s*prob\.set_n_clusters\s*\(\s*k\s*\)\s*;\s*$"
        ),
        "medoids": (
            r"(?m)^\s*prob\.centroids_ind\s*=\s*"
            r"result\.medoid_indices\s*;\s*$"
        ),
        "labels": (
            r"(?m)^\s*prob\.clusters_ind\s*=\s*result\.labels\s*;\s*$"
        ),
    },
    "fast_clara": {
        "cluster_count": (
            r"(?m)^\s*prob\.set_n_clusters\s*\(\s*"
            r"opts\.n_clusters\s*\)\s*;\s*$"
        ),
        "medoids": (
            r"(?m)^\s*prob\.centroids_ind\s*=\s*"
            r"best_result\.medoid_indices\s*;\s*$"
        ),
        "labels": (
            r"(?m)^\s*prob\.clusters_ind\s*=\s*"
            r"best_result\.labels\s*;\s*$"
        ),
    },
    "clarans": {
        "cluster_count": (
            r"(?m)^\s*prob\.set_n_clusters\s*\(\s*"
            r"opts\.n_clusters\s*\)\s*;\s*$"
        ),
        "medoids": (
            r"(?m)^\s*prob\.centroids_ind\s*=\s*"
            r"best\.medoid_indices\s*;\s*$"
        ),
        "labels": (
            r"(?m)^\s*prob\.clusters_ind\s*=\s*best\.labels\s*;\s*$"
        ),
    },
    "hierarchical": {
        "cluster_count": (
            r"(?m)^\s*prob\.set_n_clusters\s*\(\s*k\s*\)\s*;\s*$"
        ),
        "medoids": (
            r"(?m)^\s*prob\.centroids_ind\s*=\s*"
            r"result\.medoid_indices\s*;\s*$"
        ),
        "labels": (
            r"(?m)^\s*prob\.clusters_ind\s*=\s*result\.labels\s*;\s*$"
        ),
    },
}

SETTER_EXERCISE = (
    "set_method",
    "set_random_seed",
    "set_tadpole_dc",
    "set_lb_strategy",
    "set_storage_policy",
    "set_verbose",
    "set_output_folder",
    "set_name",
    "set_data",
)

PRIVATE_DIAGNOSTICS = tuple(
    "F19_PRIVATE_data_and_resize" if field == "data"
    else f"F19_PRIVATE_{field}"
    for field in PRIVATE_FIELDS
)
GETTER_DIAGNOSTICS = tuple(
    f"F19_GETTER_{field}" for field in PRIVATE_FIELDS
)
INHERITED_MISSING_SETTERS = (
    "set_tadpole_dc",
    "set_verbose",
    "set_output_folder",
    "set_name",
)

EXPECTED = {
    "inherited": {
        "compile_success": False,
        "declaration_count": 1,
        "backing_declaration_count": 0,
        "violations": 10,
        "backings": 0,
        "helpers": 1,
        "calls": 4,
        "mex_write_count": 1,
        "mex_writes": 3,
        "core_write_count": 1,
        "core_writes": 12,
        "python_direct_accesses": {
            "method": 1,
            "random_seed": 1,
            "last_iterations": 0,
            "tadpole_dc": 0,
            "lb_strategy": 1,
            "storage_policy": 1,
            "verbose": 1,
            "output_folder": 2,
            "name": 2,
            "data": 0,
        },
        "matlab_direct_accesses": {
            "method": 0,
            "random_seed": 0,
            "last_iterations": 0,
            "tadpole_dc": 0,
            "lb_strategy": 0,
            "storage_policy": 0,
            "verbose": 5,
            "output_folder": 1,
            "name": 2,
            "data": 0,
        },
    },
    "final": {
        "compile_success": True,
        "declaration_count": 0,
        "backing_declaration_count": 1,
        "violations": 0,
        "backings": 10,
        "helpers": 0,
        "calls": 0,
        "mex_write_count": 0,
        "mex_writes": 0,
        "core_write_count": 1,
        "core_writes": 12,
        "python_direct_accesses": dict.fromkeys(PRIVATE_FIELDS, 0),
        "matlab_direct_accesses": dict.fromkeys(PRIVATE_FIELDS, 0),
    },
}


class GateFailure(RuntimeError):
    """A preregistered F19 gate condition was not met."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect",
        choices=tuple(EXPECTED),
        required=True,
        help="Explicitly select the inherited red or final green contract.",
    )
    parser.add_argument(
        "--compiler",
        help="C++20 compiler executable (defaults to CXX, clang++, then g++).",
    )
    parser.add_argument(
        "--rapidcsv-include",
        type=Path,
        help=(
            "Directory containing rapidcsv.h (defaults to a fetched copy "
            "under build/)."
        ),
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise GateFailure(f"cannot read {path}: {error}") from error


def find_compiler(requested: str | None) -> str:
    candidates = []
    if requested:
        candidates.append(requested)
    elif os.environ.get("CXX"):
        candidates.append(os.environ["CXX"])
    else:
        candidates.extend(("clang++", "g++"))

    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    raise GateFailure(
        "no C++ compiler found; pass --compiler or set CXX "
        "(searched: " + ", ".join(candidates) + ")"
    )


def find_rapidcsv_include(requested: Path | None) -> Path:
    candidates = []
    if requested:
        candidates.append(requested)
    candidates.append(
        ROOT / "build" / "highs-1151" / "_deps" / "rapidcsv-src" / "src"
    )
    candidates.extend(
        sorted(
            (ROOT / "build").glob(
                "cpm-cache/rapidcsv/*/src"
            )
        )
    )
    candidates.extend(
        sorted(
            (ROOT / "build").glob(
                "*/_deps/rapidcsv-src/src"
            )
        )
    )
    for candidate in candidates:
        resolved = candidate.resolve()
        if (resolved / "rapidcsv.h").is_file():
            return resolved
    raise GateFailure(
        "rapidcsv.h was not found under build/; configure a normal DTWC "
        "build first or pass --rapidcsv-include"
    )


def compile_command(
    compiler: str,
    rapidcsv_include: Path,
    source: str,
    *,
    defines: tuple[str, ...] = (),
) -> list[str]:
    compiler_name = Path(compiler).name.lower()
    if compiler_name in {"cl", "cl.exe"}:
        raise GateFailure(
            "MSVC cl.exe is not supported by this syntax-only runner; "
            "use clang++ or g++"
        )
    return [
        compiler,
        "-std=c++20",
        "-fsyntax-only",
        "-I",
        str(ROOT),
        "-I",
        str(rapidcsv_include),
        *(f"-D{define}" for define in defines),
        source,
    ]


def run_compiler(
    compiler: str,
    rapidcsv_include: Path,
    source: str,
    *,
    stdin: str | None = None,
    defines: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    command = compile_command(
        compiler,
        rapidcsv_include,
        source,
        defines=defines,
    )
    if stdin is not None:
        command[1:1] = ["-x", "c++"]
    return subprocess.run(
        command,
        cwd=ROOT,
        input=stdin,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def compiler_failure_text(result: subprocess.CompletedProcess[str]) -> str:
    parts = []
    if result.stdout:
        parts.append(result.stdout.rstrip())
    if result.stderr:
        parts.append(result.stderr.rstrip())
    return "\n".join(parts)


def require_compile_state(
    *,
    probe: str,
    result: subprocess.CompletedProcess[str],
    expected_success: bool,
) -> str:
    success = result.returncode == 0
    if success != expected_success:
        observed = "passed" if success else "failed"
        expected = "pass" if expected_success else "fail"
        detail = compiler_failure_text(result)
        suffix = f"\n{detail}" if detail else ""
        raise GateFailure(
            f"{probe} compile {observed}, expected it to {expected}{suffix}"
        )
    return compiler_failure_text(result)


def require_diagnostic_set(
    *,
    probe: str,
    diagnostics: str,
    prefix: str,
    expected: tuple[str, ...],
) -> int:
    observed = set(re.findall(rf"\b{prefix}[A-Za-z_]+\b", diagnostics))
    expected_set = set(expected)
    if observed != expected_set:
        raise GateFailure(
            f"{probe} diagnostics do not identify the registered failures: "
            f"missing={sorted(expected_set - observed)} "
            f"unexpected={sorted(observed - expected_set)}"
        )
    return len(observed)


def require_error_count(
    *,
    probe: str,
    diagnostics: str,
    expected: int,
) -> None:
    error_lines = re.findall(r"(?m)^.*\berror:.*$", diagnostics)
    if len(error_lines) != expected:
        raise GateFailure(
            f"{probe} emitted {len(error_lines)} compiler error lines, "
            f"expected exactly {expected}; errors={error_lines}"
        )


def run_attribution_probes(
    *,
    profile: str,
    compiler: str,
    rapidcsv_include: Path,
) -> tuple[int, int, int]:
    fixture_source = str(FIXTURE)
    compatibility = run_compiler(
        compiler,
        rapidcsv_include,
        fixture_source,
        defines=(
            "DTWC_F19_SKIP_PRIVATE_ASSERTS",
            "DTWC_F19_SKIP_GETTER_ASSERTS",
            "DTWC_F19_SKIP_SETTER_EXERCISE",
        ),
    )
    require_compile_state(
        probe="retained compatibility",
        result=compatibility,
        expected_success=True,
    )

    private_probe = run_compiler(
        compiler,
        rapidcsv_include,
        fixture_source,
        defines=(
            "DTWC_F19_SKIP_GETTER_ASSERTS",
            "DTWC_F19_SKIP_SETTER_EXERCISE",
        ),
    )
    getter_probe = run_compiler(
        compiler,
        rapidcsv_include,
        fixture_source,
        defines=(
            "DTWC_F19_SKIP_PRIVATE_ASSERTS",
            "DTWC_F19_SKIP_SETTER_EXERCISE",
        ),
    )
    setter_probe = run_compiler(
        compiler,
        rapidcsv_include,
        fixture_source,
        defines=(
            "DTWC_F19_SKIP_PRIVATE_ASSERTS",
            "DTWC_F19_SKIP_GETTER_ASSERTS",
        ),
    )

    final_profile = profile == "final"
    private_diagnostics = require_compile_state(
        probe="private-field",
        result=private_probe,
        expected_success=final_profile,
    )
    getter_diagnostics = require_compile_state(
        probe="read-only-getter",
        result=getter_probe,
        expected_success=final_profile,
    )
    setter_diagnostics = require_compile_state(
        probe="setter",
        result=setter_probe,
        expected_success=final_profile,
    )

    if final_profile:
        return (0, 0, 0)

    private_count = require_diagnostic_set(
        probe="private-field",
        diagnostics=private_diagnostics,
        prefix="F19_PRIVATE_",
        expected=PRIVATE_DIAGNOSTICS,
    )
    require_error_count(
        probe="private-field",
        diagnostics=private_diagnostics,
        expected=len(PRIVATE_DIAGNOSTICS),
    )
    getter_count = require_diagnostic_set(
        probe="read-only-getter",
        diagnostics=getter_diagnostics,
        prefix="F19_GETTER_",
        expected=GETTER_DIAGNOSTICS,
    )
    require_error_count(
        probe="read-only-getter",
        diagnostics=getter_diagnostics,
        expected=len(GETTER_DIAGNOSTICS),
    )
    missing_setters = tuple(
        setter
        for setter in INHERITED_MISSING_SETTERS
        if re.search(rf"\b{setter}\b", setter_diagnostics)
    )
    if missing_setters != INHERITED_MISSING_SETTERS:
        raise GateFailure(
            "setter diagnostics do not identify all four inherited accessor "
            f"gaps: observed={missing_setters} "
            f"expected={INHERITED_MISSING_SETTERS}"
        )
    require_error_count(
        probe="setter",
        diagnostics=setter_diagnostics,
        expected=len(INHERITED_MISSING_SETTERS),
    )
    return (private_count, getter_count, len(missing_setters))


def audit_fixture(fixture_text: str) -> None:
    assertion_partitions = {
        "private": len(
            re.findall(
                r"static_assert\s*\(\s*!raw_private_[a-z_]+"
                r"<dtwc::Problem>",
                fixture_text,
            )
        ),
        "getters": len(
            re.findall(
                r"static_assert\s*\(\s*has_const_[a-z_]+"
                r"<dtwc::Problem>",
                fixture_text,
            )
        ),
        "retained": len(
            re.findall(
                r"static_assert\s*\(\s*raw_retained_[a-z_]+"
                r"<dtwc::Problem>",
                fixture_text,
            )
        ),
    }
    expected_partitions = {"private": 10, "getters": 10, "retained": 11}
    if assertion_partitions != expected_partitions:
        raise GateFailure(
            "static-assert inventory differs from the registered 10/10/11 "
            f"partition: {assertion_partitions}"
        )

    total_assertions = len(
        re.findall(r"\bstatic_assert\s*\(", fixture_text)
    )
    if total_assertions != 31:
        raise GateFailure(
            f"static-assert inventory is {total_assertions}, expected 31"
        )

    expected_diagnostics = {
        *PRIVATE_DIAGNOSTICS,
        *GETTER_DIAGNOSTICS,
        "F19_RETAINED_max_iter",
        "F19_RETAINED_n_repetition",
        "F19_RETAINED_band",
        "F19_RETAINED_variant_params",
        "F19_RETAINED_missing_strategy",
        "F19_RETAINED_distance_strategy",
        "F19_RETAINED_cuda_settings",
        "F19_RETAINED_mip_settings",
        "F19_RETAINED_init_fun",
        "F19_RETAINED_clusters",
        "F19_RETAINED_centroids",
    }
    actual_diagnostics = set(
        re.findall(r'"(F19_(?:PRIVATE|GETTER|RETAINED)_[a-z_]+)"',
                   fixture_text)
    )
    if actual_diagnostics != expected_diagnostics:
        raise GateFailure(
            "diagnostic marker inventory differs from the 31 registered "
            f"assertions: missing={sorted(expected_diagnostics - actual_diagnostics)} "
            f"unexpected={sorted(actual_diagnostics - expected_diagnostics)}"
        )

    getter_accesses = {
        field: (
            len(re.findall(
                rf"\{{\s*problem\.{field}\s*\(\s*\)\s*\}}\s*->",
                fixture_text,
            )),
            len(re.findall(
                rf"\{{\s*const_problem\.{field}\s*\(\s*\)\s*\}}\s*->",
                fixture_text,
            )),
        )
        for field in PRIVATE_FIELDS
    }
    wrong_getter_accesses = {
        field: counts
        for field, counts in getter_accesses.items()
        if counts != (1, 1)
    }
    if wrong_getter_accesses:
        raise GateFailure(
            "each getter concept must constrain both mutable and const Problem "
            f"reads exactly once: {wrong_getter_accesses}"
        )

    resize_checks = len(re.findall(
        r"!public_resize_callable<dtwc::Problem>",
        fixture_text,
    ))
    if resize_checks != 1:
        raise GateFailure(
            "the 31-assertion fixture must include exactly one public-resize "
            f"rejection, found {resize_checks}"
        )

    setter_counts = {
        setter: len(re.findall(rf"\bproblem\.{setter}\s*\(", fixture_text))
        for setter in SETTER_EXERCISE
    }
    wrong_setters = {
        setter: count for setter, count in setter_counts.items() if count != 1
    }
    if wrong_setters:
        raise GateFailure(
            "setter exercise must contain each registered setter exactly once: "
            f"{wrong_setters}"
        )


def require_private_member(
    problem_text: str,
    *,
    member: str,
    declaration_pattern: str,
) -> None:
    declarations = list(re.finditer(declaration_pattern, problem_text))
    if len(declarations) != 1:
        raise GateFailure(
            f"{member} declaration inventory is {len(declarations)}, expected 1"
        )
    access_labels = [
        match
        for match in re.finditer(
            r"(?m)^\s*(public|private|protected):\s*$",
            problem_text,
        )
        if match.start() < declarations[0].start()
    ]
    if not access_labels or access_labels[-1].group(1) != "private":
        observed = access_labels[-1].group(1) if access_labels else "none"
        raise GateFailure(
            f"{member} must remain private; preceding access label={observed}"
        )


def problem_aliases(source_text: str) -> set[str]:
    patterns = (
        r"(?:const\s+)?dtwc::Problem\s*[*&]\s*([A-Za-z_]\w*)",
        (
            r"auto\s*&\s*([A-Za-z_]\w*)\s*=\s*\*"
            r"HandleManager<dtwc::Problem>::"
        ),
        (
            r"auto\s+([A-Za-z_]\w*)\s*=\s*"
            r"std::make_shared<dtwc::Problem>"
        ),
        r"dtwc::Problem\s+([A-Za-z_]\w*)\s*(?:\(|\{)",
    )
    return {
        alias
        for pattern in patterns
        for alias in re.findall(pattern, source_text)
    }


def direct_problem_field_accesses(
    source_text: str,
    *,
    include_nanobind_field_pointers: bool,
) -> dict[str, int]:
    aliases = problem_aliases(source_text)
    counts = {}
    for field in PRIVATE_FIELDS:
        count = sum(
            len(re.findall(
                rf"\b{re.escape(alias)}\s*(?:\.|->)\s*"
                rf"{field}\b(?!\s*\()",
                source_text,
            ))
            for alias in aliases
        )
        if include_nanobind_field_pointers:
            count += len(re.findall(
                r"\.def_rw\s*\(\s*\"[^\"]+\"\s*,\s*&\s*"
                rf"dtwc::Problem::{field}\b",
                source_text,
            ))
        counts[field] = count
    return counts


def require_expected_direct_accesses(
    *,
    profile: str,
    surface: str,
    actual: dict[str, int],
    expected: dict[str, int],
) -> int:
    if set(actual) != set(PRIVATE_FIELDS):
        raise GateFailure(
            f"{surface} direct-access audit did not inspect all ten private "
            f"targets: {sorted(actual)}"
        )
    if actual != expected:
        mismatches = {
            field: {"actual": actual[field], "expected": expected[field]}
            for field in PRIVATE_FIELDS
            if actual[field] != expected[field]
        }
        raise GateFailure(
            f"{profile} {surface} direct private-field access mismatch: "
            f"{mismatches}; full inventory={actual}"
        )
    return sum(actual.values())


def audit_sources(
    profile: str,
    problem_text: str,
    python_text: str,
    matlab_text: str,
    authoritative_texts: dict[str, str],
) -> tuple[int, int, int, int, int, int, int, int]:
    expectation = EXPECTED[profile]
    require_private_member(
        problem_text,
        member="Problem::resize()",
        declaration_pattern=r"(?m)^\s*void\s+resize\s*\(\s*\)\s*;",
    )
    last_iteration_setters = len(re.findall(
        r"\bset_last_iterations\s*\(",
        problem_text,
    ))
    if last_iteration_setters != 0:
        raise GateFailure(
            "last_iterations is registered read-only, but found "
            f"{last_iteration_setters} set_last_iterations declarations"
        )

    declaration_counts = {
        field: len(re.findall(pattern, problem_text))
        for field, pattern in RAW_PRIVATE_DECLARATIONS.items()
    }
    expected_declaration_count = expectation["declaration_count"]
    declaration_mismatches = {
        field: count
        for field, count in declaration_counts.items()
        if count != expected_declaration_count
    }
    if declaration_mismatches:
        raise GateFailure(
            f"{profile} raw-field inventory mismatch: "
            f"{declaration_mismatches}; full inventory={declaration_counts}"
        )
    violations = sum(declaration_counts.values())

    backing_matches = {
        field: list(re.finditer(pattern, problem_text))
        for field, pattern in PRIVATE_BACKING_DECLARATIONS.items()
    }
    backing_counts = {
        field: len(matches) for field, matches in backing_matches.items()
    }
    expected_backing_count = expectation["backing_declaration_count"]
    backing_mismatches = {
        field: count
        for field, count in backing_counts.items()
        if count != expected_backing_count
    }
    if backing_mismatches:
        raise GateFailure(
            f"{profile} private-backing inventory mismatch: "
            f"{backing_mismatches}; full inventory={backing_counts}"
        )
    backings = sum(backing_counts.values())

    if expected_backing_count:
        retained_max_iter = re.search(
            RAW_PRIVATE_DECLARATIONS["method"].replace(
                r"Method\s+method\s*\{", r"int\s+maxIter\s*\{"
            ),
            problem_text,
        )
        if retained_max_iter is None:
            raise GateFailure(
                "cannot locate retained public maxIter declaration to delimit "
                "the final private backing block"
            )
        public_blocks = [
            match
            for match in re.finditer(r"(?m)^\s*public:\s*$", problem_text)
            if match.start() < retained_max_iter.start()
        ]
        if not public_blocks:
            raise GateFailure(
                "cannot locate the public block containing retained fields"
            )
        relevant_public = public_blocks[-1]
        private_blocks = [
            match
            for match in re.finditer(r"(?m)^\s*private:\s*$", problem_text)
            if match.start() < relevant_public.start()
        ]
        if not private_blocks:
            raise GateFailure(
                "cannot locate a private block before the retained public block"
            )
        relevant_private = private_blocks[-1]
        misplaced_backings = [
            field
            for field, matches in backing_matches.items()
            if not (
                relevant_private.end()
                < matches[0].start()
                < relevant_public.start()
            )
        ]
        if misplaced_backings:
            raise GateFailure(
                "final backing declarations are not inside the private region "
                f"before the retained public block: {misplaced_backings}"
            )

    helper_pattern = (
        r"(?m)^\s*static\s+void\s+store_result_in_problem\s*\("
    )
    invocation_pattern = r"\bstore_result_in_problem\s*\("
    helpers = len(re.findall(helper_pattern, matlab_text))
    invocations = len(re.findall(invocation_pattern, matlab_text))
    calls = invocations - helpers
    if calls < 0:
        raise GateFailure(
            "MATLAB helper inventory is internally inconsistent: "
            f"helpers={helpers} invocations={invocations}"
        )

    actual = (violations, backings, helpers, calls)
    expected = (
        expectation["violations"],
        expectation["backings"],
        expectation["helpers"],
        expectation["calls"],
    )
    if actual != expected:
        raise GateFailure(
            f"{profile} source inventory is "
            f"violations/backings/helpers/calls={actual}, "
            f"expected {expected}"
        )

    mex_write_counts = {
        name: len(re.findall(pattern, matlab_text))
        for name, pattern in MEX_REDUNDANT_WRITEBACKS.items()
    }
    expected_mex_write_count = expectation["mex_write_count"]
    wrong_mex_writes = {
        name: count
        for name, count in mex_write_counts.items()
        if count != expected_mex_write_count
    }
    if wrong_mex_writes:
        raise GateFailure(
            f"{profile} redundant MEX writeback inventory mismatch: "
            f"{wrong_mex_writes}; full inventory={mex_write_counts}"
        )
    mex_writes = sum(mex_write_counts.values())
    if mex_writes != expectation["mex_writes"]:
        raise GateFailure(
            f"{profile} redundant MEX writeback total is {mex_writes}, "
            f"expected {expectation['mex_writes']}"
        )

    core_write_counts = {
        f"{implementation}.{token}": len(
            re.findall(pattern, authoritative_texts[implementation])
        )
        for implementation, patterns in AUTHORITATIVE_WRITEBACKS.items()
        for token, pattern in patterns.items()
    }
    expected_core_write_count = expectation["core_write_count"]
    wrong_core_writes = {
        token: count
        for token, count in core_write_counts.items()
        if count != expected_core_write_count
    }
    if wrong_core_writes:
        raise GateFailure(
            f"{profile} authoritative core writeback inventory mismatch: "
            f"{wrong_core_writes}; full inventory={core_write_counts}"
        )
    core_writes = sum(core_write_counts.values())
    if core_writes != expectation["core_writes"]:
        raise GateFailure(
            f"{profile} authoritative core writeback total is {core_writes}, "
            f"expected {expectation['core_writes']}"
        )

    python_access_inventory = direct_problem_field_accesses(
        python_text,
        include_nanobind_field_pointers=True,
    )
    python_directs = require_expected_direct_accesses(
        profile=profile,
        surface="Python binding",
        actual=python_access_inventory,
        expected=expectation["python_direct_accesses"],
    )

    matlab_access_inventory = direct_problem_field_accesses(
        matlab_text,
        include_nanobind_field_pointers=False,
    )
    matlab_directs = require_expected_direct_accesses(
        profile=profile,
        surface="MATLAB MEX",
        actual=matlab_access_inventory,
        expected=expectation["matlab_direct_accesses"],
    )

    return (
        violations,
        backings,
        helpers,
        calls,
        mex_writes,
        core_writes,
        python_directs,
        matlab_directs,
    )


def main() -> int:
    args = parse_args()
    try:
        fixture_text = read_text(FIXTURE)
        problem_text = read_text(PROBLEM_HEADER)
        python_text = read_text(PYTHON_BINDING)
        matlab_text = read_text(MATLAB_MEX)
        authoritative_texts = {
            name: read_text(path)
            for name, path in AUTHORITATIVE_WRITEBACK_FILES.items()
        }
        audit_fixture(fixture_text)
        (
            violations,
            backings,
            helpers,
            calls,
            mex_writes,
            core_writes,
            python_directs,
            matlab_directs,
        ) = audit_sources(
            args.expect,
            problem_text,
            python_text,
            matlab_text,
            authoritative_texts,
        )

        compiler = find_compiler(args.compiler)
        rapidcsv_include = find_rapidcsv_include(args.rapidcsv_include)
        smoke = run_compiler(
            compiler,
            rapidcsv_include,
            "-",
            stdin='#include "dtwc/Problem.hpp"\n',
        )
        if smoke.returncode != 0:
            raise GateFailure(
                "public-header smoke compile failed independently of the "
                "contract fixture:\n" + compiler_failure_text(smoke)
            )

        (
            private_diagnostics,
            getter_diagnostics,
            setter_diagnostics,
        ) = run_attribution_probes(
            profile=args.expect,
            compiler=compiler,
            rapidcsv_include=rapidcsv_include,
        )

        contract = run_compiler(
            compiler, rapidcsv_include, str(FIXTURE)
        )
        compile_success = contract.returncode == 0
        expected_compile_success = EXPECTED[args.expect]["compile_success"]
        if compile_success != expected_compile_success:
            state = "passed" if compile_success else "failed"
            expected_state = (
                "pass" if expected_compile_success else "fail"
            )
            detail = compiler_failure_text(contract)
            suffix = f"\n{detail}" if detail else ""
            raise GateFailure(
                f"contract compile {state}, expected it to {expected_state}"
                f"{suffix}"
            )

        compile_marker = "passed" if compile_success else "failed"
        print(
            "F19_PROBLEM_ENCAPSULATION "
            f"profile={args.expect} compile={compile_marker} "
            f"violations={violations} backings={backings} "
            f"helpers={helpers} calls={calls} "
            f"mex_writes={mex_writes} core_writes={core_writes} "
            f"python_directs={python_directs} "
            f"matlab_directs={matlab_directs} "
            f"privacy_diagnostics={private_diagnostics} "
            f"getter_diagnostics={getter_diagnostics} "
            f"setter_diagnostics={setter_diagnostics} "
            "compatibility_compile=passed "
            "assertions=31 verdict=PASS"
        )
        return 0
    except GateFailure as error:
        print(f"F19_PROBLEM_ENCAPSULATION_ERROR {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
