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
        "--self-test",
        action="store_true",
        help="Run non-mutating adversarial probes against the gate itself.",
    )
    return parser.parse_args()


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        raise GateFailure(f"cannot read {path}: {error}") from error


def scrub_cpp_lexically(text: str, *, strip_literals: bool) -> str:
    """Replace comments and optionally literals with spaces, preserving lines."""
    chars = list(text)
    size = len(chars)

    def blank(start: int, end: int) -> None:
        for index in range(start, min(end, size)):
            if chars[index] not in "\r\n":
                chars[index] = " "

    index = 0
    while index < size:
        if text.startswith("//", index):
            end = text.find("\n", index + 2)
            end = size if end < 0 else end
            blank(index, end)
            index = end
            continue
        if text.startswith("/*", index):
            close = text.find("*/", index + 2)
            end = size if close < 0 else close + 2
            blank(index, end)
            index = end
            continue
        if text.startswith('R"', index):
            delimiter_end = text.find("(", index + 2)
            if delimiter_end >= 0:
                delimiter = text[index + 2:delimiter_end]
                terminator = ")" + delimiter + '"'
                close = text.find(terminator, delimiter_end + 1)
                if close >= 0:
                    end = close + len(terminator)
                    if strip_literals:
                        blank(index, end)
                    index = end
                    continue
        if chars[index] in {'"', "'"}:
            quote = chars[index]
            end = index + 1
            while end < size:
                if chars[end] == "\\":
                    end += 2
                    continue
                end += 1
                if text[end - 1] == quote:
                    break
            if strip_literals:
                blank(index, end)
            index = end
            continue
        index += 1
    return "".join(chars)


def remove_literal_if_zero_blocks(text: str) -> str:
    """Remove literal #if 0 branches while retaining active #else branches."""
    output = []
    stack: list[tuple[bool, bool]] = []
    active = True
    for line in text.splitlines(keepends=True):
        directive = re.match(r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b(.*)",
                             line)
        if directive:
            keyword = directive.group(1)
            argument = directive.group(2).strip()
            if keyword in {"if", "ifdef", "ifndef"}:
                literal_zero = keyword == "if" and argument == "0"
                stack.append((literal_zero, active))
                if literal_zero:
                    active = False
                    output.append("\n" if line.endswith("\n") else "")
                    continue
            elif keyword in {"else", "elif"} and stack:
                literal_zero, parent_active = stack[-1]
                if literal_zero:
                    active = parent_active
                    output.append("\n" if line.endswith("\n") else "")
                    continue
            elif keyword == "endif" and stack:
                literal_zero, parent_active = stack.pop()
                if literal_zero:
                    active = parent_active
                    output.append("\n" if line.endswith("\n") else "")
                    continue
            if active:
                output.append(line)
            else:
                output.append("\n" if line.endswith("\n") else "")
            continue
        if active:
            output.append(line)
        else:
            output.append("\n" if line.endswith("\n") else "")
    return "".join(output)


def active_cpp_source(text: str) -> str:
    commentless = scrub_cpp_lexically(text, strip_literals=False)
    return remove_literal_if_zero_blocks(commentless)


def preprocessor_depth_zero_source(text: str) -> str:
    """Keep only code outside every conditional-preprocessor region."""
    commentless = scrub_cpp_lexically(text, strip_literals=False)
    output = []
    depth = 0
    for line in commentless.splitlines(keepends=True):
        directive = re.match(
            r"^\s*#\s*(if|ifdef|ifndef|elif|else|endif)\b",
            line,
        )
        if directive:
            keyword = directive.group(1)
            if keyword in {"if", "ifdef", "ifndef"}:
                depth += 1
            elif keyword == "endif":
                if depth == 0:
                    raise GateFailure("unmatched #endif in C++ source audit")
                depth -= 1
            output.append("\n" if line.endswith("\n") else "")
            continue
        if depth == 0:
            output.append(line)
        else:
            output.append("\n" if line.endswith("\n") else "")
    if depth != 0:
        raise GateFailure(
            f"unterminated conditional-preprocessor depth in source audit: {depth}"
        )
    return "".join(output)


def unconditional_cpp_code(text: str) -> str:
    return scrub_cpp_lexically(
        preprocessor_depth_zero_source(text),
        strip_literals=True,
    )


def cpp_code_only(text: str) -> str:
    return scrub_cpp_lexically(active_cpp_source(text), strip_literals=True)


def cpp_function_span(code: str, name: str) -> tuple[int, int]:
    signature = re.search(
        rf"(?m)^\s*static\s+void\s+{re.escape(name)}\s*\(",
        code,
    )
    if signature is None:
        raise GateFailure(f"cannot locate C++ function {name}")
    opening = code.find("{", signature.end())
    if opening < 0:
        raise GateFailure(f"cannot locate opening brace for {name}")
    depth = 0
    for index in range(opening, len(code)):
        if code[index] == "{":
            depth += 1
        elif code[index] == "}":
            depth -= 1
            if depth == 0:
                return (signature.start(), index + 1)
    raise GateFailure(f"cannot locate closing brace for {name}")


def cpp_function_body(active_source: str, code: str, name: str) -> str:
    start, end = cpp_function_span(code, name)
    return active_source[start:end]


def blank_cpp_functions(
    active_source: str,
    code: str,
    names: tuple[str, ...],
) -> str:
    chars = list(active_source)
    for name in names:
        start, end = cpp_function_span(code, name)
        for index in range(start, end):
            if chars[index] not in "\r\n":
                chars[index] = " "
    return "".join(chars)


def cpp_statement(active_source: str, code: str, start_token: str) -> str:
    start = code.find(start_token)
    if start < 0:
        raise GateFailure(f"cannot locate C++ statement starting {start_token}")
    parens = braces = brackets = 0
    for index in range(start, len(code)):
        char = code[index]
        if char == "(":
            parens += 1
        elif char == ")":
            parens -= 1
        elif char == "{":
            braces += 1
        elif char == "}":
            braces -= 1
        elif char == "[":
            brackets += 1
        elif char == "]":
            brackets -= 1
        elif char == ";" and parens == braces == brackets == 0:
            return active_source[start:index + 1]
    raise GateFailure(f"cannot locate statement terminator for {start_token}")


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


def compile_command(
    compiler: str,
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
        *(f"-D{define}" for define in defines),
        source,
    ]


def run_compiler(
    compiler: str,
    source: str,
    *,
    stdin: str | None = None,
    defines: tuple[str, ...] = (),
) -> subprocess.CompletedProcess[str]:
    command = compile_command(
        compiler,
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
) -> tuple[int, int, int]:
    fixture_source = str(FIXTURE)
    compatibility = run_compiler(
        compiler,
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
        fixture_source,
        defines=(
            "DTWC_F19_SKIP_GETTER_ASSERTS",
            "DTWC_F19_SKIP_SETTER_EXERCISE",
        ),
    )
    getter_probe = run_compiler(
        compiler,
        fixture_source,
        defines=(
            "DTWC_F19_SKIP_PRIVATE_ASSERTS",
            "DTWC_F19_SKIP_SETTER_EXERCISE",
        ),
    )
    setter_probe = run_compiler(
        compiler,
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
    fixture_code = cpp_code_only(fixture_text)
    assertion_partitions = {
        "private": len(
            re.findall(
                r"static_assert\s*\(\s*!raw_private_[a-z_]+"
                r"<dtwc::Problem>",
                fixture_code,
            )
        ),
        "getters": len(
            re.findall(
                r"static_assert\s*\(\s*has_const_[a-z_]+"
                r"<dtwc::Problem>",
                fixture_code,
            )
        ),
        "retained": len(
            re.findall(
                r"static_assert\s*\(\s*raw_retained_[a-z_]+"
                r"<dtwc::Problem>",
                fixture_code,
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
        re.findall(r"\bstatic_assert\s*\(", fixture_code)
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
                fixture_code,
            )),
            len(re.findall(
                rf"\{{\s*const_problem\.{field}\s*\(\s*\)\s*\}}\s*->",
                fixture_code,
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

    member_pointer_checks = {
        field: len(re.findall(
            rf"\{{\s*&T::{field}\s*\}}\s*->\s*std::same_as<"
            rf"{field}_getter_pointer<T>>",
            fixture_code,
        ))
        for field in PRIVATE_FIELDS
    }
    wrong_member_pointer_checks = {
        field: count
        for field, count in member_pointer_checks.items()
        if count != 1
    }
    if wrong_member_pointer_checks:
        raise GateFailure(
            "each getter must prove one exact unambiguous const member pointer: "
            f"{wrong_member_pointer_checks}"
        )

    readonly_last_iteration_checks = {
        "setter": len(re.findall(
            r"problem\.set_last_iterations\s*\(\s*7\s*\)",
            fixture_code,
        )),
        "overload": len(re.findall(
            r"problem\.last_iterations\s*\(\s*7\s*\)",
            fixture_code,
        )),
    }
    if readonly_last_iteration_checks != {"setter": 1, "overload": 1}:
        raise GateFailure(
            "last_iterations getter concept must reject both setter and "
            f"one-argument overload: {readonly_last_iteration_checks}"
        )

    retained_accesses = {
        field: (
            len(re.findall(rf"\{{\s*problem\.{field}\s*\}}\s*->",
                           fixture_code)),
            len(re.findall(rf"\{{\s*const_problem\.{field}\s*\}}\s*->",
                           fixture_code)),
        )
        for field in (
            "maxIter",
            "N_repetition",
            "band",
            "variant_params",
            "missing_strategy",
            "distance_strategy",
            "cuda_settings",
            "mip_settings",
            "init_fun",
            "clusters_ind",
            "centroids_ind",
        )
    }
    wrong_retained_accesses = {
        field: counts
        for field, counts in retained_accesses.items()
        if counts != (1, 1)
    }
    if wrong_retained_accesses:
        raise GateFailure(
            "each retained concept must constrain exact mutable and const "
            f"reads: {wrong_retained_accesses}"
        )

    resize_checks = len(re.findall(
        r"!public_resize_callable<dtwc::Problem>",
        fixture_code,
    ))
    if resize_checks != 1:
        raise GateFailure(
            "the 31-assertion fixture must include exactly one public-resize "
            f"rejection, found {resize_checks}"
        )

    setter_counts = {
        setter: len(re.findall(rf"\bproblem\.{setter}\s*\(", fixture_code))
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


PYTHON_PRIVATE_PROPERTIES = {
    "method": "set_method",
    "random_seed": "set_random_seed",
    "lb_strategy": "set_lb_strategy",
    "storage_policy": "set_storage_policy",
    "verbose": "set_verbose",
    "output_folder": "set_output_folder",
    "name": "set_name",
}


def erase_regex_matches(text: str, pattern: str, *, expected: int) -> str:
    matches = list(re.finditer(pattern, text))
    if len(matches) != expected:
        raise GateFailure(
            f"exact API spelling count is {len(matches)}, expected {expected}: "
            f"{pattern}"
        )
    chars = list(text)
    for match in matches:
        for index in range(match.start(), match.end()):
            if chars[index] not in "\r\n":
                chars[index] = " "
    return "".join(chars)


def audit_final_python_api(python_active: str, python_code: str) -> None:
    chain_count = len(re.findall(
        r"\bnb::class_<\s*dtwc::Problem\s*>",
        python_code,
    ))
    if chain_count != 1:
        raise GateFailure(
            "final Python binding must contain exactly one canonical Problem "
            f"class chain, found {chain_count}"
        )
    chain = cpp_statement(
        python_active,
        python_code,
        "nb::class_<dtwc::Problem>",
    )
    chain_residual = chain
    global_residual = python_active
    for field, setter in PYTHON_PRIVATE_PROPERTIES.items():
        pattern = (
            rf"\.def_prop_rw\s*\(\s*\"{field}\"\s*,\s*"
            rf"&\s*dtwc::Problem::{field}\s*,\s*"
            rf"&\s*dtwc::Problem::{setter}\b"
        )
        chain_residual = erase_regex_matches(
            chain_residual,
            pattern,
            expected=1,
        )
        global_residual = erase_regex_matches(
            global_residual,
            pattern,
            expected=1,
        )

    for name, member in (
        ("set_method", "set_method"),
        ("set_random_seed", "set_random_seed"),
    ):
        pattern = (
            rf"\.def\s*\(\s*\"{name}\"\s*,\s*"
            rf"&\s*dtwc::Problem::{member}\b"
        )
        chain_residual = erase_regex_matches(
            chain_residual,
            pattern,
            expected=1,
        )
        global_residual = erase_regex_matches(
            global_residual,
            pattern,
            expected=1,
        )

    chain_residual = cpp_code_only(chain_residual)
    global_residual = cpp_code_only(global_residual)
    forbidden_tokens = [
        token
        for field in PRIVATE_FIELDS
        for token in (field + "_",)
        if re.search(rf"\b{re.escape(token)}\b", global_residual)
    ]
    if forbidden_tokens:
        raise GateFailure(
            "final Python Problem binding mentions private backing tokens: "
            f"{sorted(set(forbidden_tokens))}"
        )

    generic_spellings = re.findall(
        r"\b(?:const_cast|reinterpret_cast|decltype|std::invoke|std::mem_fn)\b",
        global_residual,
    )
    if generic_spellings:
        raise GateFailure(
            "final Python Problem binding uses forbidden generic state access: "
            f"{generic_spellings}"
        )

    private_member_spellings = re.findall(
        r"(?:\.|->|::)\s*("
        + "|".join(re.escape(field) for field in PRIVATE_FIELDS)
        + r")\b(?!\s*\()",
        chain_residual,
    )
    if private_member_spellings:
        raise GateFailure(
            "final Python Problem binding contains non-API private state "
            f"spellings: {private_member_spellings}"
        )

    private_member_pointers = re.findall(
        r"&\s*dtwc::Problem::([A-Za-z_]\w*)",
        global_residual,
    )
    forbidden_pointers = [
        token
        for token in private_member_pointers
        if token in PRIVATE_FIELDS
        or token.rstrip("_") in PRIVATE_FIELDS
        or token in SETTER_EXERCISE
    ]
    if forbidden_pointers:
        raise GateFailure(
            "final Python Problem binding contains unpinned private member "
            f"pointers: {forbidden_pointers}"
        )


MATLAB_FINAL_COMMAND_APIS = {
    "cmd_Problem_new": (
        r"\bprob\s*->\s*set_verbose\s*\(\s*false\s*\)\s*;",
    ),
    "cmd_Problem_get_info": (
        r"\bprob\.name\s*\(\s*\)",
        r"\bprob\.verbose\s*\(\s*\)",
    ),
    "cmd_Problem_set_data": (
        r"\bprob\.set_data\s*\(",
    ),
    "cmd_Problem_set_verbose": (
        r"\bprob\.set_verbose\s*\(\s*mxIsLogicalScalarTrue\s*\(",
    ),
    "cmd_Problem_get_name": (
        r"\bprob\.name\s*\(\s*\)\s*\.c_str\s*\(\s*\)",
    ),
    "cmd_Problem_set_method": (
        r"\bprob\.set_method\s*\(\s*parse_method\s*\(",
    ),
    "cmd_Problem_set_lb_strategy": (
        r"\bprob\.set_lb_strategy\s*\(\s*candidate\s*\)",
    ),
    "cmd_Problem_set_storage_policy": (
        r"\bprob\.set_storage_policy\s*\(\s*candidate\s*\)",
    ),
    "cmd_Problem_set_output_folder": (
        r"\bprob\.set_output_folder\s*\(\s*"
        r"std::filesystem::path\s*\(\s*get_string\s*\(",
    ),
    "cmd_compute_distance_matrix": (
        r"\bprob\.set_verbose\s*\(\s*false\s*\)\s*;",
        r"\bprob\.set_data\s*\(",
    ),
    "cmd_cluster_legacy": (
        r"\bprob\.set_verbose\s*\(\s*false\s*\)\s*;",
        r"\bprob\.set_data\s*\(",
    ),
}


def audit_final_matlab_api(matlab_active: str, matlab_code: str) -> None:
    for command, patterns in MATLAB_FINAL_COMMAND_APIS.items():
        body = cpp_function_body(matlab_active, matlab_code, command)
        for pattern in patterns:
            count = len(re.findall(pattern, body))
            if count != 1:
                raise GateFailure(
                    f"final MATLAB {command} exact API pattern count is "
                    f"{count}, expected 1: {pattern}"
                )

    code = matlab_code
    backing_tokens = [
        field + "_"
        for field in PRIVATE_FIELDS
        if re.search(rf"\b{re.escape(field)}_\b", code)
    ]
    if backing_tokens:
        raise GateFailure(
            "final MATLAB MEX mentions private backing tokens: "
            f"{backing_tokens}"
        )

    generic_spellings = re.findall(
        r"\b(?:const_cast|reinterpret_cast|decltype|std::invoke|std::mem_fn)\b",
        code,
    )
    if generic_spellings:
        raise GateFailure(
            "final MATLAB MEX uses forbidden generic state access: "
            f"{generic_spellings}"
        )

    code = erase_regex_matches(
        code,
        r"\bopts\.random_seed\b",
        expected=2,
    )
    private_member_spellings = re.findall(
        r"(?:\.|->|::)\s*("
        + "|".join(re.escape(field) for field in PRIVATE_FIELDS)
        + r")\b(?!\s*\()",
        code,
    )
    if private_member_spellings:
        raise GateFailure(
            "final MATLAB MEX contains non-API private state spellings: "
            f"{private_member_spellings}"
        )


def semantic_mex_write_counts(
    matlab_active: str,
    matlab_code: str,
) -> dict[str, int]:
    without_legitimate_commands = blank_cpp_functions(
        matlab_active,
        matlab_code,
        (
            "cmd_Problem_set_n_clusters",
            "cmd_Problem_get_centroids",
            "cmd_Problem_get_clusters",
        ),
    )
    code = cpp_code_only(without_legitimate_commands)
    return {
        "cluster_count": len(re.findall(
            r"(?:\.|->|::)\s*(?:set_n_clusters|set_numberOfClusters)"
            r"\s*\(",
            code,
        )),
        "medoids": len(re.findall(
            r"(?:\.|->|::)\s*centroids_ind\b",
            code,
        )),
        "labels": len(re.findall(
            r"(?:\.|->|::)\s*clusters_ind\b",
            code,
        )),
    }


def expect_gate_rejection(label: str, action) -> int:
    try:
        action()
    except GateFailure:
        return 1
    raise GateFailure(f"adversarial self-probe was not rejected: {label}")


def run_fixture_concept_self_probe(compiler: str) -> int:
    source = r'''
#define DTWC_F19_SKIP_PRIVATE_ASSERTS
#define DTWC_F19_SKIP_GETTER_ASSERTS
#define DTWC_F19_SKIP_RETAINED_ASSERTS
#define DTWC_F19_SKIP_SETTER_EXERCISE
#include "scripts/fixtures/f19_problem_encapsulation.cpp"

struct StaticLast {
  static int last_iterations();
};
static_assert(!has_const_last_iterations_getter<StaticLast>);

struct OverloadedLast {
  int last_iterations() const;
  void last_iterations(int);
};
static_assert(!has_const_last_iterations_getter<OverloadedLast>);

struct SetterLast {
  int last_iterations() const;
  void set_last_iterations(int);
};
static_assert(!has_const_last_iterations_getter<SetterLast>);

struct CallableMethod {
  struct Callable {
    dtwc::Method operator()();
    dtwc::Method operator()() const;
  };
  Callable method;
};
static_assert(!raw_private_method_assignable<CallableMethod>);
static_assert(!has_const_method_getter<CallableMethod>);

struct WriteOnlyInt {
  void operator=(int);
};
struct RetainedProxy {
  WriteOnlyInt maxIter;
};
static_assert(!raw_retained_max_iter_assignable<RetainedProxy>);

struct RetainedExact {
  int maxIter;
};
static_assert(raw_retained_max_iter_assignable<RetainedExact>);
'''
    result = run_compiler(
        compiler,
        "-",
        stdin=source,
    )
    require_compile_state(
        probe="adversarial fixture concepts",
        result=result,
        expected_success=True,
    )
    return 6


def synthetic_final_python_binding() -> str:
    properties = "\n".join(
        f'.def_prop_rw("{field}", &dtwc::Problem::{field}, '
        f'&dtwc::Problem::{setter})'
        for field, setter in PYTHON_PRIVATE_PROPERTIES.items()
    )
    return (
        'nb::class_<dtwc::Problem>(m, "Problem")\n'
        + properties
        + '\n.def("set_method", &dtwc::Problem::set_method)\n'
        + '.def("set_random_seed", &dtwc::Problem::set_random_seed);\n'
    )


def synthetic_final_matlab_binding() -> str:
    return r'''
static void cmd_Problem_new() { prob->set_verbose(false); }
static void cmd_Problem_get_info() { use(prob.name()); use(prob.verbose()); }
static void cmd_Problem_set_data() { prob.set_data(data); }
static void cmd_Problem_set_verbose() {
  prob.set_verbose(mxIsLogicalScalarTrue(prhs[2]));
}
static void cmd_Problem_get_name() { use(prob.name().c_str()); }
static void cmd_Problem_set_method() {
  prob.set_method(parse_method(get_string(prhs[2])));
}
static void cmd_Problem_set_lb_strategy() {
  prob.set_lb_strategy(candidate);
}
static void cmd_Problem_set_storage_policy() {
  prob.set_storage_policy(candidate);
}
static void cmd_Problem_set_output_folder() {
  prob.set_output_folder(std::filesystem::path(get_string(prhs[2])));
}
static void cmd_compute_distance_matrix() {
  prob.set_verbose(false);
  prob.set_data(data);
}
static void cmd_cluster_legacy() {
  prob.set_verbose(false);
  prob.set_data(data);
}
static void cmd_Problem_set_n_clusters() {
  prob.set_n_clusters(k);
}
static void cmd_Problem_get_centroids() { use(prob.centroids_ind); }
static void cmd_Problem_get_clusters() { use(prob.clusters_ind); }
static void option_writes() {
  opts.random_seed = 1;
  opts.random_seed = 2;
}
'''


def run_source_adversarial_self_probes(
    authoritative_texts: dict[str, str],
) -> int:
    probes = 0

    commented_setter = cpp_code_only(
        "void set_last_iterations /* comment */ (int);\n"
    )
    if len(re.findall(r"\bset_last_iterations\s*\(", commented_setter)) != 1:
        raise GateFailure(
            "comment stripping did not expose set_last_iterations mutator"
        )
    probes += 1

    backing_declarations = "\n".join((
        "Method method_{};",
        "std::uint64_t random_seed_{};",
        "int last_iterations_{};",
        "double tadpole_dc_{};",
        "LowerBoundStrategy lb_strategy_{};",
        "core::StoragePolicy storage_policy_{};",
        "bool verbose_{};",
        "path_t output_folder_{};",
        "std::string name_{};",
        "Data data_;",
    ))
    for condition in ("0", "(0)", "0u", "false"):
        inactive_backings = (
            f"#if {condition}\n{backing_declarations}\n#endif\n"
        )
        inactive_code = unconditional_cpp_code(inactive_backings)
        counts = {
            field: len(re.findall(pattern, inactive_code))
            for field, pattern in PRIVATE_BACKING_DECLARATIONS.items()
        }
        if any(count != 0 for count in counts.values()):
            raise GateFailure(
                f"conditional backing evidence counted for #if {condition}: "
                f"{counts}"
            )
        probes += 1

    for condition in ("0", "(0)", "0u", "false"):
        counts = {
            f"{implementation}.{token}": len(re.findall(
                pattern,
                unconditional_cpp_code(
                    f"#if {condition}\n"
                    + authoritative_texts[implementation]
                    + "\n#endif\n"
                ),
            ))
            for implementation, patterns in AUTHORITATIVE_WRITEBACKS.items()
            for token, pattern in patterns.items()
        }
        if any(count != 0 for count in counts.values()):
            raise GateFailure(
                f"conditional core evidence counted for #if {condition}: "
                f"{counts}"
            )
        probes += 1

    python_valid = synthetic_final_python_binding()
    audit_final_python_api(
        active_cpp_source(python_valid),
        cpp_code_only(python_valid),
    )
    python_mutants = {
        "backing": ".def(\"hack\", [](auto &problem) { problem.method_ = x; })",
        "generic": (
            ".def(\"hack\", [](auto &problem) { "
            "auto member = &dtwc::Problem::method; use(member); })"
        ),
        "const_cast": (
            ".def(\"hack\", [](auto &problem) { "
            "const_cast<std::string &>(problem.name()) = x; })"
        ),
        "direct": (
            ".def(\"hack\", [](auto &problem) { "
            "problem.output_folder = x; })"
        ),
    }
    for label, mutation in python_mutants.items():
        mutant = python_valid.replace(
            ".def(\"set_random_seed\", &dtwc::Problem::set_random_seed);",
            ".def(\"set_random_seed\", &dtwc::Problem::set_random_seed)\n"
            + mutation
            + ";",
        )
        probes += expect_gate_rejection(
            f"Python {label}",
            lambda mutant=mutant: audit_final_python_api(
                active_cpp_source(mutant),
                cpp_code_only(mutant),
            ),
        )

    second_chain = (
        python_valid
        + '\nnb::class_<dtwc::Problem>(m, "ProblemShadow")'
        + '.def("hack", [](auto &problem) { problem.method_ = x; });\n'
    )
    probes += expect_gate_rejection(
        "Python second Problem chain",
        lambda: audit_final_python_api(
            active_cpp_source(second_chain),
            cpp_code_only(second_chain),
        ),
    )

    matlab_valid = synthetic_final_matlab_binding()
    matlab_active = active_cpp_source(matlab_valid)
    matlab_code = cpp_code_only(matlab_valid)
    audit_final_matlab_api(matlab_active, matlab_code)
    matlab_mutants = {
        "backing": "static void hack() { prob.verbose_ = false; }\n",
        "generic": (
            "static void hack() { decltype(auto) alias = prob; "
            "alias.verbose_ = false; }\n"
        ),
        "const_cast": (
            "static void hack() { "
            "const_cast<bool &>(prob.verbose()) = false; }\n"
        ),
        "member_pointer": (
            "static void hack() { auto member = &dtwc::Problem::verbose; }\n"
        ),
        "direct": "static void hack() { prob.verbose = false; }\n",
    }
    for label, mutation in matlab_mutants.items():
        mutant = matlab_valid + mutation
        probes += expect_gate_rejection(
            f"MATLAB {label}",
            lambda mutant=mutant: audit_final_matlab_api(
                active_cpp_source(mutant),
                cpp_code_only(mutant),
            ),
        )

    semantic_mutant = matlab_valid + r'''
static void renamed_writeback() {
  prob.set_n_clusters(k);
  prob.centroids_ind.assign(result.medoid_indices.begin(),
                            result.medoid_indices.end());
  prob.clusters_ind.swap(result.labels);
}
'''
    semantic_counts = semantic_mex_write_counts(
        active_cpp_source(semantic_mutant),
        cpp_code_only(semantic_mutant),
    )
    if semantic_counts != {
        "cluster_count": 1,
        "medoids": 1,
        "labels": 1,
    }:
        raise GateFailure(
            "semantic MATLAB writeback mutant was not detected: "
            f"{semantic_counts}"
        )
    probes += 1

    inactive_semantic = matlab_valid + "#if 0\n" + semantic_mutant + "\n#endif\n"
    inactive_counts = semantic_mex_write_counts(
        active_cpp_source(inactive_semantic),
        cpp_code_only(inactive_semantic),
    )
    if inactive_counts != {
        "cluster_count": 0,
        "medoids": 0,
        "labels": 0,
    }:
        raise GateFailure(
            "inactive semantic MATLAB writeback was counted: "
            f"{inactive_counts}"
        )
    probes += 1
    return probes


def audit_sources(
    profile: str,
    problem_text: str,
    python_text: str,
    matlab_text: str,
    authoritative_texts: dict[str, str],
) -> tuple[int, int, int, int, int, int, int, int]:
    expectation = EXPECTED[profile]
    problem_code = cpp_code_only(problem_text)
    problem_evidence_code = unconditional_cpp_code(problem_text)
    python_active = active_cpp_source(python_text)
    python_code = cpp_code_only(python_text)
    matlab_active = active_cpp_source(matlab_text)
    matlab_code = cpp_code_only(matlab_text)
    authoritative_code = {
        name: unconditional_cpp_code(text)
        for name, text in authoritative_texts.items()
    }

    require_private_member(
        problem_code,
        member="Problem::resize()",
        declaration_pattern=r"(?m)^\s*void\s+resize\s*\(\s*\)\s*;",
    )
    last_iteration_setters = len(re.findall(
        r"\bset_last_iterations\s*\(",
        problem_code,
    ))
    if last_iteration_setters != 0:
        raise GateFailure(
            "last_iterations is registered read-only, but found "
            f"{last_iteration_setters} set_last_iterations declarations"
        )

    declaration_counts = {
        field: len(re.findall(pattern, problem_evidence_code))
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
        field: list(re.finditer(pattern, problem_evidence_code))
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
            problem_evidence_code,
        )
        if retained_max_iter is None:
            raise GateFailure(
                "cannot locate retained public maxIter declaration to delimit "
                "the final private backing block"
            )
        public_blocks = [
            match
            for match in re.finditer(
                r"(?m)^\s*public:\s*$",
                problem_evidence_code,
            )
            if match.start() < retained_max_iter.start()
        ]
        if not public_blocks:
            raise GateFailure(
                "cannot locate the public block containing retained fields"
            )
        relevant_public = public_blocks[-1]
        private_blocks = [
            match
            for match in re.finditer(
                r"(?m)^\s*private:\s*$",
                problem_evidence_code,
            )
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
    helpers = len(re.findall(helper_pattern, matlab_code))
    invocations = len(re.findall(invocation_pattern, matlab_code))
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

    mex_write_counts = semantic_mex_write_counts(matlab_active, matlab_code)
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
            re.findall(pattern, authoritative_code[implementation])
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
        python_active,
        include_nanobind_field_pointers=True,
    )
    python_directs = require_expected_direct_accesses(
        profile=profile,
        surface="Python binding",
        actual=python_access_inventory,
        expected=expectation["python_direct_accesses"],
    )

    matlab_access_inventory = direct_problem_field_accesses(
        matlab_active,
        include_nanobind_field_pointers=False,
    )
    matlab_directs = require_expected_direct_accesses(
        profile=profile,
        surface="MATLAB MEX",
        actual=matlab_access_inventory,
        expected=expectation["matlab_direct_accesses"],
    )

    if profile == "final":
        audit_final_python_api(python_active, python_code)
        audit_final_matlab_api(matlab_active, matlab_code)

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
        self_probes = 0
        if args.self_test:
            self_probes += run_fixture_concept_self_probe(compiler)
            self_probes += run_source_adversarial_self_probes(
                authoritative_texts,
            )
            print(
                "F19_PROBLEM_ENCAPSULATION_SELF_TEST "
                f"probes={self_probes} verdict=PASS"
            )
        smoke = run_compiler(
            compiler,
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
        )

        contract = run_compiler(compiler, str(FIXTURE))
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
            f"self_probes={self_probes} "
            "assertions=31 verdict=PASS"
        )
        return 0
    except GateFailure as error:
        print(f"F19_PROBLEM_ENCAPSULATION_ERROR {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
