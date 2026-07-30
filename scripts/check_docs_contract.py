#!/usr/bin/env python3
"""Adversarial documentation drift checks against live code and CLI output."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def cpp_string_literals(function_name: str, source: str) -> list[str]:
    match = re.search(
        rf"std::string {re.escape(function_name)}\([^)]*\)\s*\{{(.*?)^\}}",
        source,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"cannot locate {function_name} in env.cpp")
    tokens = re.findall(r'"(?:\\.|[^"\\])*"', match.group(1))
    return [json.loads(token) for token in tokens]


def assert_env_messages() -> None:
    source = (ROOT / "dtwc/env.cpp").read_text(encoding="utf-8")
    docs = (ROOT / "docs/content/guides/devices.md").read_text(encoding="utf-8")

    no_env = "".join(cpp_string_literals("msg_no_env_file", source))
    missing_parts = cpp_string_literals("msg_missing_key", source)
    missing = missing_parts[0] + "<key>" + "".join(missing_parts[1:])
    auth_parts = cpp_string_literals("msg_auth_failure", source)
    auth = (auth_parts[0] + "<host>" + auth_parts[1] + "<user>"
            + auth_parts[2] + auth_parts[3] + "<user>" + auth_parts[4]
            + "<host>" + "".join(auth_parts[5:]))

    for label, message in (("missing file", no_env), ("missing key", missing),
                           ("bad host", auth)):
        if message not in docs:
            raise AssertionError(f"{label} HPC error in devices.md is not verbatim")


def compact(text: str) -> str:
    return " ".join(text.split())


def braced_body(source: str, signature: str, label: str) -> str:
    """Return one source body so implementation guards cannot match comments."""
    start = source.find(signature)
    if start < 0:
        raise AssertionError(f"cannot locate {label} signature")
    opening = source.find("{", start + len(signature))
    if opening < 0:
        raise AssertionError(f"cannot locate {label} opening brace")

    depth = 0
    for pos in range(opening, len(source)):
        char = source[pos]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return source[opening + 1:pos]
    raise AssertionError(f"cannot locate {label} closing brace")


def assert_ordered_markers(
    source: str, label: str, markers: tuple[str, ...]
) -> None:
    cursor = 0
    for marker in markers:
        position = source.find(marker, cursor)
        if position < 0:
            raise AssertionError(
                f"{label} omits or reorders marker after offset {cursor}: {marker}"
            )
        cursor = position + len(marker)


def assert_freeze_governance() -> None:
    status = (ROOT / "docs/api-contract-2.0.md").read_text(
        encoding="utf-8"
    ).splitlines()[0]
    if ("STATUS: FROZEN" not in status
            or "changes require a PLAN.md decision entry" not in status):
        raise AssertionError(
            "frozen API contract status must require a PLAN.md decision entry"
        )


def assert_contract_audit_state() -> None:
    contract = (ROOT / "docs/api-contract-2.0.md").read_text(encoding="utf-8")
    stale = (
        "[new bind]",
        "unbound today",
        "before FROZEN",
        "On adversarial sign-off",
        "to be backed by `Env`",
        "**Reserved:** `Method::LRCore`",
        "today it does not",
        "directory checkpoint = `distances.csv` + `metadata.txt`",
        "CLI `--resume` reads but currently discards",
        "current CLI only reads and reports its binary result checkpoint",
        "encapsulation/accessor cleanup is incomplete",
        "MATLAB retains redundant",
        "three missing C++ accessors remain",
        "[gap F19",
        "F19 owns removal",
        "raw configuration/result fields remain public (F19)",
        "store_result_in_problem",
        "`tadpole_dc` prop",
        "[gap F20",
        "advisory stored",
        "advisory only",
        "F21 covers four missing canonical C++ names",
        "remain unimplemented frozen promises (F21)",
        "spellings that are still absent",
        "aliases currently do not emit the frozen runtime warnings",
        "F22 owns their missing deprecation diagnostics",
        "F22 records that retained aliases do not all emit",
        "Rows 35–38 are implemented",
        "F22 records incomplete diagnostics",
        "get_/set_distance_matrix",
        "do not yet emit the required diagnostics (F22)",
        "F22 owns missing diagnostics",
        "do not currently emit runtime warnings (F22)",
        "absent `[gap F23]`",
        "Python binary bindings are missing (F23)",
        "lacks the two direct binary bindings",
    )
    present = [marker for marker in stale if marker in contract]
    if present:
        raise AssertionError(
            f"frozen contract retains pre-implementation markers: {present}"
        )

    required = (
        "[introduced-2.0]",
        "## 10. Adjudicated reviewer decisions",
        "CURRENT",
        "generations/<id>",
        "passive configuration carrier",
        "independent copy",
        "Python `Problem.set_view_data` currently constructs owning",
        "completed-result replay",
        "additional continuation budget",
        "same `<output>/<name>`",
        "Ten C++ `Problem` fields are private",
        "`last_iterations()` is intentionally read-only",
        "`data()` returns `const Data&`",
        "neither binding\nrepeats the assignment",
        "private C++ state; CLI exposes `--dc`",
        "`start_column(int)` and `start_row(int)` own the loader mutations",
        "four configuration properties warn on assignment",
        "deprecated actual `int` fields",
        "C++ retained 1.x alias",
        "caller-attributed `DeprecationWarning`",
        "dedicated `+dtwc/*.m` compatibility wrappers",
        "Every retained\ncallable alias in this table emits",
        "`set_distance_matrix` is canonical and warning-silent",
        "33 C++ diagnostic entities, 13 Python alias operations, and 15 MATLAB",
        "`ClusterResult` (`python/dtwcpp/__init__.py:306-326`)",
        "`Result.medoid_indices` (`python/dtwcpp/_api.py:102-107`)",
        "`dtwc_mex.cpp:224-230`",
        "`__init__.py:213-238`",
        "`_api.py:281-301`",
        "`dtwc_cl.cpp:1343-1419`",
        "`DataLoader.hpp:291-297`",
        "`DataLoader.hpp:299-306`",
        "`distance.hpp:35-42`",
        "scores.hpp:38-39",
        "settings.hpp:88-94",
        "dtwc_cl.cpp:717-725",
        "`_hpc.py:527-601`",
    )
    missing = [marker for marker in required if marker not in contract]
    if missing:
        raise AssertionError(
            f"frozen contract omits audited current-state markers: {missing}"
        )

    if contract.count("**Resolved:**") != 8:
        raise AssertionError(
            "frozen contract must record exactly eight reviewer resolutions"
        )
    missing_findings = [
        f"F{number}" for number in (18, *range(24, 27))
        if f"F{number}" not in contract
    ]
    if missing_findings:
        raise AssertionError(
            f"frozen contract hides 2.0 implementation gaps: {missing_findings}"
        )


def assert_python_binary_checkpoint_contract() -> None:
    paths = {
        "contract": ROOT / "docs/api-contract-2.0.md",
        "checkpointing": ROOT / "docs/content/getting-started/checkpointing.md",
        "python_site": ROOT / "docs/content/getting-started/python.md",
    }
    text = {
        name: compact(path.read_text(encoding="utf-8"))
        for name, path in paths.items()
    }
    signatures = (
        "`save_binary_checkpoint(result, path) -> None`",
        "`load_binary_checkpoint(path) -> ClusteringResult`",
    )
    required = {
        "contract": (
            *signatures,
            "`str | os.PathLike[str]`",
            "valid-Unicode",
            "releases the GIL",
            "Native write failures raise `dtwcpp.IOError`",
            "`dtwcpp.IOError` subclasses both `DtwcError` and `OSError`",
            "F56",
        ),
        "checkpointing": (
            *signatures,
            "`dtwcpp.ClusteringResult`",
            "valid-Unicode",
            "assert isinstance(replayed, dtwcpp.ClusteringResult)",
            "except dtwcpp.IOError as error:",
            "also a `DtwcError` and `OSError`",
            "does not establish that labels, medoids, N, k, or the producing "
            "configuration are semantically compatible",
        ),
        "python_site": (
            *signatures,
            "`dtwcpp.ClusteringResult`",
            "valid-Unicode",
            "assert isinstance(restored, dtwcpp.ClusteringResult)",
            "except dtwcpp.IOError as error:",
            "both `DtwcError` and `OSError`",
            "validates the binary wire structure, not dataset/configuration "
            "provenance or N/k compatibility",
        ),
    }
    exact_read_error = (
        "load_binary_checkpoint: cannot read a valid binary result "
        "checkpoint from '<path>'."
    )
    drift = {
        name: [
            marker
            for marker in (*markers, exact_read_error)
            if marker not in text[name]
        ]
        for name, markers in required.items()
    }
    drift = {name: markers for name, markers in drift.items() if markers}
    if drift:
        raise AssertionError(
            f"Python binary-checkpoint documentation drift: {drift}"
        )

    binding = (ROOT / "python/src/_dtwcpp_core.cpp").read_text(encoding="utf-8")
    save_body = compact(braced_body(
        binding, 'm.def("save_binary_checkpoint"', "Python binary writer"
    ))
    load_body = compact(braced_body(
        binding, 'm.def("load_binary_checkpoint"', "Python binary reader"
    ))
    save_required = (
        "const dtwc::core::ClusteringResult snapshot = result;",
        "nb::gil_scoped_release release;",
        "dtwc::save_binary_checkpoint(snapshot, path);",
    )
    load_required = (
        "const std::string path_text = utf8_path_text(path);",
        "dtwc::core::ClusteringResult result;",
        "bool loaded = false;",
        "nb::gil_scoped_release release;",
        "loaded = dtwc::load_binary_checkpoint(result, path);",
        "} if (!loaded)",
        "throw dtwc::IOError(",
        '"load_binary_checkpoint: cannot read a valid binary result "',
        '"checkpoint from \'" + path_text + "\'."',
        "return result;",
    )
    assert_ordered_markers(
        save_body, "Python binary-checkpoint writer", save_required
    )
    assert_ordered_markers(
        load_body, "Python binary-checkpoint reader", load_required
    )

    package = (ROOT / "python/dtwcpp/__init__.py").read_text(encoding="utf-8")
    package_tree = ast.parse(package)
    native_imports = [
        alias.name
        for node in ast.walk(package_tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "dtwcpp._dtwcpp_core"
        for alias in node.names
    ]
    all_assignments = [
        node.value
        for node in package_tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    ]
    if len(all_assignments) != 1:
        raise AssertionError("cannot locate one literal dtwcpp.__all__ assignment")
    all_exports = ast.literal_eval(all_assignments[0])
    export_drift = {
        name: {
            "native_import": native_imports.count(name),
            "__all__": all_exports.count(name),
        }
        for name in ("save_binary_checkpoint", "load_binary_checkpoint")
        if native_imports.count(name) != 1 or all_exports.count(name) != 1
    }
    if export_drift:
        raise AssertionError(
            "Python binary-checkpoint exports must each appear once in the "
            f"native import and once in __all__: {export_drift}"
        )


def assert_migration_behaviors() -> None:
    migration = (ROOT / "docs/content/guides/migration.md").read_text(
        encoding="utf-8"
    )
    migration = compact(migration)
    required = (
        "`Result.distance_matrix` is `None` for matrix-free methods",
        "Explicit GPU requests no longer warn and run on CPU",
        "Requesting an unavailable MIP solver no longer prints and returns",
        "Ten C++ `Problem` fields are now private",
        "Every retained callable alias in this table emits its required C++ compile "
        "diagnostic or Python/MATLAB runtime warning",
        "`set_distance_matrix` is canonical and warning-silent",
        "33 C++ diagnostic entities, 13 Python alias operations, and 15 "
        "MATLAB alias operations",
    )
    missing = [item for item in required if item not in migration]
    if missing:
        raise AssertionError(f"migration guide omits behaviors: {missing}")


def assert_f22_changelog() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    unreleased_start = changelog.index("# Unreleased")
    next_release = changelog.index("# 2.0.0rc1", unreleased_start)
    unreleased = compact(changelog[unreleased_start:next_release])
    required = (
        "all 33 retained 1.x compatibility entities",
        "all 13 retained alias operations",
        "all 15 retained alias operations",
        "`ClusterResult` remains an uncached, identity-preserving alias",
        "canonical operations stay silent",
    )
    missing = [item for item in required if item not in unreleased]
    if missing:
        raise AssertionError(
            f"Unreleased changelog omits F22 behaviors: {missing}"
        )


def assert_rc1_changelog() -> None:
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    release_heading = "# 2.0.0rc1 - 2026-07-10"
    history_heading = "# Development history absorbed into 2.0.0rc1"
    if release_heading not in changelog or history_heading not in changelog:
        raise AssertionError("rc1 changelog release/history headings are missing")
    release_start = changelog.index(release_heading)
    history_start = changelog.index(history_heading)
    if history_start <= release_start:
        raise AssertionError("absorbed development history must follow the rc1 summary")
    summary = changelog[release_start:history_start]
    required = (
        "raise `DeviceError`",
        "raises\n  `SolverError`",
        "`dist_by_ind` rebind race",
        "Windows `0xc0000409`",
    )
    missing = [item for item in required if item not in summary]
    if missing:
        raise AssertionError(f"rc1 summary omits behaviors: {missing}")
    if "- API contract 2.0 frozen" in changelog:
        raise AssertionError("stray API-contract bullet remains in CHANGELOG")


def assert_tier1_signatures() -> None:
    header = compact((ROOT / "dtwc/api.hpp").read_text(encoding="utf-8"))
    cpp_api = compact((ROOT / "dtwc/api.cpp").read_text(encoding="utf-8"))
    python = compact((ROOT / "python/dtwcpp/_api.py").read_text(encoding="utf-8"))
    matlab = compact((ROOT / "bindings/matlab/+dtwc/cluster.m").read_text(encoding="utf-8"))
    contract = compact((ROOT / "docs/api-contract-2.0.md").read_text(encoding="utf-8"))

    cpp_needles = (
        'std::string device(std::string_view name);',
        'std::string device();',
        'Dataset load(const std::filesystem::path &source, int skip_cols = 0, char delimiter = 0, std::string_view name = "");',
        'Result cluster(const Dataset &data, int k, std::string_view method = "pam", int band = -1, std::string_view device = "", int max_iter = 100);',
        'double score(std::string_view name) const;',
        'void save(const std::filesystem::path &directory) const;',
    )
    for signature in cpp_needles:
        if signature not in header:
            raise AssertionError(f"C++ Tier-1 signature drift: {signature}")

    py_signature = 'def cluster(data, k, *, method="pam", band=-1, device=None, max_iter=100):'
    if py_signature not in python:
        raise AssertionError("Python Tier-1 cluster signature drift")
    if "function res = cluster(data, k, varargin)" not in matlab:
        raise AssertionError("MATLAB Tier-1 cluster signature drift")

    contract_needles = (
        'std::string dtwc::device(std::string_view name)',
        'std::string dtwc::device()',
        'dtwc::Result dtwc::cluster(const Dataset& data, int k, std::string_view method="pam", int band=-1, std::string_view device="", int max_iter=100)',
        'cluster(data, k, *, method="pam", band=-1, device=None, max_iter=100) -> Result',
        "res = dtwc.cluster(data, k, 'method','pam', 'band',-1, 'device','', 'max_iter',100)",
    )
    for signature in contract_needles:
        if signature not in contract:
            raise AssertionError(f"contract signature drift: {signature}")

    py_methods_match = re.search(r'_METHODS\s*=\s*\((.*?)\)', python, re.DOTALL)
    if py_methods_match is None:
        raise AssertionError("cannot locate Python _METHODS")
    py_methods = set(re.findall(r'"([a-z]+)"', py_methods_match.group(1)))
    required = {"auto", "pam", "onebatch", "clara", "kmedoids", "mip",
                "lrcore", "hierarchical", "tadpole"}
    if py_methods != required:
        raise AssertionError(f"Python method registry drift: {sorted(py_methods)}")
    for method in required:
        if f'"{method}"' not in cpp_api or f'"{method}"' not in contract:
            raise AssertionError(f"method {method!r} is not aligned across code/contract")


def assert_method_catalog() -> None:
    metrics = (ROOT / "docs/content/method/metrics.md").read_text(encoding="utf-8")
    variants = (ROOT / "docs/content/method/dtw-variants.md").read_text(
        encoding="utf-8"
    )
    algorithms = (ROOT / "docs/content/method/algorithms.md").read_text(
        encoding="utf-8"
    )

    if re.search(r"(?mi)^(?:#+\s+Huber\b|\|\s*Huber\s*\|)", metrics):
        raise AssertionError("metrics.md advertises unsupported Huber metric")
    if "significantly accelerating distance matrix construction" in metrics:
        raise AssertionError("metrics.md repeats the falsified exact-matrix speed claim")
    for name in ("L1", "L2", "Squared L2"):
        if name not in metrics:
            raise AssertionError(f"metrics.md omits live metric {name}")

    for name in ("DDTW", "WDTW", "ADTW", "Soft-DTW", "MSM", "TWE"):
        if re.search(rf"(?m)^## {re.escape(name)}\b", variants) is None:
            raise AssertionError(f"dtw-variants.md omits live variant {name}")

    method_headings = {
        "FastPAM": r"^## FastPAM\b",
        "OneBatchPAM": r"^## OneBatchPAM\b",
        "FastCLARA": r"^## FastCLARA\b",
        "Lloyd": r"^## Lloyd",
        "MIP": r"^## Mixed-Integer Programming",
        "LR-core": r"^## LR-core\b",
        "Hierarchical": r"^## Hierarchical\b",
        "TADPole": r"^## TADPole\b",
    }
    for name, pattern in method_headings.items():
        if re.search(pattern, algorithms, re.MULTILINE) is None:
            raise AssertionError(f"algorithms.md omits live method {name}")
    if "Information Systems" not in algorithms or "10.1016/j.is.2021.101804" not in algorithms:
        raise AssertionError("algorithms.md has stale FasterPAM provenance")


def assert_dtw_derivation_sync() -> None:
    paths = {
        "derivation": ROOT / "docs/derivations/01-dtw-recurrence-sakoe-chiba.md",
        "index": ROOT / "docs/derivations/README.md",
        "site": ROOT / "docs/content/method/dtw.md",
        "citations": ROOT / ".claude/CITATIONS.md",
        "plan": ROOT / "PLAN.md",
        "kernel": ROOT / "dtwc/core/dtw_kernel.hpp",
        "wrapper": ROOT / "dtwc/warping.hpp",
    }
    missing_paths = [
        str(path.relative_to(ROOT))
        for path in paths.values()
        if not path.is_file()
    ]
    if missing_paths:
        raise AssertionError(f"D1 derivation drift: missing files {missing_paths}")

    text = {
        name: path.read_text(encoding="utf-8")
        for name, path in paths.items()
    }
    required = {
        "derivation": (
            "`band >= |n-m|`",
            "`DTW_w(x,y) >= DTW_full(x,y)`",
            "`numeric_limits<T>::max()`",
            "No approximation is used",
            "not a metric",
            "## Code-conformance table",
            "**DISCREPANCY**",
        ),
        "index": (
            "01-dtw-recurrence-sakoe-chiba.md",
            "CPU **CONFIRMED**",
            "**DISCREPANCY** F12",
        ),
        "site": (
            "$$|i-j| \\le w$$",
            "w \\ge |n-m|",
            "`numeric_limits<T>::max()`",
            "not a metric",
            "non-increasing",
            "CPU routes",
            "CUDA's fixed-window geometry and public no-path sentinel are confirmed on the local RTX.",
            "Metal source implements the same fixed geometry and sentinel translation",
            "`[BLOCKED-ENV]` under F12",
        ),
        "citations": (
            "10.1109/TASSP.1978.1163055",
            "Sakoe-Chiba-DTW.pdf",
            "**[confirmed]**",
            "equations (6)--(8)",
        ),
        "plan": (
            "- [x] **D1. DTW recurrence + Sakoe–Chiba band.**",
            "**F12 — cross-backend fixed-band geometry and no-path sentinel diverge.**",
            "Metal's no-LB source",
            "**F33 — CPU banded DTW used an endpoint-scaled corridor",
        ),
        "kernel": (
            "The adjustment window is |row-column| <= band.",
            "if (n_long - n_short > band_width) return maxValue;",
        ),
        "wrapper": (
            "canonical fixed Sakoe-Chiba window `|i-j| <= band`",
            "if (max_sz - min_sz > band_width)",
        ),
    }
    drift = {
        name: [
            marker
            for marker in markers
            if marker not in compact(text[name])
        ]
        for name, markers in required.items()
    }
    drift = {name: markers for name, markers in drift.items() if markers}
    if drift:
        raise AssertionError(f"D1 derivation drift: missing markers {drift}")

    unsupported_math = [
        marker
        for marker in ("\\(", "\\)", "\\[", "\\]")
        if marker in text["derivation"]
    ]
    if unsupported_math:
        raise AssertionError(
            "D1 derivation drift: unsupported GitHub math delimiters "
            f"{unsupported_math}"
        )

    broken_table_math = [
        line
        for line in text["derivation"].splitlines()
        if line.lstrip().startswith("|") and ("$|" in line or "|$" in line)
    ]
    if broken_table_math:
        raise AssertionError(
            f"D1 derivation drift: raw table math pipes {broken_table_math}"
        )


def assert_lb_keogh_derivation_sync() -> None:
    paths = {
        "derivation": ROOT / "docs/derivations/02-envelopes-lb-keogh.md",
        "index": ROOT / "docs/derivations/README.md",
        "citations": ROOT / ".claude/CITATIONS.md",
        "lessons": ROOT / ".claude/LESSONS.md",
        "gpu_site": ROOT / "docs/content/method/gpu-backends.md",
        "metrics_site": ROOT / "docs/content/method/metrics.md",
        "multivariate_site": ROOT / "docs/content/method/multivariate.md",
        "algorithms_site": ROOT / "docs/content/method/algorithms.md",
        "dtw_site": ROOT / "docs/content/method/dtw.md",
        "python_site": ROOT / "docs/content/getting-started/python.md",
        "python_api": ROOT / "python/dtwcpp/__init__.py",
        "python_binding": ROOT / "python/src/_dtwcpp_core.cpp",
        "method_enum": ROOT / "dtwc/enums/Method.hpp",
        "cli_source": ROOT / "dtwc/dtwc_cl.cpp",
        "lower_bound_api": ROOT / "dtwc/core/lower_bounds.hpp",
        "lower_bound": ROOT / "dtwc/core/lower_bound_impl.hpp",
        "cuda_header": ROOT / "dtwc/cuda/cuda_dtw.cuh",
        "cuda_source": ROOT / "dtwc/cuda/cuda_dtw.cu",
        "metal_source": ROOT / "dtwc/metal/metal_dtw.mm",
        "public_distance": ROOT / "dtwc/core/public_distance.hpp",
        "pruned_header": ROOT / "dtwc/core/pruned_distance_matrix.hpp",
        "pruned_source": ROOT / "dtwc/core/pruned_distance_matrix.cpp",
        "tadpole": ROOT / "dtwc/algorithms/tadpole.cpp",
        "tadpole_header": ROOT / "dtwc/algorithms/tadpole.hpp",
        "changelog": ROOT / "CHANGELOG.md",
        "oracle": (
            ROOT / "tests/unit/adversarial/test_lb_keogh_derivation.cpp"
        ),
        "ctest": ROOT / "tests/CMakeLists.txt",
        "baseline": ROOT / ".claude/baselines/2026-07-30-d2-lb-keogh.md",
    }
    missing_paths = [
        str(path.relative_to(ROOT))
        for path in paths.values()
        if not path.is_file()
    ]
    if missing_paths:
        raise AssertionError(f"D2 derivation drift: missing files {missing_paths}")

    text = {
        name: path.read_text(encoding="utf-8")
        for name, path in paths.items()
    }
    required = {
        "derivation": (
            "# D2 — envelopes and LB_Keogh admissibility",
            "Proposition 1",
            "`r >= w`",
            "`w >= |n-m|`",
            "$U^2$",
            "Their sum is not generally admissible",
            "## Unequal-length prefix theorem",
            "No approximation is used",
            "## Full-DTW call sites",
            "## The negative-band discrepancy",
            "## Executable oracle",
            "## Code-conformance table",
            "direct $\\Theta(m\\min(m,2r+1))$ extrema scans",
            "CUDA `dtwc/cuda/cuda_dtw.cu:782-813`",
            "`dtwc/algorithms/tadpole.cpp:149-160,178-190,219-224`",
            "`dtwc/core/lower_bound_impl.hpp:419-527,593-611`",
            "D2_LB_KEOGH_GATE envelope_cases=2004 equal_cases=28602 unequal_cases=17712 call_sites=2/2 skips=0 verdict=PASS",
            "All tests passed (65 assertions in 1 test case)",
            "2,004",
            "28,602",
            "17,712",
            "`min(n,m)`",
            "channels share a commensurate unit $U$",
            "Raw heterogeneous physical units require an explicit scaling/weighting model",
            "Independent DTW allows each channel",
            "production multivariate bounds are `3 U` and `3 U^2`",
            "**DISCREPANCY** F46",
            "**DISCREPANCY** F47",
            "**DISCREPANCY** F30: requests can silently disable or fall back",
            "`dtwc/metal/metal_dtw.mm:1512-1515`",
            "TADPole's empty domain is **DISCREPANCY** F48",
            "direct-call band/cache provenance is **DISCREPANCY** F49",
            "**DISCREPANCY** F50",
            "D17",
        ),
        "index": (
            "02-envelopes-lb-keogh.md",
            "Scalar CPU L1/squared, feasible unequal prefix, and additive dependent/independent MV **CONFIRMED**",
            "API/metric/domain/provenance **DISCREPANCY** F46–F49",
            "GPU **DISCREPANCY/OPEN** F27–F30/F50",
        ),
        "citations": (
            "10.1007/s10115-004-0154-9",
            "KAIS_2004_warping.pdf",
            "Proposition 1",
            "sequences of the same length",
            "10.1109/ICDE.2001.914875",
            "closed/paywalled full text was not accessed",
            "**[inferred]** Formula-level compatibility",
            "10.1145/2339530.2339576",
            "SIGKDD_trillion.pdf",
            "normalized, equal-length subsequence search",
            "https://arxiv.org/abs/cs/0610046",
            "monotone-deque running extrema algorithm",
            "distinct two-pass LB_Improved result",
        ),
        "gpu_site": (
            "envelope coverage—not equality—is the admissibility condition",
            "`r >= w`",
            "summing only the first `min(n,m)` query rows is still admissible",
            "Full DTW requires the global envelope",
            "Only when the bound is admissible does `LB > threshold` certify",
            "`numeric_limits<double>::max()`",
            "not IEEE infinity",
            "exact-arithmetic-admissible Metal",
        ),
        "lessons": (
            "LB_Keogh admissibility is a domain contract",
            "centered envelope whose radius covers the actual fixed DTW window",
            "shared-path multivariate Euclidean, cosine, or Huber",
            "Admissibility does not imply tightness or a prune rate",
            "A bound-decision counter is not automatically an avoided-work counter",
            "Additive multivariate objectives need a unit model",
            "bit-level identity at a floating threshold remains D17",
        ),
        "metrics_site": (
            "unrooted squared-L2 bound squares each excess and has units `U^2`",
            "require finite input samples and ordered finite envelope bounds",
            "F47",
        ),
        "multivariate_site": (
            "coordinatewise box argument is a DTWC++ extension",
            "original Keogh proposition is scalar and same-length",
            "channels share a commensurate unit after scaling or nondimensionalization",
            "For independent DTW",
            "independent objectives `4/4`",
            "both multivariate proofs",
        ),
        "algorithms_site": (
            "For finite, nonempty, equal-length Standard-L1 pairs",
            "representable by the integer band API",
            "TADPole constructs a global-minimum/global-maximum envelope",
            "exactly representable regression confirms that regime",
            "threshold analysis remains D17",
            "F48",
        ),
        "dtw_site": (
            "CUDA's fixed-window geometry and public no-path sentinel are confirmed on the local RTX.",
            "Metal source implements the same fixed geometry and sentinel translation",
            "`[BLOCKED-ENV]` under F12",
        ),
        "python_site": (
            "legacy CPU LB-guided exact-matrix path",
            "`band=-1` disables LB_Keogh",
            "recomputed",
        ),
        "python_api": (
            "legacy CPU LB-guided exact-matrix path",
            "``band=-1``",
            "disables LB_Keogh",
            "recomputed",
        ),
        "python_binding": (
            "legacy LB-guided exact-matrix",
            "band=-1 disables LB_Keogh",
            "recomputed",
            "`band >= 0`",
            "finite double-max",
            "not IEEE infinity",
        ),
        "method_enum": (
            "conditionally admissible LB/UB pruning",
            "Exact-arithmetic identity covers finite, nonempty, equal-length Standard-L1",
            "floating thresholds remain D17",
            "empty series remain F48",
        ),
        "cli_source": (
            "TADPole density-peaks with conditionally admissible LB/UB DTW pruning",
        ),
        "lower_bound": (
            "A negative band is coerced to radius zero",
            "full-DTW envelope",
            "Envelope carries no source-length or radius provenance",
            "repository-derived extension",
            "separately minimized per-channel",
            "Channels must share a commensurate unit",
            "The current implementation is in L1 units",
        ),
        "cuda_header": (
            "finite public",
            "double-max no-result sentinel",
            "not IEEE infinity",
        ),
        "metal_source": (
            "finite FLT_MAX device sentinel",
            "normalized to public double-max on copy",
        ),
        "changelog": (
            "F48 records the empty-series TADPole inconsistency",
            "F49 records direct-call band/cache provenance",
            "floating threshold identity remains open under D17",
            "TADPole LB/UB counters describe density decisions",
            "CUDA's Python pruning documentation now states the required nonnegative band",
        ),
        "tadpole_header": (
            "finite, nonempty, equal-length Standard-L1",
            "The live TADPole route does not call",
            "Admissibility alone does not promise any prune rate",
            "Density-stage NOT-neighbour decisions by LB",
            "Exact arithmetic preserves",
            "Floating bit-level identity at a threshold remains D17",
            "Empty series are a known exception (F48)",
        ),
        "pruned_source": (
            "recomputed without a cutoff; this legacy route does not skip required work.",
        ),
        "baseline": (
            "## Remaining closure bands registered before execution",
            "D2_LB_KEOGH_GATE envelope_cases=2004 equal_cases=28602 unequal_cases=17712 call_sites=2/2 skips=0 verdict=PASS",
            "All tests passed (65 assertions in 1 test case)",
            "canonical `build/highs-1151`: 123/123",
            "llfio-OFF `build/nollfio`: 123/123",
            "Arrow-ON `build/arrow-pyarrow-23`: 125/125",
            "decisive target must now report exactly 65 assertions in one case",
        ),
    }
    drift = {
        name: [
            marker
            for marker in markers
            if marker not in compact(text[name])
        ]
        for name, markers in required.items()
    }
    drift = {name: markers for name, markers in drift.items() if markers}
    if drift:
        raise AssertionError(f"D2 derivation drift: missing markers {drift}")

    compact_required = {
        "lower_bound": (
            "const std::size_t w = static_cast<std::size_t>(std::max(band, 0));",
            "if (w >= n)",
            "const auto n = std::min(query.size(), env.upper.size());",
            "return std::max(lb1, lb2);",
            "sum += excess * excess;",
        ),
        "lower_bound_api": (
            "lb_kim_valid<SquaredL2Metric> = true;",
        ),
        "cuda_source": (
            "const bool do_lb_pruning = opts.use_lb_keogh && (opts.band >= 0);",
        ),
        "metal_source": (
            "const bool pipeline_uses_pair_indices = !use_banded_row && !use_regtile;",
            "bool lb_active = lb_requested && pipeline_uses_pair_indices && num_pairs > 0;",
            "if (!buf_upper || !buf_lower || !buf_lb || !buf_pair_indices || !buf_active_count)",
            "lb_active = false;",
        ),
        "pruned_source": (
            "const bool use_lb_keogh = use_lb_keogh_flag && (band >= 0);",
            "const bool use_lb_keogh = use_lb && (band >= 0);",
            "const bool equal_len = prob.series(i).size() == prob.series(j).size();",
            "dist = dtw_with_abandon(-1.0);",
        ),
        "tadpole": (
            "const int env_band = (band < 0) ? static_cast<int>(s.size()) : band;",
            "if (can_prune && si.size() == sj.size())",
            "if (lb >= dc)",
            "LB_Keogh + the diagonal L1",
            "permanent exactly representable regression",
            "Bit-level identity when a floating reduction",
        ),
        "oracle": (
            "REQUIRE(envelope_cases == 2004);",
            "REQUIRE(equal_cases == 28602);",
            "REQUIRE(unequal_cases == 17712);",
            "REQUIRE(discriminator.forward_l1 == 8.0);",
            "REQUIRE(discriminator.reverse_l1 == 2.0);",
            "REQUIRE(discriminator.forward_squared == 22.0);",
            "REQUIRE(discriminator.reverse_squared == 4.0);",
            "REQUIRE(singleton.symmetric_l1 == 0.5);",
            "REQUIRE(singleton.symmetric_squared == 0.25);",
            "REQUIRE(independent_l1 == 4.0);",
            "REQUIRE(independent_squared == 4.0);",
            "REQUIRE(mv_l1 == 3.0);",
            "REQUIRE(mv_squared == 3.0);",
            "REQUIRE(dependent_l1 == 8.0);",
            "REQUIRE(dependent_squared == 20.0);",
            "REQUIRE(unsafe_negative_bound == 2.0);",
            "REQUIRE(unsafe_negative_bound > dtwc::dtwFull_L<double>(x, y));",
            "REQUIRE(matrix_stats.pruned_by_lb_keogh == 0);",
            "REQUIRE(pruned_stats.pruned_by_lb == 0);",
            "REQUIRE(separated_pruned_stats.pruned_by_lb == 1);",
            "REQUIRE(call_sites == 2);",
        ),
    }
    compact_drift = {
        name: [
            marker
            for marker in markers
            if marker not in compact(text[name])
        ]
        for name, markers in compact_required.items()
    }
    compact_drift = {
        name: markers
        for name, markers in compact_drift.items()
        if markers
    }
    if compact_drift:
        raise AssertionError(
            f"D2 production/oracle drift: missing markers {compact_drift}"
        )

    implementation_bodies = {
        "public_float_normalization": braced_body(
            text["public_distance"],
            "inline constexpr double normalize_public_distance(float value) noexcept",
            "Float32 public-distance normalization",
        ),
        "cuda_compaction": braced_body(
            text["cuda_source"],
            "__global__ void compact_active_pairs_kernel(",
            "CUDA LB compaction kernel",
        ),
        "cuda_envelopes": braced_body(
            text["cuda_source"],
            "__global__ void compute_envelopes_kernel(",
            "CUDA envelope kernel",
        ),
        "cuda_copy": braced_body(
            text["cuda_source"],
            "std::vector<double> convert_result_matrix(",
            "CUDA result conversion",
        ),
        "metal_compaction": braced_body(
            text["metal_source"],
            "kernel void compact_active_pairs(",
            "Metal LB compaction kernel",
        ),
        "metal_envelopes": braced_body(
            text["metal_source"],
            "kernel void compute_envelopes(",
            "Metal envelope kernel",
        ),
        "metal_copy": text["metal_source"],
    }
    implementation_required = {
        "public_float_normalization": (
            "value == std::numeric_limits<float>::max()",
            "? std::numeric_limits<double>::max()",
            ": static_cast<double>(value)",
        ),
        "cuda_compaction": (
            "lb_values[pid] <= threshold",
            "static_cast<T>(3.402823466e+38f)",
            "static_cast<T>(1.7976931348623157e+308)",
            "result_matrix[si * N + sj] = INF;",
            "result_matrix[sj * N + si] = INF;",
        ),
        "cuda_envelopes": (
            "const int lo = (k >= w) ? k - w : 0;",
            "const int hi = (k + w + 1 < L) ? k + w + 1 : L;",
            "for (int j = lo + 1; j < hi; ++j)",
        ),
        "cuda_copy": (
            "dtwc::gpu::detail::normalize_public_distance(src[row_offset + j])",
        ),
        "metal_compaction": (
            "lb_values[pid] <= threshold",
            "const float INF = 3.402823466e+38f;",
            "result_matrix[si * N_series + sj] = INF;",
            "result_matrix[sj * N_series + si] = INF;",
        ),
        "metal_envelopes": (
            "const int lo = (k >= w) ? k - w : 0;",
            "const int hi = (k + w + 1 < L) ? k + w + 1 : L;",
            "for (int j = lo + 1; j < hi; ++j)",
        ),
        "metal_copy": (
            "result.matrix[i * N + j] = dtwc::gpu::detail::normalize_public_distance(out_ptr[i * N + j]);",
        ),
    }
    implementation_drift = {
        name: [
            marker
            for marker in markers
            if marker not in compact(implementation_bodies[name])
        ]
        for name, markers in implementation_required.items()
    }
    implementation_drift = {
        name: markers
        for name, markers in implementation_drift.items()
        if markers
    }
    if implementation_drift:
        raise AssertionError(
            "D2 GPU sentinel implementation drift: "
            f"missing code markers {implementation_drift}"
        )

    ctest_subjects = re.findall(
        r"if\(TARGET test_lb_keogh_derivation\)(.*?)endif\(\)",
        text["ctest"],
        flags=re.DOTALL,
    )
    if len(ctest_subjects) != 1:
        raise AssertionError(
            "D2 CTest drift: expected one test_lb_keogh_derivation policy block"
        )
    ctest_subject = compact(ctest_subjects[0])
    ctest_markers = (
        "PROPERTY SKIP_RETURN_CODE)",
        'ENVIRONMENT "OMP_NUM_THREADS=1"',
        'FAIL_REGULAR_EXPRESSION "[Ss][Kk][Ii][Pp]([Pp]|[ :])"',
        "PASS_REGULAR_EXPRESSION",
        "D2_LB_KEOGH_GATE envelope_cases=2004 equal_cases=28602 unequal_cases=17712 call_sites=2/2 skips=0 verdict=PASS",
        r"All tests passed \\((6[5-9]|[7-9][0-9]|[1-9][0-9][0-9]+) assertions in 1 test case\\)",
        "RUN_SERIAL TRUE",
    )
    missing_ctest = [
        marker for marker in ctest_markers if marker not in ctest_subject
    ]
    if missing_ctest:
        raise AssertionError(
            f"D2 CTest drift: missing non-skipping execution markers {missing_ctest}"
        )

    unsupported_math = [
        marker
        for marker in ("\\(", "\\)", "\\[", "\\]")
        if marker in text["derivation"]
    ]
    if unsupported_math:
        raise AssertionError(
            "D2 derivation drift: unsupported GitHub math delimiters "
            f"{unsupported_math}"
        )
    if text["derivation"].count("$$") % 2:
        raise AssertionError("D2 derivation drift: unbalanced display-math delimiters")
    broken_table_math = [
        line
        for line in text["derivation"].splitlines()
        if line.lstrip().startswith("|") and ("$|" in line or "|$" in line)
    ]
    if broken_table_math:
        raise AssertionError(
            f"D2 derivation drift: raw table math pipes {broken_table_math}"
        )

    stale = {
        "dtw_site": (
            "CUDA source currently uses an endpoint-scaled corridor",
            "Metal source uses fixed geometry, but its double-returning no-path route widens `FLT_MAX`",
        ),
        "gpu_site": (
            "without a validity proof (F29)",
            "supported only for equal-length L1 series with a matching admissible envelope",
            "pruned pairs remain +inf",
            "stamp +∞ into result",
        ),
        "metrics_site": (
            "valid L1, L2, and Squared L2 specializations",
            "valid for the same three pointwise metrics",
        ),
        "python_site": (
            "Enable LB_Keogh pruning (CPU only)",
        ),
        "python_api": (
            "Use LB_Keogh pruning (CPU only)",
        ),
        "python_binding": (
            "for faster computation (L1 metric only)",
            "pruned (+inf in result)",
            "result entry +inf",
        ),
        "method_enum": (
            "Result is provably identical to brute-force",
            "plain L1/SquaredL2 DTW",
        ),
        "cuda_header": (
            "get INF (no DTW)",
        ),
        "metal_source": (
            "pruned pairs have their +∞",
            "pruned (+∞ stamped",
            "Pruned pairs get +∞",
            "stamp +∞ for pruned",
        ),
        "pruned_header": (
            "saving ~30-60%",
            "early-abandon helped",
        ),
        "pruned_source": (
            "saving 30-60%",
        ),
        "algorithms_site": (
            "pins the result to the brute-force",
            "Exact relative to its brute-force",
        ),
        "lessons": (
            '"identical to brute force" guarantee',
        ),
        "tadpole": (
            "bit-for-bit the same either way",
            "LB_Keogh + the Euclidean upper bound",
        ),
        "tadpole_header": (
            "pairs are far (LB ≥ dc) and prune",
            "by LB ≥ dc (no DTW)",
            "by UB < dc (no DTW)",
            "plain L1/SquaredL2",
            "tighter cascade bounds (LB_Webb",
            "Labels are identical either way",
            "labels are identical to the brute-force",
        ),
    }
    present_stale = {
        name: [
            marker
            for marker in markers
            if marker in compact(text[name])
        ]
        for name, markers in stale.items()
    }
    present_stale = {
        name: markers
        for name, markers in present_stale.items()
        if markers
    }
    if present_stale:
        raise AssertionError(
            f"D2 documentation retains stale claims: {present_stale}"
        )


def assert_gpu_backend_page() -> None:
    page = (ROOT / "docs/content/method/gpu-backends.md").read_text(
        encoding="utf-8"
    )
    stale = (
        "MetalKernelOverride",
        "`max_length_hint > 0` skips the runtime length scan",
        "`Problem::lower_bound_strategy` (coming in a later commit)",
        "Selecting `DistanceMatrixStrategy::Auto` on a build with both backends enabled picks CUDA",
        "CPU with DistanceMatrixStrategy::Pruned",
        "always satisfies $$\\mathrm{LB}_{\\mathrm{Keogh}} \\le \\mathrm{DTW}$$",
        "The CUDA reference implementation in DTWC++ is a direct port",
        "**103×**",
        "**7.6×**",
        "`N < 20`, `L < 100`",
        "Unified memory removes H2D/D2H",
        "per-pair envelope + LB cost is O(L)",
    )
    present = [marker for marker in stale if marker in page]
    if present:
        raise AssertionError(f"GPU backend page retains stale claims: {present}")

    required = (
        "equal-length L1",
        "thresholded",
        "`numeric_limits<double>::max()`",
        "not IEEE infinity",
        "`dtwc::KernelOverride`",
        "`Problem::lb_strategy()` is CPU-only",
        "`Problem::set_lb_strategy(LowerBoundStrategy)`",
        "`DistanceMatrixStrategy::Auto` is CPU-only",
        "Θ(N·L·min(L, 2r+1))",
        "O(N²·L)",
        "benchmarks/results/mac_m2max/metal_vs_cpu.json",
        "historical, advisory",
        "inspired by cuDTW++",
        "F27",
        "F28",
        "F29",
        "F30",
        "F31",
    )
    missing = [marker for marker in required if marker not in page]
    if missing:
        raise AssertionError(f"GPU backend page omits current truth: {missing}")


def assert_remaining_docs_truth() -> None:
    paths = (
        "docs/content/getting-started/examples.md",
        "docs/content/getting-started/python.md",
        "docs/content/api/interface-parity.md",
        "docs/content/method/multivariate.md",
        "docs/content/method/scores.md",
        "dtwc/warping.hpp",
        "cmake/StandardProjectSettings.cmake",
        "dtwc/core/twe.hpp",
    )
    text = "\n".join(
        (ROOT / relative).read_text(encoding="utf-8") for relative in paths
    )
    stale = (
        "set_numberOfClusters",
        "set_number_of_clusters",
        "daviesBouldinIndex",
        "dunnIndex",
        "calinskiHarabaszIndex",
        "adjustedRandIndex",
        "normalizedMutualInformation",
        "davies_bouldin_index",
        "dunn_index",
        "calinski_harabasz_index",
        "adjusted_rand_index",
        "normalized_mutual_information",
        "cluster_labels",
        "All DTW variants have `_mv` counterparts",
        "Zero Overhead for Univariate",
        "without performance penalty",
        "loading added consistently later",
        "task R1",
        "zero overhead",
        "all other fast-math optimizations",
        "Full safe fast-math subset",
        "DTWC_ENABLE_SIMD",
        "Highway",
        "the build is -ffast-math",
    )
    present = [marker for marker in stale if marker in text]
    if present:
        raise AssertionError(f"remaining docs retain stale claims: {present}")

    required = (
        "`MVMode::Dependent`",
        "`MVMode::Independent`",
        "MSM and TWE reject",
        "`MVL2Dist`",
        "low-level primitives",
        "`prob.set_data(data)`",
        "unclustered",
        "singleton",
        "zero medoid separation",
        "unsquared dissimilarities",
        "denominator-zero",
        "empty input",
        "selected relaxations",
        "`-ffinite-math-only`",
        "`MAX + cost`",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise AssertionError(f"remaining docs omit current truth: {missing}")


def assert_f22_ordinary_call_hygiene() -> None:
    paths = (
        ".claude/commands/cluster.md",
        ".claude/commands/evaluate.md",
        ".claude/commands/help.md",
        ".claude/commands/visualize.md",
        ".claude/openmp-crashcourse.md",
        ".claude/reports/test_kasper_analysis/REPORT.md",
        ".claude/skills/python-wrapper-skill.md",
        ".claude/skills/matlab-wrapper-skill.md",
        "docs/content/api/interface-parity.md",
        "docs/content/method/gpu-backends.md",
        ".claude/reports/test_kasper_analysis/rerun_znorm.py",
        ".claude/reports/test_kasper_analysis/rescore_kasper.py",
        ".claude/reports/test_kasper_analysis/run_extended.py",
        ".claude/reports/test_kasper_analysis/run_kasper.py",
        ".claude/reports/test_kasper_analysis/run_preprocessed.py",
    )
    sources = {
        relative: (ROOT / relative).read_text(encoding="utf-8")
        for relative in paths
    }
    cxx_calls = (
        "set_numberOfClusters",
        "refreshDistanceMatrix",
        "readDistanceMatrix",
        "maxDistance",
        "distByInd",
        "isDistanceMatrixFilled",
        "printDistanceMatrix",
        "writeDistanceMatrix",
        "printClusters",
        "writeClusters",
        "writeMedoidMembers",
        "writeSilhouettes",
        "findTotalCost",
        "assignClusters",
        "calculateMedoids",
        "cluster_by_MIP",
        "cluster_by_kMedoidsLloyd",
        "cluster_size",
        "daviesBouldinIndex",
        "dunnIndex",
        "calinskiHarabaszIndex",
        "adjustedRandIndex",
        "normalizedMutualInformation",
        "setDataPath",
        "setResultsPath",
    )
    score_aliases = (
        "davies_bouldin_index",
        "dunn_index",
        "calinski_harabasz_index",
        "adjusted_rand_index",
        "normalized_mutual_information",
    )
    python_calls = (
        "set_number_of_clusters",
        "distance_matrix_numpy",
        "set_distance_matrix_from_numpy",
        *score_aliases,
    )
    cxx_unique = set(cxx_calls) | {
        "fillDistanceMatrix",
        "startColumn",
        "startRow",
        "maxIter",
        "N_repetition",
    }
    python_unique = set(python_calls) | {
        "n_repetition",
        "cluster_size",
        "ClusterResult",
        "medoid_indices",
    }
    inventory_shape = (
        len(cxx_unique),
        33,  # three overloads make 33 diagnostic entities from 30 names.
        len(python_unique),
        13,  # n_repetition has independently warned read/write operations.
        4 + 5 + 1 + len(score_aliases),
    )
    if inventory_shape != (30, 33, 12, 13, 15):
        raise AssertionError(
            f"F22 ordinary-call checker inventory drift: {inventory_shape}"
        )
    patterns = (
        (
            "C++ callable alias",
            re.compile(
                rf"\b(?:{'|'.join(map(re.escape, cxx_calls))})\s*\("
            ),
        ),
        (
            "C++ fill-distance alias",
            re.compile(r"\bfillDistanceMatrix(?!_BruteForce)\s*\("),
        ),
        (
            "C++ loader-setter alias",
            re.compile(
                r"\b(?:startColumn|startRow)\s*\(\s*(?!\))"
                r"|\.\s*(?:startColumn|startRow)\s*="
            ),
        ),
        (
            "C++ deprecated field",
            re.compile(r"(?:\.|->)\s*(?:maxIter|N_repetition)\b"),
        ),
        (
            "Python callable alias",
            re.compile(
                rf"\b(?:{'|'.join(map(re.escape, python_calls))})\s*\("
            ),
        ),
        (
            "Python property alias",
            re.compile(r"\.(?:n_repetition|cluster_size)\b"),
        ),
        (
            "Python module alias",
            re.compile(r"\bClusterResult\b"),
        ),
        (
            "Python Result.medoid_indices alias",
            re.compile(r"\bResult\s*\.\s*medoid_indices\b"),
        ),
        (
            "MATLAB configuration-property write alias",
            re.compile(
                r"\.\s*(?:Band|Verbose|MaxIter|NRepetition)\s*=(?!=)"
            ),
        ),
        (
            "MATLAB callable alias",
            re.compile(
                rf"\b(?:get_distance_matrix|"
                rf"{'|'.join(map(re.escape, score_aliases))})\s*\("
            ),
        ),
        (
            "MATLAB dependent-property read alias",
            re.compile(
                r"(?:\b(?:prob|problem)\s*\.|\bdtwc\.Problem\.)\s*"
                r"(?:Size|ClusterSize|Name|CentroidsInd|ClustersInd)\b"
            ),
        ),
    )
    pattern_by_label = dict(patterns)
    positive_controls = [
        *(
            ("C++ callable alias", f"obj.{name}(1)")
            for name in cxx_calls
        ),
        ("C++ fill-distance alias", "prob.fillDistanceMatrix()"),
        ("C++ loader-setter alias", "loader.startColumn(1)"),
        ("C++ loader-setter alias", "loader.startRow = 1"),
        ("C++ deprecated field", "prob.maxIter"),
        ("C++ deprecated field", "prob->N_repetition"),
        *(
            ("Python callable alias", f"obj.{name}()")
            for name in python_calls
        ),
        ("Python property alias", "prob.n_repetition"),
        ("Python property alias", "prob.cluster_size"),
        ("Python module alias", "from dtwcpp import ClusterResult"),
        ("Python Result.medoid_indices alias", "Result.medoid_indices"),
        *(
            (
                "MATLAB configuration-property write alias",
                f"prob.{name} = value",
            )
            for name in ("Band", "Verbose", "MaxIter", "NRepetition")
        ),
        ("MATLAB callable alias", "prob.get_distance_matrix()"),
        *(
            ("MATLAB callable alias", f"dtwc.{name}()")
            for name in score_aliases
        ),
        *(
            ("MATLAB dependent-property read alias", f"prob.{name}")
            for name in ("Size", "ClusterSize", "Name", "CentroidsInd", "ClustersInd")
        ),
    ]
    missed_controls = [
        f"{label}: {sample!r}"
        for label, sample in positive_controls
        if not pattern_by_label[label].search(sample)
    ]
    if missed_controls:
        raise AssertionError(
            "F22 ordinary-call checker misses its controls: "
            + "; ".join(missed_controls)
        )
    negative_controls = (
        ("C++ fill-distance alias", "Problem::fillDistanceMatrix_BruteForce"),
        ("C++ loader-setter alias", "loader.startColumn()"),
        ("C++ loader-setter alias", "loader.startRow()"),
        ("MATLAB configuration-property write alias", "'Band', 10"),
        ("Python Result.medoid_indices alias", "native_result.medoid_indices"),
    )
    false_positive_controls = [
        f"{label}: {sample!r}"
        for label, sample in negative_controls
        if pattern_by_label[label].search(sample)
    ]
    if false_positive_controls:
        raise AssertionError(
            "F22 ordinary-call checker rejects canonical/private controls: "
            + "; ".join(false_positive_controls)
        )
    violations: list[str] = []
    for relative, source in sources.items():
        for label, pattern in patterns:
            match = pattern.search(source)
            if match:
                line = source.count("\n", 0, match.start()) + 1
                violations.append(
                    f"{relative}:{line}: {label}: {match.group(0)!r}"
                )
    if violations:
        raise AssertionError(
            "ordinary F22 documentation/call sites retain aliases: "
            + "; ".join(violations)
        )
    text = "\n".join(sources.values())
    required = (
        "prob.set_band(10);",
        "fill_distance_matrix",
        "dist_by_ind",
        ".set_n_clusters(",
        "davies_bouldin(",
        "calinski_harabasz(",
        "adjusted_rand(",
        "normalized_mutual_info(",
    )
    missing = [marker for marker in required if marker not in text]
    if missing:
        raise AssertionError(
            f"ordinary F22 documentation/call sites omit canonical forms: {missing}"
        )


def cli_flags(text: str) -> set[str]:
    return set(re.findall(r"(?<![\w-])--[a-z][a-z0-9-]*", text))


def assert_cli_reference(binary: Path) -> None:
    proc = subprocess.run([str(binary), "--help"], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          encoding="utf-8", errors="replace", check=True)
    live = cli_flags(proc.stdout)
    docs_text = (ROOT / "docs/content/getting-started/cli.md").read_text(encoding="utf-8")
    documented = cli_flags(docs_text)
    missing = live - documented
    dead = documented - live
    if missing or dead:
        raise AssertionError(
            "CLI reference drift:\n"
            f"  live but undocumented: {sorted(missing)}\n"
            f"  documented but not live: {sorted(dead)}"
        )
    if "`float64`" not in docs_text:
        raise AssertionError("CLI dtype default is not documented as float64")

    config_text = (
        ROOT / "docs/content/getting-started/configuration.md"
    ).read_text(encoding="utf-8")
    config_flags = cli_flags(config_text)
    config_missing = live - config_flags
    config_dead = config_flags - live
    if config_missing or config_dead:
        raise AssertionError(
            "configuration reference drift:\n"
            f"  live but undocumented: {sorted(config_missing)}\n"
            f"  documented but not live: {sorted(config_dead)}"
        )

    source = (ROOT / "dtwc/dtwc_cl.cpp").read_text(encoding="utf-8")
    yaml_keys = set(re.findall(r'set_if_unset\("([a-z0-9-]+)"', source))
    missing_yaml = sorted(
        key for key in yaml_keys if f"`{key}`" not in config_text
    )
    if missing_yaml:
        raise AssertionError(
            f"configuration page omits canonical YAML keys: {missing_yaml}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path,
                        help="built dtwc_cl binary for exact flag comparison")
    args = parser.parse_args()

    subprocess.run([sys.executable, str(ROOT / "scripts/generate_docs.py"), "--check"],
                   check=True)
    assert_freeze_governance()
    assert_contract_audit_state()
    assert_python_binary_checkpoint_contract()
    assert_migration_behaviors()
    assert_f22_changelog()
    assert_rc1_changelog()
    assert_env_messages()
    assert_tier1_signatures()
    assert_method_catalog()
    assert_dtw_derivation_sync()
    assert_lb_keogh_derivation_sync()
    assert_gpu_backend_page()
    assert_remaining_docs_truth()
    assert_f22_ordinary_call_hygiene()
    if args.cli is not None:
        assert_cli_reference(args.cli.resolve())
    print("documentation contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
