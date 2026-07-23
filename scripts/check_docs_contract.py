#!/usr/bin/env python3
"""Adversarial documentation drift checks against live code and CLI output."""

from __future__ import annotations

import argparse
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
        f"F{number}" for number in range(18, 27)
        if f"F{number}" not in contract
    ]
    if missing_findings:
        raise AssertionError(
            f"frozen contract hides 2.0 implementation gaps: {missing_findings}"
        )


def assert_migration_behaviors() -> None:
    migration = (ROOT / "docs/content/guides/migration.md").read_text(
        encoding="utf-8"
    )
    required = (
        "`Result.distance_matrix` is `None` for matrix-free methods",
        "Explicit GPU requests no longer warn and run on CPU",
        "Requesting an unavailable MIP solver no longer prints and returns",
    )
    missing = [item for item in required if item not in migration]
    if missing:
        raise AssertionError(f"migration guide omits behaviors: {missing}")


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
            "finding F12",
            "CUDA source currently uses an endpoint-scaled corridor",
        ),
        "citations": (
            "10.1109/TASSP.1978.1163055",
            "Sakoe-Chiba-DTW.pdf",
            "**[confirmed]**",
            "equations (6)--(8)",
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
        name: [marker for marker in markers if marker not in text[name]]
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
        if line.startswith("|") and ("$|" in line or "|$" in line)
    ]
    if broken_table_math:
        raise AssertionError(
            f"D1 derivation drift: raw table math pipes {broken_table_math}"
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
        "`+inf`",
        "`dtwc::KernelOverride`",
        "`Problem::lb_strategy` is CPU-only",
        "`DistanceMatrixStrategy::Auto` is CPU-only",
        "O(N·L·r)",
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
    assert_migration_behaviors()
    assert_rc1_changelog()
    assert_env_messages()
    assert_tier1_signatures()
    assert_method_catalog()
    assert_dtw_derivation_sync()
    assert_gpu_backend_page()
    assert_remaining_docs_truth()
    if args.cli is not None:
        assert_cli_reference(args.cli.resolve())
    print("documentation contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
