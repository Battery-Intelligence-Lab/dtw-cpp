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


def assert_migration_behaviors() -> None:
    migration = (ROOT / "docs/content/guides/migration.md").read_text(
        encoding="utf-8"
    )
    if "`Result.distance_matrix` is `None` for matrix-free methods" not in migration:
        raise AssertionError(
            "migration guide omits matrix-free Result.distance_matrix behavior"
        )


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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path,
                        help="built dtwc_cl binary for exact flag comparison")
    args = parser.parse_args()

    subprocess.run([sys.executable, str(ROOT / "scripts/generate_docs.py"), "--check"],
                   check=True)
    assert_freeze_governance()
    assert_migration_behaviors()
    assert_env_messages()
    assert_tier1_signatures()
    if args.cli is not None:
        assert_cli_reference(args.cli.resolve())
    print("documentation contract checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
