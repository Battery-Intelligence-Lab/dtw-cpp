#!/usr/bin/env python3
"""Permanent dual-release mutation gate for the F22 MATLAB deprecations.

The public ``.m`` compatibility layer is mutated one case at a time.  Every
mutant is driven through the real Release MEX on MATLAB R2024b and R2025b,
with independent preference/temp directories and exact source/MEX provenance.
Recovery snapshots live under the configured build directory.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid
from typing import Iterable


PASS_MARKER = (
    "F22_MATLAB_DEPRECATION aliases=15/15 "
    "warning_profiles=15/15 messages=15/15 "
    "canonical_silent=15/15 equivalence=15/15 "
    "constructor_silent=1/1 tier1_silent=1/1 skips=0 verdict=PASS"
)
FAIL_MARKER_TEMPLATE = (
    "F22_MATLAB_DEPRECATION aliases=15/15 "
    "warning_profiles={warning_profiles}/15 messages={messages}/15 "
    "canonical_silent={canonical_silent}/15 equivalence={equivalence}/15 "
    "constructor_silent=1/1 tier1_silent={tier1_silent}/1 "
    "skips=0 verdict=FAIL"
)
RESULT_PATTERN = re.compile(
    r"F22_MATLAB_MUTANT_RESULT id=(?P<id>M[WBP]\d{2}) "
    r"release=(?P<release>R202[45]b) passed=(?P<passed>[01]) "
    r"failed=(?P<failed>[01]) incomplete=(?P<incomplete>[01])"
)
CONTROL_PATTERN = re.compile(
    r"F22_MATLAB_CONTROL_RESULT label=(?P<label>initial|final) "
    r"release=(?P<release>R202[45]b) passed=(?P<passed>[01]) "
    r"failed=(?P<failed>[01]) incomplete=(?P<incomplete>[01])"
)
INVALID_FAILURE_FRAGMENTS = (
    "parse error",
    "undefined function",
    "unrecognized function or variable",
    "invalid mex-file",
)


@dataclass(frozen=True)
class Edit:
    path: str
    needle: str
    replacement: str
    expected_count: int = 1


@dataclass(frozen=True)
class Mutation:
    name: str
    category: str
    source: str
    symbol: str
    diagnostic: str
    fail_marker: str
    marker_required: bool
    expected_incomplete: int
    edits: tuple[Edit, ...]


PROBLEM = "bindings/matlab/+dtwc/Problem.m"
ESTIMATOR = "bindings/matlab/+dtwc/DTWClustering.m"
DB_OLD = "bindings/matlab/+dtwc/davies_bouldin_index.m"
DUNN_OLD = "bindings/matlab/+dtwc/dunn_index.m"
CH_OLD = "bindings/matlab/+dtwc/calinski_harabasz_index.m"
ARI_OLD = "bindings/matlab/+dtwc/adjusted_rand_index.m"
NMI_OLD = "bindings/matlab/+dtwc/normalized_mutual_information.m"
TARGETS = (
    PROBLEM,
    ESTIMATOR,
    DB_OLD,
    DUNN_OLD,
    CH_OLD,
    ARI_OLD,
    NMI_OLD,
)
MEX_SOURCE = "bindings/matlab/dtwc_mex.cpp"
TEST_SOURCE = "tests/matlab/test_contract_parity.m"
IMMUTABLE_SOURCES = (
    MEX_SOURCE,
    TEST_SOURCE,
    "bindings/matlab/+dtwc/+test/parallelisation.m",
    "bindings/matlab/+dtwc/davies_bouldin.m",
    "bindings/matlab/+dtwc/dunn.m",
    "bindings/matlab/+dtwc/calinski_harabasz.m",
    "bindings/matlab/+dtwc/adjusted_rand.m",
    "bindings/matlab/+dtwc/normalized_mutual_info.m",
)
CATEGORIES = ("warning_removal", "behavior_corruption", "policy")


def fail_marker(
    *,
    warning_profiles: int = 15,
    messages: int = 15,
    canonical_silent: int = 15,
    equivalence: int = 15,
    tier1_silent: int = 1,
) -> str:
    return FAIL_MARKER_TEMPLATE.format(
        warning_profiles=warning_profiles,
        messages=messages,
        canonical_silent=canonical_silent,
        equivalence=equivalence,
        tier1_silent=tier1_silent,
    )


def warning_block(old_name: str, new_name: str, indent: str) -> str:
    return (
        f"{indent}warning('dtwc:deprecatedAlias', ...\n"
        f"{indent}    ['''{old_name}'' is deprecated; use ' ...\n"
        f"{indent}     '''{new_name}'' instead.']);"
    )


def warning_removal(
    name: str,
    path: str,
    symbol: str,
    old_name: str,
    new_name: str,
    indent: str,
    *,
    marker_required: bool = True,
    expected_incomplete: int = 0,
) -> Mutation:
    return Mutation(
        name=name,
        category="warning_removal",
        source=path,
        symbol=symbol,
        diagnostic=f"F22 warning ID/count mismatch for {old_name}.",
        fail_marker=fail_marker(warning_profiles=14, messages=14),
        marker_required=marker_required,
        expected_incomplete=expected_incomplete,
        edits=(
            Edit(
                path,
                warning_block(old_name, new_name, indent),
                f"{indent}% F22 mutation: deprecation warning removed.",
            ),
        ),
    )


def behavior_mutation(
    name: str,
    path: str,
    symbol: str,
    old_name: str,
    new_name: str,
    needle: str,
    replacement: str,
    *,
    expected_incomplete: int,
) -> Mutation:
    return Mutation(
        name=name,
        category="behavior_corruption",
        source=path,
        symbol=symbol,
        diagnostic=f"F22 behavior mismatch: {old_name} versus {new_name}.",
        fail_marker=fail_marker(equivalence=14),
        marker_required=False,
        expected_incomplete=expected_incomplete,
        edits=(Edit(path, needle, replacement),),
    )


MUTATIONS = (
    warning_removal(
        "MW01", PROBLEM, "dtwc.Problem",
        "dtwc.Problem.Band", "dtwc.Problem.set_band", "            ",
        marker_required=False,
        expected_incomplete=1,
    ),
    warning_removal(
        "MW02", PROBLEM, "dtwc.Problem",
        "dtwc.Problem.Verbose", "dtwc.Problem.set_verbose", "            ",
        marker_required=False,
        expected_incomplete=1,
    ),
    warning_removal(
        "MW03", PROBLEM, "dtwc.Problem",
        "dtwc.Problem.MaxIter", "dtwc.Problem.set_max_iter", "            ",
        marker_required=False,
        expected_incomplete=1,
    ),
    warning_removal(
        "MW04", PROBLEM, "dtwc.Problem",
        "dtwc.Problem.NRepetition",
        "dtwc.Problem.set_n_repetitions",
        "            ",
        marker_required=False,
        expected_incomplete=1,
    ),
    warning_removal(
        "MW05", PROBLEM, "dtwc.Problem",
        "dtwc.Problem.get_distance_matrix",
        "dtwc.Problem.distance_matrix",
        "            ",
    ),
    warning_removal(
        "MW06", PROBLEM, "dtwc.Problem",
        "dtwc.Problem.Size", "dtwc.Problem.size", "            ",
    ),
    warning_removal(
        "MW07", PROBLEM, "dtwc.Problem",
        "dtwc.Problem.ClusterSize", "dtwc.Problem.n_clusters", "            ",
    ),
    warning_removal(
        "MW08", PROBLEM, "dtwc.Problem",
        "dtwc.Problem.Name", "dtwc.Problem.name", "            ",
    ),
    warning_removal(
        "MW09", PROBLEM, "dtwc.Problem",
        "dtwc.Problem.CentroidsInd", "dtwc.Problem.medoids", "            ",
    ),
    warning_removal(
        "MW10", PROBLEM, "dtwc.Problem",
        "dtwc.Problem.ClustersInd", "dtwc.Problem.labels", "            ",
    ),
    warning_removal(
        "MW11", DB_OLD, "dtwc.davies_bouldin_index",
        "dtwc.davies_bouldin_index", "dtwc.davies_bouldin", "    ",
    ),
    warning_removal(
        "MW12", DUNN_OLD, "dtwc.dunn_index",
        "dtwc.dunn_index", "dtwc.dunn", "    ",
    ),
    warning_removal(
        "MW13", CH_OLD, "dtwc.calinski_harabasz_index",
        "dtwc.calinski_harabasz_index", "dtwc.calinski_harabasz", "    ",
    ),
    warning_removal(
        "MW14", ARI_OLD, "dtwc.adjusted_rand_index",
        "dtwc.adjusted_rand_index", "dtwc.adjusted_rand", "    ",
    ),
    warning_removal(
        "MW15", NMI_OLD, "dtwc.normalized_mutual_information",
        "dtwc.normalized_mutual_information",
        "dtwc.normalized_mutual_info",
        "    ",
    ),
    behavior_mutation(
        "MB01", PROBLEM, "dtwc.Problem",
        "dtwc.Problem.Band", "dtwc.Problem.set_band",
        "            obj.set_band(val);",
        "            obj.set_band(val + 1);",
        expected_incomplete=1,
    ),
    behavior_mutation(
        "MB02", PROBLEM, "dtwc.Problem",
        "dtwc.Problem.Verbose", "dtwc.Problem.set_verbose",
        "            obj.set_verbose(val);",
        "            obj.set_verbose(~val);",
        expected_incomplete=1,
    ),
    behavior_mutation(
        "MB03", PROBLEM, "dtwc.Problem",
        "dtwc.Problem.MaxIter", "dtwc.Problem.set_max_iter",
        "            obj.set_max_iter(val);",
        "            obj.set_max_iter(val + 1);",
        expected_incomplete=1,
    ),
    behavior_mutation(
        "MB04", PROBLEM, "dtwc.Problem",
        "dtwc.Problem.NRepetition", "dtwc.Problem.set_n_repetitions",
        "            obj.set_n_repetitions(val);",
        "            obj.set_n_repetitions(val + 1);",
        expected_incomplete=1,
    ),
    behavior_mutation(
        "MB05", PROBLEM, "dtwc.Problem",
        "dtwc.Problem.get_distance_matrix", "dtwc.Problem.distance_matrix",
        "            D = obj.distance_matrix();",
        "            D = obj.distance_matrix();\n"
        "            D(1, 2) = D(1, 2) + 1;",
        expected_incomplete=1,
    ),
    behavior_mutation(
        "MB06", PROBLEM, "dtwc.Problem",
        "dtwc.Problem.Size", "dtwc.Problem.size",
        "            val = obj.size();",
        "            val = obj.size() + 1;",
        expected_incomplete=1,
    ),
    behavior_mutation(
        "MB07", PROBLEM, "dtwc.Problem",
        "dtwc.Problem.ClusterSize", "dtwc.Problem.n_clusters",
        "            val = obj.n_clusters();",
        "            val = obj.n_clusters() + 1;",
        expected_incomplete=1,
    ),
    behavior_mutation(
        "MB08", PROBLEM, "dtwc.Problem",
        "dtwc.Problem.Name", "dtwc.Problem.name",
        "            val = obj.name();",
        "            val = [obj.name() '_mutant'];",
        expected_incomplete=1,
    ),
    behavior_mutation(
        "MB09", PROBLEM, "dtwc.Problem",
        "dtwc.Problem.CentroidsInd", "dtwc.Problem.medoids",
        "            val = obj.medoids();",
        "            val = obj.medoids();\n"
        "            val(1) = mod(val(1), 6) + 1;",
        expected_incomplete=0,
    ),
    behavior_mutation(
        "MB10", PROBLEM, "dtwc.Problem",
        "dtwc.Problem.ClustersInd", "dtwc.Problem.labels",
        "            val = obj.labels();",
        "            val = obj.labels();\n"
        "            val(1) = 3 - val(1);",
        expected_incomplete=0,
    ),
    behavior_mutation(
        "MB11", DB_OLD, "dtwc.davies_bouldin_index",
        "dtwc.davies_bouldin_index", "dtwc.davies_bouldin",
        "    db = dtwc.davies_bouldin(prob);",
        "    db = -dtwc.davies_bouldin(prob);",
        expected_incomplete=1,
    ),
    behavior_mutation(
        "MB12", DUNN_OLD, "dtwc.dunn_index",
        "dtwc.dunn_index", "dtwc.dunn",
        "    di = dtwc.dunn(prob);",
        "    di = -dtwc.dunn(prob);",
        expected_incomplete=1,
    ),
    behavior_mutation(
        "MB13", CH_OLD, "dtwc.calinski_harabasz_index",
        "dtwc.calinski_harabasz_index", "dtwc.calinski_harabasz",
        "    ch = dtwc.calinski_harabasz(prob);",
        "    ch = -dtwc.calinski_harabasz(prob);",
        expected_incomplete=1,
    ),
    behavior_mutation(
        "MB14", ARI_OLD, "dtwc.adjusted_rand_index",
        "dtwc.adjusted_rand_index", "dtwc.adjusted_rand",
        "    ari = dtwc.adjusted_rand(int32(labels1(:)'), int32(labels2(:)'));",
        "    ari = -dtwc.adjusted_rand(int32(labels1(:)'), int32(labels2(:)'));",
        expected_incomplete=1,
    ),
    behavior_mutation(
        "MB15", NMI_OLD, "dtwc.normalized_mutual_information",
        "dtwc.normalized_mutual_information", "dtwc.normalized_mutual_info",
        "    nmi = dtwc.normalized_mutual_info( ...",
        "    nmi = -dtwc.normalized_mutual_info( ...",
        expected_incomplete=1,
    ),
    Mutation(
        name="MP01",
        category="policy",
        source=PROBLEM,
        symbol="dtwc.Problem",
        diagnostic="F22 warning ID/count mismatch for dtwc.Problem.Band.",
        fail_marker=fail_marker(warning_profiles=14),
        marker_required=True,
        expected_incomplete=0,
        edits=(
            Edit(
                PROBLEM,
                warning_block(
                    "dtwc.Problem.Band", "dtwc.Problem.set_band", "            "
                ),
                warning_block(
                    "dtwc.Problem.Band", "dtwc.Problem.set_band", "            "
                )
                + "\n"
                + warning_block(
                    "dtwc.Problem.Band", "dtwc.Problem.set_band", "            "
                ),
            ),
        ),
    ),
    Mutation(
        name="MP02",
        category="policy",
        source=PROBLEM,
        symbol="dtwc.Problem",
        diagnostic=(
            "F22 canonical operation warned for "
            "dtwc.Problem.distance_matrix."
        ),
        fail_marker=fail_marker(
            warning_profiles=14, canonical_silent=14
        ),
        marker_required=True,
        expected_incomplete=0,
        edits=(
            Edit(
                PROBLEM,
                "            D = dtwc_mex('Problem_get_distance_matrix', "
                "obj.Handle);",
                warning_block(
                    "dtwc.Problem.get_distance_matrix",
                    "dtwc.Problem.distance_matrix",
                    "            ",
                )
                + "\n"
                + "            D = dtwc_mex("
                "'Problem_get_distance_matrix', obj.Handle);",
            ),
        ),
    ),
    Mutation(
        name="MP03",
        category="policy",
        source=ESTIMATOR,
        symbol="dtwc.DTWClustering",
        diagnostic=(
            "Canonical DTWClustering.fit emitted a deprecation warning."
        ),
        fail_marker=fail_marker(tier1_silent=0),
        marker_required=True,
        expected_incomplete=0,
        edits=(
            Edit(
                ESTIMATOR,
                "                prob.set_band(obj.Band);",
                "                prob.Band = obj.Band;",
            ),
        ),
    ),
)


class GateError(RuntimeError):
    """Harness failure; never counts as a mutation kill."""


@dataclass(frozen=True)
class MatlabRelease:
    name: str
    executable: Path


@dataclass(frozen=True)
class Invocation:
    release: str
    output: str
    mex_hash_checks: int
    killed: bool | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-dir", type=Path, default=Path("build/mex-verify-msvc")
    )
    parser.add_argument(
        "--matlab-r2024b",
        type=Path,
        default=Path(r"C:\Program Files\MATLAB\R2024b\bin\matlab.exe"),
    )
    parser.add_argument(
        "--matlab-r2025b",
        type=Path,
        default=Path(r"C:\Program Files\MATLAB\R2025b\bin\matlab.exe"),
    )
    parser.add_argument(
        "--mode", choices=("preflight", "execute", "restore"),
        default="preflight",
    )
    parser.add_argument(
        "--confirm-exclusive-build-access", action="store_true"
    )
    return parser.parse_args()


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def normal(path: Path | str) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def require_inside(path: Path, parent: Path, description: str) -> Path:
    resolved = path.resolve()
    try:
        common = os.path.commonpath((normal(resolved), normal(parent)))
    except ValueError as error:
        raise GateError(f"invalid {description}: {resolved}") from error
    if common != normal(parent):
        raise GateError(f"{description} escapes its root: {resolved}")
    return resolved


def run(
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise GateError(f"command failed to run: {command!r}: {error}") from error


def replace_edit(text: str, edit: Edit, mutation_name: str) -> str:
    needle = edit.needle
    replacement = edit.replacement
    observed = text.count(needle)
    if observed == 0 and "\n" in needle and "\r\n" in text:
        needle = needle.replace("\n", "\r\n")
        replacement = replacement.replace("\n", "\r\n")
        observed = text.count(needle)
    if observed != edit.expected_count:
        raise GateError(
            f"{mutation_name}: {edit.path} expected {edit.expected_count} "
            f"occurrence(s), observed {observed}: {edit.needle!r}"
        )
    return text.replace(needle, replacement)


def materialized(
    mutation: Mutation, originals: dict[str, bytes]
) -> dict[str, bytes]:
    edits_by_path: dict[str, list[Edit]] = {}
    for edit in mutation.edits:
        edits_by_path.setdefault(edit.path, []).append(edit)
    result: dict[str, bytes] = {}
    for relative, edits in edits_by_path.items():
        try:
            text = originals[relative].decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise GateError(f"{relative} is not strict UTF-8") from error
        for edit in edits:
            text = replace_edit(text, edit, mutation.name)
        payload = text.encode("utf-8")
        if payload == originals[relative]:
            raise GateError(f"{mutation.name}: materialization changed no bytes")
        result[relative] = payload
    return result


def assert_inventory(originals: dict[str, bytes]) -> None:
    if len(MUTATIONS) != 33:
        raise GateError(f"inventory={len(MUTATIONS)}, expected 33")
    counts = {
        category: sum(item.category == category for item in MUTATIONS)
        for category in CATEGORIES
    }
    expected = {
        "warning_removal": 15,
        "behavior_corruption": 15,
        "policy": 3,
    }
    if counts != expected:
        raise GateError(f"category drift: {counts!r}")
    if len({item.name for item in MUTATIONS}) != 33:
        raise GateError("mutation names are not unique")
    expected_names = {
        *(f"MW{index:02d}" for index in range(1, 16)),
        *(f"MB{index:02d}" for index in range(1, 16)),
        *(f"MP{index:02d}" for index in range(1, 4)),
    }
    if {item.name for item in MUTATIONS} != expected_names:
        raise GateError("mutation ID inventory drift")
    fatal_names = {
        *(f"MW{index:02d}" for index in range(1, 5)),
        *(f"MB{index:02d}" for index in range(1, 9)),
        *(f"MB{index:02d}" for index in range(11, 16)),
    }
    observed_fatal = {
        item.name for item in MUTATIONS if item.expected_incomplete == 1
    }
    if observed_fatal != fatal_names:
        raise GateError("fatal/incomplete mutation inventory drift")
    if any(item.expected_incomplete not in (0, 1) for item in MUTATIONS):
        raise GateError("invalid expected incomplete count")
    for mutation in MUTATIONS:
        if mutation.source not in TARGETS:
            raise GateError(f"{mutation.name}: source is outside target allowlist")
        payloads = materialized(mutation, originals)
        if set(payloads) != {mutation.source}:
            raise GateError(f"{mutation.name}: must mutate exactly one source")


def assert_clean_oracles(repository: Path) -> None:
    guarded = (*TARGETS, *IMMUTABLE_SOURCES)
    completed = run(
        ["git", "diff", "--quiet", "HEAD", "--", *guarded],
        cwd=repository,
        timeout=30,
    )
    if completed.returncode != 0:
        raise GateError(
            "mutation targets or immutable oracles have staged/unstaged changes"
        )


def find_mex(build: Path) -> Path:
    expected = (build / "bin/dtwc_mex.mexw64").resolve()
    if not expected.is_file():
        raise GateError(f"fresh Release MEX is missing: {expected}")
    return expected


def write_recovery(
    recovery: Path,
    repository: Path,
    originals: dict[str, bytes],
    hashes: dict[str, str],
    mex: Path,
) -> Path:
    recovery.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    for index, relative in enumerate(TARGETS):
        snapshot = recovery / f"{index:02d}-{Path(relative).name}.bin"
        snapshot.write_bytes(originals[relative])
        records.append(
            {
                "kind": "source",
                "relative": relative,
                "snapshot": snapshot.name,
                "sha256": hashes[relative],
            }
        )
    mex_payload = mex.read_bytes()
    mex_snapshot = recovery / "dtwc_mex.mexw64.bin"
    mex_snapshot.write_bytes(mex_payload)
    records.append(
        {
            "kind": "mex",
            "relative": os.fspath(mex.relative_to(repository)),
            "snapshot": mex_snapshot.name,
            "sha256": sha256(mex_payload),
        }
    )
    for record in records:
        snapshot = recovery / record["snapshot"]
        if sha256(snapshot.read_bytes()) != record["sha256"]:
            raise GateError(f"recovery snapshot mismatch: {snapshot}")
    manifest = recovery / "manifest.json"
    manifest.write_text(
        json.dumps({"state": "active", "files": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def restore_from_manifest(repository: Path, recovery: Path) -> None:
    manifest = recovery / "manifest.json"
    if not manifest.is_file():
        raise GateError(f"missing recovery manifest: {manifest}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    records = data.get("files", [])
    if len(records) != 8:
        raise GateError("recovery manifest record count drift")
    for record in records:
        relative = record["relative"]
        if record["kind"] == "source" and relative not in TARGETS:
            raise GateError(f"unexpected recovery source: {relative}")
        target = require_inside(
            repository / relative, repository, "recovery target"
        )
        snapshot = require_inside(
            recovery / record["snapshot"], recovery, "recovery snapshot"
        )
        payload = snapshot.read_bytes()
        if sha256(payload) != record["sha256"]:
            raise GateError(f"corrupt recovery snapshot: {snapshot}")
        target.write_bytes(payload)
        if sha256(target.read_bytes()) != record["sha256"]:
            raise GateError(f"restore hash mismatch: {target}")
    print(
        "F22_MATLAB_MUTATION_RESTORE sources=7/7 mex=1/1 verdict=PASS"
    )


def restore_sources(
    repository: Path,
    originals: dict[str, bytes],
    hashes: dict[str, str],
    relatives: Iterable[str] = TARGETS,
) -> None:
    for relative in relatives:
        (repository / relative).write_bytes(originals[relative])
    for relative in TARGETS:
        observed = sha256((repository / relative).read_bytes())
        if observed != hashes[relative]:
            raise GateError(
                f"source restoration failed: {relative}: "
                f"{observed} != {hashes[relative]}"
            )


def assert_immutable(
    repository: Path, immutable_hashes: dict[str, str]
) -> None:
    for relative, expected in immutable_hashes.items():
        observed = sha256((repository / relative).read_bytes())
        if observed != expected:
            raise GateError(
                f"immutable source changed: {relative}: "
                f"{observed} != {expected}"
            )


def fresh_mex_build(
    repository: Path,
    build: Path,
    immutable_hashes: dict[str, str],
) -> tuple[Path, str]:
    cache = build / "CMakeCache.txt"
    if not cache.is_file():
        raise GateError(f"CMake cache is missing: {cache}")
    cache_text = cache.read_text(encoding="utf-8", errors="replace")
    if "DTWC_ALLOW_SEQUENTIAL:BOOL=OFF" not in cache_text:
        raise GateError("MEX build is not the registered OpenMP build")
    cmake = shutil.which("cmake")
    if cmake is None:
        raise GateError("cmake is unavailable on PATH")
    mex_source = repository / MEX_SOURCE
    source_hash = sha256(mex_source.read_bytes())
    os.utime(mex_source, None)
    completed = run(
        [
            cmake,
            "--build",
            os.fspath(build),
            "--config",
            "Release",
            "--target",
            "dtwc_mex",
        ],
        cwd=repository,
        timeout=600,
    )
    if completed.returncode != 0:
        raise GateError(
            f"fresh MEX build failed\n{completed.stdout[-16000:]}"
        )
    if "dtwc_mex.cpp" not in completed.stdout:
        raise GateError(
            "fresh MEX build did not compile dtwc_mex.cpp\n"
            f"{completed.stdout[-8000:]}"
        )
    mex = find_mex(build)
    if ".mexw64" not in completed.stdout:
        raise GateError(
            "fresh MEX build did not report the linked MEX\n"
            f"{completed.stdout[-8000:]}"
        )
    if sha256(mex_source.read_bytes()) != source_hash:
        raise GateError("fresh MEX build changed dtwc_mex.cpp bytes")
    assert_immutable(repository, immutable_hashes)
    mex_hash = sha256(mex.read_bytes())
    print(
        "F22_MATLAB_MEX_BUILD source_compiled=1 linked=1 "
        "openmp_cache=1 "
        f"sha256={mex_hash} bytes={mex.stat().st_size} verdict=PASS"
    )
    return mex, mex_hash


def matlab_quote(value: Path | str) -> str:
    return os.fspath(value).replace("'", "''")


def matlab_command(
    *,
    repository: Path,
    mex: Path,
    expected_source: Path,
    symbol: str,
    release: str,
    result_statement: str,
) -> str:
    bindings = repository / "bindings/matlab"
    tests = repository / "tests/matlab"
    test_file = repository / TEST_SOURCE
    return (
        "restoredefaultpath;"
        f"addpath('{matlab_quote(bindings)}');"
        f"addpath('{matlab_quote(tests)}');"
        f"addpath('{matlab_quote(mex.parent)}');"
        "rehash path;"
        f"expectedMex='{matlab_quote(mex)}';"
        "observedMex=which('dtwc_mex');"
        "assert(strcmpi(observedMex,expectedMex),"
        "'F22:staleMex','Unexpected MEX: %s',observedMex);"
        "allMex=which('dtwc_mex','-all');"
        "if ischar(allMex),allMex={allMex};end;"
        "assert(numel(allMex)==1,'F22:ambiguousMex',"
        "'Expected one MEX, observed %d.',numel(allMex));"
        f"expectedSource='{matlab_quote(expected_source)}';"
        f"observedSource=which('{symbol}');"
        "assert(strcmpi(observedSource,expectedSource),"
        "'F22:staleSource','Unexpected source: %s',observedSource);"
        f"observedRelease=['R' version('-release')];"
        f"assert(strcmp(observedRelease,'{release}'),"
        "'F22:wrongRelease','Unexpected release: %s',observedRelease);"
        "probe=dtwc.test.parallelisation();"
        "expectedFields={'available';'max_threads';'pass';'reason';"
        "'threads_engaged'};"
        "assert(isstruct(probe)&&isscalar(probe)&&"
        "isequal(sort(fieldnames(probe)),expectedFields),"
        "'F22:openmpSchema','OpenMP probe schema drift.');"
        "assert(islogical(probe.available)&&isscalar(probe.available)&&"
        "islogical(probe.pass)&&isscalar(probe.pass)&&"
        "isnumeric(probe.max_threads)&&isscalar(probe.max_threads)&&"
        "isfinite(probe.max_threads)&&probe.max_threads=="
        "floor(probe.max_threads)&&"
        "isnumeric(probe.threads_engaged)&&"
        "isscalar(probe.threads_engaged)&&"
        "isfinite(probe.threads_engaged)&&probe.threads_engaged=="
        "floor(probe.threads_engaged)&&"
        "(ischar(probe.reason)||"
        "(isstring(probe.reason)&&isscalar(probe.reason)))&&"
        "probe.available&&probe.pass&&"
        "probe.max_threads>=2&&probe.threads_engaged>=2&&"
        "isempty(probe.reason),'F22:openmpProbe',"
        "'OpenMP Release MEX did not engage.');"
        f"allSuite=testsuite('{matlab_quote(test_file)}');"
        "suite=allSuite(contains({allSuite.Name},"
        "'test_f22_matlab_deprecation_policy'));"
        "assert(numel(suite)==1,'F22:testSelection',"
        "'Expected one focused test, observed %d.',numel(suite));"
        "runner=matlab.unittest.TestRunner.withTextOutput("
        "'OutputDetail',matlab.unittest.Verbosity.Detailed);"
        "results=runner.run(suite);"
        "passed=sum([results.Passed]);"
        "failed=sum([results.Failed]);"
        "incomplete=sum([results.Incomplete]);"
        + result_statement
    )


def unique_env(build: Path, label: str) -> dict[str, str]:
    env = os.environ.copy()
    unique = uuid.uuid4().hex
    runtime = build / "tests/f22-matlab-runtime" / f"{label}-{unique}"
    prefs = runtime / "prefs"
    temp = runtime / "temp"
    prefs.mkdir(parents=True, exist_ok=False)
    temp.mkdir(parents=True, exist_ok=False)
    env["MATLAB_PREFDIR"] = os.fspath(prefs)
    env["TEMP"] = os.fspath(temp)
    env["TMP"] = os.fspath(temp)
    env["OMP_NUM_THREADS"] = "2"
    return env


def invoke_matlab(
    *,
    repository: Path,
    build: Path,
    mex: Path,
    mex_hash: str,
    release: MatlabRelease,
    expected_source: Path,
    symbol: str,
    label: str,
    result_statement: str,
) -> tuple[str, int]:
    before = sha256(mex.read_bytes())
    if before != mex_hash:
        raise GateError(
            f"{label}/{release.name}: MEX hash drift before invocation"
        )
    command = matlab_command(
        repository=repository,
        mex=mex,
        expected_source=expected_source,
        symbol=symbol,
        release=release.name,
        result_statement=result_statement,
    )
    completed = run(
        [os.fspath(release.executable), "-batch", command],
        cwd=repository,
        env=unique_env(build, f"{label}-{release.name}"),
        timeout=240,
    )
    after = sha256(mex.read_bytes())
    if after != mex_hash:
        raise GateError(
            f"{label}/{release.name}: MEX hash drift after invocation"
        )
    if completed.returncode != 0:
        raise GateError(
            f"{label}/{release.name}: MATLAB exited "
            f"{completed.returncode}\n{completed.stdout[-16000:]}"
        )
    return completed.stdout, 2


def run_control(
    *,
    label: str,
    repository: Path,
    build: Path,
    mex: Path,
    mex_hash: str,
    releases: tuple[MatlabRelease, MatlabRelease],
) -> list[Invocation]:
    def one(release: MatlabRelease) -> Invocation:
        statement = (
            "fprintf('F22_MATLAB_CONTROL_RESULT label="
            f"{label} release={release.name} passed=%d failed=%d "
            "incomplete=%d\\n',passed,failed,incomplete);"
        )
        output, checks = invoke_matlab(
            repository=repository,
            build=build,
            mex=mex,
            mex_hash=mex_hash,
            release=release,
            expected_source=repository / PROBLEM,
            symbol="dtwc.Problem",
            label=f"control-{label}",
            result_statement=statement,
        )
        matches = list(CONTROL_PATTERN.finditer(output))
        if len(matches) != 1:
            raise GateError(
                f"control {label}/{release.name}: result marker "
                f"count={len(matches)}\n{output[-16000:]}"
            )
        values = matches[0].groupdict()
        expected = {
            "label": label,
            "release": release.name,
            "passed": "1",
            "failed": "0",
            "incomplete": "0",
        }
        if values != expected:
            raise GateError(
                f"control {label}/{release.name}: result={values!r}"
            )
        if output.count(PASS_MARKER) != 1:
            raise GateError(
                f"control {label}/{release.name}: PASS marker count drift\n"
                f"{output[-16000:]}"
            )
        return Invocation(release.name, output, checks, None)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {release.name: pool.submit(one, release) for release in releases}
        results = [futures[release.name].result() for release in releases]
    for result in results:
        marker = CONTROL_PATTERN.search(result.output)
        assert marker is not None
        print(marker.group(0))
    return results


def classify_mutant(
    mutation: Mutation,
    release: MatlabRelease,
    output: str,
) -> bool:
    matches = list(RESULT_PATTERN.finditer(output))
    if len(matches) != 1:
        raise GateError(
            f"{mutation.name}/{release.name}: result marker "
            f"count={len(matches)}\n{output[-16000:]}"
        )
    values = matches[0].groupdict()
    if values["id"] != mutation.name or values["release"] != release.name:
        raise GateError(
            f"{mutation.name}/{release.name}: result marker identity drift"
        )
    result = (
        int(values["passed"]),
        int(values["failed"]),
        int(values["incomplete"]),
    )
    if result == (1, 0, 0):
        if output.count(PASS_MARKER) != 1:
            raise GateError(
                f"{mutation.name}/{release.name}: survivor lacked PASS marker"
            )
        return False
    expected_result = (0, 1, mutation.expected_incomplete)
    if result != expected_result:
        raise GateError(
            f"{mutation.name}/{release.name}: expected result "
            f"ledger={expected_result}, observed={result}"
        )
    if PASS_MARKER in output:
        raise GateError(
            f"{mutation.name}/{release.name}: failed test emitted PASS marker"
        )
    lowered = output.lower()
    for invalid in INVALID_FAILURE_FRAGMENTS:
        if invalid in lowered:
            raise GateError(
                f"{mutation.name}/{release.name}: invalid failure "
                f"fragment={invalid!r}\n{output[-16000:]}"
            )
    if mutation.diagnostic not in output:
        raise GateError(
            f"{mutation.name}/{release.name}: missing registered diagnostic "
            f"{mutation.diagnostic!r}\n{output[-16000:]}"
        )
    observed_markers = re.findall(
        r"F22_MATLAB_DEPRECATION[^\r\n]+", output
    )
    if observed_markers:
        if observed_markers != [mutation.fail_marker]:
            raise GateError(
                f"{mutation.name}/{release.name}: wrong FAIL marker "
                f"{observed_markers!r}"
            )
    elif mutation.marker_required:
        raise GateError(
            f"{mutation.name}/{release.name}: registered FAIL marker missing"
        )
    return True


def run_mutant(
    *,
    mutation: Mutation,
    repository: Path,
    build: Path,
    mex: Path,
    mex_hash: str,
    releases: tuple[MatlabRelease, MatlabRelease],
) -> list[Invocation]:
    expected_source = (repository / mutation.source).resolve()

    def one(release: MatlabRelease) -> Invocation:
        statement = (
            "fprintf('F22_MATLAB_MUTANT_RESULT "
            f"id={mutation.name} release={release.name} "
            "passed=%d failed=%d incomplete=%d\\n',"
            "passed,failed,incomplete);"
        )
        output, checks = invoke_matlab(
            repository=repository,
            build=build,
            mex=mex,
            mex_hash=mex_hash,
            release=release,
            expected_source=expected_source,
            symbol=mutation.symbol,
            label=mutation.name,
            result_statement=statement,
        )
        killed = classify_mutant(mutation, release, output)
        return Invocation(release.name, output, checks, killed)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = {release.name: pool.submit(one, release) for release in releases}
        results = [futures[release.name].result() for release in releases]
    for result in results:
        marker = RESULT_PATTERN.search(result.output)
        assert marker is not None
        print(marker.group(0))
    return results


def execute(
    *,
    repository: Path,
    build: Path,
    recovery: Path,
    originals: dict[str, bytes],
    hashes: dict[str, str],
    immutable_hashes: dict[str, str],
    releases: tuple[MatlabRelease, MatlabRelease],
) -> bool:
    mex, mex_hash = fresh_mex_build(repository, build, immutable_hashes)
    controls = 0
    mex_checks = 0
    initial = run_control(
        label="initial",
        repository=repository,
        build=build,
        mex=mex,
        mex_hash=mex_hash,
        releases=releases,
    )
    controls += len(initial)
    mex_checks += sum(item.mex_hash_checks for item in initial)
    manifest = write_recovery(
        recovery, repository, originals, hashes, mex
    )
    mutation_kills = 0
    release_kills = 0
    restored = 0
    category_kills = {category: 0 for category in CATEGORIES}
    try:
        for mutation in MUTATIONS:
            payloads = materialized(mutation, originals)
            results: list[Invocation] = []
            try:
                for relative, payload in payloads.items():
                    (repository / relative).write_bytes(payload)
                    if sha256(payload) == hashes[relative]:
                        raise GateError(
                            f"{mutation.name}: mutant hash equals clean hash"
                        )
                results = run_mutant(
                    mutation=mutation,
                    repository=repository,
                    build=build,
                    mex=mex,
                    mex_hash=mex_hash,
                    releases=releases,
                )
                kills = sum(bool(result.killed) for result in results)
                release_kills += kills
                if kills == 2:
                    mutation_kills += 1
                    category_kills[mutation.category] += 1
            finally:
                restore_sources(
                    repository, originals, hashes, payloads.keys()
                )
                restored += 1
            mex_checks += sum(item.mex_hash_checks for item in results)
            print(
                f"F22_MATLAB_MUTATION id={mutation.name} "
                f"category={mutation.category} "
                f"R2024b_killed={int(bool(results[0].killed))} "
                f"R2025b_killed={int(bool(results[1].killed))} "
                f"release_kills={sum(bool(item.killed) for item in results)}/2 "
                "restore=pass"
            )
        final = run_control(
            label="final",
            repository=repository,
            build=build,
            mex=mex,
            mex_hash=mex_hash,
            releases=releases,
        )
        controls += len(final)
        mex_checks += sum(item.mex_hash_checks for item in final)
        assert_immutable(repository, immutable_hashes)
        if sha256(mex.read_bytes()) != mex_hash:
            raise GateError("final MEX hash differs from fresh build")
        if restored != 33 or controls != 4 or mex_checks != 140:
            raise GateError(
                f"ledger mismatch controls={controls} restored={restored} "
                f"mex_hash_checks={mex_checks}"
            )
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["state"] = "complete"
        manifest.write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )
        survived = 33 - mutation_kills
        verdict = "PASS" if survived == 0 and release_kills == 66 else "FALSIFIED"
        print(
            f"F22_MATLAB_MUTATIONS controls={controls}/4 versions=2/2 "
            f"mutations={mutation_kills}/33 "
            f"release_kills={release_kills}/66 "
            f"warning_removals={category_kills['warning_removal']}/15 "
            "behavior_corruptions="
            f"{category_kills['behavior_corruption']}/15 "
            f"policy_mutations={category_kills['policy']}/3 "
            f"source_restores={restored}/33 source_files=7/7 "
            f"mex_hash_checks={mex_checks}/140 skips=0 "
            f"survived={survived} verdict={verdict}"
        )
        return verdict == "PASS"
    finally:
        restore_sources(repository, originals, hashes)


def main() -> int:
    args = parse_args()
    repository = Path(__file__).resolve().parents[1]
    build = require_inside(
        args.build_dir if args.build_dir.is_absolute()
        else repository / args.build_dir,
        repository,
        "build directory",
    )
    recovery = require_inside(
        build / "tests/f22-matlab-mutation-recovery",
        build,
        "recovery directory",
    )
    releases = (
        MatlabRelease("R2024b", args.matlab_r2024b.resolve()),
        MatlabRelease("R2025b", args.matlab_r2025b.resolve()),
    )
    try:
        if args.mode == "restore":
            restore_from_manifest(repository, recovery)
            return 0
        if not build.is_dir():
            raise GateError(f"build directory does not exist: {build}")
        for release in releases:
            if not release.executable.is_file():
                raise GateError(
                    f"{release.name} executable is missing: "
                    f"{release.executable}"
                )
        paths = {
            relative: require_inside(
                repository / relative, repository, "mutation target"
            )
            for relative in TARGETS
        }
        assert_clean_oracles(repository)
        originals = {
            relative: paths[relative].read_bytes() for relative in TARGETS
        }
        hashes = {
            relative: sha256(payload)
            for relative, payload in originals.items()
        }
        immutable_hashes = {
            relative: sha256((repository / relative).read_bytes())
            for relative in IMMUTABLE_SOURCES
        }
        assert_inventory(originals)
        mex = find_mex(build)
        print(
            "F22_MATLAB_MUTATION_PREFLIGHT inventory=33/33 "
            "warning_removals=15/15 behavior_corruptions=15/15 "
            "policy_mutations=3/3 versions=2/2 source_files=7/7 "
            f"mex={mex} verdict=PASS"
        )
        if args.mode == "preflight":
            return 0
        if not args.confirm_exclusive_build_access:
            raise GateError(
                "execute requires --confirm-exclusive-build-access"
            )
        passed = execute(
            repository=repository,
            build=build,
            recovery=recovery,
            originals=originals,
            hashes=hashes,
            immutable_hashes=immutable_hashes,
            releases=releases,
        )
        return 0 if passed else 1
    except (GateError, json.JSONDecodeError) as error:
        print(f"F22_MATLAB_MUTATION_HARNESS_ERROR {error}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
