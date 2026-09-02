#!/usr/bin/env python3
"""Permanent 31-mutant gate for the F22 Python deprecation policy.

Native mutants are accepted only when the extension recompiles successfully,
the freshly built binary is copied into the venv, a new interpreter proves the
imported hash, and the focused public test fails in its registered dimension.
Pure-Python mutants run first against the clean native extension.  Exact source
and extension recovery snapshots live under the configured build directory.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Iterable


PASS_MARKER = (
    "F22_PYTHON_GATE alias_symbols=13 operations=14 "
    "primary_warn_once=14 canonical_silent=14 equivalent=14 "
    "class_routes=3 identity=2 ordinary_legacy=0 verdict=PASS"
)
SUMMARY = re.compile(r"(?P<failed>\d+) failed, (?P<passed>\d+) passed")
PURE_CATEGORIES = {"warning_removal", "behavior_identity", "policy"}


@dataclass(frozen=True)
class Edit:
    path: str
    needle: str
    replacement: str
    expected_count: int = 1
    occurrence: int | None = None


@dataclass(frozen=True)
class Mutation:
    name: str
    category: str
    native: bool
    failed: int
    passed: int
    fragments: tuple[str, ...]
    edits: tuple[Edit, ...]


CORE = "python/src/_dtwcpp_core.cpp"
INIT = "python/dtwcpp/__init__.py"
API = "python/dtwcpp/_api.py"
TARGETS = (CORE, INIT, API)


def native_warning_removal(
    name: str,
    old_name: str,
    fragment: str,
    *,
    occurrence: int | None = None,
    expected_count: int = 1,
) -> Mutation:
    needle = f'warn_deprecated_alias("{old_name}"'
    return Mutation(
        name,
        "warning_removal",
        True,
        1,
        18,
        (fragment,),
        (
            Edit(
                CORE,
                needle,
                f'if constexpr (false) warn_deprecated_alias("{old_name}"',
                expected_count,
                occurrence,
            ),
        ),
    )


NATIVE_CASES = (
    "set_number_of_clusters",
    "n_repetition_get",
    "n_repetition_set",
    "cluster_size",
    "distance_matrix_numpy",
    "set_distance_matrix_from_numpy",
    "davies_bouldin_index",
    "dunn_index",
    "calinski_harabasz_index",
    "adjusted_rand_index",
    "normalized_mutual_information",
)


MUTATIONS = (
    # Pure-Python mutants run first so the native artifact remains clean.
    Mutation(
        "warning-remove-cluster-result",
        "warning_removal",
        False,
        4,
        15,
        ("ClusterResult", "lookup", "from_import", "star_import"),
        (Edit(INIT, "    if not importlib_preflight:", "    if False:"),),
    ),
    Mutation(
        "warning-remove-result-medoid-indices",
        "warning_removal",
        False,
        1,
        18,
        ("Result.medoid_indices",),
        (
            Edit(
                API,
                '        warnings.warn("Result.medoid_indices is deprecated; '
                'use Result.medoids",\n'
                "                      DeprecationWarning, stacklevel=2)",
                "        pass  # F22 mutant removed warning",
            ),
        ),
    ),
    Mutation(
        "behavior-cluster-result-wrong-class",
        "behavior_identity",
        False,
        4,
        15,
        ("ClusterResult", "lookup", "from_import", "star_import"),
        (Edit(INIT, "    return Result\n", "    return ClusteringResult\n"),),
    ),
    Mutation(
        "behavior-result-medoid-indices-copy",
        "behavior_identity",
        False,
        1,
        18,
        ("Result.medoid_indices",),
        (
            Edit(
                API,
                "        return self.medoids\n",
                "        return self.medoids.copy()\n",
            ),
        ),
    ),
    Mutation(
        "policy-result-medoid-attribution",
        "policy",
        False,
        1,
        18,
        ("Result.medoid_indices",),
        (Edit(API, "DeprecationWarning, stacklevel=2)",
              "DeprecationWarning, stacklevel=200)"),),
    ),
    Mutation(
        "policy-cluster-result-import-preflight",
        "policy",
        False,
        2,
        17,
        ("from_import", "star_import"),
        (Edit(INIT, "    if not importlib_preflight:", "    if True:"),),
    ),
    native_warning_removal(
        "warning-remove-set-number-of-clusters",
        "Problem.set_number_of_clusters",
        "set_number_of_clusters",
    ),
    native_warning_removal(
        "warning-remove-n-repetition-get",
        "Problem.n_repetition",
        "n_repetition_get",
        occurrence=1,
        expected_count=2,
    ),
    native_warning_removal(
        "warning-remove-n-repetition-set",
        "Problem.n_repetition",
        "n_repetition_set",
        occurrence=2,
        expected_count=2,
    ),
    native_warning_removal(
        "warning-remove-cluster-size",
        "Problem.cluster_size",
        "cluster_size",
    ),
    native_warning_removal(
        "warning-remove-distance-matrix-numpy",
        "Problem.distance_matrix_numpy",
        "distance_matrix_numpy",
    ),
    native_warning_removal(
        "warning-remove-set-distance-matrix-from-numpy",
        "Problem.set_distance_matrix_from_numpy",
        "set_distance_matrix_from_numpy",
    ),
    native_warning_removal(
        "warning-remove-davies-bouldin-index",
        "dtwcpp.davies_bouldin_index",
        "davies_bouldin_index",
    ),
    native_warning_removal(
        "warning-remove-dunn-index",
        "dtwcpp.dunn_index",
        "dunn_index",
    ),
    native_warning_removal(
        "warning-remove-calinski-harabasz-index",
        "dtwcpp.calinski_harabasz_index",
        "calinski_harabasz_index",
    ),
    native_warning_removal(
        "warning-remove-adjusted-rand-index",
        "dtwcpp.adjusted_rand_index",
        "adjusted_rand_index",
    ),
    native_warning_removal(
        "warning-remove-normalized-mutual-information",
        "dtwcpp.normalized_mutual_information",
        "normalized_mutual_information",
    ),
    Mutation(
        "behavior-set-number-of-clusters",
        "behavior_identity",
        True,
        1,
        18,
        ("set_number_of_clusters",),
        (Edit(CORE, "           p.set_n_clusters(n);",
              "           p.set_n_clusters(n + 1);"),),
    ),
    Mutation(
        "behavior-n-repetition-get",
        "behavior_identity",
        True,
        1,
        18,
        ("n_repetition_get",),
        (Edit(CORE, "                   return p.n_repetitions();",
              "                   return p.n_repetitions() + 1;"),),
    ),
    Mutation(
        "behavior-n-repetition-set",
        "behavior_identity",
        True,
        1,
        18,
        ("n_repetition_set",),
        (Edit(CORE, "                   p.set_n_repetitions(value);",
              "                   p.set_n_repetitions(value + 1);"),),
    ),
    Mutation(
        "behavior-cluster-size",
        "behavior_identity",
        True,
        1,
        18,
        ("cluster_size",),
        (Edit(CORE, "                   return p.n_clusters();",
              "                   return p.n_clusters() + 1;"),),
    ),
    Mutation(
        "behavior-distance-matrix-numpy",
        "behavior_identity",
        True,
        1,
        18,
        ("distance_matrix_numpy",),
        (
            Edit(
                CORE,
                "           return read_distance_matrix_np(p);",
                "           auto result = read_distance_matrix_np(p);\n"
                "           result.data()[0] += 1.0;\n"
                "           return result;",
            ),
        ),
    ),
    Mutation(
        "behavior-set-distance-matrix-from-numpy",
        "behavior_identity",
        True,
        1,
        18,
        ("set_distance_matrix_from_numpy",),
        (
            Edit(
                CORE,
                "           write_distance_matrix_np(p, dm);",
                "           write_distance_matrix_np(p, dm);\n"
                "           p.dense_distance_matrix().set("
                "0, 1, dm.data()[1] + 1.0);",
            ),
        ),
    ),
    Mutation(
        "behavior-davies-bouldin-index",
        "behavior_identity",
        True,
        1,
        18,
        ("davies_bouldin_index",),
        (
            Edit(
                CORE,
                "return dtwc::scores::davies_bouldin(prob);",
                "return dtwc::scores::dunn(prob);",
                2,
                2,
            ),
        ),
    ),
    Mutation(
        "behavior-dunn-index",
        "behavior_identity",
        True,
        1,
        18,
        ("dunn_index",),
        (
            Edit(
                CORE,
                "return dtwc::scores::dunn(prob);",
                "return dtwc::scores::davies_bouldin(prob);",
                2,
                2,
            ),
        ),
    ),
    Mutation(
        "behavior-calinski-harabasz-index",
        "behavior_identity",
        True,
        1,
        18,
        ("calinski_harabasz_index",),
        (
            Edit(
                CORE,
                "return dtwc::scores::calinski_harabasz(prob);",
                "return dtwc::scores::inertia(prob);",
                2,
                2,
            ),
        ),
    ),
    Mutation(
        "behavior-adjusted-rand-index",
        "behavior_identity",
        True,
        1,
        18,
        ("adjusted_rand_index",),
        (
            Edit(
                CORE,
                "return dtwc::scores::adjusted_rand(labels_true, labels_pred);",
                "return dtwc::scores::normalized_mutual_info("
                "labels_true, labels_pred);",
                2,
                2,
            ),
        ),
    ),
    Mutation(
        "behavior-normalized-mutual-information",
        "behavior_identity",
        True,
        1,
        18,
        ("normalized_mutual_information",),
        (
            Edit(
                CORE,
                "return dtwc::scores::normalized_mutual_info("
                "labels_true, labels_pred);",
                "return dtwc::scores::adjusted_rand(labels_true, labels_pred);",
                2,
                2,
            ),
        ),
    ),
    Mutation(
        "policy-native-warning-category",
        "policy",
        True,
        11,
        8,
        NATIVE_CASES,
        (Edit(CORE, "PyExc_DeprecationWarning", "PyExc_FutureWarning"),),
    ),
    Mutation(
        "policy-native-warning-duplicate",
        "policy",
        True,
        11,
        8,
        NATIVE_CASES,
        (
            Edit(
                CORE,
                "  if (PyErr_WarnEx(PyExc_DeprecationWarning, "
                "message.c_str(), 1) < 0)\n"
                "    throw nb::python_error();",
                "  if (PyErr_WarnEx(PyExc_DeprecationWarning, "
                "message.c_str(), 1) < 0)\n"
                "    throw nb::python_error();\n"
                "  if (PyErr_WarnEx(PyExc_DeprecationWarning, "
                "message.c_str(), 1) < 0)\n"
                "    throw nb::python_error();",
            ),
        ),
    ),
    Mutation(
        "policy-canonical-distance-warning",
        "policy",
        True,
        2,
        17,
        ("distance_matrix_numpy", "set_distance_matrix_from_numpy"),
        (
            Edit(
                CORE,
                "  auto read_distance_matrix_np = [](dtwc::Problem &prob) {",
                "  auto read_distance_matrix_np = [](dtwc::Problem &prob) {\n"
                "    warn_deprecated_alias("
                "\"Problem.distance_matrix\", \"Problem.distance_matrix\");",
            ),
        ),
    ),
)


class GateError(RuntimeError):
    """Harness failure; never counts as a mutation kill."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-dir", type=Path, default=Path("build/cfg-gate-normal")
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
    timeout: int = 300,
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
    if edit.occurrence is None:
        return text.replace(needle, replacement)
    if not 1 <= edit.occurrence <= observed:
        raise GateError(f"{mutation_name}: invalid occurrence")
    pieces = text.split(needle)
    index = edit.occurrence
    return needle.join(pieces[:index]) + replacement + needle.join(
        pieces[index:]
    )


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
    if len(MUTATIONS) != 31:
        raise GateError(f"inventory={len(MUTATIONS)}, expected 31")
    counts = {
        category: sum(item.category == category for item in MUTATIONS)
        for category in PURE_CATEGORIES
    }
    expected = {
        "warning_removal": 13,
        "behavior_identity": 13,
        "policy": 5,
    }
    if counts != expected:
        raise GateError(f"category drift: {counts!r}")
    if sum(item.native for item in MUTATIONS) != 25:
        raise GateError("native mutant inventory is not 25")
    if len({item.name for item in MUTATIONS}) != 31:
        raise GateError("mutation names are not unique")
    for mutation in MUTATIONS:
        materialized(mutation, originals)


def assert_clean_targets(repository: Path) -> None:
    completed = run(
        ["git", "diff", "--quiet", "HEAD", "--", *TARGETS],
        cwd=repository,
        timeout=30,
    )
    if completed.returncode != 0:
        raise GateError("mutation targets have staged or unstaged changes")


def find_artifacts(
    repository: Path, build: Path
) -> tuple[Path, Path]:
    built = sorted((build / "python").glob("_dtwcpp_core*.pyd"))
    installed_dir = repository / ".venv/Lib/site-packages/dtwcpp"
    installed = sorted(installed_dir.glob("_dtwcpp_core*.pyd"))
    if len(built) != 1 or len(installed) != 1:
        raise GateError(
            f"expected one built/installed extension, observed "
            f"{len(built)}/{len(installed)}"
        )
    return built[0].resolve(), installed[0].resolve()


def write_recovery(
    recovery: Path,
    repository: Path,
    originals: dict[str, bytes],
    hashes: dict[str, str],
    built: Path,
    installed: Path,
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
    for kind, artifact in (("built", built), ("installed", installed)):
        payload = artifact.read_bytes()
        snapshot = recovery / f"{kind}-extension.bin"
        snapshot.write_bytes(payload)
        records.append(
            {
                "kind": kind,
                "relative": os.fspath(artifact.relative_to(repository)),
                "snapshot": snapshot.name,
                "sha256": sha256(payload),
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
    if len(records) != 5:
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
        "F22_PYTHON_MUTATION_RESTORE sources=3/3 artifacts=2/2 "
        "verdict=PASS"
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


def unique_env(
    repository: Path, build: Path, label: str
) -> dict[str, str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "2"
    pycache = build / "tests/f22-python-pycache" / label
    pycache.mkdir(parents=True, exist_ok=True)
    env["PYTHONPYCACHEPREFIX"] = os.fspath(pycache)
    return env


def probe_extension(
    repository: Path,
    build: Path,
    installed: Path,
    expected_hash: str,
    label: str,
) -> None:
    script = (
        "from pathlib import Path\n"
        "import hashlib\n"
        "import dtwcpp\n"
        "import dtwcpp._dtwcpp_core as core\n"
        "package=Path(dtwcpp.__file__).resolve()\n"
        "binary=Path(core.__file__).resolve()\n"
        "digest=hashlib.sha256(binary.read_bytes()).hexdigest().upper()\n"
        "assert core._F22_DEPRECATION_POLICY is True\n"
        "print(f'F22_PYTHON_EXTENSION package={package} core={binary} "
        "sha256={digest} discriminator=1 verdict=PASS')\n"
    )
    completed = run(
        [sys.executable, "-B", "-c", script],
        cwd=repository,
        env=unique_env(repository, build, f"probe-{label}"),
        timeout=60,
    )
    if completed.returncode != 0:
        raise GateError(
            f"{label}: extension import failed\n{completed.stdout[-8000:]}"
        )
    expected_package = (repository / "python/dtwcpp/__init__.py").resolve()
    marker = (
        f"F22_PYTHON_EXTENSION package={expected_package} core={installed} "
        f"sha256={expected_hash} discriminator=1 verdict=PASS"
    )
    if completed.stdout.count(marker) != 1:
        raise GateError(
            f"{label}: extension provenance marker mismatch\n"
            f"{completed.stdout[-8000:]}"
        )


def pytest_command(repository: Path) -> list[str]:
    return [
        sys.executable,
        "-B",
        "-m",
        "pytest",
        "-q",
        "-s",
        os.fspath(repository / "tests/python/test_deprecation_policy.py"),
    ]


def run_control(
    label: str,
    repository: Path,
    build: Path,
    cmake: str,
    built: Path,
    installed: Path,
) -> str:
    compiled = run(
        [
            cmake,
            "--build",
            os.fspath(build),
            "--config",
            "Release",
            "--target",
            "_dtwcpp_core",
        ],
        cwd=repository,
    )
    if compiled.returncode != 0:
        raise GateError(
            f"{label}: clean extension build failed\n{compiled.stdout[-12000:]}"
        )
    shutil.copy2(built, installed)
    built_hash = sha256(built.read_bytes())
    installed_hash = sha256(installed.read_bytes())
    if installed_hash != built_hash:
        raise GateError(f"{label}: built/installed extension hash mismatch")
    probe_extension(
        repository, build, installed, built_hash, f"control-{label}"
    )
    tested = run(
        pytest_command(repository),
        cwd=repository,
        env=unique_env(repository, build, f"control-{label}"),
        timeout=120,
    )
    if tested.returncode != 0:
        raise GateError(
            f"{label}: focused control failed\n{tested.stdout[-16000:]}"
        )
    if tested.stdout.count(PASS_MARKER) != 1:
        raise GateError(f"{label}: PASS marker count drift")
    if re.search(r"\b19 passed\b", tested.stdout) is None:
        raise GateError(f"{label}: focused 19-pass ledger missing")
    print(
        f"F22_PYTHON_MUTATION_CONTROL label={label} tests=19/19 "
        f"sha256={built_hash} verdict=PASS"
    )
    return built_hash


def assert_mutant_failure(
    mutation: Mutation, completed: subprocess.CompletedProcess[str]
) -> None:
    if completed.returncode != 1:
        if completed.returncode == 0:
            raise GateError(f"{mutation.name}: SURVIVED focused gate")
        raise GateError(
            f"{mutation.name}: pytest exited {completed.returncode}\n"
            f"{completed.stdout[-16000:]}"
        )
    if PASS_MARKER in completed.stdout:
        raise GateError(f"{mutation.name}: emitted a false PASS marker")
    matches = list(SUMMARY.finditer(completed.stdout))
    if len(matches) != 1:
        raise GateError(
            f"{mutation.name}: pytest summary count={len(matches)}\n"
            f"{completed.stdout[-16000:]}"
        )
    match = matches[0]
    observed = (int(match.group("failed")), int(match.group("passed")))
    expected = (mutation.failed, mutation.passed)
    if observed != expected:
        raise GateError(
            f"{mutation.name}: expected failed/passed={expected}, "
            f"observed={observed}\n{completed.stdout[-16000:]}"
        )
    summary_tail = completed.stdout[match.start():]
    if re.search(r"\b\d+ (?:errors?|skipped|xfailed|xpassed)\b", summary_tail):
        raise GateError(f"{mutation.name}: non-failure pytest outcome present")
    for fragment in mutation.fragments:
        if fragment not in completed.stdout:
            raise GateError(
                f"{mutation.name}: missing failure fragment {fragment!r}"
            )


def run_mutant(
    mutation: Mutation,
    repository: Path,
    build: Path,
    cmake: str,
    built: Path,
    installed: Path,
    clean_hash: str,
) -> None:
    if mutation.native:
        before_mtime = built.stat().st_mtime_ns
        before_hash = sha256(built.read_bytes())
        compiled = run(
            [
                cmake,
                "--build",
                os.fspath(build),
                "--config",
                "Release",
                "--target",
                "_dtwcpp_core",
            ],
            cwd=repository,
        )
        if compiled.returncode != 0:
            raise GateError(
                f"{mutation.name}: native mutant did not compile\n"
                f"{compiled.stdout[-16000:]}"
            )
        if "_dtwcpp_core.cpp" not in compiled.stdout:
            raise GateError(
                f"{mutation.name}: build did not compile the mutated source\n"
                f"{compiled.stdout[-8000:]}"
            )
        mutant_hash = sha256(built.read_bytes())
        if (
            built.stat().st_mtime_ns == before_mtime
            or mutant_hash == before_hash
            or mutant_hash == clean_hash
        ):
            raise GateError(f"{mutation.name}: stale native artifact")
        shutil.copy2(built, installed)
        if sha256(installed.read_bytes()) != mutant_hash:
            raise GateError(f"{mutation.name}: installed mutant hash mismatch")
        probe_extension(
            repository, build, installed, mutant_hash, mutation.name
        )
    completed = run(
        pytest_command(repository),
        cwd=repository,
        env=unique_env(repository, build, mutation.name),
        timeout=120,
    )
    assert_mutant_failure(mutation, completed)


def execute(
    repository: Path,
    build: Path,
    recovery: Path,
    originals: dict[str, bytes],
    hashes: dict[str, str],
) -> None:
    cmake = shutil.which("cmake")
    if cmake is None:
        raise GateError("cmake is unavailable on PATH")
    built, installed = find_artifacts(repository, build)
    clean_hash = run_control(
        "initial", repository, build, cmake, built, installed
    )
    manifest = write_recovery(
        recovery, repository, originals, hashes, built, installed
    )
    killed = 0
    restored = 0
    category_kills = {
        "warning_removal": 0,
        "behavior_identity": 0,
        "policy": 0,
    }
    native_kills = 0
    pure_kills = 0
    try:
        for mutation in MUTATIONS:
            payloads = materialized(mutation, originals)
            try:
                for relative, payload in payloads.items():
                    (repository / relative).write_bytes(payload)
                    if sha256(payload) == hashes[relative]:
                        raise GateError(
                            f"{mutation.name}: mutant hash equals clean hash"
                        )
                run_mutant(
                    mutation,
                    repository,
                    build,
                    cmake,
                    built,
                    installed,
                    clean_hash,
                )
                killed += 1
                category_kills[mutation.category] += 1
                native_kills += int(mutation.native)
                pure_kills += int(not mutation.native)
            finally:
                restore_sources(
                    repository, originals, hashes, payloads.keys()
                )
                restored += 1
            print(
                f"F22_PYTHON_MUTATION name={mutation.name} "
                f"category={mutation.category} "
                f"kind={'native' if mutation.native else 'pure'} "
                f"failed={mutation.failed} passed={mutation.passed} "
                "killed=1 restore=pass"
            )
        final_hash = run_control(
            "final", repository, build, cmake, built, installed
        )
        print(
            "F22_PYTHON_NATIVE_RESTORE clean_rebuild=1 "
            f"initial_sha256={clean_hash} final_sha256={final_hash} "
            "built_imported_match=1 verdict=PASS"
        )
        if killed != 31 or restored != 31:
            raise GateError(
                f"ledger mismatch killed={killed} restored={restored}"
            )
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["state"] = "complete"
        manifest.write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "F22_PYTHON_MUTATIONS controls=2/2 inventory=31/31 "
            f"warning_removals={category_kills['warning_removal']}/13 "
            f"behavior_identity={category_kills['behavior_identity']}/13 "
            f"policy={category_kills['policy']}/5 "
            f"native_kills={native_kills}/25 pure_kills={pure_kills}/6 "
            "killed=31/31 survived=0 source_restores=31/31 "
            "source_files=3/3 native_restore=pass skips=0 verdict=PASS"
        )
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
        build / "tests/f22-python-mutation-recovery",
        build,
        "recovery directory",
    )
    try:
        if args.mode == "restore":
            restore_from_manifest(repository, recovery)
            return 0
        if not build.is_dir():
            raise GateError(f"build directory does not exist: {build}")
        paths = {
            relative: require_inside(
                repository / relative, repository, "mutation target"
            )
            for relative in TARGETS
        }
        assert_clean_targets(repository)
        originals = {
            relative: paths[relative].read_bytes() for relative in TARGETS
        }
        hashes = {
            relative: sha256(payload)
            for relative, payload in originals.items()
        }
        assert_inventory(originals)
        built, installed = find_artifacts(repository, build)
        print(
            "F22_PYTHON_MUTATION_PREFLIGHT inventory=31/31 "
            "warning_removals=13/13 behavior_identity=13/13 policy=5/5 "
            "native=25/25 pure=6/6 source_files=3/3 "
            f"built={built} installed={installed} verdict=PASS"
        )
        if args.mode == "preflight":
            return 0
        if not args.confirm_exclusive_build_access:
            raise GateError(
                "execute requires --confirm-exclusive-build-access"
            )
        execute(repository, build, recovery, originals, hashes)
        return 0
    except (GateError, json.JSONDecodeError) as error:
        print(f"F22_PYTHON_MUTATION_HARNESS_ERROR {error}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
