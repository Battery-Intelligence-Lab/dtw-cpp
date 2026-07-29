#!/usr/bin/env python3
"""Adversarial mutation gate for the frozen F22 C++ compatibility policy.

The registered ledger is 46 mutants:

* 33 per-entity deprecation-attribute removals;
* seven legacy I/O forwarding corruptions;
* four deprecated-field read/write cross-wires;
* one canonical-name warning leak; and
* one wrong-message-family corruption.

Every mutation is materialised from an exact byte snapshot, restored in a
``finally`` block, and followed by SHA-256 verification.  Recovery snapshots
remain under the selected build directory so an externally killed process can
be repaired with ``--mode restore``.
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


DIAGNOSTIC_PASS = (
    "F22_CPP_DIAGNOSTICS inventory=33/33 legacy=33/33 "
    "canonical_silent=33/33 overloads=31/31 fields=2/2 "
    "skips=0 verdict=PASS"
)
BEHAVIOR_PASS = (
    "F22_CPP_COMPAT inventory=33/33 behavior=33/33 "
    "field_routes=4/4 io_routes=7/7 file_identity=6/6 "
    "stdout_identity=2/2 skips=0 verdict=PASS"
)
CATCH_PASS = re.compile(
    r"All tests passed \((?P<assertions>\d+) assertions in "
    r"(?P<cases>\d+) test cases\)"
)
DIAGNOSTIC_FAIL = re.compile(
    r"^F22_CPP_DIAGNOSTICS .* skips=0 verdict=FAIL$", re.MULTILINE
)
BEHAVIOR_FAIL = re.compile(
    r"^F22_CPP_COMPAT .* skips=0 verdict=FAIL$", re.MULTILINE
)


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
    gate: str
    edits: tuple[Edit, ...]


TARGETS = (
    "dtwc/Problem.hpp",
    "dtwc/Problem.cpp",
    "dtwc/scores.hpp",
    "dtwc/DataLoader.hpp",
    "dtwc/settings.hpp",
)


def removal(
    name: str,
    path: str,
    message: str,
    *,
    occurrence: int | None = None,
    expected_count: int = 1,
) -> Mutation:
    return Mutation(
        name=name,
        category="diagnostic_removal",
        gate="diagnostic",
        edits=(
            Edit(
                path,
                f'[[deprecated("{message}")]]',
                "/* F22 mutant removed deprecation */",
                expected_count,
                occurrence,
            ),
        ),
    )


MUTATIONS = (
    removal("diag-max-iter-field", "dtwc/Problem.hpp",
            "use set_max_iter/max_iter"),
    removal("diag-n-repetition-field", "dtwc/Problem.hpp",
            "use set_n_repetitions/n_repetitions"),
    removal("diag-cluster-size", "dtwc/Problem.hpp", "use n_clusters"),
    removal("diag-refresh-distance-matrix", "dtwc/Problem.hpp",
            "use refresh_distance_matrix"),
    removal("diag-read-distance-matrix", "dtwc/Problem.hpp",
            "use read_distance_matrix"),
    removal("diag-set-number-of-clusters", "dtwc/Problem.hpp",
            "use set_n_clusters"),
    removal("diag-max-distance", "dtwc/Problem.hpp", "use max_distance"),
    removal("diag-dist-by-ind", "dtwc/Problem.hpp", "use dist_by_ind"),
    removal("diag-is-distance-matrix-filled", "dtwc/Problem.hpp",
            "use is_distance_matrix_filled"),
    removal("diag-fill-distance-matrix", "dtwc/Problem.hpp",
            "use fill_distance_matrix"),
    removal("diag-print-distance-matrix", "dtwc/Problem.hpp",
            "use print_distance_matrix"),
    removal("diag-write-distance-matrix-named", "dtwc/Problem.hpp",
            "use write_distance_matrix", occurrence=1, expected_count=2),
    removal("diag-write-distance-matrix-default", "dtwc/Problem.hpp",
            "use write_distance_matrix", occurrence=2, expected_count=2),
    removal("diag-print-clusters", "dtwc/Problem.hpp", "use print_clusters"),
    removal("diag-write-clusters", "dtwc/Problem.hpp", "use write_clusters"),
    removal("diag-write-medoid-members", "dtwc/Problem.hpp",
            "use write_medoid_members"),
    removal("diag-write-silhouettes", "dtwc/Problem.hpp",
            "use write_silhouettes"),
    removal("diag-cluster-by-mip", "dtwc/Problem.hpp",
            "use cluster_by_mip"),
    removal("diag-cluster-by-kmedoids-lloyd", "dtwc/Problem.hpp",
            "use cluster_by_kmedoids_lloyd"),
    removal("diag-find-total-cost", "dtwc/Problem.hpp",
            "use find_total_cost"),
    removal("diag-assign-clusters", "dtwc/Problem.hpp",
            "use assign_clusters"),
    removal("diag-calculate-medoids", "dtwc/Problem.hpp",
            "use calculate_medoids"),
    removal("diag-davies-bouldin", "dtwc/scores.hpp",
            "use scores::davies_bouldin"),
    removal("diag-dunn", "dtwc/scores.hpp", "use scores::dunn"),
    removal("diag-calinski-harabasz", "dtwc/scores.hpp",
            "use scores::calinski_harabasz"),
    removal("diag-adjusted-rand", "dtwc/scores.hpp",
            "use scores::adjusted_rand"),
    removal("diag-normalized-mutual-info", "dtwc/scores.hpp",
            "use scores::normalized_mutual_info"),
    removal("diag-loader-start-column", "dtwc/DataLoader.hpp",
            "use start_column"),
    removal("diag-loader-start-row", "dtwc/DataLoader.hpp",
            "use start_row"),
    removal("diag-set-data-path-fs", "dtwc/settings.hpp",
            "use set_data_path", occurrence=1, expected_count=2),
    removal("diag-set-data-path-cstring", "dtwc/settings.hpp",
            "use set_data_path", occurrence=2, expected_count=2),
    removal("diag-set-results-path-fs", "dtwc/settings.hpp",
            "use set_results_path", occurrence=1, expected_count=2),
    removal("diag-set-results-path-cstring", "dtwc/settings.hpp",
            "use set_results_path", occurrence=2, expected_count=2),
    Mutation(
        "io-read-forwarding",
        "io_forwarding",
        "behavior",
        (
            Edit(
                "dtwc/Problem.hpp",
                "void readDistanceMatrix(const fs::path &p) "
                "{ read_distance_matrix(p); }",
                "void readDistanceMatrix(const fs::path &p) "
                "{ read_distance_matrix(p); refresh_distance_matrix(); }",
            ),
        ),
    ),
    Mutation(
        "io-write-distance-named-forwarding",
        "io_forwarding",
        "behavior",
        (
            Edit(
                "dtwc/Problem.hpp",
                "    write_distance_matrix(name_);\n"
                "  }\n"
                "  [[deprecated(\"use write_distance_matrix\")]]\n"
                "  void writeDistanceMatrix() const",
                "    write_distance_matrix(name_ + \".mutant\");\n"
                "  }\n"
                "  [[deprecated(\"use write_distance_matrix\")]]\n"
                "  void writeDistanceMatrix() const",
            ),
        ),
    ),
    Mutation(
        "io-write-distance-default-forwarding",
        "io_forwarding",
        "behavior",
        (
            Edit(
                "dtwc/Problem.hpp",
                "void writeDistanceMatrix() const { write_distance_matrix(); }",
                "void writeDistanceMatrix() const "
                "{ write_distance_matrix(\"f22-mutant.csv\"); }",
            ),
        ),
    ),
    Mutation(
        "io-print-clusters-forwarding",
        "io_forwarding",
        "behavior",
        (
            Edit(
                "dtwc/Problem.hpp",
                "void printClusters() const { print_clusters(); }",
                "void printClusters() const { print_distance_matrix(); }",
            ),
        ),
    ),
    Mutation(
        "io-write-clusters-forwarding",
        "io_forwarding",
        "behavior",
        (
            Edit(
                "dtwc/Problem.hpp",
                "void writeClusters() { write_clusters(); }",
                "void writeClusters() { write_silhouettes(); }",
            ),
        ),
    ),
    Mutation(
        "io-write-medoid-members-forwarding",
        "io_forwarding",
        "behavior",
        (
            Edit(
                "dtwc/Problem.hpp",
                "    write_medoid_members(iter, rep);",
                "    write_medoid_members(iter + 1, rep);",
            ),
        ),
    ),
    Mutation(
        "io-write-silhouettes-forwarding",
        "io_forwarding",
        "behavior",
        (
            Edit(
                "dtwc/Problem.hpp",
                "void writeSilhouettes() { write_silhouettes(); }",
                "void writeSilhouettes() { write_clusters(); }",
            ),
        ),
    ),
    Mutation(
        "field-max-iter-write-crosswire",
        "field_crosswire",
        "behavior",
        (Edit("dtwc/Problem.cpp", "  maxIter = n;", "  N_repetition = n;"),),
    ),
    Mutation(
        "field-max-iter-read-crosswire",
        "field_crosswire",
        "behavior",
        (Edit("dtwc/Problem.cpp", "  return maxIter;", "  return N_repetition;"),),
    ),
    Mutation(
        "field-n-repetition-write-crosswire",
        "field_crosswire",
        "behavior",
        (Edit("dtwc/Problem.cpp", "  N_repetition = n;", "  maxIter = n;"),),
    ),
    Mutation(
        "field-n-repetition-read-crosswire",
        "field_crosswire",
        "behavior",
        (Edit("dtwc/Problem.cpp", "  return N_repetition;", "  return maxIter;"),),
    ),
    Mutation(
        "canonical-warning-leak",
        "canonical_warning",
        "diagnostic",
        (
            Edit(
                "dtwc/Problem.hpp",
                "  void fill_distance_matrix();\n"
                "  [[deprecated(\"use fill_distance_matrix\")]]",
                "  [[deprecated(\"F22 mutant canonical warning leak\")]]\n"
                "  void fill_distance_matrix();\n"
                "  [[deprecated(\"use fill_distance_matrix\")]]",
            ),
        ),
    ),
    Mutation(
        "wrong-message-family",
        "wrong_message",
        "diagnostic",
        (
            Edit("dtwc/Problem.hpp", '[[deprecated("use ',
                 '[[deprecated("prefer ', 22),
            Edit("dtwc/scores.hpp", '[[deprecated("use ',
                 '[[deprecated("prefer ', 5),
            Edit("dtwc/DataLoader.hpp", '[[deprecated("use ',
                 '[[deprecated("prefer ', 2),
            Edit("dtwc/settings.hpp", '[[deprecated("use ',
                 '[[deprecated("prefer ', 4),
        ),
    ),
)


class GateError(RuntimeError):
    """A harness/precondition failure, never a mutation kill."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--build-dir", type=Path, default=Path("build/highs-1151")
    )
    parser.add_argument(
        "--mode", choices=("preflight", "execute", "restore"),
        default="preflight",
    )
    parser.add_argument(
        "--confirm-exclusive-build-access", action="store_true",
        help="required for execute because configured builds are not isolated",
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
        raise GateError(f"{description} escapes repository/build root: {resolved}")
    return resolved


def run(
    command: list[str],
    *,
    cwd: Path,
    timeout: int = 300,
) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            command,
            cwd=cwd,
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
        raise GateError(
            f"{mutation_name}: invalid occurrence {edit.occurrence}/{observed}"
        )
    pieces = text.split(needle)
    index = edit.occurrence
    return needle.join(pieces[:index]) + replacement + needle.join(
        pieces[index:]
    )


def materialized_payloads(
    mutation: Mutation,
    originals: dict[str, bytes],
) -> dict[str, bytes]:
    changed: dict[str, bytes] = {}
    by_path: dict[str, list[Edit]] = {}
    for edit in mutation.edits:
        by_path.setdefault(edit.path, []).append(edit)
    for relative, edits in by_path.items():
        try:
            text = originals[relative].decode("utf-8", errors="strict")
        except UnicodeDecodeError as error:
            raise GateError(f"{relative} is not strict UTF-8") from error
        for edit in edits:
            text = replace_edit(text, edit, mutation.name)
        payload = text.encode("utf-8")
        if payload == originals[relative]:
            raise GateError(f"{mutation.name}: mutation made no byte change")
        changed[relative] = payload
    return changed


def assert_inventory(originals: dict[str, bytes]) -> None:
    if len(MUTATIONS) != 46:
        raise GateError(f"registered inventory is {len(MUTATIONS)}, expected 46")
    counts = {
        category: sum(m.category == category for m in MUTATIONS)
        for category in {
            "diagnostic_removal",
            "io_forwarding",
            "field_crosswire",
            "canonical_warning",
            "wrong_message",
        }
    }
    expected = {
        "diagnostic_removal": 33,
        "io_forwarding": 7,
        "field_crosswire": 4,
        "canonical_warning": 1,
        "wrong_message": 1,
    }
    if counts != expected:
        raise GateError(f"registered category drift: {counts!r}")
    names = [mutation.name for mutation in MUTATIONS]
    if len(set(names)) != len(names):
        raise GateError("mutation names are not unique")
    for mutation in MUTATIONS:
        materialized_payloads(mutation, originals)


def assert_clean_targets(repository: Path) -> None:
    result = run(
        ["git", "diff", "--quiet", "HEAD", "--", *TARGETS],
        cwd=repository,
        timeout=30,
    )
    if result.returncode != 0:
        raise GateError("mutation targets have staged or unstaged changes")


def write_recovery(
    recovery: Path,
    repository: Path,
    originals: dict[str, bytes],
    original_hashes: dict[str, str],
) -> Path:
    recovery.mkdir(parents=True, exist_ok=True)
    records = []
    for index, relative in enumerate(TARGETS):
        snapshot = recovery / f"{index:02d}-{Path(relative).name}.bin"
        snapshot.write_bytes(originals[relative])
        if sha256(snapshot.read_bytes()) != original_hashes[relative]:
            raise GateError(f"recovery snapshot hash mismatch: {snapshot}")
        records.append(
            {
                "relative": relative,
                "sha256": original_hashes[relative],
                "snapshot": snapshot.name,
            }
        )
    manifest = recovery / "manifest.json"
    manifest.write_text(
        json.dumps({"state": "active", "files": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def restore_from_manifest(repository: Path, recovery: Path) -> None:
    manifest = recovery / "manifest.json"
    if not manifest.is_file():
        raise GateError(f"recovery manifest is missing: {manifest}")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    records = data.get("files")
    if not isinstance(records, list):
        raise GateError("recovery manifest has no file records")
    expected_relatives = set(TARGETS)
    observed_relatives = {
        record.get("relative") for record in records if isinstance(record, dict)
    }
    if observed_relatives != expected_relatives:
        raise GateError("recovery manifest target allowlist mismatch")
    for record in records:
        relative = record["relative"]
        target = require_inside(repository / relative, repository, "restore target")
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
        "F22_CPP_MUTATION_RESTORE files=5/5 source_restore=pass verdict=PASS"
    )


def restore_payloads(
    repository: Path,
    originals: dict[str, bytes],
    original_hashes: dict[str, str],
    relatives: Iterable[str] = TARGETS,
) -> None:
    for relative in relatives:
        (repository / relative).write_bytes(originals[relative])
    for relative in TARGETS:
        observed = sha256((repository / relative).read_bytes())
        if observed != original_hashes[relative]:
            raise GateError(
                f"source restoration failed for {relative}: "
                f"{observed} != {original_hashes[relative]}"
            )


def diagnostic_command(repository: Path, build: Path) -> list[str]:
    return [
        sys.executable,
        os.fspath(repository / "scripts/test_f22_cpp_deprecations.py"),
        "--build-dir",
        os.fspath(build),
        "--source-dir",
        os.fspath(repository),
    ]


def assert_control(
    label: str,
    repository: Path,
    build: Path,
    cmake: str,
    ctest: str,
) -> None:
    built = run(
        [cmake, "--build", os.fspath(build), "--config", "Release",
         "--target", "test_problem_api_2_0"],
        cwd=repository,
    )
    if built.returncode != 0:
        raise GateError(
            f"{label} control build exited {built.returncode}\n{built.stdout[-8000:]}"
        )
    tested = run(
        [
            ctest,
            "--test-dir",
            os.fspath(build),
            "-C",
            "Release",
            "-R",
            "^test_problem_api_2_0$",
            "--output-on-failure",
            "-V",
        ],
        cwd=repository,
    )
    if tested.returncode != 0:
        raise GateError(
            f"{label} control test exited {tested.returncode}\n"
            f"{tested.stdout[-12000:]}"
        )
    if tested.stdout.count(DIAGNOSTIC_PASS) != 1:
        raise GateError(f"{label} control diagnostic marker count drift")
    if tested.stdout.count(BEHAVIOR_PASS) != 1:
        raise GateError(f"{label} control behavior marker count drift")
    catch = CATCH_PASS.search(tested.stdout)
    if catch is None:
        raise GateError(f"{label} control omitted Catch2 execution ledger")
    assertions = int(catch.group("assertions"))
    cases = int(catch.group("cases"))
    if assertions < 229 or cases < 5:
        raise GateError(
            f"{label} control floor failed: {assertions} assertions/{cases} cases"
        )
    print(
        f"F22_CPP_MUTATION_CONTROL label={label} diagnostic=33/33 "
        f"behavior=33/33 assertions={assertions} cases={cases} verdict=PASS"
    )


def kill_diagnostic(
    mutation: Mutation,
    repository: Path,
    build: Path,
) -> None:
    result = run(diagnostic_command(repository, build), cwd=repository)
    if result.returncode != 1:
        raise GateError(
            f"{mutation.name}: diagnostic mutant exited {result.returncode}\n"
            f"{result.stdout[-12000:]}"
        )
    if DIAGNOSTIC_FAIL.search(result.stdout) is None:
        raise GateError(
            f"{mutation.name}: missing diagnostic FAIL marker\n"
            f"{result.stdout[-12000:]}"
        )
    if DIAGNOSTIC_PASS in result.stdout or "F22_CPP_HARNESS_ERROR" in result.stdout:
        raise GateError(f"{mutation.name}: invalid diagnostic kill")


def kill_behavior(
    mutation: Mutation,
    repository: Path,
    build: Path,
    cmake: str,
    ctest: str,
) -> None:
    built = run(
        [cmake, "--build", os.fspath(build), "--config", "Release",
         "--target", "test_problem_api_2_0"],
        cwd=repository,
    )
    if built.returncode != 0:
        raise GateError(
            f"{mutation.name}: mutant did not compile\n{built.stdout[-12000:]}"
        )
    tested = run(
        [
            ctest,
            "--test-dir",
            os.fspath(build),
            "-C",
            "Release",
            "-R",
            "^test_problem_api_2_0$",
            "--output-on-failure",
            "-V",
        ],
        cwd=repository,
    )
    if tested.returncode == 0:
        raise GateError(f"{mutation.name}: SURVIVED behavior gate")
    if tested.stdout.count(DIAGNOSTIC_PASS) != 1:
        raise GateError(
            f"{mutation.name}: diagnostic prerequisite did not execute/pass\n"
            f"{tested.stdout[-12000:]}"
        )
    if BEHAVIOR_FAIL.search(tested.stdout) is None:
        raise GateError(
            f"{mutation.name}: missing behavior FAIL marker\n"
            f"{tested.stdout[-12000:]}"
        )
    if BEHAVIOR_PASS in tested.stdout:
        raise GateError(f"{mutation.name}: emitted a behavior PASS marker")


def execute(
    repository: Path,
    build: Path,
    recovery: Path,
    originals: dict[str, bytes],
    original_hashes: dict[str, str],
) -> None:
    cmake = shutil.which("cmake")
    ctest = shutil.which("ctest")
    if cmake is None or ctest is None:
        raise GateError("cmake and ctest must be available on PATH")
    manifest = write_recovery(
        recovery, repository, originals, original_hashes
    )
    killed = 0
    restored = 0
    category_kills = {
        "diagnostic_removal": 0,
        "io_forwarding": 0,
        "field_crosswire": 0,
        "canonical_warning": 0,
        "wrong_message": 0,
    }
    try:
        assert_control("initial", repository, build, cmake, ctest)
        for mutation in MUTATIONS:
            payloads = materialized_payloads(mutation, originals)
            try:
                for relative, payload in payloads.items():
                    (repository / relative).write_bytes(payload)
                    if sha256((repository / relative).read_bytes()) == (
                        original_hashes[relative]
                    ):
                        raise GateError(
                            f"{mutation.name}: materialized hash did not change"
                        )
                if mutation.gate == "diagnostic":
                    kill_diagnostic(mutation, repository, build)
                else:
                    kill_behavior(
                        mutation, repository, build, cmake, ctest
                    )
                killed += 1
                category_kills[mutation.category] += 1
            finally:
                restore_payloads(
                    repository,
                    originals,
                    original_hashes,
                    payloads.keys(),
                )
                restored += 1
            print(
                f"F22_CPP_MUTATION name={mutation.name} "
                f"category={mutation.category} gate={mutation.gate} "
                "killed=1 restore=pass"
            )
        assert_control("final", repository, build, cmake, ctest)
        if killed != 46 or restored != 46:
            raise GateError(
                f"registered ledger mismatch: killed={killed} restored={restored}"
            )
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["state"] = "complete"
        manifest.write_text(
            json.dumps(data, indent=2) + "\n", encoding="utf-8"
        )
        print(
            "F22_CPP_MUTATIONS controls=2/2 inventory=46/46 "
            f"diagnostic_removals={category_kills['diagnostic_removal']}/33 "
            f"io_forwarding={category_kills['io_forwarding']}/7 "
            f"field_crosswires={category_kills['field_crosswire']}/4 "
            f"canonical_warning={category_kills['canonical_warning']}/1 "
            f"wrong_message={category_kills['wrong_message']}/1 "
            "killed=46/46 survived=0 source_restores=46/46 "
            "source_files=5/5 skips=0 verdict=PASS"
        )
    finally:
        restore_payloads(repository, originals, original_hashes)


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
        build / "tests/f22-cpp-mutation-recovery",
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
        for relative, path in paths.items():
            if not path.is_file():
                raise GateError(f"mutation target is missing: {relative}")
        assert_clean_targets(repository)
        originals = {
            relative: paths[relative].read_bytes() for relative in TARGETS
        }
        original_hashes = {
            relative: sha256(payload)
            for relative, payload in originals.items()
        }
        assert_inventory(originals)
        print(
            "F22_CPP_MUTATION_PREFLIGHT inventory=46/46 "
            "diagnostic_removals=33/33 io_forwarding=7/7 "
            "field_crosswires=4/4 canonical_warning=1/1 "
            "wrong_message=1/1 source_files=5/5 verdict=PASS"
        )
        if args.mode == "preflight":
            return 0
        if not args.confirm_exclusive_build_access:
            raise GateError(
                "execute requires --confirm-exclusive-build-access"
            )
        execute(
            repository, build, recovery, originals, original_hashes
        )
        return 0
    except (GateError, json.JSONDecodeError) as error:
        print(f"F22_CPP_MUTATION_HARNESS_ERROR {error}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
