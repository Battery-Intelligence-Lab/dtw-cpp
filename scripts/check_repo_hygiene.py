#!/usr/bin/env python3
"""Enforce the permanent R1 repository-hygiene invariants."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BANNED_TRACKED_PATHS = (
    "benchmarks/baselines/.gitkeep",
    "benchmarks/results/_autorun/bench_dtw_baseline_20260412_221108.json",
    "docs/docs_logo.png",
    "media/cluster_matrix_formation4.svg",
    "media/Merged_document.png",
)

INTENTIONAL_ZERO_BYTE_PATHS = {
    "python/dtwcpp/py.typed",
}

TARGETED_DUPLICATE_PAIRS = (
    ("docs/docs_logo.png", "docs/static/docs_logo.png"),
    (
        "media/cluster_matrix_formation4.svg",
        "docs/static/method/cluster_matrix_formation4.svg",
    ),
    ("media/Merged_document.png", "docs/static/method/dtw_image.png"),
)

REQUIRED_IGNORE_TARGETS = (
    "/tools/emsdk/",
    "/tools/node/",
    "/web/pkg/",
    "node_modules/",
    "/web/.vite/",
    "/web/coverage/",
    "/web/playwright-report/",
    "/web/test-results/",
    "*.pyd",
    "*.mexw64",
    "*.mexa64",
    "*.mexmaci64",
    "*.mexmaca64",
    "/.pytest_cache/",
    "/.ruff_cache/",
    "/.mypy_cache/",
    "/htmlcov/",
    "/.coverage",
    "/coverage.xml",
    "/CMakeUserPresets.json",
    "/.claude/scheduled_tasks.lock",
    ".env.*",
    "!.env.example",
)

REQUIRED_BENCHMARK_IGNORE_ROUTES = (
    "/benchmarks/results/_autorun/*.json",
    "!/benchmarks/results/_autorun/bench_dtw_baseline_20260412_221116.json",
    "!/benchmarks/results/_autorun/bench_metal_dtw_20260412_212302.json",
)

HIGH_CONFIDENCE_SECRET_PATTERNS = (
    re.compile(
        rb"-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----"
    ),
    re.compile(rb"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{36,255}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{50,255}\b"),
    re.compile(rb"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(rb"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bnpm_[A-Za-z0-9]{36}\b"),
    re.compile(rb"\bpypi-[A-Za-z0-9_-]{50,}\b"),
    re.compile(rb"\bAIza[0-9A-Za-z_-]{35}\b"),
)


def git_output(arguments: list[str], *, input_data: bytes | None = None) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        input=input_data,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout


def index_entries() -> dict[str, tuple[str, str]]:
    entries: dict[str, tuple[str, str]] = {}
    for record in git_output(["ls-files", "-s", "-z"]).split(b"\0"):
        if not record:
            continue
        metadata, path_bytes = record.split(b"\t", 1)
        mode_bytes, object_id_bytes, stage_bytes = metadata.split()
        relative = path_bytes.decode("utf-8")
        if stage_bytes != b"0":
            raise AssertionError(f"unmerged index entry remains: {relative}")
        entries[relative] = (mode_bytes.decode(), object_id_bytes.decode())
    return entries


def object_metadata(
    entries: dict[str, tuple[str, str]],
) -> dict[str, tuple[str, int]]:
    object_ids = tuple(dict.fromkeys(object_id for _, object_id in entries.values()))
    output = git_output(
        ["cat-file", "--batch-check=%(objectname) %(objecttype) %(objectsize)"],
        input_data=("\n".join(object_ids) + "\n").encode(),
    )
    metadata: dict[str, tuple[str, int]] = {}
    for line in output.decode().splitlines():
        object_id, object_type, size = line.split()
        metadata[object_id] = (object_type, int(size))
    return metadata


def read_index_blob(
    entries: dict[str, tuple[str, str]], relative: str
) -> bytes:
    try:
        _, object_id = entries[relative]
    except KeyError as error:
        raise AssertionError(f"required tracked path is absent: {relative}") from error
    return git_output(["cat-file", "blob", object_id])


def read_index_text(
    entries: dict[str, tuple[str, str]], relative: str
) -> str:
    return read_index_blob(entries, relative).decode("utf-8")


def is_ignored(relative: str) -> bool:
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-q", "--", relative],
        cwd=ROOT,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise AssertionError(
            f"git check-ignore failed for {relative}: exit {result.returncode}"
        )
    return result.returncode == 0


def worktree_matches_index(relative: str) -> bool:
    result = subprocess.run(
        ["git", "diff", "--quiet", "--", relative],
        cwd=ROOT,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise AssertionError(
            f"git diff failed for {relative}: exit {result.returncode}"
        )
    return result.returncode == 0


def scan_high_confidence_secrets(
    entries: dict[str, tuple[str, str]],
    metadata: dict[str, tuple[str, int]],
) -> list[tuple[str, int]]:
    hits: list[tuple[str, int]] = []
    paths_by_object: dict[str, list[str]] = {}
    for relative, (_, object_id) in entries.items():
        object_type, _ = metadata[object_id]
        if object_type != "blob":
            continue
        paths_by_object.setdefault(object_id, []).append(relative)

    process = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    assert process.stdin is not None and process.stdout is not None
    for object_id, object_paths in paths_by_object.items():
        process.stdin.write(object_id.encode() + b"\n")
        process.stdin.flush()
        header = process.stdout.readline().rstrip(b"\n")
        header_id, object_type, size_bytes = header.split()
        size = int(size_bytes)
        if header_id.decode() != object_id or object_type != b"blob":
            raise AssertionError(f"unexpected git cat-file header: {header!r}")
        data = process.stdout.read(size)
        if process.stdout.read(1) != b"\n":
            raise AssertionError(f"missing git cat-file terminator: {object_id}")
        if b"\0" in data:
            continue
        for pattern_index, pattern in enumerate(HIGH_CONFIDENCE_SECRET_PATTERNS):
            if pattern.search(data):
                hits.extend(
                    (relative, pattern_index) for relative in object_paths
                )
    process.stdin.close()
    if process.wait(timeout=30) != 0:
        raise AssertionError("git cat-file --batch failed")
    return hits


def main() -> int:
    entries = index_entries()
    metadata = object_metadata(entries)
    errors: list[str] = []

    banned = [
        relative for relative in BANNED_TRACKED_PATHS
        if relative in entries
    ]
    if banned:
        errors.append(f"banned tracked paths remain: {banned}")

    unexpected_zero = [
        relative
        for relative, (_, object_id) in entries.items()
        if metadata[object_id] == ("blob", 0)
        and relative not in INTENTIONAL_ZERO_BYTE_PATHS
    ]
    if unexpected_zero:
        errors.append(f"unexpected tracked zero-byte files remain: {unexpected_zero}")

    duplicate_groups = [
        pair
        for pair in TARGETED_DUPLICATE_PAIRS
        if all(relative in entries for relative in pair)
        and entries[pair[0]][1] == entries[pair[1]][1]
    ]
    if duplicate_groups:
        errors.append(f"targeted duplicate groups remain: {duplicate_groups}")

    doxyfile = read_index_text(entries, "docs/Doxyfile")
    hugo = read_index_text(entries, "docs/hugo.toml")
    dtw_page = read_index_text(entries, "docs/content/method/dtw.md")
    mip_page = read_index_text(entries, "docs/content/method/mip.md")
    asset_checks = (
        "PROJECT_LOGO           = ./docs/static/docs_logo.png" in doxyfile
        and "docs/static/docs_logo.png" in entries,
        'logo = "docs_logo.png"' in hugo
        and "docs/static/docs_logo.png" in entries,
        'src="/method/dtw_image.png"' in dtw_page
        and "docs/static/method/dtw_image.png" in entries,
        'src="/method/cluster_matrix_formation4.svg"' in mip_page
        and "docs/static/method/cluster_matrix_formation4.svg" in entries,
    )
    asset_routes = sum(asset_checks)
    if asset_routes != len(asset_checks):
        errors.append(f"registered asset routes failed: {asset_checks}")

    ignore_lines = {
        line.strip()
        for line in read_index_text(entries, ".gitignore").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    present_ignore_targets = [
        target for target in REQUIRED_IGNORE_TARGETS if target in ignore_lines
    ]
    missing_ignore_targets = [
        target for target in REQUIRED_IGNORE_TARGETS if target not in ignore_lines
    ]
    if missing_ignore_targets:
        errors.append(f"required ignore targets missing: {missing_ignore_targets}")
    missing_benchmark_routes = [
        target
        for target in REQUIRED_BENCHMARK_IGNORE_ROUTES
        if target not in ignore_lines
    ]
    if missing_benchmark_routes:
        errors.append(
            f"required benchmark ignore routes missing: {missing_benchmark_routes}"
        )
    if "benchmarks/baselines/*.json" in ignore_lines:
        errors.append("obsolete benchmarks/baselines/*.json ignore rule remains")
    if not worktree_matches_index(".gitignore"):
        errors.append("working-tree .gitignore differs from the staged index blob")
    ignore_semantic_expectations = (
        (
            "benchmarks/results/_autorun/bench_dtw_baseline_20260412_221116.json",
            False,
        ),
        (
            "benchmarks/results/_autorun/bench_metal_dtw_20260412_212302.json",
            False,
        ),
        ("benchmarks/results/_autorun/future.json", True),
        ("benchmarks/results/future.json", True),
        ("benchmarks/baselines/future.json", False),
        (".env", True),
        (".env.local", True),
        (".env.example", False),
    )
    wrong_ignore_semantics: list[tuple[str, bool, bool]] = []
    for relative, expected in ignore_semantic_expectations:
        actual = is_ignored(relative)
        if actual != expected:
            wrong_ignore_semantics.append((relative, expected, actual))
    if wrong_ignore_semantics:
        errors.append(
            "git ignore semantic probes differ from expectations "
            f"(path, expected, actual): {wrong_ignore_semantics}"
        )

    secret_hits = scan_high_confidence_secrets(entries, metadata)
    if secret_hits:
        hit_paths = sorted({relative for relative, _ in secret_hits})
        errors.append(f"high-confidence secret shapes found in: {hit_paths}")

    readme = read_index_text(entries, "README.md")
    codecov_query_hits = len(
        re.findall(r"https://codecov\.io/[^\s)\]]+badge\.svg\?[^\s)\]]+", readme)
    )
    if codecov_query_hits:
        errors.append("the public Codecov badge URL still contains a query string")

    changelog = read_index_text(entries, "CHANGELOG.md")
    unreleased_positions = [
        match.start() for match in re.finditer(r"(?m)^# Unreleased$", changelog)
    ]
    rc1_positions = [
        match.start()
        for match in re.finditer(
            r"(?m)^# 2\.0\.0rc1 - 2026-07-10$", changelog
        )
    ]
    structure_checks = (
        len(unreleased_positions) == 1,
        len(rc1_positions) == 1,
        len(unreleased_positions) == 1
        and len(rc1_positions) == 1
        and unreleased_positions[0] < rc1_positions[0],
        changelog.count("**Breaking:** `--ram-limit`") == 1,
        changelog.count(
            "**Breaking:** the CLI now rejects `--device cuda`"
        ) == 1,
    )
    changelog_structure = all(structure_checks)
    if not changelog_structure:
        errors.append(f"CHANGELOG structure/F7 checks failed: {structure_checks}")

    compatibility_headers = (
        "**Breaking compatibility (portable-v1):**",
        "**Breaking compatibility (seed-42 default):**",
    )
    normalized_changelog = " ".join(changelog.split())
    compatibility_checks = (
        changelog.count(compatibility_headers[0]) == 1
        and all(
            marker in normalized_changelog
            for marker in (
                "same explicit seed",
                "vendor-defined mappings",
                "literal medoids, labels, and barycenters",
                "unseeded Tier-2",
            )
        ),
        changelog.count(compatibility_headers[1]) == 1
        and all(
            marker in normalized_changelog
            for marker in (
                "ambiguous default outputs",
                "Lloyd local optima",
                "time-limited solver trajectories",
                "exact optimum value",
            )
        ),
    )
    seed_compatibility_markers = sum(compatibility_checks)
    if seed_compatibility_markers != len(compatibility_checks):
        errors.append(
            "CHANGELOG omits one or more registered seed compatibility disclosures"
        )
    forbidden_seed_promises = (
        "seed 29 reproduces rc1",
        "seed 29 recreates rc1",
        "seed 29 restores rc1",
    )
    present_promises = [
        marker
        for marker in forbidden_seed_promises
        if marker in normalized_changelog.lower()
    ]
    if present_promises:
        errors.append(f"CHANGELOG retains forbidden rc1 seed promise: {present_promises}")

    print(f"banned_tracked_paths={len(banned)}")
    print(f"unexpected_zero_byte_files={len(unexpected_zero)}")
    print(f"targeted_duplicate_groups={len(duplicate_groups)}")
    print(f"asset_routes={asset_routes}/{len(asset_checks)}")
    print(
        "required_ignore_targets="
        f"{len(present_ignore_targets)}/{len(REQUIRED_IGNORE_TARGETS)}"
    )
    print(f"high_confidence_secret_hits={len(secret_hits)}")
    print(f"codecov_badge_query_hits={codecov_query_hits}")
    print(f"changelog_structure={'PASS' if changelog_structure else 'FAIL'}")
    print(
        "seed_compatibility_markers="
        f"{seed_compatibility_markers}/{len(compatibility_checks)}"
    )
    if errors:
        print("VERDICT=FAIL")
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print("VERDICT=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
