#!/usr/bin/env python3
"""Kill the three registered F45 diagnostic-containment mutants."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
from pathlib import Path
import re
import subprocess
import sys


PASS_MARKER = (
    "F45_LLFIO_DIAGNOSTIC_STATE probes=4/4 diagnostics=4/4 "
    "raw_include=1/1 wrapper_routes=2/2 skips=0 verdict=PASS"
)
MARKER_PATTERN = re.compile(
    r"^F45_LLFIO_DIAGNOSTIC_STATE (?P<fields>.+) verdict=(?P<verdict>PASS|FAIL)$",
    re.MULTILINE,
)


@dataclass(frozen=True)
class Mutation:
    name: str
    relative_path: str
    needle: bytes
    replacement: bytes
    expected_diagnostics: str
    expected_raw_include: str
    expected_wrapper_routes: str


MUTATIONS = (
    Mutation(
        "missing-pop",
        "dtwc/core/llfio_include.hpp",
        (
            b"\n#if defined(__clang__)\n"
            b"#  pragma clang diagnostic pop\n"
            b"#endif\n"
        ),
        b"\n",
        "1/4",
        "1/1",
        "2/2",
    ),
    Mutation(
        "raw-data-store-include",
        "dtwc/core/mmap_data_store.hpp",
        b'#include "llfio_include.hpp"',
        b"#include <llfio/v2.0/llfio.hpp>",
        "3/4",
        "2/1",
        "1/2",
    ),
    Mutation(
        "raw-distance-matrix-include",
        "dtwc/core/mmap_distance_matrix.hpp",
        b'#include "llfio_include.hpp"',
        b"#include <llfio/v2.0/llfio.hpp>",
        "3/4",
        "2/1",
        "1/2",
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=Path("build/highs-1151"))
    parser.add_argument(
        "--off-build-dir", type=Path, default=Path("build/nollfio")
    )
    parser.add_argument("--source-dir", type=Path)
    return parser.parse_args()


def require_inside(path: Path, root: Path, description: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise RuntimeError(
            f"{description} escapes the repository root: {path}"
        ) from error


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest().upper()


def replace_once(payload: bytes, needle: bytes, replacement: bytes, name: str) -> bytes:
    count = payload.count(needle)
    if count != 1:
        raise RuntimeError(
            f"{name} expected one mutation site, observed {count}"
        )
    return payload.replace(needle, replacement, 1)


def run_gate(
    source_dir: Path, build_dir: Path, off_build_dir: Path
) -> tuple[int, str, re.Match[str] | None]:
    gate = source_dir / "scripts/test_f45_llfio_diagnostic_state.py"
    result = subprocess.run(
        [
            sys.executable,
            str(gate),
            "--source-dir",
            str(source_dir),
            "--build-dir",
            str(build_dir),
            "--off-build-dir",
            str(off_build_dir),
        ],
        cwd=source_dir,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    matches = list(MARKER_PATTERN.finditer(result.stdout))
    marker = matches[0] if len(matches) == 1 else None
    return result.returncode, result.stdout, marker


def marker_fields(marker: re.Match[str]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for token in marker.group("fields").split():
        key, value = token.split("=", 1)
        fields[key] = value
    fields["verdict"] = marker.group("verdict")
    return fields


def require_control(
    label: str, source_dir: Path, build_dir: Path, off_build_dir: Path
) -> None:
    returncode, output, marker = run_gate(
        source_dir, build_dir, off_build_dir
    )
    if returncode != 0 or marker is None or marker.group(0) != PASS_MARKER:
        raise RuntimeError(
            f"{label} control failed with exit {returncode}\n{output}"
        )
    print(f"F45_MUTATION_CONTROL label={label} verdict=PASS")


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

    originals: dict[Path, bytes] = {}
    original_hashes: dict[Path, str] = {}
    killed = 0
    restored = 0
    try:
        require_inside(build_dir, source_dir, "build directory")
        require_inside(
            off_build_dir, source_dir, "LLFIO-OFF build directory"
        )
        for mutation in MUTATIONS:
            target = (source_dir / mutation.relative_path).resolve()
            require_inside(target, source_dir, mutation.name)
            if target not in originals:
                originals[target] = target.read_bytes()
                original_hashes[target] = sha256(originals[target])

        require_control("initial", source_dir, build_dir, off_build_dir)

        for mutation in MUTATIONS:
            target = (source_dir / mutation.relative_path).resolve()
            for path, original in originals.items():
                if path.read_bytes() != original:
                    raise RuntimeError(
                        f"pre-mutation source drift before {mutation.name}: {path}"
                    )

            mutant = replace_once(
                originals[target],
                mutation.needle,
                mutation.replacement,
                mutation.name,
            )
            mutant_hash = sha256(mutant)
            if mutant_hash == original_hashes[target]:
                raise RuntimeError(f"{mutation.name} did not change source bytes")
            target.write_bytes(mutant)

            try:
                returncode, output, marker = run_gate(
                    source_dir, build_dir, off_build_dir
                )
                if returncode != 1 or marker is None:
                    raise RuntimeError(
                        f"{mutation.name} was not killed cleanly; "
                        f"exit={returncode}\n{output}"
                    )
                fields = marker_fields(marker)
                if (
                    fields.get("verdict") != "FAIL"
                    or fields.get("diagnostics")
                    != mutation.expected_diagnostics
                    or fields.get("raw_include")
                    != mutation.expected_raw_include
                    or fields.get("wrapper_routes")
                    != mutation.expected_wrapper_routes
                ):
                    raise RuntimeError(
                        f"{mutation.name} produced the wrong failure marker: "
                        f"{marker.group(0)}\n{output}"
                    )
                killed += 1
            finally:
                target.write_bytes(originals[target])

            restored_payload = target.read_bytes()
            restored_hash = sha256(restored_payload)
            if (
                restored_payload != originals[target]
                or restored_hash != original_hashes[target]
            ):
                raise RuntimeError(
                    f"{mutation.name} failed exact source restoration"
                )
            restored += 1
            print(
                "F45_MUTATION "
                f"name={mutation.name} result=KILLED "
                f"mutant_sha256={mutant_hash} "
                f"restored_sha256={restored_hash} restore=pass"
            )

        require_control("final", source_dir, build_dir, off_build_dir)
        if killed != 3 or restored != 3:
            raise RuntimeError(
                f"registered ledger mismatch: killed={killed} restored={restored}"
            )
        print(
            "F45_LLFIO_DIAGNOSTIC_MUTATIONS "
            "controls=2/2 mutations=3/3 killed=3/3 survived=0 "
            "restored=3/3 skips=0 verdict=PASS"
        )
        return 0
    except (OSError, RuntimeError, ValueError) as error:
        print(f"F45_MUTATION_HARNESS_ERROR {error}")
        return 2
    finally:
        for path, original in originals.items():
            path.write_bytes(original)


if __name__ == "__main__":
    sys.exit(main())
