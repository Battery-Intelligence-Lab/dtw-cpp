#!/usr/bin/env python3
"""Run and fail-closed adjudicate the registered D3 closure gates."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Final


ROOT: Final = Path(__file__).resolve().parents[1]

D3_MARKER: Final = (
    "D3_LB_ENHANCED_WEBB_GATE envelope_cases=2004 path_cases=35982 "
    "full_cover_cases=7380 enhanced_cases=68787 enhanced_v5=4/4 "
    "webb_cases=35982 webb_branches=4/4 webb_strict=2/2 "
    "tail_cases=35982 tail_strict=2/2 metric_cases=140 "
    "order_witnesses=2/2 cascade_routes=2/2 skips=0 verdict=PASS"
)
F57_MARKER: Final = (
    "F57_LB_WEBB_INTMAX l1=4/4 squared=8/8 global_parity=2/2 "
    "admissible=2/2 skips=0 verdict=PASS"
)

FOCUSED_TESTS: Final = (
    "test_lb_enhanced_webb",
    "test_lb_enhanced_webb_derivation",
    "test_lb_webb_intmax",
    "unit_test_lower_bounds",
    "unit_test_pruned_distance_matrix",
)

ARROW_SUBJECT_TESTS: Final = (
    "test_lb_enhanced_webb_derivation",
    "test_lb_webb_intmax",
    "test_io_readers",
    "test_distance_matrix_csv_contract",
    "test_cli_resume_state",
    "test_fast_clara_parquet_parity",
    "test_fast_clara_assignment_contract",
)

ARROW_SUBJECT_MARKERS: Final = (
    D3_MARKER,
    F57_MARKER,
    "All tests passed (390 assertions in 11 test cases)",
    (
        "F14_CSV_PUBLIC subject=real_dtwc_cl+native_result runs=2/2 "
        "route_markers=1/1 matrix_pairs=1/1 rows=27/27 lf=27/27 cr=0/0 "
        "final_lf=2/2 blank_tail=0 mmap=unavailable skips=0"
    ),
    (
        "F17_CLI_RESUME subject=real_dtwc_cl writer=production_serializer "
        "runs=12/12 replay_fields=10/10 markers=2/2 algorithm_skipped=1/1 "
        "fresh_discriminator=4/4 checkpoint_preserved=10/10 "
        "rejection_cases=9/9 sources_preserved=2/2 skips=0"
    ),
    (
        "F8_PARITY subject=real_dtwc_cl runs=6 route_markers=12/12 "
        "parity=9/9 configs=3/3_distinct "
        "fixture_sha256=2F259F418A6BB9C62CA0004CB334C05E8309213F5A76DC890F83C15D5BDA3CA8"
    ),
    (
        "F13_ASSIGNMENT_CONTRACT subject=real_dtwc_cl runs=4/4 "
        "route_markers=8/8 artifact_pairs=6/6 payloads=4/4 "
        "objective_bytes=4/4 stream_rejections=2/2 skips=0 "
        "fixture_sha256=2F259F418A6BB9C62CA0004CB334C05E8309213F5A76DC890F83C15D5BDA3CA8"
    ),
)


@dataclass(frozen=True)
class MatrixBand:
    build_dir: str
    total: int
    skips: frozenset[str]


MATRIX_BANDS: Final = {
    "canonical": MatrixBand(
        "build/highs-1151",
        125,
        frozenset(
            {
                "test_cuda_correctness",
                "test_cuda_lb_keogh",
                "test_io_readers",
                "test_metal_correctness",
                "test_metal_lb_keogh",
                "test_metal_mmap",
            }
        ),
    ),
    "nollfio": MatrixBand(
        "build/nollfio",
        125,
        frozenset(
            {
                "unit_test_mmap_data_store",
                "unit_test_mmap_distance_matrix",
                "test_cuda_correctness",
                "test_cuda_lb_keogh",
                "test_io_readers",
                "test_metal_correctness",
                "test_metal_lb_keogh",
                "test_metal_mmap",
                "unit_test_benders",
            }
        ),
    ),
    "arrow": MatrixBand(
        "build/arrow-pyarrow-23",
        127,
        frozenset(
            {
                "unit_test_mmap_data_store",
                "unit_test_mmap_distance_matrix",
                "test_cuda_correctness",
                "test_cuda_lb_keogh",
                "test_metal_correctness",
                "test_metal_lb_keogh",
                "test_metal_mmap",
                "unit_test_benders",
            }
        ),
    ),
}


def _exactly_once(text: str, literal: str) -> bool:
    return text.count(literal) == 1


def _inventory_names(text: str) -> tuple[str, ...]:
    return tuple(
        re.findall(r"^\s*Test\s+#\d+:\s+(\S+)\s*$", text, re.MULTILINE)
    )


def _skipped_names(text: str) -> frozenset[str]:
    return frozenset(
        re.findall(
            r"^\s*\d+\s+-\s+(\S+)\s+\(Skipped\)\s*$", text, re.MULTILINE
        )
    )


def _test_passed(text: str, name: str) -> bool:
    return bool(
        re.search(
            rf"^\s*\d+/\d+\s+Test\s+#\d+:\s+{re.escape(name)}\s+"
            r"\.{2,}\s+Passed\s+",
            text,
            re.MULTILINE,
        )
    )


def _summary_is_exact(text: str, total: int) -> bool:
    return _exactly_once(text, f"100% tests passed, 0 tests failed out of {total}")


def adjudicate_full(
    matrix: str, inventory_text: str, output: str, returncode: int
) -> tuple[bool, str]:
    band = MATRIX_BANDS[matrix]
    inventory = _inventory_names(inventory_text)
    required = ("test_lb_enhanced_webb_derivation", "test_lb_webb_intmax")
    inventory_ok = len(inventory) == band.total and all(
        inventory.count(name) == 1 for name in required
    )
    skip_names = _skipped_names(output)
    subjects_passed = all(_test_passed(output, name) for name in required)
    ok = all(
        (
            returncode == 0,
            inventory_ok,
            _summary_is_exact(output, band.total),
            skip_names == band.skips,
            subjects_passed,
        )
    )
    detail = (
        f"rc={returncode} inventory={len(inventory)}/{band.total} "
        f"subjects={int(subjects_passed)}/1 "
        f"skips={len(skip_names)}/{len(band.skips)} "
        f"skip_set_match={skip_names == band.skips} "
        f"summary_exact={_summary_is_exact(output, band.total)}"
    )
    return ok, detail


def adjudicate_selected(
    names: tuple[str, ...], markers: tuple[str, ...], output: str, returncode: int
) -> tuple[bool, str]:
    passed = tuple(name for name in names if _test_passed(output, name))
    found_markers = tuple(marker for marker in markers if _exactly_once(output, marker))
    no_skip_result = not _skipped_names(output) and not re.search(
        r"^\s*\d+/\d+\s+Test.*\*\*\*Skipped", output, re.MULTILINE
    )
    ok = all(
        (
            returncode == 0,
            len(passed) == len(names),
            len(found_markers) == len(markers),
            _summary_is_exact(output, len(names)),
            no_skip_result,
        )
    )
    detail = (
        f"rc={returncode} subjects={len(passed)}/{len(names)} "
        f"markers={len(found_markers)}/{len(markers)} "
        f"summary_exact={_summary_is_exact(output, len(names))} "
        f"skip_free={no_skip_result}"
    )
    return ok, detail


def _passed_line(index: int, total: int, name: str) -> str:
    return f"{index}/{total} Test #{index}: {name} .......   Passed    0.01 sec"


def _valid_full_fixture(matrix: str) -> tuple[str, str]:
    band = MATRIX_BANDS[matrix]
    inventory = "\n".join(
        f"  Test #{index}: synthetic_{index}" for index in range(1, band.total - 1)
    )
    inventory += (
        f"\n  Test #{band.total - 1}: test_lb_enhanced_webb_derivation"
        f"\n  Test #{band.total}: test_lb_webb_intmax\n"
    )
    output = "\n".join(
        (
            _passed_line(1, band.total, "test_lb_enhanced_webb_derivation"),
            _passed_line(2, band.total, "test_lb_webb_intmax"),
            f"100% tests passed, 0 tests failed out of {band.total}",
            "The following tests did not run:",
            *(
                f"  {index} - {name} (Skipped)"
                for index, name in enumerate(sorted(band.skips), start=1)
            ),
        )
    )
    return inventory, output


def _valid_selected_fixture(
    names: tuple[str, ...], markers: tuple[str, ...]
) -> str:
    return "\n".join(
        (
            *markers,
            *(_passed_line(index, len(names), name) for index, name in enumerate(names, 1)),
            f"100% tests passed, 0 tests failed out of {len(names)}",
        )
    )


def self_test() -> bool:
    mutations = 0
    rejected = 0

    for matrix, band in MATRIX_BANDS.items():
        inventory, output = _valid_full_fixture(matrix)
        ok, _ = adjudicate_full(matrix, inventory, output, 0)
        if not ok:
            return False
        variants = (
            (inventory, output, 1),
            (inventory.replace("test_lb_webb_intmax", "missing_subject"), output, 0),
            (inventory, output.replace(f"out of {band.total}", "out of 999"), 0),
            (inventory, output.replace(" (Skipped)", "", 1), 0),
            (inventory, output.replace("Passed", "***Skipped", 1), 0),
        )
        for mutated_inventory, mutated_output, rc in variants:
            mutations += 1
            mutated_ok, _ = adjudicate_full(
                matrix, mutated_inventory, mutated_output, rc
            )
            rejected += int(not mutated_ok)

    selected_cases = (
        (FOCUSED_TESTS, (D3_MARKER, F57_MARKER)),
        (ARROW_SUBJECT_TESTS, ARROW_SUBJECT_MARKERS),
    )
    for names, markers in selected_cases:
        output = _valid_selected_fixture(names, markers)
        ok, _ = adjudicate_selected(names, markers, output, 0)
        if not ok:
            return False
        variants = [
            (output, 1),
            (output.replace(markers[0], "missing_marker"), 0),
            (output.replace(names[0], "missing_subject"), 0),
            (output.replace(f"out of {len(names)}", "out of 999"), 0),
            (output.replace("Passed", "***Skipped", 1), 0),
        ]
        for mutated_output, rc in variants:
            mutations += 1
            mutated_ok, _ = adjudicate_selected(names, markers, mutated_output, rc)
            rejected += int(not mutated_ok)

    ok = rejected == mutations
    print(
        f"D3_CLOSURE_SELF_TEST mutations={mutations} rejected={rejected} "
        f"verdict={'PASS' if ok else 'FAIL'}"
    )
    return ok


def _run(command: list[str]) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )
    return completed.returncode, completed.stdout


def run_full(matrix: str) -> bool:
    band = MATRIX_BANDS[matrix]
    inventory_rc, inventory = _run(
        ["ctest", "--test-dir", band.build_dir, "-N"]
    )
    if inventory_rc != 0:
        print(inventory, end="")
        print(f"D3_CLOSURE_{matrix.upper()} inventory_rc={inventory_rc} verdict=FAIL")
        return False

    command = [
        "ctest",
        "--test-dir",
        band.build_dir,
        "-C",
        "Release",
        "--output-on-failure",
        "--no-tests=error",
        "-j",
        "1",
    ]
    returncode, output = _run(command)
    print(output, end="")
    ok, detail = adjudicate_full(matrix, inventory, output, returncode)
    print(
        f"D3_CLOSURE_{matrix.upper()} {detail} "
        f"verdict={'PASS' if ok else 'FAIL'}"
    )
    return ok


def run_selected(mode: str) -> bool:
    if mode == "focused":
        band = MATRIX_BANDS["canonical"]
        names = FOCUSED_TESTS
        markers = (D3_MARKER, F57_MARKER)
    else:
        band = MATRIX_BANDS["arrow"]
        names = ARROW_SUBJECT_TESTS
        markers = ARROW_SUBJECT_MARKERS

    expression = "^(" + "|".join(re.escape(name) for name in names) + ")$"
    command = [
        "ctest",
        "--test-dir",
        band.build_dir,
        "-C",
        "Release",
        "-R",
        expression,
        "--output-on-failure",
        "--no-tests=error",
        "-V",
        "-j",
        "1",
    ]
    returncode, output = _run(command)
    print(output, end="")
    ok, detail = adjudicate_selected(names, markers, output, returncode)
    print(
        f"D3_CLOSURE_{mode.upper()} {detail} "
        f"verdict={'PASS' if ok else 'FAIL'}"
    )
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("self-test", "focused", "canonical", "nollfio", "arrow", "arrow-subjects")
    )
    args = parser.parse_args()

    if args.mode == "self-test":
        return 0 if self_test() else 1
    if args.mode in MATRIX_BANDS:
        return 0 if run_full(args.mode) else 1
    return 0 if run_selected(args.mode) else 1


if __name__ == "__main__":
    sys.exit(main())
