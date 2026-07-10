"""Executable Phase 8 M26/M40 CLI missing-data boundary gate.

Run explicitly with the freshly built binary::

    python tests/integration/test_cli_missing_data.py --cli build/bin/dtwc_cl
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import tempfile


def run(cli: Path, args: list[str], threads: int) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = str(threads)
    return subprocess.run(
        [str(cli), *args], capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False, env=env, timeout=30,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    ns = parser.parse_args()
    cli = ns.cli.resolve()
    if not cli.is_file():
        raise AssertionError(f"CLI does not exist: {cli}")

    with tempfile.TemporaryDirectory(prefix="dtwc-m26-m40-") as temp:
        root = Path(temp)
        fixture = root / "missing.tsv"
        fixture.write_text(
            "0\t0\tnan\n0\t0\t0\n0\t0\t0\n",
            encoding="utf-8", newline="\n",
        )

        common = [
            "--input", str(fixture), "--output", str(root / "results"),
            "--n-clusters", "1", "--method", "pam", "--max-iter", "1",
        ]
        for threads in (1, 4):
            name = f"error-{threads}"
            result = run(cli, [*common, "--name", name,
                               "--missing-strategy", "error"], threads)
            combined = result.stdout + result.stderr
            assert result.returncode == 1, (threads, result.returncode, combined)
            assert "Error: fill_distance_matrix: NaN detected" in result.stderr, combined
            assert "Set missing_strategy to ZeroCost, AROW, or Interpolate" in result.stderr
            assert "terminate" not in combined.lower(), combined
            assert "abort" not in combined.lower(), combined
            assert not (root / "results" / f"{name}_labels.csv").exists()
            assert not (root / "results" / f"{name}_checkpoint.bin").exists()

        zero = run(cli, [*common, "--name", "zero", "--verbose",
                         "--missing-strategy", "zero_cost"], 4)
        assert zero.returncode == 0, (zero.stdout, zero.stderr)
        assert "3 series, 3 avg length" in zero.stdout, zero.stdout
        assert "Total cost: 0" in zero.stdout, zero.stdout
        assert (root / "results" / "zero_labels.csv").is_file()

        malformed = root / "malformed.tsv"
        malformed.write_text("0\t0\t0\n0\toops\t2\n", encoding="utf-8", newline="\n")
        bad = run(cli, [
            "--input", str(malformed), "--output", str(root / "bad-results"),
            "--name", "bad", "--n-clusters", "1", "--method", "pam",
            "--missing-strategy", "zero_cost",
        ], 4)
        assert bad.returncode == 1, (bad.stdout, bad.stderr)
        assert "row 2, column 2: invalid numeric field 'oops'" in bad.stderr
        assert not (root / "bad-results" / "bad_labels.csv").exists()

    print("M26/M40 CLI missing-data gate: Error 1/4, ZeroCost, malformed passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
