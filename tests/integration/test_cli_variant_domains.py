"""Executable Phase 8 M34 CLI domain gate.

Run explicitly with the freshly built binary::

    python tests/integration/test_cli_variant_domains.py --cli build/bin/dtwc_cl

The invalid matrix uses a missing input and a fresh output path deliberately:
parameter validation must win before filesystem/data effects.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import subprocess
import sys
import tempfile


INVALID_CASES = [
    (["--variant", "standard", "--adtw-penalty", "-1"],
     "ADTW penalty must be finite and non-negative."),
    (["--variant", "wdtw", "--wdtw-g", "-1"],
     "WDTW g must be finite and non-negative."),
    (["--variant", "adtw", "--adtw-penalty", "-1"],
     "ADTW penalty must be finite and non-negative."),
    (["--variant", "softdtw", "--sdtw-gamma", "0"],
     "Soft-DTW gamma must be finite and positive."),
    (["--variant", "softdtw", "--sdtw-gamma", "nan"],
     "Soft-DTW gamma must be finite and positive."),
    (["--variant", "msm", "--msm-c", "0"],
     "MSM c must be finite and positive."),
    (["--variant", "twe", "--twe-nu", "0"],
     "TWE nu must be finite and positive."),
    (["--variant", "twe", "--twe-lambda", "0"],
     "TWE lambda must be finite and positive."),
]


VALID_CASES = [
    ["--variant", "standard"],
    ["--variant", "wdtw", "--wdtw-g", "0"],
    ["--variant", "adtw", "--adtw-penalty", "0"],
    ["--variant", "softdtw", "--sdtw-gamma", str(sys.float_info.min)],
    ["--variant", "msm", "--msm-c", str(sys.float_info.min)],
    ["--variant", "twe", "--twe-nu", str(sys.float_info.min),
     "--twe-lambda", "0.8"],
]


def run(cli: Path, args: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(cli), *args], capture_output=True, text=True,
        encoding="utf-8", errors="replace", check=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cli", type=Path, required=True)
    ns = parser.parse_args()
    cli = ns.cli.resolve()
    if not cli.is_file():
        raise AssertionError(f"CLI does not exist: {cli}")

    with tempfile.TemporaryDirectory(prefix="dtwc-m34-") as temp:
        root = Path(temp)

        for index, (variant_args, expected) in enumerate(INVALID_CASES):
            output = root / f"invalid-{index}"
            result = run(cli, [
                "--input", str(root / "must-not-be-read.tsv"),
                "--output", str(output), "--name", f"m34-invalid-{index}",
                "--n-clusters", "1", *variant_args,
            ])
            assert result.returncode != 0, (variant_args, result.stdout, result.stderr)
            assert expected in result.stderr, (variant_args, result.stderr)
            assert not output.exists(), f"invalid case created output: {output}"

        fixture = root / "series.tsv"
        fixture.write_text("s0\t0\ns1\t0\t0\n", encoding="utf-8", newline="\n")
        for index, variant_args in enumerate(VALID_CASES):
            output = root / f"valid-{index}"
            result = run(cli, [
                "--input", str(fixture), "--skip-cols", "1",
                "--output", str(output), "--name", f"m34-valid-{index}",
                "--n-clusters", "1", "--method", "pam", "--max-iter", "1",
                *variant_args,
            ])
            assert result.returncode == 0, (variant_args, result.stdout, result.stderr)

    print(f"M34 CLI domains: {len(INVALID_CASES)} invalid + "
          f"{len(VALID_CASES)} valid cases passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
