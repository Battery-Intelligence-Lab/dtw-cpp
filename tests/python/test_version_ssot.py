"""Release-version single-source-of-truth regression tests."""

from pathlib import Path
import os
import subprocess

import dtwcpp


ROOT = Path(__file__).resolve().parents[2]


def _cli_path() -> Path:
    override = os.environ.get("DTWC_CL_PATH")
    candidates = [
        Path(override) if override else None,
        ROOT / "build" / "cfg-gate-normal" / "dtwc_cl.exe",
        ROOT / "build" / "cfg-gate-normal" / "bin" / "dtwc_cl.exe",
    ]
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate
    raise AssertionError("dtwc_cl executable not found; set DTWC_CL_PATH")


def test_version_file_python_metadata_and_cli_match():
    expected = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    assert expected == "2.0.0rc1"
    assert dtwcpp.__version__ == expected
    completed = subprocess.run(
        [str(_cli_path()), "--version"], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == expected
