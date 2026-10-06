"""Release-version single-source-of-truth regression tests."""

from pathlib import Path
import subprocess

import dtwcpp


ROOT = Path(__file__).resolve().parents[2]


def test_version_file_python_metadata_and_cli_match(dtwc_cl):
    expected = (ROOT / "VERSION").read_text(encoding="utf-8").strip()
    assert expected == "2.0.0rc1"
    assert dtwcpp.__version__ == expected
    completed = subprocess.run(
        [dtwc_cl, "--version"], capture_output=True, text=True, check=True
    )
    assert completed.stdout.strip() == expected
