"""The Python quickstart printed in the Hugo site must remain executable."""

import runpy
from pathlib import Path


def test_documented_python_quickstart(monkeypatch):
    repo = Path(__file__).resolve().parents[2]
    monkeypatch.chdir(repo)
    monkeypatch.setattr("sys.argv", ["quickstart.py"])
    runpy.run_path(str(repo / "docs" / "examples" / "quickstart.py"),
                   run_name="__main__")
