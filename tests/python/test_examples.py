"""
@file test_examples.py
@brief Every examples/python/*.py runs to exit 0, each in its own temporary
directory. None touches the network; one that needs an optional package skips
naming it.
@author Volkan Kumtepeli
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

_EXAMPLES = Path(__file__).resolve().parents[2] / "examples" / "python"
_NEEDS = {"08_parquet_io.py": "pyarrow", "09_device_clustering.py": "matplotlib"}
_ARGS = {"09_device_clustering.py": ["cpu"]}  # its default device is the GPU; hpc needs a cluster


@pytest.mark.parametrize("example", sorted(path.name for path in _EXAMPLES.glob("*.py")))
def test_example_runs(example, tmp_path):
    needed = _NEEDS.get(example)
    if needed and importlib.util.find_spec(needed) is None:
        pytest.skip(f"{example} needs {needed}, which is not installed")
    env = dict(os.environ, MPLBACKEND="Agg", PYTHONUTF8="1",
               TMPDIR=str(tmp_path), TEMP=str(tmp_path), TMP=str(tmp_path))
    run = subprocess.run([sys.executable, str(_EXAMPLES / example), *_ARGS.get(example, [])],
                         cwd=tmp_path, env=env, capture_output=True, text=True,
                         encoding="utf-8", errors="replace", timeout=600)
    assert run.returncode == 0, run.stdout + run.stderr
