"""
@file test_mip.py
@brief Method.MIP through highspy, the route of an extension that links no HiGHS (the wheel).
@details
    C++ builds the model as arrays, highspy solves them and C++ decodes the solution.
    highspy's HiGHS may not be the version dtwc_cl links, so the optimum is compared,
    within the MIP gap, and not the labels.
"""
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import dtwcpp
from dtwcpp import _hpc

ROOT = Path(__file__).resolve().parents[2]


# The Python CI job installs the mip extra and checks the extension links no HiGHS,
# so neither skip can happen there.
@pytest.mark.skipif(dtwcpp.HIGHS_AVAILABLE, reason="this extension links HiGHS, so method 'mip' "
                    "never reaches highspy (and two HiGHS builds in one process crash)")
@pytest.mark.parametrize("n, length, k, seed", [(12, 20, 3, 20261006), (18, 16, 4, 7)])
def test_highspy_reaches_the_optimum_of_linked_highs(tmp_path, n, length, k, seed):
    pytest.importorskip("highspy")  # the mip extra
    cli = _hpc.find_dtwc_binary(str(ROOT))
    assert cli is not None, "dtwc_cl executable not found; set DTWC_CL_PATH"
    csv = tmp_path / "series.csv"
    np.savetxt(csv, np.random.default_rng(seed).standard_normal((n, length)), delimiter=",", fmt="%.17g")
    run = subprocess.run([cli, "-i", str(csv), "-k", str(k), "-m", "mip", "-o", str(tmp_path / "cli")],
                         capture_output=True, text=True, check=True)
    linked = float(re.search(r"Total cost:\s*(\S+)", run.stdout).group(1))

    result = dtwcpp.cluster(csv, k, method="mip")

    assert result.cost == pytest.approx(linked, rel=dtwcpp.MIPSettings().mip_gap)
    labels, medoids = result.labels, result.medoids
    assert len(set(medoids.tolist())) == k
    assert sorted(set(labels.tolist())) == list(range(k))
    assert [labels[m] for m in medoids] == list(range(k))  # each medoid in its own cluster
    assert result.cost == pytest.approx(result.distance_matrix[np.arange(n), medoids[labels]].sum())


def test_mip_without_highspy_names_the_extra(monkeypatch):
    monkeypatch.setattr(dtwcpp, "HIGHS_AVAILABLE", False)
    monkeypatch.setitem(sys.modules, "highspy", None)  # `import highspy` raises ImportError
    with pytest.raises(dtwcpp.SolverError, match=r"pip install dtwcpp\[mip\]"):
        dtwcpp.cluster([[0.0, 1.0], [5.0, 6.0], [0.5, 1.5]], 2, method="mip")
