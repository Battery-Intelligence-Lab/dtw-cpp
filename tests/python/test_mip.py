"""
@file test_mip.py
@brief Method.MIP, and Method.LRCore's kelley root, through highspy, the route of an
       extension that links no HiGHS (the wheel).
@details
    C++ builds the model as arrays, highspy solves them and C++ decodes the solution.
    highspy's HiGHS may not be the version dtwc_cl links, so the optimum is compared,
    within the MIP gap, and not the labels. LR-core runs in C++ and solves its Kelley
    root's master LP with highspy; its oracle is brute force over every medoid set.
"""
import itertools
import re
import subprocess
import sys

import numpy as np
import pytest

import dtwcpp


# highspy_route (conftest.py) skips without highspy or beside linked HiGHS; the Python CI job
# sets DTWC_REQUIRE_HIGHSPY, which makes that a failure. dtwc_cl (conftest.py) is the binary
# DTWC_CL_PATH names, the linked-HiGHS oracle.
@pytest.mark.parametrize("n, length, k, seed", [(12, 20, 3, 20261006), (18, 16, 4, 7)])
def test_highspy_reaches_the_optimum_of_linked_highs(highspy_route, dtwc_cl, tmp_path, n, length, k, seed):
    cli = dtwc_cl
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
    with pytest.raises(dtwcpp.SolverError, match=r'pip install "dtwcpp\[mip\]"'):
        dtwcpp.cluster([[0.0, 1.0], [5.0, 6.0], [0.5, 1.5]], 2, method="mip")


@pytest.mark.parametrize("n, length, k, seed", [(12, 20, 3, 20261008), (14, 16, 4, 7)])
def test_lrcore_kelley_root_on_highspy_reaches_the_brute_force_optimum(highspy_route, monkeypatch,
                                                                       n, length, k, seed):
    """lr_root="kelley" solves LR-core's root on the highspy master (each solve is counted; the
    default root solves none) and reaches the optimum of the default subgradient root, which
    brute force over every medoid set confirms; it is unique here, so the clusterings agree."""
    from dtwcpp import _mip
    solves = []
    run = _mip._KelleyMaster.run
    monkeypatch.setattr(_mip._KelleyMaster, "run", lambda self: solves.append(1) or run(self))
    series = np.random.default_rng(seed).standard_normal((n, length))

    kelley = dtwcpp.cluster(series, k, method="lrcore", lr_root="kelley")
    kelley_solves = len(solves)
    default = dtwcpp.cluster(series, k, method="lrcore")

    assert kelley_solves > 0 and len(solves) == kelley_solves
    d = kelley.distance_matrix
    costs = sorted(d[:, list(m)].min(axis=1).sum() for m in itertools.combinations(range(n), k))
    assert costs[1] > costs[0] * (1 + 1e-9)  # one optimal medoid set
    assert kelley.cost == pytest.approx(costs[0], rel=1e-12)
    assert default.cost == pytest.approx(costs[0], rel=1e-12)
    assert np.array_equal(kelley.medoids, default.medoids)
    assert np.array_equal(kelley.labels, default.labels)


def test_lrcore_kelley_without_highspy_names_the_extra_before_any_work(monkeypatch):
    monkeypatch.setattr(dtwcpp, "HIGHS_AVAILABLE", False)
    monkeypatch.setitem(sys.modules, "highspy", None)  # `import highspy` raises ImportError
    prob = dtwcpp.Problem("kelley_without_highspy")
    prob.set_data([[0.0, 1.0], [5.0, 6.0], [0.5, 1.5]])
    prob.set_n_clusters(2)
    prob.method = dtwcpp.Method.LRCore
    settings = dtwcpp.MIPSettings()
    settings.lr_root = "kelley"
    prob.mip_settings = settings
    with pytest.raises(dtwcpp.SolverError, match=r'pip install "dtwcpp\[mip\]"'):
        prob.cluster()
    assert not prob.is_distance_matrix_filled()  # refused before any distance was computed
