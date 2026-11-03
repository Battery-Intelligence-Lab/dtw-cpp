"""
@file test_run.py
@brief _dtwcpp_core._run and _Result: Python's route to C++ dtwc::run, the run dtwc_cl makes.
@details
    _run runs a copy of a Config on a source, a path C++ reads or a Data in memory, with the GIL released for the
    whole run, and returns the C++ dtwc::Result as _Result. With external=True, Method.MIP on HiGHS and LR-core's
    kelley root solve with dtwcpp._mip.solve(prob), prob the run's own Problem (the wheel links no HiGHS); these tests
    install a fake dtwcpp._mip, as test_mip.py patches highspy. dtwc_cl (DTWC_CL_PATH, conftest.py) is the oracle of
    the runs' results.
@author Volkan Kumtepeli
"""
import gc
import json
import os
import subprocess
import sys
import threading
import types
from pathlib import Path

import numpy as np
import pytest

import dtwcpp
from dtwcpp import _dtwcpp_core as core

_READER = Path(__file__).resolve().parents[1] / "data" / "reader"


def _config(k, method, **keys):
    config = core.Config()
    config.n_clusters = k
    config.method = method
    for key, value in keys.items():
        setattr(config, key, value)
    return config


def _column(path, index):
    """Column `index` of a result file dtwc_cl writes (labels: name,cluster; medoids: cluster,medoid_index,...)."""
    return [int(line.split(",")[index]) for line in path.read_text(encoding="utf-8").splitlines()[1:]]


def _cli(dtwc_cl, args, env=None):
    """dtwc_cl's run: its stdout, and the 'Name: value' lines of its results."""
    run = subprocess.run([dtwc_cl, *args], capture_output=True, text=True, env=env, check=False)
    assert run.returncode == 0, run.stderr
    results = run.stdout.split("=== Results ===")[1]
    return run.stdout, dict((part.strip() for part in line.split(": ", 1))
                            for line in results.splitlines() if ": " in line)


def _install_solve(monkeypatch, solve):
    """dtwcpp._mip as the run imports it, with `solve` its only function."""
    fake = types.ModuleType("dtwcpp._mip")
    fake.solve = solve
    monkeypatch.setitem(sys.modules, "dtwcpp._mip", fake)


def _clustering(labels, medoids, cost):
    result = dtwcpp.ClusteringResult()
    result.labels, result.medoid_indices, result.total_cost = labels, medoids, cost
    result.converged = True
    return result


def test_a_csv_runs_as_dtwc_cl_runs_it_and_writes_nothing(dtwc_cl, tmp_path, monkeypatch):
    """A path runs as dtwc_cl runs it: labels, medoids, cost, method, iterations. The run writes no file (Config's
    output, "./results" by default, is cleared) and changes no setting of the caller's Config: the same Config then
    runs the file's series in memory, which a Config holding the path as its input would refuse."""
    csv = _READER / "clara_256.csv"
    _, cli = _cli(dtwc_cl, ["-i", str(csv), "-k", "3", "-m", "pam", "--name", "t", "-o", str(tmp_path / "cli")])
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    monkeypatch.chdir(cwd)
    config = _config(3, "pam", name="t")

    result = core._run(config, str(csv))

    assert list(cwd.iterdir()) == []
    assert (config.name, config.n_clusters, config.method) == ("t", 3, "pam")
    assert result.labels.tolist() == _column(tmp_path / "cli" / "t_labels.csv", 1)
    assert result.medoids.tolist() == _column(tmp_path / "cli" / "t_medoids.csv", 1)
    assert result.cost == float(cli["Total cost"])  # dtwc_cl prints the double's shortest round trip
    assert (result.method, result.iterations, result.converged) == (cli["Method"], int(cli["Iterations"]),
                                                                    cli["Converged"] == "yes")
    labels = result.labels
    labels[0] += 1  # an int64 copy: the result keeps its own
    assert labels.dtype == np.int64 and result.labels[0] == labels[0] - 1

    in_memory = core._run(config, dtwcpp.load(csv).as_data())
    assert in_memory.labels.tolist() == result.labels.tolist() and in_memory.cost == result.cost


def test_an_arrow_ipc_file_in_ram_and_streamed_runs_as_dtwc_cl(dtwc_cl, tmp_path):
    """clara_256_lz4.arrow as test_cli_ipc_stream runs it, four OpenMP threads (the stream's price counts its decoding
    threads): with no ram_limit the series are read into RAM, with 300,000 bytes FastCLARA streams them; both give
    dtwc_cl's labels, medoids and cost. The streamed result holds no series: score and _distance_matrix raise
    InvalidInput naming the stream, and save writes dtwc_cl's labels and medoids files, byte for byte, and the report."""
    ipc = _READER / "clara_256_lz4.arrow"
    env = {**os.environ, "OMP_NUM_THREADS": "4"}
    code = r"""
import json, sys
import dtwcpp
from dtwcpp import _dtwcpp_core as core
out = {}
for limit in (0, 300000):
    config = core.Config()
    for key, value in (("n_clusters", 3), ("method", "clara"), ("sample_size", 8), ("n_samples", 2),
                       ("seed", 42), ("device", "cpu"), ("name", "t"), ("ram_limit", limit)):
        setattr(config, key, value)
    result = core._run(config, sys.argv[1])
    run = {"labels": result.labels.tolist(), "medoids": result.medoids.tolist(), "cost": result.cost}
    for name, call in (("score", lambda: result.score("silhouette")), ("matrix", lambda: result._distance_matrix())):
        try:
            call()
            run[name] = "returned"
        except dtwcpp.InvalidInput as error:
            run[name] = str(error)
    result.save(sys.argv[2] + "/" + str(limit))
    out[limit] = run
print(json.dumps(out))
"""
    run = subprocess.run([sys.executable, "-c", code, str(ipc), str(tmp_path / "py")], capture_output=True,
                         text=True, env=env, check=False)
    assert run.returncode == 0, run.stderr
    python = json.loads(run.stdout)
    for limit, route in (("0", "ram"), ("300000", "stream")):
        out = tmp_path / route
        stdout, cli = _cli(dtwc_cl, ["-i", str(ipc), "-o", str(out), "--name", "t", "--method", "clara",
                                     "--n-clusters", "3", "--sample-size", "8", "--n-samples", "2", "--seed", "42",
                                     "--device", "cpu", "--ram-limit", limit, "--verbose"], env=env)
        assert ("Arrow IPC walk selected streaming: 256 series" in stdout) == (route == "stream")
        got = python[limit]
        assert got["labels"] == _column(out / "t_labels.csv", 1)
        assert got["medoids"] == _column(out / "t_medoids.csv", 1)
        assert got["cost"] == float(cli["Total cost"])
        saved = tmp_path / "py" / limit
        for table in ("labels", "medoids"):
            assert (saved / f"t_{table}.csv").read_bytes() == (out / f"t_{table}.csv").read_bytes()
        assert (saved / "t_report.txt").stat().st_size > 0
    assert python["0"]["score"] == "returned" and python["0"]["matrix"] == "returned"
    for name, call in (("score", "score"), ("matrix", "distance_matrix")):
        assert python["300000"][name].startswith(f"Result::{call}: the run streamed its 256 series")


def test_no_report_leaves_no_report(tmp_path):
    """Config.no_report inverts the run's report, as --no-report: no text and no <name>_report.txt."""
    series = dtwcpp.Data([[0.0, 1.0], [0.5, 1.5], [9.0, 8.0], [9.5, 8.5]], ["a", "b", "c", "d"])
    config = _config(2, "pam", name="r")
    assert config.no_report is False
    reported = core._run(config, series)
    config.no_report = True
    silent = core._run(config, series)

    assert reported.report.startswith("DTWC++ run report")
    assert silent.report == ""
    reported.save(tmp_path / "on")
    silent.save(tmp_path / "off")
    assert (tmp_path / "on" / "r_report.txt").read_text(encoding="utf-8") == reported.report
    assert not (tmp_path / "off" / "r_report.txt").exists() and (tmp_path / "off" / "r_labels.csv").exists()


def test_a_float32_matrix_is_float32_at_its_stored_width():
    """matrix_dtype float32 stores each distance rounded to a float: _distance_matrix is that array, float32, the
    float64 run's matrix rounded; float64 stays float64."""
    rows = np.random.default_rng(3).standard_normal((9, 14))
    series = dtwcpp.Data(rows.tolist(), [str(i) for i in range(9)])
    wide = core._run(_config(2, "pam"), series)._distance_matrix()
    narrow = core._run(_config(2, "pam", matrix_dtype="float32"), series)._distance_matrix()

    assert wide.dtype == np.float64 and narrow.dtype == np.float32 and narrow.shape == (9, 9)
    assert np.array_equal(narrow, wide.astype(np.float32))


def test_the_seam_hands_solve_the_runs_own_problem(monkeypatch):
    """external=True: MIP on HiGHS and LR-core's kelley root call dtwcpp._mip.solve(prob) with the run's own Problem,
    its series set, and its ClusteringResult is the run's; PAM never calls it, nor does a run without external. The
    Problem outlives the run's Result: the object solve kept co-owns it."""
    calls = []

    def solve(prob):
        calls.append(prob)
        return _clustering([0, 0, 1, 1, 1], [1, 3], 2.5)

    _install_solve(monkeypatch, solve)
    names = ["a", "b", "c", "d", "e"]
    series = dtwcpp.Data([[0.0, 1.0], [0.2, 1.1], [7.0, 6.0], [7.2, 6.1], [6.9, 6.2]], names)

    mip = core._run(_config(2, "mip", name="seam"), series, external=True)
    assert len(calls) == 1 and calls[0].size == 5 and calls[0].method == dtwcpp.Method.MIP
    assert (mip.labels.tolist(), mip.medoids.tolist(), mip.cost) == ([0, 0, 1, 1, 1], [1, 3], 2.5)
    kelley = core._run(_config(2, "lrcore", lr_root="kelley"), series, external=True)
    assert len(calls) == 2 and calls[1].method == dtwcpp.Method.LRCore and kelley.cost == 2.5
    core._run(_config(2, "pam"), series, external=True)
    try:
        core._run(_config(2, "mip"), series)  # linked HiGHS solves it; without, SolverError
    except dtwcpp.SolverError:
        pass
    assert len(calls) == 2

    kept = calls[0]
    del mip, kelley
    calls.clear()
    gc.collect()
    assert kept.size == 5 and kept.series_name(4) == "e" and kept.labels().tolist() == [0, 0, 1, 1, 1]


@pytest.mark.parametrize("error", [dtwcpp.SolverError("highspy failed (fake)"), KeyError("fake")])
def test_an_exception_solve_raises_reaches_the_caller_as_itself(monkeypatch, error):
    """The exception crosses the C++ run and is raised again by _run: the same object, so SolverError stays
    SolverError and any other keeps its type."""
    def solve(prob):
        raise error

    _install_solve(monkeypatch, solve)
    series = dtwcpp.Data([[0.0], [1.0], [5.0]], ["a", "b", "c"])
    with pytest.raises(type(error)) as caught:
        core._run(_config(2, "mip"), series, external=True)
    assert caught.value is error


def test_the_gil_is_free_while_the_run_runs(monkeypatch, tmp_path):
    """While _run reads its file in C++, another Python thread runs: it sees this thread's frame in the _run call
    before solve starts. Holding the GIL from the call to solve, this thread would let it see none: no line between
    them yields the GIL (the call's arguments are locals and constants)."""
    csv = tmp_path / "long.csv"
    np.savetxt(csv, np.random.default_rng(1).standard_normal((20, 10000)), delimiter=",")
    state = {"solve": False}

    def solve(prob):
        state["solve"] = True
        return _clustering([0] * 10 + [1] * 10, [0, 10], 1.0)

    _install_solve(monkeypatch, solve)
    main, code, seen = threading.get_ident(), sys._getframe().f_code, []
    stop = threading.Event()

    def watch():
        while not stop.is_set():
            before = state["solve"]
            frame = sys._current_frames().get(main)
            inside = frame is not None and frame.f_code is code and frame.f_lineno == line
            if inside and not before and not state["solve"]:
                seen.append(1)

    run, config, path = core._run, _config(2, "mip"), str(csv)
    watcher = threading.Thread(target=watch)
    line = sys._getframe().f_lineno + 3  # the line of the _run call
    watcher.start()
    try:
        result = run(config, path, external=True)
    finally:
        stop.set()
        watcher.join()
    assert state["solve"] and result.cost == 1.0
    assert seen, "no other thread ran while _run read its file: the GIL was held"
