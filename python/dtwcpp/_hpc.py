"""
@file _hpc.py
@brief Run a clustering on a SLURM cluster (e.g. Oxford ARC) and bring the labels back.
@details
    A run crosses as one file, job.toml: one ``key = value`` line per key, in the
    grammar ``dtwc_cl --config`` reads, and C++ checks the keys on the cluster as
    it checks any config file. The transport (ssh / rsync / sbatch) is the shell
    wrapper _slurm/slurm_remote.sh, package data found with importlib.resources,
    so an installed wheel needs no source checkout. This module
      1. writes the run directory: job.toml and, for series in memory, input.tsv,
      2. drives the wrapper: submit-job, then status until the job has left the
         queue, then download-cluster,
      3. parses the downloaded NAME_labels.csv into labels in input order.
    The cluster is out of a laptop's reach: tests/python/test_hpc.py runs the
    wrapper against local stand-ins for ssh, rsync and sbatch, and job.toml
    through a local dtwc_cl.
@author Volkan Kumtepeli
"""
import csv
import importlib.resources
import os
import re
import shutil
import subprocess
import tempfile
import time

import numpy as np

# A string as dtwc_cl's config writer quotes it (dtwc/cli/config.cpp
# config_value): these escapes, any other control byte as \u00XX, UTF-8 kept.
_ESCAPES = {"\b": "\\b", "\t": "\\t", "\n": "\\n", "\f": "\\f", "\r": "\\r",
            '"': '\\"', "\\": "\\\\"}


def _toml_value(value):
    """A Config value as a job.toml token: a bool, an integer or a float bare
    (``repr``, the shortest text that reads back to the same double), a string
    quoted."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return '"' + "".join(
        _ESCAPES.get(c) or (f"\\u{ord(c):04x}" if c < " " or c == "\x7f" else c)
        for c in value) + '"'


def _normalize_wait_controls(poll_seconds, timeout_seconds):
    """Validate polling controls before a remote job can be submitted."""
    normalized = []
    for name, value, minimum, strict in (
        ("poll_seconds", poll_seconds, 0.0, True),
        ("timeout_seconds", timeout_seconds, 0.0, True),
    ):
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"{name} must be a finite number")
        value = float(value)
        if not np.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if value < minimum or (strict and value == minimum):
            comparator = "greater than zero" if strict else "non-negative"
            raise ValueError(f"{name} must be {comparator}")
        normalized.append(value)
    return tuple(normalized)


def write_series_tsv(series, path):
    """Write a list of 1-D series to a tab-delimited file, one series per row.

    Pure data — no header, no id column — matching ``dtwc_cl --skip-cols 0``.
    Ragged series are allowed (rows may differ in length). Values are written
    with ``repr``, the shortest text that reads back to the same double, so an
    HPC run clusters the same numbers as a local one (``:.10g`` rounded them).
    """
    path = str(path)
    with open(path, "w", newline="") as f:
        for s in series:
            f.write("\t".join(repr(float(v)) for v in s))
            f.write("\n")
    return path


def parse_labels_csv(path, n=None):
    """Parse dtwc_cl's ``NAME_labels.csv`` into labels in INPUT order.

    The file has header ``name,cluster``; dtwc_cl names batch-row series ``1..N``
    (1-based). Row order is not guaranteed (it may be lexically sorted), so the
    mapping is by name: ``labels[i] = cluster_of[str(i + 1)]``.

    ``n`` is the expected number of series; when ``None`` (e.g. an HPC path source
    whose length isn't known locally) it is inferred from the file row count.
    Raises KeyError if any series ``1..n`` is absent from the file.
    """
    mapping = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            mapping[str(row["name"]).strip()] = int(row["cluster"])
    if n is None:
        n = len(mapping)
    labels = np.empty(n, dtype=np.int64)
    for i in range(n):
        labels[i] = mapping[str(i + 1)]          # KeyError if a series is missing
    return labels


def slurm_wrapper_path():
    """The SLURM wrapper shipped in this package; ``cluster_generic.slurm`` is beside it."""
    return str(importlib.resources.files("dtwcpp") / "_slurm" / "slurm_remote.sh")


def gpu_request(gpu_device=None):
    """The sbatch arguments that ask SLURM for a GPU of type ``gpu_device``
    (``None``: any GPU of compute capability 8.0 or newer), from
    _slurm/gpu_devices.txt, the table slurm_remote.sh reads too. A type below
    the CUDA floor, or one the table does not name, is a DeviceError."""
    from dtwcpp import DeviceError
    text = (importlib.resources.files("dtwcpp") / "_slurm" / "gpu_devices.txt").read_text(
        encoding="utf-8")
    rows = {fields[0]: fields[1:] for fields in map(str.split, text.splitlines())
            if len(fields) >= 3 and not fields[0].startswith("#")}
    types = ", ".join(t for t, (_, *request) in rows.items() if t != "*" and request != ["-"])
    name = "*" if gpu_device is None else str(gpu_device).lower()
    if name not in rows or (name == "*") != (gpu_device is None):
        raise DeviceError(f"gpu_device={gpu_device!r} is not a GPU type SLURM can be "
                          f"asked for: {types}.")
    capability, *request = rows[name]
    if request == ["-"]:
        raise DeviceError(f"gpu_device={gpu_device!r} has CUDA compute capability "
                          f"{capability}, below the 8.0 DTWC++ needs: ask for {types}.")
    return request


class SlurmRemoteRunner:
    """Thin wrapper around the packaged slurm_remote.sh (ssh + rsync + sbatch).

    ``repo_root`` is the project directory: it holds ``.env`` and receives
    ``results/``. It need not be a source checkout.
    """

    def __init__(self, repo_root):
        self.repo_root = os.path.abspath(str(repo_root))
        self.wrapper = slurm_wrapper_path()

    def preflight(self):
        """Refuse, before anything is written or sent, a device='hpc' this
        machine cannot drive: no bash, no wrapper, or no .env."""
        from dtwcpp import DeviceError
        if shutil.which("bash") is None:
            raise DeviceError(
                "device='hpc' needs bash: on Windows install Git Bash (ships ssh + rsync)."
            )
        if not os.path.isfile(self.wrapper):
            raise DeviceError(
                f"SLURM wrapper not found: {self.wrapper}. The dtwcpp install is "
                "incomplete; reinstall the package."
            )
        if not os.path.isfile(os.path.join(self.repo_root, ".env")):
            raise DeviceError(
                f"Missing .env in {self.repo_root} (the working directory, or "
                "DTWC_REPO_ROOT). Create it with SLURM_USER, SLURM_HOST and "
                "SLURM_REMOTE_BASE; scripts/slurm/env.example in a source "
                "checkout is a template."
            )

    def _run(self, *args, timeout=None):
        # The wrapper resolves its project directory from DTWC_REPO_ROOT; pass
        # ours so an ambient value cannot point it at another .env.
        return subprocess.run(["bash", self.wrapper, *args],
                              cwd=self.repo_root, capture_output=True, text=True,
                              env={**os.environ, "DTWC_REPO_ROOT": self.repo_root},
                              timeout=timeout)

    def submit_job(self, rundir, *, device="cpu", gpu_device=None):
        """Submit the run directory ``rundir`` (relative to the project
        directory) and return its SLURM job ID. ``device="gpu"`` asks for a GPU:
        one of type ``gpu_device``, else any the CUDA floor allows."""
        request = [] if device != "gpu" else (
            ["--gpu-device", gpu_device] if gpu_device else ["--gpu"])
        res = self._run("submit-job", rundir, *request)
        out = (res.stdout or "") + (res.stderr or "")
        if res.returncode != 0:
            raise RuntimeError(
                f"submit-job failed (exit {res.returncode}).\n"
                f"Wrapper output:\n{out or '<empty>'}"
            )
        m = re.search(r"Job ID:\s*(\d+)", out)
        if not m:
            raise RuntimeError(
                f"submit-job returned no Job ID (exit {res.returncode}). Check that "
                f".env has the right SLURM_USER/SLURM_HOST, that ssh to the cluster works, "
                f"and that a build exists there ('slurm_remote.sh build').\n"
                f"Wrapper output:\n{out or '<empty>'}"
            )
        return m.group(1)

    def wait(self, job_id, *, poll_seconds=20, timeout_seconds=86400):
        poll_seconds, timeout_seconds = _normalize_wait_controls(
            poll_seconds, timeout_seconds,
        )
        job_id = str(job_id)
        if re.fullmatch(r"[1-9][0-9]*", job_id) is None:
            raise ValueError("job_id must be a positive decimal Slurm job ID")
        deadline = time.monotonic() + timeout_seconds
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(
                    f"job {job_id} exceeded the {timeout_seconds:g}s timeout"
                )
            try:
                status = self._run("status", timeout=remaining)
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError(
                    f"status check for job {job_id} exceeded the "
                    f"{timeout_seconds:g}s timeout"
                ) from exc
            if status.returncode != 0:
                out = (status.stdout or "") + (status.stderr or "")
                raise RuntimeError(
                    f"status failed (exit {status.returncode}) while waiting for "
                    f"job {job_id}.\nWrapper output:\n{out or '<empty>'}"
                )
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"job {job_id} exceeded the {timeout_seconds:g}s timeout"
                )
            active = re.search(
                rf"(?m)^\s*{re.escape(job_id)}(?:\s|$)", status.stdout or "",
            )
            if active is None:
                return
            sleep_seconds = min(poll_seconds, deadline - time.monotonic())
            if sleep_seconds <= 0:
                raise TimeoutError(
                    f"job {job_id} exceeded the {timeout_seconds:g}s timeout"
                )
            time.sleep(sleep_seconds)

    def download_labels(self, name, job_id):
        """Download job ``job_id``'s labels; return the local ``<name>_labels.csv``."""
        job_id = str(job_id)
        if re.fullmatch(r"[1-9][0-9]*", job_id) is None:
            raise ValueError("job_id must be a positive decimal Slurm job ID")
        result = self._run("download-cluster", job_id)
        if result.returncode != 0:
            out = (result.stdout or "") + (result.stderr or "")
            raise RuntimeError(
                f"download-cluster failed (exit {result.returncode}) for job "
                f"{job_id}.\nWrapper output:\n{out or '<empty>'}"
            )
        labels = os.path.join(self.repo_root, "results", "slurm",
                              f"cluster_{job_id}", f"{name}_labels.csv")
        if not os.path.isfile(labels):
            raise FileNotFoundError(
                f"job {job_id} wrote no labels ({labels} is missing after the "
                f"download): dtwc_cl's messages are in logs/cluster_{job_id}.err "
                "under SLURM_REMOTE_BASE/src on the cluster, which "
                "'slurm_remote.sh download' fetches."
            )
        return labels


def cluster_on_hpc(data, config, keys, *, device="hpc", gpu_device=None,
                   poll_seconds=20, timeout_seconds=86400, repo_root=None,
                   runner=None):
    """Cluster ``data`` by ``config`` on a SLURM cluster; return the labels in input order.

    ``data`` is a :class:`dtwcpp.Dataset`. A path names a file on the cluster,
    read there and never here, so it is absolute; series in memory travel as
    input.tsv. job.toml holds the input, ``n-clusters``, ``name``, ``device``
    (``gpu`` for ``"hpc:gpu"``), a path's ``skip-rows``, ``skip-cols`` and
    ``delimiter`` when set, and the Config keys named in ``keys``, the ones the
    caller gave: a key not given is not written, so the cluster's dtwc_cl
    applies its own default. An ``"hpc:gpu"`` run asks SLURM for a GPU of type
    ``gpu_device`` (:func:`gpu_request`) and runs the build made for it.

    The project directory (``repo_root``, else ``$DTWC_REPO_ROOT``, else the
    working directory) holds ``.env`` and receives ``results/``. The cluster
    needs a build, made once from a source checkout with
    ``bash scripts/slurm/slurm_remote.sh upload`` and then ``build``.
    """
    from dtwcpp import InvalidInput
    gpu = device.strip().lower() == "hpc:gpu"
    if gpu_device is not None:
        gpu_device = str(gpu_device).lower()
    if gpu:
        gpu_request(gpu_device)  # an unknown or too old type: refused here
    poll_seconds, timeout_seconds = _normalize_wait_controls(
        poll_seconds, timeout_seconds,
    )
    name = config.name or data.name
    # The run's files on the cluster are <name>_labels.csv, ... in its results
    # directory, downloaded by that file name: a name must stay a file name.
    if name in ("", ".", "..") or "/" in name or "\\" in name:
        raise InvalidInput(
            f"cluster: device='hpc' names the run's files <name>_labels.csv, ..., so "
            f"name must be a file name, without '/' or '\\'; got {name!r}.")
    series = None
    if data.is_path:
        source = os.fspath(data.source).replace("\\", "/")
        if not source.startswith("/"):
            raise InvalidInput(
                f"cluster: device='hpc' reads '{source}' on the cluster; give its "
                "absolute path there.")
        items = [("input", source)]
    else:
        series = data.as_series()  # load() has dropped skip_rows and skip_cols
        # Problem::cluster()'s guards, which the cluster would meet only after the queue.
        if not series:
            raise InvalidInput("cluster: dataset is empty.")
        if config.n_clusters > len(series):
            raise InvalidInput("cluster: k must not exceed the number of series.")
        items = [("input", "input.tsv")]
    items += [("n-clusters", config.n_clusters), ("name", name),
              ("device", "gpu" if gpu else "cpu")]
    if data.is_path:
        items += [(key, value) for key, value in (
            ("skip-rows", data.skip_rows), ("skip-cols", data.skip_cols),
            ("delimiter", data.delimiter)) if value]
    items += [(key.replace("_", "-"), getattr(config, key))
              for key in keys if key != "name"]

    repo_root = repo_root or os.environ.get("DTWC_REPO_ROOT", os.getcwd())
    runner = runner or SlurmRemoteRunner(repo_root)
    runner.preflight()
    runs = os.path.join(repo_root, "results", "hpc")
    os.makedirs(runs, exist_ok=True)
    rundir = tempfile.mkdtemp(prefix="job.", dir=runs)  # one per run: no shared path
    if series is not None:
        write_series_tsv(series, os.path.join(rundir, "input.tsv"))
    with open(os.path.join(rundir, "job.toml"), "w", encoding="utf-8",
              newline="\n") as f:
        f.writelines(f"{key} = {_toml_value(value)}\n" for key, value in items)

    # Relative to the project directory, so Git Bash's rsync cannot read
    # 'C:/...' as host:path.
    job_id = runner.submit_job(
        os.path.relpath(rundir, repo_root).replace(os.sep, "/"),
        device="gpu" if gpu else "cpu", gpu_device=gpu_device)
    runner.wait(job_id, poll_seconds=poll_seconds, timeout_seconds=timeout_seconds)
    labels = runner.download_labels(name, job_id)
    return parse_labels_csv(labels, None if series is None else len(series))
