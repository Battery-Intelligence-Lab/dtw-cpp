"""
@file _hpc.py
@brief Offload a clustering job to a SLURM cluster (e.g. Oxford ARC) and bring labels back.
@details
    The transport (ssh / rsync / sbatch) is delegated to the tested shell wrapper
    scripts/slurm/slurm_remote.sh. This module owns the Python side:
      1. serialize the series to a TSV the cluster's dtwc_cl reads,
      2. drive the wrapper (submit -> poll -> download),
      3. parse the downloaded NAME_labels.csv back into per-series cluster labels.

    The pure helpers (serialize, parse, command build, binary discovery) are unit
    tested AND validated end-to-end against a local dtwc_cl binary in
    tests/python/test_hpc.py. The remote submission itself cannot be exercised on a
    dev laptop — it must be verified on the actual cluster.
@author Volkan Kumtepeli
"""
import csv
import glob
import os
import re
import shutil
import subprocess
import time

import numpy as np


# ─────────────────────────────────────────────────────────────────────────
# Pure helpers (unit tested)
# ─────────────────────────────────────────────────────────────────────────
def write_series_tsv(series, path):
    """Write a list of 1-D series to a tab-delimited file, one series per row.

    Pure data — no header, no id column — matching ``dtwc_cl --skip-cols 0``.
    Ragged series are allowed (rows may differ in length).
    """
    path = str(path)
    with open(path, "w", newline="") as f:
        for s in series:
            f.write("\t".join(f"{float(v):.10g}" for v in s))
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
    labels = np.empty(n, dtype=int)
    for i in range(n):
        labels[i] = mapping[str(i + 1)]          # KeyError if a series is missing
    return labels


def build_dtwc_command(binary, input_path, k, name, output_dir, *,
                       method="pam", device="cpu", band=-1,
                       dtype="float64", skip_cols=0):
    """Construct the dtwc_cl argument list (pure — does not execute)."""
    return [
        str(binary),
        "-i", str(input_path),
        "-k", str(k),
        "--skip-cols", str(skip_cols),
        "--dtype", dtype,
        "--method", method,
        "-d", device,
        "-b", str(band),
        "--name", name,
        "-o", str(output_dir),
        "-v",
    ]


def find_dtwc_binary(root):
    """Return a path to a built dtwc_cl binary under ``root``, or ``None``.

    Prefers ``build*/bin`` (current builds) over a possibly-stale top-level
    ``bin/``; within a group, the most recently modified wins.
    """
    _skip = (".pdb", ".ipdb", ".iobj", ".recipe", ".idx", ".obj", ".lib")
    for pat in (os.path.join(root, "build*", "bin", "dtwc_cl*"),
                os.path.join(root, "bin", "dtwc_cl*")):
        cands = [p for p in glob.glob(pat)
                 if os.path.isfile(p) and not p.endswith(_skip)
                 and (p.endswith(".exe") or os.path.splitext(p)[1] == "")]
        if cands:
            cands.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return cands[0]
    return None


# ─────────────────────────────────────────────────────────────────────────
# Remote orchestration (drives scripts/slurm/slurm_remote.sh — NOT laptop-testable)
# ─────────────────────────────────────────────────────────────────────────
class SlurmRemoteRunner:
    """Thin wrapper around scripts/slurm/slurm_remote.sh (ssh + rsync + sbatch)."""

    def __init__(self, repo_root):
        self.repo_root = str(repo_root)
        self.wrapper = os.path.join(self.repo_root, "scripts", "slurm", "slurm_remote.sh")

    def preflight(self):
        if shutil.which("bash") is None:
            raise RuntimeError(
                "bash not found. On Windows install Git Bash (ships ssh + rsync)."
            )
        if not os.path.isfile(self.wrapper):
            raise RuntimeError(f"SLURM wrapper not found: {self.wrapper}")
        if not os.path.isfile(os.path.join(self.repo_root, ".env")):
            raise RuntimeError(
                "Missing .env at repo root. Copy scripts/slurm/env.example -> .env "
                "and set SLURM_USER / SLURM_HOST / SLURM_REMOTE_BASE."
            )

    def _run(self, *args):
        return subprocess.run(["bash", self.wrapper, *args],
                              cwd=self.repo_root, capture_output=True, text=True)

    def submit_cluster(self, input_tsv, k, *, method="pam", device="cpu",
                       band=-1, skip_cols=0, name="dtwc_job", upload=True):
        res = self._run("submit-cluster", input_tsv, str(k), method, device,
                        str(band), name, str(skip_cols), "1" if upload else "0")
        out = (res.stdout or "") + (res.stderr or "")
        m = re.search(r"Job ID:\s*(\d+)", out)
        if not m:
            raise RuntimeError(
                f"submit-cluster returned no Job ID (exit {res.returncode}). Check that "
                f".env has the right SLURM_USER/SLURM_HOST, that ssh to the cluster works, "
                f"and that a build exists there ('slurm_remote.sh build').\n"
                f"Wrapper output:\n{out or '<empty>'}"
            )
        return m.group(1)

    def wait(self, job_id, *, poll_seconds=20, timeout_seconds=86400):
        waited = 0
        while job_id in self._run("status").stdout:
            if waited >= timeout_seconds:
                raise TimeoutError(f"job {job_id} still queued after {waited}s")
            time.sleep(poll_seconds)
            waited += poll_seconds

    def download_labels(self, name):
        self._run("download")
        hits = glob.glob(os.path.join(self.repo_root, "results", "slurm", "**",
                                      f"{name}_labels.csv"), recursive=True)
        if not hits:
            raise FileNotFoundError(
                f"{name}_labels.csv not found under results/slurm/ after download"
            )
        return max(hits, key=os.path.getmtime)


def cluster_on_hpc(source, n_clusters, *, method="pam", device="cpu", band=-1,
                   skip_cols=0, name="dtwc_job", poll_seconds=20,
                   timeout_seconds=86400, repo_root=None, runner=None):
    """Offload clustering to a SLURM cluster and return labels in input order.

    ``source`` is either an in-memory list of series (serialized + uploaded) or a
    path string interpreted **on the cluster** (pre-staged data — never read or
    uploaded locally, so it scales to data too large to hold on a laptop).

    Requires a configured ``.env`` at the repo root and ssh + rsync (Git Bash on
    Windows). The build must already exist on the cluster — run
    ``bash scripts/slurm/slurm_remote.sh build htc-cpu`` once beforehand.

    NOTE: the remote submission cannot be verified on a dev laptop; run on ARC.
    """
    repo_root = repo_root or os.environ.get("DTWC_REPO_ROOT", os.getcwd())
    rundir = os.path.join(repo_root, "results", "hpc", name)
    os.makedirs(rundir, exist_ok=True)

    if isinstance(source, (str, os.PathLike)):
        # Cluster-side path: pass through, no local read, no upload.
        input_arg, upload, n = str(source).replace(os.sep, "/"), False, None
    else:
        # In-memory series: serialize to a repo-relative TSV and upload it.
        # (Repo-relative so Git Bash rsync doesn't read 'C:/...' as 'host:path'.)
        n = len(source)
        tsv = write_series_tsv(source, os.path.join(rundir, "input.tsv"))
        input_arg, upload = os.path.relpath(tsv, repo_root).replace(os.sep, "/"), True

    runner = runner or SlurmRemoteRunner(repo_root)
    runner.preflight()
    job_id = runner.submit_cluster(input_arg, n_clusters, method=method, device=device,
                                   band=band, skip_cols=skip_cols, name=name, upload=upload)
    runner.wait(job_id, poll_seconds=poll_seconds, timeout_seconds=timeout_seconds)
    labels_csv = runner.download_labels(name)
    return parse_labels_csv(labels_csv, n)
