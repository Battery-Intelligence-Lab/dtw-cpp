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


_UINT64_MAX = (1 << 64) - 1
_CLI_INT_MAX = (1 << 31) - 1
_CLI_UINT_MAX = (1 << 32) - 1


def _validate_restart_schedule(n_init, seed):
    """Validate and normalize the deterministic PAM restart schedule."""
    if isinstance(n_init, (bool, np.bool_)) or not isinstance(
        n_init, (int, np.integer)
    ):
        raise TypeError("n_init must be an integer")
    n_init = int(n_init)
    if n_init < 1:
        raise ValueError("n_init must be at least 1")
    if n_init > _CLI_INT_MAX:
        raise ValueError("n_init exceeds the dtwc_cl int range")

    if seed is None:
        return n_init, None
    if isinstance(seed, (bool, np.bool_)) or not isinstance(
        seed, (int, np.integer)
    ):
        raise TypeError("seed must be an integer or None")
    seed = int(seed)
    if not 0 <= seed <= _UINT64_MAX:
        raise ValueError("seed must fit in uint64")
    if n_init - 1 > _UINT64_MAX - seed:
        raise ValueError("seed + n_init - 1 overflows uint64")
    if seed > _CLI_UINT_MAX:
        raise ValueError("seed exceeds the dtwc_cl unsigned range")
    return n_init, seed


def _normalize_choice(name, value, aliases):
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    normalized = aliases.get(value.lower())
    if normalized is None:
        raise ValueError(
            f"unsupported {name}={value!r}; expected one of "
            f"{sorted(set(aliases.values()))}"
        )
    return normalized


def _normalize_finite_real(name, value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{name} must be a real number")
    value = float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def _validate_remote_configuration(
    *, device, max_iter, variant, wdtw_g, adtw_penalty, msm_c,
    twe_nu, twe_lambda, mv_mode, missing_strategy, metric,
):
    """Normalize the exact CLI configuration and reject silent substitutions."""
    if isinstance(max_iter, (bool, np.bool_)) or not isinstance(
        max_iter, (int, np.integer)
    ):
        raise TypeError("max_iter must be an integer")
    max_iter = int(max_iter)
    if not 1 <= max_iter <= _CLI_INT_MAX:
        raise ValueError("max_iter must be in the dtwc_cl positive int range")

    cuda_match = (
        re.fullmatch(r"cuda:([0-9]+)", device.lower())
        if isinstance(device, str) else None
    )
    if cuda_match is not None:
        if int(cuda_match.group(1)) > _CLI_INT_MAX:
            raise ValueError("device CUDA ordinal exceeds the dtwc_cl int range")
        device = device.lower()
    else:
        device = _normalize_choice(
            "device", device, {"cpu": "cpu", "cuda": "cuda"},
        )
    variant = _normalize_choice(
        "variant", variant,
        {
            "standard": "standard", "ddtw": "ddtw", "wdtw": "wdtw",
            "adtw": "adtw", "msm": "msm", "twe": "twe",
        },
    )
    mv_mode = _normalize_choice(
        "mv_mode", mv_mode,
        {"dependent": "dependent", "independent": "independent"},
    )
    missing_strategy = _normalize_choice(
        "missing_strategy", missing_strategy,
        {
            "error": "error", "zero_cost": "zero_cost",
            "zero-cost": "zero_cost", "zerocost": "zero_cost",
            "arow": "arow", "interpolate": "interpolate",
        },
    )
    metric = _normalize_choice(
        "metric", metric,
        {
            "l1": "l1", "squared_euclidean": "squared_euclidean",
            "sqeuclidean": "squared_euclidean", "l2sq": "squared_euclidean",
        },
    )

    if mv_mode == "independent" and (
        variant != "standard" or missing_strategy != "error"
    ):
        raise ValueError(
            "mv_mode='independent' requires variant='standard' and "
            "missing_strategy='error'"
        )
    if variant != "standard" and missing_strategy != "error":
        raise ValueError(
            "variant/missing_strategy combination is unsupported: non-Error "
            "missing handling would replace the requested variant"
        )

    is_cuda = device == "cuda" or device.startswith("cuda:")
    if is_cuda and variant != "standard":
        raise ValueError("remote CUDA supports variant='standard' only")
    if is_cuda and missing_strategy != "error":
        raise ValueError("remote CUDA does not support missing_strategy")
    if is_cuda and mv_mode != "dependent":
        raise ValueError("remote CUDA does not support mv_mode='independent'")
    if not is_cuda and metric != "l1":
        raise ValueError(
            "metric='squared_euclidean' is unsupported by the remote CPU CLI"
        )

    return {
        "device": device,
        "max_iter": max_iter,
        "variant": variant,
        "wdtw_g": _normalize_finite_real("wdtw_g", wdtw_g),
        "adtw_penalty": _normalize_finite_real("adtw_penalty", adtw_penalty),
        "msm_c": _normalize_finite_real("msm_c", msm_c),
        "twe_nu": _normalize_finite_real("twe_nu", twe_nu),
        "twe_lambda": _normalize_finite_real("twe_lambda", twe_lambda),
        "mv_mode": mv_mode,
        "missing_strategy": missing_strategy,
        "metric": metric,
    }


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
                       dtype="float64", skip_cols=0, n_init=1, seed=None,
                       max_iter=100, variant="standard", wdtw_g=0.05,
                       adtw_penalty=1.0, msm_c=1.0, twe_nu=0.001,
                       twe_lambda=1.0, mv_mode="dependent",
                       missing_strategy="error", metric="l1"):
    """Construct the dtwc_cl argument list (pure — does not execute).

    ``seed=None`` deliberately omits ``--seed`` so the C++ CLI remains the
    single source of truth for its default; explicit schedules use
    ``seed, seed + 1, ...`` for ``n_init`` restarts.
    """
    n_init, seed = _validate_restart_schedule(n_init, seed)
    config = _validate_remote_configuration(
        device=device, max_iter=max_iter, variant=variant, wdtw_g=wdtw_g,
        adtw_penalty=adtw_penalty, msm_c=msm_c, twe_nu=twe_nu,
        twe_lambda=twe_lambda, mv_mode=mv_mode,
        missing_strategy=missing_strategy, metric=metric,
    )
    command = [
        str(binary),
        "-i", str(input_path),
        "-k", str(k),
        "--skip-cols", str(skip_cols),
        "--dtype", dtype,
        "--method", method,
        "-d", config["device"],
        "-b", str(band),
        "--max-iter", str(config["max_iter"]),
        "--variant", config["variant"],
        "--wdtw-g", str(config["wdtw_g"]),
        "--adtw-penalty", str(config["adtw_penalty"]),
        "--msm-c", str(config["msm_c"]),
        "--twe-nu", str(config["twe_nu"]),
        "--twe-lambda", str(config["twe_lambda"]),
        "--mv-mode", config["mv_mode"],
        "--missing-strategy", config["missing_strategy"],
        "--metric", config["metric"],
        "--name", name,
        "-o", str(output_dir),
        "--n-init", str(n_init),
        "-v",
    ]
    if seed is not None:
        command.extend(["--seed", str(seed)])
    return command


def find_dtwc_binary(root):
    """Return a path to a built dtwc_cl binary under ``root``, or ``None``.

    Prefers build-tree binaries (including nested ``build/*/bin`` verification
    trees) over a possibly-stale top-level ``bin/``; within a group, the most
    recently modified wins.
    """
    _skip = (".pdb", ".ipdb", ".iobj", ".recipe", ".idx", ".obj", ".lib")
    pattern_groups = (
        (
            os.path.join(root, "build", "bin", "dtwc_cl*"),
            os.path.join(root, "build", "*", "bin", "dtwc_cl*"),
            os.path.join(root, "build*", "bin", "dtwc_cl*"),
        ),
        (os.path.join(root, "bin", "dtwc_cl*"),),
    )
    for patterns in pattern_groups:
        cands = {
            p
            for pat in patterns
            for p in glob.glob(pat)
            if os.path.isfile(p) and not p.endswith(_skip)
            and (p.endswith(".exe") or os.path.splitext(p)[1] == "")
        }
        if cands:
            return max(cands, key=os.path.getmtime)
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
                       band=-1, skip_cols=0, name="dtwc_job", upload=True,
                       n_init=1, seed=None, max_iter=100, variant="standard",
                       wdtw_g=0.05, adtw_penalty=1.0, msm_c=1.0,
                       twe_nu=0.001, twe_lambda=1.0, mv_mode="dependent",
                       missing_strategy="error", metric="l1"):
        n_init, seed = _validate_restart_schedule(n_init, seed)
        config = _validate_remote_configuration(
            device=device, max_iter=max_iter, variant=variant, wdtw_g=wdtw_g,
            adtw_penalty=adtw_penalty, msm_c=msm_c, twe_nu=twe_nu,
            twe_lambda=twe_lambda, mv_mode=mv_mode,
            missing_strategy=missing_strategy, metric=metric,
        )
        res = self._run("submit-cluster", input_tsv, str(k), method,
                        config["device"],
                        str(band), name, str(skip_cols), "1" if upload else "0",
                        str(n_init), "" if seed is None else str(seed),
                        str(config["max_iter"]), config["variant"],
                        str(config["wdtw_g"]), str(config["adtw_penalty"]),
                        str(config["msm_c"]), str(config["twe_nu"]),
                        str(config["twe_lambda"]), config["mv_mode"],
                        config["missing_strategy"], config["metric"])
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
                   timeout_seconds=86400, repo_root=None, runner=None,
                   n_init=1, seed=None, max_iter=100, variant="standard",
                   wdtw_g=0.05, adtw_penalty=1.0, msm_c=1.0,
                   twe_nu=0.001, twe_lambda=1.0, mv_mode="dependent",
                   missing_strategy="error", metric="l1"):
    """Offload clustering to a SLURM cluster and return labels in input order.

    ``source`` is either an in-memory list of series (serialized + uploaded) or a
    path string interpreted **on the cluster** (pre-staged data — never read or
    uploaded locally, so it scales to data too large to hold on a laptop).

    ``n_init`` and ``seed`` are carried unchanged to the remote CLI. When seed is
    omitted, the remote CLI's own default supplies the first restart seed.

    Requires a configured ``.env`` at the repo root and ssh + rsync (Git Bash on
    Windows). The build must already exist on the cluster — run
    ``bash scripts/slurm/slurm_remote.sh build htc-cpu`` once beforehand.

    NOTE: the remote submission cannot be verified on a dev laptop; run on ARC.
    """
    n_init, seed = _validate_restart_schedule(n_init, seed)
    config = _validate_remote_configuration(
        device=device, max_iter=max_iter, variant=variant, wdtw_g=wdtw_g,
        adtw_penalty=adtw_penalty, msm_c=msm_c, twe_nu=twe_nu,
        twe_lambda=twe_lambda, mv_mode=mv_mode,
        missing_strategy=missing_strategy, metric=metric,
    )
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
    job_id = runner.submit_cluster(input_arg, n_clusters, method=method,
                                   device=config["device"],
                                   band=band, skip_cols=skip_cols, name=name,
                                   upload=upload, n_init=n_init, seed=seed,
                                   max_iter=config["max_iter"],
                                   variant=config["variant"],
                                   wdtw_g=config["wdtw_g"],
                                   adtw_penalty=config["adtw_penalty"],
                                   msm_c=config["msm_c"],
                                   twe_nu=config["twe_nu"],
                                   twe_lambda=config["twe_lambda"],
                                   mv_mode=config["mv_mode"],
                                   missing_strategy=config["missing_strategy"],
                                   metric=config["metric"])
    runner.wait(job_id, poll_seconds=poll_seconds, timeout_seconds=timeout_seconds)
    labels_csv = runner.download_labels(name)
    return parse_labels_csv(labels_csv, n)
