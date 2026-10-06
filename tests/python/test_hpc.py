"""
@file test_hpc.py
@brief The device='hpc' route: job.toml, the wrapper's envelope, the job script.
@details The cluster is out of reach here: the wrapper runs against local
         stand-ins for ssh, rsync, scp and sbatch, and the job against the
         local dtwc_cl (DTWC_CL_PATH), the oracle for the cluster's binary.
@author Volkan Kumtepeli
"""
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import types

import numpy as np
import pytest

import dtwcpp
from dtwcpp import _hpc


# ---------------------------------------------------------------------------
# Serialization: list-of-series -> TSV
# ---------------------------------------------------------------------------
class TestWriteSeriesTSV:
    def test_writes_one_row_per_series(self, tmp_path):
        series = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        p = tmp_path / "data.tsv"
        _hpc.write_series_tsv(series, p)
        lines = p.read_text().strip().splitlines()
        assert len(lines) == 2

    def test_roundtrips_through_loadtxt(self, tmp_path):
        rng = np.random.default_rng(0)
        series = [list(rng.standard_normal(10)) for _ in range(4)]
        p = tmp_path / "data.tsv"
        _hpc.write_series_tsv(series, p)
        back = np.loadtxt(p, delimiter="\t")
        np.testing.assert_array_almost_equal(back, np.array(series), decimal=5)

    def test_values_keep_full_precision(self, tmp_path):
        """FX-10: ``:.10g`` rounded every value, so an HPC run clustered
        different numbers from a local one."""
        series = [[0.1 + 0.2, 1.0 / 3.0, 5e-324, -2.5e300], [123456.78901234567]]
        p = tmp_path / "data.tsv"
        _hpc.write_series_tsv(series, p)
        back = [[float(v) for v in line.split("\t")]
                for line in p.read_text().splitlines()]
        assert back == series


# ---------------------------------------------------------------------------
# Label parsing: dtwc_cl's NAME_labels.csv -> labels in input order
# ---------------------------------------------------------------------------
class TestParseLabelsCSV:
    def test_maps_one_based_names_to_input_order(self, tmp_path):
        p = tmp_path / "j_labels.csv"
        p.write_text("name,cluster\n1,0\n2,0\n3,1\n")
        labels = _hpc.parse_labels_csv(p, n=3)
        np.testing.assert_array_equal(labels, [0, 0, 1])

    def test_robust_to_lexical_row_order(self, tmp_path):
        """dtwc_cl may emit rows lexically sorted (1,10,11,2,...); mapping is by
        name, so input order must be recovered regardless of row order."""
        p = tmp_path / "j_labels.csv"
        # 12 series, rows deliberately scrambled, series 7..12 are cluster 1
        rows = "\n".join(f"{i},{0 if i <= 6 else 1}" for i in [1, 10, 11, 12, 2, 3, 4, 5, 6, 7, 8, 9])
        p.write_text("name,cluster\n" + rows + "\n")
        labels = _hpc.parse_labels_csv(p, n=12)
        np.testing.assert_array_equal(labels, [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

    def test_raises_when_a_series_is_missing(self, tmp_path):
        p = tmp_path / "j_labels.csv"
        p.write_text("name,cluster\n1,0\n2,0\n")        # series 3 missing
        with pytest.raises((KeyError, ValueError)):
            _hpc.parse_labels_csv(p, n=3)


# ---------------------------------------------------------------------------
# job.toml: the run as one file, in the grammar dtwc_cl --config reads
# ---------------------------------------------------------------------------
class _Submitted(Exception):
    """What a stand-in for the wrapper received: the run directory, its files
    and submit_job's keywords."""


def _submit(monkeypatch, tmp_path, run):
    """Call ``run``, a cluster() or fit() on device 'hpc', up to the submission."""
    class Runner:
        def __init__(self, repo_root):
            pass

        def preflight(self):
            pass

        def submit_job(self, rundir, **request):
            files = {path.name: path.read_text(encoding="utf-8")
                     for path in (tmp_path / rundir).iterdir()}
            raise _Submitted(rundir, files, request)

    monkeypatch.setattr(_hpc, "SlurmRemoteRunner", Runner)
    monkeypatch.setenv("DTWC_REPO_ROOT", str(tmp_path))
    with pytest.raises(_Submitted) as caught:
        run()
    return caught.value.args


class TestJobToml:
    def test_a_path_names_a_file_on_the_cluster(self, monkeypatch, tmp_path):
        """The file is read there, never here, with load()'s reader keys; counts
        cross as index_t; C++ canonicalised the keys (hclust is hierarchical),
        and CPU squared_euclidean crosses, which the 2.0 previews refused."""
        source = dtwcpp.load("/cluster/données.csv", skip_rows=2,
                             skip_cols=np.int64(1 << 40), delimiter=";")

        def run():
            dtwcpp.cluster(source, k=np.int64(1 << 40), device="hpc",
                           max_iter=np.int64((1 << 31) - 1), method="hclust",
                           metric="squared_euclidean")

        _, files, request = _submit(monkeypatch, tmp_path, run)
        assert files == {"job.toml": (
            'input = "/cluster/données.csv"\n'
            "n-clusters = 1099511627776\n"
            'name = "données"\n'
            'device = "cpu"\n'
            "skip-rows = 2\n"
            "skip-cols = 1099511627776\n"
            'delimiter = ";"\n'
            "max-iter = 2147483647\n"
            'method = "hierarchical"\n'
            'metric = "squared_euclidean"\n')}
        assert request == {"device": "cpu", "gpu_device": None}
        # A relative path has no meaning on the cluster: refused before a run
        # directory is written.
        with pytest.raises(dtwcpp.InvalidInput, match="absolute path"):
            dtwcpp.cluster("data/series.tsv", k=2, device="hpc")
        assert len(list((tmp_path / "results/hpc").iterdir())) == 1

    def test_series_in_memory_travel_as_input_tsv(self, monkeypatch, tmp_path):
        """load() cut the columns, so the file is not cut again; hpc:gpu computes
        on the GPU gpu_device names; two runs of one name never share a directory."""
        source = dtwcpp.load([[9.0, 0.1, 2.0], [9.0, 3.0, 4.0]], skip_cols=1)

        def run():
            dtwcpp.cluster(source, k=2, device="hpc:gpu", gpu_device="L40S",
                           name="café", wdtw_g=0.17)

        first, files, request = _submit(monkeypatch, tmp_path, run)
        assert files == {
            "input.tsv": "0.1\t2.0\n3.0\t4.0\n",
            "job.toml": ('input = "input.tsv"\n'
                         "n-clusters = 2\n"
                         'name = "café"\n'
                         'device = "gpu"\n'
                         "wdtw-g = 0.17\n")}
        assert request == {"device": "gpu", "gpu_device": "l40s"}
        assert _submit(monkeypatch, tmp_path, run)[0] != first

    @pytest.mark.parametrize(("gpu_device", "expected"), [
        (None, ["--gres=gpu:1",
                "--constraint=gpu_cc:8.0|gpu_cc:8.6|gpu_cc:8.9|gpu_cc:9.0"]),
        ("a100", ["--gres=gpu:a100:1"]),
        ("a6000", ["--gres=gpu:1", "--constraint=gpu_cc:8.6"]),
        ("L40S", ["--gres=gpu:1", "--constraint=gpu_cc:8.9"]),
        ("h100", ["--gres=gpu:1", "--constraint=gpu_cc:9.0"]),
        ("p100", "compute capability 6.0, below the 8.0"),
        ("v100", "compute capability 7.0, below the 8.0"),
        ("rtx8000", "compute capability 7.5, below the 8.0"),
        ("rtx", "compute capability 7.5, below the 8.0"),
        ("mi210", "not a GPU type"),
        ("*", "not a GPU type"),
    ])
    def test_gpu_device_is_an_arc_request(self, gpu_device, expected):
        """The table as data: ARC's gres type where it documents one (A100),
        else any GPU of the type's compute capability (gpu_cc:); no type asks
        for every capability at or above the floor, so the job cannot land on
        a refused V100; a type below the floor is a DeviceError naming it."""
        if isinstance(expected, list):
            assert _hpc.gpu_request(gpu_device) == expected
        else:
            with pytest.raises(dtwcpp.DeviceError, match=expected):
                _hpc.gpu_request(gpu_device)

    def test_gpu_device_belongs_to_an_hpc_gpu_run(self, monkeypatch, tmp_path):
        """A local device, or hpc (a CPU run), refuses it as InvalidInput; a type
        the table refuses fails before anything is written."""
        for device in ("cpu", "hpc"):
            with pytest.raises(dtwcpp.InvalidInput, match="gpu_device"):
                dtwcpp.cluster([[0.0], [1.0]], k=2, device=device, gpu_device="a100")
        monkeypatch.setenv("DTWC_REPO_ROOT", str(tmp_path))
        with pytest.raises(dtwcpp.DeviceError, match="below the 8.0"):
            dtwcpp.cluster([[0.0], [1.0]], k=2, device="hpc:gpu", gpu_device="v100")
        assert not (tmp_path / "results").exists()

    def test_the_estimator_sends_its_parameters(self, monkeypatch, tmp_path):
        """DTWClustering(device='hpc') sends every parameter, its own defaults
        included (method pam; random_state None is seed 42), and batch_size,
        which the positional transport refused."""
        model = dtwcpp.DTWClustering(n_clusters=2, n_init=2, variant="twe",
                                     twe_nu=0.02, batch_size=8, device="hpc")

        def run():
            model.fit([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])

        _, files, _ = _submit(monkeypatch, tmp_path, run)
        assert files["job.toml"] == (
            'input = "input.tsv"\n'
            "n-clusters = 2\n"
            'name = "dataset"\n'
            'device = "cpu"\n'
            'method = "pam"\n'
            "max-iter = 100\n"
            "n-init = 2\n"
            "batch-size = 8\n"
            "seed = 42\n"
            'mv-mode = "dependent"\n'
            'variant = "twe"\n'
            "band = -1\n"
            'metric = "l1"\n'
            'missing-strategy = "error"\n'
            "wdtw-g = 0.05\n"
            "adtw-penalty = 1.0\n"
            "msm-c = 1.0\n"
            "twe-nu = 0.02\n"
            "twe-lambda = 1.0\n")


# ---------------------------------------------------------------------------
# The wrapper and the job script, behind local stand-ins for the cluster
# ---------------------------------------------------------------------------
def _bash_path(path):
    """Translate an absolute Windows path for Git Bash or WSL bash."""
    path = Path(path).resolve().as_posix()
    if os.name != "nt" or len(path) < 3 or path[1:3] != ":/":
        return path
    uname = subprocess.run(
        ["bash", "-c", "uname -s"], check=True, capture_output=True, text=True,
    ).stdout.strip().lower()
    drive, suffix = path[0].lower(), path[2:]
    return f"/mnt/{drive}{suffix}" if uname.startswith("linux") else f"/{drive}{suffix}"


def _packaged_job():
    """The job script the packaged wrapper uploads (it ships beside it)."""
    return Path(_hpc.slurm_wrapper_path()).with_name("cluster_generic.slurm")


def _isolated_slurm_wrapper(tmp_path):
    """A checkout-shaped sandbox behind local SSH/transfer/sbatch executables.

    ``project`` holds the real scripts/slurm/ (forwarder and job files), a copy
    of the packaged python/dtwcpp/_slurm/, a dtwc/ tree and a run directory
    ``run/`` holding a job.toml, so the forwarder runs as in a clone;
    ``wrapper.parents[2]`` is that checkout, whose .env and results/ the
    wrapper uses. The remote holds the builds submit-job runs.
    """
    root = Path(__file__).resolve().parents[2]
    project = tmp_path / "project"
    shutil.copytree(root / "scripts/slurm", project / "scripts/slurm")
    shutil.copytree(
        Path(_hpc.slurm_wrapper_path()).parent, project / "python/dtwcpp/_slurm",
    )
    (project / "dtwc").mkdir()
    (project / "run").mkdir()
    (project / "run/job.toml").write_text("n-clusters = 2\n", encoding="utf-8")
    wrapper = project / "scripts/slurm/slurm_remote.sh"

    remote = tmp_path / "remote"
    for build in ("htc-cpu", "htc-gpu"):
        remote_binary = remote / f"src/build-{build}/bin/dtwc_cl"
        remote_binary.parent.mkdir(parents=True)
        remote_binary.write_text("#!/bin/sh\n", encoding="utf-8", newline="\n")
        remote_binary.chmod(0o755)
    (project / ".env").write_text(
        "SLURM_USER=test_user\n"
        "SLURM_HOST=test_host\n"
        f"SLURM_REMOTE_BASE={_bash_path(remote)}\n",
        encoding="utf-8",
    )

    fake_bin = tmp_path / "fake-bin"
    fake_bin.mkdir()
    fake_ssh = fake_bin / "ssh"
    fake_ssh.write_text(
        "#!/usr/bin/env bash\nexec sh -c \"${2:-}\"\n",
        encoding="utf-8", newline="\n",
    )
    fake_ssh.chmod(0o755)
    for transfer_name in ("rsync", "scp"):
        transfer = fake_bin / transfer_name
        transfer.write_text(
            "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8", newline="\n",
        )
        transfer.chmod(0o755)
    capture = tmp_path / "sbatch-args.txt"
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$@\" > \"$CAPTURE_SBATCH\"\n"
        "echo 12345\n",
        encoding="utf-8", newline="\n",
    )
    fake_sbatch.chmod(0o755)
    return wrapper, fake_bin, capture


class TestSlurmRunner:
    """The Python orchestration glue around slurm_remote.sh (no real cluster)."""

    def test_submit_parses_job_id(self):
        calls = []
        r = _hpc.SlurmRemoteRunner(".")
        r._run = lambda *a: calls.append(a) or types.SimpleNamespace(
            stdout="  Job ID: 98765\n", stderr="", returncode=0)
        assert r.submit_job("run", device="gpu") == "98765"
        assert calls == [("submit-job", "run", "--gpu")]

    def test_submit_raises_without_job_id(self):
        r = _hpc.SlurmRemoteRunner(".")
        r._run = lambda *a: types.SimpleNamespace(stdout="kaboom", stderr="", returncode=0)
        with pytest.raises(RuntimeError, match="Job ID"):
            r.submit_job("run")

    def test_submit_rejects_nonzero_exit_even_with_job_id(self):
        r = _hpc.SlurmRemoteRunner(".")
        r._run = lambda *a: types.SimpleNamespace(
            stdout="Job ID: 98765\n", stderr="submission failed", returncode=1,
        )
        with pytest.raises(RuntimeError, match=r"submit-job.*exit 1"):
            r.submit_job("run")

    def test_wait_polls_until_job_absent(self):
        r = _hpc.SlurmRemoteRunner(".")
        seq = iter(["111 running\n", "111 running\n", "no jobs in queue"])
        r._run = lambda *a, **kw: types.SimpleNamespace(
            stdout=next(seq), stderr="", returncode=0,
        )
        r.wait("111", poll_seconds=0.001)

    def test_wait_rejects_status_failure_and_substring_job_ids(self):
        r = _hpc.SlurmRemoteRunner(".")
        r._run = lambda *a, **kw: types.SimpleNamespace(
            stdout="1110 RUNNING\n", stderr="", returncode=0,
        )
        r.wait("111", poll_seconds=0.001, timeout_seconds=1)

        r._run = lambda *a, **kw: types.SimpleNamespace(
            stdout="", stderr="ssh failed", returncode=255,
        )
        with pytest.raises(RuntimeError, match=r"status.*exit 255"):
            r.wait("111", poll_seconds=0.001, timeout_seconds=1)

    @pytest.mark.parametrize(
        ("poll_seconds", "timeout_seconds", "message"),
        [
            (0, 1, "poll_seconds"),
            (True, 1, "poll_seconds"),
            (1, 0, "timeout_seconds"),
            (1, -1, "timeout_seconds"),
            (1, np.inf, "timeout_seconds"),
        ],
    )
    def test_wait_rejects_invalid_controls_before_status(
        self, poll_seconds, timeout_seconds, message,
    ):
        r = _hpc.SlurmRemoteRunner(".")
        r._run = lambda *a: pytest.fail(f"status invoked: {a}")
        with pytest.raises((TypeError, ValueError), match=message):
            r.wait(
                "111", poll_seconds=poll_seconds,
                timeout_seconds=timeout_seconds,
            )

    def test_wait_uses_wall_clock_and_bounds_status_call(self, monkeypatch):
        r = _hpc.SlurmRemoteRunner(".")
        clock = [100.0]
        observed = {}

        monkeypatch.setattr(_hpc.time, "monotonic", lambda: clock[0])
        monkeypatch.setattr(
            _hpc.time, "sleep", lambda value: pytest.fail(f"slept {value}"),
        )

        def slow_status(*args, **kwargs):
            observed["timeout"] = kwargs.get("timeout")
            clock[0] += 2.0
            return types.SimpleNamespace(
                stdout="111 RUNNING\n", stderr="", returncode=0,
            )

        r._run = slow_status
        with pytest.raises(TimeoutError, match="1s timeout"):
            r.wait("111", poll_seconds=0.1, timeout_seconds=1)
        assert observed["timeout"] == pytest.approx(1.0)

        def hung_status(*args, **kwargs):
            raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

        r._run = hung_status
        clock[0] = 200.0
        with pytest.raises(TimeoutError, match="status check.*1s timeout"):
            r.wait("111", poll_seconds=0.1, timeout_seconds=1)

    def test_download_requires_success_and_exact_job_path(self, tmp_path):
        exact = tmp_path / "results/slurm/cluster_123/safe_labels.csv"
        stale = tmp_path / "results/slurm/cluster_999/safe_labels.csv"
        exact.parent.mkdir(parents=True)
        stale.parent.mkdir(parents=True)
        exact.write_text("exact", encoding="utf-8")
        stale.write_text("stale", encoding="utf-8")
        runner = _hpc.SlurmRemoteRunner(tmp_path)
        calls = []

        def successful_download(*args):
            calls.append(args)
            return types.SimpleNamespace(stdout="", stderr="", returncode=0)

        runner._run = successful_download
        assert runner.download_labels("safe", "123") == str(exact)
        assert calls == [("download-cluster", "123")]

        runner._run = lambda *a: types.SimpleNamespace(
            stdout="", stderr="transfer failed", returncode=23,
        )
        with pytest.raises(RuntimeError, match=r"download.*exit 23"):
            runner.download_labels("safe", "123")

        def successful_but_missing(*args):
            exact.unlink()
            return types.SimpleNamespace(stdout="", stderr="", returncode=0)

        runner._run = successful_but_missing
        with pytest.raises(FileNotFoundError, match=r"job 123 wrote no labels"):
            runner.download_labels("safe", "123")
        assert stale.read_text(encoding="utf-8") == "stale"

class TestSlurmLastMile:
    """Pin runner exports and job-script flags without contacting SLURM."""

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    def test_build_profile_injection_is_rejected_before_ssh(self, tmp_path):
        wrapper, fake_bin, _ = _isolated_slurm_wrapper(tmp_path)
        injected = tmp_path / "build-profile-injected.txt"
        ssh_called = tmp_path / "ssh-called.txt"
        fake_ssh = fake_bin / "ssh"
        fake_ssh.write_text(
            "#!/usr/bin/env bash\n"
            "printf called > \"$SSH_CALLED\"\n"
            "exec sh -c \"${2:-}\"\n",
            encoding="utf-8", newline="\n",
        )
        malicious = (
            f"htc-cpu'; touch {_bash_path(injected)}; echo 'injected"
        )
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"export SSH_CALLED={shlex.quote(_bash_path(ssh_called))}; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} build "
            f"{shlex.quote(malicious)}"
        )
        completed = subprocess.run(
            ["bash", "-c", command], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        assert completed.returncode != 0
        assert "build profile" in completed.stderr.lower()
        assert not ssh_called.exists()
        assert not injected.exists()

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    def test_benchmark_gpu_type_injection_is_rejected_before_ssh(self, tmp_path):
        wrapper, fake_bin, _ = _isolated_slurm_wrapper(tmp_path)
        injected = tmp_path / "gpu-type-injected.txt"
        ssh_called = tmp_path / "ssh-called.txt"
        fake_ssh = fake_bin / "ssh"
        fake_ssh.write_text(
            "#!/usr/bin/env bash\n"
            "printf called > \"$SSH_CALLED\"\n"
            "exec sh -c \"${2:-}\"\n",
            encoding="utf-8", newline="\n",
        )
        malicious = f"a100; touch {_bash_path(injected)}; #"
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"export SSH_CALLED={shlex.quote(_bash_path(ssh_called))}; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} "
            f"submit-benchmark-gpu {shlex.quote(malicious)}"
        )
        completed = subprocess.run(
            ["bash", "-c", command], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        assert completed.returncode != 0
        assert "gpu type" in completed.stderr.lower()
        assert not ssh_called.exists()
        assert not injected.exists()

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    @pytest.mark.parametrize(
        "profile", ["arc", "htc-cpu", "htc-gpu", "htc-v4", "h100", "grace"],
    )
    def test_every_documented_build_profile_is_forwarded_as_data(
        self, tmp_path, profile,
    ):
        wrapper, fake_bin, capture = _isolated_slurm_wrapper(tmp_path)
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"export CAPTURE_SBATCH={shlex.quote(_bash_path(capture))}; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} build "
            f"{shlex.quote(profile)}"
        )
        completed = subprocess.run(
            ["bash", "-c", command], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        args = capture.read_text(encoding="utf-8").splitlines()
        exports = next(arg for arg in args if arg.startswith("--export="))
        assert f"DTWC_BUILD_PROFILE={profile}" in exports

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    def test_documented_build_profile_flag_form_is_accepted(self, tmp_path):
        wrapper, fake_bin, capture = _isolated_slurm_wrapper(tmp_path)
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"export CAPTURE_SBATCH={shlex.quote(_bash_path(capture))}; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} "
            "build --profile htc-cpu"
        )
        completed = subprocess.run(
            ["bash", "-c", command], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        args = capture.read_text(encoding="utf-8").splitlines()
        exports = next(arg for arg in args if arg.startswith("--export="))
        assert "DTWC_BUILD_PROFILE=htc-cpu" in exports

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    @pytest.mark.parametrize(
        ("args", "message"),
        [
            ("htc-gpu --gpu-device a6000", None),
            ("htc-gpu", None),
            ("htc-gpu --gpu-device p100", "below the 8.0"),
            ("htc-cpu --gpu-device a100", "needs a GPU profile"),
        ],
    )
    def test_a_gpu_build_runs_on_that_gpu(self, tmp_path, args, message):
        """build --gpu-device asks for that GPU as its jobs do, on the .env
        partition, so build-arc.sh builds natively into build-<type>; without
        it the build asks for no GPU, on an interactive node, and is portable."""
        wrapper, fake_bin, capture = _isolated_slurm_wrapper(tmp_path)
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"export CAPTURE_SBATCH={shlex.quote(_bash_path(capture))}; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} build {args}"
        )
        completed = subprocess.run(
            ["bash", "-c", command], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        if message is not None:
            assert completed.returncode != 0
            assert message in completed.stderr
            assert not capture.exists()
            return
        assert completed.returncode == 0, completed.stdout + completed.stderr
        sbatch_args = capture.read_text(encoding="utf-8").splitlines()
        gpu_args = [arg for arg in sbatch_args if arg.startswith(("--gres=", "--constraint="))]
        exports = next(arg for arg in sbatch_args if arg.startswith("--export="))
        if "--gpu-device" in args:
            assert gpu_args == _hpc.gpu_request("a6000")
            assert "--partition=short" in sbatch_args
            assert exports.endswith(",DTWC_BUILD_DIR=build-a6000")
        else:
            assert gpu_args == []
            assert "--partition=interactive" in sbatch_args
            assert exports.endswith(",DTWC_BUILD_DIR=build-htc-gpu")

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    @pytest.mark.parametrize("mode", ["cpu", "gpu", "bogus"])
    def test_a_smoke_mode_reaches_sbatch(self, tmp_path, mode):
        """submit-smoke <mode> runs smoke.slurm with MODE; the GPU mode asks for
        a GPU at or above the CUDA floor, so it cannot land on a refused V100."""
        wrapper, fake_bin, capture = _isolated_slurm_wrapper(tmp_path)
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"export CAPTURE_SBATCH={shlex.quote(_bash_path(capture))}; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} submit-smoke {mode}"
        )
        completed = subprocess.run(
            ["bash", "-c", command], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        if mode == "bogus":
            assert completed.returncode != 0
            assert "submit-smoke syntax" in completed.stderr
            return
        assert completed.returncode == 0, completed.stdout + completed.stderr
        args = capture.read_text(encoding="utf-8").splitlines()
        assert args[-1] == "scripts/slurm/jobs/smoke.slurm"
        assert f"--export=ALL,MODE={mode}" in args
        gpu_args = [arg for arg in args if arg.startswith(("--gres=", "--constraint="))]
        assert gpu_args == (_hpc.gpu_request() if mode == "gpu" else [])

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    @pytest.mark.parametrize("gpu_type", ["", "a100", "l40s", "h100"])
    def test_documented_benchmark_gpu_types_are_exact_argv(
        self, tmp_path, gpu_type,
    ):
        wrapper, fake_bin, capture = _isolated_slurm_wrapper(tmp_path)
        type_arg = f" {shlex.quote(gpu_type)}" if gpu_type else ""
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"export CAPTURE_SBATCH={shlex.quote(_bash_path(capture))}; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} "
            f"submit-benchmark-gpu{type_arg}"
        )
        completed = subprocess.run(
            ["bash", "-c", command], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        args = capture.read_text(encoding="utf-8").splitlines()
        # The wrapper reads the table _hpc.gpu_request reads: one table, two readers.
        gpu_args = [arg for arg in args if arg.startswith(("--gres=", "--constraint="))]
        assert gpu_args == _hpc.gpu_request(gpu_type or None)

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    def test_commented_empty_optional_config_is_really_empty(self, tmp_path):
        wrapper, fake_bin, _ = _isolated_slurm_wrapper(tmp_path)
        env_file = wrapper.parents[2] / ".env"
        with env_file.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                "SLURM_CLUSTER=                     # optional\n"
                "SLURM_EMAIL=                       # optional\n"
            )
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} test"
        )
        completed = subprocess.run(
            ["bash", "-c", command], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    @pytest.mark.parametrize(
        ("key", "value"),
        [
            ("SLURM_USER", "bad user"),
            ("SLURM_HOST", "host;command"),
            ("SLURM_REMOTE_BASE", "relative/remote"),
            ("SLURM_PARTITION", "short,long"),
            ("SLURM_CLUSTER", "arc;command"),
            ("SLURM_EMAIL", "not-an-email"),
        ],
    )
    def test_unsafe_transport_config_is_rejected_before_ssh(
        self, tmp_path, key, value,
    ):
        wrapper, fake_bin, _ = _isolated_slurm_wrapper(tmp_path)
        env_file = wrapper.parents[2] / ".env"
        with env_file.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(f"{key}={value}\n")
        ssh_called = tmp_path / "ssh-called.txt"
        fake_ssh = fake_bin / "ssh"
        fake_ssh.write_text(
            "#!/usr/bin/env bash\n"
            "printf called > \"$SSH_CALLED\"\n"
            "exit 0\n",
            encoding="utf-8", newline="\n",
        )
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"export SSH_CALLED={shlex.quote(_bash_path(ssh_called))}; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} test"
        )
        completed = subprocess.run(
            ["bash", "-c", command], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        assert completed.returncode != 0
        assert key in completed.stderr
        assert not ssh_called.exists()

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    def test_remote_base_injection_is_rejected_before_remote_shell(self, tmp_path):
        wrapper, fake_bin, _ = _isolated_slurm_wrapper(tmp_path)
        injected = tmp_path / "remote-base-injected.txt"
        env_file = wrapper.parents[2] / ".env"
        with env_file.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                "SLURM_REMOTE_BASE="
                f"/tmp/remote;touch {_bash_path(injected)};#\n"
            )
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} upload"
        )
        completed = subprocess.run(
            ["bash", "-c", command], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        assert completed.returncode != 0
        assert "SLURM_REMOTE_BASE" in completed.stderr
        assert not injected.exists()

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    def test_unsafe_job_id_is_rejected_before_remote_shell(self, tmp_path):
        """download-cluster's one argument, the job ID, crosses the shell."""
        wrapper, fake_bin, _ = _isolated_slurm_wrapper(tmp_path)
        injected = tmp_path / "injected.txt"
        malicious = f"123;printf injected>{_bash_path(injected)};#"
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} download-cluster "
            f"{shlex.quote(malicious)}"
        )
        completed = subprocess.run(
            ["bash", "-c", command], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        assert completed.returncode != 0
        assert "job ID" in completed.stderr
        assert not injected.exists()

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    @pytest.mark.parametrize(
        ("args", "message"),
        [
            ("run,other", "run directory must be"),
            ("-run", "run directory must be"),
            ("C:/run", "run directory must be"),
            ("missing", "no job.toml"),
            ("run --bogus", "submit-job syntax"),
            ("run --gpu-device v100", "below the 8.0"),
            ("run --gpu-device '#'", "unknown GPU type"),
        ],
    )
    def test_submit_job_rejects_an_unsafe_envelope_before_ssh(
        self, tmp_path, args, message,
    ):
        wrapper, fake_bin, _ = _isolated_slurm_wrapper(tmp_path)
        ssh_called = tmp_path / "ssh-called.txt"
        fake_ssh = fake_bin / "ssh"
        fake_ssh.write_text(
            "#!/usr/bin/env bash\nprintf called > \"$SSH_CALLED\"\nexit 97\n",
            encoding="utf-8", newline="\n",
        )
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"export SSH_CALLED={shlex.quote(_bash_path(ssh_called))}; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} submit-job {args}"
        )
        completed = subprocess.run(
            ["bash", "-c", command], cwd=wrapper.parents[2], check=False,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        assert completed.returncode != 0
        assert message in completed.stderr
        assert not ssh_called.exists()

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    def test_allocator_traversal_output_is_rejected_before_submit(self, tmp_path):
        wrapper, fake_bin, capture = _isolated_slurm_wrapper(tmp_path)
        remote = _bash_path(tmp_path / "remote")
        fake_mktemp = fake_bin / "mktemp"
        fake_mktemp.write_text(
            "#!/usr/bin/env bash\n"
            f"echo {shlex.quote(remote + '/data/userjobs/job.ABCDEFGH/../../src')}\n",
            encoding="utf-8", newline="\n",
        )
        fake_mktemp.chmod(0o755)
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"export CAPTURE_SBATCH={shlex.quote(_bash_path(capture))}; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} submit-job run"
        )
        completed = subprocess.run(
            ["bash", "-c", command], cwd=wrapper.parents[2], check=False,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        assert completed.returncode != 0
        assert "allocator returned an unsafe path" in completed.stderr
        assert not capture.exists()

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    def test_configured_cluster_status_failure_is_not_unscoped(self, tmp_path):
        wrapper, fake_bin, _ = _isolated_slurm_wrapper(tmp_path)
        env_file = wrapper.parents[2] / ".env"
        with env_file.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write("SLURM_CLUSTER=arc\n")
        fake_squeue = fake_bin / "squeue"
        fake_squeue.write_text(
            "#!/usr/bin/env bash\n"
            "case \" $* \" in\n"
            "  *' --clusters=arc '*) exit 7 ;;\n"
            "  *) echo 'unscoped empty queue'; exit 0 ;;\n"
            "esac\n",
            encoding="utf-8", newline="\n",
        )
        fake_squeue.chmod(0o755)
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} status"
        )
        completed = subprocess.run(
            ["bash", "-c", command], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        assert completed.returncode == 7
        assert "unscoped empty queue" not in completed.stdout

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    def test_safe_optional_config_reaches_sbatch_as_single_argv(self, tmp_path):
        """The optional settings and the GPU request, its constraint's '|'
        included, reach sbatch as single arguments; the job runs build-l40s."""
        wrapper, fake_bin, capture = _isolated_slurm_wrapper(tmp_path)
        cluster = "arc-prod"
        email = "first.last+dtwc@eng.ox.ac.uk"
        env_file = wrapper.parents[2] / ".env"
        with env_file.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                f"SLURM_CLUSTER={cluster}\n"
                f"SLURM_EMAIL={email}\n"
            )
        binary = tmp_path / "remote/src/build-l40s/bin/dtwc_cl"
        binary.parent.mkdir(parents=True)
        binary.write_text("#!/bin/sh\n", encoding="utf-8", newline="\n")
        binary.chmod(0o755)
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"export CAPTURE_SBATCH={shlex.quote(_bash_path(capture))}; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} submit-job run "
            "--gpu-device l40s"
        )
        completed = subprocess.run(
            ["bash", "-c", command], cwd=wrapper.parents[2], check=False,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        sbatch_args = capture.read_text(encoding="utf-8").splitlines()
        assert f"--clusters={cluster}" in sbatch_args
        assert f"--mail-user={email}" in sbatch_args
        assert [arg for arg in sbatch_args if arg.startswith(("--gres=", "--constraint="))] \
            == _hpc.gpu_request("l40s")
        assert sbatch_args[-2].endswith(",DTWC_BUILD=l40s")

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    def test_each_submission_uploads_its_run_to_a_fresh_directory(self, tmp_path):
        """The run directory and the job script sbatch runs go to one new remote
        directory per submission; the upload is local and option-terminated."""
        wrapper, fake_bin, capture = _isolated_slurm_wrapper(tmp_path)
        transfer_capture = tmp_path / "transfer-args.txt"
        script_capture = tmp_path / "script-transfer-args.txt"
        for tool, variable in (("rsync", "CAPTURE_TRANSFER"), ("scp", "CAPTURE_SCRIPT")):
            (fake_bin / tool).write_text(
                f"#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"${variable}\"\n",
                encoding="utf-8", newline="\n",
            )
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"export CAPTURE_SBATCH={shlex.quote(_bash_path(capture))}; "
            f"export CAPTURE_TRANSFER={shlex.quote(_bash_path(transfer_capture))}; "
            f"export CAPTURE_SCRIPT={shlex.quote(_bash_path(script_capture))}; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} submit-job run"
        )
        remote_dirs = []
        for _ in range(2):
            completed = subprocess.run(
                ["bash", "-c", command], cwd=wrapper.parents[2], check=False,
                capture_output=True, text=True, encoding="utf-8", errors="replace",
            )
            assert completed.returncode == 0, completed.stdout + completed.stderr
            transfer = transfer_capture.read_text(encoding="utf-8").splitlines()
            assert transfer[:3] == ["-az", "--", "run/"]
            host, remote_dir = transfer[3].rstrip("/").split(":", 1)
            assert host == "test_user@test_host"
            script = script_capture.read_text(encoding="utf-8").splitlines()
            assert script[-1] == f"{host}:{remote_dir}/cluster_generic.slurm"
            assert capture.read_text(encoding="utf-8").splitlines()[-2:] == [
                f"--export=ALL,DTWC_JOB={remote_dir},DTWC_BUILD=htc-cpu",
                f"{remote_dir}/cluster_generic.slurm",
            ]
            remote_dirs.append(remote_dir)
        assert remote_dirs[0] != remote_dirs[1]

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    def test_exact_download_cannot_fall_back_to_stale_labels(self, tmp_path):
        wrapper, fake_bin, _ = _isolated_slurm_wrapper(tmp_path)
        project = wrapper.parents[2]
        stale = project / "results/slurm/cluster_123/safe_labels.csv"
        stale.parent.mkdir(parents=True)
        stale.write_text("stale", encoding="utf-8")
        fake_rsync = fake_bin / "rsync"
        fake_rsync.write_text(
            "#!/usr/bin/env bash\nexit 23\n",
            encoding="utf-8", newline="\n",
        )
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} "
            "download-cluster 123"
        )
        completed = subprocess.run(
            ["bash", "-c", command], cwd=project, check=False,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        assert completed.returncode == 23
        assert not stale.exists()

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    @pytest.mark.parametrize("extra", ["", "newer-key = 1\n"])
    def test_the_job_runs_job_toml_on_the_named_build(self, tmp_path, dtwc_cl, extra):
        """cluster_generic.slurm runs build-$DTWC_BUILD's dtwc_cl on
        $DTWC_JOB/job.toml into results/cluster_<job id>/. A key that binary
        does not know (a newer wheel against an older build; the local binary
        is the oracle) fails the job loudly, naming the key, with no labels."""
        binary = tmp_path / "build-test/bin/dtwc_cl"
        binary.parent.mkdir(parents=True)
        binary.write_text(
            f'#!/usr/bin/env bash\nexec {shlex.quote(_bash_path(dtwc_cl))} "$@"\n',
            encoding="utf-8", newline="\n",
        )
        binary.chmod(0o755)
        run = tmp_path / "run"
        run.mkdir()
        _hpc.write_series_tsv([[0.0, 0.1], [0.1, 0.0], [9.0, 9.1], [9.1, 9.0]],
                              run / "input.tsv")
        (run / "job.toml").write_text(
            'input = "input.tsv"\nn-clusters = 2\nname = "job"\n' + extra,
            encoding="utf-8", newline="\n",
        )
        job_env = {
            "SLURM_SUBMIT_DIR": _bash_path(tmp_path), "SLURM_JOB_ID": "123",
            "DTWC_JOB": _bash_path(run), "DTWC_BUILD": "test",
        }
        exports = " ".join(f"{key}={shlex.quote(value)}" for key, value in job_env.items())
        completed = subprocess.run(
            ["bash", "-c", f"export {exports}; exec bash {shlex.quote(_bash_path(_packaged_job()))}"],
            check=False, capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        labels = tmp_path / "results/cluster_123/job_labels.csv"
        if extra:
            assert completed.returncode != 0
            assert "newer-key" in completed.stderr
            assert not labels.exists()
        else:
            assert completed.returncode == 0, completed.stdout + completed.stderr
            assert _hpc.parse_labels_csv(labels, 4).tolist() in ([0, 0, 1, 1], [1, 1, 0, 0])


class TestPackagedWrapper:
    """device='hpc' needs no source checkout: the wrapper is package data (FX-5)."""

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    @pytest.mark.skipif(
        os.name == "nt",
        reason="Git Bash started from Python puts its own ssh/scp ahead of the "
        "fakes on PATH; the bash -c tests below cover the wrapper there",
    )
    def test_cluster_runs_outside_a_checkout(self, tmp_path, monkeypatch, dtwc_cl):
        """dtwcpp.cluster(..., device='hpc') through the real wrapper and job
        script, the local dtwc_cl standing in for the cluster's: only ssh,
        rsync, scp, sbatch and squeue are local stand-ins (sbatch runs the job
        here), and neither the working directory nor the project directory holds
        a scripts/ tree."""
        project, remote, fake_bin = (
            tmp_path / "project", tmp_path / "remote", tmp_path / "bin",
        )
        binary = remote / "src/build-htc-cpu/bin/dtwc_cl"
        for directory in (project, binary.parent, fake_bin):
            directory.mkdir(parents=True)
        binary.write_text(
            f'#!/usr/bin/env bash\nexec {shlex.quote(dtwc_cl)} "$@"\n', encoding="utf-8",
        )
        binary.chmod(0o755)
        (project / ".env").write_text(
            "SLURM_USER=u\nSLURM_HOST=h\n"
            f"SLURM_REMOTE_BASE={_bash_path(remote)}\n",
            encoding="utf-8",
        )
        # The last two arguments, source and destination, host: dropped.
        copy = 'src="${@: -2:1}"; src="${src#*:}"; dst="${@: -1}"; dst="${dst#*:}"\n'
        tools = {
            "ssh": 'exec sh -c "$2"',  # run the "remote" command right here
            "rsync": copy + 'mkdir -p "$dst" && cp -R "$src." "$dst"',
            "scp": 'printf "%s\\n" "$@" > "$CAPTURE_DIR/scp"\n' + copy + 'cp "$src" "$dst"',
            # Run the job here as SLURM would, with the --export values.
            "sbatch": 'printf "%s\\n" "$@" > "$CAPTURE_DIR/sbatch"\n'
                      'export_arg="${@: -2:1}"\n'
                      'IFS=, read -r -a exports <<< "${export_arg#--export=ALL,}"\n'
                      'env "${exports[@]}" SLURM_SUBMIT_DIR="$PWD" SLURM_JOB_ID=12345 '
                      'bash "${@: -1}" > "$CAPTURE_DIR/job.log" 2>&1\n'
                      "echo 12345",
            "squeue": "echo JOBID",
        }
        for name, body in tools.items():
            tool = fake_bin / name
            tool.write_text(
                f"#!/usr/bin/env bash\n{body}\n", encoding="utf-8", newline="\n",
            )
            tool.chmod(0o755)
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("DTWC_REPO_ROOT", str(project))
        monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
        monkeypatch.setenv("CAPTURE_DIR", str(tmp_path))

        rng = np.random.default_rng(7)
        series = [rng.standard_normal(8) * 0.1 + (0.0 if i < 5 else 9.0)
                  for i in range(10)]
        res = dtwcpp.cluster(series, k=2, device="hpc", name="café", method="pam")

        assert len(set(res.labels[:5])) == len(set(res.labels[5:])) == 1
        assert res.labels[0] != res.labels[9]
        assert res.device == "hpc" and res.distance_matrix is None
        assert (project / "results/slurm/cluster_12345/café_labels.csv").is_file()
        package = Path(dtwcpp.__file__).resolve().parent
        scp = (tmp_path / "scp").read_text(encoding="utf-8").splitlines()
        assert Path(scp[-2]).resolve() == package / "_slurm/cluster_generic.slurm"
        sbatch = (tmp_path / "sbatch").read_text(encoding="utf-8").splitlines()
        run = sbatch[-1].rsplit("/", 1)[0]
        assert run.startswith(f"{_bash_path(remote)}/data/userjobs/job.")
        assert sbatch[-2] == f"--export=ALL,DTWC_JOB={run},DTWC_BUILD=htc-cpu"

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    @pytest.mark.parametrize("command", ["upload", "submit-benchmark-cpu"])
    def test_checkout_run_ignores_an_ambient_repo_root(self, tmp_path, command):
        """scripts/slurm/slurm_remote.sh (the benchmark skill's entry point)
        reads .env from, and sends files of, its own checkout: a DTWC_REPO_ROOT
        left in the shell for device='hpc' must not redirect it."""
        wrapper, fake_bin, capture = _isolated_slurm_wrapper(tmp_path)
        checkout = wrapper.parents[2]
        elsewhere = tmp_path / "elsewhere"
        shutil.copytree(checkout / "scripts", elsewhere / "scripts")
        (elsewhere / "dtwc").mkdir()
        (elsewhere / ".env").write_text(
            "SLURM_USER=other_user\nSLURM_HOST=other_host\n"
            f"SLURM_REMOTE_BASE={_bash_path(tmp_path / 'other-remote')}\n",
            encoding="utf-8",
        )
        sent = tmp_path / "sent.txt"
        for tool in ("scp", "rsync"):
            (fake_bin / tool).write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" >> \"$CAPTURE_SENT\"\n",
                encoding="utf-8", newline="\n",
            )
        command_line = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"export CAPTURE_SBATCH={shlex.quote(_bash_path(capture))}; "
            f"export CAPTURE_SENT={shlex.quote(_bash_path(sent))}; "
            f"export DTWC_REPO_ROOT={shlex.quote(_bash_path(elsewhere))}; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} {command}"
        )
        completed = subprocess.run(
            ["bash", "-c", command_line], cwd=tmp_path, check=False,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        args = sent.read_text(encoding="utf-8").splitlines()
        local = [arg for arg in args if arg.startswith("/")]
        remote = [arg for arg in args if ":" in arg and not arg.startswith("/")]
        assert local and all(
            arg.startswith(_bash_path(checkout) + "/") for arg in local
        ), args
        assert remote and all(
            arg.startswith("test_user@test_host:") for arg in remote
        ), args
        if command.startswith("submit-"):
            assert capture.read_text(encoding="utf-8").splitlines()[-1] == (
                "scripts/slurm/jobs/ucr_benchmark_cpu.slurm"
            )

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    @pytest.mark.parametrize(
        "command", ["upload", "submit-smoke cpu", "submit-benchmark-gpu"],
    )
    def test_source_commands_refuse_to_run_outside_a_checkout(
        self, tmp_path, command,
    ):
        """An installed wrapper has no source tree to send. It says so before
        any SSH or transfer, even when DTWC_REPO_ROOT names a checkout."""
        wrapper, fake_bin, _ = _isolated_slurm_wrapper(tmp_path)
        checkout = wrapper.parents[2]
        installed = tmp_path / "venv/lib/python3.12/site-packages/dtwcpp/_slurm"
        shutil.copytree(checkout / "python/dtwcpp/_slurm", installed)
        called = tmp_path / "called.txt"
        for tool in ("ssh", "scp", "rsync"):
            (fake_bin / tool).write_text(
                "#!/usr/bin/env bash\nprintf called > \"$CALLED\"\nexit 97\n",
                encoding="utf-8", newline="\n",
            )
        command_line = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"export CALLED={shlex.quote(_bash_path(called))}; "
            f"export DTWC_REPO_ROOT={shlex.quote(_bash_path(checkout))}; "
            f"exec bash {shlex.quote(_bash_path(installed / 'slurm_remote.sh'))} "
            f"{command}"
        )
        completed = subprocess.run(
            ["bash", "-c", command_line], cwd=tmp_path, check=False,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        assert completed.returncode != 0
        assert "source checkout" in completed.stderr, completed.stderr
        assert not called.exists()


class TestLocalRoundTrip:
    def test_required_input_message_names_toml_first(self, dtwc_cl):
        completed = subprocess.run(
            [dtwc_cl, "--n-clusters", "2"],
            check=False, capture_output=True, text=True,
        )
        assert completed.returncode != 0
        assert completed.stderr == (
            "Error: --input is required via CLI or config file (TOML or YAML)\n"
        )

    def test_the_local_binary_runs_job_toml(self, monkeypatch, tmp_path, dtwc_cl):
        """The cluster's job minus the transport: the local dtwc_cl runs the run
        directory as the job does (--config job.toml --output <dir>), and writes
        back every line Python wrote (--print-config), a UTF-8 name and a float
        among them."""
        out = tmp_path / "out"

        class Runner:
            def __init__(self, repo_root):
                pass

            def preflight(self):
                pass

            def submit_job(self, rundir, **request):
                self.rundir = tmp_path / rundir
                subprocess.run([dtwc_cl, "--config", "job.toml", "--output", str(out)],
                               cwd=self.rundir, check=True, capture_output=True)
                return "1"

            def wait(self, job_id, **controls):
                pass

            def download_labels(self, name, job_id):
                return out / f"{name}_labels.csv"

        runner = Runner(tmp_path)
        monkeypatch.setattr(_hpc, "SlurmRemoteRunner", lambda repo_root: runner)
        monkeypatch.setenv("DTWC_REPO_ROOT", str(tmp_path))
        rng = np.random.default_rng(7)
        series = [rng.standard_normal(8) * 0.1 + (0.0 if i < 5 else 9.0)
                  for i in range(10)]
        res = dtwcpp.cluster(series, k=2, device="hpc", name="café", method="pam",
                             variant="wdtw", wdtw_g=0.17)

        assert len(set(res.labels[:5])) == len(set(res.labels[5:])) == 1
        assert res.labels[0] != res.labels[9]
        written = (runner.rundir / "job.toml").read_text(encoding="utf-8").splitlines()
        printed = subprocess.run(
            [dtwc_cl, "--config", "job.toml", "--print-config"],
            cwd=runner.rundir, check=True, capture_output=True,
        ).stdout.decode("utf-8").splitlines()
        assert 'name = "café"' in written and "wdtw-g = 0.17" in written
        assert set(written) <= set(printed)

    def test_a_string_crosses_as_dtwc_cl_writes_it(self, tmp_path, dtwc_cl):
        """job.toml quotes a string as dtwc_cl's own writer does (quotes,
        backslashes and control bytes escaped, UTF-8 kept), so dtwc_cl reads
        it and writes the same line back."""
        line = "name = " + _hpc._toml_value('q"b\\s\t\x01\x7f é#,[x]')
        job = tmp_path / "job.toml"
        job.write_text(line + "\n", encoding="utf-8", newline="\n")
        printed = subprocess.run(
            [dtwc_cl, "--config", str(job), "--print-config"],
            check=True, capture_output=True,
        ).stdout.decode("utf-8").splitlines()
        assert line == 'name = "q\\"b\\\\s\\t\\u0001\\u007f é#,[x]"'
        assert line in printed
