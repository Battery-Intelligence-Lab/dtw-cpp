"""
@file test_hpc.py
@brief Tests for the HPC offload helpers (data serialization, label parsing,
       command construction). The remote SLURM transport itself is not exercised
       here — only the local, verifiable contract with the dtwc_cl binary.
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
# Command construction (pure — no execution)
# ---------------------------------------------------------------------------
class TestBuildCommand:
    def test_has_core_flags(self):
        cmd = _hpc.build_dtwc_command(
            "dtwc_cl", "in.tsv", k=3, name="job", output_dir="out",
            method="pam", device="cpu",
        )
        s = " ".join(cmd)
        assert "in.tsv" in s and "-k 3" in s.replace("'", "")
        assert "--method pam" in s and "--name job" in s
        assert "--skip-cols 0" in s

    def test_device_flag_passed_through(self):
        cmd = _hpc.build_dtwc_command(
            "dtwc_cl", "in.tsv", k=2, name="j", output_dir="o", device="cuda",
        )
        assert "cuda" in cmd
        assert "-d" in cmd or "--device" in cmd

    def test_restart_schedule_flags_passed_through(self):
        cmd = _hpc.build_dtwc_command(
            "dtwc_cl", "in.tsv", k=3, name="job", output_dir="out",
            method="pam", device="cpu", n_init=2, seed=42,
        )
        assert "--n-init" in cmd
        assert cmd[cmd.index("--n-init") + 1] == "2"
        assert "--seed" in cmd
        assert cmd[cmd.index("--seed") + 1] == "42"

    def test_default_restart_uses_cli_seed_single_source_of_truth(self):
        cmd = _hpc.build_dtwc_command(
            "dtwc_cl", "in.tsv", k=3, name="job", output_dir="out",
        )
        assert cmd[cmd.index("--n-init") + 1] == "1"
        assert "--seed" not in cmd

    @pytest.mark.parametrize(
        ("n_init", "seed", "error", "message"),
        [
            (0, None, ValueError, "at least 1"),
            (True, None, TypeError, "integer"),
            (1, True, TypeError, "integer or None"),
            (1 << 31, None, ValueError, "dtwc_cl int range"),
            (2, 1 << 32, ValueError, "dtwc_cl unsigned range"),
            (2, (1 << 64) - 1, ValueError, "overflows uint64"),
            (1, 1 << 64, ValueError, "fit in uint64"),
        ],
    )
    def test_invalid_restart_schedule_rejected(
        self, n_init, seed, error, message,
    ):
        with pytest.raises(error, match=message):
            _hpc.build_dtwc_command(
                "dtwc_cl", "in.tsv", k=3, name="job", output_dir="out",
                n_init=n_init, seed=seed,
            )

    def test_full_remote_configuration_flags_passed_through(self):
        cmd = _hpc.build_dtwc_command(
            "dtwc_cl", "in.tsv", k=3, name="job", output_dir="out",
            max_iter=17, variant="twe", wdtw_g=0.17,
            adtw_penalty=2.5, msm_c=3.5, twe_nu=0.02,
            twe_lambda=4.0, mv_mode="dependent",
            missing_strategy="error", metric="l1",
        )
        expected = {
            "--max-iter": "17",
            "--variant": "twe",
            "--wdtw-g": "0.17",
            "--adtw-penalty": "2.5",
            "--msm-c": "3.5",
            "--twe-nu": "0.02",
            "--twe-lambda": "4.0",
            "--mv-mode": "dependent",
            "--missing-strategy": "error",
            "--metric": "l1",
        }
        for flag, value in expected.items():
            assert cmd[cmd.index(flag) + 1] == value

        independent = _hpc.build_dtwc_command(
            "dtwc_cl", "in.tsv", k=3, name="job", output_dir="out",
            mv_mode="independent",
        )
        assert independent[independent.index("--mv-mode") + 1] == "independent"
        missing = _hpc.build_dtwc_command(
            "dtwc_cl", "in.tsv", k=3, name="job", output_dir="out",
            missing_strategy="zero_cost",
        )
        assert missing[missing.index("--missing-strategy") + 1] == "zero_cost"


# ---------------------------------------------------------------------------
# Real end-to-end contract: serialize -> run LOCAL dtwc_cl -> parse.
# This is the cluster job minus the ssh/rsync transport. Skips if no binary.
# ---------------------------------------------------------------------------
def _local_binary():
    return _hpc.find_dtwc_binary("C:/D/git/dtw-cpp")


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


def _isolated_slurm_wrapper(tmp_path):
    """Copy the real wrapper behind local SSH/transfer/sbatch executables."""
    root = Path(__file__).resolve().parents[2]
    project = tmp_path / "project"
    wrapper = project / "scripts/slurm/slurm_remote.sh"
    wrapper.parent.mkdir(parents=True)
    shutil.copy2(root / "scripts/slurm/slurm_remote.sh", wrapper)
    job = project / "scripts/slurm/jobs/cluster_generic.slurm"
    job.parent.mkdir(parents=True)
    shutil.copy2(root / "scripts/slurm/jobs/cluster_generic.slurm", job)

    remote = tmp_path / "remote"
    remote_binary = remote / "src/build-test/bin/dtwc_cl"
    remote_binary.parent.mkdir(parents=True)
    remote_binary.write_text("", encoding="utf-8")
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
        r = _hpc.SlurmRemoteRunner(".")
        r._run = lambda *a: types.SimpleNamespace(stdout="  Job ID: 98765\n", stderr="", returncode=0)
        assert r.submit_cluster("in.tsv", 3) == "98765"

    def test_submit_raises_without_job_id(self):
        r = _hpc.SlurmRemoteRunner(".")
        r._run = lambda *a: types.SimpleNamespace(stdout="kaboom", stderr="", returncode=0)
        with pytest.raises(RuntimeError, match="Job ID"):
            r.submit_cluster("in.tsv", 3)

    def test_submit_rejects_nonzero_exit_even_with_job_id(self):
        r = _hpc.SlurmRemoteRunner(".")
        r._run = lambda *a: types.SimpleNamespace(
            stdout="Job ID: 98765\n", stderr="submission failed", returncode=1,
        )
        with pytest.raises(RuntimeError, match=r"submit-cluster.*exit 1"):
            r.submit_cluster("in.tsv", 3)

    def test_submit_forwards_restart_schedule(self):
        captured = {}
        r = _hpc.SlurmRemoteRunner(".")

        def fake_run(*args):
            captured["args"] = args
            return types.SimpleNamespace(
                stdout="  Job ID: 98765\n", stderr="", returncode=0,
            )

        r._run = fake_run
        assert r.submit_cluster("in.tsv", 3, n_init=2, seed=42) == "98765"
        assert captured["args"][9:11] == ("2", "42")

    def test_submit_forwards_full_remote_configuration(self):
        captured = {}
        r = _hpc.SlurmRemoteRunner(".")

        def fake_run(*args):
            captured["args"] = args
            return types.SimpleNamespace(
                stdout="Job ID: 98765\n", stderr="", returncode=0,
            )

        r._run = fake_run
        r.submit_cluster(
            "in.tsv", 3, max_iter=17, variant="twe", wdtw_g=0.17,
            adtw_penalty=2.5, msm_c=3.5, twe_nu=0.02,
            twe_lambda=4.0, mv_mode="dependent",
            missing_strategy="error", metric="l1",
        )
        assert captured["args"][-10:] == (
            "17", "twe", "0.17", "2.5", "3.5", "0.02", "4.0",
            "dependent", "error", "l1",
        )

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
        exact = tmp_path / "results/slurm/safe_123/safe_labels.csv"
        stale = tmp_path / "results/slurm/safe_999/safe_labels.csv"
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
        assert calls == [("download-cluster", "safe", "123")]

        runner._run = lambda *a: types.SimpleNamespace(
            stdout="", stderr="transfer failed", returncode=23,
        )
        with pytest.raises(RuntimeError, match=r"download.*exit 23"):
            runner.download_labels("safe", "123")

        def successful_but_missing(*args):
            exact.unlink()
            return types.SimpleNamespace(stdout="", stderr="", returncode=0)

        runner._run = successful_but_missing
        with pytest.raises(FileNotFoundError, match=r"exact labels.*123"):
            runner.download_labels("safe", "123")
        assert stale.read_text(encoding="utf-8") == "stale"

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"k": 0}, "n_clusters"),
            ({"method": "pam;touch"}, "method"),
            ({"band": -2}, "band"),
            ({"skip_cols": -1}, "skip_cols"),
            ({"name": "../escape"}, "name"),
            ({"input_tsv": "/cluster/input,other.tsv"}, "input"),
            ({"input_tsv": "--delete"}, "must not start"),
            ({"input_tsv": None}, "input_tsv"),
            ({"input_tsv": "C:/data.tsv", "upload": True}, "must not contain"),
        ],
    )
    def test_submit_rejects_unsafe_envelope_before_wrapper(
        self, kwargs, message,
    ):
        values = {
            "input_tsv": "/cluster/input.tsv", "k": 2, "method": "pam",
            "band": -1, "skip_cols": 0, "name": "safe_job",
        }
        values.update(kwargs)
        runner = _hpc.SlurmRemoteRunner(".")
        runner._run = lambda *args: pytest.fail(f"wrapper invoked: {args}")
        with pytest.raises((TypeError, ValueError), match=message):
            runner.submit_cluster(**values)


class TestDTWClusteringHpcDispatch:
    """DTWClustering(device='hpc') offloads instead of computing locally."""

    def test_fit_dispatches_to_cluster_on_hpc(self, monkeypatch):
        import dtwcpp
        from dtwcpp import _hpc

        captured = {}

        def fake_cluster_on_hpc(series, n_clusters, **kwargs):
            captured["n"] = len(series)
            captured["k"] = n_clusters
            captured.update(kwargs)
            return np.zeros(len(series), dtype=int)

        monkeypatch.setattr(_hpc, "cluster_on_hpc", fake_cluster_on_hpc)
        X = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
        clf = dtwcpp.DTWClustering(
            n_clusters=2, n_init=2, max_iter=17, variant="twe",
            wdtw_g=0.17, adtw_penalty=2.5, msm_c=3.5, twe_nu=0.02,
            twe_lambda=4.0, mv_mode="dependent", missing_strategy="error",
            metric="l1", device="hpc",
        )
        labels = clf.fit_predict(X)

        assert captured["k"] == 2 and captured["n"] == 3
        assert captured["n_init"] == 2
        assert captured["seed"] == dtwcpp.DEFAULT_RANDOM_SEED
        assert captured["max_iter"] == 17
        assert captured["variant"] == "twe"
        assert captured["wdtw_g"] == 0.17
        assert captured["adtw_penalty"] == 2.5
        assert captured["msm_c"] == 3.5
        assert captured["twe_nu"] == 0.02
        assert captured["twe_lambda"] == 4.0
        assert captured["mv_mode"] == "dependent"
        assert captured["missing_strategy"] == "error"
        assert captured["metric"] == "l1"
        assert len(labels) == 3
        assert clf.medoid_indices_ is None       # remote fit: only labels_ populated

    def test_cluster_on_hpc_forwards_schedule_to_runner(self, tmp_path):
        labels_path = tmp_path / "forwarded_labels.csv"
        labels_path.write_text("name,cluster\n1,0\n2,1\n", encoding="utf-8")

        class FakeRunner:
            def preflight(self):
                pass

            def submit_cluster(self, *args, **kwargs):
                self.submit_args = args
                self.submit_kwargs = kwargs
                return "123"

            def wait(self, *args, **kwargs):
                pass

            def download_labels(self, name, job_id):
                return labels_path

        runner = FakeRunner()
        labels = _hpc.cluster_on_hpc(
            "/cluster/input.tsv", 2, repo_root=tmp_path, runner=runner,
            n_init=2, seed=42, max_iter=17, variant="twe", wdtw_g=0.17,
            adtw_penalty=2.5, msm_c=3.5, twe_nu=0.02,
            twe_lambda=4.0, mv_mode="dependent",
            missing_strategy="error", metric="l1",
        )

        assert runner.submit_kwargs["n_init"] == 2
        assert runner.submit_kwargs["seed"] == 42
        assert runner.submit_kwargs["max_iter"] == 17
        assert runner.submit_kwargs["variant"] == "twe"
        assert runner.submit_kwargs["wdtw_g"] == 0.17
        assert runner.submit_kwargs["adtw_penalty"] == 2.5
        assert runner.submit_kwargs["msm_c"] == 3.5
        assert runner.submit_kwargs["twe_nu"] == 0.02
        assert runner.submit_kwargs["twe_lambda"] == 4.0
        assert runner.submit_kwargs["mv_mode"] == "dependent"
        assert runner.submit_kwargs["missing_strategy"] == "error"
        assert runner.submit_kwargs["metric"] == "l1"
        np.testing.assert_array_equal(labels, [0, 1])

    def test_same_name_in_memory_submissions_use_distinct_inputs(self, tmp_path):
        labels_path = tmp_path / "labels.csv"
        labels_path.write_text("name,cluster\n1,0\n2,1\n", encoding="utf-8")

        class FakeRunner:
            def __init__(self):
                self.inputs = []

            def preflight(self):
                pass

            def submit_cluster(self, input_tsv, *args, **kwargs):
                path = tmp_path / input_tsv
                self.inputs.append((path, path.read_text(encoding="utf-8")))
                return str(100 + len(self.inputs))

            def wait(self, *args, **kwargs):
                pass

            def download_labels(self, *args):
                return labels_path

        runner = FakeRunner()
        _hpc.cluster_on_hpc(
            [[0.0], [1.0]], 2, repo_root=tmp_path, runner=runner,
            name="same",
        )
        _hpc.cluster_on_hpc(
            [[10.0], [11.0]], 2, repo_root=tmp_path, runner=runner,
            name="same",
        )
        assert runner.inputs[0][0] != runner.inputs[1][0]
        assert runner.inputs[0][1] != runner.inputs[1][1]

    @pytest.mark.parametrize(
        ("kwargs", "error", "message"),
        [
            ({"device": "cpu", "metric": "squared_euclidean"},
             ValueError, "metric"),
            ({"device": f"cuda:{1 << 31}"}, ValueError, "device"),
            ({"device": "cuda:\u0661"}, ValueError, "device"),
            ({"device": 1}, TypeError, "device"),
            ({"device": "cuda", "variant": "twe"}, ValueError, "variant"),
            ({"device": "cuda", "missing_strategy": "zero_cost"},
             ValueError, "missing"),
            ({"device": "cuda", "mv_mode": "independent"},
             ValueError, "mv_mode"),
            ({"variant": "twe", "missing_strategy": "zero_cost"},
             ValueError, "combination"),
            ({"variant": "twe", "mv_mode": "independent"},
             ValueError, "mv_mode"),
            ({"missing_strategy": "zero_cost", "mv_mode": "independent"},
             ValueError, "mv_mode"),
            ({"variant": "unknown"}, ValueError, "variant"),
            ({"missing_strategy": "unknown"}, ValueError, "missing_strategy"),
            ({"mv_mode": "unknown"}, ValueError, "mv_mode"),
            ({"metric": "unknown"}, ValueError, "metric"),
            ({"max_iter": 0}, ValueError, "max_iter"),
            ({"max_iter": 1 << 31}, ValueError, "max_iter"),
            ({"max_iter": True}, TypeError, "max_iter"),
            ({"wdtw_g": np.inf}, ValueError, "wdtw_g"),
            ({"twe_lambda": np.nan}, ValueError, "twe_lambda"),
            ({"msm_c": True}, TypeError, "msm_c"),
            ({"poll_seconds": 0}, ValueError, "poll_seconds"),
            ({"timeout_seconds": 0}, ValueError, "timeout_seconds"),
            ({"timeout_seconds": True}, TypeError, "timeout_seconds"),
        ],
    )
    def test_incompatible_remote_configuration_fails_before_side_effects(
        self, tmp_path, kwargs, error, message,
    ):
        class UntouchedRunner:
            def preflight(self):
                raise AssertionError("preflight must not run")

        run_dir = tmp_path / "results/hpc/rejected"
        with pytest.raises(error, match=message):
            _hpc.cluster_on_hpc(
                "/cluster/input.tsv", 2, repo_root=tmp_path,
                runner=UntouchedRunner(), name="rejected", **kwargs,
            )
        assert not run_dir.exists()

    @pytest.mark.parametrize(
        ("source", "kwargs", "message"),
        [
            ("/cluster/input.tsv", {"name": "../escape"}, "name"),
            ("/cluster/input,other.tsv", {"name": "safe_job"}, "input"),
            ("/cluster/input.tsv", {"name": "safe_job", "method": "pam;touch"},
             "method"),
        ],
    )
    def test_submission_envelope_fails_before_run_directory_or_runner(
        self, tmp_path, source, kwargs, message,
    ):
        class UntouchedRunner:
            def preflight(self):
                raise AssertionError("preflight must not run")

        with pytest.raises((TypeError, ValueError), match=message):
            _hpc.cluster_on_hpc(
                source, 2, repo_root=tmp_path, runner=UntouchedRunner(), **kwargs,
            )
        assert not (tmp_path / "results").exists()


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
        gres_args = [arg for arg in args if arg.startswith("--gres=")]
        assert gres_args == ([f"--gres=gpu:{gpu_type}:1"] if gpu_type else [])

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
            ("SLURM_GPU_GRES", "gpu:a100:1;command"),
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

    def test_restart_schedule_reaches_dtwc_cl(self):
        root = Path(__file__).resolve().parents[2]
        wrapper = (root / "scripts/slurm/slurm_remote.sh").read_text(
            encoding="utf-8",
        )
        job = (root / "scripts/slurm/jobs/cluster_generic.slurm").read_text(
            encoding="utf-8",
        )

        assert "DTWC_N_INIT=${N_INIT}" in wrapper
        assert "DTWC_SEED=${SEED}" in wrapper
        assert 'DTWC_N_INIT="${DTWC_N_INIT:-1}"' in job
        assert '--n-init "${DTWC_N_INIT}"' in job
        assert 'SEED_ARGS=(--seed "${DTWC_SEED}")' in job
        for name in (
            "MAX_ITER", "VARIANT", "WDTW_G", "ADTW_PENALTY", "MSM_C",
            "TWE_NU", "TWE_LAMBDA", "MV_MODE", "MISSING_STRATEGY", "METRIC",
        ):
            assert f"DTWC_{name}" in wrapper
            assert f"DTWC_{name}" in job
        for flag in (
            "--max-iter", "--variant", "--wdtw-g", "--adtw-penalty",
            "--msm-c", "--twe-nu", "--twe-lambda", "--mv-mode",
            "--missing-strategy", "--metric",
        ):
            assert flag in job

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    @pytest.mark.parametrize(("seed_arg", "expected_export"), [("", ""), ("42", "42")])
    def test_seed_export_overrides_inherited_slurm_environment(
        self, tmp_path, seed_arg, expected_export,
    ):
        """``--export=ALL`` must not override omission or an explicit seed."""
        wrapper, fake_bin, capture = _isolated_slurm_wrapper(tmp_path)

        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"export CAPTURE_SBATCH={shlex.quote(_bash_path(capture))}; "
            "export DTWC_SEED=29; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} submit-cluster "
            "/remote/input:v1+tag@host%=a.tsv 2 pam cpu -1 seed_export 0 0 1 "
            f"{shlex.quote(seed_arg)}"
        )
        completed = subprocess.run(
            ["bash", "-c", command], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        sbatch_args = capture.read_text(encoding="utf-8").splitlines()
        exports = next(arg for arg in sbatch_args if arg.startswith("--export="))
        assert "DTWC_INPUT=/remote/input:v1+tag@host%=a.tsv" in exports
        assert f",DTWC_SEED={expected_export}" in exports
        assert ",DTWC_SEED=29" not in exports

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    def test_unsafe_job_name_is_rejected_before_remote_shell(self, tmp_path):
        wrapper, fake_bin, capture = _isolated_slurm_wrapper(tmp_path)
        injected = tmp_path / "injected.txt"
        malicious_name = f"safe;printf injected>{_bash_path(injected)};#"
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"export CAPTURE_SBATCH={shlex.quote(_bash_path(capture))}; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} submit-cluster "
            "/remote/input.tsv 2 pam cpu -1 "
            f"{shlex.quote(malicious_name)} 0 0 1 ''"
        )
        completed = subprocess.run(
            ["bash", "-c", command], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        assert completed.returncode != 0
        assert "job name" in completed.stderr.lower()
        assert not injected.exists()

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({0: "/remote/input,other.tsv"}, "input path"),
            ({0: "--delete"}, "must not start"),
            ({1: "0"}, "n_clusters"),
            ({2: "pam;true"}, "method"),
            ({4: "-2"}, "band"),
            ({5: "../escape"}, "job name"),
            ({6: "-1"}, "skip_cols"),
            ({7: "2"}, "upload"),
            ({0: "C:/data.tsv", 7: "1"}, "must not contain"),
            ({0: "definitely-missing-m28.tsv", 7: "1"}, "input not found"),
        ],
    )
    def test_wrapper_rejects_unsafe_envelope_before_ssh(
        self, tmp_path, overrides, message,
    ):
        wrapper, fake_bin, _ = _isolated_slurm_wrapper(tmp_path)
        ssh_called = tmp_path / "ssh-called.txt"
        fake_ssh = fake_bin / "ssh"
        fake_ssh.write_text(
            "#!/usr/bin/env bash\nprintf called > \"$SSH_CALLED\"\nexit 97\n",
            encoding="utf-8", newline="\n",
        )
        args = [
            "/remote/input.tsv", "2", "pam", "cpu", "-1", "safe_job",
            "0", "0",
        ]
        for index, value in overrides.items():
            args[index] = value
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"export SSH_CALLED={shlex.quote(_bash_path(ssh_called))}; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} submit-cluster "
            + " ".join(shlex.quote(value) for value in args)
        )
        completed = subprocess.run(
            ["bash", "-c", command], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
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
            f"echo {shlex.quote(remote + '/data/userjobs/safe.ABCDEFGH/../../src')}\n",
            encoding="utf-8", newline="\n",
        )
        fake_mktemp.chmod(0o755)
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"export CAPTURE_SBATCH={shlex.quote(_bash_path(capture))}; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} submit-cluster "
            "/remote/input.tsv 2 pam cpu -1 safe 0 0"
        )
        completed = subprocess.run(
            ["bash", "-c", command], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
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
        wrapper, fake_bin, capture = _isolated_slurm_wrapper(tmp_path)
        cluster = "arc-prod"
        email = "first.last+dtwc@eng.ox.ac.uk"
        gres = "gpu:a100:1"
        env_file = wrapper.parents[2] / ".env"
        with env_file.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(
                f"SLURM_CLUSTER={cluster}\n"
                f"SLURM_EMAIL={email}\n"
                f"SLURM_GPU_GRES={gres}\n"
            )
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"export CAPTURE_SBATCH={shlex.quote(_bash_path(capture))}; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} submit-cluster "
            "/remote/input.tsv 2 pam cuda -1 quote_config 0 0"
        )
        completed = subprocess.run(
            ["bash", "-c", command], check=False, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        assert completed.returncode == 0, completed.stdout + completed.stderr
        sbatch_args = capture.read_text(encoding="utf-8").splitlines()
        assert f"--clusters={cluster}" in sbatch_args
        assert f"--mail-user={email}" in sbatch_args
        assert f"--gres={gres}" in sbatch_args

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    def test_safe_upload_is_local_and_option_terminated(self, tmp_path):
        wrapper, fake_bin, capture = _isolated_slurm_wrapper(tmp_path)
        project = wrapper.parents[2]
        source = project / "input+tag%=a.tsv"
        source.write_text("0\t1\n", encoding="utf-8")
        transfer_capture = tmp_path / "transfer-args.txt"
        fake_rsync = fake_bin / "rsync"
        fake_rsync.write_text(
            "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$CAPTURE_TRANSFER\"\n",
            encoding="utf-8", newline="\n",
        )
        script_capture = tmp_path / "script-transfer-args.txt"
        fake_scp = fake_bin / "scp"
        fake_scp.write_text(
            "#!/usr/bin/env bash\nprintf '%s\\n' \"$@\" > \"$CAPTURE_SCRIPT\"\n",
            encoding="utf-8", newline="\n",
        )
        command = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"export CAPTURE_SBATCH={shlex.quote(_bash_path(capture))}; "
            f"export CAPTURE_TRANSFER={shlex.quote(_bash_path(transfer_capture))}; "
            f"export CAPTURE_SCRIPT={shlex.quote(_bash_path(script_capture))}; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} submit-cluster "
            "input+tag%=a.tsv 2 pam cpu -1 safe_upload 0 1"
        )
        destinations = []
        exported_inputs = []
        submitted_scripts = []
        sbatch_scripts = []
        for _ in range(2):
            completed = subprocess.run(
                ["bash", "-c", command], cwd=project, check=False,
                capture_output=True, text=True, encoding="utf-8",
                errors="replace",
            )
            assert completed.returncode == 0, completed.stdout + completed.stderr
            transfer_args = transfer_capture.read_text(
                encoding="utf-8",
            ).splitlines()
            assert transfer_args[:3] == ["-az", "--", "input+tag%=a.tsv"]
            destinations.append(transfer_args[3])
            script_args = script_capture.read_text(
                encoding="utf-8",
            ).splitlines()
            assert script_args[-2].endswith("/cluster_generic.slurm")
            submitted_scripts.append(script_args[-1])
            sbatch_args = capture.read_text(encoding="utf-8").splitlines()
            exports = next(
                arg for arg in sbatch_args if arg.startswith("--export=")
            )
            sbatch_scripts.append(sbatch_args[-1])
            exported_inputs.append(
                exports.split("DTWC_INPUT=", 1)[1].split(",", 1)[0]
            )

        assert destinations[0] != destinations[1]
        assert exported_inputs == [
            target.split(":", 1)[1] for target in destinations
        ]
        assert submitted_scripts[0] != submitted_scripts[1]
        assert sbatch_scripts == [
            target.split(":", 1)[1] for target in submitted_scripts
        ]
        assert sbatch_scripts[0] != sbatch_scripts[1]
        assert [target.rsplit("/", 1)[0] for target in destinations] == [
            target.rsplit("/", 1)[0] for target in submitted_scripts
        ]
        assert all(
            target.endswith("/input+tag%=a.tsv") for target in destinations
        )

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    def test_exact_download_cannot_fall_back_to_stale_labels(self, tmp_path):
        wrapper, fake_bin, _ = _isolated_slurm_wrapper(tmp_path)
        project = wrapper.parents[2]
        stale = project / "results/slurm/safe_123/safe_labels.csv"
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
            "download-cluster safe 123"
        )
        completed = subprocess.run(
            ["bash", "-c", command], cwd=project, check=False,
            capture_output=True, text=True, encoding="utf-8", errors="replace",
        )
        assert completed.returncode == 23
        assert not stale.exists()

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    @pytest.mark.parametrize(
        ("seed", "expect_seed", "overrides"),
        [
            ("42", True, {}),
            ("", False, {
                "DTWC_MAX_ITER": "17",
                "DTWC_VARIANT": "twe",
                "DTWC_WDTW_G": "0.17",
                "DTWC_ADTW_PENALTY": "2.5",
                "DTWC_MSM_C": "3.5",
                "DTWC_TWE_NU": "0.02",
                "DTWC_TWE_LAMBDA": "4.0",
            }),
            ("", False, {
                "DTWC_DEVICE": "cuda",
                "DTWC_METRIC": "squared_euclidean",
            }),
            ("", False, {"DTWC_MISSING_STRATEGY": "zero_cost"}),
            ("", False, {"DTWC_MV_MODE": "independent"}),
        ],
    )
    def test_job_executes_final_restart_arguments(
        self, tmp_path, seed, expect_seed, overrides,
    ):
        root = Path(__file__).resolve().parents[2]
        job = root / "scripts/slurm/jobs/cluster_generic.slurm"
        fake_bin = tmp_path / "build-fake/bin/dtwc_cl"
        fake_bin.parent.mkdir(parents=True)
        fake_bin.write_bytes(
            b'#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAPTURE_ARGS"\n'
        )
        fake_bin.chmod(0o755)

        input_path = tmp_path / "input.tsv"
        input_path.write_text("0\t1\n", encoding="utf-8")
        capture_path = tmp_path / "args.txt"
        job_env = {
            "SLURM_SUBMIT_DIR": _bash_path(tmp_path),
            "SLURM_JOB_ID": "123",
            "SLURMD_NODENAME": "test-node",
            "SLURM_CPUS_PER_TASK": "1",
            "DTWC_INPUT": _bash_path(input_path),
            "DTWC_K": "2",
            "DTWC_NAME": "restart_test",
            "DTWC_DEVICE": "cpu",
            "DTWC_N_INIT": "2",
            "DTWC_SEED": seed,
            "DTWC_MAX_ITER": "100",
            "DTWC_VARIANT": "standard",
            "DTWC_WDTW_G": "0.05",
            "DTWC_ADTW_PENALTY": "1.0",
            "DTWC_MSM_C": "1.0",
            "DTWC_TWE_NU": "0.001",
            "DTWC_TWE_LAMBDA": "1.0",
            "DTWC_MV_MODE": "dependent",
            "DTWC_MISSING_STRATEGY": "error",
            "DTWC_METRIC": "l1",
            "CAPTURE_ARGS": _bash_path(capture_path),
        }
        job_env.update(overrides)
        exports = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in job_env.items()
        )
        command = f"export {exports}; exec bash {shlex.quote(_bash_path(job))}"

        subprocess.run(
            ["bash", "-c", command], check=True, cwd=tmp_path,
            capture_output=True, text=True,
        )
        args = capture_path.read_text(encoding="utf-8").splitlines()
        assert args[args.index("--n-init") + 1] == "2"
        if expect_seed:
            assert args[args.index("--seed") + 1] == seed
        else:
            assert "--seed" not in args
        expected = {
            "--max-iter": "DTWC_MAX_ITER",
            "--variant": "DTWC_VARIANT",
            "--wdtw-g": "DTWC_WDTW_G",
            "--adtw-penalty": "DTWC_ADTW_PENALTY",
            "--msm-c": "DTWC_MSM_C",
            "--twe-nu": "DTWC_TWE_NU",
            "--twe-lambda": "DTWC_TWE_LAMBDA",
            "--mv-mode": "DTWC_MV_MODE",
            "--missing-strategy": "DTWC_MISSING_STRATEGY",
            "--metric": "DTWC_METRIC",
        }
        for flag, variable in expected.items():
            assert args[args.index(flag) + 1] == job_env[variable]


@pytest.mark.skipif(_local_binary() is None, reason="no local dtwc_cl binary built")
class TestLocalRoundTrip:
    def test_required_input_message_names_toml_first(self):
        completed = subprocess.run(
            [_local_binary(), "--n-clusters", "2"],
            check=False, capture_output=True, text=True,
        )
        assert completed.returncode != 0
        assert completed.stderr == (
            "Error: --input is required via CLI or config file "
            "(TOML; YAML if built with DTWC_ENABLE_YAML)\n"
        )

    def test_two_groups_recovered(self, tmp_path):
        rng = np.random.default_rng(7)
        series = [list(rng.standard_normal(8) * 0.1 + (0.0 if i < 5 else 9.0))
                  for i in range(10)]
        tsv = tmp_path / "input.tsv"
        _hpc.write_series_tsv(series, tsv)

        out = tmp_path / "out"
        out.mkdir()
        cmd = _hpc.build_dtwc_command(
            _local_binary(), str(tsv), k=2, name="rt", output_dir=str(out),
            method="pam", device="cpu",
        )
        subprocess.run(cmd, check=True, capture_output=True, text=True)

        labels = _hpc.parse_labels_csv(out / "rt_labels.csv", n=10)
        # two groups of 5; first 5 share a label, last 5 share the other
        assert len(set(labels[:5])) == 1
        assert len(set(labels[5:])) == 1
        assert labels[0] != labels[9]

    def test_seeded_restart_improves_registered_fixture(self, tmp_path):
        base = np.array([0.0, 0.01, -0.02, 0.03])
        series = [base + offset for offset in range(8)]
        tsv = tmp_path / "restart_input.tsv"
        _hpc.write_series_tsv(series, tsv)

        observed = []
        for n_init in (1, 2):
            out = tmp_path / f"restart_{n_init}"
            out.mkdir()
            cmd = _hpc.build_dtwc_command(
                _local_binary(), str(tsv), k=3, name="restart",
                output_dir=str(out), method="pam", device="cpu",
                n_init=n_init, seed=42,
            )
            completed = subprocess.run(
                cmd, check=True, capture_output=True, text=True,
            )
            observed.append(completed.stdout)
            assert (out / "restart_labels.csv").is_file()

        assert "cost=24" in observed[0]
        assert "cost=20" in observed[1]

    def test_cli_rejects_zero_restart_count(self, tmp_path):
        tsv = tmp_path / "invalid_restart.tsv"
        _hpc.write_series_tsv([[0.0, 1.0], [1.0, 2.0]], tsv)
        cmd = _hpc.build_dtwc_command(
            _local_binary(), str(tsv), k=2, name="invalid_restart",
            output_dir=str(tmp_path), n_init=1,
        )
        cmd[cmd.index("--n-init") + 1] = "0"

        completed = subprocess.run(
            cmd, check=False, capture_output=True, text=True,
        )
        assert completed.returncode != 0
        assert "n-init" in completed.stderr

    def test_cli_missing_strategy_is_honored(self, tmp_path):
        tsv = tmp_path / "series.tsv"
        _hpc.write_series_tsv(
            [[0.0, 0.5, 1.0], [0.0, 0.6, 1.0], [9.0, 9.5, 10.0]],
            tsv,
        )

        accepted_dir = tmp_path / "accepted"
        accepted_dir.mkdir()
        accepted_cmd = _hpc.build_dtwc_command(
            _local_binary(), str(tsv), k=2, name="missing_zero_cost",
            output_dir=str(accepted_dir), missing_strategy="zero_cost",
        )
        completed = subprocess.run(
            accepted_cmd, check=True, capture_output=True, text=True,
        )
        assert "Missing:  zero_cost" in completed.stdout
        assert (accepted_dir / "missing_zero_cost_labels.csv").is_file()
