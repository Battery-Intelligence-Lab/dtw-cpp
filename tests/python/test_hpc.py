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


class TestSlurmRunner:
    """The Python orchestration glue around slurm_remote.sh (no real cluster)."""

    def test_submit_parses_job_id(self):
        r = _hpc.SlurmRemoteRunner(".")
        r._run = lambda *a: types.SimpleNamespace(stdout="  Job ID: 98765\n", stderr="", returncode=0)
        assert r.submit_cluster("in.tsv", 3) == "98765"

    def test_submit_raises_without_job_id(self):
        r = _hpc.SlurmRemoteRunner(".")
        r._run = lambda *a: types.SimpleNamespace(stdout="kaboom", stderr="", returncode=1)
        with pytest.raises(RuntimeError, match="Job ID"):
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
        assert captured["args"][-2:] == ("2", "42")

    def test_wait_polls_until_job_absent(self):
        r = _hpc.SlurmRemoteRunner(".")
        seq = iter(["...111 running...", "...111 running...", "no jobs in queue"])
        r._run = lambda *a: types.SimpleNamespace(stdout=next(seq), stderr="")
        r.wait("111", poll_seconds=0)        # returns once "111" no longer present


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
        clf = dtwcpp.DTWClustering(n_clusters=2, n_init=2, device="hpc")
        labels = clf.fit_predict(X)

        assert captured["k"] == 2 and captured["n"] == 3
        assert captured["n_init"] == 2
        assert captured["seed"] == dtwcpp.DEFAULT_RANDOM_SEED
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

            def download_labels(self, name):
                return labels_path

        runner = FakeRunner()
        labels = _hpc.cluster_on_hpc(
            "/cluster/input.tsv", 2, repo_root=tmp_path, runner=runner,
            n_init=2, seed=42,
        )

        assert runner.submit_kwargs["n_init"] == 2
        assert runner.submit_kwargs["seed"] == 42
        np.testing.assert_array_equal(labels, [0, 1])


class TestSlurmLastMile:
    """Pin runner exports and job-script flags without contacting SLURM."""

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

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    @pytest.mark.parametrize(("seed", "expect_seed"), [("42", True), ("", False)])
    def test_job_executes_final_restart_arguments(
        self, tmp_path, seed, expect_seed,
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
            "DTWC_N_INIT": "2",
            "DTWC_SEED": seed,
            "CAPTURE_ARGS": _bash_path(capture_path),
        }
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
