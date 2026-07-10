"""
@file test_hpc.py
@brief Tests for the HPC offload helpers (data serialization, label parsing,
       command construction). The remote SLURM transport itself is not exercised
       here — only the local, verifiable contract with the dtwc_cl binary.
@author Volkan Kumtepeli
"""
import os
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


# ---------------------------------------------------------------------------
# Real end-to-end contract: serialize -> run LOCAL dtwc_cl -> parse.
# This is the cluster job minus the ssh/rsync transport. Skips if no binary.
# ---------------------------------------------------------------------------
def _local_binary():
    return _hpc.find_dtwc_binary("C:/D/git/dtw-cpp")


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
        clf = dtwcpp.DTWClustering(n_clusters=2, device="hpc")
        labels = clf.fit_predict(X)

        assert captured["k"] == 2 and captured["n"] == 3
        assert len(labels) == 3
        assert clf.medoid_indices_ is None       # remote fit: only labels_ populated


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
