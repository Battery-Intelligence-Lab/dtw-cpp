"""
@file test_cdist_workflow.py
@brief Tests for cross-collection cdist, the files and checkpoint cluster() writes, zero-copy
       ingest, lazy sklearn import, and extended SLURM HPC staging/hydration.
@author Volkan Kumtepeli
"""
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

import numpy as np
import pytest

import dtwcpp
from dtwcpp import _hpc


def _bash_path(path):
    path = Path(path).resolve().as_posix()
    if os.name != "nt" or len(path) < 3 or path[1:3] != ":/":
        return path
    uname = subprocess.run(
        ["bash", "-c", "uname -s"], check=True, capture_output=True, text=True
    ).stdout.strip().lower()
    drive, suffix = path[0].lower(), path[2:]
    return f"/mnt/{drive}{suffix}" if uname.startswith("linux") else f"/{drive}{suffix}"


class TestCdist:
    def test_rectangular_and_self_cdist_match_dtw_oracle(self):
        rng = np.random.default_rng(123)
        X = rng.normal(size=(5, 16))
        Y = rng.normal(loc=2.0, size=(19, 16))  # 19 tests SIMD tail-panel padding

        D = dtwcpp.cdist(X, Y, band=3, metric="squared_euclidean")
        assert D.shape == (5, 19) and D.dtype == np.float64
        for i in range(5):
            for j in range(19):
                ref = dtwcpp.distance.dtw(X[i], Y[j], band=3, metric="squared_euclidean")
                assert D[i, j] == pytest.approx(ref, rel=1e-12)

        # Y=None computes X against X via symmetric fast path
        D_self = dtwcpp.distance.cdist(X, band=3, metric="squared_euclidean")
        D_xx = dtwcpp.cdist(X, X, band=3, metric="squared_euclidean")
        np.testing.assert_allclose(D_self, D_xx, rtol=1e-12, atol=1e-12)

    def test_ragged_variants_float32_and_multivariate_3d(self):
        X_rag = [np.array([0.0, 1.0, 2.0, 1.0]), np.array([3.0, 3.5, 4.0, 4.2, 3.8, 3.1])]
        Y_rag = [np.array([0.2, 0.9, 1.8, 1.2, 0.5]), np.array([3.1, 3.6, 3.9, 4.1]), np.array([-1.0, -2.0, -1.5])]

        D_rag = dtwcpp.cdist(X_rag, Y_rag, variant="wdtw", wdtw_g=0.07)
        assert D_rag.shape == (2, 3)
        for i, xi in enumerate(X_rag):
            for j, yj in enumerate(Y_rag):
                assert D_rag[i, j] == pytest.approx(
                    dtwcpp.distance.dtw(xi, yj, variant="wdtw", wdtw_g=0.07), rel=1e-12
                )

        # matrix_dtype="float32" returns float32 array
        D_f32 = dtwcpp.cdist(X_rag, Y_rag, variant="wdtw", wdtw_g=0.07, matrix_dtype="float32")
        assert D_f32.dtype == np.float32
        np.testing.assert_array_equal(D_f32, D_rag.astype(np.float32))

        # 3-D multivariate array (n_series, n_steps, ndim) when ndim > 1
        rng = np.random.default_rng(9)
        X_3d = rng.normal(size=(4, 10, 3))
        Y_3d = rng.normal(size=(6, 10, 3))
        D_3d = dtwcpp.cdist(X_3d, Y_3d, ndim=3, mv_mode="dependent", band=2)
        D_flat = dtwcpp.cdist(X_3d.reshape(4, 30), Y_3d.reshape(6, 30), ndim=3, mv_mode="dependent", band=2)
        np.testing.assert_allclose(D_3d, D_flat, rtol=1e-12)

        with pytest.raises(dtwcpp.InvalidInput, match="ndim"):
            dtwcpp.cdist(X_3d, Y_3d, ndim=2)

    def test_hpc_device_rejected_on_cdist(self):
        X = np.zeros((2, 4))
        with pytest.raises(dtwcpp.DeviceError, match="hpc"):
            dtwcpp.cdist(X, X, device="hpc")

    def test_gpu_and_hybrid_cdist_when_available(self):
        if not dtwcpp.gpu_available():
            return
        rng = np.random.default_rng(77)
        X = rng.normal(size=(6, 20))
        Y = rng.normal(loc=1.5, size=(15, 20))
        D_cpu = dtwcpp.cdist(X, Y, band=4, device="cpu")
        D_gpu = dtwcpp.cdist(X, Y, band=4, device="gpu", gpu_precision="fp64")
        np.testing.assert_allclose(D_gpu, D_cpu, rtol=1e-12)
        if dtwcpp.test.gpu()["hybrid"]:
            D_hyb = dtwcpp.cdist(X, Y, band=4, device="cpu+gpu", gpu_precision="fp64")
            np.testing.assert_allclose(D_hyb, D_cpu, rtol=1e-12)


class TestClusterFilesAndFastIngest:
    def test_cluster_writes_the_files_and_checkpoint_it_is_asked_for(self, tmp_path, monkeypatch):
        csv = tmp_path / "sensor.csv"
        csv.write_text(
            "id,t0,t1,t2,t3\n"
            "a,0.0,0.1,0.2,0.1\n"
            "b,0.1,0.0,0.1,0.2\n"
            "c,0.05,0.15,0.1,0.05\n"
            "d,9.0,9.1,9.2,9.0\n"
            "e,9.2,9.0,9.1,9.3\n"
            "f,8.9,9.1,9.0,9.2\n",
            encoding="utf-8",
        )
        auto_out = tmp_path / "auto_out"
        ckpt_dir = tmp_path / "ckpt"
        manual_out = tmp_path / "manual_out"
        keys = {"method": "pam", "band": 2}

        ds = dtwcpp.load(csv, skip_rows=1, skip_cols=1, name="sensor")
        res = dtwcpp.cluster(ds, 2, output=auto_out, checkpoint=ckpt_dir, **keys)
        assert res.labels.shape == (6,)
        assert len(set(res.labels[:3])) == 1 and len(set(res.labels[3:])) == 1
        assert res.labels[0] != res.labels[3]
        assert (auto_out / "sensor_labels.csv").is_file()
        assert (auto_out / "sensor_medoids.csv").is_file()
        assert (auto_out / "sensor_report.txt").is_file()
        assert (ckpt_dir / "sensor.dtwm").is_file()

        # The same checkpoint resumes.
        resumed = dtwcpp.cluster(ds, 2, checkpoint=ckpt_dir, **keys)
        assert re.search(r"the checkpoint\s+resumed", resumed.report)
        np.testing.assert_array_equal(resumed.labels, res.labels)

        # No output, or None or "", writes nothing; Result.save writes on request.
        cwd = tmp_path / "cwd"
        cwd.mkdir()
        monkeypatch.chdir(cwd)
        for given in ({}, {"output": None}, {"output": ""}):
            mem_res = dtwcpp.cluster(ds, 2, **given, **keys)
            np.testing.assert_array_equal(mem_res.labels, res.labels)
            assert mem_res.cost == pytest.approx(res.cost, rel=1e-12)
        assert list(cwd.iterdir()) == []
        mem_res.save(manual_out)
        assert (manual_out / "sensor_labels.csv").is_file()
        assert (manual_out / "sensor_medoids.csv").is_file()

        # An interval saves a checkpoint, so it needs one.
        with pytest.raises(dtwcpp.InvalidInput, match="checkpoint-interval requires"):
            dtwcpp.cluster(ds, 2, checkpoint_interval=5, **keys)

    def test_a_dataset_refuses_a_dtype_it_was_not_made_with(self):
        """A Dataset is read with the options it was made with: a dtype given to cluster(),
        even at its default, or to cdist(), that is another value is refused, naming both;
        the same value, spelled either way, or none, runs."""
        ds = dtwcpp.load(np.vstack([np.zeros((3, 6)), np.full((3, 6), 9.0)]), skip_rows=1)
        with pytest.raises(dtwcpp.InvalidInput, match="cdist: the Dataset was made with dtype='float64'"):
            dtwcpp.cdist(ds, dtype="float32")
        ds32 = dtwcpp.load(ds.source, dtype="float32")
        with pytest.raises(dtwcpp.InvalidInput, match="dtype='float32' and dtype='float64' was given"):
            dtwcpp.cdist(ds32, dtype="float64")
        with pytest.raises(dtwcpp.InvalidInput, match="dtype='float32' and dtype='float64' was given"):
            dtwcpp.cluster(ds32, 2, dtype="float64")
        with pytest.raises(dtwcpp.InvalidInput, match="cdist: the Data was made with dtype='float64'"):
            dtwcpp.cdist(ds.as_data(), dtype="float32")
        assert dtwcpp.cluster(ds, 2).n_series == 5
        assert dtwcpp.cluster(ds32, 2, dtype="float32").n_series == 6
        assert dtwcpp.cluster(ds32, 2, dtype="f32").n_series == 6
        assert dtwcpp.cluster(ds32, 2).n_series == 6

    def test_direct_numpy_and_3d_ingest_and_estimator_transform(self):
        rng = np.random.default_rng(5)
        X_f32 = np.vstack([
            rng.normal(0.0, 0.1, size=(5, 12)).astype(np.float32),
            rng.normal(8.0, 0.1, size=(5, 12)).astype(np.float32),
        ])
        ds_f32 = dtwcpp.load(X_f32, dtype="float32")
        data_cpp = ds_f32.as_data()
        assert data_cpp.is_f32() and data_cpp.size == 10

        est = dtwcpp.DTWClustering(n_clusters=2, band=3, metric="squared_euclidean", n_init=2).fit(X_f32)
        Q = rng.normal(4.0, 1.0, size=(3, 12))
        T = est.transform(Q)
        assert T.shape == (3, 2)
        for i in range(3):
            for m_idx, medoid in enumerate(est.cluster_centers_):
                assert T[i, m_idx] == pytest.approx(
                    dtwcpp.distance.dtw(Q[i], medoid, band=3, metric="squared_euclidean"), rel=1e-12
                )

        # Problem.set_data accepts 3-D (n_series, n_steps, ndim) when ndim > 1
        X_mv = rng.normal(size=(6, 8, 2))
        prob = dtwcpp.Problem("mv3d")
        prob.set_data(X_mv, ndim=2)
        assert prob.size == 6

    def test_import_dtwcpp_does_not_eagerly_import_sklearn(self):
        proc = subprocess.run(
            [sys.executable, "-c", "import sys, dtwcpp; assert 'sklearn' not in sys.modules"],
            capture_output=True,
            text=True,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr


class TestHpcExtensions:
    def test_upload_input_stages_local_file_and_hydrates_medoids_and_report(self, monkeypatch, tmp_path):
        local_csv = tmp_path / "local_series.csv"
        local_csv.write_text("0.0,0.1\n0.1,0.0\n9.0,9.1\n9.1,9.0\n", encoding="utf-8")
        out_dir = tmp_path / "remote_out"
        out_dir.mkdir()
        (out_dir / "staged_labels.csv").write_text("name,cluster\n1,0\n2,0\n3,1\n4,1\n", encoding="utf-8")
        (out_dir / "staged_medoids.csv").write_text(
            "cluster,medoid_index,medoid_name\n0,1,2\n1,2,3\n", encoding="utf-8"
        )
        (out_dir / "staged_report.txt").write_text("DTWC++ run report\nstatus: ok\n", encoding="utf-8")

        captured = {}

        class FakeRunner:
            def __init__(self, repo_root):
                pass

            def preflight(self):
                pass

            def submit_job(self, rundir, **request):
                full = tmp_path / rundir
                captured["files"] = sorted(p.name for p in full.iterdir())
                captured["job_toml"] = (full / "job.toml").read_text(encoding="utf-8")
                return "77"

            def wait(self, job_id, **controls):
                pass

            def download_labels(self, name, job_id):
                return out_dir / f"{name}_labels.csv"

        monkeypatch.setattr(_hpc, "SlurmRemoteRunner", FakeRunner)
        monkeypatch.setenv("DTWC_REPO_ROOT", str(tmp_path))

        res = dtwcpp.cluster(local_csv, 2, method="pam", device="hpc", upload_input=True, name="staged",
                             output=tmp_path / "auto_copied")

        assert captured["files"] == ["input.csv", "job.toml"]
        assert 'input = "input.csv"' in captured["job_toml"]
        np.testing.assert_array_equal(res.labels, [0, 0, 1, 1])
        np.testing.assert_array_equal(res.medoids, [1, 2])
        assert res.report.startswith("DTWC++ run report")

        assert (tmp_path / "auto_copied" / "staged_labels.csv").is_file()  # output receives the downloaded files
        saved_dir = tmp_path / "copied_local"
        res.save(saved_dir)
        assert (saved_dir / "staged_labels.csv").is_file()
        assert (saved_dir / "staged_medoids.csv").is_file()
        assert (saved_dir / "staged_report.txt").is_file()

        # Extensionless local input stages as "input" (comma default), never "input.tsv"
        raw_input = tmp_path / "sensor_rows"
        raw_input.write_text("0.0,0.1\n0.1,0.0\n9.0,9.1\n9.1,9.0\n", encoding="utf-8")
        dtwcpp.cluster(raw_input, 2, method="pam", device="hpc", upload_input=True, name="staged")
        assert captured["files"] == ["input", "job.toml"]
        assert 'input = "input"' in captured["job_toml"]

    @pytest.mark.skipif(shutil.which("bash") is None, reason="bash unavailable")
    def test_slurm_account_qos_and_modules_validation(self, tmp_path):
        root = Path(__file__).resolve().parents[2]
        project = tmp_path / "proj"
        shutil.copytree(root / "scripts/slurm", project / "scripts/slurm")
        shutil.copytree(Path(_hpc.slurm_wrapper_path()).parent, project / "python/dtwcpp/_slurm")
        (project / "dtwc").mkdir()
        (project / "run").mkdir()
        (project / "run/job.toml").write_text("n-clusters = 2\n", encoding="utf-8")
        remote = tmp_path / "remote"
        b = remote / "src/build-htc-cpu/bin/dtwc_cl"
        b.parent.mkdir(parents=True)
        b.write_text("#!/bin/sh\n", encoding="utf-8", newline="\n")
        b.chmod(0o755)
        (project / ".env").write_text(
            "SLURM_USER=u\nSLURM_HOST=h\n"
            f"SLURM_REMOTE_BASE={_bash_path(remote)}\n"
            "SLURM_ACCOUNT=engs-ox\n"
            "SLURM_QOS=normal\n"
            "SLURM_MODULES=GCC/13.2.0 CUDA/12.4\n",
            encoding="utf-8",
            newline="\n",
        )
        fake_bin = tmp_path / "fbin"
        fake_bin.mkdir()
        (fake_bin / "ssh").write_text('#!/usr/bin/env bash\nexec sh -c "${2:-}"\n', encoding="utf-8", newline="\n")
        for t in ("rsync", "scp"):
            (fake_bin / t).write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8", newline="\n")
        cap = tmp_path / "sbatch.txt"
        (fake_bin / "sbatch").write_text(
            '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$CAP"\necho 42\n', encoding="utf-8", newline="\n"
        )
        for p in fake_bin.iterdir():
            p.chmod(0o755)

        wrapper = project / "scripts/slurm/slurm_remote.sh"
        cmd = (
            f"export PATH={shlex.quote(_bash_path(fake_bin))}:\"$PATH\"; "
            f"export CAP={shlex.quote(_bash_path(cap))}; "
            f"exec bash {shlex.quote(_bash_path(wrapper))} submit-job run"
        )
        res = subprocess.run(["bash", "-c", cmd], cwd=project, capture_output=True, text=True, encoding="utf-8")
        assert res.returncode == 0, res.stdout + res.stderr
        sbatch_args = cap.read_text(encoding="utf-8").splitlines()
        assert "--account=engs-ox" in sbatch_args
        assert "--qos=normal" in sbatch_args
        assert any("DTWC_MODULES=GCC/13.2.0:CUDA/12.4" in a for a in sbatch_args)

