"""
@file test_single_thread_warning.py
@brief No-silent-fallback on the Python compute path (Task 3.6, review H1).

The high-level Python compute entry points never construct dtwc::env(), so before
this fix a build with OpenMP present but only 1 usable thread (OMP_NUM_THREADS=1,
common on SLURM/containers) ran SILENTLY single-threaded. This test reproduces
exactly that scenario in a fresh subprocess — OMP_NUM_THREADS=1, and NO
dtwcpp.device() call — and asserts the loud RuntimeSingleThread warning reaches
stderr. Subprocess because the warning is process-once.

Skipped unless the extension was built with OpenMP AND the host is multicore
(the runtime single-thread warning is intentionally suppressed on a genuine
single-core host — no crying wolf).
"""

import os
import subprocess
import sys
import textwrap

import pytest

import dtwcpp


_MULTICORE = (os.cpu_count() or 1) >= 2
_OPENMP = bool(getattr(dtwcpp, "OPENMP_AVAILABLE", False))


@pytest.mark.skipif(
    not (_OPENMP and _MULTICORE),
    reason="needs an OpenMP extension build on a multicore host",
)
def test_compute_warns_single_threaded_without_device_call():
    """OMP_NUM_THREADS=1 + high-level compute (no device() call) -> loud stderr."""
    code = textwrap.dedent(
        """
        import dtwcpp
        # High-level compute WITHOUT any dtwcpp.device() call. This path never
        # constructs dtwc::env(); it must still warn loudly when serialised.
        series = [[0.0, 1.0, 2.0, 3.0], [1.0, 1.0, 1.0, 1.0], [3.0, 2.0, 1.0, 0.0]]
        dm = dtwcpp.compute_distance_matrix(series, band=2, metric="l1")
        assert dm.shape == (3, 3)
        """
    )
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"

    r = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )

    assert r.returncode == 0, f"subprocess failed:\n{r.stderr}"
    # The RuntimeSingleThread cause specifically (not merely "single-threaded").
    assert "OpenMP is available but only 1 thread is usable" in r.stderr, r.stderr
    assert "running SINGLE-THREADED" in r.stderr, r.stderr
    # Exactly one loud line — the process-once guard is not spamming.
    assert r.stderr.count("[DTWC++ WARNING]") == 1, r.stderr
