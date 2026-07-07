"""
@file test_test_api.py
@brief Tests for the dtwcpp.test introspection API (Task 3.3).

Drives the LIVE public entry points dtwcpp.test.parallelisation() and
dtwcpp.test.gpu() (bound from the header-only dtwc::test::* probes) — the same
functions exercised by the C++ (tests/unit/test_test_api.cpp) and MATLAB
(tests/matlab/test_test_api.m) suites, asserting the SAME field names.

Registered expectations (this OpenMP-ON / CUDA-OFF extension build):
  parallelisation() -> available, pass, threads_engaged >= 2 on a multicore host;
  gpu()             -> unavailable branch, reason non-empty, no exception.
The gpu() assertions branch on the reported `available` flag so the file is also
correct against a GPU-enabled build.
"""

import dtwcpp
import dtwcpp.test


_PARALLEL_KEYS = {"available", "max_threads", "threads_engaged", "pass", "reason"}
_GPU_KEYS = {"available", "backend", "device_name", "validated", "pass", "reason"}


def test_parallelisation_schema_and_engagement():
    """dtwcpp.test.parallelisation(): real OMP region, distinct-thread proof."""
    r = dtwcpp.test.parallelisation()

    # Exact schema parity with C++ / MATLAB (same field names).
    assert set(r.keys()) == _PARALLEL_KEYS

    assert isinstance(r["available"], bool)
    assert r["max_threads"] >= 1
    assert r["threads_engaged"] >= 1

    if r["available"]:
        assert r["reason"] == ""
        if r["max_threads"] >= 2:
            # Load-bearing: a REAL parallel region engaged >= 2 distinct threads.
            assert r["threads_engaged"] >= 2
            assert r["threads_engaged"] <= r["max_threads"]
            assert r["pass"] is True
        else:
            assert r["pass"] is True
    else:
        # Sequential build: honest loud unavailable, never a fake pass.
        assert r["reason"] != ""
        assert r["pass"] is False


def test_gpu_schema_and_validation_or_reason():
    """dtwcpp.test.gpu(): execute+validate, or a loud reason; never raises."""
    r = dtwcpp.test.gpu()  # must not raise

    assert set(r.keys()) == _GPU_KEYS

    if r["available"]:
        assert r["backend"] != ""
        assert r["device_name"] != ""
        assert r["validated"] is True
        assert r["pass"] is True
    else:
        # Unavailable branch (CUDA-OFF extension here): reason names what's missing.
        assert r["reason"] != ""
        assert r["validated"] is False
        assert r["pass"] is False


def test_check_system_reports_metal(capsys):
    """check_system() surfaces Metal (metal_available was bound but unreported)."""
    dtwcpp.check_system()
    out = capsys.readouterr().out
    assert "Metal:" in out
