"""
@file test_distance_matrix.py
@brief Tests for DenseDistanceMatrix Python bindings.
@author Volkan Kumtepeli
"""

import math

import numpy as np
import pytest

import dtwcpp


class TestDenseDistanceMatrix:
    """Tests for dtwcpp.DenseDistanceMatrix."""

    def test_construction_size(self):
        """Constructed matrix has correct size."""
        dm = dtwcpp.DenseDistanceMatrix(5)
        assert dm.size == 5

    def test_set_get(self):
        """set(i,j,v) and get(i,j) round-trip correctly."""
        dm = dtwcpp.DenseDistanceMatrix(4)
        dm.set(1, 2, 3.14)
        assert dm.get(1, 2) == pytest.approx(3.14)

    def test_symmetry_enforcement(self):
        """set(i,j,v) makes get(j,i) == v (symmetric storage)."""
        dm = dtwcpp.DenseDistanceMatrix(4)
        dm.set(0, 3, 7.5)
        assert dm.get(3, 0) == pytest.approx(7.5)

    def test_to_numpy_shape(self):
        """to_numpy() returns (n, n) array."""
        n = 6
        dm = dtwcpp.DenseDistanceMatrix(n)
        arr = dm.to_numpy()
        assert arr.shape == (n, n)

    def test_to_numpy_values(self):
        """to_numpy() reflects values set via set()."""
        dm = dtwcpp.DenseDistanceMatrix(3)
        dm.set(0, 1, 5.0)
        dm.set(0, 2, 7.0)
        dm.set(1, 2, 3.0)
        arr = dm.to_numpy()
        assert arr[0, 1] == pytest.approx(5.0)
        assert arr[1, 0] == pytest.approx(5.0)
        assert arr[0, 2] == pytest.approx(7.0)
        assert arr[2, 0] == pytest.approx(7.0)
        assert arr[1, 2] == pytest.approx(3.0)
        assert arr[2, 1] == pytest.approx(3.0)

    def test_to_numpy_symmetric(self):
        """The full numpy matrix is symmetric."""
        dm = dtwcpp.DenseDistanceMatrix(4)
        dm.set(0, 1, 1.0)
        dm.set(0, 2, 2.0)
        dm.set(0, 3, 3.0)
        dm.set(1, 2, 4.0)
        dm.set(1, 3, 5.0)
        dm.set(2, 3, 6.0)
        arr = dm.to_numpy()
        np.testing.assert_array_almost_equal(arr, arr.T)

    def test_to_numpy_is_independent_copy(self):
        """to_numpy() returns an independent copy — DenseDistanceMatrix uses
        packed triangular storage so a zero-copy full-NxN view is structurally
        impossible. Mutating the returned array must NOT affect the C++ matrix;
        use ``set(i, j, v)`` for that."""
        dm = dtwcpp.DenseDistanceMatrix(4)
        dm.set(0, 1, 1.0)
        arr = dm.to_numpy()
        arr[0, 1] = 42.0
        arr[1, 0] = 99.0
        assert dm.get(0, 1) == pytest.approx(1.0)

    def test_to_numpy_dtype(self):
        """to_numpy() returns float64 array."""
        dm = dtwcpp.DenseDistanceMatrix(3)
        arr = dm.to_numpy()
        assert arr.dtype == np.float64

    def test_resize(self):
        """resize() changes the matrix size."""
        dm = dtwcpp.DenseDistanceMatrix(3)
        assert dm.size == 3
        dm.resize(7)
        assert dm.size == 7

    def test_max(self):
        """max() returns the largest stored value."""
        dm = dtwcpp.DenseDistanceMatrix(3)
        dm.set(0, 1, 10.0)
        dm.set(0, 2, 5.0)
        dm.set(1, 2, 20.0)
        assert dm.max() == pytest.approx(20.0)


class TestOwnedBufferHandover:
    """A10/G5: numpy arrays returned by the bindings own their buffer.

    The buffers used to be raw `new double[n*n]` handed to a capsule only
    afterwards, so anything throwing in between leaked the whole N^2 matrix,
    and the GPU paths built the array inside a `gil_scoped_acquire` nested in a
    live release. Both are now one `adopt_as_ndarray` call.
    """

    def test_matrix_outlives_every_other_reference(self):
        arr = dtwcpp.compute_distance_matrix(
            [[1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [9.0, 1.0, 5.0]]
        )
        import gc

        copy = arr.copy()
        gc.collect()
        assert arr.shape == (3, 3)
        np.testing.assert_array_equal(arr, copy)

    def test_empty_input_returns_empty_matrix(self):
        arr = dtwcpp.compute_distance_matrix([])
        assert arr.shape == (0, 0)

    def test_dense_to_numpy_owns_its_data(self):
        dm = dtwcpp.DenseDistanceMatrix(3)
        dm.set(0, 1, 2.0)
        arr = dm.to_numpy()
        del dm
        import gc

        gc.collect()
        assert arr[0, 1] == pytest.approx(2.0)
        assert arr[1, 0] == pytest.approx(2.0)

    def test_zero_sized_dense_to_numpy(self):
        dm = dtwcpp.DenseDistanceMatrix(0)
        assert dm.to_numpy().shape == (0, 0)


class TestPdlpBinding:
    """E2: the PDLP LP-relaxation arbiter was reachable only from C++."""

    def test_gpu_capability_query_exists(self):
        from dtwcpp import _dtwcpp_core as core

        assert isinstance(core.PDLP_GPU_AVAILABLE, bool)
        assert core.pdlp_gpu_available() == core.PDLP_GPU_AVAILABLE

    def test_lp_bound_rejects_a_non_square_matrix(self):
        from dtwcpp import _dtwcpp_core as core

        D = np.zeros((3, 4), dtype=np.float64)
        with pytest.raises(dtwcpp.InvalidInput):
            core.pdlp_lp_bound(D, 2)

    def test_lp_bound_is_a_lower_bound_or_reports_no_solver(self):
        from dtwcpp import _dtwcpp_core as core

        D = np.array(
            [[0.0, 1.0, 5.0, 6.0],
             [1.0, 0.0, 6.0, 5.0],
             [5.0, 6.0, 0.0, 1.0],
             [6.0, 5.0, 1.0, 0.0]],
            dtype=np.float64,
        )
        if not dtwcpp.HIGHS_AVAILABLE:
            with pytest.raises(dtwcpp.SolverError):
                core.pdlp_lp_bound(D, 2)
            return
        result = core.pdlp_lp_bound(D, 2)
        # Optimal 2-medoid cost of this instance is 1 + 1 = 2; the LP
        # relaxation can only be at or below it.
        assert result.lp_bound <= 2.0 + 1e-6
        assert result.lp_bound >= -1e-9


class TestProblemThreadSafety:
    """Threading policy (audit 2026-09-02, D1).

    A `Problem` is single-thread-only, exactly like the C++ class. Every
    binding releases the GIL around its native work so other Python threads
    keep running, and no binding takes a per-object lock: two threads calling
    methods on the SAME Problem race on its lazily-filled distance cache.
    The supported patterns are one Problem per thread, or
    `fill_distance_matrix()` first and read-only access afterwards.
    """

    @staticmethod
    def _series(n=12, length=24):
        return [
            [
                math.sin(0.3 * t + 0.17 * i) * (1.0 + 0.05 * i)
                for t in range(length)
            ]
            for i in range(n)
        ]

    def _problem(self, series):
        p = dtwcpp.Problem("gil_thread_safety")
        p.set_data(series, [f"s{i}" for i in range(len(series))])
        return p

    def test_prefilled_problem_reads_the_same_values_from_several_threads(self):
        """The supported concurrent pattern: fill first, then only read."""
        import threading

        series = self._series()
        n = len(series)
        pairs = [(i, j) for i in range(n) for j in range(n)]

        serial = self._problem(series)
        expected = [serial.dist_by_ind(i, j) for (i, j) in pairs]

        shared = self._problem(series)
        shared.fill_distance_matrix()  # no lazy mutation left to race on
        assert shared.is_distance_matrix_filled()

        n_threads = 8
        results = [None] * n_threads
        barrier = threading.Barrier(n_threads)

        def worker(slot):
            barrier.wait()  # maximise overlap
            results[slot] = [shared.dist_by_ind(i, j) for (i, j) in pairs]

        threads = [
            threading.Thread(target=worker, args=(k,)) for k in range(n_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for slot, got in enumerate(results):
            assert got is not None, f"thread {slot} did not finish"
            assert got == expected, f"thread {slot} disagreed with the serial run"

        assert all(math.isfinite(v) for v in expected)
        assert all(shared.dist_by_ind(i, i) == 0.0 for i in range(n))

    def test_dist_by_ind_releases_the_gil(self):
        """The policy is *consistent release*: no binding pins the GIL.

        A pure-Python ticker thread advances only while it holds the GIL, so
        a native call that kept the GIL would freeze it. `fill_distance_matrix`
        (known to release) calibrates the measurement, so the assertion does
        not depend on machine speed.

        Each measurement carries GIL hand-off noise, which only ever ADDS
        ticks, so the minimum over a few repeats is the discriminating
        statistic: a minimum well above zero means the ticker really ran
        inside the call.
        """
        import sys
        import threading
        import time

        # Long enough that one unbanded pair is tens of ms of native work.
        long_series = [
            [math.sin(0.01 * t) for t in range(3000)],
            [math.cos(0.011 * t) for t in range(3000)],
        ]

        ticks = 0
        stop = False

        def ticker():
            nonlocal ticks
            while not stop:
                ticks += 1

        def measure(call):
            problem = dtwcpp.Problem("gil_probe")
            problem.set_data(long_series, ["x", "y"])
            before = ticks
            call(problem)
            return ticks - before

        previous_interval = sys.getswitchinterval()
        sys.setswitchinterval(0.001)
        thread = threading.Thread(target=ticker, daemon=True)
        thread.start()
        try:
            time.sleep(0.05)  # let the ticker reach steady state
            lazy = min(measure(lambda p: p.dist_by_ind(0, 1)) for _ in range(5))
            control = min(
                measure(lambda p: p.fill_distance_matrix()) for _ in range(5)
            )
        finally:
            stop = True
            thread.join(timeout=5.0)
            sys.setswitchinterval(previous_interval)

        # Calibration: the known-releasing control really does let the ticker run.
        assert control > 1000, (
            "control fill_distance_matrix() did not let the ticker run "
            f"({control} ticks); the probe is not calibrated"
        )
        # dist_by_ind computes the same single pair and must release as well.
        assert lazy > control // 10, (
            f"dist_by_ind let the ticker advance only {lazy} ticks against "
            f"{control} for the releasing control -- it is holding the GIL"
        )


class TestCheckpointMetricFingerprint:
    """Audit 2026-09-02, item 3: the A5 metric fingerprint did not reach
    Python, so a SquaredL2 checkpoint was still accepted by a later L1 run."""

    def _filled_problem(self, name):
        p = dtwcpp.Problem(name)
        p.set_data(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            ["a", "b", "c"],
        )
        p.fill_distance_matrix()
        return p

    def test_squared_l2_checkpoint_is_rejected_by_an_l1_load(self, tmp_path):
        directory = str(tmp_path / "ckpt_sq")
        dtwcpp.save_checkpoint(
            self._filled_problem("ckpt_src"), directory, dtwcpp.MetricType.SquaredL2
        )

        # Default (L1) load must refuse the SquaredL2 fingerprint.
        target = dtwcpp.Problem("ckpt_dst")
        target.set_data(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            ["a", "b", "c"],
        )
        assert dtwcpp.load_checkpoint(target, directory) is False
        assert not target.is_distance_matrix_filled()

        # The matching metric still loads, so the rejection is the fingerprint
        # and not a broken write.
        assert (
            dtwcpp.load_checkpoint(target, directory, dtwcpp.MetricType.SquaredL2)
            is True
        )

    def test_l1_checkpoint_round_trips_on_the_default_metric(self, tmp_path):
        directory = str(tmp_path / "ckpt_l1")
        source = self._filled_problem("ckpt_l1_src")
        dtwcpp.save_checkpoint(source, directory)

        target = dtwcpp.Problem("ckpt_l1_dst")
        target.set_data(
            [[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0], [1.0, 1.0, 1.0, 1.0]],
            ["a", "b", "c"],
        )
        assert dtwcpp.load_checkpoint(target, directory) is True
        np.testing.assert_allclose(target.distance_matrix(), source.distance_matrix())
