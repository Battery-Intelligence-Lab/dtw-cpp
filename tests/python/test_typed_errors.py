"""docs/api-contract-2.0.md §5 at the seams the GT-4 sweep left (GT-4b).

Each case raised an untyped C++ error (Python ``RuntimeError``), the wrong
leaf, or nothing: a checkpoint directory that cannot be created, ``skip_cols``
wider than a file's rows, and a GPU PDLP solve on a build without the GPU
backend, which warned and ran on the CPU.
"""

import os
import re
import sys

import numpy as np
import pytest

import dtwcpp
from dtwcpp import _dtwcpp_core as core


def _problem():
    prob = dtwcpp.Problem("gt4b")
    prob.set_data([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [5.0, 6.0, 7.0]], ["a", "b", "c"])
    return prob


class TestSaveCheckpointPath:
    """A filesystem failure is ``dtwcpp.IOError`` (an ``OSError``) naming the path."""

    def test_path_is_a_regular_file(self, tmp_path):
        plain = tmp_path / "plain"
        plain.write_text("x")
        with pytest.raises(dtwcpp.IOError, match=re.escape(str(plain))):
            dtwcpp.save_checkpoint(_problem(), str(plain))

    def test_parent_is_a_regular_file(self, tmp_path):
        plain = tmp_path / "plain"
        plain.write_text("x")
        target = plain / "checkpoint"
        with pytest.raises(dtwcpp.IOError, match=re.escape(str(target))):
            dtwcpp.save_checkpoint(_problem(), str(target))

    @pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0,
                        reason="POSIX permission bits bind only an unprivileged user")
    def test_parent_is_read_only(self, tmp_path):
        locked = tmp_path / "locked"
        locked.mkdir()
        locked.chmod(0o555)
        target = locked / "checkpoint"
        try:
            with pytest.raises(dtwcpp.IOError, match=re.escape(str(target))):
                dtwcpp.save_checkpoint(_problem(), str(target))
        finally:
            locked.chmod(0o755)


class TestSkipColsWiderThanARow:
    """§1.2: ``skip_cols`` beyond a series is ``InvalidInput`` from memory; a file
    source is the same request, not a failed read."""

    def test_file_source_raises_what_the_in_memory_source_raises(self, tmp_path):
        with pytest.raises(dtwcpp.InvalidInput):
            dtwcpp.load([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], skip_cols=5).as_series()
        csv = tmp_path / "three.csv"
        csv.write_text("1,2,3\n4,5,6\n", encoding="utf-8")
        with pytest.raises(dtwcpp.InvalidInput, match="fewer than start_col=5"):
            dtwcpp.load(csv, skip_cols=5).as_series()


class TestPdlpGpuRequest:
    @pytest.mark.skipif(not dtwcpp.HIGHS_AVAILABLE or core.PDLP_GPU_AVAILABLE,
                        reason="needs HiGHS built without its GPU backend")
    def test_gpu_request_without_the_gpu_backend_is_device_error(self):
        D = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
        params = core.PdlpParams()
        params.use_gpu = True
        with pytest.raises(dtwcpp.DeviceError, match="DTWC_HIGHS_GPU"):
            core.pdlp_lp_bound(D, 1, params)
