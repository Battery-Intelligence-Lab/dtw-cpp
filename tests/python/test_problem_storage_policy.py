"""F20: the public Python Problem storage policy governs owning set_data."""

import csv
import gc
import math
import os
from pathlib import Path
import shutil
import struct
import tempfile
import zlib

import dtwcpp
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "data" / "test" / "nonUnimodular_1_Nc_2.csv"
BUILD_ROOT = ROOT / "build"
NDIM = 2
MMAP_UNAVAILABLE = (
    "Problem::set_data: StoragePolicy::Mmap requested but mmap support "
    "(llfio) is not compiled in. Rebuild with -DDTWC_ENABLE_LLFIO=ON."
)
UPPER_PAIR_BITS = (
    0x4071FDCAC083126F,
    0x406456B851EB851F,
    0x406C7E76C8B43958,
    0x4070914395810625,
    0x4072B9EF9DB22D0F,
    0x4071476C8B439581,
    0x406ECDB22D0E5604,
    0x406CD3EF9DB22D0E,
    0x406A6B7CED916872,
    0x406B8978D4FDF3B7,
    0x406E59C28F5C28F6,
    0x40713C147AE147AF,
    0x4057EBD70A3D70A4,
    0x406D6A353F7CED91,
    0x406FDB3333333332,
)


def _fixture():
    with FIXTURE.open(newline="", encoding="utf-8") as stream:
        series = [[float(value) for value in row] for row in csv.reader(stream)]
    assert len(series) == 6
    assert all(len(row) == 6 for row in series)
    return series, [str(index) for index in range(1, 7)]


def _bits(value):
    return struct.unpack("<Q", struct.pack("<d", value))[0]


def _dependent_l1_dtw(lhs, rhs):
    """Independent full-matrix dependent-MV L1 recurrence."""
    assert len(lhs) % NDIM == 0
    assert len(rhs) % NDIM == 0
    n_steps = len(lhs) // NDIM
    m_steps = len(rhs) // NDIM
    matrix = [
        [math.inf for _ in range(m_steps + 1)]
        for _ in range(n_steps + 1)
    ]
    matrix[0][0] = 0.0
    for i in range(1, n_steps + 1):
        for j in range(1, m_steps + 1):
            local = sum(
                abs(
                    lhs[(i - 1) * NDIM + dimension]
                    - rhs[(j - 1) * NDIM + dimension]
                )
                for dimension in range(NDIM)
            )
            matrix[i][j] = local + min(
                matrix[i - 1][j],
                matrix[i][j - 1],
                matrix[i - 1][j - 1],
            )
    return matrix[n_steps][m_steps]


def _distance_oracle(series):
    upper = [
        _dependent_l1_dtw(series[i], series[j])
        for i in range(len(series))
        for j in range(i + 1, len(series))
    ]
    assert tuple(_bits(value) for value in upper) == UPPER_PAIR_BITS
    return [
        [
            0.0 if i == j else _dependent_l1_dtw(series[i], series[j])
            for j in range(len(series))
        ]
        for i in range(len(series))
    ]


def _verify_problem(problem, series, names, distances):
    assert problem.size == len(series)
    for index, expected in enumerate(series):
        assert problem.series_name(index) == names[index]
        observed = problem.series(index)
        assert struct.pack(f"<{len(observed)}d", *observed) == struct.pack(
            f"<{len(expected)}d", *expected
        )
    for i in range(len(series)):
        for j in range(len(series)):
            assert _bits(problem.dist_by_ind(i, j)) == _bits(distances[i][j])


def _fast_pam_snapshot(problem):
    result = dtwcpp.fast_pam_seeded(
        problem, 2, dtwcpp.DEFAULT_RANDOM_SEED, 100
    )
    return (
        tuple(result.labels),
        tuple(result.medoid_indices),
        _bits(result.total_cost),
        result.iterations,
        result.converged,
    )


def _verify_mmap_artifact(path, series):
    contents = path.read_bytes()
    assert len(contents) == 408
    assert contents[:4] == b"DTWS"
    assert struct.unpack_from("<H", contents, 4)[0] == 1
    assert struct.unpack_from("<I", contents, 6)[0] == 0x01020304
    assert contents[10] == 8
    assert struct.unpack_from("<Q", contents, 12)[0] == 6
    assert struct.unpack_from("<Q", contents, 20)[0] == NDIM
    assert struct.unpack_from("<I", contents, 28)[0] == 0xAB81A0E4
    assert zlib.crc32(contents[:28]) == 0xAB81A0E4
    assert struct.unpack_from("<7Q", contents, 64) == (
        0,
        48,
        96,
        144,
        192,
        240,
        288,
    )
    payload = b"".join(
        struct.pack(f"<{len(row)}d", *row) for row in series
    )
    assert contents[120:] == payload


def test_problem_storage_policy_routes_real_owning_set_data(monkeypatch):
    """Run the mmap success or exact llfio-off transaction; never skip."""
    series, names = _fixture()
    distances = _distance_oracle(series)
    scratch = (
        BUILD_ROOT / f"f20-python-storage-{os.getpid()}"
    ).resolve()
    assert scratch.parent == BUILD_ROOT.resolve()
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)

    old_tempdir = tempfile.tempdir
    for variable in ("TMP", "TEMP", "TMPDIR"):
        monkeypatch.setenv(variable, str(scratch))
    tempfile.tempdir = str(scratch)

    heap = None
    candidate = None
    try:
        heap = dtwcpp.Problem("f20_python_heap")
        heap.storage_policy = dtwcpp.StoragePolicy.Heap
        heap.set_data(series, names, NDIM)
        assert heap.storage_policy == dtwcpp.StoragePolicy.Heap
        assert not list(scratch.glob("*.dtws"))
        _verify_problem(heap, series, names, distances)
        heap_clustering = _fast_pam_snapshot(heap)

        candidate = dtwcpp.Problem("f20_python_candidate")
        sentinel = [[-7.0, -3.0, 11.0, 2.0]]
        candidate.storage_policy = dtwcpp.StoragePolicy.Heap
        candidate.set_data(sentinel, ["sentinel"], NDIM)
        candidate.set_distance_matrix(np.array([[0.0]], dtype=np.float64))
        assert candidate.is_distance_matrix_filled()

        candidate.storage_policy = dtwcpp.StoragePolicy.Mmap
        assert candidate.storage_policy == dtwcpp.StoragePolicy.Mmap
        assert candidate.series(0) == sentinel[0]
        assert candidate.series_name(0) == "sentinel"

        try:
            candidate.set_data(series, names, NDIM)
        except dtwcpp.IOError as error:
            assert type(error) is dtwcpp.IOError
            assert str(error) == MMAP_UNAVAILABLE
            assert candidate.size == 1
            assert candidate.series(0) == sentinel[0]
            assert candidate.series_name(0) == "sentinel"
            assert candidate.is_distance_matrix_filled()
            assert candidate.distance_matrix().tolist() == [[0.0]]
            assert not list(scratch.glob("*.dtws"))
            print(
                "F20_PYTHON_STORAGE build=llfio-off route=rejected "
                "transaction=pass artifacts=0 distances=36/36 "
                "subject_skips=0 verdict=PASS"
            )
        else:
            artifacts = list(scratch.glob("*.dtws"))
            assert len(artifacts) == 1
            _verify_problem(candidate, series, names, distances)
            assert _fast_pam_snapshot(candidate) == heap_clustering
            candidate = None
            gc.collect()
            _verify_mmap_artifact(artifacts[0], series)
            print(
                "F20_PYTHON_STORAGE build=llfio-on route=mmap "
                "artifact=pass distances=72/72 fastpam=exact "
                "subject_skips=0 verdict=PASS"
            )
    finally:
        candidate = None
        heap = None
        gc.collect()
        tempfile.tempdir = old_tempdir
        if scratch.exists():
            shutil.rmtree(scratch)
