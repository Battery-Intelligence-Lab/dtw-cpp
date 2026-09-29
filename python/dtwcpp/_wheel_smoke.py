"""Runtime release-artifact gate used by cibuildwheel.

The six series are a compact subset of the cross-language conformance fixture:
two representatives from each of its three deliberately separated groups.
"""

import os
import tempfile


_SERIES = [
    [-4, -2, 0, 2, 0, -2, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4],
    [-3, -1, 1, 3, 1, -1, -3, -3, -3, -3, -3, -3, -3, -3, -3, -3],
    [96, 96, 96, 96, 96, 96, 98, 100, 102, 100, 98, 96, 96, 96, 96, 96],
    [97, 97, 97, 97, 97, 97, 99, 101, 103, 101, 99, 97, 97, 97, 97, 97],
    [196, 196, 196, 196, 196, 196, 196, 196, 196, 196, 198, 200, 202, 200, 198, 196],
    [197, 197, 197, 197, 197, 197, 197, 197, 197, 197, 199, 201, 203, 201, 199, 197],
]


def _canonical(labels):
    mapping = {}
    return [mapping.setdefault(int(label), len(mapping)) for label in labels]


def _mapped_matrix_persists(dtwcpp):
    """Map a distance matrix to a .dtwm file, fill it, and reopen it filled."""
    series = [[float(v) for v in s] for s in _SERIES]
    names = [str(i) for i in range(len(series))]
    with tempfile.TemporaryDirectory() as scratch:
        path = os.path.join(scratch, "smoke.dtwm")
        first = dtwcpp.Problem("smoke")
        first.set_data(series, names)
        first.use_mmap_distance_matrix(path)
        first.fill_distance_matrix()
        second = dtwcpp.Problem("smoke")
        second.set_data(series, names)
        second.use_mmap_distance_matrix(path)
        filled = second.is_distance_matrix_filled()
        size = os.path.getsize(path)
        del first, second  # unmap before the directory goes
    assert filled, "the mapped distance matrix did not persist"
    assert size == 48 + 21 * 8, size
    return size


def run():
    """Assert compiled parallelism, bundled HiGHS, a real MIP solve and a mapped matrix."""
    import dtwcpp
    import dtwcpp.test

    parallel = dtwcpp.test.parallelisation()
    assert parallel["available"], parallel["reason"]
    if os.cpu_count() is not None and os.cpu_count() >= 2:
        assert parallel["threads_engaged"] >= 2, parallel

    assert dtwcpp.HIGHS_AVAILABLE, "wheel was built without bundled HiGHS"
    result = dtwcpp.cluster(_SERIES, 3, method="mip", band=3, device="cpu")
    assert _canonical(result.labels) == [0, 0, 1, 1, 2, 2], result.labels
    mapped_bytes = _mapped_matrix_persists(dtwcpp)
    print(
        "dtwcpp wheel smoke OK:",
        dtwcpp.__version__,
        "OpenMP threads =", parallel["threads_engaged"],
        "HiGHS MIP cost =", result.cost,
        "mapped .dtwm bytes =", mapped_bytes,
    )


if __name__ == "__main__":
    run()
