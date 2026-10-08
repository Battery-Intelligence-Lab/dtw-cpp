"""One WM measurement process: one shape, one thread setting, the wheel of the venv whose python runs it.

python measure.py <shape> <threads> [--smoke]
  shape    a | b | c (Problem.fill_distance_matrix) or d_full | d_band (dtwcpp.distance.dtw on one pair, in a loop)
  threads  1: OMP_NUM_THREADS must be 1 (driver.py sets it); this process pins itself to logical CPU 12 (a P-core)
           before dtwcpp loads. 24: no OMP_* variable, no pinning (the OpenMP default).
  --smoke  the same paths on a few series and a 0.05 s interval; for proving the kit, never for a table.
Prints one line of key=value fields. Gcell/s counts the DP cells of the shape: |i-j| <= band, every cell unbanded.
"""
import ctypes
import hashlib
import json
import os
import pathlib
import sys
import time

KIT = pathlib.Path(__file__).resolve().parent
CPU = 12
BAND = 100
SMOKE_N = {"a": 160, "b": 40, "c": 120}


def band_cells(n, band):
    """DP cells of an n x n matrix inside |i - j| <= band (all n * n when band < 0)."""
    if band < 0 or band >= n - 1:
        return n * n
    return n * (2 * band + 1) - band * (band + 1)


def pin_to_cpu():
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.SetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    k32.GetProcessAffinityMask.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t),
                                           ctypes.POINTER(ctypes.c_size_t)]
    process = k32.GetCurrentProcess()
    if not k32.SetProcessAffinityMask(process, 1 << CPU):
        raise OSError(ctypes.get_last_error(), "SetProcessAffinityMask")
    mask, system = ctypes.c_size_t(), ctypes.c_size_t()
    if not k32.GetProcessAffinityMask(process, ctypes.byref(mask), ctypes.byref(system)):
        raise OSError(ctypes.get_last_error(), "GetProcessAffinityMask")
    if mask.value != 1 << CPU:
        raise SystemExit(f"affinity readback {mask.value:#x}, expected {1 << CPU:#x}")
    return mask.value


def load_data(np):
    manifest = json.loads((KIT / "data" / "manifest.json").read_text())
    arrays = {}
    for name, digest in manifest["files"].items():
        path = KIT / "data" / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise SystemExit(f"{path} does not match manifest.json")
        arrays[path.stem] = np.load(path)
    return arrays


def series_of(shape, arrays, smoke):
    """The shape's series and band: a 2-D array for equal lengths, a list of 1-D arrays for (c)."""
    if shape == "c":
        values, lengths = arrays["c_values"], arrays["c_lengths"]
        offsets = [0, *lengths.cumsum()]
        series = [values[offsets[i]:offsets[i + 1]] for i in range(len(lengths))]
        return (series[:SMOKE_N["c"]] if smoke else series), -1
    series = arrays[shape]
    return (series[:SMOKE_N[shape]] if smoke else series), (BAND if shape == "b" else -1)


def fill_cells(series, band):
    lengths = [len(s) for s in series]
    if band < 0:
        total = sum(lengths)
        return (total * total - sum(n * n for n in lengths)) // 2
    if len(set(lengths)) != 1:
        raise SystemExit("a banded shape here has equal lengths")
    return len(lengths) * (len(lengths) - 1) // 2 * band_cells(lengths[0], band)


def time_fill(dtwcpp, series, band, interval):
    """Warm up on the first quarter of the series, then refill the whole set until `interval` s are timed."""
    warm = dtwcpp.Problem("wm-warm-up")
    warm.set_distance(band=band, metric="l1")
    warm.set_data(series[:len(series) // 4])
    start = time.perf_counter()
    while time.perf_counter() - start < 0.2:
        warm.refresh_distance_matrix()
        warm.fill_distance_matrix()
    prob = dtwcpp.Problem("wm")
    prob.set_distance(band=band, metric="l1")
    prob.set_data(series)
    reps, seconds = 0, 0.0
    while seconds < interval:
        prob.refresh_distance_matrix()  # drops the matrix: the next fill computes every pair again
        t0 = time.perf_counter()
        prob.fill_distance_matrix()
        seconds += time.perf_counter() - t0
        reps += 1
    digest = hashlib.sha256(prob.distance_matrix().tobytes()).hexdigest()[:16]
    return reps, seconds, digest


def time_pair(dtwcpp, x, y, band, interval):
    """dtwcpp.distance.dtw(x, y) in a loop, one pair per call; the clock is read every 16 calls."""
    start = time.perf_counter()
    while time.perf_counter() - start < 0.1:
        dtwcpp.distance.dtw(x, y, band=band)
    reps, t0 = 0, time.perf_counter()
    while True:
        for _ in range(16):
            value = dtwcpp.distance.dtw(x, y, band=band)
        reps += 16
        seconds = time.perf_counter() - t0
        if seconds >= interval:
            break
    return reps, seconds, value.hex()


def main():
    shape, threads, smoke = sys.argv[1], int(sys.argv[2]), "--smoke" in sys.argv[3:]
    if shape not in ("a", "b", "c", "d_full", "d_band") or threads not in (1, 24):
        raise SystemExit(__doc__)
    if threads == 1:
        if os.environ.get("OMP_NUM_THREADS") != "1":
            raise SystemExit("threads 1 needs OMP_NUM_THREADS=1 (driver.py sets it)")
        affinity = f"{pin_to_cpu():#x}"
    else:
        if any(k.startswith(("OMP_", "KMP_")) for k in os.environ):
            raise SystemExit("threads 24 runs with no OMP_*/KMP_* variable (driver.py clears them)")
        affinity = "default"

    import numpy as np
    import dtwcpp
    import dtwcpp.test

    module = pathlib.Path(dtwcpp._dtwcpp_core.__file__)
    if not module.is_relative_to(pathlib.Path(sys.prefix)):
        raise SystemExit(f"dtwcpp loaded from {module}, outside this venv {sys.prefix}")
    parallel = dtwcpp.test.parallelisation()
    if parallel["threads_engaged"] != (1 if threads == 1 else os.cpu_count()):
        raise SystemExit(f"threads {threads}: OpenMP engaged {parallel}")

    arrays = load_data(np)
    interval = 0.05 if smoke else 0.5
    if shape.startswith("d_"):
        band = BAND if shape == "d_band" else -1
        x, y = arrays["d"]
        reps, seconds, digest = time_pair(dtwcpp, x, y, band, interval)
        cells = band_cells(len(x), band)
    else:
        series, band = series_of(shape, arrays, smoke)
        reps, seconds, digest = time_fill(dtwcpp, series, band, interval)
        cells = fill_cells(series, band)
    print(f"wheel={pathlib.Path(sys.prefix).name} pyd={hashlib.sha256(module.read_bytes()).hexdigest()[:12]} "
          f"shape={shape} threads={threads} affinity={affinity} omp_max={parallel['max_threads']} "
          f"engaged={parallel['threads_engaged']} smoke={int(smoke)} reps={reps} seconds={seconds:.4f} "
          f"cells_per_rep={cells} gcells={reps * cells / seconds / 1e9:.4f} result={digest}")


if __name__ == "__main__":
    main()
