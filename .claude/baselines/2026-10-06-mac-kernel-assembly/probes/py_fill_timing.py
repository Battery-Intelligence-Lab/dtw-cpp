"""Times dtwcpp.compute_distance_matrix (the wheel's Problem fill) on ragged series (lengths 90..110:
the per-pair path) and on equal-length series (the lanes path), at the OpenMP default (all cores)
and OMP_NUM_THREADS from the environment; 5 repetitions; ms and Gcell/s."""
import os, statistics, time
import numpy as np
import dtwcpp
rng = np.random.default_rng(3)
def walk(n): return np.cumsum(rng.uniform(-0.5, 0.5, n))
shapes = {
    "ragged N1000 L100+-10 band-1": [walk(int(n)) for n in rng.integers(90, 111, 1000)],
    "equal  N2000 L100     band-1": [walk(100) for _ in range(2000)],
}
for name, S in shapes.items():
    lens = np.array([len(s) for s in S], dtype=float)
    cells = (lens.sum() ** 2 - (lens ** 2).sum()) / 2
    ts, ref = [], None
    for r in range(5):
        t0 = time.perf_counter()
        D = dtwcpp.compute_distance_matrix(S, band=-1, metric="l1", device="cpu")
        ts.append(time.perf_counter() - t0)
        ref = D if ref is None else ref
    print(f"py fill {name} threads={os.environ.get('OMP_NUM_THREADS', 'default')}: ms median {1e3 * statistics.median(ts):.1f} min {1e3 * min(ts):.1f}  Gcell/s {cells / statistics.median(ts) / 1e9:.2f}  checksum {float(np.nansum(D)):.6e}", flush=True)
