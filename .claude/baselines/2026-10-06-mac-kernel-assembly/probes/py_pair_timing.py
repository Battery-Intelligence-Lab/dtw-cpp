"""Times the wheel's single-pair dtwcpp.dtw (the -Os binding TU's distance::dtw<double>) per DP cell.
Random walks, L1 (default) and squared, unbanded and band L/10; 7 repetitions of a batch; ns per cell."""
import statistics, sys, time
import numpy as np
from dtwcpp import _dtwcpp_core as core
rng = np.random.default_rng(1)
out = []
for L in (100, 1000):
    xs = [np.cumsum(rng.uniform(-0.5, 0.5, L)) for _ in range(16)]
    for band in (-1, L // 10):
        cells = L * L if band < 0 else sum(min(L, j + band + 1) - max(0, j - band) for j in range(L))
        for metric in ("l1", "squared_euclidean"):
            calls = max(8, int(4e7 // cells))
            reps = []
            for r in range(7):
                t0 = time.perf_counter_ns()
                for c in range(calls):
                    core.dtw(xs[c % 16], xs[(c + 1 + c // 16) % 16], band=band, metric=metric)
                reps.append((time.perf_counter_ns() - t0) / (calls * cells))
            line = f"py dtwcpp.dtw L={L:5d} band={band:4d} metric={metric:10s} ns/cell median {statistics.median(reps):.4f} min {min(reps):.4f}"
            print(line, flush=True)
# z_normalize (binding TU, header-inlined): ns per element on a 1e6 array (copy in, three passes)
import dtwcpp as _d
zn = getattr(core, "z_normalize", None) or getattr(_d, "z_normalize", None)
if zn is not None:
    v = rng.normal(5.0, 2.0, 1_000_000)
    reps = []
    for r in range(7):
        t0 = time.perf_counter_ns()
        for _ in range(20):
            zn(v)
        reps.append((time.perf_counter_ns() - t0) / (20 * v.size))
    print(f"py z_normalize n=1e6 ns/element median {statistics.median(reps):.4f} min {min(reps):.4f}", flush=True)
