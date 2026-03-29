import numpy as np
import time

def random_series(length, seed):
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=length)

results = {}
for L in [1000, 4000, 8000, 16000]:
    x = random_series(L, 42)
    y = random_series(L, 43)
    results[L] = {}

    # dtwcpp
    import dtwcpp
    t0 = time.perf_counter()
    d = dtwcpp.dtw_distance(x, y)
    results[L]['dtwcpp'] = time.perf_counter() - t0

    # dtaidistance
    from dtaidistance import dtw
    t0 = time.perf_counter()
    d = dtw.distance_fast(x, y)
    results[L]['dtaidistance'] = time.perf_counter() - t0

    # aeon
    from aeon.distances import dtw_distance
    t0 = time.perf_counter()
    d = dtw_distance(x, y)
    results[L]['aeon'] = time.perf_counter() - t0

    # tslearn
    from tslearn.metrics import dtw as tslearn_dtw
    t0 = time.perf_counter()
    if L <= 8000:  # tslearn may be very slow at L=16000
        d = tslearn_dtw(x.reshape(-1, 1), y.reshape(-1, 1))
        results[L]['tslearn'] = time.perf_counter() - t0
    else:
        results[L]['tslearn'] = None

for L, libs in results.items():
    print(f"\nL={L}:")
    for lib, t in libs.items():
        if t is not None:
            print(f"  {lib:20s} {t*1000:10.1f} ms")
        else:
            print(f"  {lib:20s}    skipped")
    # Compute ratios vs dtwcpp
    if 'dtwcpp' in libs and libs['dtwcpp']:
        base = libs['dtwcpp']
        for lib, t in libs.items():
            if t and lib != 'dtwcpp':
                print(f"  {lib} / dtwcpp = {base/t:.2f}x")
