# 2026-10-01 — W7g: one `distance.dtw` per language

Question: does the one Python entry (`_dtwcpp_core.dtw` over `dtwc::distance::dtw`, GIL released, zero-copy
float64 arrays) cost anything against the base's `distance.dtw` → `standard` → `dtw_distance` route?

Branch pb/W7g, base 143f138 (pb/W7c). Commits fdb005d (Python), f2f48bf (MATLAB). Logs `C:/D/git/wt/W7g-*.log`
(disposable); scripts were scratch.

## Method [confirmed]

MSVC wheels of base and head, each `uv pip install --reinstall`ed into its own venv (`venv/W7g-base`,
`venv/W7g`), `CMAKE_GENERATOR` unset. `x`, `y`: `default_rng(20261001).standard_normal(10_000)` each; one warm-up
call, then 7 timed calls of `dtwcpp.distance.dtw(x, y)` (full L1 DTW, 10^8 cells), `time.perf_counter`. Base and
head interleaved, two rounds. The machine was shared with other agents' builds.

## Results [inferred: under load]

| Round | Base min / median (s) | Head min / median (s) |
|---|---|---|
| 1 | 0.0879 / 0.0881 | 0.0879 / 0.0883 |
| 2 | 0.1314 / 0.1319 | 0.1298 / 0.1318 |

Both return `0x1.6b86f74f1a321p+12` [confirmed]. Four threads, one call each, take 1.02× (base) and 1.03× (head)
one call's time: the GIL is released in both [inferred: under load]. Conclusion: no slowdown; the entry's
overhead (parsing four names) is invisible at this size.
