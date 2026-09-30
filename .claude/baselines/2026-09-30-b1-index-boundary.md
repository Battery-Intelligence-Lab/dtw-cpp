# 2026-09-30 — B1: the Python and MATLAB boundaries check the indices they pass into unchecked C++

Question: after W6m's `dist_by_ind` checks, which other binding entry points hand a raw index or count to unchecked
C++, and what did each do at the base?

Branch pb/B1, base 6c102b3. Commit 2f8dc96 (Python). No MATLAB commit: the MEX has no `series`, `name` or
`centroid_of` command (below). Logs are `C:/D/git/wt/B1-*.log` (disposable); the probe scripts were scratch.

## Method [confirmed]

Python: MSVC wheel, `uv pip install --reinstall` into a fresh venv, `CMAKE_GENERATOR` unset; every probe in its own
interpreter (an access violation is exit 3221225477). N = 3 series `[[1,2,3],[4,5,6],[1.5,2.5,3.5]]` (N = 4 for the
count probes). MATLAB: clang MEX of the same tree (`build-mex`, R2024b), one `matlab -batch` per probe group.

## Base outcomes, Python (N = 3) [confirmed]

| Call | Base |
|---|---|
| `series(3)` / `series(100)` / `series(1000000)` / `series(-1)` | ValueError "vector too long" / MemoryError / returned `[]` / TypeError |
| `series_name(3)` / `(100)` / `(1000000)` / `(-1)` | MemoryError / MemoryError / returned `''` / TypeError |
| `series`, `series_name`, `centroid_of` on an empty Problem | access violation |
| `centroid_of(0)`, `centroid_of(-1)` before any clustering | access violation |
| `centroid_of(3)` after `cluster()` / `centroid_of(-1)` after `cluster()` | returned 0 / access violation |
| `cluster()` k = 2, then `set_n_clusters(1)`, `centroid_of(i)` of a series in cluster 1 | returned 1 from a one-element medoid list |
| `centroids_ind = [99]`, then `centroid_of(0)` and `inertia` | 99 and 7.27e223 |
| `clusters_ind = [0]` (wrong length), then `centroid_of(2)` | access violation |

After 2f8dc96 each of these raises `InvalidInput` ("series: i = 3 is outside [0, N) with N = 3."; "centroid_of: this Problem
holds no clustering; cluster it first."; "centroid_of: series 1 has label 1 but the Problem holds 1 medoids; cluster it
again.") or, for the two assignments, `AttributeError` (the fields are read-only; `Problem.set_result(ClusteringResult)`
is the write route). Left as it was: after `set_n_clusters(k)` alone `resize()` zero-fills both vectors, so
`centroid_of(0)` returns 0 and `inertia` a number; a Problem cannot tell that state from a clustering without a flag.

## The MEX [confirmed]

`dtwc_mex('Problem_series' | 'Problem_series_name' | 'Problem_centroid_of', ...)` is `Unknown command`. The only command
that takes a series index is `Problem_dist_by_ind` (W6m); dendrogram merge ids are checked in C++ (`hierarchical.cpp:188`)
and in the MEX. `matlab_suite` on the base build: `141 run, 140 passed, 0 failed, 1 incomplete`; `git diff 6c102b3 HEAD --
bindings dtwc tests/matlab` is empty, so the same MEX and suite stand at HEAD.

## Other entry points that pass a raw count or state into unchecked C++ (listed, not fixed) [confirmed at 2f8dc96]

1. `find_total_cost()` on a Problem never clustered (data only, or filled): access violation in Python, and MATLAB
   `Problem_find_total_cost` (R2024b: "Access violation", exit 0xc0000005). Python `write_clusters()` before clustering:
   access violation. `write_silhouettes()` and the MEX scores raise "Cluster before calculating ...", so the guard
   exists there, not in `find_total_cost` / `write_clusters`.
2. `set_n_clusters(k)` accepts k = 0 and k > N in both bindings, and k = 2**31 - 1 in Python (a 2**31-element medoid
   vector); k = -1 is `ValueError: vector too long` / `dtwc:internal` (an accident of `resize`). `cluster()` rejects
   k outside [1, N] later, so the error arrives at another call.
3. `set_band(-5)` is accepted in both (Python then fills and `dist_by_ind(0,1)` returns 9.0); what a band below -1 means is
   not established.
4. MEX `fast_pam(h, 2, max_iter)` accepts 0 and -1 (returns a struct); MEX `Problem_set_cuda_settings(h, -1)` is accepted.

Checked already, same probes: `fast_pam` / `fast_clara` k, n_samples, max_iter and sample_size 0; `cut_dendrogram` k;
`build_dendrogram` max_points; `tier1_cluster` k; `set_max_iter(0)`; `set_n_repetitions(0)`; `set_data` ndim <= 0;
`dtw_barycenter` series_indices `[7]`.

## Gates [confirmed]

- Python, base 6c102b3 with its own tests (B1b worktree, `build/` junctioned): `1109 passed, 19 skipped` = the brief's
  last green. Same wheel, the 27 new tests: `26 failed, 1 passed` (`test_last_index_is_valid`).
- Python, 2f8dc96, rebuilt wheel: `1136 passed, 19 skipped, 0 failed` = 1109 + 27 (26 in `test_problem.py`, 1 in
  `test_contract_parity.py`).
- `check_docs.py --cli build/bin/dtwc_cl.exe` PASS; `check_pins.py` failures=0; `generate_docs.py --check` current.
