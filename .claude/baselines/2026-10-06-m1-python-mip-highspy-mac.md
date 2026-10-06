# 2026-10-06 — M1: Python solves the MIP with highspy; the wheel drops HiGHS (Mac)

Unit M1 from design-2.0 33302535. Apple M5 Pro (18 OpenMP threads), macOS (Darwin 25.6.0), Apple clang 21,
`clang-macos` preset (`build/`), MATLAB R2026a (`build-matlab/`, the preset's flags by hand plus
`-DDTWC_BUILD_MATLAB=ON`), Python 3.12.14 through uv, highspy 1.15.1 from PyPI. Commits: a6de1c71 (C++ model
builder), 6aa7048c (Python route, `mip` extra), the step-3 commit (wheel without HiGHS, CI, MEX static HiGHS).
Every number below is [confirmed] by the command named; none is a timing claim.

## Registered before the runs

- C: serial ctest 95 = 94 passed + 1 MAY_SKIP (base). CLI: `dtwc_cl -m mip` outputs byte-identical to the base
  binary. Conformance: regenerated reference differs from the tracked one by the silhouette ulp only (D-19).
- M: matlab_suite 140 run, 139 passed, 0 failed, 1 filtered by assumption (base on this Mac).
- P: with the extra, base 970 passed / 12 skipped + 3 new cases = 973 / 12; without it 969 / 16 (the int64 mip
  case, the wheel smoke and the 2 comparison cases skip; the missing-extra case passes).
- W: the extension below 1.6 MB with no HiGHS object (base 5,034,320 bytes).

## Results

| check | base 33302535 | after |
|---|---|---|
| serial ctest (`ctest --test-dir build -C Release -j1`) | 95: 94 + 1 skip (CUDA) | the same after each of the 3 steps |
| matlab_suite (`ctest --test-dir build-matlab -R matlab_suite -V`) | 140 / 139 / 0 failed / 1 filtered (the orchestrator's base run) | step 1: same; step 3: same (`TIER1_METHODS routed=10/10`, test_cluster_mip ran) |
| MEX `dtwc_mex.mexmaca64` bytes | 1,147,936 (step 1, `@rpath/libhighs.1.dylib`) | 4,557,072, `otool -L` no libhighs, `nm -U` 2,249 highs |
| pytest, extra (`.[test,dev,io,mip]` + matplotlib) | 970 / 12 / 0 (linked HiGHS) | 973 / 12 / 0 |
| pytest, no extra (`.[test,dev,io]` + matplotlib) | — | 969 / 16 / 0 |
| `_dtwcpp_core.cpython-312-darwin.so` bytes | 5,034,320 | 1,203,728 (−3,830,592, −76.1 %) |
| wheel file (`uv build --python 3.12 --wheel`) | 2,407,684 (HiGHS on, step-2 code) | 576,019 |
| `nm -U <so> \| grep -ci highs` | 1,762 | 2: `dtwc::MIP_clustering_byHiGHS`, `dtwc::highs_solver_available` |
| `nm -U <so> \| grep -c "Highs\|HighsTask\|HEkk\|HighsMip\|_ZN5highs"` | — | 0 |
| extension link line (ninja -v, `build.verbose=true`) | `nanobind-static.a libdtwc++.a lib/libhighs.a libomp.dylib` | `nanobind-static.a libdtwc++.a libomp.dylib` |
| wheel CMakeCache | `DTWC_ENABLE_HIGHS:BOOL=ON` | `DTWC_ENABLE_HIGHS:BOOL=OFF` (pyproject `cmake.args`; CMAKE_ARGS only `-DOpenMP_ROOT`) |

Step 2 alone (wheel built with CMAKE_ARGS `-DDTWC_ENABLE_HIGHS=OFF`, pyproject still ON): with highspy
972 / 13 / 0, without 969 / 16 / 0 (the wheel smoke then still asked for linked HiGHS). highspy 1.8.0 (the declared
floor): `tests/python/test_mip.py` + `test_index_types.py` 33 passed.

## The CLI and the highspy route

`cli_mip.sh <dtwc_cl> <out>`: `-i data/dummy --skip-rows 1 --skip-cols 1 -k 3 -m mip`, and `-i synth.csv -k 3 -m mip`
(12 x 20, `random.Random(20261006)`, `gauss` around (i % 3) * 2, 6 decimals; sha256 c1ac99d8…bd9d), plus the synthetic set
with `--no-warm-start`. Base binary copied with its dylibs to `$S/base`, run with `DYLD_LIBRARY_PATH=$S/base`.
`diff -r` of labels, medoids, distance matrix and silhouettes files, and of stdout without its Output/Time lines:
identical after step 1 and after step 3, cold start included. Costs: dummy 148361.91988495924, synthetic
129.372049; medoids dummy {2, 10, 16}, synthetic {0, 2, 7}.

`dtwcpp.cluster(load(...), 3, method="mip")` on the HiGHS-off extension with highspy 1.15.1 (the route): the same
labels, medoids and costs, digit for digit.

`DTWC_CONFORMANCE_REGEN=1 ./build/bin/cpp_conformance`, `git diff` of the reference: the one line
`silhouette,0.96894972764334841` → `0.9689497276433483` at base and after each step (`cmp` of the diffs);
reference restored with `git checkout`.

## The model and its views

`_mip_model` on 4 series, k 2: num_col 16, num_row 17, nnz 44 (3N² − N); `col_cost` float64, `a_start`/`a_index`/
`integrality` int32, read-only, C-contiguous, `owndata` False, the same buffer on two reads, still readable after
the model's name is deleted; an empty `start` without a warm start is `array([], float64)`. HiGHS's pointer
`passModel` with kRowwise transposes the CSR to CSC with ascending rows per column (`HighsSparseMatrix::
ensureColwise`, HiGHS 1.15.1), the order the sorted triplets gave. `model_equiv.py` (a third computation): the
builder's CSR put through a transcription of `ensureColwise` equals a transcription of the old triplets + sort
exactly, and the row bounds, column bounds and integrality are the old ones, for (N, k) = (1,1), (2,1), (3,2), (5,2),
(12,3), (20,4). Costs: both binaries compute `1.0 / max(max_distance * 0.5, 1.0)` with one `fdiv` and scale each
distance with `fmul` (`objdump --disassemble-symbols` of `MIP_clustering_byHiGHS` in the base and new `dtwc_cl`;
`-freciprocal-math`), and the builder's costs equal `d * (1 / s)` bitwise; `d / s` differs by 1 ulp in some entries.

## The highspy floor (hs_floor_api.py: every call the route makes on a 3-variable MIP)

| highspy | Python | result |
|---|---|---|
| 1.5.3 | 3.11 | no `Highs.setSolution`, no `Highs.resetGlobalScheduler` |
| 1.7.1, 1.7.2 | 3.12 | `setSolution` ok; no `resetGlobalScheduler` |
| 1.8.0, 1.8.1, 1.9.0, 1.10.0 | 3.12 | every call ok, optimal |

The route resets HiGHS's per-thread scheduler before and after its run because HiGHS refuses a run whose
`threads` differs from the scheduler the calling thread holds: a caller's own highspy solve with `threads` 2 before
and after three dtwcpp MIP solves (threads 18) all succeed (`sched.py`); a second count without a reset gives
`kError`.

## Two HiGHS builds in one process (found on the way)

The HiGHS-linked extension (step-2 code built with HiGHS on) exports 1,762 highs symbols, 19 of them weak, among them
`HighsTaskExecutor::run_worker`, which highspy's `libhighs.1.15.dylib` exports weak too (34 weak). In that process
a caller's own highspy MIP (a 30-variable knapsack, no dtwcpp call) exits 139, with or without a dtwcpp MIP before it;
beside the HiGHS-off extension it solves (`user_highspy.sh`). The highspy route forced onto the HiGHS-linked
extension exits 139 in `run()` too, so the comparison test skips where HiGHS is linked. The cause (dyld coalescing
weak definitions across images) is [inferred].

## Deleted names

`git grep -c "solver_types\|solver::Triplet\|RowMajor\|ColumnMajor\|isFractional\|isAround\|CompElement" --
':!CHANGELOG.md' ':!.claude/plans/*' ':!.claude/baselines/*'`: no match (exit 1). CHANGELOG keeps them in the
removal entry.

## Lines

a6de1c71 +109 / −235 (8 files); 6aa7048c +220 / −4 (11 files); step 3 +68 / −33 before this record and the brief.

## Not proven here

The Gurobi build (mip_Gurobi.cpp is untouched); CI (the three workflows were read and parsed, not run); Linux and
Windows (the MEX's static HiGHS there; the wheel on manylinux and Windows); MATLAB R2024b (CI's release).

## Merged on the main tree (orchestrator, `c0580948` = `fed1f37d` + `pb/M1`), all `[confirmed]`

`build/` (clang-macos Release, incremental): zero warnings. `ctest -j1`: 100 % of 95, `test_cuda_correctness`
skipped, 41.71 s; `CODEGEN_NO_CALLS tool=clang++ inner_loops=108 calls=0 verdict=PASS`. Conformance regenerated:
the one silhouette ulp only (D-19). `check_docs` PASS, `check_pins` 0 failures, `generate_docs --check` current.
`build-matlab/` reconfigured by the merge (HiGHS now static for a MATLAB-on tree): zero warnings;
`dtwc_mex.mexmaca64` 4,557,072 bytes, `otool -L` lists no libhighs; `matlab_suite: 140 run, 139 passed, 0 failed,
1 incomplete` (the registered filter). Fresh venv, `uv pip install ".[test,dev,io,mip]" matplotlib pandas`:
**974 passed, 11 skipped, 0 failed, 97.83 s** (the agent's 973/12 plus the pandas DataFrame case, now installed; skips:
9 CUDA, 1 GPU present, 1 scipy present). The CI Python job now installs those extras too (`python-tests.yml`, edited
here, not run here).
