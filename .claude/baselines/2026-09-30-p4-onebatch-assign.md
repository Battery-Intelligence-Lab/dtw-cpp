# 2026-09-30 — P4: OneBatchPAM's final exact assignment runs in parallel

Question: `one_batch_pam` labels each of the N series against the k medoids after the swap phase. At the base
(`1fc8776`) that loop is serial (`dtwc/algorithms/one_batch_pam.cpp`, "exact() updates the evaluation counter, so
keep this loop serial"). Does running it through `run_openmp`, one thread per point, make the call faster without
changing a label, a medoid, the total cost or the evaluation count?

## Model and bands, registered BEFORE any code change [registered]

Workload (N = 2000, k = 10, L = 200 random-walk series, full band, L1, automatic batch): the batch is
m = 20·ceil(log2(2001)) = 220. The table fill is m(N-1) = 439,780 DTW calls and is already parallel. The final
assignment is j(N-1) calls, j = the medoids that are not in the batch (j <= 10): <= 19,990 calls, serial at base.
Every call costs the same T (equal lengths, full band). With P_eff the parallel throughput in single-thread units
(24 logical CPUs, 8 P + 16 E cores, shared machine: P_eff between 12 and 20):

- fill time ≈ 439,780·T / P_eff, base assignment time ≈ 19,990·T, so base assignment / fill ≈ 0.045·P_eff = 0.5–0.9.
- The swap sweeps read the N×m table only (no DTW call): O(N·m) cheap operations per sweep, negligible against T.

| id | quantity | band | predicted |
|---|---|---|---|
| B1 | assignment phase wall time, base / head, 24 threads, median of 5 | >= 4x | PASS, about P_eff (12–20x); FAIL if < 4x |
| B2 | whole `one_batch_pam` wall time, base / head, 24 threads, median of 5 (the brief's "head at least 4x faster than base", read literally) | >= 4x | FALSIFIED. Head time is at least the fill, so the ratio is bounded by (fill + base assignment) / fill = 1.5–1.9; 4x needs base assignment >= 3 fill, i.e. P_eff >= 66 |
| B3 | hash of labels + medoids + total_cost bits, and `distance_evaluations`: base, head at 1 thread, head at 24 threads | identical (exact) | identical: a point's label, distance and count are written by its one thread and the cost is summed in point order after the region |
| B4 | head at 1 thread vs base, whole call (serial cost of the change: the function pointer is resolved once instead of calling the `Problem::dtw_function` accessor per pair) | within 10% of 1.0 | about 1.0 (advisory) |
| B5 | TSan (R1 WSL harness) on `unit_test_one_batch_pam`; assembly of the assignment loop | 0 reports; no added call or lock | 0 reports |

Timing protocol: base and head binaries of the same scratch benchmark, run alternately (base, head, base, ...),
5 repetitions each (6 were run at 24 threads and 3 at 1 thread; see below), `OMP_NUM_THREADS=24`, the machine shared (load noted at each run), medians. The phase time is
from a scratch timer around the assignment loop, inserted into both trees and not committed. Wall-clock is
advisory; B3 and B5 decide.

## Change [confirmed]

`dtwc/algorithms/one_batch_pam.cpp` (commit `9168ed9`, base `1fc8776`): the final scan is `assign_point`, run by
`run_openmp(assign_point, n, n > 64)` (the guard the table fill already uses). Point p writes `labels[p]`,
`point_cost[p]`, `point_evaluations[p]`; `exact()` is const and adds to the caller's tally; the tallies and the
objective are combined serially after the region. The dispatcher is resolved once in the constructor and read by
`FixedBatchDistances::dtw()` (the table fill and the assignment shared two copies of the f32/f64 branch). No
`omp critical`, `omp atomic`, `std::mutex` or `std::atomic` in `dtwc/algorithms` (`git grep`: empty).

## Timing [confirmed]

Intel Core Ultra 9 285, 24 logical CPUs, shared: other agents' CPU load 21–67% before each 24-thread run, 65–90% before
each 1-thread run. Scratch benchmark (`N = 2000, k = 10, L = 200`, random walks from `mt19937_64(20260930)`, automatic batch m = 220,
full band, L1) built from base and from head, each with the same scratch stage timers; 6 alternating runs at
`OMP_NUM_THREADS=24`, 3 at 1. Medians, seconds:

| run | threads | fill | swap | assignment | whole call |
|---|---|---|---|---|---|
| base | 24 | 0.920 | 0.0040 | 0.904 (0.893–0.909) | 1.823 |
| head | 24 | 0.917 | 0.0041 | 0.0414 (0.0393–0.0426) | 0.963 |
| base | 1 | 20.02 | 0.0037 | 0.897 | 20.92 |
| head | 1 | 19.37 | 0.0045 | 0.914 | 20.29 |

Every run, both binaries, both thread counts: `evals = 459770` (= 439,780 table + 19,990 assignment: all 10 medoids
outside the batch), `cost = 578760.0340637482`, FNV-1a hash of labels + medoids + cost bits `260f15542ca87325`.

| id | result | verdict |
|---|---|---|
| B1 assignment phase, base / head, 24 threads | 21.9x (0.904 s -> 0.0414 s) | PASS (>= 4x). The prediction (12–20x) was 10% low: the fill's own speed-up, 20.0 s at 1 thread over 0.92 s at 24, is 21.8x, so P_eff was about 22 |
| B2 whole call, base / head, 24 threads | 1.89x (1.823 s -> 0.963 s) | FALSIFIED as predicted (needs >= 4x). The fill (0.92 s, already parallel) is now 95% of the call; even a free assignment gives 1.823 / 0.921 = 1.98x |
| B3 identity | hash, cost and `evals` equal in all 18 runs; `unit_test_one_batch_pam` pins 1 thread = 4 threads = a serial scan | PASS |
| B4 head at 1 thread vs base | whole call 1.03x (20.92 s -> 20.29 s), assignment 0.90 s vs 0.91 s: the per-pair accessor calls removed from the loop are below the noise of the load | PASS (within 10%) |

Reading. The brief's "at least 4x on N = 2000, k = 10, L = 200" holds for the step P4 changes (21.9x) and cannot hold
for the whole call, because the call is dominated by the parallel table fill: m(N-1) = 439,780 calls against the
assignment's 19,990. By the cost model (not measured) the whole-call factor grows with k (more out-of-batch medoids)
and shrinks with m. The CHANGELOG line states both factors.

## Assembly (clang 21.1.8, the flags of `build/compile_commands.json`, `-flto` dropped, `-S -masm=intel`) [confirmed]

| loop | base | head |
|---|---|---|
| final assignment, calls per pair | 4 accessor calls (`preflight_float32_distance_semantics`, `ensure_dtw_function_configuration_current`, `validate_fill_request_once`, `validated_dtw_function_f32`; the Float64 arm has its three) then `call qword ptr [rax+16]` (the `std::function`) | `call qword ptr [rax+16]` only; the lambda's other calls are the cold throws (`throw_nonfinite_medoid_distance`, `_Xbad_function_call`, `throw_wrong_precision` x2) |
| `run_openmp` wrapper of the assignment | none (inline serial loop) | `omp_get_thread_num`, `__kmpc_dispatch_init_8/next_8/deinit`, the slot's `__ExceptionPtr*` calls: the same set as the fill's |
| table fill row | lambda called from `run_openmp`; one `call qword ptr [rax+16]` | lambda inlined into `run_openmp`; two `call qword ptr [rax+16]` (the Float32 and the Float64 arm); no library call added |
| `lock`, `xchg`, `cmpxchg` in the TU | 0 | 0 |

The assembly files, benchmark sources and logs are scratch (`C:/D/git/wt/tmp/P4/`: `obp_base.s`, `obp_head.s`,
`bench_24.log`, `bench_1.log`, `ctest_base.log`, `ctest_head.log`, `pytest_head.log`); the benchmark and the stage timers
were never committed.

## Gates [confirmed]

- Serial `ctest -j1 -C Release`, base `1fc8776`: 123 tests, 100% passed, 0 failed, 3 skipped (`test_cuda_correctness`,
  `test_metal_correctness`, `test_metal_mmap`, all MAY_SKIP). Head: the same, 123, 3 skipped, the same three.
  `cpp_conformance` passes against the unchanged `conformance_reference.txt`; that reference has no OneBatchPAM row, so
  B3 and the unit tests are the identity evidence.
- New tests (`unit_test_one_batch_pam`, 11 cases, 10554 assertions): the final assignment is the serial scan at 1 and 4
  threads; a parallel non-finite distance gives the same message at both. Bite checks (source mutated, test rebuilt,
  source restored): dropping `point_evaluations[point] = calls` fails `outside >= 1` (3/3 runs); hoisting `best`,
  `label`, `calls` out of the lambda fails the labels/outcome checks (3/3 runs); counting into the shared
  `distances.evaluations` fails the count check in 2 of 3 runs (a race is not deterministic; TSan below is).
- TSan, R1's harness (WSL2, clang 18.1.3, LLVM libomp + Archer, `OMP_NUM_THREADS=8`), a clone of `pb/P4` at `9168ed9`
  in `~/dtwc-tsan-p4`, `unit_test_one_batch_pam` (11 cases, 10554 assertions): exit 0, 0 `WARNING: ThreadSanitizer`.
  Bite: with every worker adding to the shared `distances.evaluations`, exit 66 and 8 reports, all at
  `one_batch_pam.cpp:346` (the mutated line). The positive control of the harness is R1's (`2026-09-30-r1-tsan.md`).
  Log: `~/tsan/logs-p4/unit_test_one_batch_pam-head.log` (WSL).
- Python gate: fresh venv (`C:/D/git/wt/venv/P4`, wheel built from the worktree with `--reinstall`), `DTWC_CL_PATH` the
  worktree's `dtwc_cl.exe`: 1172 passed, 19 skipped, 0 failed (4 min 7 s). `tests/python` is untouched by this unit, so
  its collection is the base's; no base wheel was built to count it (the 19 skips equal the last recorded count).
  `test_api.py` and `test_sklearn_estimator.py` run `method="onebatch"` through the binding.
- `check_docs.py --cli build/bin/dtwc_cl.exe` (VERDICT=PASS, 385 flags), `check_pins.py` (0 failures) and
  `generate_docs.py --check` (current).

## Conclusion

The serial assignment was 19,990 DTW calls of about 45 µs on one thread (0.90 s); the per-pair accessor calls were not
its cost (the head at 1 thread takes 0.91 s). With one writer per point it runs on every thread without a lock, and
labels, medoids, cost and call count are unchanged at 1 and 24 threads. Step speed-up 21.9x; whole-call speed-up 1.89x,
bounded by the table fill. Nothing remains open in this unit. What the whole call still spends is the table fill (0.92 s
of 0.96 s); whether the lane kernel of the distance-matrix fill can serve its (series, batch column) pairs is not
examined here.
