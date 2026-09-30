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
5 repetitions each, `OMP_NUM_THREADS=24`, the machine shared (load noted at each run), medians. The phase time is
from a scratch timer around the assignment loop, inserted into both trees and not committed. Wall-clock is
advisory; B3 and B5 decide.
