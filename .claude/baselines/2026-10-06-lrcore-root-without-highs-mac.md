# 2026-10-06 — lrcore without HiGHS: the subgradient root against the Kelley root (Mac)

Question (M1 review): the wheel links no HiGHS, so `lagrangian_root_exact` (`lagrangian_root.cpp:590`) runs the subgradient root (`lagrangian_root`, cap 4000 iterations) instead of the
Kelley cutting-plane root (`lagrangian_root_kelley`, cap 500 majors) and branch-and-bound closes the gap. What does `-m lrcore` lose: root certification, nodes, iterations, wall time, and is
the answer the same? Measurement only, no product change. Apple M5 Pro (18 threads), macOS 26.6.2, Apple clang 21.0.0, CMake 4.4.3, worktree of design-2.0 bf82dc1b. Counters, bounds,
costs, labels, medoids: [confirmed] (each tabled input ran twice per build, dtwc_cl and probe: 194 pairs, r1 = r2 in all; MIP, thread and OpenMP-scratch runs once). Wall-clock: [inferred] (shared machine, 1-min load average 2.7-50.3, median 8.9).

## Builds, method, inputs
- ON `build/`: `cmake --preset clang-macos -DOpenMP_ROOT=/opt/homebrew/opt/libomp` (HiGHS v1.15.1 shared). OFF `build-nohighs/`: `cmake -S . -B build-nohighs -G Ninja
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ -DOpenMP_ROOT=/opt/homebrew/opt/libomp -DDTWC_ENABLE_HIGHS=OFF -DDTWC_BUILD_TESTING=OFF`
  (the wheel's `cmake.args` has `-DDTWC_ENABLE_HIGHS=OFF` and builds with arch level v3: same LR code). Each: `cmake --build <tree> --target dtwc_cl`. The caches differ in DTWC_ENABLE_HIGHS and DTWC_BUILD_TESTING only
  (+ COVERAGE=OFF and the HiGHS / Catch2 sub-project options, ON tree). `ninja -t commands dtwc_cl`: every library file has equal flags (-O3 -flto=thin -march=native, fast-math) but `-DDTWC_ENABLE_HIGHS` (+ `-Wno-invalid-offsetof` on 3 mip files).
- `dtwc_cl -v` prints no LR counter (progress, `cost=` to 6 digits, the clock; nodes only in the SolverError at the cap). So each input ran per build, twice, through (1) `<tree>/bin/dtwc_cl -i <in>
  --skip-rows 1 --skip-cols 1 -k <k> -m lrcore -v -o <dir>` (exit code, `Total cost`, SHA-256 of labels / medoids / matrix files) and (2) `lrcore_probe` (scratch, NOT in the repo; dtwc_cl.cpp's flags, linked to each
  tree's libdtwc++.a): the CLI's argv -> Config -> Problem as `run()`, the preamble of `LR_core_clustering` (matrix, FastPAM UB at --seed 42, dense D), then `lagrangian_root`, `lagrangian_root_kelley` (a SolverError
  in OFF), `lagrangian_root_exact` (--lr-max-nodes 2000000), `prob.cluster()`; doubles `%.17g`. iterations = subgradient iterations / Kelley majors (501 = cap of 500 used up); LR ms = exact stage (root + tree). 900 s cap: only X7's compact MIP
  passed it (> 15 min, not finished). Checks: probe `cluster()` = dtwc_cl on 192 of 192 pairs (cost digits, medoids, SolverError text); `lagrangian_root_exact`'s multipliers = the root the build uses on 214 of 214 probes; costs recomputed from
  the CLI's matrix CSV (`math.fsum`) within 1.7e-15 on 70 runs, every label its nearest medoid; `dummy`: brute force over C(25,3) triples = the same cost and medoids; compact MIP (`dtwc_cl -m mip --mip-gap 0`, ON tree, unregistered): Findings 1, 4.
  Scripts in `$S`: build_probe.sh on|off; run_all.py <ID[:k=K][:seed=S],..> on,off 2; run_mip.py; run_threads.py; build_probe_omp.sh; run_omp.py; run_omp_ab.py; collate/stats/determinism/crosscheck/recompute/mip_final.py (uv run --no-project python).
- Inputs (stdlib generators, data and every log in `$S` = the session scratchpad `.../scratchpad/lrcore`, not in the repo; full-band L1 DTW, length 40, one draw per set): `data/dummy` (N=25, `-k 3`; cpp_conformance runs pam, no lrcore);
  `gen_data.py` (4e11e94ade93): A N=200, B N=800, C N=200 = 4 templates (sin 1 period, sin 2 periods, ramp, bump) + N(0, s), s = 0.1 / 0.1 / 0.5, k=4, shuffled, random.Random(20261006) (A, C share draws) and 20261007; `gen_stress.py` (95e78154b1c5,
  seed 20261008+i, i = 1..9 for X1..X9, 10 Z1, 11 Z2): X1 N=800 s0.5, X2 s0.5 k8, X3 s1, X4 s2 (templates + noise, shuffled), X5-X7 random walks, X8-X9 i.i.d. noise, Z1 N=3200 s0.1, Z2 N=1600 s0.5 (shuffled); `gen_line.py` (29649878b44b, seeds
  20261030..35): Y1..Y6 constant series c_i = 100 (i mod 3) + U(-1,1) + 0.001 i (Y2: 10, +-5), k=3, in index order, so D = 40 |c_i - c_j|, a line metric. `<id>k<k>` changes -k, `<id>s<s>` --seed; N and k: tables and the capped line.

## Registered before each round (`$S/registered*.md`, sha256[:12], local time) and the verdict (FALSIFIED is a result)
- R 9b30472d83a3 14:43: equal answers; Kelley certifies D, A, B, C <= 1e4 nodes; OFF certifies D, A, not B; nodes_OFF 1-100x ON on C; OFF <= 100x ON: CONFIRMED except B (OFF certifies, 21 it), C (0 nodes OFF, 1,157 ON): FALSIFIED.
- S ae483c9efc49 14:49 (X1-X9): OFF misses the root where Kelley certifies; node cap only on the noise sets; nodes_OFF >= nodes_ON: FALSIFIED (Kelley certified no X; X2 capped; X4, X8 fewer nodes OFF); equal answers, no LR run killed: CONFIRMED.
- T 42ef1e750db5 14:55: a different frontier k per build FALSIFIED (X6 data k <= 6, X4 data k = 4, both); seeds 1-5 on X3, X5, X1: same cost, OFF root 15/15, ON 0/15: CONFIRMED; Z1 both certify at the root FALSIFIED (224 majors); Z2 CONFIRMED.
- U fdeedbfc615d 15:00, V 379429304c57 15:02, 3e8324709237 15:06 (Y1-Y6): Kelley <= 50 majors CONFIRMED (13, 13, 12, 8), Y2 <= 100 FALSIFIED (414); OFF gap <= 1e-4 on Y1 CONFIRMED (3.72e-5); OFF 4000 iterations at every N and OFF time >= 3x per
   doubling FALSIFIED (Y5 certifies at 2,585; CLI 69.98 s vs 28.04 s on Y4); Y6 CLI 3x-5x of Y5, < 15 min: 4.2x on r1 CONFIRMED, 5.5x on r2 FALSIFIED.
- W 347e3b119870 15:03 (Y k = 6, 10, 20): Kelley certifies at the root on >= 4 of 6 FALSIFIED (0); OFF caps where ON certifies FALSIFIED (the reverse: Y1k6, Y3k6 cap in ON only).
- Q bc8b777b0e2e 15:13 (1 thread): same counters CONFIRMED (12 of 12), 1 thread >= 2x slower FALSIFIED (LR is serial); O 538f9b2e21e0 15:15 (scratch -fopenmp): same fields CONFIRMED (10 of 10), OFF root >= 3x faster on Y4, Y5 CONFIRMED.

## The four requested inputs, every number as printed (cost, medoids, labels SHA-256 equal in ON and OFF; fill ms = DTW matrix, for scale)
```text
input (N,k)   build root         cert iters root LB                 root gap               core nodes LR ms   CLI Time  fill ms
dummy (25,3)  ON   Kelley      yes     76 148361.91988495915     5.885033803681092e-16    6      0    5.082    1.888s   1447.2
dummy (25,3)  OFF  subgradient yes     17 148361.91988495924     0                        4      0    0.017    1.482s   1442.5
A (200,4)     ON   Kelley      yes     56 604.4265550000007      -7.523616345520489e-16   6      0   12.994  0.02256s      0.8
A (200,4)     OFF  subgradient yes     17 604.426555             3.7618081727602447e-16   4      0    0.586 0.008984s      0.8
B (800,4)     ON   Kelley      yes     27 2422.5037709999897     3.566640337272286e-15    4      0   56.671   0.1214s      9.3
B (800,4)     OFF  subgradient yes     21 2422.503770999998      1.8771791248801504e-16   4      0   16.699   0.0769s     10.5
C (200,4)     ON   Kelley      no     501 2531.491763335803      8.924125484666247e-05   12   1157  291.844    0.313s      0.9
C (200,4)     OFF  subgradient yes     28 2531.7176969999996     5.388602585019541e-16    5      0    0.963    0.011s      1.5
answers ON = OFF: dummy: cost 148361.91988495924, medoids [2, 10, 16], labels 4004360e77fd  |  A: cost 604.4265550000002, medoids [2, 47, 157, 190], labels 2db617e41a61
                    B: cost 2422.5037709999983, medoids [70, 476, 488, 636], labels ec70c9174be4  |  C: cost 2531.717697000001, medoids [10, 47, 54, 196], labels 0973af760c4e
```

## More inputs (ON = Kelley root; OFF = subgradient root; gaps to 3 digits, LR ms rounded; cost verbatim; `=`: cost, medoids, labels files identical)
```text
id        N  k | ON: cert majors  gap       core  nodes   LR ms | OFF: cert  iters  gap       core  nodes   LR ms | answer cost
X1       800  4 | no   501  2.23e-04  11     749   3393 | YES   49  3.62e-16   5       0     68 | =        10047.03983
X3       200  4 | no   501  1.41e-03  16    3373    522 | YES   29  7.67e-07   4       0      1 | =        4805.517453000001
X4       200  4 | no   501  5.80e-03  63  156147   5637 | no  4000  5.63e-03  59  145795    221 | =        9275.629668000001
X5       200  4 | no   501  5.89e-03  35   50331    967 | YES  123  8.86e-07   4       0      8 | =        8747.844634000008
X8       200  4 | no   501  1.07e-03  15    3639   1417 | no  4000  4.07e-04  14    2089    192 | =        4339.357224999997
X6k5     200  5 | no   501  8.55e-03  55  853079   1527 | no  4000  1.21e-02  45  649925    349 | =        7733.510131000004
X6k6     200  6 | no   501  1.18e-03  27  487365    779 | YES  195  9.97e-07  10       0     10 | =        7201.7867229999965
Z2      1600  4 | no   501  1.14e-04  14    2729   6866 | YES   66  7.14e-07   6       0    295 | =        20267.267666999996
Z1      3200  4 | no   224  1.87e-05   5      19   9370 | YES   49  6.98e-07   7       0    906 | =        9652.883812000016
Y3       200  3 | YES   13  1.29e-13   5       0      5 | no  4000  7.46e-06   7      41    195 | =        3701.6139599999997
Y1       800  3 | YES   13  6.47e-13   5       0     34 | no  4000  3.72e-05  25    1657   4539 | =        16834.787040000025
Y2       800  3 | YES  414  7.36e-07  43       0   5025 | no  4000  1.16e-05  23    2589   4668 | =        78251.65444000009
Y4      1600  3 | YES   12  9.47e-07   6       0    162 | no  4000  9.49e-06  44    4869  37213 | =        37612.18572000003
Y5      3200  3 | YES    8  2.31e-07   7       0    411 | YES 2585  8.37e-07  31       0  53944 | =        115393.94399999999
Y6      6400  3 | CLI only, no probe: dtwc_cl Time ON 0:6.898 s | OFF 4:55.37 s | =        424709.96975999966
Y1k6     800  6 | no   501  3.42e-03 629     CAP   6489 | no  4000  3.83e-05  30  158757   4504 | ON FAILS 8663.682039999992
Y3k6     200  6 | no   501  3.39e-03 116     CAP   1226 | no  4000  3.21e-04  26     385    204 | ON FAILS 1891.2166399999978
capped in both builds (13; (N,k)): X2 (200,8) X6 (200,10) X7 (800,8) X9 (200,10) X6k7 (200,7) X6k8 (200,8) X4k5 (200,5) X4k6 (200,6) X4k8 (200,8) Y1k10 (800,10) Y1k20 (800,20) Y3k10 (200,10) Y3k20 (200,20); root gap ON 7.31e-04..9.50e-03, OFF 6.24e-04..3.15e-02; OFF root LB higher on 12 of 13 (not Y3k10)
```

## Findings
1. Answer [confirmed]: both builds close 19 of 34 inputs (33 probed + Y6 by the CLI alone) and on all 19 the cost digits, medoids and labels files are identical; 13 hit the node cap in both (same SolverError, exit 1); 2 (Y1k6, Y3k6) close in OFF
   only. "Certified" = relative gap <= 1e-6 (`lagrangian_root.cpp:52,319`): OFF's root certificates sit at 6.98e-7..9.97e-7 on X3, X5, X6k6, Z1, Z2, Y5, ON's at 7.36e-7, 9.47e-7, 2.31e-7 on Y2, Y4, Y5 (the others <= 1e-12). Independent optimum: brute
   force on dummy, and the compact MIP (HiGHS via `dtwc_cl -m mip --mip-gap 0` in the ON tree, not highspy) on the 15 inputs LR closed that it was run on (13 of the 19 and Y1k6, Y3k6; A B C X1 X3 X4 X5 X6k5 X6k6 X8 Y1 Y2 Y3): the same optimum, worst
   relative cost difference 5.6e-16 (on Y2, Y3k6 other medoids one rank away on the line, cost difference 3.3e-13 / 2.8e-14: a tie); 27 of 28 MIP runs finished. Z1, Z2, Y4-Y6 have no oracle beyond the two builds agreeing.
2. Root [confirmed]: Kelley (ON) certifies at the root on 8 of the 33 probed inputs (dummy, A, B, Y1-Y5), the subgradient (OFF) on 11 (dummy, A, B, C, X1, X3, X5, X6k6, Y5, Z1, Z2); Kelley uses all 500 majors on 23 of 33 and left the loop early
   without certifying on Z1 (224) and Y3k10 (310) (:543-544, :573; cause unknown); OFF all 4000 on each of its 22 misses. Noise / random-walk sets (C, X1, X3, X5, X6k6, Z1, Z2): OFF certifies in 28-195 iterations with 0 nodes, ON needs 19-487,365
   nodes. X4, X8, X6k5 (both need a tree): OFF has fewer nodes (table). Line metric, k = 3 (Y1-Y5): ON certifies in 8-13 majors (Y2: 414); OFF runs 4000 iterations (Y5 stops at 2,585), gap 7.46e-6..3.72e-5, 41-4,869 nodes on Y1-Y4.
3. Wall, LR phase OFF/ON [inferred], r1 | r2: 0.003-0.30 | 0.003-0.29 on the 13 inputs where OFF is faster; Y2 0.93 | 1.35; slower on Y3 43 | 46, Y1 133 | 105, Y5 131 | 95, Y4 230 | 149 (r1 ms OFF / ON: 195 / 5, 4,539 / 34, 53,944 / 411, 37,213 /
   162). dtwc_cl end to end, OFF vs ON, r1 | r2: Y4 28.04 | 24.81 s vs 0.4531 | 0.486 s, Y5 69.98 | 58.01 vs 1.803 | 1.546, Y6 (N = 6400) 295.37 | 321.33 vs 6.898 | 7.043 (OFF/ON 62 | 51, 39 | 38, 43 | 46); repetitions differ by up to 1.6x (load).
4. Capped: Y1k6 / Y3k6 cap in ON (2M nodes, table) and close in OFF (158,757 / 385 nodes, 4.5 s / 0.2 s). The 13 capped in both: OFF's root LB is the higher on 12 (not Y3k10; the gap column mixes in each root's UB); the MIP found their optimum in
   0.66-124 s on 12 (X7: > 15 min, not finished); the LR incumbent the error discards was that optimum on 4 (ON) / 5 (OFF) of 12, at most 0.40% (ON) / 3.25% (OFF) above it. The error (Y1k6, ON): `Error: LR-core did not prove optimality (gap 0.003420
   after 2000000 branch-and-bound nodes). The incumbent is a heuristic, not a certificate; raise Problem::mip_settings.lr_max_nodes (currently 2000000), use Method::MIP for an exact MIP backend, or Method::Kmedoids for a heuristic answer.`
5. The LR phase is serial in both trees on AppleClang/GCC [flags confirmed; timing [inferred]]: `lagrangian_root.cpp` (the `mip-solvers` object library) is compiled without `-Xclang -fopenmp` / `-DDTWC_HAS_OPENMP` (`OpenMP::OpenMP_CXX` is PUBLIC on
   dtwc++ only, dtwc/CMakeLists.txt:115; for MSVC `dtwc_options` carries /openmp:experimental to mip-solvers, :107-112), so the `#pragma omp` loops of `evaluate_dual` are compiled out, and `warn_if_single_threaded` (env.cpp) cannot see it.
   OMP_NUM_THREADS=1 vs 18 (one run each): OFF stage times unchanged (Y1 root 4,263 vs 4,319 ms, X4 192.5 vs 193.2, Y3 194.7 vs 195.4), ON root_kelley within -7..+13% on X3, X4, Y1, Y3, Y4 and +43% on C (431 vs 302 ms). The unmodified file compiled
   in scratch with those flags gives identical fields on 10 of 10 probes (one run each); in interleaved serial / OpenMP A/B runs (two pairs each, load 3.1-6.9, `run_omp_ab.py`) the OFF root is 3.6-3.8x (B), 4.1-4.2x (X1), 3.9x (Y1), 5.4-5.6x (Y4)
   faster and 0.36x at N = 200 (Y3: 195 vs 548 ms); Kelley 1.3-1.4x (Y1, Y4), 0.66-2.2x (Y3); Y5 4.8x (one pair; its serial root took 61.5 s in r1 and 91.7 s in r2).
6. FastPAM's seed-42 UB = the final optimum on 16 of 18 closed inputs (not X4, X5); other seeds ran on X1, X3, X5 only. The subgradient root called in the ON tree matches the OFF tree's on certified / iterations / core on 33 of 33, bit for bit on 8.

## Reading
Without HiGHS the wheel's user loses no answer: the 19 inputs both builds close are identical (the compact MIP and a brute force agree where run), and 13 inputs fail identically in both (node-cap SolverError), where the MIP solved 12 (highspy, M1:
not run here). What changes is the root, by regime, on synthetic length-40 full-band L1 data only. On templates + noise and random walks the subgradient root is the faster one (OFF/ON LR phase 0.003-0.30 on 13 inputs; Kelley uses its 500 majors on 23
of 33 inputs overall), so HiGHS buys nothing there: its LR phase takes up to 9.4 s (Z1) where OFF's takes 0.9 s. On constant series (a line metric: Y1, Y3-Y6) Kelley certifies in 8-13 majors and the subgradient runs up to 4000 serial O(N^2)
iterations: 43-230x slower in the LR phase (r1; 46-149x r2) and 43-46x end to end at N = 6400 (295-321 s against 6.9-7.0 s); Y2 (overlapping clusters) is a tie. Independent of HiGHS, `lagrangian_root.cpp` is built without OpenMP on AppleClang/GCC:
built with it (scratch) the results are bit-identical and the OFF root 3.6-5.6x faster at N >= 800 (Y5 4.8x, one pair), 0.36x at N = 200 (OFF/ON LR ratio on Y1, Y4, Y5 would be near 36x, 31x, 47x, not 133x, 230x, 131x).
