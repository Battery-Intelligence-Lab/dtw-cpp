# DTWC++ — Decisions

Killed ideas, the standing rules, and one line per dated ruling. A change to a frozen surface (§2, rule 1)
needs a dated line in §3. Anything older or longer is in git history (`git log -p -- .claude/DECISIONS.md`).

## 1. Killed ideas — reopen only by overturning the recorded evidence

- **FastDTW** — slower than exact DTW (Wu & Keogh, TKDE 2022).
- **BanditPAM / BanditPAM++** — dominated by FasterPAM on a precomputed matrix.
- **Elkan / triangle-inequality pruning, faiss / hnswlib** — DTW is not a metric.
- **ONNX export, R / Julia bindings, a C ABI** — no consumer; native competitors saturate.
- **The 2023 custom x-space LP solver (OSLP, tableau, OSQP/ADMM)** — retired in `f7064b3`.
- **PDLP as the production p-median solver** — matrix-free Kelley wins on either device
  (`baselines/2026-07-08-pdlp-bench.md`); PDLP itself is deleted in phase B.
- **Lower bounds to cut DTW calls on an exact matrix** — an exact matrix needs every pair; the Pruned fill
  re-runs every abandoned pair and does more work than brute force (`baselines/2026-07-08-lb-cascade.md`).
  Exact-matrix savings come only from EAP cell pruning; pair skipping belongs to TADPole.
- **LB pruning inside PAM, MIP or LR-core** — they read the whole matrix.
- **≥ 10× PAM swap from the FastPAM1 decomposition on a cached matrix** — measured 2.95–8.06×, and FastPAM1
  stops unconverged at k = 200 where FasterPAM converges (`baselines/2026-07-08-faster-pam-bench.md`).
  FasterPAM is the only swap (2026-09-28).
- **Parallel `find_best_swap`** — measured ~10× slower than the sequential scan.
- **Eigen** — removed (X-27); the recorded slowdown did not reproduce under interleaved A/B
  (`baselines/2026-09-22-x27-drop-eigen-band.md`, `baselines/2026-09-23-x27-eigen-gap.md`).
- **Highway / xsimd; SIMD within one pair** — tried March–April 2026, not worth it (Volkan, 2026-09-23); the
  row recurrence does not vectorise (`baselines/2026-09-22-x04-codegen-report.md`). SIMD lanes across pairs in
  plain C++ passed their probe (2026-09-29, §3) and enter the CPU fill.
- **EAPruned as the per-pair exact kernel** — after K1 the call-free linear kernel is 1.5–3.6× faster on 7 of 7 UCR
  datasets; EAP wins only when it visits under about a third of the cells (`baselines/2026-09-29-p2-eap-vs-linear.md`).
- **A runtime check on a count** (series, clusters, labels) — counts are `index_t`; see §2 rule 2.
- **Extending the hand-written CMake URL-pin deny-list** — replaced by `scripts/check_pins.py`.
- **Not planned** (reopen with clustering evidence): ERP, LCSS, EDR, ShapeDTW, Itakura parallelogram.
- **Libraries not adopted** (2026-09-22; each duplicates in-tree code chosen on purpose): xxHash (SHA-256 is
  identity), PCG / xoshiro (`portable_random.hpp`), magic_enum (the name tables are the cross-language
  contract), toml++ / nlohmann-json (CLI11 + fkYAML read the config), {fmt}, kokkos-mdspan, rapidcheck
  (Catch2 `GENERATE`), Taskflow / TBB (OpenMP). Adopted: fast_float (FX-6), libpfm4 (benchmarks only), llfio
  header-only for the mapped matrix (Q2 closed 2026-09-28: not mio).

## 2. Standing rules

1. **Compatibility.** Frozen = what v1.0.0 shipped: C++ `Problem`, `DataLoader`, `Data`, `Method`, `Solver`,
   `init::*`, `scores::silhouette`, `dtwBanded` / `dtwFull` / `dtwFull_L`, `Range`, `Index`, the root headers
   `parallelisation.hpp`, `settings.hpp`, `timing.hpp`, `utility.hpp`, the `Problem_IO` filenames, and the CLI.
   Everything born during 2.0 is pre-tag: change it freely, no shims. No wire format is frozen before the tag.
   Breaking a frozen item needs a reason — R1 silently wrong, R2 unsound, R3 blocks the cross-language contract
   with no additive route, R4 unreachable and not in v1.0.0 — and one dated line in §3. "Cleaner" is not a
   reason. `docs/api-contract-2.0.md` retires in phase G. A user-visible change against v1.0.0 gets a
   CHANGELOG line.
2. **Integers.** `using index_t = std::int64_t;` counts of series, clusters and rows, labels, medoids and
   `dist_by_ind` indices are `index_t`; tuning values (`band`, `max_iter`, `n_init`, `n_samples`) are `int`;
   seeds `uint64_t`; products are `size_t` / `int64_t` by type. No count guard anywhere. The only checks left
   are the two inline `N·N > INT_MAX` throws where HiGHS and Gurobi take `int`. Labels: C++ `int64_t`, Python
   `np.int64`, MATLAB double, 1-based.
3. **Conventions (v1.0.0 / JOSS):** local cost L1, band in integer cells, no final square root, Σd medoid
   objectives with k-medoids++ seeding. Interoperability is added by tokens, never by flipping a default.
4. **No silent fallback** — device, threads, format, solver. A request that cannot be honoured is a typed error
   naming the fix; a guard that must fire without a dependency lives outside its `#ifdef`. A feature this build
   lacks takes its subsystem's error: `DeviceError`, `SolverError`, `IOError`.
5. **Optional dependencies stay optional** (OpenMP aside): HiGHS, Gurobi, CUDA, Metal, llfio, Arrow, YAML.
6. **Race-free by design** (Volkan, 2026-09-02; 2026-09-28: "we should be race-free by design (except the file
   operations), so no mutex for data reading from the distance matrix etc."). In a parallel region each element or
   slot has one writer and shared data is read-only; reductions and error capture go through per-thread slots
   combined serially after the region. No `std::mutex`, `omp critical` or atomic on a data path; only file
   operations and one-time process init (`std::call_once`) may lock.
7. **FP model:** `-fassociative-math` without `-ffinite-math-only`; NaN means missing or not computed; no
   `infinity()` sentinel.
8. **Gates assert their subject ran** (CTest scores a skip as a pass); CLI behaviour is proven on the real
   binary. A gate checks behaviour, never a sentence.
9. **Register the band before the run;** FALSIFIED is a deliverable; two attempts, then record. Numbers go to
   `.claude/baselines/` verbatim, tagged `[confirmed]` or `[inferred]`; a baseline stays while something cites it.
10. **"No-op" means digit-identical conformance output.** Wall-clock is advisory, counters decide. Stash and
    re-run before calling a failure pre-existing; disagreement → a third computation.
11. **One commit per proven step**, locally on `design-2.0` (Volkan, 2026-09-28).
12. **Python `Problem` is one thread per instance**, the GIL released consistently.
13. **Tier-1 routes have no side effects** (no working-directory writes); names are UTF-8 end to end.
14. **Libraries:** prefer a mature, portable, permissively licensed library over in-tree code, and name the code
    it deletes. No copyleft: MIT / BSD / Apache-2.0 / BSL-1.0 / zlib; never GPL / LGPL (Volkan, 2026-09-22).
15. **Config files go through CLI11** (`from_config` + fkYAML): flags beat the file, an unknown key is an error.
16. **Agents never push, tag, publish, SSH out, submit to ARC, rewrite history, or delete an untracked
    `build*/`.** Those are Volkan's actions.

## 3. Rulings (newest last)

- 2026-09-07 — Volkan "Go": every III.10 recommendation of the 2.0 spec adopted (Auto = brute force, move
  foundation headers to `dtwc/base/`, one `DTWC_` option prefix, shims kept, CLARANS kept, MPI deferred).
- 2026-09-21 — records reorganised into CHARTER / MAP / PLAN / DECISIONS (history at `9c08074`).
- 2026-09-22 — X-29 / X-31: CLI archives lacked an rpath and were built `-march=native`; fixed
  (`baselines/2026-09-22-x29-release-archive-not-portable.md`).
- 2026-09-22 — X-04: no DTW kernel loop vectorises; the codegen report stays a manual tool
  (`baselines/2026-09-22-x04-codegen-report.md`).
- 2026-09-22 — X-15 / S-03: FP relaxations ride on `dtwc_options`, not directory scope; `DTWC_FP_MODEL=strict`
  gives digit-identical conformance (`baselines/2026-09-22-x15-s03-fp-flag-scope.md`).
- 2026-09-22 — X-24: benchmark counters through libpfm4; a counter-less run is renamed, never passed off
  (`baselines/2026-09-22-x24-benchmark-pmu-counters.md`).
- 2026-09-22 — dependency review: nothing missing (§1 list); third-party notices ship with every artefact (X-23).
- 2026-09-23 — YAGNI pass (CHARTER entry): keep an item only for interface, speed, a silent wrong answer, or a
  helper that deletes two copies.
- 2026-09-23 — Volkan: no Highway; the 1.x output filenames stay; the Eigen gap is investigated, not accepted.
- 2026-09-23 — test floors removed (GT-1): a test passes on ≥ 1 assertion in ≥ 1 case, no failure, and no skip
  unless `MAY_SKIP`.
- 2026-09-23 — D-B: fast_float vendored for the text readers; macOS wheel target 13.3; an interior blank line
  and an empty series are errors.
- 2026-09-24 — IF-1: `Problem::set_device(Device, index)`; `DeviceError` for what the device cannot run,
  `InvalidInput` for an infeasible band (D-12: an error, never widened silently).
- 2026-09-24 — IF-2: `Config` keyed by the CLI long names; `cli::bind` is the one key table; `run(Config)` the
  one pipeline; `--print-config`; Python and MATLAB keywords are rendered to config text and parsed by it.
- 2026-09-24 — IF-3: `cuda` is an alias of `gpu` in every language.
- Standing D-rows: D-10 C++ has no `hpc` submission (CLI / Python only); D-11 tag after the release review;
  D-13 Python ≥ 3.10; D-14 one estimator, `DTWClustering`; D-15 MATLAB's thread setting governs the MEX;
  D-19 cross-platform agreement is a tolerance, not a pinned libm.
- 2026-09-28 — records floor (phase A, W1): `reports/` (with the PRIVATE Kasper folder), `specs/`, the plan
  archive, the TODO list, the design document and old handoffs deleted from the tree; removing the Kasper folder from
  pushed history is Volkan's.

**2026-09-28 — design review approved (Volkan).** `plans/2026-09-27-design-review.md` replaces the plan's §3–§6
with phases A–G; implementer agents may commit each proven step locally on `design-2.0` (never push, tag or
rewrite). GPU assignment for CLARA is in 2.0, the last performance item after phase C. OneBatchPAM vs CLARA:
"Are they equivalent methods? If they have trade-offs they both can stay, but first measure if they are equal."
llfio → mio: open until a measured answer to "how is mio performance?". This overturns D-16 (Pruned kept as a
diagnostic) and D-22 (no widening: counts become `index_t = std::int64_t`). The review's §2 table is the
decision record for integers, compatibility, labels per language, `hpc`, `Method`, device vocabulary, `auto`,
PAM swap, exact solvers, the pruned fill, persistence, backends, diagnostics, Python I/O, records, gates and the
CHANGELOG rule.
- **2026-09-28 — OneBatchPAM and CLARA both stay.** Measured (exact costs and DTW-evaluation counts, deterministic;
  wall-clock advisory because COMSOL and builds shared the machine): a trade-off — OneBatchPAM 1.4–4.7 % lower cost in 8
  of 10 cells, CLARA fewer DTW evaluations for k ≤ 20, OneBatchPAM fewer at k = 100. Volkan's rule: trade-offs → both.
  Evidence: `plans/2026-09-27-audit/phaseA_measurements.md`. OneBatchPAM's swap loop and final assignment are serial.
- **2026-09-28 — mmap library: one that takes updates; not mio.** Volkan: "If you think mio is fine as it is, if not I
  would prefer a library that takes updates." mio is not fine as it is: no commit since 2023-03, `CreateFileMapping`
  failure checked against the wrong sentinel (issue #102), a destructor that syncs and drops errors, no exclusive
  create. Access speed is not the reason: mio equals llfio when the file is created the same way. llfio stays (D-18),
  pulled in header-only without the quickcpplib superbuild if the spike proves it builds in wheels; otherwise
  Boost.Interprocess. New files are created non-sparse (`win_disable_sparse_file_creation`): a sparse mapped file cost
  1.8× on random reads on Windows.
- **2026-09-28 — llfio header-only: the spike held** (`78af336` on `worktree-agent-aff6e262d581bae10`, base
  `80c4dcb`). Pinned archives, no superbuild (`Dependencies.cmake` −196/+69); warm configure 198 s → 44 s; ctest green;
  a wheel with llfio ON passed pytest 1174/19/0 with a bit-identical mapped round trip; an R2025b MSVC MEX with llfio ON
  passed its storage test. The superbuild had compiled quickcpplib from master and outcome from develop, unpinned; both
  are pinned now. wg14_signals is Apache-2.0 (notice added). Boost.Interprocess is not needed; D-18 stands.
- 2026-09-28 — W2a FALSIFIED: LB_Webb prunes exactly as LB_Keogh in TADPole (78.0 %, N = 200;
  `baselines/2026-09-28-webb-vs-keogh-tadpole.md`), so it went. `docs/api-contract-2.0.md` was edited to drop
  `lb_strategy` and `Pruned`.
- 2026-09-28 — W5b: the auto-spill to `.dtws` ran only after the heap load had filled RAM, so it never served data
  larger than RAM (it copied it and leaked the temp file). Data beyond RAM = list-per-row Parquet streamed by CLARA /
  OneBatchPAM under `--ram-limit`, plus the mapped distance matrix.
- 2026-09-28 — W5e: an internal precondition whose public entry already raises a typed error is an `assert`, and a
  test that expected it to throw goes (`ParquetChunkReader::read_rows` on a scalar column; `fast_clara` rejects it).
- 2026-09-28 — Sophos 'Generic ML PUA' on X2's Release `dtwc_cl.exe`: no exclusion. Volkan: "It is alright probably
  it will be resolved when we add more and more features". X2 re-merges once the binary has changed; if still
  quarantined, that merge's CLI tests run on a Debug build of the same tree; every other gate stays Release; CI runs
  the Release CLI tests.
- 2026-09-29 — PF-5 probe PASS: W equal-length pairs in SIMD lanes, plain C++, run 3.7–4.2× (W = 4) and 5.2–7.9×
  (W = 8) the fixed scalar kernel for f64, full and banded, every lane bitwise equal; a 24-thread fill 3.6× / 4.8×
  (`baselines/2026-09-29-pf5-simd-lanes-probe.md`). The kill criterion (1.5×) is not met, so lanes enter the CPU fill
  now (unit P1, after K1 and Y2), not after phase G: Volkan asked on 2026-09-29 that the code "generates decent
  assembly like SIMD where needed".
- 2026-09-29 — K1 merged (`4441969`): no DP cell makes a library call; the linear kernel 7.0× and the banded 5.6× faster
  on Windows (pinned P-core), digit-identical. The fill band (≥ 2×) FALSIFIED at 1.36×: the unbanded fill runs the
  EAPruned kernel, which made no call; P1 measures lanes against it. `test_codegen_no_calls` (clang builds) fails a
  build whose DP inner loop calls; registered with `add_test` because it runs a Python script, its PASS regex requires
  `inner_loops` ≥ 1.
- 2026-09-29 — Y2 merged (`959dc5b`): heap and mapped matrices are one `core::DistanceMatrix`; get/set index a raw
  `double *`. The mapped file's fingerprint is checked once, when it is mapped; a matrix assigned through the mutable
  accessor is the caller's (review point 4: a per-lookup compare to catch deliberate C++ misuse is the defensive design
  the charter rules out). A checkpoint is `<dir>/<name>.dtwm` (`distances.dtwm` unnamed), flushed to the device before
  the rename; writing values into a mapped Problem is `InvalidInput`.
- 2026-09-29 — Y3 merged (`b260415`): the CMake named-critical scanner goes with `run_openmp`'s critical (no header has
  one); `dtwc_main` goes, overriding the ledger's keep-as-tutorial (B-07/O-19) per the 09-27 instruction; `--ram-limit`
  rounds a fractional byte up; `gpu_precision_names` keeps its spellings (docs and SLURM scripts use f32 / f64).
- 2026-09-29 — P1 merged (`d61c499`): the CPU fill runs W = 64 / sizeof(T) equal-length pairs per call in SIMD lanes,
  bitwise equal to the per-pair kernels. lld-link's LTO backend runs no SLP vectoriser, so `core/dtw_lanes.cpp` compiles
  `-fno-lto` when clang targets the MSVC ABI. cl packs nothing (/Qvec-report 1200), so MSVC builds — the Windows wheel,
  the MEX — get unpacked lanes. A value-returning min let cl pack the float lanes but, under `/fp:contract`, fused
  squared L2 in the lanes only (a bitwise mismatch), so it was reverted; whether MSVC keeps `/fp:contract` is Volkan's.
- 2026-09-29 — W4a: no CUDA kernel variant is within 5 % of its replacement; W4d deletes the forcing machinery only.
- 2026-09-30 — Volkan: no Fable; simpler delegated tasks run on Sonnet 5.5 at xhigh effort (CHARTER). Workflow agents
  take `model` and `effort`; the Agent tool does not, so Sonnet units run as one-agent workflows.
- 2026-09-30 — X2 merged (`4de2ce9`): its Release `dtwc_cl.exe` ran with no Sophos event; the 09-28 quarantine did not
  recur once the binary changed.
- 2026-09-30 — P3 merged (`5e15459`): no CHANGELOG line. v1.0.0 already ran `dtwFull_L` for unbanded DTW; EAP was
  2.0-born, so P2's 1.48–3.57× is against a kernel no release shipped.
- 2026-09-30 — Y4 merged (`93eefe1`): the F18 MATLAB oracle's seed-42 L1 `fast_pam` expectation follows FasterPAM (a
  different swap-local optimum; pre-registered "`pam` results"). F43 did not reproduce: a scratch mapped route in an
  llfio-ON MSVC MEX under R2024b ran clean, and no MEX route maps a matrix, so no `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR`.
- 2026-09-30 — G1 merged (`1bb9413`): only `mip` needs HiGHS; LR-core is exact without it (subgradient root, then
  branch and bound), so a HiGHS-OFF build asserts `SolverError` for `mip` alone.
- 2026-09-30 — W6e's MATLAB half: `cmd_cluster_legacy` and snake_case MATLAB keys move to W9e, where `run(Config)`
  replaces them (chair.md line 20 already puts `cmd_cluster_legacy` in W9); renaming the keys twice is waste.
- 2026-09-30 — W6e, Python half: `Problem.cluster_size` is a method again, silent (v1 bound it as a method; the 2.0
  warning property made `prob.cluster_size()` a TypeError). The bindings of `set_view_data`, `OneBatchPAMStats` and
  `one_batch_pam_with_stats` go as chair.md §4 W6 approved; the C++ stays (FastCLARA and the CLI use it).
- 2026-09-30 — W4d: one verbose line per GPU fill, printed by the backend (it knows the kernel and precision used);
  `Problem`'s duplicate print and its GPU pre-checks go, so the refusal wording lives in the backends (pinned by
  `test_run_resolution` and `test_cli_device_matrix`). The device limits are read once under `std::call_once`, whose
  callable does not throw (GCC PR 66146).
- 2026-09-30 — W6m: MATLAB integers are read exactly (`get_exact_int`); an index is >= 1, a label is any integer (labels
  are values: ARI/NMI take 0 and negatives, as C++ and Python). `Problem::dist_by_ind` stays unchecked (hot path); the
  bindings check indices at the language boundary (MATLAB and Python returned 0 or read out of bounds).
- 2026-09-30 — B1 merged (`4968d44`): Python's `clusters_ind` / `centroids_ind` are read-only (v1's Python never bound
  them) and `set_result` is the bound write route; a `Result` with no medoids raises `InvalidInput` from `score()` and
  `save()` instead of scoring against the zero-filled medoids `set_n_clusters` leaves.
- 2026-09-30 — B2 (scope): "cluster first" is one C++ check per call — clustered means both outputs have the sizes N
  and k — in the functions that read them wholesale; `set_n_clusters` stops pre-sizing them; the setters refuse
  k < 1 and band < −1. The per-element accessors stay unchecked (hot path); the bindings check indices.
- 2026-09-30 — X3b: `-Werror=switch` / C4062 cover C++ sources only; `.cu` and `.mm` switches are not gated (OBJCXX
  cannot be verified here). `unit_test_invalid_distance_enums` / `_public_selectors` keep their names until W12b.
- 2026-09-30 — Volkan: the CUDA floor is the A30's generation — compute capability 8.0 (Ampere, 2021). Default
  architectures 80;86;89;90 with PTX of the newest for later GPUs; a device below 8.0 is a typed `DeviceError` at
  selection; code for older architectures goes (C1).
- 2026-09-30 — C1 merged (`ceb7f91`): the CUDA toolkit floor stays 12.0 (ARC loads CUDA 12.4), so the shared-memory
  carveout (−17–20 % at FP32 L 2049–2644, its band passed) is dropped — its launch attribute needs CUDA 12.5. The
  global-memory wavefront removes every length limit on the GPU.
