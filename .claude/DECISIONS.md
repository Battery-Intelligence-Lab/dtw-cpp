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
   reason. `docs/api-contract-2.0.md` retired with W14a (e7f6153a); the tier pages describe the 2.0 surface. A user-visible change against v1.0.0 gets a
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
  build whose DP inner loop calls (since 2026-10-05, any loop of a probe kernel); registered with `add_test` because it
  runs a Python script, its PASS regex requires `inner_loops` ≥ 1.
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
- 2026-09-30 — Volkan: test deletions that the permission system refused are approved for W11c, W12a and W6f
  (each deleted test names the kept test that covers its subject). CPU floor for release archives and wheels:
  x86-64-v3 (AVX2+FMA; every ARC/HTC node has it). HPC builds the most specialised code for the target: detected
  on the node when the build runs there (native CPU flags, CUDA arch native), else named with `gpu_device=`.
- 2026-09-30 — E1 (W7a+W7b): `dist_by_ind` is a read of the packed matrix; a method that needs the matrix fills it
  serially at entry, the rest call the bound DTW function directly. No lock or atomic remains in `dtwc/`. PAM swap
  5.6–5.9× faster (inferred under load; counters identical). `resolve_dtw_fn` takes the Data so the WDTW weights are
  bound by value (part of W7e, done early).
- 2026-09-30 — Volkan: trivial tests go (duplicates, existence lists, source greps, wall-clock, vacuous or
  never-run cases), each deletion naming the oracle that stays; a unit adds a test only where a contract has
  none. Agents may delete under this approval; a refused deletion is listed and applied by the orchestrator.
- 2026-09-30 — W6f: `unit_test_invalid_distance_enums` and `_public_selectors` are deleted (the X3b line's "keep
  their names until W12b" is moot).
- 2026-10-01 — Volkan: FP contraction stays on every compiler (GCC's default `fast`, clang's `on`, MSVC
  `/fp:contract`; the open "drop /fp:contract" question is closed). Distances may differ between compilers and
  between the SIMD lanes and per-pair routes at the epsilon level; clustering results (conformance labels and
  medoids) may not. Cross-route tests compare within a bound scaled to the path length; bitwise only where both
  sides run the same code.
- 2026-10-01 — V3: the lanes fill at x86-64-v3 is ~1.0× unbanded and 1.17× banded against SSE2, not 2×: the
  8-lane double kernel waits on its min-then-add chain (inferred); lead: 16 double lanes.
- 2026-10-01 — V4: a symlinked checkpoint root is followed like any path (W5d removed the rejection; checkpoint is
  2.0-born and no page promised it); the POSIX-only row that expected the rejection goes.
- 2026-10-01 — F2: `DTWC_CL_PATH` names the dtwc_cl the Python wrapper and tests run; set but not a file (or
  empty) is an error.
- 2026-10-01 — W13b: a distance matrix is scanned for ±inf once where it enters (an accessor write, a CSV load, a
  checkpoint, a mapped .dtwm, Python/MATLAB `set_distance_matrix`); loops that only read a filled matrix read it
  unchecked. A fill of finite series can still give ±inf (±DBL_MAX series, soft-DTW at γ = DBL_MAX): no fill-end
  scan, and Lloyd's assignment keeps its check (overflow is not guarded, Volkan 09-30).
- 2026-10-01 — C3: the Shared wavefront compiles its preload branch for FP32 only (FP64 79 → 62 registers; FP64
  L 513–1024 at 0.84–0.89 of base); the gain is probably fewer FP64 instructions per cell, not occupancy (inferred).
- 2026-10-01 — C4 FALSIFIED: rounding each block's shared memory to the 128-byte unit makes the route rule match the
  driver's occupancy at every L 2049–10000 and moves FP32 L 2751–2757 to the global route at 0.945–0.963 of base,
  against the registered ≤ 0.95; not landed (the patch is kept in the C2 baseline folder).
- 2026-10-01 — Volkan (Parquet, file formats): the bindings ship the essentials, not file readers. The wheel and the
  MEX link no Arrow C++ (arrow.dll 21.4 MB + parquet.dll 6.4 MB on Windows, and a second libarrow beside the user's
  pyarrow); Python reads Parquet through the installed pyarrow (the `parquet` extra) into the Arrow C stream that the
  compiled-in nanoarrow reads zero-copy; MATLAB uses its own `parquetread`. The C++ readers serve the CLI and C++
  users (`DTWC_ENABLE_ARROW` opt-in). Next: L1 measures what the bindings link, L2 splits the library.
- 2026-10-01 — W8a: one reader entry `dtwc::read_data`; Parquet folder names unique and UTF-8; Arrow IPC through
  the C stream (nulls refused); `dtwc_cl -v` prints "Data loaded …" instead of the loader's own lines (CHANGELOG).
  Python reads list-per-row Parquet through pyarrow; a `.arrow` file in Python moves to pyarrow with L2.
- 2026-10-01 — W7c: one `core::validate(DistanceConfig, f32)` where a config is set or a facade entry runs; the
  metric rule is the facade's (Standard with any missing strategy and DDTW take the non-L1 metrics; L2 + AROW
  multivariate is refused: no such cost); `core/dtw.*`, `DTWOptions`, `ConstraintType`, `distance_semantics.hpp`,
  `variant_validation.hpp` and 11 per-pair validate calls deleted. Python/MATLAB `DTWClustering` and Python
  `distance.dtw` keep a copy of the old rule until W7g/W8c/W9b; three C4244 in wdtw_weights<float> go with W7e.
- 2026-10-01 — L1 (record only): HiGHS is 75 % of the Python extension (4.6 of 6.1 MB) and 78 % of the local MEX
  (6.2 of 7.9 MB); cli/ + CLI11 + fkYAML and the readers are ~2 %, pulled in by two edges (Python `device()` in
  api.cpp beside `cluster()` → `run()`; MEX `tier1_cluster` → `run()`). The CI MEX ships without HiGHS (0.88 MB);
  a MEX built with Gurobi ON needs gurobi130.dll (38.7 MB) to load.
- 2026-10-01 — Volkan (MIP solver in the bindings): Python solves the MIP with the user's highspy (an optional
  extra; the wheel drops HiGHS); MATLAB keeps HiGHS linked in the MEX (intlinprog would need the paid Optimization
  Toolbox), and the shipped MEX turns HiGHS on and keeps Gurobi off. C++ builds the model as arrays, Python hands
  them to highspy zero-copy, C++ decodes the solution; C++/CLI/MEX call linked HiGHS. Tests compare the optimal
  cost (highspy's HiGHS version may differ). M1, after L2.
- 2026-10-01 — Volkan (build files): each source folder of dtwc/ lists its own files in its own CMakeLists.txt
  (as mip/ does); dtwc/CMakeLists.txt keeps the targets, options and links (L2).
- 2026-10-01 — W7g: one `distance.dtw` per language over `dtwc::distance::dtw` (Python zero-copy with the GIL
  released; MATLAB `dtwc_mex('dtw', x, y, Name, Value…)`); the per-variant helpers and MATLAB's own metric list go
  (none in v1.0.0); keywords take the dtwc_cl names (Python `wdtw_g`, `adtw_penalty`, `sdtw_gamma`, `msm_c`,
  `twe_nu`, `twe_lambda`; MATLAB the same in CamelCase, as its DTWClustering spelled them, until W9e);
  `Problem.set_distance` in both bindings; both DTWClusterings validate in fit through C++; contract §10 item 8 is
  superseded (`dtw(x, y)` is Standard DTW in every language).
- 2026-10-01 — W7d: `core::orient` + `core::run_dtw` replace 26 orientation preambles. ab72231 kept (orchestrator):
  each kernel copies its Cost into a local, so no build reloads x and y per cell (shipped ThinLTO dtwc_cl: 63 such
  DP loops at base, 0 after); values unchanged (conformance, 31 CLI runs). An out-of-range MetricType now computes
  L1 where it threw per pair, so core::validate refuses an out-of-range enum (rule 3; W7ef).
- 2026-10-01 — W12b closed: tests/unit/adversarial/ is gone; the missing-data rules have one oracle
  (tests/support/missing_dtw_oracle.hpp); `dtw_routes_agree` lets an infinity agree only with itself (it passed any
  value against inf).
- 2026-10-01 — L2a: each dtwc/ folder lists its own sources in its own CMakeLists.txt; a source property set from a
  folder file names `TARGET_DIRECTORY dtwc++`; compile commands and executable link lines are identical in every tree
  (dtwc++.lib's member order follows the folders). 41 headers are listed nowhere, as before: L2b decides.
- 2026-10-01 — W8b: one `open_output` / `close_output` (it creates the parent directory) and one
  `detail::write_result_files` for Result::save and the CLI. A RAM-limited (streamed) Parquet result writes
  `series_<i>` labels and medoids and no matrix or silhouettes (Result::save asked for them: InvalidInput; it read
  an empty Data). Silhouettes: UndefinedScore is a warning; k = 1 writes no file, silently, in C++, the CLI, MATLAB
  and Python. `--name sub/x` writes into a subdirectory of the output directory (accepted).
- 2026-10-01 — W7ef: Soft-DTW and v1's dtwFull run on the linear kernel (Soft-DTW peak working set 3,431 → 14 MB at
  8 × 8,000 samples; values digit-identical: SoftCell sums diag, left, up as the full kernel did); Interpolate fills
  thread_local buffers; Problem's mutable matrix accessor is `writable_distance_matrix()`; TimeSeries/View,
  softmin_gamma, the DDTW pointer overloads and the full-matrix kernel go; core::validate refuses an out-of-range
  enum. SoftGammaScale stays: a subnormal gamma is valid and pinned.
- 2026-10-01 — W10: `set_device` + `set_gpu_precision` are a Problem's device surface (DistanceMatrixStrategy,
  CUDASettings and their mappers gone); `gpu_available()` / `gpu_info()` are the discovery functions in Python and
  MATLAB. A distance cache is keyed by its computed precision, not its device (orchestrator, applying Volkan 10-01
  on epsilon-level FP): a CPU FP64 cache serves a CUDA FP64 run, GPU 0 and GPU 1 agree, FP32 is refused by an FP64
  run. Metal's Auto is FP32; CUDA's Auto is refused for a persistent cache (it depends on the GPU). Python
  Result.device is "gpu"; `gpu_precision_names` stays (the GpuPrecision name table).
- 2026-10-01 — W14b: an explicit build option that cannot be honoured stops the configure (CUDA without a usable
  compiler or on macOS, Metal off Apple, Arrow found but unlinkable, Gurobi or HiGHS absent, MATLAB not found);
  Gurobi defaults OFF (CHANGELOG; the default MEX imports no gurobi130.dll); Arrow without Parquet is an IPC-only
  build that says so. Build trees that cached the old defaults need -DDTWC_ENABLE_METAL=OFF (and ARROW=OFF where
  Arrow is absent) once. A MATLAB without its executable needs DTWC_BUILD_TESTING=OFF.
- 2026-10-01 — Volkan (key names across languages): "okay similar enough names are ok. camelcase and snake case can
  change between languages." The same words in every language; Python and the Config / CLI keys use snake_case
  (flags kebab-case), MATLAB CamelCase name-value keys (`WdtwG`, `MissingStrategy`, `NClusters`, `MaxIter`); one
  convention within a language (W9e moves `dtwc.cluster`'s snake_case keys to CamelCase).
- 2026-10-01 — W9a: one `Method` (nine values) and one name table; `Problem::cluster()` runs all nine and returns
  its ClusteringResult; `run()` = apply the Config, load, `prob.cluster()`, write; `auto` = PAM on a GPU or at
  N <= 5000, else CLARA (one `resolve_method` for the CLI and Problem). `k` is required (rule 10: v1.0.0 with no
  `--Nc` exited 0 having clustered nothing); a run is named after its input; the C++ Tier-1 default method is
  `auto`; v1.0.0's option names are hidden warn-once spellings and its `--Nc i..j` range is refused.
- 2026-10-01 — W8c: Python and MATLAB `compute_distance_matrix` are one Problem fill on every device (byte-identical
  matrices, about 3x faster); an infeasible band, a band below -1 and an empty series are InvalidInput there too.
- 2026-10-02 — Volkan (question tool) — Windows wheels: "Measure first": time an MSVC wheel against a clang-cl
  wheel on the fill, quiet machine, before choosing (clang packs the lane loop, cl does not; a clang-cl wheel ships
  libomp.dll, which can abort with OMP Error #15 beside Intel's OpenMP). `dtwc_cl.exe` gets a VERSIONINFO resource
  ("Add it"). ARC builds keep Arrow ON and stop loudly if ARC's Arrow is unusable ("Keep ON, fail loudly"). Headers:
  "whichever the best practices for modern CMake" — read as each folder listing its headers in a `FILE_SET HEADERS`
  (CMake 3.23; the base directory is never the repo root), done in L2b. The cache rulings stand ("Keep as is": keyed
  by the data, the distance settings and the computed precision; CUDA Auto refused with a persistent cache; Metal
  Auto = FP32); `k` stays required. `9056fcb9` was his commit. Metal is noted for his next session on the Mac.
- 2026-10-02 — SW: comments state their reason, never a tracker id (259 → 17 lines; the 17 name GPUs, kernel phases,
  UNIMODULAR.md or test bands); `.claude/baselines` paths in comments stay (live records); tests keep `index_t`.
- 2026-10-02 — VI: `dtwc_cl.exe` carries VERSIONINFO (LegalCopyright = the LICENSE's copyright line; FILEVERSION from
  the project version, the text version in the strings); no prerelease flag; the wheel and the MEX get none.
- 2026-10-02 — GC: on CUDA, FastCLARA's assignment runs on the fill's kernels (one rectangle decode beside
  `decode_pair`, no second kernel family) and its host loop keeps the one tie rule and finite check; a test of an
  exact tie checks the cost and the groups, not which tied medoids win; streamed samples print under `-v` like
  in-memory ones; Metal still assigns on the CPU and says so.
- 2026-10-02 — Volkan (question tool), v1 Python names: "nobody depends on them don't worry documenting the changes.
  We just need to have a proper documentation of the latest version for now." No shims; v1.0.0's Python was never
  published (README and CHANGELOG silent, no package name). W14a documents 2.0 first.
- 2026-10-02 — Volkan, readers: "Pritorise reading the same file in the same way in all languages if possible. So you
  can bind some reader and probably assume pyarrow reads things same I guess. Also there needs to be a way for python
  and matlab users to pass their already-read data. Because they can read with numpy so they should be able to pass
  it." The bindings bind the one C++ text reader and writer (not copies); Parquet and Arrow go through the installed
  pyarrow (MATLAB: its own reader, checked against C++); every entry takes already-read data. Supersedes the 10-01
  plan to move text reading into Python; Arrow, CLI11 and fkYAML stay out of the bindings.
- 2026-10-02 — Volkan, clean-up: "Yes to both": a unit's worktree and build folder go once it has merged.
- 2026-10-05 — Volkan (question tool), Mac timing: "Merge W9b, then Mac (Recommended)": W9b merges here first, then
  the Mac pass (PLAN "Blocked on another machine") before L2b reshapes the CMake; a second short pass after L2b.
- 2026-10-05 — Volkan (question tool), W9b needing 2.5–4 more hours: "Go now; W9b finishes here (Recommended)": the
  Mac pass runs on design-2.0 without W9b, W9b merges on Windows, and the Mac runs pytest again once W9b lands.
- 2026-10-05 — Mac pass (orchestrator): a build tree cached before W14b needs `-DDTWC_ENABLE_GUROBI=OFF` once, like
  METAL and ARROW. The Mac's silhouette is one ulp off the Windows-recorded conformance reference with identical
  labels and medoids: accepted under D-19 and Volkan 10-01, the reference stays as recorded. Apple clang's
  `memset_pattern16` idiom is fixed in the kernels' code, not by a flag (the targeted no-builtin flag does nothing).
- 2026-10-06 — Volkan (chat, "sure also merge and commit"): the banded kernel's bounds-as-arithmetic simplification
  (`pb/banded-bounds-arith`, 25 lines fewer, bit-identical) merges although the Mac timed it 1.08–1.20× slower at
  bands 5–12 without early abandon (faster with it); the Windows box times it on x86 and the numbers decide if it stays.
- 2026-10-06 — W9b (merged 4265e3a6): Python's cluster(), DTWClustering, load() and set_data run on the C++ core
  (Config and apply() in the core); text and distance-matrix files are read as bytes (a Ctrl-Z is refused, a CR ends a
  line only before its LF); one conversion takes already-read data for every entry and refuses what is not series with
  `_NotSeries(TypeError, ValueError)`, since scikit-learn's estimator checks need ValueError (numpy's AxisError has the
  same two-parent form); Volkan may veto. `refuse_gpu_method` is inlined into apply() (one caller after GC).
- 2026-10-06 — Volkan (chat): "You can pull and merge the macos changes": origin's Mac pass merged on Windows.
- 2026-10-06 — M1 (merged c0580948, Mac): the wheel links no HiGHS (an extension that links HiGHS exports its weak
  symbols, and a user's own highspy MIP in the same process crashed on macOS, exit 139); `method="mip"` on the wheel
  solves with the installed highspy, the `mip` extra (`highspy>=1.8`, the first with `Highs.resetGlobalScheduler`),
  a missing highspy is SolverError naming the extra; the MEX links HiGHS statically (a MATLAB-on configure sets
  `BUILD_SHARED_LIBS OFF` for HiGHS, so that tree's dtwc_cl and tests link it statically too); `Problem.solver` is
  read-only in Python so a Gurobi request stays with C++; Gurobi's builder keeps its point-major model (little to
  delete, not compilable on the Mac). The CI Python job installs `[test,dev,io,mip]` and pandas so the pyarrow,
  pandas, scikit-learn and highspy cases run there instead of skipping (edited on the Mac, CI not run here).
- 2026-10-06 — W9e (merged on the Mac): MATLAB's refusals use `dtwc:invalidArgument` (contract §5's identifier for
  InvalidInput; the brief's `dtwc:invalidInput` was wrong); `DTWClustering` gains Method, MsmC, TweNu, TweLambda,
  MvMode, BatchSize and RandomState, `TotalCost` is `Inertia`, predict/transform/score read the fitted medoids, no
  `'precomputed'` metric; in-memory SkipRows/SkipCols are applied in MATLAB as Python applies them (C++'s copy sits in
  api.cpp, which the MEX no longer links); a logical reads as 1 for a numeric key (Python refuses a bool); the
  Parquet route is pinned against pyarrow, not the C++ Arrow reader (none on the Mac); `CheckpointOptions` keeps
  snake_case keys (outside the brief, MATLAB's last). Open for Volkan: the Windows R2024b matlab_suite run (the brief
  stays until it passes).
- 2026-10-06 — M1's adversarial review (merged bb35d337): the wheel's `method="lrcore"` runs the subgradient root
  because the Kelley cutting-plane root (`lagrangian_root.cpp:590`) needs linked HiGHS — the brief's "LR-core needs no
  HiGHS" was false; documented in CHANGELOG and the solvers guide, measured on the Mac (`baselines/2026-10-06-lrcore-
  root-without-highs-mac.md`); Volkan rules whether the wheel accepts it. A skipped highspy case fails the CI Python
  job from inside pytest (`DTWC_REQUIRE_HIGHSPY`); the CI MEX job asserts `test_cluster_mip` ran. Extras are printed
  quoted (`"dtwcpp[mip]"`: zsh). `.claude/CLAUDE.md`'s Python gate line should add `mip` to its extras (Volkan's file).
- 2026-10-06 — W9c (merged 834b7904, Mac): `hpc` crosses as one `job.toml` written by Python (to_config_text renders
  through CLI11, which the wheel no longer links; the grammar is `config_value`'s, round-tripped through the real
  `dtwc_cl --print-config`); C++ `apply()` on a scratch Problem checks every value before anything is written, the
  cluster's build judges only what its GPU must; the GPU table is a data file beside the wrapper (bash needs it
  without Python); A6000, L40S and H100 are asked for by `gpu_cc:` constraint (ARC lists no gres type for them —
  the node tags are inferred until the ARC leg runs); a pre-staged path is absolute; a `.env` that sets the removed
  `SLURM_GPU_GRES` is refused; a missing `.env`, bash or wrapper is `DeviceError`. `build --gpu-device <type>` keeps
  the CPU code portable (`DTWC_NATIVE_CPU=OFF`: one GPU type sits on nodes of different CPUs on ARC), which narrows
  the 09-30 ruling's "native CPU flags" to builds run by hand on the node — Volkan to confirm. The ARC leg is his.
- 2026-10-06 — lrcore without HiGHS measured (`3cb49bc8`, `baselines/2026-10-06-lrcore-root-without-highs-mac.md`):
  the wheel's subgradient root and the Kelley root give the same cost, medoids and labels on every input both close
  (19) and fail identically on 13 hard ones; on noisy DTW data the subgradient root certifies where Kelley does not
  (X1, X3, X5, Z1, Z2: 0 nodes against 19–487,365) and is faster; Kelley wins only on constant level-shifted series
  (a line metric: 43–230× on the LR phase, Y6 at N = 6400 295 s against 6.9 s). Orchestrator's recommendation: accept
  (the wheel keeps no HiGHS); Volkan rules. Unasked finding: `mip-solvers` links no OpenMP, so the LR phase is serial
  in every build (a scratch `-fopenmp` build: bit-identical, 3.6–5.6× faster root at N ≥ 800) — a candidate unit;
  `lagrangian_root.cpp:585` and `docs/content/math/lr-core.md:231,245` claim Kelley closes more roots: true on the
  line metric only.
- 2026-10-06 — L2b (merged on the Mac): `dtwc_core`, `dtwc_cli` and `dtwc_io` (STATIC) behind an INTERFACE `dtwc++`
  (consumers' link line unchanged; a non-INTERFACE command on `dtwc++` now fails at configure, CHANGELOG says so);
  `dtwc_cli` is STATIC, not OBJECT (an object library through an INTERFACE target delivers no objects); `dtwc_io`
  exists only in a build with Arrow; nanoarrow stays in the core (the brief's "no nanoarrow in the wheel" was wrong:
  Python's `data_from_arrow_c_array` needs it and it links no Arrow); FastCLARA's Parquet streaming is `dtwc_io`'s
  `algorithms::fast_clara_parquet`, and `fast_clara` refuses a stream request instead of ignoring `parquet_path`
  (review finding; a silent fallback gap that predated the split for Arrow-off builds); the core's `read_data` keeps
  the Arrow-off wording for Parquet in every build. Headers: 83 under dtwc/, each listed once in its folder's
  FILE_SET (Volkan 10-01). The Arrow-ON proof ran on the Mac through a pyarrow-25 shim; Windows (`build/arrow-
  pyarrow-23`) and GNU ld link order are still to run (commands in the L2b record). Seen: HiGHS caches
  `BUILD_SHARED_LIBS=ON`, so a re-configured tree builds Catch2 shared; in the Arrow-25 shim tree `test_io_readers`'s
  streamed `Result::save` case fails at base too (Windows's pyarrow 23 passes it; cause not found).
- 2026-10-06 — Mac kernel assembly audit (`baselines/2026-10-06-mac-kernel-assembly.md`, Opus; the headline numbers
  re-measured by the orchestrator): the CLI, the wheel and the MEX carry the same kernel loops, and `-march`/`-mcpu`
  change none; the lanes kernel sits on its 7-cycle `fcmgt→bif→fadd` chain (AArch64 has no one-instruction
  `b < a ? b : a`), the per-pair kernels on `fcmp→fcsel→fadd`. Fixed: nanobind compiled the binding file at `-Os` and
  the whole module ran that copy of the per-pair kernel (`NOMINSIZE`, 659f889f). For Volkan: AArch64 lanes with
  `fminnm` and 128-byte blocks (1.41–2.00×, fill 1.48–1.72×; exact on every input the checked paths admit) and
  two-column per-pair kernels (1.44–1.98× unbanded); the recommendation is both, AArch64 only for the first.
- 2026-10-06 — Volkan, asked whether to run the two kernel units: "okay run whatever is left sure". Run on the Mac:
  the reverse `check_docs` gate, arm-lanes, W13c, then W13e, W9f, lr-omp, pair-2col, the PF follow-ups, W14a, W14c.
- 2026-10-06 — arm-lanes (`e26d5680`): on `__aarch64__` the lanes' min is `std::fmin` (`LanesCell`, one `fminnm`)
  and W is 128 bytes (16 doubles, 32 floats); x86 (`fmin` = 3 instructions) and the per-pair kernels keep
  `std::min`. Exact: the fill refuses NaN and ±inf before any lane runs. MSVC ARM64 (no `__aarch64__`) keeps W 8.
- 2026-10-06 — W13c (`b36fad43`): `fast_pam(prob, k, max_iter = 100, seed = 42)` absorbs `fast_pam_seeded` in C++,
  Python and MATLAB; the unseeded-engine contract retires (chair.md:121, MAP §6). Only v1's one-argument
  `init::random` / `init::Kmeanspp` read `randGenerator`, one draw each as their seed (the pre-registered sequence
  change). `init::*_seeded` went rather than becoming overloads: an overload set breaks v1's
  `prob.init_fun = init::Kmeanspp`, so `init_with_seed` keeps its function-identity check.
- 2026-10-07 — PF (`30284994`): Python reads Parquet by the C++ reader's column rule, as MATLAB does (the first
  Float32/Float64 or list-of-float column; a scalar column is one series named by the file, list rows `series_<i>`);
  a set_solver that cannot be honoured prints nothing, the SolverError speaks.
- 2026-10-07 — Volkan, a Parquet file of scalar columns only (`save_dataset_parquet`'s layout), which read as one
  series: "read each row as series I think or have some option right? It is probably not rare to have multiple time
  series in the same file"; and (question tool) the rows' names: "First string column". So, in C++, Python and
  MATLAB alike: several numeric scalar columns → each row is a series (as CSV; `--column` stays the option that
  reads one column); one float column → one series per file; a list column → one series per row; a file's first
  string column names its rows (both layouts), else `series_<i>` (unit PQ).
- 2026-10-07 — PQ (`b2687171`): `--skip-cols` / `--skip-rows` drop a Parquet file's leading columns / rows as they
  drop a CSV's fields / lines (Parquet refused them before): a dropped column is neither read nor a name, a dropped
  row is not read; `--column` picks among the columns left; a column that is neither float nor string among a row's
  samples is an IOError naming it.
- 2026-10-07 — pair-2col (`0edc388b`): the unbanded per-pair kernel computes two columns per pass (1.42–1.98× f64 L1
  in every placement); the banded kernel's two-column form FALSIFIED its band and was reverted. On the M5 a per-pair
  loop's speed moves ±30 % with its code placement, and 64-byte loop alignment does not remove that (f64 squared runs
  1.3× slower aligned): a single build's banded timing is not evidence about banded code.
