# DTWC++ — Decisions

Killed ideas, the standing rules, and the rulings still in force, one dated line each. A change to a frozen surface
(§2, rule 1) needs a dated line in §3. Rulings about finished work were pruned on 2026-10-09 (Volkan: "remove the
obsolete ones, or already completed ones"); they are in git: `git show 034d8994:.claude/DECISIONS.md`.

## 1. Killed ideas — reopen only by overturning the recorded evidence

- **FastDTW** — slower than exact DTW (Wu & Keogh, TKDE 2022).
- **BanditPAM / BanditPAM++** — dominated by FasterPAM on a precomputed matrix.
- **Elkan / triangle-inequality pruning, faiss / hnswlib** — DTW is not a metric.
- **ONNX export, R / Julia bindings, a C ABI** — no consumer; native competitors saturate.
- **The 2023 custom x-space LP solver (OSLP, tableau, OSQP/ADMM)** — retired in `f7064b3`.
- **PDLP as the production p-median solver** — matrix-free Kelley wins on either device
  (`baselines/2026-07-08-pdlp-bench.md`); PDLP was deleted in phase B.
- **Lower bounds to cut DTW calls on an exact matrix** — an exact matrix needs every pair; the Pruned fill re-ran
  every abandoned pair and did more work than brute force (`baselines/2026-07-08-lb-cascade.md`). Pair skipping
  belongs to TADPole.
- **LB pruning inside PAM, MIP or LR-core** — they read the whole matrix.
- **LB_Webb in TADPole** — prunes exactly as LB_Keogh (78.0 % at N = 200, 2026-09-28); deleted.
- **≥ 10× PAM swap from the FastPAM1 decomposition on a cached matrix** — measured 2.95–8.06×, and FastPAM1 stops
  unconverged at k = 200 where FasterPAM converges (`baselines/2026-07-08-faster-pam-bench.md`). FasterPAM is the
  only swap (2026-09-28).
- **Parallel `find_best_swap`** — measured ~10× slower than the sequential scan. Reopened 2026-10-09 for unit SW
  (Volkan): a blocked evaluation over contiguous runs whose accepted swaps equal the sequential sweep's.
- **Eigen** — removed (X-27); the recorded slowdown did not reproduce under interleaved A/B
  (`baselines/2026-09-22-x27-drop-eigen-band.md`, `baselines/2026-09-23-x27-eigen-gap.md`).
- **Highway / xsimd; SIMD within one pair** — tried March–April 2026, not worth it (Volkan, 2026-09-23); the row
  recurrence does not vectorise (`baselines/2026-09-22-x04-codegen-report.md`). SIMD lanes across pairs, in plain
  C++, are the CPU fill's.
- **EAPruned as the per-pair exact kernel** — the call-free linear kernel is 1.5–3.6× faster on 7 of 7 UCR datasets;
  EAP won only below about a third of the cells (`baselines/2026-09-29-p2-eap-vs-linear.md`); deleted.
- **mio for the mapped matrix** — unmaintained since 2023-03, a wrong failure sentinel, a destructor that drops errors
  (2026-09-28). llfio stays, header-only; mapped files are created non-sparse (a sparse one cost 1.8× on Windows).
- **The `.dtws` auto-spill** — it ran only after the heap load had filled RAM, so it never served data larger than
  RAM (2026-09-28). Data beyond RAM is CLARA's Parquet stream and the mapped matrix.
- **A runtime check on a count** (series, clusters, labels) — counts are `index_t`; see §2 rule 2.
- **Extending the hand-written CMake URL-pin deny-list** — replaced by `scripts/check_pins.py`.
- **Not planned** (reopen with clustering evidence): ERP, LCSS, EDR, ShapeDTW, Itakura parallelogram.
- **Libraries not adopted** (2026-09-22; each duplicates in-tree code chosen on purpose): xxHash (SHA-256 is
  identity), PCG / xoshiro (`portable_random.hpp`), magic_enum (the name tables are the cross-language
  contract), toml++ / nlohmann-json (CLI11 + fkYAML read the config), {fmt}, kokkos-mdspan, rapidcheck
  (Catch2 `GENERATE`), Taskflow / TBB (OpenMP). Adopted: fast_float, libpfm4 (benchmarks only), llfio
  header-only for the mapped matrix.

## 2. Standing rules

1. **Compatibility.** Frozen = what v1.0.0 shipped: C++ `Problem`, `DataLoader`, `Data`, `Method`, `Solver`,
   `init::*`, `scores::silhouette`, `dtwBanded` / `dtwFull` / `dtwFull_L`, `Range`, `Index`, the root headers
   `parallelisation.hpp`, `settings.hpp`, `timing.hpp`, `utility.hpp`, the `Problem_IO` filenames, and the CLI.
   Everything born during 2.0 is pre-tag: change it freely, no shims. No wire format is frozen before the tag.
   Breaking a frozen item needs a reason — R1 silently wrong, R2 unsound, R3 blocks the cross-language contract
   with no additive route, R4 unreachable and not in v1.0.0 — and one dated line in §3. "Cleaner" is not a
   reason. The tier pages describe the 2.0 surface. A user-visible change against v1.0.0 gets a CHANGELOG line.
2. **Integers.** `using index_t = std::int64_t;` counts of series, clusters and rows, labels, medoids and
   `dist_by_ind` indices are `index_t`; tuning values (`band`, `max_iter`, `n_init`, `n_samples`) are `int`;
   seeds `uint64_t`; products are `size_t` / `int64_t` by type. No count guard anywhere. The only checks left
   are the two inline `N·N > INT_MAX` throws where HiGHS and Gurobi take `int`. Labels: C++ `int64_t`, Python
   `np.int64`, MATLAB double, 1-based.
3. **Conventions (v1.0.0 / JOSS):** local cost L1, band in integer cells, no final square root, Σd medoid
   objectives with k-medoids++ seeding. Interoperability is added by tokens, never by flipping a default. Double
   is the default for series and the matrix; float32 is one word away (2026-10-08, §3).
4. **No silent fallback** — device, threads, format, solver. A request that cannot be honoured is a typed error
   naming the fix; a guard that must fire without a dependency lives outside its `#ifdef`. A feature this build
   lacks takes its subsystem's error: `DeviceError`, `SolverError`, `IOError`. An explicit build option that cannot
   be honoured stops the configure (W14b).
5. **Optional dependencies stay optional** (OpenMP aside): HiGHS, Gurobi, CUDA, Metal, llfio, Arrow, YAML.
6. **Race-free by design** (Volkan, 2026-09-02; 2026-09-28: "we should be race-free by design (except the file
   operations), so no mutex for data reading from the distance matrix etc."). In a parallel region each element or
   slot has one writer and shared data is read-only; reductions and error capture go through per-thread slots
   combined serially after the region. No `std::mutex`, `omp critical` or atomic on a data path; only file
   operations and one-time process init (`std::call_once`) may lock.
7. **FP model:** `-fassociative-math` without `-ffinite-math-only`; NaN means missing or not computed; no
   `infinity()` sentinel. FP contraction stays on every compiler (GCC's `fast`, clang's `on`, MSVC `/fp:contract`):
   distances may differ between compilers and between routes at the epsilon level, clustering results (labels,
   medoids) may not; cross-route tests compare within a bound scaled to the path length, bitwise only where both
   sides run the same code (Volkan, 2026-10-01).
8. **Gates assert their subject ran** (CTest scores a skip as a pass); CLI behaviour is proven on the real
   binary. A gate checks behaviour, never a sentence. A test passes on ≥ 1 assertion in ≥ 1 case with no failure
   and no skip unless `MAY_SKIP`.
9. **Register the band before the run;** FALSIFIED is a deliverable; two attempts, then record. Numbers go to
   `.claude/baselines/` (on disk, out of git) verbatim, tagged `[confirmed]` or `[inferred]`.
10. **"No-op" means digit-identical conformance output.** Wall-clock is advisory, counters decide; timing only on a
    quiet machine, and cache misses are a speed metric beside it (Volkan, 2026-10-08). Stash and re-run before
    calling a failure pre-existing; disagreement → a third computation.
11. **One commit per proven step**, locally on `design-2.0` (Volkan, 2026-09-28).
12. **Python `Problem` is one thread per instance**, the GIL released consistently.
13. **Tier-1 routes have no side effects** (no working-directory writes); names are UTF-8 end to end.
14. **Libraries:** prefer a mature, portable, permissively licensed library over in-tree code, and name the code
    it deletes. No copyleft: MIT / BSD / Apache-2.0 / BSL-1.0 / zlib; never GPL / LGPL (Volkan, 2026-09-22).
    Third-party notices ship with every artefact.
15. **Config files go through CLI11** (`from_config` + fkYAML): flags beat the file, an unknown key is an error.
16. **Agents never push, tag, publish, SSH out, submit to ARC, rewrite history, or delete an untracked
    `build*/`.** Those are Volkan's actions. A unit's worktree and build folder go once it has merged (Volkan,
    2026-10-02). Never Fable; simpler delegated tasks run on Sonnet at xhigh effort (Volkan, 2026-09-30).
17. **Names across languages** (Volkan, 2026-10-01: "okay similar enough names are ok. camelcase and snake case can
    change between languages."): the same words everywhere; Python and the Config / CLI keys in snake_case (flags
    kebab-case), MATLAB CamelCase name-value keys; one convention within a language.
18. **Hot paths stay unchecked.** Per-element accessors (`dist_by_ind`, `series`, the result vectors) check nothing;
    a binding checks every index at its language boundary; a method that needs the matrix fills it serially at
    entry; a matrix is scanned for ±inf once where it enters, never per lookup.
19. **Tests:** trivial tests go (duplicates, existence lists, source greps, wall-clock, vacuous or never-run cases),
    each deletion naming the oracle that stays; a unit adds a test only where a contract has none (Volkan,
    2026-09-30).
20. **Comments state their reason** — never a tracker id (2026-10-02) or a run record's path (2026-10-09).

## 3. Rulings in force (newest last)

- 2026-09-07 — Volkan "Go" on the 2.0 spec: Auto = brute force; foundation headers in `dtwc/base/`; one `DTWC_`
  option prefix.
- 2026-09-22 — X-15 / S-03: FP relaxations ride on `dtwc_options`, not directory scope; `DTWC_FP_MODEL=strict`
  gives digit-identical conformance. X-24: benchmark counters through libpfm4; a counter-less run is renamed,
  never passed off.
- 2026-09-23 — YAGNI: keep an item only for interface, speed, a silent wrong answer, or a helper that deletes two
  copies. Volkan: the 1.x output filenames stay.
- 2026-09-23 — D-B: fast_float vendored for the text readers; macOS wheel target 13.3; an interior blank line and an
  empty series are errors.
- 2026-09-24 — IF-1: `Problem::set_device(Device, index)`; `DeviceError` for what the device cannot run,
  `InvalidInput` for an infeasible band (D-12: an error, never widened silently). IF-2: `Config` keyed by the CLI
  long names, `cli::bind` the one key table, `run(Config)` the one pipeline, `--print-config`. IF-3: `cuda` is an
  alias of `gpu` in every language.
- Standing D-rows: D-10 C++ has no `hpc` submission (CLI / Python only); D-11 tag after the release review;
  D-13 Python ≥ 3.10; D-14 one estimator, `DTWClustering`; D-15 MATLAB's thread setting governs the MEX;
  D-19 cross-platform agreement is a tolerance, not a pinned libm.
- 2026-09-28 — The records floor deleted `reports/` with its PRIVATE Kasper folder from the tree; removing that
  folder from pushed history is Volkan's (open).
- 2026-09-28 — OneBatchPAM and CLARA both stay: a measured trade-off (OneBatchPAM 1.4–4.7 % lower cost in 8 of 10
  cells; CLARA fewer DTW evaluations for k ≤ 20, OneBatchPAM fewer at k = 100); Volkan: trade-offs → both.
- 2026-09-28 — W5e: an internal precondition whose public entry already raises a typed error is an `assert`.
- 2026-09-30 — Volkan: the CUDA floor is compute capability 8.0 (the A30's generation): architectures 80;86;89;90
  with PTX of the newest; below 8.0 a typed `DeviceError` at selection. The CUDA toolkit floor stays 12.0 (ARC
  loads 12.4).
- 2026-09-30 — Volkan: the CPU floor for release archives and wheels is x86-64-v3 (AVX2 + FMA). HPC builds the most
  specialised code for its target: native CPU flags when built by hand on the node; `build --gpu-device` keeps the
  CPU code portable (`DTWC_NATIVE_CPU=OFF`; Volkan 10-08: "Confirm portable CPU").
- 2026-09-30 — X3b: `-Werror=switch` / C4062 cover C++ sources only; `.cu` and `.mm` switches are not gated.
- 2026-10-01 — Volkan (MIP solver in the bindings): Python solves the MIP with the user's highspy (the `mip` extra;
  the wheel links no HiGHS); MATLAB keeps HiGHS linked in the MEX, Gurobi off; C++ builds the model as arrays.
- 2026-10-01 — Volkan (build files): each source folder of `dtwc/` lists its own sources, and its headers in a
  `FILE_SET HEADERS` (10-02), in its own CMakeLists.txt; `dtwc/CMakeLists.txt` keeps the targets, options and links.
- 2026-10-01 — W10: a distance cache is keyed by the data, the distance settings and the computed precision, not the
  device; Metal's Auto is FP32; CUDA's Auto is refused for a persistent cache (Volkan 10-02: "Keep as is").
- 2026-10-01 — W9a: one `Method` (nine values); `auto` = PAM on a GPU or at N ≤ 5000, else CLARA; `k` is required
  (v1.0.0 with no `--Nc` exited 0 having clustered nothing); v1.0.0's option names are warn-once spellings and its
  `--Nc i..j` is refused.
- 2026-10-02 — Volkan: ARC builds keep Arrow ON and stop loudly if ARC's Arrow is unusable ("Keep ON, fail
  loudly"); `dtwc_cl.exe` carries a VERSIONINFO resource ("Add it").
- 2026-10-02 — Volkan, v1 Python names: "nobody depends on them don't worry documenting the changes. We just need
  to have a proper documentation of the latest version for now." No shims; v1.0.0's Python was never published.
- 2026-10-02 — Volkan, readers: "Pritorise reading the same file in the same way in all languages if possible. So
  you can bind some reader and probably assume pyarrow reads things same I guess. Also there needs to be a way for
  python and matlab users to pass their already-read data. Because they can read with numpy so they should be able
  to pass it." The bindings bind the one C++ text reader and writer; Parquet and Arrow go through the installed
  pyarrow (MATLAB: `parquetread`, checked against C++); the wheel and the MEX link no Arrow C++ (10-01); every entry
  takes already-read data.
- 2026-10-05 — The Mac's silhouette is one ulp off the Windows conformance reference with identical labels and
  medoids: accepted under D-19 and Volkan 10-01; the reference stays as recorded.
- 2026-10-07 — Volkan, a Parquet file of scalar columns only: "read each row as series I think or have some option
  right? It is probably not rare to have multiple time series in the same file"; the rows' names: "First string
  column". In C++, Python and MATLAB alike: several numeric scalar columns → each row is a series; one float column
  → one series per file; a list column → one series per row; a file's first string column names its rows.
- 2026-10-07 — Volkan (lrcore root): "When highspy is not there use subgradient but when highspy available then we
  should be able to use, but maybe default to subgradient for all like CLI etc. so default behaviour is the same."
  The subgradient root is the default in every language and build; Kelley's root is an option, run by linked HiGHS
  or the user's highspy, a SolverError without either.
- 2026-10-08 — Matrix reads (Volkan, three rounds of the question tool): NaN marks a pair never computed (Soft-DTW
  of a series with itself is about −1.76·γ·n, unbounded, so no finite mark is safe); `not_computed` = nan
  (default) | negative; under every distance but Soft-DTW a negative entry is refused by default, naming the option.
- 2026-10-08 — Volkan: "Okay move the wheel to cl if you cannot make it happen, unless there is a problem with the
  license. Because speed is the most important right now." The Windows wheels and archive are built by a pinned
  clang-cl (libomp.dll bundled; no licence fee); the archive carries cl's build beside it as `dtwc_cl_msvc.exe`
  ("yes archive too, you can also keep msvc compiled versions with "_msvc" suffix"), and the docs recommend
  `dtwc_cl.exe` with the measured speed-up.
- 2026-10-08 — Volkan, "Adopt, cl one-pair": without a band the fill's lanes take series of different lengths under
  clang and GCC; under cl, blocks of unequal length keep the one-pair route (cl's lanes are scalar).
- 2026-10-08 — Volkan: "Also have a look at cache misses, it is a very important metric for speed." Kits report cache
  misses beside wall time, base against candidate. VTune cannot sample on the Windows box (Hyper-V with VBS);
  Windows' ETW counters can, from an elevated `.cmd` that Volkan runs.
- 2026-10-08 — Precision (Volkan, after asking why the default is double): "Okay let's make it double, but it should
  be easily changable like DataLoader(d_type = f32) then data is loaded as f32. Same when creating a distance
  matrix but I guess since it can go between huge ranges it should be double most of the time." Double stays the
  default for series and matrix; float32 becomes a one-word option where data is loaded, in every language, and an
  opt-in for the matrix.
- 2026-10-08 — Memory access (Volkan): "Please algorithms should also prioritise having a good memory access
  pattern. Especially when we have mmap files, we should be accessing memory block by block like a well-written
  matrix multiplication algorithm, so that we don't randomly read and write time series all the time. So read
  enough time series that fit in the ram, then do the parallel thing, then load the other block etc." An algorithm
  that sweeps series or the matrix does so in blocks sized to a memory budget (RAM for mapped files, cache for data
  in memory), parallel within a block.
- 2026-10-09 — Spec `plans/2026-10-09-dtype-and-blocked-access.md` approved by Volkan (question tool), each answer
  the recommended one: the blocked SWAP "Yes, identical swaps" (its accepted swaps and medoids equal the
  sequential sweep's: MAP §7's "sequential on purpose" holds for the result, not the evaluation); mapped series
  "Measure first" (after CS); memory "One key, automatic" (`ram_limit` defaults to what the process may use, sizes
  slabs and batches, maps the matrix by its bytes; `--mmap-threshold` goes). Then "AL right after DT": CLARA's
  assignment through the lanes (17.9 Gcell/s per-pair today against the lanes fill's 105–131); the order is DT, AL,
  BF, CS, SW, FM.
- 2026-10-09 — Volkan: "Can you also throw an agent in parallel adding  * @author Volkan Kumtepeli in files that
  doesn't have it. It could be above the Claude if there is a Claude author." Every source file names Volkan first
  (unit AU); vendored code and `.claude/` records are left alone.
- 2026-10-09 — Volkan: "Items inside .claude folder are really special, so I wouldn't want everybody to see. So could
  you remove "baselines", "CHARTER" from git tracking? I don't want to see them on github. Also CITATIONS.md should
  be moved to docs folder. For "DECISIONS.md", "LESSONS.md" and "PLAN.md" you could remove the obsolete ones, or
  already completed ones." Done in `034d8994` and the prune that follows; the copies already pushed
  (`origin/design-2.0` holds 310 baseline files and the charter) stay in GitHub's history unless Volkan rewrites it
  (open, with the Kasper folder of 09-28).
- 2026-10-09 — Volkan (question tool), the machine not quiet since 02:13 (Sophos and Tanium a steady 8–10 % of
  the CPU, nothing of ours running): "Run now, load recorded". Timed runs may start at total CPU < 12 %, the
  background recorded beside every run, 20 alternating pairs per 24-thread shape, a narrow miss getting one
  registered second attempt; < 6 % stays the gate whenever the scanners allow it.
- 2026-10-09 — Volkan (question tool) on BF, whose GCC s1t (N 1000, 24 threads, a 40 ms fill) missed ≥ 0.98×
  twice (0.892×, 0.951×): "Merge now, misses recorded"; GCC s1t, e3 and c1's closing save stay recorded as
  missed, and the design's 8-column tasks for small fills become a PLAN candidate.
- 2026-10-09 — Volkan (question tool) on PE, the Parquet stream overshooting its RAM limit (1.48 GB under
  `--ram-limit 1G`): "Next, before SW". PE runs after CS, before SW, FM and MS.
- 2026-10-10 — Volkan (question tool) on FM, the opt-in float32 matrix, each answer the recommended one: `.dtwm` v5
  records the rows written ("Yes, rows-written field": a new mapped matrix skips its NaN prefill, rows past the count
  are not computed); a float32 matrix exports at its width ("Stored width": NumPy float32, MATLAB `single`); v4 files
  are read and mapped as float64 ("Read and map v4"); `matrix_dtype` also sets CLARA's sample matrices and TADPole's
  cache ("Yes, one rule").
- 2026-10-10 — Volkan (question tool) on SW's 8-wide swap loop, which under GCC 13.3 reads the stack 0.5 times a
  lookup where base's never did (fewer instructions, one-thread swaps 1.25–1.64× faster): "Fix it first". The
  hot-loop rule holds without exception: SW returns to its implementer before any merge.
- 2026-10-10 — Volkan (question tool) on MS's measurement: disk space, "I opened up some space for you now" (C:
  224 GB free at 13:21, WSL's ext4.vhdx on it); "WSL decides" (every cold and scale run in WSL, Windows only the
  NVMe's raw ceiling); scale "2^25 series, float64" (34.5 GB, extrapolated to 10^8).
- 2026-10-10 — Volkan (question tool) on SW at af18d202, re-timed (t1, t5, t6a, t6b, p1–p4 and the scores PASS; t2,
  t3, t4, m1 and e1 FALSIFIED; GCC's 24-thread t4–t6b not measurable in WSL while a host core was busy): "Merge now
  (Recommended)", the misses recorded; he re-ran e1's ETW at 21:17 ("Done: it finished").
- 2026-10-10 — Volkan (question tool) on MS's failed validity gate (the band-20 control decoder-bound, the assignment
  4.5× faster than designed; W agrees with the model at both bands): "Accept as is". The registered decision stands:
  decoding binds, so mapped series, branch M (a kit-only prototype) first.
