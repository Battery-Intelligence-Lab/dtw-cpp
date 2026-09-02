# DTWC++ Lessons Learned

Critical knowledge to avoid repeating mistakes.

---

## Mathematical

- **DTW is NOT a metric.** It violates the triangle inequality. Generic MILP
  lower/upper-bound certificates remain valid for a well-formed DTW cost
  matrix; what cannot be transferred automatically are metric-only
  approximation or integrality-gap results.
- **DTW-AROW ≠ zero-cost DTW.** AROW constrains missing values to diagonal alignment.
- **LB_Keogh admissibility is a domain contract, not just a metric name.**
  For finite scalar inputs and a centered envelope whose radius covers the
  actual fixed DTW window, the projection sum is admissible for additive L1
  and unrooted squared-L2 objectives. Coordinatewise boxes extend that proof
  to dependent additive multivariate DTW; summing the scalar proofs extends it
  to independent additive DTW. It does not prove the shared-path multivariate
  Euclidean, cosine, or Huber forms, and a negative helper band is radius zero,
  not a full-DTW envelope.
- **The public `lb_webb` name denotes local `LB_Webb_NoLR` plus a tail cap, not full Algorithm 2.** Its first pass is directional Keogh and its supported L1/squared corrections are nonnegative, so the local directional result dominates matching-direction Keogh and its symmetric maximum dominates symmetric Keogh. Full Algorithm 2 includes `MinLRPaths`; omitting it has no universal order (the paper's Wafer table has NoLR 0.96904 versus full Webb 0.96891). Only the separate trailing-flag cap has the proved direction `production <= exact-predicate NoLR`. Name the variant before transferring a theorem or benchmark.
- **LB_Enhanced/Keogh ordering depends on effective V.** At `V=1`, forced-corner costs dominate endpoint interval projections, so directional Enhanced dominates matching-direction Keogh. At effective `V>=2`, exact D3 witnesses—not an attribution to SDM 2019—establish both strict directions. The 2026-07-08 bench measured Enhanced gains of 3.3% at a 10% band and 10.9% at a 40% band on that fixture; those observations are not a universal band-width law. The live cascade takes the maximum.
- **A DTW lower bound cannot reduce DTW calls when building an EXACT full distance matrix.** LBs skip work only where the exact value is not needed (NN-search: skip if LB ≥ best-so-far). An exact matrix needs every entry, so a tighter LB (Webb/Enhanced) buys nothing there — its value is for NN/query paths and for pair-skipping density pruning (TADPole). Exact-matrix speedups come from cell-pruning (PrunedDTW/EAPruned, returns exact), not from tighter bounds.
- **LB/UB pair-skip pruning needs a density/NN consumer — k-medoids, MIP and LR-core are NOT it** (they read essentially the whole N×N matrix: `fast_pam_swap` fills it and the SWAP loop reads every candidate pair). That is WHY TADPole (Task 5.3) is a NEW `Method` (density-peaks clustering), not an accelerator bolted onto the existing methods. The prunable step is the cutoff-kernel density `ρ_i=|{j: d<dc}|` (binary "d<dc" test) + the δ nearest-higher-density search. Cutoff kernel only — the Gaussian ρ=Σexp(−(d/dc)²) needs every exact distance and cannot prune.
- **TADPole's δ(global densest point) = max of all OTHER points' δ (Begum Table 2), NOT Rodriguez–Laio's max_j d(densest,j).** Use TADPole's convention or the highest-density point's γ=ρ·δ (hence its center-selection) diverges. Also fix a strict total order for ρ-ties (ρ desc, index asc) or δ/parents/labels are nondeterministic. Exact arithmetic preserves prune/brute decisions in the D2 domain and the exactly representable regression confirms that case; bit-level identity at a floating threshold remains D17.
- **`compute_envelopes(series, band<0)` gives the band-0 envelope (= the series itself), NOT a full-warp envelope.** It does `w = max(band,0)`, so passing `band=-1` (full DTW) yields `LB_Keogh = Σ|q−s| = ED ≥ full-DTW` — an INVALID lower bound (prunes true neighbours). For full DTW pass a window ≥ series length (global min/max envelope) to get a valid weak LB. Cost TADPole a wrong-ρ bug in review; caught before merge by the LB≤DTW bounds test.
- **A lower bound's provenance is part of its value.** A bare pair of
  upper/lower vectors does not say which source, metric units, radius/full
  mode, or length produced it. Shape-correct arrays from a narrower window can
  make flawless LB_Keogh algebra inadmissible; truncating to the shortest
  vector merely hides the contract breach. Carry and validate provenance at
  the public boundary, then keep unchecked pointer kernels internal (F46).
- **A derivation-pinned line is a contract; run `check_docs_contract` before
  changing it.** `lb_keogh`'s `const auto n = std::min(query.size(),
  env.upper.size());` is the live witness for D2's unequal-length prefix
  theorem and its 17,712-case oracle, and
  `scripts/check_docs_contract.py` pins it verbatim. Replacing it with an
  equal-length reject while fixing a genuine ragged-envelope OOB read silently
  invalidated the whole D2 campaign — the compiler and the unit tests were both
  happy. The correct fix keeps the derived truncation and adds a coverage check
  (`envelope_covers`: every array indexed over the prefix must reach it),
  reserving exact equality for `lb_enhanced`/`lb_webb`, which have no prefix
  theorem. Before editing any line inside a derivation's implementation, grep
  `scripts/check_docs_contract.py` for it and run
  `uv run python scripts/check_docs_contract.py`.
- **Metric compatibility traits must test units, not family names.** Scalar L2
  is `|a-b|`, but unrooted squared L2 is `(a-b)^2`; advertising one raw
  absolute-value LB_Kim implementation for both makes `0.5` a claimed lower
  bound on a true squared cost of `0.25`. A compile-time “valid” trait is a
  scientific claim and needs a metric-discriminating executable fixture (F47).
- **A zero pruning count does not prove a pruning stage ran.** The safe
  full-DTW TADPole fixture and a disabled LB both report zero decisions. Pair
  it with a separated-range case that must prune: the joint `(0,1)` fingerprint
  distinguishes disabled, unsafe radius-zero, and correct global-envelope
  implementations. Reachability needs both a negative and a positive control.
- **Admissibility does not imply tightness or a prune rate.** From
  `LB <= d` and `d >= dc` one cannot infer `LB >= dc`; a valid weak bound may
  clear no thresholds. Register an executed decision count or bound-gap
  distribution before claiming work reduction.
- **A bound-decision counter is not automatically an avoided-work counter.**
  TADPole deliberately pre-triggers pair `(0,1)` to bind its lazy distance
  function before parallel work. That pair can later increment
  `pruned_by_lb` even though its DTW was already computed. `pruned_by_lb` and
  `pruned_by_ub` count density decisions; only the deduplicated `dtw_calls`
  ledger can support an avoided-work claim.
- **Additive multivariate objectives need a unit model.** Summing channels is
  dimensionally meaningful only when they share a commensurate unit or have
  been scaled/nondimensionalized. A correct coordinatewise inequality does
  not license a single `U`/`U^2` ledger for raw heterogeneous physical units.
- **Re-run counterexamples after changing upstream geometry.** F29's inherited
  unequal-length band-zero fixture stopped being a finite-DTW case when D1
  repaired the window to `|i-j| <= band`. The old narrative was not evidence:
  an explicit path arbiter showed prefix LB_Keogh is admissible for every
  registered feasible fixed window. A killed premise must be replaced by a
  feasible discriminator, never carried forward by name.
- **Bound shortcuts must share the exact distance's empty/no-path domain.**
  Returning zero from an empty envelope or diagonal upper bound while exact
  DTW returns the maximum sentinel changes TADPole neighbour and medoid
  decisions. Either reject empty series once or bypass every shortcut under
  one documented policy; mixing local “harmless” empty defaults is unsafe
  (F48).
- **Published caches must bind every computation parameter.** A helper that
  accepts a band independent of `Problem::band()` can compute under one window
  and publish into a cache whose validity snapshot names another. Reject the
  mismatch before the first write or remove the redundant parameter; output
  equality on easy fixtures cannot establish provenance (F49).
- **Clamp logical full windows before device integer arithmetic.** A public
  `INT_MAX` radius is semantically just global coverage, but evaluating
  `k+w+1` in signed GPU-kernel arithmetic can overflow before clipping. Clamp
  on the host to `max_length-1`, validate narrowing, and execute the extreme
  case on the real backend; source inspection alone cannot assert the wrapped
  numeric result (F50).
- **WDTW/ADTW/DDTW/Soft-DTW are distinct recurrence/cost policies, not metric swaps.** They share the unified kernel family, but each still changes DTW semantics in a real way.
- **EAPruned (exact cell-pruning) needs a RELAXED prune threshold under the Clang/GCC Release reassociation flag set.** **[confirmed]** The historical randomized gate recorded a no-slack threshold returning +inf on approximately 1/200 pairs (`.claude/baselines/2026-07-08-eap.md`); the retained fix prunes against `thr = ub·(1 + n_long·16·ε)`, and relaxing a pruning threshold only adds computed cells. **[inferred]** The likely mechanism is association/rounding disagreement between the DP sum and diagonal-UB sum: the build permits reassociation through `-fassociative-math` together with `-fno-signed-zeros` and `-fno-trapping-math`, but no exact failing bytes or flag-ablation artifact survives to prove causation. The factor 16 is regression-tested, not a proved worst-case bound; its derivation remains PLAN R2-D4. The build is not `-ffast-math`: `cmake/StandardProjectSettings.cmake` supplies an explicit flag set without `-ffinite-math-only`, and `dtwc/CMakeLists.txt` adds `-fno-finite-math-only` to `dtwc++`.
- **EAPruned's exact-matrix speedup is data-cohesion-dependent, NOT a flat multiplier.** With a diagonal UB and no NN cutoff (all-pairs exact), pruning is heavy only for near-diagonal pairs (DTW ≈ Euclidean): near-diagonal 6–12×, mild-warp 5–10×, but UNRELATED cross-cluster pairs prune ~nothing (~98% cells) and get only ~1.5× (from the leaner inner loop, not pruning). So a cohesive dataset (many within-cluster pairs) speeds the matrix build a lot; an incoherent one barely. The paper's headline 2.88× is NN-search with a tightening cutoff — a DIFFERENT regime; do not register it as the exact-matrix band. Never slower than plain DP in any measured case.
- **Match a reference implementation by reading its SOURCE, not just the paper.** aeon's TWE names a term `del_x_squared_dist` but the function is `_univariate_euclidean_distance` = √(squared) = |a−b| for univariate (NOT squared); and it front-zero-pads both series (`_pad_arrs`), so the `|x̂ᵢ₋₁ − ŷⱼ₋₁|` match term is `|0−0|=0` at the first cell. Coding MSM/TWE from the paper alone would have mismatched the oracle. Both MSM and TWE are exact metrics (verified `d(x,y)==d(y,x)` numerically) — so orient n_short≤n_long and roll a buffer of the shorter axis (O(min); a full O(n·m) matrix is ~512MB/thread at n≈8k).
- **Rolling-buffer DP: write row 0 into the buffer that becomes `prev` after the first swap.** TWE returned the MAX sentinel because I initialised `prev[0]=0` but the loop does `swap(prev,curr)` FIRST — moving the initialised row into `curr` and reading an all-MAX `prev`. Fix: write row 0 into `curr` (so the first swap makes it `prev`), matching the MSM kernel. For the padding cells use `numeric_limits::max()` plus a guarded `add(u,v)=u==MAX?MAX:u+v`, matching the DTW no-neighbour sentinel and preventing `MAX + cost` overflow.
- **MSM/TWE are univariate + unbanded in v1; reject MV at BIND time, not per-call.** A multivariate request throws `InvalidInput` in `make_msm`/`make_twe` (called serially by `rebind_dtw_fn` before the parallel fill) — never inside the per-pair `dtw_fn_` (an exception in an OpenMP region is UB). Documented, not a silent single-channel collapse.
- **A rounded hand-repro can hide a floating-point bug.** The EAP +inf bug reproduced on full-precision random inputs but NOT on the 3-decimal values printed for it — the rounding lost the ULP-level precision that triggered the association/rounding-sensitive overshoot. Debug from the exact failing bytes (rerun the seeded generator), never from a pretty-printed copy.

## Apple Silicon / macOS (benchmarked 2026-04-12, M2 Max 8P+4E)

- **Don't worry about E-cores vs P-cores on Apple Silicon.** Measured on `fillDistanceMatrix/100/1000/-1`: 1→8 threads gives 6.88× (all P-cores), 8→12 threads gives another 1.33× (E-cores help). Full 12-thread speedup is 9.14× (76% efficiency). E-cores are a net win with dynamic scheduling — do NOT cap to P-core count.
- **Thread-placement knobs did not matter in the recorded M2 Max sweep.** With
  that machine's Homebrew libomp at T=12, default/close/spread differed by less
  than 0.3% (1606/1610/1609 ms). This is one architecture/runtime result, not
  proof that `OMP_PROC_BIND` or `OMP_PLACES` are universal Darwin no-ops.
- **`sysctlbyname("hw.perflevel0.logicalcpu")` is not worth calling** for thread-count tuning. `omp_get_max_threads()` gives the right answer. Skip the P-core auto-cap code path unless a future benchmark contradicts this.

## Python packaging / bindings

- **Never infer Python import provenance from the checkout layout.** The
  2026-07-23 probe found a mixed environment: `dtwcpp` and `_api.py` resolve to
  `python/dtwcpp/` in the repository, while `_dtwcpp_core` resolves to the venv
  site-packages `.pyd`. Print all three `__file__` paths before a gate. Manage
  dependencies only through `uv add/remove`. For extension evidence, rebuild
  through the configured wheel/CMake recipe, copy a fresh `.pyd` + `libomp.dll`,
  then verify import and one newly added symbol before pytest;
  an editable pure-Python layer can otherwise mask a stale native core.
- **An exact-base Python arbiter must defeat editable-install import hooks.**
  Merely prepending a detached worktree to `PYTHONPATH` still allowed the
  current checkout's editable finder to supply the package. Remove that finder
  from the isolated environment, remove the current source path, and assert
  both the package and native-extension `__file__` paths plus the extension
  digest before judging whether a failure is pre-existing.
- **`dtwc_cl` names batch-row series `1..N` (1-based row index), ignoring any id column.** With a TSV of one series per row + `--skip-cols 0`, the `NAME_labels.csv` (`name,cluster`) uses names `1..N`. Its rows are **not guaranteed in input order** (the binary may lexically sort: `1,10,11,2,...`). ALWAYS map labels by name (`labels[i] = clusters[str(i+1)]`), never by row position. Verified against `build/bin/dtwc_cl.exe` 2026-06-30.
- **`./bin/dtwc_cl.exe` can be stale.** A top-level `bin/` binary may predate current flags (it rejected `--skip-cols`/`-k`). Prefer `build-*/bin/dtwc_cl` (current builds) — `find_dtwc_binary` does this.

## C++ Performance

- **Generalising can beat specialising.** Phase 1 unified the
  Standard/ADTW/WDTW/DDTW/ZeroCost-missing paths behind policy kernels. The
  inherited benchmark artifacts record: `BM_dtwFull_L/1000`, 2967 → 1924 μs
  (**1.54×**); `BM_dtwBanded/1000/50`, 258 → 146 μs (**1.77×**); and
  `BM_wdtwBanded_g/1000/50`, 450 → 159 μs (**2.83×**). These exact labels
  matter: the earlier lesson attached the first two ratios to the wrong
  variants. Template Cost/Cell policies keep one audited recurrence while
  allowing compile-time specialization.
- **The old SIMD ceiling was an operation-count estimate, not a measured
  maximum.** Commit `f3af033` estimated 1.29× from 28 operations/cell versus 9
  for scalar multi-pair DTW. Earlier experiments were route-specific: the
  initial scatter/gather batch-DTW was slower, while the later equal-length SoA
  batch measured about 2.8× versus four sequential calls (`2dac179`,
  `367a9b4`). The dispatched route was correctness-invalid because it ignored
  bands and variants (`923f723`), and the Highway surface was removed in
  `d670143`. The profiler baseline later denied PMU access and did not establish
  the old latency, L1-bandwidth, or cache claims. SIMD stays killed unless new
  registered correctness and performance evidence overturns that decision.
- **The recorded FastPAM1 table shows 2.95×–8.06×, not k×.** At N=1000, the
  naive/FastPAM1 ratios were 2.95, 3.45, 2.95, 4.30, and 8.06 for
  k={10,20,50,100,200}; the trend is not monotone. These are advisory
  wall-clock numbers from `.claude/baselines/2026-07-08-faster-pam-bench.md`.
  Code structure supports the **[inferred]** explanation that both variants
  retain N² distance-matrix reads per iteration while FastPAM1 removes O(k)
  arithmetic, but no PMU artifact proves a memory-bound bottleneck. FasterPAM's
  eager-swap result is a separate column and must not be conflated with this
  decomposition ratio.
- **FasterPAM's removal-loss decomposition is UNDEFINED at k=1 (`second_dist = +inf`).** `ρ(m)=Σ(d₂−d₁)=inf` and the per-point correction `d₁−d₂=−inf`, so `ρ + correction = inf + (−inf) = NaN`; `NaN < best` is false ⇒ no swap accepted ⇒ the routine silently returns the BUILD medoid instead of the 1-medoid optimum. Special-case k=1 to the direct `argmin_x Σ_o d(x,o)`. The general k≥2 path is fine (every point has a finite second-nearest). Any decomposition that subtracts a possibly-infinite `d₂` must guard the degenerate no-second case.
- **The "Pruned" distance-matrix strategy is a PESSIMISATION for an exact matrix — it does STRICTLY MORE work than BruteForce.** `fill_distance_matrix_pruned` feeds the LB into an early-abandon threshold, but the kernel's abandon returns the `maxValue` sentinel (never the exact value), so every abandoned pair is recomputed fully: `work = brute + LB_overhead + Σ partial-DTW ≥ brute`. Since `lb ≤ dtw`, the entry test `lb > threshold ⇒ dtw > threshold ⇒` abandon always fires ⇒ always recomputes. The `pruned_by_lb_*` counters count pairs that were made MORE expensive; the pre-existing tests only checked digit-identity, never a speedup, so this went unnoticed. Do NOT tighten the LB expecting a matrix-build speedup — a tighter bound only moves more pairs into the worse bucket. The honest exact-matrix speedup is cell-pruning (PrunedDTW/EAPruned), not a tighter LB.
- **One historical triangular-index lookup-table experiment regressed 2.61 →
  2.75 ms (+5%).** No raw transcript survives, so it does not establish cycle
  counts or a universal cache mechanism. Keep the simple arithmetic unless a
  new registered workload disproves that result.
- **Nested `std::min` helped specific legacy kernels, not every recurrence.**
  The tracked benchmark at `65e249b` measured 2.45×–3.14× for its legacy
  full/banded cases, while rolling `BM_dtwFull_L/4000` was effectively
  unchanged (1.006×). Do not attribute the result categorically to temporary
  creation; benchmark the actual kernel being changed.
- **MV dispatch must branch on `ndim` for EVERY strategy, not just the ones
  someone remembered.** `make_interpolate` and `make_soft_dtw` had no `ndim > 1`
  branch, so a multivariate request ran the *univariate* recurrence over the
  interleaved channel stream and `band` counted flat elements instead of
  timesteps — a silently wrong number, no diagnostic, no test. MSM/TWE had
  already established the pattern (reject at bind time, serial, before the
  parallel fill) and the two strategies were simply never brought in line. When
  a dispatcher grows a new axis, enumerate the FULL cross-product and give every
  cell either an implementation or an explicit `InvalidInput`.
- **A strategy enum that promises to throw must throw at every entry point.**
  `MissingStrategy::Error` was enforced only in `Problem::fill_distance_matrix`;
  `distance::dtw` and `dtw_runtime` ran the recurrence on NaN and returned NaN —
  which is also the distance matrix's "uncomputed" sentinel, so the result was
  indistinguishable from an unfilled entry and `all_computed()` never became
  true. A contract implemented in one of three entry points is not implemented.
- **A negative band means UNBANDED, never radius 0 — clamping it breaks
  admissibility.** `lb_enhanced`/`lb_webb` used `max(band, 0)`, which pins the
  elastic arms to the diagonal; an unbanded path may step around those cells,
  so the "lower bound" exceeded the true DTW (`A=[0,5,0,0]`, `B=[0,0,5,0]`:
  10 vs 0). Every bound derived under a window assumption must refuse, not
  coerce, an out-of-domain window.
- **Validate every array you will index, not just the first one.** `Envelope`
  and `WebbEnvelope` are aggregates whose members can be resized independently;
  the entry points checked `upper.size()` and then read `lower`/`ul`/`lu`.
  Same class of bug as the unchecked WDTW weight span (`weights[|i-j|]` needs
  `max(nx, ny)` entries). A size check on one member of an aggregate is not a
  size check on the aggregate.
- **Never bound a parallel region with `omp_set_num_threads`.** It mutates
  process-wide state that nothing restores, so one constrained call (a
  k-means++ init at 2 workers) pinned every later fill and made pruning
  statistics depend on call order. Use a `num_threads(...)` clause, which is
  scoped to the region.
- **Lambda capture-by-value creates stale parameter bugs.** Capture `[this]` and read at invocation time, not `[b=band]`.
- **NaN is the ONLY safe sentinel** for distance matrix uncomputed entries. Soft-DTW returns negatives — any fixed sentinel collides.
- **The mmap numbers are one hot-cache experiment, not a default-policy proof.**
  A historical N=5000 (~95 MB) run reported +5% random access, +9% sequential,
  78× faster open, and 48× faster CLARA views versus copy, but retained no raw
  transcript and did not test cold-cache or memory-pressure behavior. Current
  CLI threshold policy must stand on its own route tests.
- **DTWC++ currently uses LLFIO for its optional mmap backend.** That is a
  repository implementation fact, not a timeless LLFIO-versus-mio ranking.
  Replacing it would require an optional-dependency build proof plus mapped
  correctness, locking, and failure-path gates.

## Data Formats & I/O (exploratory work 2026-04-08)

- **The inherited I/O timings are not decisive evidence.**
  `bench_parquet_access.py` generated 30 battery files but timed DTW on only 10
  random pairs truncated to length 500 in pure Python, and no run output was
  retained. The former 10–100× compute/I/O, 0.4 μs/series, 4 ms open, and
  eight-core Zstd/NVMe rules therefore remain hypotheses requiring a fresh
  registered native-binary benchmark.
- **Commit `420f764` historically records a generated battery fixture as
  199.6 MB versus 9.7 MB under Parquet Zstd (20.58×), but no raw transcript
  survives.** Treat those sizes as inherited prose, not a confirmed
  measurement. The old 100 GB → 2 TB statement was only a linear extrapolation
  from them. Compression/storage choices must be made from the actual dataset
  and access pattern.
- **The current Parquet reader decodes/decompresses values; the Arrow IPC
  reader can expose mmap-backed Arrow buffers.** That implementation distinction
  does not prove equal end-to-end speed versus `.dtws`.
- **Raw HDF5 dataset offsets were rejected as an unverified mmap shortcut.**
  Any future HDF5 route must validate contiguous layout, filters, offsets, and
  file lifetime rather than assuming `H5Dget_offset()` alone makes a safe view.
- **Use `LargeList` when cumulative offsets may exceed `INT32_MAX`, and validate
  offsets before access.** Current Arrow/Parquet readers intentionally support
  both `List` and `LargeList`; ordinary `List` input is not itself an error.

## Float32 (benchmarked 2026-04-08)

- **Float32 can improve both storage and measured throughput.** It halves the
  series payload, and the registered dated workloads measured **1.57×–1.90×**
  speedups over Float64 (`.claude/baselines/2026-07-06-phase0.md`). Those
  wall-clock results are advisory on the shared host; they disprove the old
  categorical "identical speed / purely memory" claims rather than establish a
  universal ratio.
- **The inherited Float32 accuracy probe was narrow.** Its maximum relative
  error, **2.74e-05**, survives only as historical prose from `420f764`; no raw
  transcript survives. The associated script uses only 10 random length-500
  pairs after Float32 input rounding followed by double accumulation. Neither
  the inherited number nor that narrow design proves negligible medoid or
  clustering error; a decision-sensitive fixture would be required.
- **Float64 is the default; Float32 is opt-in.** Float32 can halve payload
  memory, but the public default and Problem/Result matrix contract remain
  double precision.

## C++ Implementation

- **NaN for missing data.** Use `quiet_NaN()`, check via `std::isnan()`. Safe because `-ffinite-math-only` is NOT set.
- **DTWC++'s compact-MIP flattening differs by backend.**
  `mip_Highs.cpp` stores `A[i,j]` row-major and `mip_Gurobi.cpp` stores it
  column-major; both place diagonals at `i*(Nb+1)`. This is an implementation
  choice in these adapters, not an inherent solver-wide convention.
- **Public invalid states require typed errors, not assertions.** View-mode
  `p_vec(i)` and `get_name(i)` currently use `assert(!data.is_view())`, which
  disappears under `NDEBUG`; this is open finding **F25**, not accepted advice.
  Assertions are for internal invariants after public validation.

## Cross-Language Bindings

- **Do not invent MEX unwinding semantics.** MathWorks documents that
  `mexErrMsgIdAndTxt` terminates the MEX call and returns to MATLAB, and that a
  C++ object which goes out of scope on an error has its destructor called. It
  does not document the old "longjmp skips destructors" claim. Keep the
  conservative owner-scope pattern—catch, leave the scope owning native
  resources, then call `mexErrMsgIdAndTxt`—without assigning it a fabricated
  mechanism.
- **The MEX gateway pairs `mexLock` with `mexAtExit` cleanup.** MathWorks
  documents that `mexLock` prevents clearing a MEX function and that
  `mexAtExit` registers cleanup before clear/termination. Those contracts
  support the current handle-lifetime pattern; they do not prove the inherited
  causal claim that `mexLock` itself prevents shutdown crashes.
- **A MATLAB batch gate requires both clean exit and executed assertions.**
  Nonzero exit or crash is a failure even if expected text appeared. Put the
  freshly built MEX directory last in `addpath` so it prepends, confirm
  `which('dtwc_mex','-all')`, then run the full gate. The 2026-07-10 evidence
  was 61/61 overall, with `test_parallelisation` engaging 24 OpenMP threads.
  The current F18 recount is 82 collected / 81 passed / 0 failed / 1 expected
  opposite-flavor capability skip on both MATLAB versions.
- **nanobind is the current Python binding generator.** Keep GIL-release and
  array behavior tied to the explicit live bindings. No retained artifact
  establishes the former stable-ABI, binary-size multiplier, or native-CUDA
  claims.

## HiGHS MIP Solver (IMPORTANT — workaround in place)

- **The HiGHS warm-start assertion is a historical v1.14.0 observation.**
  DTWC++ now pins **v1.15.1**. `cmake/Dependencies.cmake` still applies
  `NDEBUG` defensively to the dependency, but the old `ub_consistent` failure
  has not been reproduced on v1.15.1 and no current artifact proves its root
  cause or solution-correctness impact. Treat the broad assertion suppression
  as technical debt to revalidate, not as a verified upstream diagnosis.

## Build System

- **CUDA multi-version on Windows:** Generate `Directory.Build.props` with `<CudaToolkitCustomDir>`.
- **MSVC flags leak into nvcc:** Use `$<$<COMPILE_LANGUAGE:C,CXX>:...>` generator expressions.
- **CMake function-local discovery state needs an explicit return contract.**
  Variables such as `Parquet_FOUND` do not leave a function unless exported
  (F9 fixed this with `PARENT_SCOPE`). Prefer imported targets when the package
  provides them, but do not replace every valid `X_FOUND` contract with a
  blanket `TARGET` rule.
- **PyArrow can supply usable Arrow/Parquet CMake packages on Windows.** The
  decisive F9 build consumed PyArrow 23's headers, import libraries, and runtime
  DLLs successfully. Treat DLL placement as an explicit build/run recipe rather
  than prescribing a different package manager without evidence.
- **A Windows shared dependency linked PUBLIC reaches every test executable.**
  In the PyArrow-backed build, ordinary unit executables imported
  `arrow.dll`/`parquet.dll`, and `arrow.dll` in turn needed a hash-named runtime
  from `pyarrow.libs`. Adding PATH only to the two Arrow CLI tests left unit
  CTests failing before `main` with `0xC0000135`. Compute the proven runtime
  directories once and attach them to every registered test in the affected
  Arrow-linked directory; verify generated CTest metadata, not source order.
  F20 exposed the same rule in Python-spawned real-CLI children: the parent
  extension imported successfully, but the children exited `0xC0000135` until
  `dtwcpp`, `pyarrow`, and `pyarrow.libs` were all present on their inherited
  `PATH`. An in-process import is not a subprocess runtime-dependency gate.

## ARC SLURM Hardware

- **htc GPU compute capabilities (corrected from docs).** The ARC docs list CUDA toolkit version, not compute capability. Actual values: P100=6.0, V100=7.0, RTX8000/TitanRTX=7.5, A100=8.0, RTXA6000=8.6, L40S=8.9, H100/GH200=9.0.
- **Rome (htc-g019) and Broadwell (htc-g045-049) lack AVX-512.** Use
  `DTWC_ARCH_LEVEL=v3` for a portable x86 htc build; there is no one x86-64-v4
  recipe for every ARC node. Builds must be architecture-specific.
- **Grace Hopper (htc-g057) is AArch64.** It requires a separate ARM build;
  AArch64 build/runtime remains unverified because agents may not submit to
  SLURM/HPC.

## Arrow/Parquet

- **Never `static_pointer_cast<DoubleArray>` without checking value type.** Parquet list columns can store Float (32-bit) values. Casting to DoubleArray reinterprets float bits as double — silent data corruption. Always check `values->type_id()` first.
- **Parquet metadata is cheap, but `total_uncompressed_size` is not a decoded-RAM oracle.** Dictionary/page encodings can make that field far smaller than the values Arrow materialises. For a selected numeric leaf, budget at least `num_values * source_width`, take the maximum with encoded metadata, and include the target buffers plus vector/name objects. The eager Float32 route currently peaks across both its intermediate Float64 `Data` and the converted Float32 `Data`; measuring only the final representation undercounts it.
- **A streaming RAM cap must be decided before payload I/O and must model retained-plus-transient data.** Loading the full Parquet table and then opening a row-group reader only adds chunks to the resident peak. Row groups are indivisible: budget the largest selected group beside retained sample/medoid payloads, and reject with a smaller-row-group remedy when it cannot fit. State explicitly whether a cap governs series materialisation or whole-process RSS; F7 governs the former.
- **Arrow field indices and Parquet physical-column indices are different namespaces.** A preceding Struct can own multiple leaves, so passing its top-level Arrow index to `ReadTable`/`ReadRowGroups` selects the wrong physical column. Count physical leaves in every preceding top-level field and share that mapping between eager and chunked readers.
- **Scalar and list Parquet columns have different logical N.** One scalar column is one time series spanning all rows; one List/LargeList cell is one series. Eager, metadata, sparse, and streaming paths must use one schema selector and the same Float32/Float64-only rule, or `auto`, RAM planning, and output names will disagree before the algorithm starts.
- **A guard that rejects an unsupported option must live OUTSIDE `#ifdef DTWC_HAS_PARQUET`.** The canonical gate build is `DTWC_ENABLE_ARROW=OFF`, so anything inside that guard does not exist in the binary the gate actually tests. A `--ram-limit` rejection written inside the guard passed its unit test (the test TU defines the macro), passed the 113-test gate, and still let a real `dtwc_cl -i data.csv --ram-limit 1G` run to completion — the flag was accepted and ignored. Only driving the real binary caught it. Corollary: **a green unit test on a helper proves the helper works, never that it is reachable.** Route-matrix rejections belong beside the filesystem classification, which needs no Arrow.
- **Optional-dependency OFF and ON builds prove complementary contracts.** The
  canonical Arrow-OFF build intentionally skips `test_io_readers`; the fresh
  Arrow-ON CMake build against PyArrow 23 must compile the guarded production
  branch and run the subject. F9's local gate did so with **390 assertions in
  11 Arrow/Parquet cases** and no skip (`833f570`, `0c91c9b`;
  `.claude/baselines/2026-07-23-f9-arrow-gate.md`).
- **A gate that can silently SKIP its subject is not a gate—assert the subject
  RAN.** CTest treats return code 4 as a pass
  (`cmake/Coverage.cmake:10`). The Arrow-ON workflow parser therefore rejects
  skip text, requires a unique Catch2 success summary, and enforces assertion
  and case floors (`e323197`). Configure announcements alone are not evidence;
  inspect the compile definition and direct binary output. Hosted workflow
  execution remains operator-owned and unclaimed.
- **F7 and F9 are closed by executable routes.** F7's non-Parquet RAM rejection
  is pinned through the real CLI (`7c71602`). F9 is closed by the complementary
  Arrow-ON execution above, not by the intentionally skipped Arrow-OFF row.

## Refactoring Process

- **One task per commit is part of the validation protocol, not cosmetic history.** The `8debf1d` omnibus combined Tasks 5.8–5.11, release work, documentation, API changes, tests, and fixes across 103 files. That made task-level gates, review, bisection, and honest rollback inseparable. Land one conventional commit per task or independently confirmed finding, with its own tests and durable run-log; do not use a later omnibus commit to stand in for missing per-task evidence.
- **A FROZEN contract can change only through an explicit, dated decision.** Non-additive edits require a decision-log entry that names the old rule, new rule, rationale, compatibility impact, and owner. Never silently rewrite the contract or delete the governance clause that requires the decision—that removes the evidence needed to distinguish an authorized scope change from accidental drift.
- **Cross-validation is the gate for policy migrations.** Before swapping a dispatch from impl A to impl B (e.g. `dtwAROW_banded` → `dtw_kernel_banded<T, SpanAROWL1Cost<T>, AROWCell>`), write a test that runs BOTH on representative inputs (no-NaN, interior NaN, leading/trailing NaN, all-NaN × bands {1..4}) and asserts bit-for-bit agreement within 1e-10. If the test passes, migrate; if it fails, diagnose BEFORE touching production dispatch. Applied in Phase 3.2 (AROW) and 3.3 (Soft-DTW) — both landed with zero regression.
- **Silent-dispatch bugs hide in asymmetries.** `Problem::dtw_function_f32()` was hardwired to Standard DTW regardless of `variant_params.variant` / `missing_strategy` — nobody noticed because every existing test exercised only the f64 path. Found when unifying dispatch via a templated resolver. **Pattern: dual-type APIs (f32/f64) need parity tests, not just one-side tests.** Same failure mode showed up twice in this codebase (also `dtw_runtime()` silently ignoring variant pre-Phase 1).
- **Cell/Cost policy contracts are forward-extensible.** Adding `seed(cost, i, j)` to the Cell policy to support AROW's `C(0,0) = 0 on NaN` semantics did NOT require touching existing cells — `StandardCell::seed` defaulted to `return cost`, which is identical to the pre-refactor `col[0] = cost(0, 0)` assignment. Pattern: new contract methods with sensible defaults are additive, not breaking.

## Audit / Testing

- **`catch (...)` without a rethrow is a false-success generator.**
  `Problem::read_distance_matrix` wrapped its whole body in
  `catch (...) { std::cout << "could not be read"; }`. The CLI's own
  `try/catch` around the call was therefore dead code and it printed
  `"Loaded distance matrix from <path>"` *immediately after* the failure line,
  then silently recomputed the whole O(N^2) matrix (real-binary transcript,
  2026-09-02). A swallowing catch also freezes the behaviour into tests: the
  F22 compatibility gate asserted on the exact failure *stdout*, so the fix had
  to move that assertion to the exception message. Rule: a reader reports
  failure; only the caller decides whether to continue.
- **An `if/else if` dispatch chain over a string needs a terminal `else` that
  throws — the missing branch is not "do nothing", it is "publish a
  default-constructed result as a success".** `dtwc_cl`'s method chain had no
  terminal else, so a value that bypassed CLI11's `CheckedTransformer` (a YAML
  key, whose hand-written normalisation had drifted from the transformer map)
  left `ClusteringResult` default-constructed and the run went on to write a
  binary checkpoint, `labels.csv` and `medoids.csv` for it and exit 0
  (`method: obp`, real binary, exit 0, "Total cost: 0"). Same class: unknown
  `solver` silently kept HiGHS, unknown `linkage` silently became Average.
  Wherever a config file can reach a selector, validate the normalised value
  once up front AND keep the terminal `else { throw }`.
- **A "unique" identifier built from a static's address is a per-image
  constant, not entropy.** `default_series_cache_path()` used
  `reinterpret_cast<uintptr_t>(&counter)`, so two processes of the same binary
  produced byte-identical temp `.dtws` paths — observed directly: two runs both
  printed `dtwc_store_140698956918712_2047.dtws`. Its `counter++` was also
  non-atomic, and 8 threads x 256 calls lost updates on 2 of 3 runs. Use a real
  entropy source (random_device mixed with the clock) for the process tag and
  one relaxed `fetch_add` for the sequence — one atomic per *load call* costs
  nothing, one per series would.
- **`null_count()` is the cheap null check; `IsNull()` per element is not.**
  Both Parquet readers ignored nulls entirely, so a null list cell yielded a
  wrong series and a null element yielded raw buffer bytes straight into the
  DTW distances. One `null_count()` read per chunk (and per values buffer) is
  O(1) metadata and keeps the copy loop branch-free.
- **When one concept has three implementations, they will disagree at the
  boundary.** `Ndata == 0` meant 1 series to `DataLoader::count()`, ALL series
  to the folder loader and 0 to the batch loader, and `Ndata < -1` was accepted
  everywhere. The three sites each open-coded the stopping rule
  (`if (i_data == Ndata) break` vs `Ndata == -1 || n < Ndata` vs
  `Ndata >= 0 && n >= Ndata`). One named predicate plus one validator removes
  the whole class; a comment claiming a body "mirrors X EXACTLY" is not a
  mechanism.
- **An accessor that does not branch on the same discriminator as its `size()`
  is undefined behaviour waiting for a caller.** `Data::size()` branched on
  precision; `series()`/`series_f32()` did not, so `series()` on Float32 data
  indexed an empty vector. One predictable branch in the accessor is the right
  cost — the audit's own perf note demanded a branch, not a lock or a virtual.
- **`fs::directory_iterator` order is a filesystem property, not a contract.**
  Series order set names, labels, medoids and every distance-matrix index, so
  results were not reproducible across machines, and non-regular entries went
  to `readFile()`. Sort once, before the parallel load, filtering to regular
  files — O(n log n) per load, never per series. NTFS happens to enumerate in
  name order, which is exactly why this survived so long on Windows: the
  ordering half of this fix is not falsifiable on the dev machine.
- **A fingerprint that hardcodes one value of a semantic axis silently accepts
  the wrong cache.** `distance_checkpoint_identity()` passed
  `MetricType::L1` unconditionally although the CLI already computed the real
  metric, so a SquaredL2 matrix and an L1 matrix over the same data shared a
  digest. When a fingerprint enumerates axes, every axis the *caller* controls
  must be a parameter, not a literal.
- **Test gates must be one-sided.** A `PASS_REGULAR_EXPRESSION` pinning
  "300-9999 assertions in 9-99 test cases" fails on a legitimate 100th case
  and passes on any count inside the band even if the intended subject was
  deleted. Assert a floor (or that a named tag ran); never an upper bound.

- **A registered test-summary band must reconcile to the collected total before
  the decisive run.** F19 registered at least 1010 passed and 12 skipped over
  1,022 collected while also allowing one known F39 failure:
  `1010 + 12 + 1 = 1023`. The fresh run correctly reported
  `1009 passed + 12 skipped + 1 failed = 1022`. Write the outcome ledger
  explicitly before registering category floors (including failures, errors,
  xfail/xpass, and deselections where applicable); an impossible aggregate is
  FALSIFIED evidence, not permission to relabel a valid outcome or tune the
  floor after the run.
- **Repository-hygiene gates must inspect the Git index, not worktree
  existence.** A staged deletion disappears from `git ls-files -s`, while an
  index-tracked path can be absent only in the worktree; filtering tracked
  paths through `Path.exists()` silently certifies the latter. Read staged blob
  IDs/content with `git cat-file`, make malformed records print FAIL instead of
  throwing before the verdict, and mutation-review the checker itself. Ignore
  files add a second trap: rules are ordered, so line-set membership cannot
  prove that a later rule did not re-ignore an exception. Require a
  filter-aware worktree/index match and pin representative behavior with
  `git check-ignore --no-index`. Secret-shape scans should operate on bytes
  rather than skipping NUL-free non-UTF-8 blobs, and their registered formats
  must include current credential families (for example encrypted PKCS#8 and
  fine-grained GitHub PATs). The R1 checker's first four apparent greens each
  missed one of these classes before adversarial review.
- **Remote-tracking refs are mutable evidence; re-read them at closeout.** R1
  recorded `origin/Claude` 51 commits behind `Claude`, then the ref advanced to
  the local hygiene commit during final review and its reflog said
  `update by push`. Non-sample hooks were absent, but GitHub Desktop had four
  running processes predating the update; a later zero-`git.exe` probe cannot
  exclude an alternate client. The actor therefore remains unknown. Timestamp
  every branch snapshot, preserve the before/after reflog and process evidence,
  and never convert a remote-tracking ref into a server-state claim without an
  authorized fresh read. An unexpected remote update falsifies a global
  no-operation band but is not permission to push, force-reset, or “repair”
  the branch; separate agent compliance from external state, name the operator
  rollback, and continue local work.
- **A low computed-entry count does not prove low memory use.** FastCLARA's old
  test asserted that fewer than 20% of parent distances were computed, while
  the first lazy lookup had already allocated every packed `N*(N+1)/2` slot.
  Memory-contract tests must assert backing size/capacity (and allocation count
  where available), not only populated elements or work counters.
- **A regression test must execute the production arithmetic it claims to protect.** The Metal case in `tests/unit/test_decode_pair.cpp` repeated the intended `pair_offset` types and arithmetic in test-local `constexpr`s, which is tautological: the test stays green if `metal_dtw.mm` is reverted. Exercise the real host-side helper or public dispatch path (or extract a shared production helper) and prove the test fails when the production fix is removed.
- **Wider loop counters cannot extend an `int`-indexed API.** FastPAM's point indices end at `Problem::dist_by_ind(int, int)` and `vector<int>` results, so changing internal induction variables to `int64_t` only added casts and hid the real ceiling. Compare the original `size_t` to `INT_MAX`, narrow once, and keep the kernel internally consistent. Apply that check at every public entry before allocation or mutation; a guard only in a downstream delegate is too late for callers that materialise data first.
- **Always rerun ctest failures serially after the first parallel pass.** A clean handoff on another platform is not evidence of a green local tree. On Windows Release (2026-04-13), `ctest -j 4 -C Release` failed with `0xc0000409`; serial rerun showed `test_fast_pam_adversarial` was a deterministic crash while `unit_test_clustering_algorithms` was a parallel-only failure mode. Audit skills must distinguish "real blocker" from "flake".
- **A test name must match the algorithm path it actually exercises.** `tests/unit/adversarial/test_fast_pam_adversarial.cpp` sounds like FastPAM coverage, but its helper sets `prob.set_method(Method::Kmedoids)` and calls the legacy Lloyd path. That creates false confidence. For algorithm migrations, mislabeled tests are worse than missing tests because they silently certify the wrong implementation.
- **A "structure" test (label ranges, sizes, converged-flag) does NOT test OPTIMALITY — a wrong answer can satisfy it.** The FastPAM k=1 unit test only asserted `labels all == 0` and `converged`, so it passed even when the new decomposition returned the BUILD medoid (the `second_dist = +inf` NaN bug recorded above) instead of the optimum. What caught it was a downstream exact-value oracle. The current named regression starts at `tests/unit/algorithms/unit_test_fast_clara.cpp:712` and pins literal k=1 sample medians. Lesson: for every algorithm, at least one test must pin the OPTIMAL output against an independent oracle (brute force / closed form), not just its shape. The brute-force local-optimality arbiter (no improving swap by full reassignment) added in `unit_test_faster_pam.cpp` is the general form—it is tie-independent and would have caught the NaN at any k.
- **Tests must pin the LIVE code path — name the public entry point in a comment.** Phase 0 task 0.6 "fixed" multivariate L2 in `core::dispatch_mv_metric`, a dead duplicate with zero call sites; the live `detail::dispatch_mv_metric` (warping.hpp) kept aliasing L2→L1, the new test validated the dead function, and the CHANGELOG claim was false. Caught only by adversarial review tracing the real dispatch chain (`dtwBanded_mv` → warping.hpp). Rules: (1) before fixing a dispatcher, grep call sites and delete dead duplicates — two dispatchers for one concept is itself the bug; (2) every regression test states in a comment which public entry point it exercises. (Fixed in Phase 0 remediation R1, commit ffb7a8d.)
- **A finite no-path sentinel makes `isfinite()` a false-green band test.**
  DTWC++ deliberately uses `numeric_limits<T>::max()`, not infinity, inside its
  DTW kernels. Unequal-length tests that asked only for a finite,
  non-negative result therefore passed when the canonical Sakoe–Chiba window
  had no path. The adjacent ADTW “reference” was worse: it copied the
  production endpoint-scaled formula, so both sides agreed against Sakoe and
  Chiba equation (8). Pin the sentinel exactly below `|n-m|`, use a
  full-matrix oracle with independently stated `|i-j| <= band` bounds, and
  include a non-degenerate threshold case whose cost differs from a slanted
  corridor.
- **A compute-type sentinel must be translated at the public type boundary.**
  CUDA and Metal FP32 kernels stamp `numeric_limits<float>::max()` for no path,
  but their public result containers hold doubles. A blind cast therefore
  exposes widened `FLT_MAX`, not the backend-independent public `DBL_MAX`
  contract. Centralize exact sentinel translation in the result-copy boundary
  and test exact equality; `isfinite`, positivity, and approximate comparison
  all accept the wrong value. The same translation applies to threshold-pruned
  GPU pairs: their public result is finite `DBL_MAX`, not `+inf`; documentation
  and bindings must not promise an IEEE infinity that the API never returns.
- **A maximum finite value cannot double as “no best result yet.”** A valid
  assignment distance or objective may be exact finite `DBL_MAX`; initializing
  `best = DBL_MAX` and later testing `best == DBL_MAX` therefore collides with
  real data. Keep a separate boolean presence flag, preserve strict `<` for
  first-slot ties, and validate every distance before comparison.
- **Exact diagnostics require whole-message mutation tests.** A gate that
  searches for the expected text as a substring accepts an appended suffix and
  does not freeze the public error contract. Compare the complete normalized
  stderr line, then prove that changing only a suffix makes the real-binary
  gate fail.
- **Tracked-manifest inventory constants move with every new tracked
  manifest.** F13 added one permanent `.cmake` integration gate, so the
  index-owned count changed from 25 to 26 even though no supply-chain identity
  changed. Compare `git ls-files` sets against the registered base and update
  the production checker plus its direct test in the same dedicated finding;
  a filesystem walk cannot judge a Git-index contract.
- **A repaired oracle must retain a discriminator, not only new expected
  numbers.** Portable RNG changed Lloyd's seeded trajectory, and replacing the
  old literals on the original iteration-cap fixture made capped and converged
  runs identical. Build a new independently calculated fixture whose two
  states differ, then mutate the forwarded cap to the default and require the
  public binding test to fail.
- **A documentation marker gate can pass text that the renderer breaks and can
  preserve the wrong backend scope.** D1's first documentation gate was green
  while its standalone derivation used GitHub-unsupported `\(...\)`/`\[...\]`
  math delimiters, raw absolute-value pipes split two table rows, and the site
  presented CPU-only window/sentinel behavior as backend-wide. Check
  renderer-supported delimiters and table structure, require backend
  qualifiers and finding ownership as explicit drift markers, and separately
  review source attribution at equation granularity.
- **Decode binary floating-point evidence; do not copy a rounded summary into
  an exact ledger.** F8's first preregistration copied decimal spellings from a
  handoff even though the SHA-pinned checkpoint was authoritative. Before the
  decisive run, decoding its documented `total_cost` bytes at offset 24 showed
  `4.3999999999999986` and `4.4000012278556824` at 17 digits. The CLI prints
  only six significant digits, while a language's shortest-roundtrip `repr`
  may choose another correct spelling. Register the raw IEEE-754 bytes plus a
  max-digits decimal rendering whenever last-bit identity is load-bearing.
- **A CMake supply-chain scan must follow active argument grammar, not search
  for nearby words.** F11's first scanner revisions false-greened bracket
  comments, inline/semicolon-expanded and variable-expanded arguments,
  shorthand/API branch archives, `cmake_language(CALL|DEFER|EVAL ...)`, URL
  mirrors, and `URL_HASH` text captured by CPM's `OPTIONS` multi-value
  argument. The final audit found three more bypasses: an expansion beside a
  safe literal URL, a DEFER `ID`/`ID_VAR` operand named `call`, and an exact URL
  hidden in a differently named decoy package. Tokenize comments, quotes,
  bracket arguments, command indirection, DEFER option operands, package
  identity, and CPM's keyword boundaries; reject syntax that cannot be
  classified. A lower-bound count and one expected package name still permit a
  pinned decoy beside a drifted real call. Register the exact multiset of path,
  package name, URL, and digest for every tracked archive, so additions,
  removals, duplicates, renames, and substitutions all require an explicit
  gate update. A hash is active only in the `URL` multi-value segment before
  the next CPM keyword. Even an exact identity is decorative when
  `DOWNLOAD_COMMAND`, `SOURCE_DIR`, another repository method, or an in-call
  find-package route overrides acquisition; a URL declaration must reject
  every alternate source selector before comparing its identity. After two
  registered pivots, the hand-written parser was still bypassed by
  `CUSTOM_CACHE_KEY` under `CPM_SOURCE_CACHE` and by CMake's quoted
  backslash-newline normalization (`"DOWNLOAD_\<newline>COMMAND"`). Stop
  extending lexical deny-lists at that point: record the falsification and
  replace the design with an official-parser or canonical-manifest gate.
- **A wrapper timeout is not a successful subprocess exit, even when CMake
  printed `Generating done`.** F11's first fresh configure wrote a complete
  Ninja tree at 61.0 s, but the 60 s command wrapper returned 124 before it
  captured CMake's exit. The run remained FALSIFIED. Preserve the entire build
  directory under a named attempt (never delete it), verify no child process
  remains, and use the one allowed fresh retry with a wrapper longer than the
  registered workload; generated files and stamps can localise the result but
  cannot manufacture the missing exit code.
- **`CMAKE_MINIMUM_REQUIRED_VERSION` is not a stable root-project floor after
  dependencies configure.** F16 attempt 1 read it from `tests/CMakeLists.txt`
  after CPM dependencies had run and observed `3.14`, even though the root's
  first command is `cmake_minimum_required(VERSION 3.26)`. Dependency
  listfiles with 3.14 minima were present in the configured tree. A late
  metadata guard must bind the root's own first command (or a root-owned value
  captured before dependencies), not CMake's mutable most-recent minimum
  variable. The failed guard printed
  `preset=3.26.0, root=3.14, expected=3.26.0`; no test compiled or ran.
- **A timed-out Windows loader probe can leave the executable locking build
  outputs.** After CTest times out before `main`, inspect only the exact
  `ctest`/test child PIDs and terminate those confirmed descendants before a
  rebuild. Never delete the build tree or kill by a broad process-name guess;
  record the timeout output and verify the named processes are gone.
- **A gate can reject its own success marker, and randomized test order makes
  adjacency regexes flaky.** F14 initially used a broad `[Ss][Kk][Ii][Pp]`
  failure regex while requiring `skips=0`, so an otherwise-green gate was
  guaranteed to fail. It also required a marker case to appear immediately
  before Catch2's summary even though case order is randomized. Bound the
  failure token so it cannot match the success vocabulary, and permit
  intervening output while requiring both the execution marker and the
  framework's assertion/case floor. For recursive-cleanup gates, resolve the
  trusted parent and append the exact child; resolving both equal input
  strings through the same junction is a tautology, not an escape check.
- **A successful buffered insertion is not a successful file write.** Checking
  `good()` before an `ofstream` destructor runs can miss flush/close failures,
  and a destructor cannot report them through the calling API. For
  user-visible output contracts, finish all writes, close explicitly, then
  check stream state and return or throw the typed I/O failure. F14 applied
  this at all three matrix-file openers after the production audit.
- **`std::mt19937` is bit-exact by the standard; `uniform_real_distribution` /
  `generate_canonical` are not. [confirmed 2026-09-02]** The same seed gave four
  byte-different fixtures (MSVC, relaxed Clang, libstdc++, libc++), each needing
  a hand-registered profile row. `tests/support/deterministic_series.hpp` now
  converts integers itself (`genrand_res53`, `k * 0x1p-52` / `k * 0x1.4p-49`, one
  correctly rounded op, nothing to reassociate or fuse) and F15 registers ONE
  fingerprint per schedule; verified byte-identical across clang/MSVC-STL,
  g++/libstdc++ and `-O0 -ffp-contract=off`. Never hash distribution output, and
  never write `lo + (hi-lo)*u` in a fixture: `/fp:contract` or
  `-fassociative-math` turns it into an FMA and the bits move.
- **Catch2 decomposition rejects unparenthesized logical OR.** An expression
  such as `CHECK((a && b) || (c && d))` reaches Catch2's deleted/decomposition
  guard and fails to compile; force the complete predicate to `bool` with one
  more pair of parentheses: `CHECK(((a && b) || (c && d)))`. F15 attempt 1
  failed at compile time on this exact distinction.
- **CMake `string(JSON)` is not a whole-document JSON syntax arbiter.** F16's
  M10 appended trailing non-whitespace after a complete preset object.
  `cmake --list-presets=all` rejected it with `Extra non-whitespace after JSON
  value`, but the configure-time `string(JSON)` queries all succeeded and
  configuration exited 0. Use CMake's preset reader or another parser that
  proves complete-input consumption; successful field lookup proves only a
  valid leading JSON value. Do not add another lexical sentinel after a capped
  parser falsification.
- **Cross-shell diagnostic classifiers are separate subjects from the command
  they wrap.** F16's no-LLVM CMake subjects exited 1 and printed the required
  missing-`clang++` diagnostic, while two PowerShell `$output` predicates
  returned `diagnostic=False` because native stderr arrived as error records.
  A WSL wrapper also failed before CMake because an unquoted grep expression
  containing parentheses lost its intended quoting through `wsl.exe`. Capture
  native stdout/stderr through a single known shell, split execution from
  inspection, and report wrapper failure independently; never let a broken
  classifier overwrite directly observed subject evidence.
- **CTest exits 0 when `-R` matches no tests.** F16's first post-mutation
  selector omitted the tracked subject's `test_` prefix and printed `No tests
  were found!!!` with a successful process exit. Run `ctest -N` to resolve the
  exact registered name, then require the test's own execution marker and
  assertion/case floor; an exit code without a nonzero executed-subject count
  is not a gate.
- **A deserializer's success message proves no downstream state consumption.**
  F17's inherited CLI printed `Loaded checkpoint: 41 iterations, cost=1650`,
  then ran FastPAM and replaced every loaded result field with cost 976 /
  iteration 1. A resume gate needs a deliberately distinguishable complete
  state, must assert every field at the final public artifacts, and must prove
  the algorithm path did not run.
- **CTest skip regexes must not classify semantic success counters.** F17
  attempt 1 satisfied every child assertion and printed
  `algorithm_skipped=1/1`, then CTest's broad skip regex matched that marker and
  failed the decisive test. Apply skip detection to each child output and make
  any outer metadata expression line-oriented to actual diagnostics; a raw
  substring is incompatible with words such as `skipped` in success evidence.
- **A resume rehearsal must select the producer's actual state artifact.**
  F17's inherited SLURM job changed output directories for run 2, so binary
  `--resume` could not find run 1's automatic checkpoint; equal deterministic
  labels merely proved a fresh rerun. Copy or name the exact source state,
  require a replay marker plus absence of the algorithm marker, compare every
  restored field, and prove source bytes and timestamp unchanged.
- **A new tracked `.cmake` test changes the supply-chain manifest subject.**
  F17 added a legitimate real-CLI CMake driver but preregistered the inherited
  exact inventory of 27. The post-commit checker observed 28 and correctly
  failed despite all URL identities remaining pinned. Register both behavioral
  and inventory effects before implementation; never evade an index-owned scan
  by hiding executable CMake behind another extension.
- **A global device setter is not evidence that a newly constructed compute
  object uses that device.** F18's MATLAB estimator successfully changed Env to
  `gpu`, then constructed a default Auto `Problem`; Auto resolved only to CPU
  strategies, so the profiled fit published the L1 cost 9 and launched no CUDA
  kernel. Pin the final object's backend configuration or independently profile
  the public operation. Capability validation and execution reachability are
  separate subjects.
- **A profiler may exit 0 after observing no kernels.** Nsight Compute
  `--set none` is usable without performance-counter permission and names real
  DTWC CUDA launches, but its no-work control also exited 0 and printed
  `No kernels were profiled.` A GPU reachability gate must require the expected
  kernel/device/invocation row and reject the no-kernel diagnostic; the profiler
  process status alone false-greens.
- **Poison external-command seams before testing a rejection boundary.** F18
  found a repository-root `.env` (contents deliberately unread), so an
  inherited or mutated estimator call to `dtwc.device('hpc')` can reach the real
  `ssh` probe before returning the same broad `dtwc:deviceError`. An ID-only
  assertion is unsafe and can false-green. Use a fresh process whose
  `DTWC_REPO_ROOT` names a repo-local fake `.env`, whose first `PATH` resolution
  is a recording `ssh.cmd`, then pin the estimator-specific message, Env state,
  and exact zero/one shim-call counts. Never execute the unsafe mutant without
  the poison shim.
- **A symmetric distance matrix cannot validate row-major/column-major copy
  orientation.** Its transpose is identical, so even an exact public clustering
  oracle cannot distinguish `matrix[i*N+j]` from the wrong transposed ownership
  assumption. Pin the boundary expression by source/mutation review or use a
  deliberately nonsymmetric synthetic copy seam; do not claim the symmetric
  numeric fixture proves layout.
- **Recount live MATLAB suites; historical pass totals are not a floor.** F18's
  fresh five-suite runs on R2024b and R2025b collected 82 cases, not the
  inherited working rule's 61. They produced 81 passes, zero failures, and one
  intentional opposite-flavor assumption filter. Register collected, passed,
  failed, and incomplete separately and name the sole allowed filter.
- **CUDA `Auto` precision is a separate executable path, not shorthand for
  explicit FP32/FP64.** F18's four-by-two MATLAB fixture crashed the CUDA MEX
  with `0xc0000005` under `CUDAPrecision::Auto`, before a kernel row appeared,
  while the same pre-existing `Problem` route forced to FP32 and FP64 returned
  the exact distance 10. The existing `dtwc.test.gpu()` oracle also stayed
  green because it forces FP64. Source comparison isolates Auto's additional
  `query_gpu_config()` call and static mutex/cache, but a debugger seam is still
  required before naming the precise statement. Test Auto, FP32, and FP64 as
  three distinct routes in the owning DLL/MEX process; an explicit-precision
  oracle cannot certify the Auto selector.

- **PowerShell array syntax inside an `if` branch does not preserve singleton
  identity at the outer assignment. [confirmed]** F19's all-route MATLAB
  profile returned four strings and worked, but `@($Route)` in a single-route
  branch flowed through the `if` output pipeline as scalar `System.String`;
  StrictMode then rejected `$routeNames.Count`. Coerce the destination to
  `[string[]]`, return it through a non-enumerating boundary, and
  mutation-test both type and count for every singleton. A parser-only check
  cannot catch this. Evidence:
  `scripts/test_f19_matlab_route_selector.ps1`.
- **A semantic patch is not a byte-preserving mutation operator for mixed-EOL
  evidence. [confirmed]** F19's MATLAB gateway is `i/lf w/mixed`. Removing the
  intended line through a text hunk normalized surrounding newline bytes, so
  the registered profile hash rejected it. Decode the captured snapshot as
  strict UTF-8 without BOM, mutate while retaining its CR/LF characters, write
  exact bytes, and rehash every materialization and restore. When the decoder
  returns structured `{ Bytes, Text }` evidence, pass `.Text` to the mutation
  helper rather than string-coercing the record.
- **A source token is evidence only when it is active code. [confirmed]**
  F19's first privacy/source audit could be satisfied by declarations and
  writebacks under `#if 0`, `#if (0)`, `#if 0u`, or `#if false`. Scrub comments
  and literals, track preprocessor nesting, and count only active depth-zero
  tokens. The 27-probe F19 gate includes all four inactive-code mutants.
- **A runtime artifact label is not artifact identity. [confirmed]** For
  F19's six MEX profiles, requested release/profile names could still select a
  stale binary. Bind observed `version('-release')`, exact
  `which(...,'-all')` path, source SHA-256, MEX SHA-256, and before/after
  invocation hashes, then rehash the complete evidence set after the schedule.
  The decisive F19 run records 42/42 post-run rehashes.
- **Default-moving a `std::function` does not rebind a lambda that captured
  `this`. [confirmed]** F20's mapped data and cached distances survived a
  Problem move, while the derived DTW closures still called the moved-from
  object. Force uncached const and mutable work after move construction and
  move assignment, poison/reuse the source storage, and verify the closure's
  relocation state. A cached result can hide a live use-after-move path.
- **Self-referential owner bundles need pointer-identity and closed-handle
  lifetime gates. [confirmed]** Equal values did not prove that spans and
  `string_view`s targeted the relocated mmap/name owner; short SSO names could
  even keep stale bytes looking valid. Assert owner/view pointer, count, and
  name-view identity before and after both moves. Parse a Windows mapped
  artifact only after the handle-owning scope has ended, because an open
  mapping can make the file unreadable without proving bad bytes.
- **A MEX uses the host's already-loaded private MSVC runtime. [confirmed]**
  The same optimized F20 MEX used VS 14.50 constexpr mutex bytes, crashed in
  R2024b's private MSVCP140 14.36 `_Mtx_lock`, and passed with R2025b's private
  14.40 SRW representation. Record compiler headers, loaded runtime version and
  hash, optimization, and release for every compatibility claim.
  `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` is only an inferred remedy until an
  optimized two-release differential runs.
- **Nearest-export stack labels and a Debug pass do not localize an optimized
  MEX fault. [confirmed]** MATLAB labeled the F20 frame
  `Thrd_yield+184`, but RVA/import/disassembly showed a null dereference inside
  `_Mtx_lock` called by LLFIO's first initialization mutex. Verify the
  `mexFunction` export, reproduce the decisive optimization, and use PDB line
  data plus the import address before naming the failing statement.
- **Exception translation must include path discovery, not only the final I/O
  call. [confirmed by review]** F20's default series-cache route calls
  `std::filesystem::temp_directory_path()` before the `try` that translates
  mmap creation failures. A bad temp environment can therefore escape as
  `filesystem_error` instead of the public `IOError`; poison the discovery
  step and require transaction preservation before claiming taxonomy closure.
- **A public rename is a symbol-table migration, not a textual alias.
  [confirmed]** F21 could not add `DataLoader::start_row(int)` because a private
  `int start_row` already occupied the member name. The inherited compile probe
  exposed both the access violation and “called object type `int`” error before
  product work. Rename coupled private storage coherently, pin exact overloads
  with function-pointer casts (a string literal can hide a missing
  `const char*` overload through conversion), and mutation-test both canonical
  ownership and legacy forwarding. Evidence:
  `.claude/baselines/2026-07-29-f21-cpp-renames.md`.
- **A third-party public header can silently change every caller diagnostic.
  [confirmed]** The Windows-Clang quickcpplib header pulled through LLFIO uses
  a bare `#pragma clang diagnostic ignored "-Wdeprecated-declarations"` with
  no push/pop. Consequently `#include <dtwc.hpp>` made all 24 already-retained
  C++ aliases silent in canonical LLFIO-ON while identical llfio-OFF diagnosed
  24. Bracket optional-dependency includes at the DTWC++ boundary and test a
  deprecated sentinel after each public include under `-Werror`; OFF does not
  prove ON. Evidence:
  `.claude/baselines/2026-07-29-f45-llfio-diagnostic-state.md`.
- **Different CTest build matrices are not process-isolated when tests use the
  source root as their working directory. [confirmed]** Running canonical and
  llfio-OFF matrices concurrently made both `unit_test_fileOperations`
  processes create/remove the same relative `CSV` directory; one matrix
  reported locked-file failures and an unrelated adversarial executable exited
  `0xc0000409`. Both named tests then passed 2/2 in each unchanged build, and
  both complete matrices passed serially. Run configured matrices one at a
  time unless every artifact path is build-local. Evidence:
  `.claude/baselines/2026-07-29-f45-llfio-diagnostic-state.md`.
- **`mxGetScalar` does not enforce the scalar shape its name suggests.
  [confirmed]** F22 passed vector configuration candidates through a helper
  documented as extracting a scalar; native state consumed element one while
  MATLAB cached the complete vector. Check `mxGetNumberOfElements(...) == 1`
  at the shared numeric/logical boundary and mutate native state before the
  wrapper cache. This affects `get_scalar` callers, not uint64 handles, which
  use a different decoder. Evidence:
  `tests/matlab/test_contract_parity.m::f22_verify_config_setter_atomicity`.
- **A MATLAB `verify*` failure does not stop a hand-written verdict marker.
  [confirmed]** F22's first atomicity failure recorded the test as failed but
  continued to print `verdict=PASS` because the marker's counters omitted the
  nonfatal checks. Every load-bearing oracle that is not represented in the
  printed verdict must use fatal `assert*` checks before that marker (or return
  an explicit boolean into it). Evidence:
  `tests/matlab/test_contract_parity.m::test_f22_matlab_deprecation_policy`.
- **A fatal MATLAB unit-test assertion sets both `Failed` and `Incomplete`.
  [confirmed]** Treating every killed mutant as `failed=1 incomplete=0` rejects
  the strongest failures as harness errors; treating either value as
  acceptable can hide an unexpected abort. Register the exact result shape per
  mutant before execution. F22's dual-release campaign observed
  `passed=0 failed=1 incomplete=1` for its seventeen fatal-assertion mutants
  and `passed=0 failed=1 incomplete=0` for its sixteen nonfatal-verdict
  mutants. Evidence:
  `scripts/test_f22_matlab_deprecation_mutations.py` and
  `.claude/baselines/2026-07-29-f22-deprecation-policy.md`.
- **A dated documentation re-audit is false if its source anchors and checker
  inventory were not re-audited too. [confirmed]** F22's first documentation
  pass updated the policy prose but retained moved line numbers and checked
  only a handful of literal legacy calls, so variable renames and most aliases
  could bypass it. Re-resolve every local anchor against live source, table the
  complete unique-name/operation counts, execute positive controls for every
  pattern family, and pin explicit exclusions for canonical/private lookalikes.
  Evidence: `scripts/check_docs_contract.py` at `43e1c44` and
  `.claude/baselines/2026-07-29-f22-deprecation-policy.md`.
- **CTest's `LastTestsFailed.log` can remain stale after a green rerun.
  [confirmed]** The clean-first F22 canonical matrix passed 122/122 and its
  focused subject then passed 229 assertions/5 cases, while
  `build/highs-1151/Testing/Temporary/LastTestsFailed.log` still contained
  `61:test_problem_api_2_0` from an older run. Use the current command's exit
  and complete transcript for adjudication; the mutable helper file is useful
  only when timestamped and reconciled to that same run. Evidence:
  `.claude/baselines/2026-07-30-f22-final-gates.md`.
- **A clean-first scoped target deletes sibling executables required by a
  later cross-surface suite. [confirmed]** Clean-first building only
  `_dtwcpp_core` produced and installed a fresh extension but removed
  `build/cfg-gate-normal/bin/dtwc_cl.exe`. The subsequent Python suite selected
  an Arrow-build CLI for five HPC cases and could not find the intended CLI for
  its version case, yielding six setup-caused failures plus the expected F39
  red. After any scoped clean-first binding build, enumerate and rebuild every
  executable consumed by the downstream gate and run it once outside the
  harness before simulation. `test_version_ssot` honors `DTWC_CL_PATH`, but
  `test_hpc` does not: `_hpc.find_dtwc_binary` chooses the newest build-tree
  candidate by mtime, so assert that helper's selected path explicitly.
  Evidence:
  `.claude/baselines/2026-07-30-f22-final-gates.md`.
- **PowerShell can strip embedded MATLAB double-quote delimiters passed through
  `matlab -batch`. [confirmed]** An 85-test R2024b run reached its exact
  expected failed/incomplete ledger, but the post-run oracle's MATLAB string
  array became bare identifiers and exited before printing the registered
  summary. Use single-quoted MATLAB character arrays/cell arrays in an inline
  PowerShell batch argument, and emit newlines with a separate
  `fprintf('%s',newline)` to avoid a second quoting layer. Evidence:
  `.claude/baselines/2026-07-30-f22-final-gates.md`.
- **Writing native scalar objects is not an endian implementation.
  [confirmed]** Binary-v1 documented little endian but wrote and read native
  `int32_t`, `uint16_t`, and `double` objects; the local x86-64 fixture happened
  to agree while malformed signed counts allocated or threw before payload
  length was known. Encode unsigned bit patterns byte-by-byte, assert the exact
  binary64 representation, validate all canonical header bytes, and compare
  the same open stream's exact size before any count-derived allocation.
  Evidence:
  `.claude/baselines/2026-07-30-f51-binary-checkpoint-wire.md`.
- **A round-trip fixture needs both sides of every binary state.
  [confirmed]** F51's first strengthened fixture used only `converged=true` and
  nonnegative payload integers, so an always-true boolean codec or broken
  signed payload could satisfy the exact marker. Preserve the corruption
  corpus, but make accepted compatibility fixtures execute canonical zero and
  negative payload values too. Evidence:
  `tests/unit/unit_test_checkpoint_binary.cpp`.
- **Whole-file formatting can bury a small persistence fix in unrelated churn.
  [confirmed]** Running `clang-format -i` over the legacy checkpoint
  translation unit changed 1,103 diff lines around a roughly 300-line codec
  replacement. Reconstruct untouched regions from the committed source and
  format only changed line ranges before review. Evidence: F51 implementation
  preflight recorded in
  `.claude/summaries/handoff-2026-07-30-f51-binary-checkpoint-wire.md`.
- **A timed-out build wrapper can leave Ninja's log non-authoritative.
  [confirmed]** F51's first full canonical rebuild was killed by a 120-second
  wrapper ceiling; the recovery build warned `premature end of file` and a
  premature repeat still rebuilt header dependents. Inspect live process
  command lines before killing anything, run `ninja -t recompact` on the exact
  build directory, complete the rebuild, and require a separate
  `ninja: no work to do.` settling run before CTest. Evidence:
  `.claude/baselines/2026-07-30-f51-binary-checkpoint-wire.md`.
- **Pytest does not create a missing parent of `--basetemp`. [confirmed]**
  F23's first decisive Python command used
  `--basetemp=build/f23-green/focused` while `build/f23-green` did not exist.
  Pytest created neither ancestor and errored in fixture setup before the
  wire-format subject ran; two fixture-free export tests still passed, making
  the partial output especially easy to misread. Before a decisive run, create
  and verify the immediate parent of every nested `--basetemp`, or use a direct
  child of an existing build root. Evidence:
  `.claude/baselines/2026-07-30-f23-python-binary-checkpoint.md`.
- **`PathLike[str]` plus a UTF-8 exception message is not automatically an
  all-filesystem-path contract. [confirmed]** Converting a native path with
  `path.u8string()` and later passing `what()` through `PyErr_SetString`
  handles valid Unicode, but POSIX surrogateescape filenames can produce
  invalid UTF-8 and Windows lone surrogates can fail conversion before the
  native I/O call. Test raw undecodable filename bytes on POSIX and
  unpaired-surrogate inputs where the platform permits them; build typed errors
  from a representation Python can always decode. Evidence: F56 in `PLAN.md`
  and the F23 binding audit.
- **A fail-closed source checker must pin verdicts, not merely workloads.
  [confirmed]** D3 initially pinned exhaustive inventory counts and output
  markers while omitting eight `violations == 0` assertions and most of F57's
  parity/admissibility assertions; the test could still clear its assertion
  floor after those verdicts were removed. Raw marker scans also accepted
  commented-out C++, and independent CTest marker/floor substrings did not
  prove one composite pass regex. Strip comments and literals before code
  checks, pin every load-bearing zero/result assertion, and require the exact
  `PASS_REGULAR_EXPRESSION` property that couples marker to floor. Six
  in-memory mutants now reject. Evidence:
  `.claude/baselines/2026-07-30-d3-lb-enhanced-webb.md`.
- **A last-bit pin is not portable across GCC vs Apple Clang under the Release
  reassociation set.** An HPC (GCC 14) run on 2026-09-01 (PR #32) failed three
  gates that were green on Apple Clang: F13 pinned `nearest[]` bits
  `0x8000000000000000` (`-0.0`) while GCC flushed them to `+0.0` under
  `-fno-signed-zeros`; F22's canonical probe treated GCC's constructor-NSDMI
  use of deprecated `maxIter`/`N_repetition` as a canonical diagnostic; F8
  Soft-DTW `total_cost` differed by 2 ULP (`CC..` vs `CE..`) after
  resident==stream and labels/medoids already matched. Canonicalize signed
  zeros in the F13 oracle the same way as the zero objective; pragma-silence
  the *constructors* (not the fields) in `Problem.hpp`; register a second
  Soft-DTW cost encoding and keep resident==stream byte-identity. Do not skip
  the tests, loosen Standard DTW goldens, or remove `[[deprecated]]`.
- **`std::filesystem::path`'s stream operator quotes and escapes, so CLI path
  markers are not raw paths.** `operator<<` routes through `std::quoted`; on
  Windows the separator `\` is emitted as `\\`, which a naive CMake
  `string(REPLACE "\\" "/")` turns into `//`. Pinning that doubled form
  passes on Windows and silently mis-asserts on POSIX (and PR #32's
  single-slash pin did the reverse). Collapse the escaped pair first, then the
  single separator, and compare against a single-separator path. Observed
  2026-09-02 in `tests/integration/test_distance_matrix_csv_contract.cmake`.

## LR-core Solver (Phase 4)

- **Killed 2023 ideas — keep them killed.** Every 2023 attempt (removed in `f7064b3`) solved the p-median LP in x-space explicitly: dense/sparse tableau simplex, Gomory cuts, OSQP/ADMM on the N²-variable relaxation. All failed on scale. The right structure is to DUALIZE the assignment equalities and bound matrix-free: the Cardinality+Linking substructure is TU for all N (Ghouila-Houri), so the Lagrangian dual equals the LP bound (Geoffrion) WITHOUT forming the N²-column LP. Do not re-open x-space LP solving; if tempted, re-read UNIMODULAR.md §8.
- **Falsified polytope claims (registered bands, scipy/HiGHS vertex LPs + brute-force IP oracle).** p-median constraint matrix is TU only for N≤2 (6×6 3-cycle has det −2). Half-integrality of fractional vertices is FALSE (values 1/4, 1/3, 3/4 occur). "LP is 80–90% integral" is NOT a polytope property — it is data-regime-dependent: clustered non-metric surrogates were 250/250 integral, uniform N=20,k=4 was 26% integral, and the separate uniform N=10,k=3 gate had maximum gap **13.718%**. The user's "almost unimodular" observation = his data lives in the integral regime, not a theorem.
- **Two different gaps need two different tools — never conflate.** The LR primal repair recovered the brute-force optimum on the registered 40/40 clustered surrogate instances; that is not an every-N theorem. (a) The DUAL/LP gap is continuous: λ=2.0 oscillated on about 25% of separated fixtures, while the damped λ=1.0 + CFM route was the retained subgradient. On the recorded N=800 fixture, subgradient remained at gap 5.8e-5 after 4000 iterations; stabilized Kelley reached 6.6e-14 in 15 major iterations. (b) The INTEGRALITY gap is discrete—no dual method closes it; only branching does. `lagrangian_root_exact` uses Kelley for (a) and y-branching B&B for (b). Its exact gate had zero root nodes on eight clustered surrogate fixtures and engaged the tree on 24 adversarial uniform fixtures; neither result licenses a blanket real-world claim.
- **Prune-if-bound-exceeds-incumbent tests must clear the incumbent by a tolerance, never `>` exactly.** Reduced-cost fixing on a CERTIFIED instance (gap≈0, LB≈UB) wrongly eliminated an optimal medoid whose score merely TIED ρ_(k): independently-summed LB/UB/ρ round to a few-ULP difference that crosses an exact `>`. A legit alternative optimum got fixed out. Fix: only fix when the conditional bound clears UB by `tol=1e-9·(1+max(|LB|,|UB|))` — safe (real eliminations have O(scale) margin ≫ tol), caught only because the N≤14 correctness test checks every fixed facility against the brute-force optimum. General rule: any "kill if bound > best" comparison near a zero gap needs a magnitude-scaled slack.
- **PDLP (HiGHS first-order LP) is an ARBITER, not a re-opening of the killed x-space LP (Task 4.5).** The 2023 kill was hand-rolled x-space solvers (custom OSQP/ADMM/OSLP tableau) that failed on scale. HiGHS PDLP is different on two axes: (1) it's a maintained library, not custom code (honours the prefer-libraries rule); (2) it's used strictly to CROSS-CHECK the Lagrangian bound, never as the production bound engine — the matrix-free Lagrangian still dominates on the TU-structured p-median. The arbiter has real teeth: PDLP (explicit LP, first-order primal-dual) and Kelley (Lagrangian dual, matrix-free) reach the same LP optimum by different mathematics — measured agreement **7.14e-09** across 24 instances (registered band 1e-4), validating the clever bound where the brute-force IP oracle (N≤14) can't reach. Do NOT promote PDLP to a production `Method`: it's LP-only (no integer certificate) and dominated by LR-core for the bound.
- **HiGHS PDLP GPU is a build flag, not a runtime switch.** v1.15.1 vendors real cuPDLP CUDA kernels (`pdlp/cupdlp/cuda/*.cu`, `pdlp/hipdlp/pdhg.cu`) but gates them behind the HiGHS CMake option `CUPDLP_GPU` (default OFF) — a stock build gives CPU PDLP with the identical bound. To warn honestly on `use_gpu=true` without a runtime GPU query, drive the warning off OUR OWN compile flag (`DTWC_HIGHS_GPU`, set only when we forward `CUPDLP_GPU=ON`): requested-GPU-on-CPU-build → stderr warning + `gpu_used=false`, never a silent GPU claim. GPU verified live on the RTX 4000 Ada — HiGHS forces itself SHARED on Windows for CUPDLP_GPU (`highs.dll` + `cudalin.dll` land beside the exe; add the CUDA `bin` to PATH at run time for cudart/cublas/cusparse).
- **A PUBLIC compile-def on an OBJECT lib does NOT reach test TUs — use a runtime capability query.** `target_compile_definitions(mip-solvers PUBLIC DTWC_HIGHS_GPU)` reaches code compiled INTO mip-solvers (so `pdlp_lp.cpp` saw it, set `gpu_used=true`) but NOT `test_pdlp_lp.cpp` (the define stops at the dtwc++ link boundary) — an `#ifdef DTWC_HIGHS_GPU` in the test compiled the wrong branch and failed while the feature worked. This is exactly why the mip tests use runtime try/catch, not `#ifdef DTWC_ENABLE_HIGHS`. Fix: the library exposes `pdlp_gpu_available()` (compiled where the define lives) and the test branches on that. General rule: a test cannot see a dependency's private/object-scoped defines; expose capability at runtime.
- **HiGHS `CUPDLP_GPU` is a COMPILE-TIME device switch — a per-call `use_gpu` flag cannot toggle it, and reporting off it lies.** The bench first ran two columns (`use_gpu=false` vs `true`) expecting a CPU-vs-GPU comparison inside one build. On the GPU build both columns were **digit-identical in iteration count** (880/880 … 6840/6840) and time: `solver="pdlp"` always runs on the GPU once HiGHS is built with `CUPDLP_GPU=ON`, there is no per-solve CPU path. The old code set `gpu_used=true` only when the caller requested the GPU, so a `use_gpu=false` solve on a GPU build ran on the GPU yet reported `gpu_used=false` — a false report (CLAUDE.md §1). Fix: `gpu_used = pdlp_gpu_available() && variant=="pdlp"` (build + variant, not the request); `use_gpu` only drives the CPU-build warning. Cross-device comparison must therefore be **cross-build** (same bench on a CPU-only and a `DTWC_HIGHS_GPU` build), not two calls in one process. Diagnostic tell that two "different" configs are secretly identical: iteration counts match to the digit.
- **Bench verdict — PDLP is a cross-validation ARBITER, not the measured production winner.** On the registered clustered-surrogate sweep (`.claude/baselines/2026-07-08-pdlp-bench.md`), matrix-free Kelley beat CPU- and GPU-PDLP at every measured N. At N=400, `pdlp/kelley` was **944.9×** for CPU-PDLP and **126.3×** for GPU-PDLP; GPU-PDLP was only 30.0× and 84.5× slower than Kelley at N=20 and N=100, so "never within two orders" was false. GPU-PDLP was slower than CPU-PDLP at N=100 and faster at N=200; the exact crossover inside that bracket was not measured. By N=400 it was 7.4× faster than CPU-PDLP. This partially falsified the registered prediction that GPU-PDLP would be no faster throughout N≤400. The result supports retaining PDLP as an independent explicit-LP arbiter for the matrix-free Lagrangian bound; it does not make the full p-median constraint matrix TU or establish an all-device/all-instance theorem.
- **Two Windows-CUDA toolchain traps (both cost real time; both have one-line fixes).** (1) Git Bash / MSYS mangles a leading `/c` argument into `C:\`, so `cmd.exe /c "batch"` silently opens an INTERACTIVE cmd (banner + prompt) and exits doing nothing — no error, no output. Use `MSYS_NO_PATHCONV=1 cmd.exe /c …` or `cmd.exe //c …`. (2) CUDA 13.0's `nvcc` host_config REJECTS MSVC newer than VS 2022 ("unsupported Microsoft Visual Studio version! Only 2019–2022"), e.g. the VS-18 / MSVC 14.50 on this box — pass `-DCMAKE_CUDA_FLAGS=-allow-unsupported-compiler` (and the explicit `-DCMAKE_CUDA_COMPILER=<nvcc>` since vcvars does not put nvcc on PATH). Both are captured in the known-good cache at `build/cuda-verify/CMakeCache.txt` — read it before fighting a fresh CUDA build.

## Multivariate DTW (Phase 5)

- **Independent MV mode is orthogonal to the variant axis — a mode flag, not a new `DTWVariant`.** DTW_I (per-channel univariate DTW summed) vs DTW_D (one shared warping path) is a combination choice, so it belongs in `MVMode{Dependent,Independent}` on `DTWVariantParams`, not another enum value that would combinatorially multiply with every existing variant. Default Dependent keeps the pre-existing `dtwFull_L_mv`/`dtwBanded_mv` path bit-identical (zero regression); Independent intercepts at the TOP of `resolve_dtw_fn`, before the missing-strategy and variant switches, because it is a per-channel decomposition that reuses the univariate kernel.
- **`DTW_I ≤ DTW_D` is a free, independent-math arbiter — use it.** With an additive per-channel local cost (L1 or SquaredL2) and the same band, for the dependent-optimal shared path P: `DTW_D = Σ_c cost_c(P) ≥ Σ_c min_{P_c} cost_c = DTW_I` (each channel picks its own path). Provable and cheap to assert on random data — it would immediately expose a de-interleave bug, a wrong band on one side, or an accidental mode swap WITHOUT an external oracle. Does NOT hold for a non-additive metric (Euclidean-with-sqrt dependent cost), so restrict to L1/SquaredL2. Pairs with the aeon squared-L2 oracle (`Σ_c dtw_distance(channel_c)`) as the absolute check.
- **`Problem::set_variant()` rebinds the DTW function EAGERLY — bind-time validation fires at `set_variant`, not at `fill_distance_matrix`.** `set_variant` → `refresh_distance_matrix` → `rebind_dtw_fn` → `resolve_dtw_fn`, so a rejected combination (Independent + non-Standard variant) throws right there — earlier and more serial than expected. For tests: wrap `set_variant()` in `REQUIRE_THROWS`, and set fields that participate in the rebind (`missing_strategy`) BEFORE `set_variant`. Caveat: mutating `missing_strategy` by field assignment AFTER `set_variant` does NOT rebind — the stale binding stands until the next refresh (pre-existing for all variants; real callers set strategy at construction).

## Arrow ingest + optional-dep build (Phase 5 · Task 5.7)

- **"Configure exits 0" is NOT "builds without the optional dep" — you must COMPILE a TU that includes the guarded header.** `-DDTWC_ENABLE_LLFIO=OFF` was declared done after a `configure`-only run, but `mmap_distance_matrix.hpp`/`mmap_data_store.hpp` `#include <llfio/...>` unconditionally, so every TU pulling in `Problem.hpp` failed to compile without llfio — core could not build without an optional dep (non-negotiable #3) and the Python wheel was blocked for months. Lesson: an optional-dep OFF path is only verified when a real object file that transitively includes the guarded code compiles and links. Registered the no-dep build as a HARD gate, not a configure check.
- **To make an optional type vanish without touching a `std::variant` and its visit/get sites, keep the type COMPLETE and stub only the dep-touching members.** `MmapDistanceMatrix` lives in `Problem::distMat_t = std::variant<Dense, Mmap>`; dropping it from the variant would ripple through every `std::visit`/`std::get`. Instead `#ifdef DTWC_HAS_MMAP` only the 4 llfio members (the `mapped_file_handle`, the file-mapping ctor, `open()`, `sync()`) and give the `#ifndef` branch throwing replacements — the class stays a complete, default-constructible type, the variant and all dispatch sites compile unchanged, and constructing a mapped matrix throws a clear "rebuild with llfio" error (no silent degradation). `MmapDataStore` (no non-mmap fallback) is compiled out whole because every include site is already guarded.
- **polars/pandas expose `__arrow_c_stream__`, NOT `__arrow_c_array__` — support both PyCapsule protocols.** Only pyarrow.Array/DuckDB give the single-array dunder; a polars `Series`/`DataFrame` gives the batch STREAM. A consumer that checks `__arrow_c_array__` alone silently rejects the exact producer the task names ("polars large_list"). `data_from_arrow_c_array` prefers the single array, falls back to consuming the stream (`data_from_arrow_stream`, concatenating batches); `_prepare_data` detects either dunder.
- **Vendor Arrow via nanoarrow's namespaced amalgamation, not a CPM fetch.** Two files (`nanoarrow.{h,c}`), `NANOARROW_NAMESPACE=DtwcNanoarrow` (avoids ODR clashes if pyarrow/polars-arrow is loaded in the same process), zero external deps, no build-time download (a sandboxed wheel build can't fail on it — unlike llfio's quickcpplib superbuild). Keep it a PRIVATE include of the core lib and expose only forward-declared `ArrowSchema`/`ArrowArray` pointers so nanoarrow never leaks to consumers. Gotcha: the bundler's `--header-namespace nanoarrow` emitted `#include "nanoarrownanoarrow.h"` (no separator) — normalise to `"nanoarrow/nanoarrow.h"`.
- **Prove "without pyarrow" by BLOCKING it, not by uninstalling it.** The gate set `sys.modules['pyarrow']=None` so any `import pyarrow` raises, then ran the full polars ingest+cluster and asserted pyarrow was never importable. Stronger than "pyarrow absent" (which a stray transitive import could violate) — it proves the code path itself is pyarrow-free even in an env where pyarrow is installed.

## Research Process

- **Always verify citations.** Author names, venues, volume numbers can be hallucinated.

## GPU backends + Python bindings (audit 2026-09-02)

- **A pair count must be guarded at the PUBLIC entry, not at the last
  launcher — the pre-pass runs first.** `launch_dtw_kernel` guarded
  `num_pairs > INT_MAX` correctly, but the LB_Keogh pre-pass in
  `compute_distance_matrix_cuda` ran before it and narrowed `N*(N-1)/2` to
  `int`; on the RTX 4000 Ada with N=65537 (the first N whose pair count exceeds
  INT_MAX) that produced `CUDA error at cuda_dtw.cu:1353: an illegal memory
  access was encountered`, not the intended throw. A guard on an internal
  launcher is worthless when any earlier stage consumes the same quantity:
  guard once, at the top of every public entry, before the first allocation.
- **Build the smallest failing case for an INT_MAX pair count — it is cheap.**
  N=65537 series of length 1 is a few MB of host memory, while the `N*N` result
  matrix that follows is 34 GB; putting the guard before
  `result.matrix.resize(N*N)` is what makes the case run in milliseconds. Test
  the guard helper in a CUDA-OFF build too: the seam header
  (`cuda/launch_prep.hpp`) must live OUTSIDE `#ifdef DTWC_HAS_CUDA` so the
  canonical Arrow/CUDA-OFF gate executes it (18 assertions there — same F9
  pattern as the `--ram-limit` guard under `#ifdef DTWC_HAS_PARQUET`).
- **`CUDA_VISIBLE_DEVICES=-1` forces the "no device" branch on a GPU host; the
  empty string does NOT.** `CUDA_VISIBLE_DEVICES=` still showed the device in
  this environment, so the no-device case silently skipped. With `-1`,
  `cuda_available()` is false and the pre-fix code returned an all-zero `N*N`
  matrix with no exception ("no exception was thrown where one was expected") —
  exactly the silent wrong answer the no-fallback rule forbids. Any backend
  availability check needs a runtime way to force the negative branch, or that
  branch is never tested.
- **A double-checked cache whose payload holds a `std::string` cannot publish
  without a lock.** `query_gpu_config` took a process-global mutex on every
  call, including cache hits (once per kernel launch, per host thread). The
  lock-free fix is an acquire-load fast path plus the SAME mutex for the
  one-time fill — two threads racing to fill would both assign
  `GPUConfig::device_name`, a real data race, not a benign duplicate write.
  Warn-once latches (`static bool logged`) become `std::atomic<bool>::exchange`:
  same effect, no UB.
- **An exception must never escape an `omp parallel for`, and the fix is a
  per-thread slot, never `omp critical`.** A shared error flag under `critical`
  would serialise the hot loop; use one `std::exception_ptr` per
  `omp_get_max_threads()` slot, written only by its owning thread, inspected
  after the region, first non-null rethrown. NOTE: on the current core
  `dtwBanded`/`dtwFull_L` do NOT throw on NaN (they propagate it into the
  matrix), so the audit's trigger was inferred, not reproduced — this is
  hardening against a future throwing kernel and no test can express it through
  the public API today.
- **Hand a numpy buffer over as an owned object, and build the capsule while a
  `unique_ptr` still owns it.** `double *p = new double[n*n]; ...;
  nb::capsule owner(p, deleter);` leaks the whole matrix if anything between the
  two lines throws, including the capsule allocation itself; move a
  `std::vector<double>` onto the heap, build the capsule, THEN `release()`. The
  same helper removes the `nb::gil_scoped_acquire` nested inside a live
  `gil_scoped_release` in the CUDA/Metal lambdas — build the array after the
  release scope has closed, not inside a re-acquire.
- **A per-binding GIL policy is not a lock; pick one policy for the class.**
  Holding the GIL in five `Problem` bindings while ~ten siblings released it
  gave no mutual exclusion — the released binding still resized `distMat` and
  rebound `dtw_fn_` under a live reader — so the docstrings asserted a
  thread-safety property the module did not have. The policy is now *consistent
  release, no per-object lock*, with `Problem` documented as
  single-thread-per-instance (the same contract as the C++ header); a "release
  only when the cache is complete" gate was considered and rejected because the
  only complete-cache query, `is_distance_matrix_filled()`, runs an O(N^2) NaN
  scan and gating an O(1) lookup on it is quadratically worse than the GIL it
  would avoid. If a class ever really needs concurrent Python use, the fix is a
  lock acquired BEFORE `gil_scoped_release`, never a mixed hold/release policy
  (`python/src/_dtwcpp_core.cpp`).
- **A threaded Python test is a contract guard, not a race reproducer.**
  Running `dist_by_ind` from 8 threads on an unfilled `Problem` passed against
  the GIL-releasing build on this box: the `omp critical(distByInd_init)` around
  the resize is a global named lock even outside a parallel region, so the
  window is narrow. What DOES discriminate deterministically is a pure-Python
  ticker thread — it can only advance while it holds the GIL, so a native call
  that keeps the GIL freezes it. Calibrate against a call known to release
  (`fill_distance_matrix`) and take the MINIMUM tick delta over a few repeats;
  GIL hand-off convoying adds tens of milliseconds of noise, and that noise only
  ever adds ticks.
- **`std::atomic<T>` breaks a `= default` move.** `Problem::Problem(Problem &&)
  = default;` is defined out-of-line, so an atomic member makes it ill-formed,
  not merely deleted. The `mutable bool` written from a `const` method needed a
  tiny value-moving `RelaxedFlag` wrapper (`Problem.hpp`), not a bare atomic.
- **A per-pair decode is device code.** `dtwc::detail::decode_pair` is
  `__host__ __device__` and runs once per pair per launch, so a defensive
  correction loop that recomputes `row * (2N - row - 1) / 2` costs a 64-bit
  multiply AND divide on every pair for a case that cannot occur with an FP64
  seed; hoisting `row_start` (which the up-correction needs anyway) makes the
  down-correction one comparison. Also order defensive clamps so the LAST one is
  the one that must hold — clamping low then high re-introduced a negative row
  for `N < 2`.

## Clustering algorithms + score layer (audit 2026-09-02)

- **An UNNAMED `#pragma omp critical` shares ONE implementation-defined name
  across the entire program.** Four per-thread reductions (`fast_pam.cpp` x3,
  `one_batch_pam.cpp` x1) were unnamed, so they serialised against each other
  AND against any unnamed critical in any other linked TU — unbounded coupling
  to code this layer does not own. The neighbouring failure-path criticals were
  already named (`dtwc_medoid_candidate_failure`, `tadpole_density_reduce`), so
  this was oversight, not design. Every `critical` in the codebase must carry a
  distinct name; the fix is free.
- **A precision guard on the DATA is not implied by a guard on the VARIANT.**
  TADPole's `bounds_valid` checked variant + ndim + missing_strategy but not
  `is_f32()`, so on a Float32 Problem the LB/UB path called `Data::series()` on
  the EMPTY float64 `p_vec` (historically an out-of-range read, reproduced as a
  segfault; `Data::series()` now throws, so the same code would be a loud error
  today). The guard is still required for the other reason: `exact()` routes
  through `Problem::dist_by_ind`, which DOES branch on `is_f32()`, so bound and
  exact distance would come from different data and the prune is INADMISSIBLE
  regardless of memory safety — `barycenter.cpp` had this right. Two follow-on
  rules: whenever a fast path reads series directly, check `is_f32()` alongside
  the variant and hoist the read INSIDE the guard rather than parking an
  empty-span placeholder outside it; and a capability that switches itself off
  must SAY so (`TADPoleStats::pruning_enabled` exists because the fix otherwise
  turned a segfault into an unobservable 2-10x slowdown).
- **A degenerate score's "no other cluster" branch must throw, not fall through
  a sentinel.** `silhouette` left `min` at `numeric_limits<double>::max()` when
  no second cluster existed, so `(MAX - a)/MAX` evaluated to ~ +1.0 — three
  visibly different series in one cluster scored PERFECT; `dunn` likewise
  returned `DBL_MAX / max_intra` ~ 1.8e308 as an ordinary finite number. A
  `DBL_MAX`/`DBL_MIN` "not found yet" sentinel that survives into arithmetic
  produces a plausible-looking extreme value, not an obvious error. Assert the
  sentinel was replaced (`dtwc/scores.hpp`).
- **Validity indices are defined on the REALISED partition, never the declared
  `n_clusters`.** With `Nc = 3` but only label 0 in use, every "requires >= 2
  clusters" guard passed vacuously. An empty declared cluster also has no medoid
  and no scatter, yet it still divided Davies-Bouldin's `1/k` and set
  Calinski-Harabasz's `(k-1)`/`(N-k)` degrees of freedom. Count the labels that
  actually occur.
- **`d == 0` in a ratio index is the WORST case, not a case to skip.**
  Davies-Bouldin skipped pairs with `M_ij == 0`, so two clusters with coincident
  medoids and real internal spread — the worst possible configuration — left
  `max_ratio` at its `0.0` initialiser and DBI reported 0.0 = perfect. Davies &
  Bouldin (1979) require R_ij strictly decreasing in M_ij with R_ij = 0 iff
  S_i = S_j = 0, so the limit is +inf when there is any spread and 0 only in the
  genuine 0/0 case. Skipping a degenerate term silently reports its OPPOSITE.
- **The one distance read that skips the finiteness policy is the one the table
  cannot cover.** `one_batch_pam`'s `exact()` is reached exactly when a selected
  medoid is NOT in the fixed batch, i.e. the only distances the constructor's
  finiteness sweep never saw. A non-finite `d` makes `d < best` false in every
  slot, so the point keeps label 0 and the run publishes a wrong partition with
  a non-finite cost, where every other algorithm throws. Audit finiteness guards
  by asking which inputs each guard cannot see, not by counting guards.
- **Loop-invariant work in a candidate loop hides as "per-candidate state".**
  `one_batch_pam` rebuilt a `vector<double>(k)` removal-gain base and recomputed
  an O(m) tolerance ONCE PER CANDIDATE (N heap allocations plus N O(m) passes
  per sweep) although both depend only on state that changes when a swap is
  ACCEPTED. Hoisting into a `refresh_swap_state()` called after each accepted
  swap keeps the j-accumulation order, so the arithmetic stays digit-identical;
  `fast_pam.cpp` already had the hoisted pattern — copy it.
- **`dist_by_ind` ran its semantic preflight TWICE per element** — once
  directly, once inside `ensure_dense_cache_configuration_current` — and the
  SWAP kernel issues N^2 of them per iteration. The fix without any reordering
  is a `..._preflighted()` sibling that omits the leading preflight, for callers
  that already ran it. Do NOT "fix" it by dropping the outer call, which
  silently reorders the mmap-identity and semantics errors.
- **A dead template helper is worse than a duplicated loop: its unit tests
  certify behaviour that ships nowhere.** `medoid_utils`' `assign_to_nearest` /
  `compute_nearest_and_second` / `find_cluster_medoid` had no production caller
  (only 20 of their own tests) while the header's `@details` named three callers
  it did not have, and `dtwc.hpp` re-exported it publicly; their semantics had
  also drifted from every shipping copy (no `require_finite_medoid_distance`,
  plain `+=` instead of the ordered published objective, and a `DBL_MAX`
  sentinel that mis-handles a legitimately maximal distance). They were deleted,
  not adopted: the five shipping assignment scans differ in parallel-vs-serial
  execution, index space (point index vs chunk index + global offset), distance
  signature (`(int,int)` vs `(series,series)`) and per-element side effects (a
  DTW counter), so one helper would need runtime policy switches in the hottest
  loops of the library. Grep for production callers before trusting a helper's
  unit tests as coverage.
- **Digit-identity for a pure-deletion refactor is cheapest to prove at the
  OBJECT level.** Deleting the three never-instantiated templates left all 19
  `dtwc++` library objects BYTE-IDENTICAL after a real recompile — check the
  `.obj` mtimes to prove the recompile actually happened, or the claim is
  vacuous. Pair it with `catch2 -s --order decl` transcripts (which print every
  assertion's expanded values) and a real-CLI artifact diff; the templated
  `fast_clara` f32/f64 chunk merge was proven the same way against the 32
  f32/f64 x stream/resident streaming artifacts of the Arrow-ON gates.
- **The report finding D4 ("inverted `#ifndef DTWC_HAS_PARQUET` guard") is
  FALSE — verified, do not "fix" it.** `unit_test_fast_clara.cpp`'s "the missing
  capability is loud" SECTION asserts the error a build WITHOUT Parquet must
  raise, so it correctly runs only there: 842 assertions in the Arrow-OFF
  canonical build vs 841 in `build/arrow-pyarrow-23`. The genuine residual is
  the opposite one — a Parquet build has no positive coverage of the same entry
  point.
- **`barycenter_kmeans` re-ran a full `N*k` DBA-DTW assignment after the loop
  even on the converged-`break` path**, which had already stored the identical
  labels and cost from the `assign()` at the top of that same iteration (centres
  are not updated before the break). Guard the final assign with
  `if (!result.converged)` — only the max_iter-exhausted path needs it. Verified
  digit-identical on both `unit_test_barycenter` and the allocation-counting
  `unit_test_barycenter_allocations`.
- **Guard ORDER is behaviour.** A named "Pruned requires Float64" rejection
  placed ABOVE the `Pruned + mmap -> BruteForce` downgrade broke a combination
  that the downgrade already routed to an f32-capable exact fill. A rejection
  must sit below every downgrade that would make it moot.
- **Catch the narrowest type in a skip-and-warn path.**
  `catch (const InvalidInput &)` around `scores::silhouette()` swallowed corrupt
  labellings and bad data as warnings. `dtwc::UndefinedScore : InvalidInput` now
  marks only the mathematically-undefined case, and only that is skipped.

## Exact MIP backends (audit 2026-09-02)

- **An incumbent is not a lower bound.** Benders read its master LB from
  `sum(theta_j)` of the returned solution, which is the master's PRIMAL
  objective; with `mip_rel_gap` set (and `kObjectiveBound` / `kSolutionLimit`
  accepted as success) it sits ABOVE the master optimum, so
  `best_cost - theta_sum` under-reported the gap and the loop stopped while a
  strictly better medoid set existed. The dual bound is
  `getInfo().mip_dual_bound`. Whenever a decomposition uses a solver's answer as
  a bound, take the DUAL bound, and accept only `kOptimal` as master success.
- **An exact entry point that runs out of iterations must throw.** Both Benders
  and LR-core ended their loop/tree on a cap and then published the incumbent —
  Benders printed "complete", LR-core ignored its own `certified_optimal =
  false`. The compact HiGHS/Gurobi backends already threw `SolverError` on
  non-optimality; a second exact backend that returns a heuristic instead is a
  silent wrong answer on a documented API.
- **A solver option setter returns a status, and identity options are not
  tuning options.** `Highs::setOptionValue("solver", "not-a-solver")` returns
  `kError` and HiGHS then runs its DEFAULT solver — verified: before the guard,
  `pdlp_lp_bound(..., variant = "not-a-solver")` solved happily and reported a
  "PDLP" bound produced by dual simplex. Check every setter's status, but split
  the two cases first: failing hard on a rejected `solver` is right (the
  reported value would not be a PDLP value), while failing hard on a rejected
  `kkt_tolerance` just means an older HiGHS build produces no bound at all —
  hence `set_highs_option` vs `set_highs_option_best_effort`.
- **A tolerance is not one number.** Scaling the Benders `abs_eps` by
  `max(max_distance()/2, 1)` was right for the two COST comparisons and
  catastrophic for the third site that happened to read the same variable: the
  cut-coefficient filter. Every `c_i = max(0, d_nearest - d_ji)` is `>= 0`, so
  dropping a positive one makes the cut STRICTER than the valid Benders cut —
  the master's dual bound is inflated and the optimum can be cut off under a
  message claiming optimality was proved. Before rescaling a shared epsilon,
  enumerate every consumer and ask what quantity each compares; naming the
  value (`benders_abs_eps` vs `benders_cut_coefficient_threshold`) is what
  forces that.
- **A "be honest, throw" change is a default-path behaviour change.** Turning an
  unconverged Benders incumbent into a `SolverError` is correct, but
  `benders = "auto"` engages Benders at `N > 200` and no test exercised
  `N > 200` on stock settings — the largest case was `N = 12`. Any new throw on
  the documented default route needs a default-settings regression AT that size;
  the measured floor (`N = 250`, `k = 5`, separable, converging at Benders
  iteration 4) is what lets the shipped `max_benders_iter = 200` stand
  (`tests/unit/mip/test_mip_backend_guards.cpp`).
- **An error message must name a knob the caller actually has.** LR-core's
  "raise the node cap" pointed at `LagrangianParams::max_nodes`, which
  `Method::LRCore` default-constructed and never exposed;
  `MIPSettings::lr_max_nodes` now exists and is forwarded. Same rule for the
  Benders message: name the field, not the concept.
- **An index guard must bound the NARROWEST integer on the path, and is
  per-backend work.** The HiGHS model guard checked `HighsInt` while the
  triplets were built with `static_cast<int>`, so a `HIGHSINT64` build would
  have passed `N > 46340` straight into truncation — the exact failure the guard
  was added to prevent. The same claim also covered Gurobi's `addVars(N*N)`,
  which had no check at all (`dtwc/mip/index_guard.hpp`).
- **`near` is a Windows macro.** `const auto near = ...` in `dtwc/mip/` fails to
  compile on Windows (`windef.h`, pulled in via llfio/HiGHS, defines
  `near`/`far`). Use `nearest`.

## MATLAB / MEX (audit 2026-09-02)

- **A MEX inherits MATLAB's private MSVC runtime, so a constexpr `std::mutex`
  is a cross-release landmine. [confirmed 2026-09-02]** The SAME HiGHS-enabled
  `dtwc_mex.mexw64` crashed R2024b (`0xc0000005`, `Thrd_yield+184`, `RAX=0`) and
  passed R2025b; the hosts differ only in `bin/win64/MSVCP140.dll` 14.36 vs
  14.40. STL 14.40+ emits an all-zero constexpr mutex that 14.36's `_Mtx_lock`
  dereferences. Remedy: top-level
  `add_compile_definitions(_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR)` guarded by
  `DTWC_BUILD_MATLAB AND (MSVC OR (WIN32 AND CMAKE_CXX_SIMULATE_ID STREQUAL
  "MSVC"))`, placed BEFORE `Dependencies.cmake` so it reaches fetched HiGHS and
  llfio (the ctor inlines into their objects; a define on `dtwc_mex` alone
  cannot work — check `_deps/highs-build/.../highs.vcxproj`). Kill the DLL-clash
  hypothesis first with `llvm-objdump -p` (HiGHS is static; MATLAB's own
  `highs.dll` is never imported). Nearest-export stack labels name neither the
  failing function nor its caller; a binary-identical, host-different
  differential localises the fault in one run. `tests/matlab` now runs under
  CTest as `matlab_suite` (needs `DTWC_BUILD_MATLAB=ON` and `matlab` on PATH);
  MATLAB "Incomplete" (filtered-by-assumption) is a silent skip, so the gate
  fails on it outside an explicit allow-list and floors the passed count.
- **CMake 4.2 `FindMatlab` only exports `mexFunction` under `if(MSVC)`.** A
  Clang (GNU-frontend) MEX on Windows then links with no entry point and MATLAB
  says "Invalid MEX-file ...: Gateway function is missing". `build/mex-verify`
  was generated by an older CMake and still carries `-Wl,/EXPORT:mexFunction`;
  `bindings/matlab/CMakeLists.txt` now adds it explicitly for
  `WIN32 AND NOT MSVC`.
- **A helper that eliminates UB can reintroduce it at its own boundary.**
  `exact_int_from_double` accepted `INT_MIN` (exactly representable as a double)
  and every caller then evaluated `INT_MIN - 1` — signed overflow, in the helper
  added to remove signed overflow. Check the boundary the CALLER will cross, not
  the one the helper converts (`bindings/matlab/dtwc_mex.cpp`).

## Parity, checkpoint, portability campaign (2026-09-02, second pass)

- **`DenseDistanceMatrix::resize(N)` is a wipe, not a growth. [confirmed]**
  It NaN-fills every packed slot, and `fillDistanceMatrix_BruteForce` called
  it unconditionally, so `load_checkpoint` + `fill_distance_matrix` discarded
  the restored matrix and recomputed everything: resume never worked through
  the fill entry point (25 series: 15 s vs 0.15 s after `if (m.size() != N)`).
  The pruned builder had already been fixed with a comment claiming the
  brute-force path matched; read the code, not the comment. Every semantic
  setter invalidates through `refresh_distance_matrix()` → `resize(0)`, which
  is why the conditional resize is safe.
- **A checkpoint that never prunes generations is O(N³/interval) on disk.**
  Mid-fill saves published one immutable full N×N generation each and kept
  them all (6 after run 1, 7 after run 2). Retain exactly one generation,
  removed only after `CURRENT` points at the new one, and document the O(N²)
  per-save cost with a sizing rule next to the interval knob.
- **Tier-1 routes must be side-effect-free. [confirmed]**
  `Problem::cluster()` → Lloyd persisted per-repetition medoid CSVs into the
  CWD-relative `./results/` and threw when it was missing; nothing exercised it
  because the C++ tests ran from the source root, which has a gitignored
  `results/`. It surfaced only when MATLAB Tier-1 started routing `kmedoids`
  through C++. Rule: algorithms print only under `verbose()` and write nothing;
  `cluster_and_process()` owns artifacts via an explicit flag; every writer
  creates its directory. Test from a fresh temp CWD and assert it stays empty.
- **Re-route bindings through the C++ Tier-1, do not re-implement it.**
  MATLAB re-implemented `cluster` in `.m` and drifted six ways (kmedoids ran
  FastPAM, three methods missing, device mutated the global, ordinal names,
  clara ignored max_iter, no `k <= N`); Python drifted seven. One MEX/nanobind
  call into `dtwc::cluster`/`dtwc::load` closed them all at once and deleted
  the `.m` writer and the numpy parser. Remaining Python-owned pieces (HPC
  route, `elapsed_s`, `plot`) are the only legitimate wrapper logic.
- **MSVC `char8_t` → `fs::path` throws on unmappable bytes. [confirmed]**
  `fs::path(std::u8string)` raises `system_error` ("No mapping for the Unicode
  character exists") when the UTF-8 is invalid, so a native-encoded `--name`
  from `argv` would have failed every write. `utf8_to_path` validates first and
  falls back to the native narrow interpretation. Loader-produced names are
  UTF-8 on every platform (`path_to_utf8`); writers convert back with
  `utf8_to_path`; the CLI `--name` from `argv` is still ACP on Windows.
- **Generated docs bite twice.** `docs/content/contributing/guideline.md`,
  `math/lr-core.md`, `api/tier-*.md`, `getting-started/{cli,configuration,
  checkpointing}.md` are outputs of `scripts/generate_docs.py`; hand edits are
  reverted by the next contract check, and `check_docs_contract.py --cli`
  compares BOTH `cli.md` and `configuration.md` against the live `--help`.
  Edit `docs/api-contract-2.0.md` or the source, then regenerate.
- **`PASS_REGULAR_EXPRESSION` makes CTest ignore the exit code.** A "rejects
  unknown option" test written that way passes on any output containing the
  phrase. Drive the binary from a `.cmake` script that asserts the non-zero
  exit AND the flag name, not the CLI11 wording.
- **Deleted overloads close positional-argument traps.** Inserting `int
  skip_rows` before `char delimiter` let a future `load(p, 0, ',')` bind `','`
  to `skip_rows=44`; `load(path, int, char, string_view) = delete` turns it
  into a compile error. A non-dependent `static_assert(!requires{...})` on a
  deleted call is a hard error on clang 19; use a dependent `requires`
  expression through a concept.
- **`long` is not a width.** `lr_max_nodes` was `long` (32-bit on Windows,
  64-bit on Linux) and became a public setter in two bindings before anyone
  noticed. Use `std::int64_t` for anything that crosses a binding.
