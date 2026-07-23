# DTWC++ — Research & Release Campaign (PLAN v2.0)

> **For Codex (autonomous, single continuous run):** this file is the mission. Read
> `AGENTS.md` first (the working rules), then this file top to bottom, then start at
> Phase R0 and do not stop. Never wait for an operator. Improvise where the evidence
> justifies it — this plan states goals, invariants, and suggested routes, not a
> script — but record every departure as a Decision-log entry the moment you make it.
>
> The prior plan (2.0-refactor Phases 0–9 with the complete task history, full
> decision-log prose, and progress log) is archived **verbatim** at
> `.claude/PLAN-archive-2026-07-20-phases0-9.md`. Before re-opening ANY idea, search
> that archive plus `.claude/LESSONS.md` and the Killed-ideas section below.
> Re-opening a killed idea requires explicitly overturning the recorded kill
> evidence, never forgetting it.

**Status (2026-07-23):** 2.0.0rc1 release state committed (not tagged or
published). Refactor Phases 0–7 CLOSED. Phase 8: 8.0 + 8.1 CLOSED (149
protocol-clean commits `8debf1d..eda1b92`); 8.2 findings F1–F7, F10, and the
sanitizer gate CLOSED; **F8–F9 OPEN**. Phase R0 is adjudicating the remaining
2026-07-20 interrupted work before anything else. The final **2.0.0 tag gates
on R0–R6 CLEAN**; R7 (WASM Playground) is a 2.1 feature and does not gate the
tag. Tag/publication/hosted-CI/ARC/Metal-runtime remain explicit USER actions —
never wait on them.

---

## The grand goal

Ship DTWC++ 2.0.0 as the reference implementation for large-N time-series
clustering under elastic distances: **provably correct** (every algorithm backed by
a written derivation and an independent oracle), **honestly fast** (memory-bound
analysis first, machine-independent performance fences, no unverifiable speed
claims), **cross-language consistent** (C++/Python/MATLAB, CasADi model), and
**scalable in stated regimes** (exact methods to N≈10⁴, OneBatchPAM/CLARA
streaming toward the 100M×8K ambition with a derived, validated capacity model
that says exactly where the local machine ends and HPC begins). The evidence trail
must be good enough that a careful scientist can re-derive, re-run, and audit
every claim from the repository alone.

## How Codex works this plan

You have effectively unlimited tokens but a hard wall-clock limit, and the last
two runs **died mid-task with everything uncommitted**. These rules exist so a
death costs one task, never a session:

1. **Commit cadence is a law.** One conventional commit per task or finding, the
   moment its gate is green. Never hold more than one task's work uncommitted.
   Never one omnibus commit. Message style: see AGENTS.md.
2. **No pauses, no questions, no operator check-ins.** If a sub-item cannot run in
   this environment (tool absent, platform-unsupported, network-blocked): execute
   its named fallback if one is written into the task; otherwise record WHY with
   verbatim probe evidence in the Decision log, mark the checkbox `[BLOCKED-ENV]`,
   and continue to the next item. Nothing in Phases R0–R7 may stall on a human.
3. **Checkboxes are the resume protocol.** Keep them truthful in real time. If the
   session dies, the next session resumes from the checkboxes plus the latest
   `.claude/summaries/handoff-*.md`. Update the handoff file incrementally (append
   as you go), not only at session end.
4. **Registered bands before decisive runs.** Every pass/fail band is written into
   the test/bench script BEFORE the run; the verdict prints against it. A
   falsified band is recorded FALSIFIED and kept — a falsification with a
   registered target is a deliverable, never rescue-tuned past 2 attempts.
5. **Evidence or it did not happen.** Every claimed number lives in a run-log
   committed to `.claude/baselines/` (never `benchmarks/baselines/`). Decisive
   outputs quoted VERBATIM. Tag load-bearing claims **[confirmed]** (name the
   artifact) or **[inferred]** (name what would confirm it).
6. **Arbiters for disagreement.** Two computations disagree → neither judges;
   build a third from different mathematics. "No-op/refactor-only" claims require
   digit-identical outputs on a recorded case. A test failing after your change:
   stash, rerun, compare digit-for-digit before claiming "pre-existing".
7. **Findings are hypotheses until you open the evidence.** Your own included.
   A green unit test on a helper proves the helper works, never that it is
   reachable — drive the real binary for CLI-facing behavior (F7/D1 lesson).
8. **Improvise, but leave a trail.** You may reorder tasks, interleave phases
   (R2 derivation findings feed R3 rounds — that interleave is expected), split
   or merge tasks, and choose methods this plan did not anticipate. Each such
   choice = one dated Decision-log line: what changed, why, what evidence.
9. **Scope discipline.** Minimal edits; every changed line traces to a task or a
   recorded finding. No speculative rewrites. Simple > clever.
10. **Session end (or when you sense the limit):** write/refresh the handoff in
    `.claude/summaries/handoff-YYYY-MM-DD-<topic>.md`, add LESSONS/CITATIONS
    entries, ensure the working tree is clean (commit or revert), and note the
    exact resume point.

## Global constraints (apply to every task)

- Core builds with NO optional deps (OpenMP, HiGHS, Gurobi, CUDA, Metal, MPI,
  llfio, Arrow all optional). Never make an optional dep required.
- No runtime dependence on repo-relative paths.
- Every user-visible change: tests added/updated, `CHANGELOG.md` (Unreleased)
  updated, lint clean.
- C++20 minimum (C++23 preferred where toolchains allow); no naked `new`/`delete`
  in core; buffer > thread_local >> heap allocation in hot paths.
- No silent fallbacks anywhere: a requested capability that cannot be delivered
  errors or warns loudly — never quietly degrades.
- Python tooling via `uv` only, never pip.
- Data files are read-only; never write outside the git project root; never push,
  publish, tag, or SSH to remote machines (all operator actions).
- Wall-clock benchmarks on this shared machine are ADVISORY ONLY (record the
  numbers plus a "machine under load" flag); HARD perf verdicts are
  machine-independent counters or quiet-CI numbers.
- Guards that must fire on a build WITHOUT an optional dep live OUTSIDE that
  dep's `#ifdef` (F7/D1). Gates must assert their subject RAN, not merely that
  ctest went green — a skip counts as a pass to ctest (F9 lesson).

## Campaign map

```text
R0  Adjudicate the 2026-07-20 in-flight work        — FIRST, nothing else until done
R1  Repository cleanse & record reconciliation      — non-behavioral hygiene
R2  Mathematical re-derivation program              — the science core; runs alongside R3
R3  Logical-mistake hunt (8.2 lens list continues)  — F8–F10 first, then the lenses
R4  Code simplification (behavior-frozen)           — after R3's exit band
R5  Performance assurance & fences                  — after R4; no-op oracle binds
R6  Exit gate, scale rehearsal, release readiness   — gates the 2.0.0 tag
R7  WASM Playground (Phase 9 carried verbatim)      — after R6 CLEAN; does not gate tag
```

R2 is deliberately concurrent with R3: derivations generate findings; findings
demand derivations. Everything else is ordered — bug-hunting before
simplification before performance, because simplification and perf work are
behavior-frozen against an oracle that bug fixes would invalidate.

---

## Phase R0 — Adjudicate the 2026-07-20 in-flight work [OPEN — do this first]

The working tree contains uncommitted changes dated 2026-07-20 ~23:00 (a prior
Codex run that died before committing). Inventory [confirmed: `git status`/`git
diff` 2026-07-23]:

- `tests/unit/core/unit_test_distance_sampling_weights.cpp` (NEW) +
  `dtwc/algorithms/fast_pam.cpp` (+91/−36) + additions in
  `unit_test_fast_pam.cpp`, `unit_test_clustering_algorithms.cpp`, CHANGELOG
  entry — shaped like **F10** (direct tests for
  `core::distance_sampling_weights`, seeded degenerate/signed branches).
- `.github/workflows/ubuntu-unit.yml` (+98) — shaped like **F9** (Arrow-enabled
  CI job). A matching LESSONS entry ("assert the subject RAN") was added.
- `tests/unit/unit_test_cli_args.cpp` (+95) — possibly F8/F7-adjacent.
- Scholarly work: Vinod (1969) provenance in `.claude/UNIMODULAR.md` +
  `.claude/CITATIONS.md` (Crossref-verified, full text NOT read — recorded
  honestly as [inferred]); `.claude/TODO.md` staleness note; LESSONS fast-math
  corrections (build uses an explicit flag subset incl. `-fassociative-math`,
  NOT `-ffast-math` — cites `cmake/StandardProjectSettings.cmake:59-70`).
- `PLAN.md` had two 8.3 lens refinements — already absorbed into Phase R4 below.

Tasks:

- [x] Read every diff hunk. Classify each change: F8 / F9 / F10 / docs / other.
- [ ] Build + run the full canonical gate (floor: **113/113, 0 failed**, 6
      capability skips — `.claude/summaries/handoff-2026-07-13-f7-streaming.md`).
      Run the new/changed test suites explicitly and quote their output.
- [ ] For the F9 workflow change: you cannot run GitHub CI locally — verify the
      YAML by schema/actionlint if available, verify the job obeys the
      "assert-the-subject-RAN" lesson (greps the test binary's own output for
      executed assertions, not just ctest green), and mark the CI run itself as
      an operator-triggered verification in the Decision log. If the local
      equivalent (an Arrow-ON build dir running `test_io_readers` for real) is
      constructible, BUILD IT — that, not CI, is the primary F9 closure (see R3).
- [ ] Verdict per change: KEEP (gate green, contract met) → commit as its own
      conventional commit crediting the finding it closes; REPAIR (close but
      defective) → fix, then commit; REVERT (wrong or unverifiable) → revert
      with a Decision-log line naming why. No change may stay uncommitted.
- [ ] Update the F8/F9/F10 checkboxes in R3 to reflect what actually closed.

## Phase R1 — Repository cleanse & record reconciliation [OPEN]

Non-behavioral hygiene: make the repository's *record* as trustworthy as its
code. Nothing here may change program behavior (no-op oracle not required since
no core code changes — but if any item does touch code, it moves to R4's rules).

- [ ] **TODO.md full reconciliation.** `.claude/TODO.md`'s audit list is a
      2026-07-06 snapshot; Phases 4–8 closed an unknown subset without editing
      it (the 2026-07-20 note reconciled exactly one entry). Verify all ~30
      entries against the current tree: each becomes CLOSED-BY (commit/task),
      STILL-OPEN (→ becomes an R3 finding), or NOT-REPRODUCIBLE (evidence
      quoted). Rewrite the file to the reconciled state.
- [ ] **Docs truth audit.** Every claim in README.md, docs site pages, and
      `docs/api-contract-2.0.md` traces to an artifact (test, baseline run-log,
      citation) or is corrected. Run the existing drift gates
      (`check_docs_contract.py --cli <fresh dtwc_cl>`, docs internal-link gate)
      and quote results.
- [ ] **`.claude/` record hygiene.** LESSONS.md and CITATIONS.md: dedupe,
      verify file:line references still hold after Phase 8's churn (spot-check,
      fix stale ones), keep every lesson. UNIMODULAR.md / MISSING.md / READ.md:
      add a one-line freshness header (what date, what supersedes it, or mark
      current).
- [ ] **Tracked-file junk census.** Find tracked files that should not be
      tracked (stale binaries, generated artifacts, orphaned fixtures) —
      grep-verify zero references before each removal; `.gitignore` audit
      (build dirs, `tools/emsdk`, `web/pkg` when R7 arrives). Do NOT delete
      untracked local build directories — inventory them in the handoff with
      which recipe each serves; disposal is an operator decision.
- [ ] **CHANGELOG structure check.** Unreleased vs rc1 sections coherent; every
      Phase-8 breaking change present (the F7 pair is: `--ram-limit` hard-errors
      on non-Parquet input; `--device cuda` rejected for matrix-free FastCLARA
      incl. `--method auto` above 5,000 series).
- [ ] **Branch state note.** Branch `Claude` is far ahead of `main`; merging is
      an operator decision — record the current ahead-count and a proposed merge
      plan in the handoff, do not merge.

## Phase R2 — Mathematical re-derivation program [OPEN — the science core]

Re-derive every algorithm in the library from primary sources, check the
derivation against the code line-by-line, and pin each with an oracle test. This
is where "top quality science" lives — treat it as research, not paperwork.

**Rules.** One file per topic under `docs/derivations/<nn>-<topic>.md` (the
LR-core derivation stays at its enforced home `docs/sources/lr-core-derivation.md`
— extend in place). Each file: full derivation a reader can reproduce from
scratch (skipped algebra hides errors); every equation unit-checked; every
approximation named with its regime and leading-order error term; assumptions
stated where they enter; a **code-conformance table** mapping each equation to
`file:line`; a verdict line per claim — CONFIRMED (evidence named) / DISCREPANCY
(→ becomes an R3 finding with a failing test) / OPEN. Citations verified against
`.claude/CITATIONS.md` (extend it; primary sources read where accessible,
[inferred] flagged where paywalled — the Vinod entry is the template). Where a
docs-site math page exists, keep it in sync (drift gate).

Derivation targets — each is one checkbox, one file, one conformance pass:

- [ ] **D1. DTW recurrence + Sakoe–Chiba band.** Optimal substructure; boundary
      conditions; band feasibility (`band ≥ |n−m|` for a nonempty path);
      monotonicity `DTW_band ≥ DTW_full` and monotone-in-band; L1 vs squared-L2
      local costs and what "distance" each yields (squared form is not a metric
      — say so). Conformance: `dtw_kernel.hpp`, `dtwBanded`.
- [ ] **D2. Envelopes + LB_Keogh.** Keogh & Ratanamahatana admissibility proof;
      formalize the recorded gotcha that `compute_envelopes(series, band<0)`
      yields a band-0 envelope (LB invalid for full DTW) — state the correct
      construction and verify the two call-site classes.
- [ ] **D3. LB_Enhanced + LB_Webb.** Admissibility proofs (Tan SDM 2019; Webb &
      Petitjean PR 2021); prove `LB_Webb ≥ LB_Keogh`; prove our tail-cap
      column-align variant (`idx=min(j+w,n-1)`) only loosens (stays valid);
      document that NO ordering exists between Enhanced and Keogh (take-max).
- [ ] **D4. EAPruned exactness + the relaxed threshold constant.** Herrmann &
      Webb argument (every optimal-path cell ≤ DTW ≤ UB ⇒ pruning exact). Then
      derive the accumulation-error bound that justifies
      `thr = ub·(1 + n_long·16·ε)` under `-fassociative-math`: standard
      summation bound |fl(Σ)−Σ| ≤ (n−1)ε·Σ|terms|/(1−(n−1)ε) — show the
      implemented slack dominates the worst case, name the regime where it
      would not (if any), and confirm relaxation only ADDS cells (exactness
      preserved). The 16 is currently asserted, not derived — derive it.
- [ ] **D5. MSM.** Stefan/Athitsos/Das C-function case analysis; metric proof
      (esp. triangle inequality); equivalence of our recurrence with aeon's
      (read from source — the LESSONS entry on aeon naming applies).
- [ ] **D6. TWE.** Marteau's stiffness/edit derivation; metric proof; the
      front-zero-pad equivalence; dimensional roles of ν (per-index penalty)
      and λ (edit cost) — unit-check both.
- [ ] **D7. Soft-DTW + the negativity bound.** Cuturi & Blondel softmin
      smoothing; forward recurrence; adjoint/gradient derivation (verify the
      H3-tested production adjoint symbolically, not only by finite
      differences). Then derive the exact negativity bound:
      `−γ·log(#paths) ≤ d_γ(x,x) ≤ 0` with `#paths` the Delannoy number
      `D(n,m)` — giving the worst-case negative magnitude
      `≈ γ·n·log(3+2√2)` for n≈m. This quantifies how negative Soft-DTW
      dissimilarities can go, which the D-sampling translation fix (R2-D13)
      depends on.
- [ ] **D8. WDTW / ADTW / DDTW.** Weight function and penalty semantics;
      limits (WDTW g→0 → constant-half-weight; ADTW penalty→0 → Standard) —
      confirm the code honors the limits the M34 domain decisions promise.
- [ ] **D9. DTW_I ≤ DTW_D.** Write the proof (per-channel additive costs, same
      band ⇒ the D-path is feasible for each channel's independent problem);
      name the exact condition (holds for L1 and squared-L2 local costs, NOT
      for Euclidean-with-sqrt) — this inequality is a standing HARD test band,
      so the proof is load-bearing.
- [ ] **D10. FastPAM1/FasterPAM decomposition.** Schubert & Rousseeuw:
      derive `ΔTD(m, x_c) = acc + ploss[m]` and the removal loss identity;
      eager-swap termination at a local optimum; why the cached-matrix regime
      caps the speedup at removed-arithmetic only (the recorded 2.3–8.1×
      memory-bound result — connect to D17).
- [ ] **D11. OneBatchPAM.** The m=O(log n) single-batch estimator; the NNIW
      weighting; the finite-max diagonal debias as implemented (paper says +∞;
      code follows the authors' experiment code — M7 provenance) — state the
      estimator's bias/variance tradeoff and the regime where m is too small;
      relative-tolerance stopping semantics (M9).
- [ ] **D12. CLARA / FastCLARA sampling.** Sample-size vs quality; what the
      seeded per-subsample schedule guarantees (and what it does not); the
      full-sample boundary semantics fixed in F5.
- [ ] **D13. D-sampling with signed weights — bless or fix.** k-means++
      D-sampling theory (Arthur–Vassilvitskii) and the L5 decision that PAM
      seeds ∝ D (not D²). The F7-era fix translates all unselected weights by
      `−min(0, d_min)` when Soft-DTW yields negatives. **A common shift does
      NOT preserve proportional sampling** — small weights gain relative mass.
      Derive what distribution the shift actually samples; compare against
      alternatives (clamp-at-zero — degenerate when many negatives; softmax —
      changes semantics; shift by the D7 negativity bound — data-independent).
      Decide with a recorded rationale: bless the shift (state exactly what it
      guarantees and what it gives up) or replace it (failing test first).
      This is a genuine open research question — treat the current code as a
      hypothesis, not an answer.
- [ ] **D14. TADPole + density peaks.** Begum Theorem 1 (admissible LB/UB
      pruning yields the exact density-peaks result) re-derived; the
      δ-convention for the globally densest point (Begum Table 2 ≠
      Rodriguez–Laio) and why the strict total order on ρ-ties is required
      for determinism; why only the cutoff kernel admits pruning.
- [ ] **D15. LR-core solver chain.** Verify and extend
      `docs/sources/lr-core-derivation.md`: Lagrangian dual of the Balinski
      p-median program (cite the Vinod provenance note from R0); subgradient +
      Kelley convergence; reduced-cost fixing validity (Beasley); Benders
      y-branching exactness argument; the PDLP-equals-Kelley LP-bound identity
      that the 7.14e-9 arbiter result rests on. Confirm each against
      `dtwc/mip/` code.
- [ ] **D16. Barycenters: DBA, soft-DTW barycenter, SSG.** Update-rule
      derivations; the M4 inverse-Lipschitz step cap `min(η, 1/(2·max V_ii))` —
      derive the descent guarantee it provides for the fixed-path quadratic;
      the M24 convergence criterion (why a completed prior finite assignment is
      required).
- [ ] **D17. Numerical error model + roofline.** (a) Accumulation-error bounds
      for the DP kernels under `-fassociative-math`; derive the f32-vs-f64
      agreement band from conditioning (the R3 precision lens is told "derive
      the band, don't guess" — this is where it comes from). (b) Re-derive
      arithmetic intensity per kernel: plain DTW ≈ 0.125 FLOP/byte
      (memory-bound, recorded); MSM/TWE do strictly more arithmetic per cell —
      compute their intensities and state the measurable prediction (are they
      still memory-bound? feeds R5's profile pass and guards the SIMD kill).
- [ ] **D18. Cluster validity scores.** Silhouette, Davies–Bouldin,
      Calinski–Harabasz (state exactly how a centroid-free/medoid variant is
      computed here), Dunn, inertia: formula provenance, edge cases (Nc<2,
      singleton clusters, zero distances) vs the guards in `scores.hpp`.

Exit: all 18 files exist with verdict tables; every DISCREPANCY has become a
numbered R3 finding with a test; a one-page index `docs/derivations/README.md`
maps topic → file → verdict.

## Phase R3 — Logical-mistake hunt (8.2 continues) [OPEN]

**Registered exit band (unchanged from 8.2):** DONE when two consecutive full
rounds (all lenses) produce zero new confirmed findings. Every confirmed bug:
failing test → fix → full-gate re-run → own commit. Every dead hypothesis:
recorded FALSIFIED in the run-log. R2 discrepancies enter here as findings.

Open findings first (status after R0 adjudication — update these boxes there):

- [ ] **F8 — resident≡stream parity pinned by nothing.** Commit the
      8-series/4-row-group fixture; assert resident and forced-stream FastCLARA
      runs produce identical labels, medoids, and checkpoint bytes (f64, f32,
      Soft-DTW). Until then F7's headline guarantee can regress silently.
- [ ] **F9 — Parquet reader suite absent from the canonical gate.**
      `test_io_readers` (348 assertions) is skipped under
      `DTWC_ENABLE_ARROW=OFF`; the CLI's Parquet planner lives behind
      `#ifdef DTWC_HAS_PARQUET`. Primary closure = a LOCAL Arrow-ON build dir
      running the suite for real (assert executed-assertion count ≥ floor, skip
      message absent — a skip is a pass to ctest); the CI job (in R0's
      inventory) is the secondary, operator-verified layer. This guard already
      hid one live bug (F7/D1).
- [x] **F10 — signed/degenerate D-sampling half-pinned.**
      `core::distance_sampling_weights` direct unit tests (nonnegative input
      byte-identical; selected entries exactly zero; throws on non-finite);
      all-zero → `first_unselected` for seeded `Kmeanspp_seeded` and
      `fast_pam_seeded`; the k-means++ negative-distance path. R0's inventory
      suggests most of this exists uncommitted — verify, don't assume.
      Note: R2-D13 may CHANGE the sampling rule; if so, these tests pin the
      new rule, and the old one's tests are updated in the same commit.

Remaining lenses (verbatim from 8.2 — each is one round-item; run all, round
after round, to the exit band):

- [ ] **Property/metamorphic fuzz harness** (new `tests/fuzz/` or Catch2 generators, seeds committed): invariants checked on random + adversarial inputs (NaN/Inf payloads, empty, length-1, constant series, mixed lengths, huge magnitudes, denormals) across all variant×metric×mode combinations: symmetry `d(x,y)=d(y,x)`; `d(x,x)=0`; `DTW_band ≥ DTW_full` and monotone in band; `LB_* ≤ DTW` (all LBs); `LB_Webb ≥ LB_Keogh`; `DTW_I ≤ DTW_D`; `dtwFull_eap == dtwFull_L`; prune==no-prune digit-identical (TADPole, pruned matrix); MSM/TWE triangle inequality; checkpoint save→load→identical state. Every violation is a bug or a documented, justified exclusion.
- [ ] **Cross-oracle differential test** vs aeon 1.5.0 (uv env): randomized non-degenerate pairs, all shared distances (DTW/banded/MSM/TWE/DTW_I/soft-DTW value), committed seeds, band 1e-9 rel. Disagreement = numbers-ledger entry, arbitrate with a third computation before touching code.
- [ ] **Python surface fuzz:** wrong dtypes, non-contiguous/strided arrays, zero-length series, single series, k>N, k=0/negative, unicode names, generator inputs, polars/Arrow edge shapes — every failure must be a typed exception with an actionable message, never a crash or silent wrong result.
- [ ] **Integer-width & overflow audit:** grep-driven sweep for remaining `int` index arithmetic on N²-scale quantities (the Metal int32 `pair_indices` pruning cap is KNOWN and documented — verify the documented cap actually throws/warns at the boundary rather than wrapping); `n*(n+1)/2` sites; size_t↔int narrowing in OpenMP loop indices.
- [ ] **CUDA kernel sanitizing (local RTX 4000 Ada):** run `compute-sanitizer --tool memcheck` and `--tool racecheck` over `test_cuda_correctness` + `test_cuda_lb_keogh` in `build/cuda-verify` (recipe in AGENTS.md; `MSYS_NO_PATHCONV=1` gotcha applies). M50 already sanitized the kernel-override suite — this lens extends coverage to the two named correctness suites. Zero errors = band; any report triaged like the CPU sanitizers.
- [ ] **Determinism under thread variation:** OMP_NUM_THREADS ∈ {1, 2, max} → distance matrices digit-identical; clustering results identical wherever the API promises determinism (seeded paths); any promised-deterministic path that diverges across thread counts is a bug. Record which paths are documented as non-deterministic and verify the docs say so.
- [ ] **Checkpoint/config robustness fuzz:** truncated/corrupted checkpoint files, wrong-endian/wrong-version headers, TOML configs with unknown keys/wrong types/duplicate keys, CLI flag combinations (contradictory device+method, both old and new flag names at once) — typed error with actionable message, never crash, never silent misconfiguration.
- [ ] **MEX input fuzz (MATLAB installed locally, R2024b/R2025b via `-batch`):** extend the 0.4 validation suite — wrong classes, complex, sparse, empty, cell arrays with mixed shapes, huge k, NaN-laden series → `mexErrMsgIdAndTxt` every time, zero crashes (0xc0000005 class). addpath-order gotcha applies (fresh MEX LAST).
- [ ] **Compiler-warning sweep as bug detector:** one clang build with `-Wall -Wextra -Wconversion -Wshadow` and one MSVC build with `/W4` over `dtwc/` core; triage every new warning — fix real ones, suppress documented false positives. (MSVC additionally smoke-checks the `/openmp:experimental` branch that has no local coverage.)
- [ ] **Error-path audit:** every `throw` site reachable by a test (coverage-guided: run gcov/llvm-cov on the test suite, list uncovered throw branches, add tests for the reachable ones; record unreachable ones as dead-path candidates for R4).
- [ ] **Solver edge cases:** k=N, k=N−1, k=1, duplicate series (zero distances), all-identical series, infeasible-by-construction MIP settings, iteration/node caps hit — LRCore/MIP/Benders must return certified-or-loud results, never empty/garbage clusters.
- [ ] **Mutation sweep (targeted, tests-that-test check):** scripted mutation pass over the core kernels (`dtw_kernel*.hpp`, `msm.hpp`, `twe.hpp`, `lower_bound_impl.hpp`, softmin, FastPAM swap): operator swaps (`+`↔`-`, `min`↔`max`), comparison flips (`<`↔`<=`), boundary off-by-ones, constant perturbation. Each mutant: rebuild the focused test target only, run its suite. Registered band: ≥ 90% of mutants KILLED; every survivor triaged — either a new test that kills it, or a recorded equivalent-mutant justification. Mutator script + kill ledger committed to `tests/mutation/`.
- [ ] **libFuzzer harnesses on every parser boundary** (clang `-fsanitize=fuzzer,address`): delimited/CSV reader, TOML/CLI config, checkpoint manifest + payload, mmap header/footer validation, Arrow C-Data ingest. 30 min per target minimum, seed corpora committed to `tests/fuzz/corpus/`. Fallback if libFuzzer won't link on this clang/Windows: same harnesses driven by a seeded random-byte generator as Catch2 property tests, recorded as the substitute. Every crash/hang/leak = finding.
- [ ] **Revert-probe meta-lens (do the tests pin the fixes?):** sample ≥ 10 fixes across M1–M53/F1–F7 (spread over subsystems), revert each commit in a scratch worktree, run its named regression test — it MUST go red. Any probe that stays green = the fix is unpinned = a new confirmed finding (write the missing test). Record every probe's result.
- [ ] **Cross-compiler differential:** MSVC vs clang Release digit-comparison on the conformance fixture (all variants). Registered expectation: rel ≤ 1e-12 agreement except where `-fassociative-math` reassociation legitimately differs — each exception recorded with the exact pair and magnitude; anything larger is triaged as a bug with a third-computation arbiter.
- [ ] **Precision consistency:** f32 dispatch vs f64 dispatch on committed fixtures within the R2-D17 derived analytic band; OpenMP-ON build vs OpenMP-OFF (serial) build digit-identical on the same fixtures — any divergence on a promised-deterministic path is a bug.
- [ ] **Numerical-extremes lens with a compensated arbiter:** near-`DBL_MAX` magnitudes (squared-cost overflow — extend M24's guard checks to every kernel), denormals, negative zero, catastrophic-cancellation pairs, length-8k accumulation. Arbiter: a Kahan/compensated-summation DTW oracle in the test; registered band |plain − compensated| rel ≤ 1e-12·n on random fixtures; violations are accumulation bugs or recorded reassociation effects — named either way.
- [ ] **API state-machine fuzz:** seeded random sequences of valid+invalid `Problem` setter/data/compute calls (≥ 10⁴ sequences). Invariants: typed exceptions only (no crash, no hang); after any sequence, results equal a fresh object configured to the same final state (state-independence — exercises the M25/M37/M48 seams harder than any hand-written case); rejected calls leave state byte-stable.
- [ ] **Resource/leak lens:** 1000× create→fill→cluster→destroy cycles: RSS returns to baseline (registered drift band), Windows handle count stable across mmap/checkpoint open/close cycles; LeakSanitizer on the CI ubuntu job (LSan absent on Windows — that IS the fallback, record it).
- [ ] **Concurrency stress-repeat:** 500 iterations of parallel matrix fill at randomized `OMP_NUM_THREADS` with mixed-in mid-run exceptions (metadata-failure fixtures) — zero intermittents, digit-identical accumulated results; any flake = race finding, not a re-run.
- [ ] **Serialization round-trips:** checkpoint save→load→save byte-identical; CLI TOML export→import→export identical; Python option dict → C++ → back round-trip lossless. Asymmetry = bug.

## Phase R4 — Code simplification (quality raise, ZERO behavior change) [OPEN — after R3 exit band]

Hard rule: every "no-op/refactor-only" claim requires digit-identical outputs on
a recorded case. BEFORE any edit: record a fingerprint suite — the conformance
fixture + 3 distance-matrix f64 fingerprints (20 digits) + labels/medoids across
all methods — commit it as the no-op oracle. AFTER each simplification commit:
re-run, diff digit-for-digit. Any diff = STOP, revert, record.

- [ ] **Duplication census → unify:** the soft-DTW forward paths are ALREADY unified — `soft_dtw.hpp:136-138` and `dtw_dispatch.cpp:255-259` both build `SpanL1Cost<T>` + `SoftCell<T>` and call `dtw_kernel_full` (M5 discharged for the forward path). The residual is the wrappers' divergent input guards: the free function guards empty spans (`soft_dtw.hpp:123`, returns `max()`) while the dispatch lambda has no empty check, and it validates gamma per call where the dispatch path validates once at set-variant time. Decide one contract and apply it to both. `soft_dtw_gradient`'s own `thread_local` backward matrix is a documented deliberate exclusion. Also in scope: residual duplicated decode/envelope/scratch logic; duplicated `auto`-method-resolution in `api.cpp` vs `_api.py` (single C++ source of truth, Python binds it — executes the M12 owner note).
- [ ] **Dead-code purge:** coverage report from R3's error-path audit + `cppcheck`/clang `-Wunused` sweep; grep-verify zero call sites before each deletion; deprecated 1.x shims STAY (removal is a 2.1/3.0 decision — record, don't act).
- [ ] **clang-tidy pass:** a `.clang-tidy` config exists at the repo root (committed `85f79d5`) — a conservative `-*` allowlist. Curate it (record which families were deliberately withheld and why; widen toward `bugprone-*`/`performance-*` where noise is tolerable), then RUN it over `dtwc/` core — it has never been run as a gate — and fix or explicitly suppress every warning, suppressions with a reason string.
- [ ] **Altitude/readability pass on the newest code** (it had no reviewer before merge): `api.cpp`, `one_batch_pam.cpp`, `barycenter.cpp`, `sklearn.py`, plus the F5–F7 planner/streaming code in `dtwc_cl.cpp` and `fast_clara.cpp` — flatten nesting, name magic constants, hoist repeated allocations, delete narrating comments.
- [ ] **Header & interface hygiene:** include-what-you-use pass on `dtwc/` public headers; forward-declare where possible; `const`-correctness and `[[nodiscard]]` on computational results; verify the public surface still matches `docs/api-contract-2.0.md` exactly after the pass (drift gate re-run).
- [ ] **Build-system simplification:** collapse duplicated CMake logic accumulated over Phases 0–8; record compile-time and target count before/after (advisory, shared machine).
- [ ] **Complexity census:** functions >100 lines or cyclomatic >15 in `dtwc/`; simplify the worst offenders only where the no-op oracle proves neutrality; the rest go in the census artifact for 2.1 — no speculative rewrites.
- [ ] **Header contracts:** every public function in `dtwc/` headers gets a contract comment — preconditions, postconditions, complexity, thread-safety, exception behavior (only what the signature can't express; no narration). Doxygen zero-warning; docs-site gate green.
- [ ] **Standalone-header gate:** generated one-TU-per-public-header compile test, wired into ctest so it can never rot.
- [ ] **Include-DAG check:** script asserts the module graph is acyclic with the intended layering (core ← algorithms ← api ← bindings); committed as a test.
- [ ] **Architecture doc refresh:** `.claude/design.md` updated to the as-built 2.0 state (module responsibility table, dispatch/seam map from M25/M37/M47/M48, storage formats mmap-v3/checkpoint-v2, the R2 derivations index); a new reader must locate any subsystem from this file alone.
- [ ] **Flaky-test detector:** full gate 5× consecutively; ANY intermittent failure is an R3-class finding, never a re-run-until-green. Tests >30 s named in the run-log with a reason.
- [ ] **Maintainability ledger (before/after, committed):** LOC by module, functions >100 lines, max/mean cyclomatic complexity, public-header count, max include depth, full-gate wall time. Ratchets wired into CI: warnings-as-error on `dtwc/` core; a check that fails when a NEW >100-line function appears without a decision-log entry.

## Phase R5 — Performance assurance & regression fences [OPEN — after R4; oracle binds]

The R4 no-op oracle stays binding for every commit here — "fast" never buys a
digit. Wall-clock on this shared machine is ADVISORY; HARD fences are
machine-independent counters. R2-D17's intensity model is the map: fix memory
before arithmetic, always.

- [ ] **Performance-invariant suite (HARD, CI-stable — counts, not clocks):** zero heap allocations in steady-state hot kernels (`dtw_kernel_eap`, MSM/TWE loops, softmin/adjoint inner loops, pruned-build workers, FastPAM swap inner loop) asserted with the existing allocation-counting oracle; DTW cell-count and prune-rate pinned on committed fixtures (EAP near-diagonal fixture: cells ≤ registered ceiling; TADPole fixture: exact prune count; pruned-build pair-accounting invariants from M44); scratch buffers reused across calls, never reallocated. Suite added to ctest + CI workflows.
- [ ] **Fresh advisory baselines:** the 5 canonical workloads (dense matrix N=1000 len=512; banded 10%; PAM N=2000; OneBatchPAM 50k; barycenter k-means k=3) re-timed and recorded in `.claude/baselines/` next to the Phase-5 numbers. Any regression >20% vs the 5.x baselines is investigated with a numbers ledger — never hard-FAILed locally, never ignored.
- [ ] **Hot-path profile pass:** profile the 3 heaviest workloads (VTune or ETW/WPA if present, else instrumented counters — record which); top-5 hotspots per workload in the run-log; check measured behavior against the D17 intensity predictions (a mispredicted kernel is a science finding, record it). Act ONLY where the change is provably digit-identical; everything else recorded for 2.1.
- [ ] **Link-time optimization:** build release artifacts with LTO; adopt if the conformance fixture is digit-identical and advisory timing shows no loss (record win or loss verbatim). Evaluate PGO the same way; record the decision either way.
- [ ] **SIMD stays killed** unless the D17 memory-bound analysis is overturned with new evidence. Re-opening requires citing and overturning the recorded kill evidence — never by default. (If D17 finds MSM/TWE are NOT memory-bound, that is the one legitimate door — walk through it only with a registered band and a measure-first prototype, 5.11 protocol.)

## Phase R6 — Exit gate, scale rehearsal, release readiness [OPEN — gates the 2.0.0 tag]

- [ ] **Full matrix rebuild + test:** canonical clang gate dir (llfio ON), `build/nollfio` (llfio OFF), Arrow-ON dir (from F9), CUDA build (`build/cuda-verify` recipe) — ctest 0-fail each; pytest 0-fail on a fresh `.pyd`; MATLAB floors re-run. ALL floors re-recorded verbatim in `.claude/baselines/`.
- [ ] **Scale rehearsal + capacity model (the grand-goal checkpoint).** Derive the capacity model RAM/disk/time = f(N, L, method) from the storage formulas (series O(ΣL); dense matrix N(N+1)/2 doubles; OneBatchPAM O(N·m); FastCLARA O(N+s²); mmap thresholds). Validate it: (a) against the recorded M6 point (50k series, 155.691 MiB peak RSS); (b) with ONE new synthetic run at N ≥ 500k (OneBatchPAM and streaming FastCLARA), registered HARD band = measured peak RSS within ±20% of the model, timing advisory. Then extrapolate to the 100M×8K ambition with every assumption stated (I/O bandwidth, RAM, disk) and name exactly where the local machine ends and HPC/GPU begins. Deliverable: `docs/derivations/19-capacity-model.md` + run-log. HPC execution itself is operator-gated — model and rehearse locally, do not SSH.
- [ ] Sanitizer + fuzz suites wired into CI (Claude-branch trigger, `TODO(release): remove`); the R5 invariant suite, standalone-header gate, include-DAG check, and maintainability ratchets green in the same CI run (operator triggers the run; verify the YAML locally and record).
- [ ] Mutation kill-rate ledger, revert-probe results, flaky-detector 5× record, maintainability before/after ledger all committed — the gate confirms the artifacts EXIST, not just that suites pass.
- [ ] CHANGELOG: `2.0.0rc2` section (all R3 bugs found + note that R4 is behavior-neutral); docs drift gates re-run green.
- [ ] **Windows llfio-ON wheel (known OPEN since 5.7):** apply `file(TO_CMAKE_PATH ...)` on `CMAKE_MAKE_PROGRAM` in `cmake/Dependencies.cmake`; build one local llfio-ON wheel; if an upstream quickcpplib defect still blocks, record the exact error verbatim in the Decision log, keep local wheels `DTWC_ENABLE_LLFIO=OFF` (CI Linux wheels keep llfio; `pyproject.toml` untouched), and continue — this item can NOT hold the gate.
- [ ] LESSONS.md entries for every new bug class; CITATIONS.md complete; session handoff written.
- [ ] One adversarial review pass re-reads Phases R0–R6 and confirms every checkbox with named evidence, or lists what remains — verdict goes in the Progress log. The campaign's release half closes only on CLEAN. The tag itself, PyPI, hosted CI, ARC, and Metal runtime remain USER actions — proceed to R7 without waiting.

## Phase R7 — WASM build + browser GUI "DTWC++ Playground" [OPEN — after R6 CLEAN; 2.1 feature, does not gate the tag]

**Goal:** DTWC++ running client-side in the browser — upload series, configure
variant/band/method, cluster, and see the results — as (a) a static web app
deployed with the docs site under `/playground/` and (b) one self-contained
offline HTML file. Reference implementation for architecture AND visual
identity: `c:\D\git\unibatt\glide-wasm` (user's own project; explored
2026-07-12). unibatt is Rust/`wasm-bindgen`, so its build tooling does not
transfer — DTWC++ uses **Emscripten + embind** — but its frontend architecture
(Vite + vanilla-JS tabs + Plotly + Web-Worker pool + JSON config marshalling +
progress/cancel callbacks + single-file HTML distribution) and its **entire
colour system transfer verbatim**.

**Pre-made decisions (Decision log 2026-07-12 — do not re-litigate mid-run):**

- Toolchain: Emscripten via emsdk vendored at `tools/emsdk` (gitignored — never outside repo root), `emcmake cmake -G Ninja`. Flags: `-O2`, `-fwasm-exceptions` (typed `InvalidInput`/`DeviceError` errors MUST reach JS as catchable `Error`s — Emscripten disables exception catching by default, this is the #1 silent-blocker), `--bind` (embind, RTTI on), `-sMODULARIZE=1 -sEXPORT_ES6=1 -sALLOW_MEMORY_GROWTH=1 -sMAXIMUM_MEMORY=4GB`. **NO fast-math flags** in the wasm build (native↔wasm parity band depends on it; EAP's relaxed prune threshold stays valid — relaxing only adds cells).
- Optional deps ALL OFF for the wasm target: OpenMP, HiGHS, Gurobi, CUDA, MPI, llfio (`DTWC_ENABLE_LLFIO=OFF`). Consequences: solver methods (`mip`, `lrcore`) and mmap/checkpoint are OUT of wasm v1 scope — the embind layer rejects them with the same typed loud errors as elsewhere (M47/F1 validators already enforce enum validity), never silent substitution.
- wasm32: `size_t` is 32-bit. Hard guard in the embind layer: reject N where `N*(N-1)/2 > 2^31-1` (N > 65535) with a typed error BEFORE allocation; the GUI targets N ≤ 2000 (advisory band). The R3 integer-width lens findings apply here first.
- Parallelism: NO wasm pthreads / SharedArrayBuffer (COOP/COEP headers break CDN scripts — unibatt `web/vite.config.js:13-17` precedent). Instead: N independent Web Workers, each with its own single-threaded WASM instance; the main thread compiles the `WebAssembly.Module` ONCE and `postMessage`s it (transferable) to workers which call the module factory with `{ instantiateWasm }` — copy unibatt's `multi-worker.js` / `eval-worker.js:17-23` pattern, adapted from `initSync` to Emscripten's factory API. Each worker holds its own copy of the data; matrix fill is fanned out as contiguous row-blocks.
- Frontend: Vite 6 + vanilla JS ES modules (NO React/Vue), Plotly (CDN script in dev `index.html`, inlined in the single-file build), `hyparquet` for Parquet upload; config export as TOML text matching the CLI schema so a Playground session is reproducible on the CLI. Layout: new top-level `web/` — `web/index.html`, `web/vite.config.js`, `web/src/{main.js, wasm-loader.js, workers/, tabs/, components/, styles/, data/}`, generated `web/pkg/` gitignored.
- **Colour palette (copied verbatim from unibatt; hexes are the spec — do not improvise):** UI chrome CSS variables (dark default): `--bg-body:#0f0f23; --bg-panel:#1a1a2e; --bg-input:#16213e; --bg-plot:#16213e; --text-primary:#e0e0e0; --text-secondary:#a0a0b8; --text-muted:#8888a8; --border:#2a2a4a; --border-focus:#0072B2; --accent-green:#009E73; --accent-blue:#0072B2; --accent-orange:#D55E00; --accent-cyan:#00b4d8;` Oxford identity chrome-only (never for data): `--ox-blue:#002147; --ox-blue-600:#122f53; --ox-blue-300:#49B6FF;` Light theme overrides: `--bg-body:#f5f5f5; --bg-panel:#ffffff; --bg-input:#ffffff; --bg-plot:#f8f8f8; --text-primary:#1a1a2e; --text-secondary:#4a4a6a; --text-muted:#6a6a80; --border:#d0d0e0;` Data series = Okabe–Ito colorblind-safe categorical: `['#0072B2','#D55E00','#009E73','#F0E442','#CC79A7','#56B4E9','#E69F00']` (cluster k ↦ COLORS[k % 7]). Plotly `DARK_LAYOUT`/`LIGHT_LAYOUT` and the Turbo-like sequential ramp for the distance heatmap: copy from unibatt `web/src/components/chart.js:6-28,192-213`. Theme = `data-theme` attribute on `<html>`, persisted `localStorage['dtwc-theme']`, dark default, toggle relayouts all live Plotly charts. Fonts: `--font-mono:'JetBrains Mono','Fira Code','Cascadia Code'; --font-sans:'Segoe UI',system-ui`.
- Deployment: `vite build` output published with the existing Hugo docs site under `/playground/` (additive step in `documentation.yml`); plus `scripts/build-single-html.mjs` producing self-contained `dist/dtwc-playground.html` (base64 wasm + inlined JS/CSS/Plotly — port unibatt `scripts/build-single-html.js`).
- New additive C++ public API `dtwc::warping_path` (authorized; contract governance clause satisfied — additive, documented, tested): the pair inspector needs the optimal alignment path, which no public function currently returns.

**Environment preflight (Task R7.0, record-and-continue on failure):** `node --version` + `npm --version` (if absent: portable Node under `tools/node`, gitignored); emsdk install+activate needs network — if blocked, record `[BLOCKED-ENV]` verbatim and STOP R7 only (R0–R6 conclusions unaffected); Playwright downloads Chromium — if blocked, browser E2E falls back to `vite preview` + manual smoke checklist recorded in the run-log.

### Task R7.0: Toolchain + feasibility gate

- [ ] Vendor emsdk at `tools/emsdk` (gitignore it), install + activate latest LTS; record exact emsdk/emcc versions in the run-log.
- [ ] New CMake preset/dir `build/wasm`: `emcmake cmake -G Ninja` with the flag set above; core + a minimal `dtwc_wasm_smoke` embind target compile clean. Any core source that fails to compile under Emscripten gets a minimal guarded fix (same discipline as the 5.7 llfio guards), never a fork of the file.
- [ ] Node smoke test (registered band, HARD): load the ES6 module in Node, compute `dtwFull` on a committed fixture pair — value matches the native CLI on the same fixture to rel ≤ 1e-9. Record wasm binary size (advisory band: ≤ 3 MB gzipped).

### Task R7.1: embind API layer

- [ ] `web/wasm/dtwc_embind.cpp` (own CMake target `dtwc_wasm`): `version()`; `loadSeries(Float64Array data, Int32Array lengths, int ndim)` → session handle; options setter taking one JS object mirroring `DTWOptions`/CLI names (`variant`, `band`, `msm_c`, `twe_nu`, `twe_lambda`, `wdtw_g`, `adtw_penalty`, `sdtw_gamma`, `mv_mode`, `missing`, `metric`); `distancePair(i,j)`; `distanceMatrixBlock(rowStart,rowEnd)` → Float64Array (the worker fan-out unit); `warpingPath(i,j)` → Int32Array of (i,j) pairs; `cluster(method,k,seed,maxIter)` for `{pam, onebatchpam, clara, tadpole, hierarchical}` → `{labels, medoids, cost}`; `scores(labels,medoids)` → `{silhouette[], daviesBouldin, calinskiHarabasz, dunn, inertia}`.
- [ ] Typed C++ exceptions surface as catchable JS `Error` with the exact message (HARD test — one invalid-variant and one invalid-k case); solver methods and out-of-scope combos rejected by the SAME validators as native (M47/F1/M48 seams — no wasm-special path).
- [ ] Node unit tests (vitest): every entry point, the N-guard, the error paths.

### Task R7.2: `dtwc::warping_path` (additive C++ API, native-first)

- [ ] Failing test first. Read-only path extraction for Standard/banded DTW (argmin backtrack over the DP matrix; O(n·m) memory acceptable — visualization path, not the hot kernel; document that in the header).
- [ ] Registered bands (HARD): path is a valid warping path (boundary, monotonicity, continuity); sum of local costs along the returned path == `dtwFull`/`dtwBanded` value to rel ≤ 1e-9, L1 + SquaredL2, equal + unequal lengths, banded + unbanded. Update `docs/api-contract-2.0.md` per the governance clause; CHANGELOG line; full native gate re-run (no regression).

### Task R7.3: Web app scaffold + theme

- [ ] Vite scaffold at `web/` (layout above); tab bar with 5 tabs: **Data / Distance / Clustering / Evaluation / Help**; `styles/main.css` with the palette tokens verbatim; theme toggle + persistence + Plotly relayout (unibatt `main.js:28-97` pattern); `components/chart.js` with `COLORS`, `DARK_LAYOUT`, `LIGHT_LAYOUT`, heatmap ramp.
- [ ] `wasm-loader.js`: dynamic import of the Emscripten ES6 module with a clear on-page error state if wasm fails to load (no blank page, no silent mock).

### Task R7.4: Worker pool, progress, cancel

- [ ] `workers/`: compile-once + module-transfer pattern; pool size `navigator.hardwareConcurrency` capped at 8; row-block distance-matrix fan-out; assembled matrix from the pool is digit-identical to a single-worker run (HARD, registered before the run — the wasm analogue of the M43/M44 schedule-independence guarantee).
- [ ] Progress: per-block completion messages → one monotone 0→1 progress bar (HARD: monotonicity asserted in the E2E test). Cancel: `worker.terminate()` on all workers + pool rebuild; UI returns to a runnable state after cancel (E2E-tested).
- [ ] Clustering runs in ONE worker on the assembled matrix (matrix-free methods call `distanceMatrixBlock` internally — still off-main-thread).

### Task R7.5: Data I/O

- [ ] CSV upload via `FileReader` with strict parsing that mirrors M26 semantics in JS: full-token numeric parse, textual `nan` accepted only when the missing policy allows it, malformed token → error with row/column context shown in the UI, never a silently truncated dataset.
- [ ] Parquet upload via `hyparquet`; demo dataset button — seeded synthetic generator (k well-separated warped classes, lengths 64–128, the M6 recipe) so the app demos offline with zero bundled data files.
- [ ] Export: labels CSV, medoids CSV, distance matrix CSV, and the current configuration as CLI-compatible TOML (round-trip note in Help tab: same numbers reproducible via `dtwc_cl`).

### Task R7.6: Visualizations

- [ ] Data tab: series overview plot (decimated above 5k points/series for render speed — decimation is display-only, never fed to compute).
- [ ] Distance tab: distance-matrix heatmap (sequential ramp); click a cell → pair inspector: both series + the `warpingPath` alignment drawn between them + band overlay when banded.
- [ ] Clustering tab: series coloured by cluster (`COLORS[k%7]`), medoids overdrawn bold; run controls (method, k, seed, maxIter, variant/band/params) with the same defaults as the CLI (seed 42 per M13).
- [ ] Evaluation tab: silhouette bar chart per series grouped by cluster + stat tiles for Davies–Bouldin / Calinski–Harabasz / Dunn / inertia.
- [ ] All charts theme-aware (relayout on toggle), axes/labels legible in both themes.

### Task R7.7: Tests + native-parity gate

- [ ] vitest (Node): embind surface units + THE parity gate (registered, HARD): committed fixture N=30 → full distance matrix, every entry rel ≤ 1e-9 vs the native CLI run of the same fixture (run-log with both outputs committed); seeded PAM k=3 labels + medoids IDENTICAL to native. Any parity miss = numbers-ledger entry (wasm vs native vs a third hand computation), arbitrate before touching code.
- [ ] Playwright (Chromium) E2E: load app → demo dataset → run PAM k=3 → rendered labels equal native labels; cancel mid-run → UI recovers; theme toggle → charts relayout; invalid CSV → visible row/column error. Fallback if Playwright cannot download a browser: `[BLOCKED-ENV]` + manual smoke checklist in the run-log.
- [ ] Single-file build loads from `file://` with zero network requests (HARD — assert no external fetches).
- [ ] Web-code quality gates: eslint + prettier configs committed and zero-error in CI; no compute on the main thread (E2E asserts the UI stays responsive — long-task budget ≤ 100 ms — while a matrix build runs in the pool); no leftover debug logging in production paths.

### Task R7.8: Distribution, CI, docs — R7 exit gate

- [ ] `scripts/build-single-html.mjs` → `dist/dtwc-playground.html` (advisory band ≤ 8 MB); `vite build` static output wired into `documentation.yml` publishing under `/playground/` (additive job step; docs site gate stays green).
- [ ] CI job building the wasm target + running the vitest parity gate (Claude-branch trigger, `TODO(release): remove`).
- [ ] Docs: Playground page in the Hugo site (what runs locally in your browser, what is out of scope: solvers/GPU/HPC/mmap, the N ≤ 2000 advisory envelope); README badge/link; CHANGELOG entries; LESSONS for every Emscripten gotcha hit; session handoff.
- [ ] Exit gate: full native gate re-run 0-fail (R7 must not perturb native), wasm parity gate green, E2E green (or recorded `[BLOCKED-ENV]` fallback), run-logs committed. One adversarial review pass confirms every R7 checkbox with named evidence — verdict in the Progress log.

---

## Killed ideas (do not reopen without overturning the recorded evidence)

- **FastDTW:** verified trap ("much slower than exact DTW" — Wu & Keogh TKDE 2022).
- **BanditPAM/++:** dominated by FasterPAM on precomputed matrices (5.1).
- **Elkan/triangle pruning on DTW:** invalid — DTW is not a metric.
- **ONNX export; R/Julia bindings:** no sensible story / deferred (native competitors saturate).
- **The 2023 custom OSLP solver:** retired (`f7064b3`); the "third solver" is PDLP-as-arbiter, not a revival.
- **SIMD before memory:** DTW ≈ 0.125 FLOP/byte, memory-bound; 5.11 measure-first verdict stands. (R2-D17 is the only door — see R5.)
- **"≥25% fewer full DTW calls on an exact matrix" via LBs:** UNACHIEVABLE — an exact matrix needs every DTW; LB early-abandon recomputes abandoned pairs (`Pruned` strategy is a PESSIMISATION for exact matrices). Exact-matrix DTW-work reduction = EAP cell-pruning / TADPole pair-skip only.
- **≥10× swap speedup from FastPAM decomposition on a cached matrix:** falsified — both variants do N² lookups/iter; decomposition removes only cheap O(k) arithmetic (measured 2.3–8.1×).
- **LB-pruning inside PAM/MIP/LRCore:** not admissible — those consumers read the whole matrix.
- **PDLP as production p-median solver:** falsified — matrix-free Kelley dominates on either device (945× CPU / 126× GPU at N=400); PDLP remains a cross-validation arbiter only.
- Full kill-context lives in the archive (`### Explicit rejections`, Phase 5) and `.claude/LESSONS.md`.

## Binding decisions (digest — full prose in the archive §Decision log; append NEW decisions here)

- 2026-07-06: `default_data_t` = double; Float32 explicit opt-in. `MetricType::L2` is a real multivariate L2. No-silent-fallback promoted to Global Constraint. Local benchmarks ADVISORY (shared machine).
- 2026-07-10: llfio pinned `b17613f…`; codecov + all 36 workflow actions SHA-pinned; Arrow 19.0.1 SHA-256 enforced; stale tracked MEX binary removed; HiGHS bundled in wheels, Gurobi external; production publication = manual explicit-go only.
- 2026-07-10 (M-series authorizations, all still binding): soft-DTW unification boundary (M5→R4); device-aware Tier-1 `auto` (M12); mmap v1/v2 invalidated, v3 authenticated identity + crash-safe publication + payload integrity (M14/M15/M53); cross-language Tier-1 seed 42, `42+i` restarts, Lloyd/MIP included (M13/M17); estimator literal fit/predict semantics (M23); barycenter real-work convergence + finite-state rejection (M24); dense-cache semantic binding + matrix-free dispatch reconciliation (M25/M37); transactional MIP publication (M33); DTW variant parameter domains (M34); Soft-DTW helper domain + denormal scaling (M46); HPC submission envelope/grammar/identity hardening (M28/M31/M32); dense checkpoint v2 identity (M49).
- 2026-07-10 (H4): MATLAB Tier-1 keeps the original six-method set in 2.0 (parity = 2.1); C++ HPC is a throwing beta boundary (no transport, no silent CPU fallback; Oxford ARC/2.1 owns enablement).
- 2026-07-12 (F7): truthful CLI RAM policy — `--ram-limit` is the Parquet series materialisation cap (hard-errors elsewhere); `--mmap-threshold` selects distance storage; matrix-free runs emit labels/medoids + binary checkpoint, dense CSVs only when a matrix exists.
- 2026-07-12 (F4): seeded RNG schedule versioned `portable-v1` (identical across MSVC STL/libstdc++); unseeded Tier-2 mutable-engine contract unchanged.
- 2026-07-12: Phase 9/R7 authorized (Emscripten+embind, worker pool no pthreads, unibatt palette verbatim, `dtwc::warping_path` additive API). `[BLOCKED-ENV]` record-and-continue promoted into the execution contract.
- 2026-07-13 (F7 re-review): D1 guard placement (outside `#ifdef DTWC_HAS_PARQUET`) is load-bearing; D2 CUDA/auto rejection recorded as breaking. F8–F10 opened.
- 2026-07-23: PLAN v2.0 adopted (this file); prior plan archived verbatim; AGENTS.md created as the Codex working-rules SSOT.

## Progress log (append-only; older entries in the archive)

- 2026-07-23 (Fable): PLAN v2.0 written on user directive ("clean the repo,
  re-derive the maths, find logical and performance mistakes, reach the grand
  goal — detailed but flexible, Codex runs non-stop"). Campaign R0–R7 defined;
  R0 = adjudicate the uncommitted 2026-07-20 work found in the tree (F9/F10 +
  scholarly edits, unverified). AGENTS.md created. Old plan → archive.
- 2026-07-23 (R0/F10): Diff classification completed. F10 repaired and closed
  in `20b894d`: selected as well as unselected non-finite sampling inputs now
  fail closed; the direct seam and seeded/unseeded signed/degenerate routes run.
  Registered gate PASS: 114/114 CTest targets, zero failed, exactly six
  capability skips. Evidence:
  `.claude/baselines/2026-07-23-f10-sampling.md`.
- 2026-07-23 (R0/integer width): Repaired the interrupted FastPAM edit in
  `f8ff7d3`. All three public entries now reject point counts above the
  int-indexed result ABI before effects, and supported kernels use one checked
  narrowing instead of repeated casts from widened loop counters. Deliberate
  red: 2/18 assertions failed; repaired focused suites: 258/258 and 76/76;
  canonical gate: 114/114, zero failed, exactly six capability skips. Evidence:
  `.claude/baselines/2026-07-23-fast-pam-index-width.md`.
