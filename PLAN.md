# DTWC++ — Research & Release Campaign (PLAN v2.0)

> **For Codex (autonomous, single continuous run):** this file is the mission. Read
> `AGENTS.md` first (the working rules), then this file top to bottom, then start at
> the earliest OPEN phase and do not stop. Never wait for an operator. Improvise where the evidence
> justifies it — this plan states goals, invariants, and suggested routes, not a
> script — but record every departure as a Decision-log entry the moment you make it.
>
> The prior plan (2.0-refactor Phases 0–9 with the complete task history, full
> decision-log prose, and progress log) is archived **verbatim** at
> `.claude/PLAN-archive-2026-07-20-phases0-9.md`. Before re-opening ANY idea, search
> that archive plus `.claude/LESSONS.md` and the Killed-ideas section below.
> Re-opening a killed idea requires explicitly overturning the recorded kill
> evidence, never forgetting it.

**Status (2026-07-24):** 2.0.0rc1 release state committed (not tagged or
published). Refactor Phases 0–7 CLOSED. Phase 8: 8.0 + 8.1 CLOSED (149
protocol-clean commits `8debf1d..eda1b92`); 8.2 findings F1–F7, F9–F10, and
the sanitizer gate CLOSED; **F8 CLOSED**. Phases R0–R1 CLOSED; R2 active with
D1 CLOSED; R3 active. F11's archive pin is committed, but its hand-written
parser closure is FALSIFIED and routed to F36. F12's local CUDA repair is
committed and verified, but real-Metal execution remains `[BLOCKED-ENV]`;
**F14 CLOSED** and F15 is active.
The final **2.0.0 tag gates
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

## Phase R0 — Adjudicate the 2026-07-20 in-flight work [CLOSED — 2026-07-23]

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
- [x] Build + run the full canonical gate (floor: **114/114, 0 failed**, 6
      capability skips — `.claude/summaries/handoff-2026-07-13-f7-streaming.md`).
      Run the new/changed test suites explicitly and quote their output.
- [x] For the F9 workflow change: you cannot run GitHub CI locally — verify the
      YAML by schema/actionlint if available, verify the job obeys the
      "assert-the-subject-RAN" lesson (greps the test binary's own output for
      executed assertions, not just ctest green), and mark the CI run itself as
      an operator-triggered verification in the Decision log. If the local
      equivalent (an Arrow-ON build dir running `test_io_readers` for real) is
      constructible, BUILD IT — that, not CI, is the primary F9 closure (see R3).
- [x] Verdict per change: KEEP (gate green, contract met) → commit as its own
      conventional commit crediting the finding it closes; REPAIR (close but
      defective) → fix, then commit; REVERT (wrong or unverifiable) → revert
      with a Decision-log line naming why. No change may stay uncommitted.
- [x] Update the F8/F9/F10 checkboxes in R3 to reflect what actually closed.

## Phase R1 — Repository cleanse & record reconciliation [CLOSED — 2026-07-23]

Non-behavioral hygiene: make the repository's *record* as trustworthy as its
code. Nothing here may change program behavior (no-op oracle not required since
no core code changes — but if any item does touch code, it moves to R4's rules).

- [x] **TODO.md full reconciliation.** `.claude/TODO.md`'s audit list is a
      2026-07-06 snapshot; Phases 4–8 closed an unknown subset without editing
      it (the 2026-07-20 note reconciled exactly one entry). Verify all ~30
      entries against the current tree: each becomes CLOSED-BY (commit/task),
      STILL-OPEN (→ becomes an R3 finding), or NOT-REPRODUCIBLE (evidence
      quoted). Rewrite the file to the reconciled state.
- [x] **Docs truth audit.** Every claim in README.md, docs site pages, and
      `docs/api-contract-2.0.md` traces to an artifact (test, baseline run-log,
      citation) or is corrected. Run the existing drift gates
      (`check_docs_contract.py --cli <fresh dtwc_cl>`, docs internal-link gate)
      and quote results.
- [x] **`.claude/` record hygiene.** LESSONS.md and CITATIONS.md: dedupe,
      verify file:line references still hold after Phase 8's churn (spot-check,
      fix stale ones), keep every lesson. Add one current/supersession freshness
      header to UNIMODULAR.md. `.claude/MISSING.md` / `READ.md` were retired by `0449f7c`;
      do not recreate them to satisfy obsolete wording.
- [x] **Tracked-file junk census.** Find tracked files that should not be
      tracked (stale binaries, generated artifacts, orphaned fixtures) —
      grep-verify zero references before each removal; `.gitignore` audit
      (build dirs, `tools/emsdk`, `web/pkg` when R7 arrives). Do NOT delete
      untracked local build directories — inventory them in the handoff with
      which recipe each serves; disposal is an operator decision.
- [x] **CHANGELOG structure check.** Unreleased vs rc1 sections coherent; every
      Phase-8 breaking change present (the F7 pair is: `--ram-limit` hard-errors
      on non-Parquet input; `--device cuda` rejected for matrix-free FastCLARA
      incl. `--method auto` above 5,000 series).
- [x] **Branch state note.** Branch `Claude` is far ahead of `main`; merging is
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

- [x] **D1. DTW recurrence + Sakoe–Chiba band.** Optimal substructure; boundary
      conditions; band feasibility (`band ≥ |n−m|` for a nonempty path);
      monotonicity `DTW_band ≥ DTW_full` and monotone-in-band; L1 vs squared-L2
      local costs and what "distance" each yields (squared form is not a metric
      — say so). Conformance: `dtw_kernel.hpp`, `dtwBanded`. CPU conformance is
      **CONFIRMED** by `9f78212` and derivation commit `cf5b9d8`; CUDA geometry
      is **CONFIRMED** by `4583443`, while exact real-Metal parity remains
      **DISCREPANCY** F12.
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
      caps the speedup at removed-arithmetic only (the advisory table records
      2.95–8.06×, non-monotone; the memory-bound explanation remains
      **[inferred]** pending D17 counters).
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
      arithmetic intensity per kernel: reconstruct the earlier plain-DTW
      ≈0.125 FLOP/byte estimate rather than treating it as a PMU result;
      MSM/TWE do strictly more arithmetic per cell—compute their intensities
      and state measurable cache/bandwidth predictions for R5.
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

- [x] **F8 — resident≡stream parity pinned by nothing.** Commit `1df77fc`
      tracks the SHA-pinned 8-series/4-row-group fixture and a non-skippable
      real-CLI gate. Six resident/forced-stream FastCLARA runs cover f64, f32,
      and Soft-DTW: route markers pass 12/12 and all nine labels, medoids, and
      checkpoint pairs are byte-identical. The fresh Arrow-ON suite passes
      115/115; canonical Arrow-OFF remains 114/114 with its six capability
      skips. Hosted Ubuntu execution remains operator-owned and is not claimed.
- [x] **F9 — Parquet reader suite absent from the canonical gate.**
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
- [ ] **F11 — example-project dependency integrity is outside the pin gate.**
      `examples/cpp/example_project/CMakeLists.txt` downloads a mutable branch
      archive without `URL_HASH`, and `check_supply_chain_pins.py` does not scan
      it. First gate: a repo-wide checker fails on that exact fixture while all
      currently pinned main dependencies continue to pass.
      **Partial 2026-07-24:** `653b0e6` pins the exact 2.0.0rc1 commit archive
      and SHA-256; a fresh no-override configure downloaded the registered
      4,928,286-byte artifact, built the library and real example consumer, and
      the gate at that commit passes 39 actions / 7 archives / 25 manifests.
      F13 later added one tracked CMake gate and registered the current total
      of 26 without changing the seven archive identities.
      The final fail-closed parser band is FALSIFIED by the two named probes in
      F36, so this checkbox remains open.
- [ ] **F12 — cross-backend fixed-band geometry and no-path sentinel diverge.**
      The inherited CUDA used endpoint-scaled `slope`/`window` corridors and
      signed `band+1`; both FP32 GPU APIs widened `FLT_MAX` instead of returning
      the public `DBL_MAX` sentinel. First CUDA gate: on the local RTX, drive a
      non-degenerate unequal-length fixture below, at, and above `|n-m|`, plus
      `INT_MAX`,
      against the independent full-matrix oracle and require exact geometry,
      cost, and sentinel parity. First Metal gate: the same fixture on a real
      device, including exact double sentinel identity. Source inspection
      cannot close either executable backend.
      **Partial 2026-07-24:** `4583443` repairs all six CUDA kernels and both
      GPU public-copy paths. The immutable local RTX gate passes 515 assertions
      / 6 F12 cases and the unfiltered CUDA binary passes 7,827 / 61 with no
      skip; canonical and llfio-OFF gates pass 115/115. Metal's no-LB source
      repair and permanent pairwise/K-vs-N gate passed independent inspection,
      but real-Metal execution is `[BLOCKED-ENV]` on this Windows host. Leave
      F12 open and continue at F13; evidence:
      `.claude/baselines/2026-07-24-f12-gpu-fixed-band-parity.md`.
- [x] **F13 — nearest-medoid assignment has behaviorally unpinned copies.**
      FastPAM, CLARANS, and resident/f64/f32 FastCLARA retain separate scans.
      First gate: digit-identical assignments/objectives on adversarial ties and
      non-finite rejection before R4 may consolidate anything.
      **Registered 2026-07-24:** source audit found six algorithm assignment
      bodies plus the public Lloyd `Problem::assign_clusters` body. The gate
      now pins first-slot ties, ordered IEEE-754 objectives, exact finite
      `DBL_MAX`, non-finite rejection, CPU-f32 sentinel translation, both
      CLARANS copies, resident f64/f32, and real-CLI streamed f64/f32. Full
      consolidation remains R4-owned. Evidence:
      `.claude/baselines/2026-07-24-f13-medoid-assignment-contract.md`.
      **Closed 2026-07-24:** `62c6f26` normalizes the CPU-f32 public sentinel;
      `1eb8609` enforces explicit best-result presence, finite assignment and
      candidate distances, strict first-slot ties, ordered finite objectives,
      deterministic parallel failure handoff, and transactional Lloyd
      publication. `cb11c90` repairs exact-base-confirmed stale portable-RNG
      Python oracles without weakening the iteration-cap discriminator;
      `3784251` registers the added CMake manifest; `3783b12` supplies every
      Windows Arrow-linked CTest with its shared runtimes. Nine mutation
      classes fail. Final gates: canonical 116/116, llfio-OFF 116/116, Arrow
      118/118 with both real-CLI subjects, and fresh-extension Python
      1010 passed / 12 skipped. Full scan consolidation remains R4-owned.
- [x] **F14 — four CSV emitters have no byte-parity contract.** Pin locale,
      precision, signed zero, and non-finite behavior across dense stream,
      mmap stream, and visitor paths; only then may R4 remove duplication.
      **Registered 2026-07-24:** native ASCII output is general binary64 at
      `max_digits10`, comma-delimited, LF-only, locale/state independent,
      preserves raw signed zero, leaves uncomputed NaN empty, and rejects
      computed infinities before output. The hand-written 3x3 oracle is 83
      bytes with SHA-256
      `7754CFF0231360D60A69B034CA5136E88EEFE8B53A6509706D99071C436F3813`.
      Focused dense/mmap/visitor/print/Result gates, real resident/mmap CLI
      parity, a native `Result::save` helper, source audit, 13 mutation classes
      and 21 executions, manifest 27, and exact final build floors are binding.
      R4 still owns consolidation. Evidence:
      `.claude/baselines/2026-07-24-f14-csv-wire-format.md`.
      **Closed 2026-07-24:** `e5bfd20` freezes the native bytes while retaining
      four independent output loops. Focused llfio-ON/OFF gates pass 144/13
      and 88/10 assertions/cases; resident CLI, mmap CLI, and native
      `Result::save` meet the public contract; all 21 registered mutation
      executions fail. Supply-chain inventories remain 39 actions, 7
      archives, 1 Arrow pin, and 27 CMake manifests. Final canonical,
      llfio-OFF, and Arrow gates pass 118/118, 118/118, and 120/120 with
      6/9/8 capability skips. R4 retains formatter consolidation and F37
      retains cross-language parity.
- [ ] **F15 — benchmark/test generators and CPU oracles are fragmented.**
      The historical claim that eight copies were byte-identical is falsified:
      ranges and shapes differ. Inventory intentional variants and pin seeded
      bytes/oracle values before extracting any shared test utility.
      **Registered 2026-07-24:** share only the two exact generator families
      and dense symmetric production-CPU traversal: benchmark scalar/per-row
      `mt19937` with `[-1,1]`, accelerator row-major continuous `mt19937` with
      `[-10,10]`, and zero-diagonal mirrored `N*N` assembly. Preserve packed
      LB, timing-only MPI, float/range variants, caller-owned/Gaussian/NaN draw
      schedules, and independent mathematical arbiters. Exact coherent Windows
      Clang-relaxed, Windows MSVC-precise, and Linux-libstdc++ raw-byte/oracle
      fingerprints, permanent non-skipping source reachability, 13 mutation
      executions, benchmark compilation, real CUDA baselines, and final
      119/119, 119/119, 121/121 build floors are binding. Evidence:
      `.claude/baselines/2026-07-24-f15-test-support.md`.
- [ ] **F16 — CMake presets encode one developer machine and a stale floor.**
      `CMakePresets.json` hardcodes a Windows LLVM path and declares CMake 3.21
      while the root requires 3.26. First gate: portable clean configure probes
      plus a metadata check that rejects any future floor drift.
- [ ] **F17 — CLI `--resume` reads and discards clustering state.**
      `dtwc/dtwc_cl.cpp:1398-1405` loads a binary `ClusteringResult` into the
      block-local `ckpt_result`, prints its metadata, and has no later consumer;
      the ordinary clustering path then runs and overwrites the automatic
      checkpoint. First gate: drive the real CLI from a valid binary checkpoint
      with deliberately distinguishable labels, medoids, cost, and iteration
      count; the inherited CLI must fail an assertion that resumed state affects
      the result rather than merely producing the verbose “Loaded checkpoint”
      line. Define the supported continuation semantics before repair—never
      silently relabel a read-and-discard operation as resume.
- [ ] **F18 — MATLAB estimator accepts routing options that do not reach its
      `Problem`.** `DTWClustering.Metric` is stored but never read by `fit`;
      `Device` updates global `Env`, but each repetition creates a default
      `Problem` whose `distance_strategy` remains `Auto`, and `fast_pam` does not
      consult `Env` (`bindings/matlab/+dtwc/DTWClustering.m:53-155`,
      `dtwc/Problem.cpp:778-787`). First gates: a non-degenerate fixture whose
      L1 and squared-L2 medoids/cost differ must make the estimator match the
      corresponding explicit `Problem` route, and a fresh CUDA-enabled MEX must
      prove `Device='gpu'` reaches CUDA dispatch rather than merely validating
      the global device. The current constructor-only parity test is not a gate.
- [ ] **F19 — the frozen `Problem` encapsulation/accessor cleanup is
      incomplete.** Configuration and result fields remain publicly mutable;
      promised C++ `last_iterations()`, `set_output_folder(path)`, and `name()`
      accessors are absent; MATLAB still performs binding-side result writeback
      after core algorithms already write the same state
      (`dtwc/Problem.hpp:206-234`, `bindings/matlab/dtwc_mex.cpp:326-330,
      1142-1203`). First gate: a compile-time contract fixture must fail on the
      three missing canonical accessors and a source/API guard must reject raw
      mutation of fields designated private by the frozen contract; separately,
      deleting each redundant MATLAB writeback in a probe must leave the
      returned and stored labels/medoids digit-identical.
- [ ] **F20 — `Problem::set_storage_policy` is an advisory no-op for storage
      routing.** The setter only validates and stores an enum
      (`dtwc/Problem.hpp:339-345`); heap/mmap selection is owned independently
      by `DataLoader` (`dtwc/DataLoader.hpp:200-203,276-323`). This does not
      satisfy the frozen §2.1/§6.3 promise that the `Problem` setting overrides
      local series storage. First gate: derive and register which subsequent
      load/set-data operation the setter governs, then force Heap and Mmap on a
      payload above a deterministic threshold; backing mode must differ while
      series bytes and downstream distances remain identical. The inherited
      setter must fail by leaving both routes unchanged.
- [ ] **F21 — four frozen C++ snake_case entry points are absent.**
      `DataLoader` exposes only `startColumn`/`startRow`
      (`dtwc/DataLoader.hpp:124-157`), and `settings::paths` exposes only
      `setDataPath`/`setResultsPath` (`dtwc/settings.hpp:74-86`), despite the
      frozen rename table promising `start_column`, `start_row`,
      `set_data_path`, and `set_results_path` with deprecated old-name shims.
      First gate: a public-header compile fixture calling all four canonical
      spellings must fail on the inherited tree, then pass while the four legacy
      calls still compile and produce identical loader configuration/path state.
- [ ] **F22 — compatibility aliases do not obey the frozen deprecation
      policy.** C++ `maxIter`/`N_repetition` remain unannotated public fields;
      most Python aliases forward without `DeprecationWarning`; MATLAB legacy
      properties/functions forward without the promised loud notice
      (`dtwc/Problem.hpp:208-209`, `python/src/_dtwcpp_core.cpp:770-771,
      888-893,1143-1193`, `bindings/matlab/+dtwc/Problem.m:24-36,93-119,
      327-345`). First gate: table-drive every retained alias—C++ compile probes
      require a deprecation diagnostic, Python uses
      `pytest.warns(DeprecationWarning)` exactly once per call, and MATLAB
      captures one stable warning identifier/message—while asserting canonical
      names stay silent and results remain identical.
- [ ] **F23 — Python lacks the frozen binary result-checkpoint bindings.**
      Its module exposes `CheckpointOptions` and directory save/load only
      (`python/src/_dtwcpp_core.cpp:1105-1125`), while MATLAB delivered
      `save_binary_checkpoint`/`load_binary_checkpoint`
      (`bindings/matlab/dtwc_mex.cpp:956-969`). First gate: import both names
      from the freshly rebuilt extension, round-trip a non-degenerate
      `ClusteringResult` with distinct labels, medoids, cost, and iteration
      count, assert field equality and malformed/missing-file typed errors, and
      compare the emitted bytes with the C++ reader. The current extension must
      fail at import before implementation.
- [ ] **F24 — Python HPC failures bypass the frozen device-error contract.**
      `python/dtwcpp/_hpc.py:395-405,445-456` raises wrapper-specific
      `RuntimeError` messages, while Python intentionally defers HPC credential
      validation in `python/dtwcpp/__init__.py:193-204`; §5/§6 requires
      `dtwcpp.DeviceError` and the pinned Env messages. First gate: drive the
      public Python `device('hpc')`→`cluster(...)` route through isolated
      missing-file, first-missing-key, and mocked authentication-failure
      fixtures; each must raise `DeviceError`, match the registered message
      byte-for-byte, perform no local clustering fallback, and avoid a real
      network call. The inherited wrapper must fail type and text assertions.
- [ ] **F25 — public invalid states still rely on build-dependent
      assertions.** `Problem::get_name` and `Problem::p_vec` guard view mode
      with `assert(!data.is_view())` (`dtwc/Problem.hpp:250-259`), contrary to
      the frozen typed-error rule; public matrix/store accessors retain further
      bounds assertions. First gate: inventory every assertion reachable from a
      public entry, then run the same invalid view-mode/index fixtures in Debug
      and Release. Each must raise the registered typed exception with stable
      text—never abort in Debug or enter unchecked/undefined behavior in
      Release. Internal algorithm invariants may remain assertions only when a
      public validator makes them unreachable.
- [ ] **F26 — Python `Problem.set_view_data` is named as a view but copies into
      owning storage.** Its binding converts Python input to
      `std::vector<std::vector<double>>`, builds an owning `Data`, and only then
      calls the C++ view setter (`python/src/_dtwcpp_core.cpp:872-879`); true
      non-owning spans remain C++/CLARA-internal. First gate: bind a contiguous
      ndarray through the public Python method, mutate a non-degenerate element
      in the source, and require `Problem.series()`/a recomputed distance to
      observe that mutation while the Problem keeps the Python owner alive.
      Non-contiguous, readonly, dtype, and lifetime cases must be explicit and
      typed. The inherited binding must fail the aliasing assertion.
- [ ] **F27 — GPU LB_Keogh uses L1 excess under squared-L2 DTW.** CUDA and
      Metal always sum raw envelope excess (`dtwc/cuda/cuda_dtw.cu:872-892`,
      `dtwc/metal/metal_dtw.mm:945-953`) even when the distance kernel uses
      squared local costs, so threshold pruning can discard a pair whose true
      squared-DTW cost is below threshold. First gate on each real backend:
      series `{0}` and `{0.5}`, band 0, squared L2, LB enabled, threshold 0.3;
      the pair must survive and equal 0.25 rather than be pruned by the current
      L1 bound 0.5. Force Metal Wavefront so its LB stage actually runs.
- [ ] **F28 — Metal permits a narrow LB envelope for full DTW.** When DTW is
      unbanded and `lb_envelope_band` is unset, Metal chooses roughly
      `max_L/10`; it also accepts an explicitly narrower window
      (`dtwc/metal/metal_dtw.mm:1497-1502`). Such a bound is not admissible for
      the larger/full warping window. First real-Metal gate (forced
      Wavefront): `x={0,0,0,0,1,1,1,1,1,1}`,
      `y={0,0,0,0,0,0,1,1,1,1}`, full DTW, envelope band 1, threshold 0.5.
      True DTW is zero, so the pair must not be pruned. Reject or widen any LB
      envelope that does not cover the actual DTW window before dispatch.
- [ ] **F29 — GPU LB_Keogh truncates unequal lengths without a validity
      proof.** Both kernels compare equal-index prefixes only through
      `min(Li,Lj)` (`dtwc/cuda/cuda_dtw.cu:861-892`,
      `dtwc/metal/metal_dtw.mm:934-953`), while backend DTW band geometry is
      length-aware and already differs under F12. First CUDA gate:
      `a={0,0.25}`, `b={0,0,0.25}`, band 0; the current prefix L1 bound is
      0.25 while slope-adjusted banded DTW is 0, so threshold 0.1 must not prune
      the pair. A real-Metal result and a third mathematical arbiter are
      required before claiming unequal-length support there. Until proved,
      pruning must reject or bypass unequal-length pairs loudly.
- [ ] **F30 — explicit GPU option requests silently degrade.** Metal disables
      requested LB on regtile/banded-row and after LB-buffer allocation failure
      (`dtwc/metal/metal_dtw.mm:1480-1490,1523-1537`); CUDA ignores LB when
      `band<0` (`dtwc/cuda/cuda_dtw.cu:1563-1567`); unsupported kernel
      overrides silently select Auto (`dtwc/enums/KernelOverride.hpp:8-10`);
      Metal FP64 quietly becomes FP32 unless verbose
      (`dtwc/metal/metal_dtw.mm:1262-1264`). The existing Metal LB test
      explicitly expects the banded-row no-op. First gate: table-drive every
      explicit option across supported/unsupported paths and the injected
      allocation seam. Each request must execute, raise a typed error, or return
      universally inspectable fallback metadata—never an unobservable no-op.
- [ ] **F31 — operational GPU failures escape the public device-error
      taxonomy.** `Problem` raises `DeviceError` for unavailable/uncompiled and
      empty-result cases, but Metal allocation/launch failures throw
      `std::runtime_error` and pass through `Problem::fill_distance_matrix`
      (`dtwc/Problem.cpp:817-820,861-911`,
      `dtwc/metal/metal_dtw.mm:1274-1294,1564-1599,1623-1628,1725-1737`).
      First gate: inject one allocation failure and one command/kernel failure
      through the real public `Problem` route; both must raise `DeviceError`
      with backend/action context and preserve prior Problem state. Direct
      backend helpers may retain lower-level exceptions only if the public
      boundary translates them deterministically.
- [ ] **F32 — two accepted multivariate configurations run scalar logic over
      the flat interleaved buffer.** `validate_problem_distance_semantics`
      rejects multivariate MSM/TWE and unsupported independent-mode products,
      but accepts dependent Soft-DTW and `MissingStrategy::Interpolate`
      (`dtwc/core/distance_semantics.hpp:60-91`). Their resolvers ignore
      `Data::ndim`: Soft-DTW builds `SpanL1Cost` over `x.size()`/`y.size()`, and
      Interpolate transforms each flat scalar vector before calling scalar
      `dtwBanded` (`dtwc/core/dtw_dispatch.cpp:57-65,237-259`). This silently
      permits warping between adjacent channels as if they were timesteps.
      First gate: bind a non-degenerate `ndim=2` fixture through the public
      `Problem` path for each configuration and require `InvalidInput` before
      distance allocation or state mutation. A later channel-aware
      implementation may replace that rejection only after an independent
      per-channel oracle defines the recurrence and the same gate pins it.
- [x] **F33 — CPU banded DTW used an endpoint-scaled corridor instead of the
      fixed Sakoe–Chiba window.** On base `be0069b`, the preregistered public
      unequal-length fixture returned 8 at band 0 instead of the finite no-path
      sentinel and 3 at band 2 instead of 5; both matched the registered
      slanted-window fingerprint. Commit `9f78212` fixes the shared CPU kernel
      and public singleton/MV/AROW bypasses. Independent full-matrix DP and
      exhaustive path enumeration agree, the focused gates report 70/70 and
      11/11 assertions, and the canonical gate is 114/114 with exactly the six
      capability skips. Evidence:
      `.claude/baselines/2026-07-23-r2-d1-dtw.md`.
- [ ] **F34 — the supply-chain gate omits non-action workflow acquisitions and
      weaker CMake repository pins.** The current script does not classify the
      executed `llvm.sh`, two installed MS-MPI packages, the tag-only CUDA
      container, flow-style/composite/docker action references, the two CPM
      bootstraps, or four `GITHUB_REPOSITORY + VERSION` dependencies. First
      gate: inventory every tracked CMake/workflow remote acquisition, require
      zero unclassified entries, mutation-test each syntax class, and either
      cryptographically anchor each executable input or record a narrow,
      fail-closed policy exception. Package-manager resolution and operator
      transport are separate policy classes, not silently accepted inputs.
- [ ] **F35 — the CUDA workflow's CMake requirement is shell redirection.**
      `.github/workflows/cuda-mpi-detect.yml` runs
      `pip3 install cmake>=3.26` unquoted, so POSIX shell grammar passes `cmake`
      to pip and redirects stdout into `=3.26`. First gate: an executing
      fake-pip shell fixture must reproduce that inherited argv/file side
      effect; after repair it must receive the single literal `cmake>=3.26`
      argument and create no redirection file.
- [ ] **F36 — the hand-written CMake URL-pin parser is not fail-closed.**
      Two independent final audits after the capped F11 pivots left all live
      counters green for (a) `CUSTOM_CACHE_KEY` selecting an existing
      `CPM_SOURCE_CACHE` directory and skipping fetch/hash, and (b) quoted
      `"DOWNLOAD_\<newline>COMMAND"`, which CMake normalizes to the overriding
      `DOWNLOAD_COMMAND` while the scanner retains a newline. First gate:
      commit both exact production-equivalent mutations and require them red.
      Do not add a third lexical deny-list patch. Replace the design with
      CMake-interpreter-normalized arguments or a declarative pinned-archive
      helper whose closed signature constructs the CPM call. Then replay the
      63-case focused suite, the fresh no-override example fetch/build, both
      HiGHS option configurations, and an independent audit. F36 owns only
      URL-declaration grammar/override closure; F34 retains wider acquisition
      classes.
- [ ] **F37 — the frozen cross-language `Result::save` byte contract is
      violated by language-owned emitters.** The contract promises identical
      corresponding C++/Python/MATLAB/CLI files
      (`docs/api-contract-2.0.md:194-200,728-734`), but Python delegates the
      matrix to NumPy's default `savetxt` and MATLAB delegates to
      `writematrix`. The registered F14 3x3 values make real Python
      `Result.save` write 212 Windows bytes, SHA-256
      `E894009679F58362E6CCBBB821C5DCB8BF2EAAC4A21E576AD77350BBA261A23F`,
      beginning `nan,-0.000000000000000000e+00,...`, versus the 83-byte native
      oracle with an empty sentinel field. F37's first gate, after F14 has
      frozen native bytes: drive fresh C++, Python, MATLAB, resident CLI, and
      mmap CLI on one non-degenerate result and compare all four corresponding
      files byte-for-byte. Python is runtime-confirmed; MATLAB bytes remain
      `[inferred]` until the fresh MEX gate. F37 owns cross-language emitters;
      F14 remains the four-body native formatter contract.

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
- [ ] **SIMD stays killed** unless D17 plus the R5 profile establishes a
      correctness-complete route and a measured opportunity. Re-opening
      requires citing the invalid old dispatch and the conflicting
      route-specific timings, then overturning them with a registered
      measure-first prototype.

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
- **SIMD before correctness + measurement:** the old dispatched route ignored
  bands/variants and was removed; 1.29× was an operation-count estimate, while
  route-specific SIMD timings disagreed. No PMU artifact proves a universal
  memory-bound ceiling. R2-D17 is the only door—see R5.
- **"≥25% fewer full DTW calls on an exact matrix" via LBs:** UNACHIEVABLE — an exact matrix needs every DTW; LB early-abandon recomputes abandoned pairs (`Pruned` strategy is a PESSIMISATION for exact matrices). Exact-matrix DTW-work reduction = EAP cell-pruning / TADPole pair-skip only.
- **≥10× swap speedup from FastPAM decomposition on a cached matrix:**
  falsified—the advisory N=1000 table reports 2.95×–8.06× and is non-monotone.
  Both variants retain N² lookups/iteration; attributing the result to memory
  bandwidth is **[inferred]** until a counter gate proves it.
- **LB-pruning inside PAM/MIP/LRCore:** not admissible — those consumers read the whole matrix.
- **PDLP as production p-median solver:** falsified — matrix-free Kelley dominates on either device (945× CPU / 126× GPU at N=400); PDLP remains a cross-validation arbiter only.
- **Extending the hand-written CMake URL-pin lexical deny-list:** F11's two
  registered pivots reached 63/63 yet final audits still bypassed it with
  `CUSTOM_CACHE_KEY` and quoted backslash-newline normalization. F36 must
  replace the design, not add a third token special case.
- Full kill-context lives in the archive (`### Explicit rejections`, Phase 5) and `.claude/LESSONS.md`.

## Binding decisions (digest — full prose in the archive §Decision log; append NEW decisions here)

- 2026-07-06: `default_data_t` = double; Float32 explicit opt-in. `MetricType::L2` is a real multivariate L2. No-silent-fallback promoted to Global Constraint. Local benchmarks ADVISORY (shared machine).
- 2026-07-10: llfio pinned `b17613f…`; codecov + all 36 workflow actions SHA-pinned; Arrow 19.0.1 SHA-256 enforced; stale tracked MEX binary removed; HiGHS bundled in wheels, Gurobi external; production publication = manual explicit-go only.
- 2026-07-10 (M-series authorizations, all still binding): soft-DTW unification boundary (M5→R4); device-aware Tier-1 `auto` (M12); mmap v1/v2 invalidated, v3 authenticated identity + crash-safe publication + payload integrity (M14/M15/M53); cross-language Tier-1 seed 42, `42+i` restarts, Lloyd/MIP included (M13/M17); estimator literal fit/predict semantics (M23); barycenter real-work convergence + finite-state rejection (M24); dense-cache semantic binding + matrix-free dispatch reconciliation (M25/M37); transactional MIP publication (M33); DTW variant parameter domains (M34); Soft-DTW helper domain + denormal scaling (M46); HPC submission envelope/grammar/identity hardening (M28/M31/M32); dense checkpoint v2 identity (M49).
- 2026-07-10 (H4): MATLAB Tier-1 keeps the original six-method set in 2.0 (parity = 2.1); C++ HPC is a throwing beta boundary (no transport, no silent CPU fallback; Oxford ARC/2.1 owns enablement).
- 2026-07-12 (F7): truthful CLI RAM policy — `--ram-limit` is the Parquet series materialisation cap (hard-errors elsewhere); `--mmap-threshold` selects distance storage; matrix-free runs emit labels/medoids + binary checkpoint, dense CSVs only when a matrix exists.
- 2026-07-12 (F4): seeded RNG schedule versioned `portable-v1` (identical across MSVC STL/libstdc++); unseeded Tier-2 mutable-engine contract unchanged.
- 2026-07-12: Phase 9/R7 authorized (Emscripten+embind, worker pool no pthreads, unibatt palette verbatim, `dtwc::warping_path` additive API). `[BLOCKED-ENV]` record-and-continue promoted into the execution contract.
- 2026-07-23 (F9): Primary evidence is the local PyArrow-23-backed CMake build:
  390 assertions in all 11 Arrow/Parquet cases, skip absent. The complementary
  canonical Arrow-OFF build intentionally skips that target. Ubuntu 24.04 CI
  installs Arrow/Parquet and enforces ≥348 assertions plus ≥11 cases from
  Catch2's own summary; its hosted run is operator-owned and is not claimed by
  the local closure.
- 2026-07-23 (F8): Primary evidence is the local PyArrow-23-backed real CLI:
  six successful processes prove both mutually exclusive Parquet routes,
  nine resident/stream artifact pairs are byte-identical, and exact
  configuration payloads are pinned. The permanent target exists only with
  Parquet support; the Ubuntu Arrow job selects it fail-closed, but hosted
  execution remains operator-owned and is not claimed by the local closure.
- 2026-07-23 (F11 preflight): “Repo-wide” in F11 means every tracked
  `CPMAddPackage(URL ...)` archive declaration, not every internet-bearing
  workflow command or package-manager resolution. The latter audit found
  distinct workflow-integrity and shell-grammar subjects, now F34 and F35.
  Keeping them separate prevents an example-archive fix from falsely closing
  the wider supply chain and preserves one finding per implementation commit.
- 2026-07-24 (F11/F36): The exact example pin and no-override runtime proof are
  retained in `653b0e6`, but the generic parser closure is FALSIFIED. After two
  capped inventory pivots, final audits reproduced `CUSTOM_CACHE_KEY` and
  quoted-line-continuation false-greens with all production counters passing.
  Per the registered no-third-attempt band, freeze the lexical deny-list,
  leave F11 open, route replacement architecture to F36, and continue at F12.
- 2026-07-24 (F12 registration): One fixed-band behavioral task owns all six
  CUDA geometry copies and the backend-independent public double no-path
  sentinel. The D1 non-degenerate fixture, explicit path counts, exact
  pairwise/one-vs-N route matrix, `INT_MAX`, and two-attempt cap are binding.
  Local CUDA executes on the RTX. The probe found Windows 11 with neither
  `xcrun` nor a Metal compiler, so real-Metal execution is `[BLOCKED-ENV]`;
  implement and retain its permanent gate, leave F12 open, and continue after
  all locally executable bands pass.
- 2026-07-24 (F12 local verdict): Retain repair attempt 1 in `4583443`.
  Canonical CUDA geometry and exact public sentinel translation pass every
  registered real-RTX route, including both singleton orientations. Metal's
  matching no-LB source/tests survive two independent reviews, but the local
  binary necessarily skips with zero assertions. This is partial closure:
  keep F12 and D1's Metal discrepancy open, route LB integer arithmetic to
  F28-F30 as registered, and resume the campaign at F13.
- 2026-07-24 (F13 registration): Treat “nearest-medoid copies” as the six
  FastPAM/CLARANS/FastCLARA bodies plus the additionally audited public Lloyd
  body. Preserve first-slot ties, negative finite Soft-DTW distances, exact
  finite `DBL_MAX`, and FastCLARA parent-cache independence. CPU-f32 sentinel
  translation is a separate public-distance repair commit; R4 retains full
  scan consolidation. Reuse the read-only F8 Parquet fixture for the real
  streaming discriminator.
- 2026-07-24 (F13 Lloyd non-finite contract): The inherited Tier-1 case that
  required Lloyd to publish an infinite objective contradicts F13's registered
  rejection of every non-finite assignment distance. Keep the production
  rejection and update that stale test in repair attempt 2 to require
  `InvalidInput` while preserving the pre-call labels.
- 2026-07-24 (F13 Python gate): The fresh 1,022-test Python collection exposed
  three Lloyd literals left stale by the earlier portable-v1 RNG repair. An
  isolated extension built at exact F13 base `1af0aa8` reproduced all three,
  so they are not assignment regressions. Repair tests only: pin the current
  eight-series portable result and preserve the iteration-cap discriminator
  with registered singleton values `[0,1,2,3,5,4]`. The binding and production
  implementation remain unchanged.
- 2026-07-24 (F13 tracked-manifest count): The permanent F13 Arrow gate is one
  new tracked `.cmake` manifest. Exact main-index comparison against
  `e37b71a` gives 26 versus 25 and names only that file. Increment the
  supply-chain inventory floor to 26 without changing the F11/F36 parser or
  archive identities; filesystem/worktree contents are not evidence for this
  index-owned count.
- 2026-07-24 (Windows shared-Arrow CTest runtime): Arrow-linked unit
  executables import `arrow.dll`/`parquet.dll`, but inherited CMake attached
  PyArrow's runtime directories only to the real-CLI integration test. An
  ordinary unit CTest therefore stalls before `main` with loader status
  `0xC0000135`; the identical command passes with the three proven directories.
  Apply them to every test in that Windows Arrow-linked test directory.
  Arrow-OFF builds and production runtime/install behavior remain untouched.
- 2026-07-24 (F13 local verdict): Retain `62c6f26`, `1eb8609`, `cb11c90`,
  `3784251`, and `3783b12`. The registered tie, finite-state, exact-message,
  ordered-objective, streamed f64/f32, Python-forwarding, manifest, and Windows
  loader mutants all fail. Canonical and llfio-OFF pass 116/116; Arrow passes
  118/118 without a caller PATH override and executes both real-CLI gates; the
  fresh 1,022-test Python collection passes 1010 with 12 skips. This closes
  F13 behavior only: R4 still owns scan consolidation, and hosted CI is not
  claimed. Resume at F14.
- 2026-07-24 (F14 registration/F37 split): Freeze the four native C++ matrix
  formatter bodies before R4 consolidation: locale-free general
  `max_digits10`, LF-only binary files, signed-zero preservation, empty NaN
  sentinel, exact pre-output infinity rejection, and unchanged caller stream
  state. A scalar-token and finite-preflight primitive may be shared, but all
  four row/delimiter/output loops remain separate for R4. Native
  Problem/Result/resident-CLI/mmap-CLI reachability is F14.
  Python `np.savetxt` is runtime-confirmed byte-different and MATLAB
  `writematrix` is a separately owned emitter, so the frozen all-language
  four-file promise is F37 rather than an unregistered F14 expansion.
- 2026-07-24 (F14 local verdict): Retain `e5bfd20`. The four native formatter
  bodies share only raw-binary64 preflight and scalar-token primitives; their
  row/delimiter/output loops remain independent for R4. All 21 registered
  mutation executions fail, the real resident/mmap CLI and native
  `Result::save` routes satisfy the exact wire contract, and canonical,
  llfio-OFF, and Arrow gates pass 118/118, 118/118, and 120/120. Close-time
  file errors are checked explicitly. F37 remains the owner of the confirmed
  Python and runtime-unconfirmed MATLAB contradiction. Resume at F15.
- 2026-07-24 (F15 registration): The old eight-file byte-identity claim stays
  falsified. F15 may extract only two named generator contracts plus dense
  symmetric CPU-reference traversal; it must not merge packed LB values,
  MPI timing, compiler-sensitive float/range variants, structured/random-walk
  fixtures, or independent DP/path/formula arbiters. `uniform_real_distribution`
  bytes are frozen as coherent verified compiler-plus-standard-library
  profiles rather than mislabeled portable-v1 output. Real CUDA execution is
  required; real Metal remains locally unavailable and is not claimed. Two
  implementation attempts maximum.
- 2026-07-24 (F15 portability correction): Pre-run review falsified the
  assumption that compiler floating-point flags alone identify the legacy STL
  fixture bytes. WSL Ubuntu 24.04 GCC 13.3 and Clang 18.1 with libstdc++ agree
  with each other but differ from both Windows profiles, including for
  `[-1,1]`. Their exact third coherent profile is registered before the first
  decisive F15 run; the target must accept no unmeasured profile and no mixed
  generator/oracle row.
- 2026-07-23 (R0 provenance): Vinod (1969) is retained as early
  optimization-based clustering history, not evidence for DTWC++'s diagonal
  p-median matrix. The record attributes its linking rows to Balinski and the
  classical complete model to ReVelle-Swain; the interrupted “same program”
  and independent-lineage claims were removed.
- 2026-07-23 (R1 TODO scope): Reconcile all 53 live TODO records, not only the
  approximate count in the phase text. The 21 known-bug/cleanup records retain
  the three-way verdict and map STILL-OPEN defects to R3; the 32 backlog,
  question, deferred, and operator records use the same evidence standard but
  route open work to its owning campaign phase or external owner.
- 2026-07-23 (R1 TODO verdict): The 53-record ledger is binding. New residual
  defect/cleanup routes are F11–F16; the permanent resident/stream fixture
  remains F8. Historical “8K executed”, scalar-L2, dead-preload,
  byte-identical-generator, and pyproject-floor subclaims are retired rather
  than propagated. Open operator/community work is not a local test failure.
- 2026-07-23 (R1 docs/F17): The docs audit confirmed that CLI `--resume`
  deserializes a binary clustering result but never applies it. Keep the
  limitation explicit in user documentation and route the behavioral repair to
  R3 F17; its first regression must drive the real CLI and distinguish loaded
  state from the fresh clustering result.
- 2026-07-23 (R1 frozen-contract reconciliation): The old rule presented
  `docs/api-contract-2.0.md` as fully implementation-audited while retaining
  pre-implementation status text and eight unresolved reviewer choices. The
  approved documentation rule preserves every frozen 2.0 promise, labels each
  confirmed unfulfilled promise as an R3 finding (never as intended behavior),
  records `[introduced-2.0]` as historical provenance, and resolves the eight
  reviewer choices to the already shipped behavior. Compatibility effect:
  documentation only—no symbol, default, file format, or runtime behavior
  changes. The two existing 2.1 deferrals (MATLAB post-freeze Tier-1 methods and
  C++ HPC transport) remain unchanged. R1 owns the record repair; R3 owns every
  named implementation gap.
- 2026-07-23 (R1 GPU-doc truth): The inherited GPU page's universal
  LB-admissibility, option-fallback, routing, and exact-matrix speed claims are
  not retained. Current documentation is limited to verified support:
  equal-length L1 GPU LB with an envelope covering the DTW window, thresholded
  INF-stamped output, CPU-only `DistanceMatrixStrategy::Auto`/`lb_strategy`,
  and historical advisory timings only when a tracked raw artifact exists.
  Correctness/loudness repairs route to F27–F31; R1 changes no runtime behavior.
- 2026-07-23 (R1 record-retirement truth): Git history confirms
  `.claude/MISSING.md` / `READ.md` were retired by `0449f7c`. Record hygiene
  preserves that deletion and repairs the live UNIMODULAR/LESSONS/CITATIONS
  records instead of recreating obsolete ledgers.
- 2026-07-23 (R1 repository hygiene): Remove only the five registered non-data
  artifacts after routing consumers to byte-identical static assets. Retain
  the orphaned `data/test/AllGestureWiimoteX_dist_50.csv` byte-identically
  because the absolute data-read-only rule overrides orphan cleanup. The
  public Codecov badge uses its confirmed-equivalent tokenless endpoint; the
  old query remains in reachable history, so revocation/rotation is an
  operator remedy if Codecov treats it as scoped. Fresh Doxygen/Hugo rendering
  remains `[BLOCKED-ENV]` under the recorded `doxygen=NOT_FOUND`,
  `hugo=NOT_FOUND`, and `go=NOT_FOUND` probe.
- 2026-07-23 (R1 branch disposition): The first `4797c97` snapshot was 548
  commits ahead of both local and stale `origin/main`, and 51 ahead of stale
  `origin/Claude`. During final review, `origin/Claude` advanced to `4797c97`;
  its local reflog says `2026-07-23 18:14:56 +0100: update by push`. This agent
  issued no remote mutation. Non-sample hooks were absent, but four running
  GitHub Desktop processes predated the event, so actor/cause remains
  `[inferred: unknown]`. The global no-remote-operation sub-band is FALSIFIED;
  agent authorization compliance passes. Do not “repair” it locally. After
  R0–R6 close, the operator sequence is re-read server state, fetch/prune,
  fresh ancestry and divergence checks, release-gate rerun, decide
  retain/restore, review PR/hosted gates, then fast-forward only where ancestry
  permits. Rebase, merge, force update, tag, and publication remain
  unauthorized here.
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
- 2026-07-23 (R0/F7 coverage): Reclassified the purported “Arrow-OFF half” of
  F8 as F7 planner/guard coverage and repaired it in `7c71602`. Three production
  mutants were killed; focused suites pass 149/149 and 842/842; the fresh real
  CLI rejects capped CSV with exit 1 and accepts the same uncapped input with
  exit 0; canonical gate passes 114/114 with the six expected capability skips.
  F8 remains open in full. Evidence:
  `.claude/baselines/2026-07-23-f7-routing-coverage.md`.
- 2026-07-23 (R0/F9): Repaired the inherited Arrow job and closed F9. The first
  fresh configure found a scope leak that announced Parquet but omitted
  `DTWC_HAS_PARQUET`; `833f570` exports the package result. The newly reachable
  Windows suite then exposed an mmap lifetime failure; `0c91c9b` releases the
  reader before unlink. The full Arrow-ON binary passes 390 assertions in all
  11 cases, imports both Arrow and Parquet, and its CTest target runs rather
  than skips. `e323197` adds the Ubuntu 24.04 gate plus a mutation-tested
  executed-assertion/case parser. Canonical CTest passes 114/114 with exactly
  its six registered capability skips. Evidence:
  `.claude/baselines/2026-07-23-f9-arrow-gate.md`.
- 2026-07-23 (R0 closure): Every inherited hunk has a committed verdict.
  Scholarly repairs are `c627826` (floating-point record), `8775156` (TODO
  staleness plus one verified closure), and `85eabcd` (MIP provenance).
  `git status --porcelain=v1` was empty; canonical behavioral closure remains
  114/114, zero failed, exactly six capability skips. Final matrix:
  `.claude/baselines/2026-07-23-r0-adjudication.md`. Proceed to R1.
- 2026-07-23 (R1 start): Registered the TODO reconciliation at base `83a2048`.
  Inventory: 53 records (49 unchecked, one checked, three open questions);
  acceptance requires 53/53 verdicts, no unclassified records, and unique R3
  IDs for every still-open defect. Evidence:
  `.claude/baselines/2026-07-23-r1-todo-reconciliation.md`.
- 2026-07-23 (R1 TODO reconciliation): Committed the 53-row evidence ledger in
  `512bbc4` and rewrote the stale live index in `81cae08`. Final parser:
  53 expected/actual/unique, 21 known-bug/cleanup, 32 remaining, zero missing,
  unexpected, duplicate, or `UNVERIFIED` records; F11–F16 each appear once.
  Direct focused closure gate: nine binaries, 5,809 assertions in 142 cases,
  zero skips/failures. Evidence:
  `.claude/baselines/2026-07-23-r1-todo-reconciliation.md`.
- 2026-07-23 (R1 docs/F17): Source audit confirmed the CLI `--resume` defect:
  `ckpt_result` is created and loaded only at `dtwc/dtwc_cl.cpp:1398-1405`
  and has no subsequent consumer. F17 records the real-binary failing gate;
  `.claude/baselines/2026-07-23-r1-docs-truth.md` D5 records the evidence.
- 2026-07-23 (R1 frozen contract): Reconciled the freeze artifact against the
  current tree. The inherited audit guard failed on eight stale marker classes;
  final inventory is zero stale markers, exactly eight adjudicated reviewer
  decisions, and all nine implementation gaps F18–F26 named as 2.0
  obligations. Tier-1/Tier-2/migration projections are current and the real-CLI
  docs gate passes. Fresh Hugo rendering remains `[BLOCKED-ENV]`. Evidence:
  `.claude/baselines/2026-07-23-r1-docs-truth.md` D6.
- 2026-07-23 (R1 GPU docs): Registered F27–F31, then repaired the GPU page
  without changing runtime behavior. The inherited guard failed on all 12 stale
  claim classes; final live-CLI docs gate passes. Current text scopes GPU LB to
  equal-length L1 with a matching admissible window and thresholded `+inf`
  output, states CPU-only Auto/lower-bound routing, and retains only a
  raw-artifact-linked historical/advisory timing table. Evidence:
  `.claude/baselines/2026-07-23-r1-docs-truth.md` D7.
- 2026-07-23 (R1 docs closure): Reconciled the remaining examples, interface
  map, multivariate and score pages, plus floating-point/DTW source comments.
  The inherited D8 guard failed on 24 stale marker classes; the post-rebuild
  real-CLI contract gate passes and `git diff --check` is clean. A link check
  over the ignored existing site passes only as advisory evidence; the fresh
  Hugo render remains `[BLOCKED-ENV]` under the recorded `hugo=NOT_FOUND` and
  `go=NOT_FOUND` probe. Evidence:
  `.claude/baselines/2026-07-23-r1-docs-truth.md` D8.
- 2026-07-23 (R1 record hygiene): Reconciled UNIMODULAR, LESSONS, and
  CITATIONS against current code, tracked artifacts, and opened primary
  sources. Two independent residual reviews rejected the first checker green
  and forced corrections to SIMD/Float32/FastPAM/I/O evidence scope,
  LR-core/Benders architecture, the Ghouila-Houri proof, and stale
  bibliography attributions. The permanent checker now pins those classes,
  the sole freshness header, canonical citation counts/URLs, the deliberate
  retirement of MISSING/READ, and the corrected 2.95×–8.06× table reading.
  Evidence: `.claude/baselines/2026-07-23-r1-record-hygiene.md`.
- 2026-07-23 (R1 closure): `4797c97` removes five verified non-data artifacts,
  preserves all three ignored build roots and the orphaned data fixture,
  expands future generated-file ignores, removes the Codecov badge query, and
  records both RNG compatibility boundaries. The preregistered checker moves
  from 5/2/3 inherited banned/zero/duplicate failures to exact PASS; an
  adversarial reviewer rejected four earlier false-green designs before the
  index/blob/ordered-ignore gate was accepted. CHANGELOG structure and the
  real-CLI documentation contract pass. The later unexpected
  `origin/Claude` update FALSIFIED the global no-remote-operation sub-band;
  campaign-agent compliance still passes and no rollback was attempted.
  Evidence:
  `.claude/baselines/2026-07-23-r1-repo-hygiene.md`. Proceed to R2-D1 and R3-F8.
- 2026-07-23 (R2-D1/F33): Sakoe and Chiba equations (6)–(8) were checked from
  the primary scan with the paper's weighted recurrence kept distinct from
  DTWC++'s objective. The inherited CPU endpoint-scaled corridor was
  FALSIFIED, then repaired in `9f78212`; the derivation and permanent drift
  guard are `cf5b9d8`. D1 closes CPU **CONFIRMED** while F12 remains open for
  CUDA geometry and exact Metal no-path parity. The actual canonical inventory
  is 114/114, zero failed, with the same six capability skips. Evidence:
  `.claude/baselines/2026-07-23-r2-d1-dtw.md`. Resume at R3-F8.
- 2026-07-23 (R3-F8): `1df77fc` tracks the registered Parquet fixture and a
  permanent non-skippable real-CLI parity test. The first decisive run passed:
  six exits, 12/12 live route checks, 18 exact artifacts, 9/9 byte-identical
  resident/stream pairs, and 3/3 distinct configuration checkpoints. The fresh
  Arrow-ON suite passes 115/115 with `test_io_readers` and F8 executing; the
  canonical Arrow-OFF suite remains 114/114 with exactly six capability
  skips. Independent review found all 11 registered acceptance items verified.
  Hosted CI was not run locally and is not claimed. Evidence:
  `.claude/baselines/2026-07-23-f8-fast-clara-parity.md`. Resume at R3-F11.
- 2026-07-24 (R3-F11 partial/F36): `653b0e6` pins the standalone example to
  commit `eda1b92bc89ee51568b052a6af86f615d336de3c` plus SHA-256 and hardens
  the tracked archive gate through an exact seven-identity inventory and 63
  mutation cases. Fresh example configure/build, CPU/GPU HiGHS builds, real
  GPU/HiGHS test, and canonical 114/114 all pass. Final independent audits
  nonetheless FALSIFIED fail-closed coverage with `CUSTOM_CACHE_KEY` and
  quoted CMake line continuation. F11 remains open; F36 replaces the killed
  lexical design. Evidence:
  `.claude/baselines/2026-07-23-f11-supply-chain-coverage.md`. Resume at F12.
- 2026-07-24 (R3-F12 partial): `4583443` replaces six CUDA slanted corridors
  with one canonical overflow-safe predicate and normalizes FP32 GPU no-path
  values at the public double boundary. The final real-RTX gate passes 515
  assertions / 6 focused cases and 7,827 / 61 unfiltered; CUDA CTest passes
  2/2, the host normalizer passes 8/8, and canonical plus llfio-OFF gates pass
  115/115. Two independent audits found no remaining local blocker. The fresh
  Metal-OFF executable skips with zero assertions, so F12 remains open pending
  real Apple build/device evidence. Evidence:
  `.claude/baselines/2026-07-24-f12-gpu-fixed-band-parity.md`. Resume at F13.
- 2026-07-24 (R3-F13 registration): At base `1af0aa8`, audited six algorithm
  assignment bodies plus Lloyd's public seventh body and registered literal
  tie/order/non-finite/sentinel oracles before tests or production edits.
  Separate CPU-f32 and assignment repair commits, exact diagnostics and bits,
  a four-process real Arrow CLI gate, mutation probes, and final build floors
  are binding. Evidence:
  `.claude/baselines/2026-07-24-f13-medoid-assignment-contract.md`.
- 2026-07-24 (R3-F13 CPU-f32 partial): The preregistered live resolver test
  failed 1/1 with widened `FLT_MAX`. Commit `62c6f26` moves exact sentinel
  translation to a shared CPU/GPU public-distance policy; repaired focused,
  GPU-host, and full distance-semantics gates pass 5/5, 8/8, and 53/53.
  Nearest-medoid assignment validation remains the active F13 half.
- 2026-07-24 (R3-F13 CLOSED): `1eb8609` enforces the seven-body assignment
  contract, `cb11c90` repairs exact-base-confirmed stale Python oracles,
  `3784251` registers the added CMake manifest, and `3783b12` makes every
  Windows Arrow-linked CTest self-contained. The focused gate passes 114
  assertions / 8 cases; canonical and llfio-OFF pass 116/116; Arrow passes
  118/118 without caller PATH and executes both real-CLI gates; fresh-extension
  Python passes 1010 / 12 skipped over 1,022 collected. Nine mutation classes
  fail. Earlier falsifications remain recorded, full scan consolidation remains
  R4-owned, and hosted CI is not claimed. Evidence:
  `.claude/baselines/2026-07-24-f13-medoid-assignment-contract.md`. Resume at
  F14.
- 2026-07-24 (R3-F14 registration/F37): At clean base `95ffd63`, audited four
  native matrix CSV bodies and registered an independent 83-byte LF-only
  literal, hostile locale/stream state, exact infinity side effects/messages,
  dense/mmap/visitor/print/Result routes, real resident/mmap CLI parity, native
  Result parity, manifest 27, 13 mutation classes and 21 executions, and final
  inventories 118/118 canonical, 118/118 llfio-OFF, and 120/120 Arrow. The
  inherited Windows CLI artifact contains 27 CRLF rows. Python's real 212-byte
  `Result.save` matrix confirms F37; MATLAB remains runtime-unconfirmed.
  Evidence: `.claude/baselines/2026-07-24-f14-csv-wire-format.md`.
- 2026-07-24 (R3-F14 CLOSED): `e5bfd20` implements the registered native CSV
  contract in attempt 1 and retains all four formatter loops. Focused
  llfio-ON/OFF gates pass 144 assertions / 13 cases and 88 / 10; resident CLI,
  mmap CLI, and native `Result::save` pass exact public byte checks. All 13
  mutation classes and 21 executions fail. The supply-chain inventory remains
  39/7/1/27, and final canonical, llfio-OFF, and Arrow suites pass 118/118,
  118/118, and 120/120 with 6/9/8 capability skips. F37 retains cross-language
  parity; R4 retains consolidation. Evidence:
  `.claude/baselines/2026-07-24-f14-csv-wire-format.md`. Resume at F15.
- 2026-07-24 (R3-F15 registration): At clean base `e79fab3`, retired the stale
  eight-copy premise, inventoried two exact clone families plus adjacent
  intentional variants, and froze raw generator/oracle fingerprints for the
  repository's Clang-relaxed and MSVC-precise profiles. Independent full-matrix
  DP equals the production rolling CPU reference in both profiles, while the
  band-0 fixture differs from full DTW. Six benchmark executables compile;
  real CUDA passes 7,827 assertions / 61 cases and 688 / 8; CPU accuracy/SIMD
  pass 283 / 39 and 7,029 / 16. Evidence:
  `.claude/baselines/2026-07-24-f15-test-support.md`.
- 2026-07-24 (R3-F15 portability): Before implementation attempt 1 executed,
  compiled the literal preflight in WSL Ubuntu 24.04 with GCC 13.3 and Clang
  18.1 plus the repository Release relaxations. Both libstdc++ runs produced
  the same third coherent generator/full/band-0 profile and retained
  digit-identical independent/production matrices. Evidence:
  `.claude/baselines/2026-07-24-f15-test-support.md`.
