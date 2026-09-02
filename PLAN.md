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

**Status (2026-08-09, post-D3 closure):** 2.0.0rc1 release state
committed (not tagged or published). Refactor Phases 0–7 CLOSED; Phase 8
(8.0/8.1, findings F1–F10, sanitizer gate) CLOSED. Phases R0–R1 CLOSED; R2
active with **D1–D3 CLOSED** and D4–D18 outstanding (15/18 remain); R3 active
with **F13, F14, F15, F19, F21, F23, F33, F45, F51, F54, F55, F57 CLOSED**.
D3's 40-equation derivation (`7dce222`), fail-closed documentation gate
(`639e1c4`), F54 cascade-max repair (`d09cf9c`+`53506a9`), F55 provenance
corrections (`6abff20`), and F57 saturation repair (`29f9103`) pass the exact
focused/WSL-UBSan gates and the serial canonical, llfio-OFF, and Arrow-ON
matrices at **125/125, 125/125, and 127/127** with exact 6/9/8 skip sets;
Arrow runtime evidence confirms D3 115/1, F57 24/1, reader 390/11, and all four
real-CLI markers.
Repair-retained but closure-FALSIFIED, both attempts exhausted, evidence-only
checkboxes (never rescue-tune): **F11** (parser replacement → F36), **F16**
(fail-closed metadata → F38), **F17** (manifest reconciliation → F39),
**F18** (residuals → F40/F41/F42), **F20** (residuals → F43/F44). **F12**
partially closed (CUDA verified on the local RTX; real Metal
`[BLOCKED-ENV]`). **F22** products + docs retained; Python 31/31 and MATLAB
66/66 mutation gates pass; the C++ mutation band is FALSIFIED 33/46 with both
attempts exhausted. Its final serial gates pass all three native matrices,
fresh Python with the registered F39 red, and both MATLAB releases with the
registered F18 red; F22 stays unchecked. **Campaign cursor: (1) the GPU-LB
CUDA cluster (F27, F29, F47, F50 local-RTX halves; Metal halves
`[BLOCKED-ENV]`, gates implemented-and-retained) — D2/D3 now supply the
admissibility oracles it was waiting for; (2) D4 (EAPruned) per the cadence
rule.** The final **2.0.0 tag gates on R0–R6
CLEAN**; R7 (WASM
Playground) is a 2.1 feature and does not gate the tag.
Tag/publication/hosted-CI/ARC/Metal-runtime remain explicit USER actions —
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

**You have infinite tokens. You do not have infinite hours.** Never economize
on thinking, reading source, re-deriving mathematics, or verification — spend
tokens lavishly there. The scarce resources are wall-clock and evidence
integrity, so spend HOURS in proportion to scientific value (rule 11) and
assume the session can die at any moment (rules 1, 3). Earlier runs
**died mid-task with everything uncommitted**. These rules exist so a death
costs one task, never a session:

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
11. **Evidence effort is proportional to consequence.** A wrong distance value,
    an inadmissible lower bound, or a silently misrouted device deserves the
    full ceremony (registered bands, mutation campaigns, arbiters). A
    deprecation warning or a docs anchor deserves a focused test and a commit.
    The F22 lesson: a 110-mutant cross-language campaign for warning text is
    hours R2 never got. Before building a harness, ask what a false-green
    would cost the science; size the gate to that answer. Rigor floors stay
    absolute — registered bands before decisive runs, no unverified claims —
    but the DEPTH of the campaign is yours to right-size, and the choice is a
    Decision-log line when you go notably lighter than precedent.
12. **Rolling hygiene is part of every task, not a phase.** Each finding or
    derivation closure ends with a sweep before its final commit: temporary
    probe scripts deleted or promoted into `tests/`/`scripts/` with an owner;
    no stray files in the tree (`git status` clean means INSPECTED clean, not
    just committed); dead branches of your own exploration removed; PLAN.md
    slimmed by archiving closed prose when it exceeds ~1,200 lines (precedent:
    the three `.claude/PLAN-archive-*.md` files); AGENTS.md floors updated the
    moment a gate legitimately changes them. The repository must be LESS
    cluttered after your session than before it — a bloated repo is a failed
    session even if every gate is green.

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

**R2/R3 cadence rule (binding from 2026-07-29):** R3 has consumed every session
since R0 while R2 sits at 1/18 — yet R2 is where the campaign's scientific
value lives, and several open R3 findings cannot even be judged without their
derivation (the pairing map in R2). From now on, alternate: after each R3
finding reaches its verdict, complete at least one R2 derivation file before
opening the next finding. When a finding's subject has a paired derivation,
do the derivation FIRST — it is the finding's oracle. Deviating from the
cadence is allowed (rule 8) but costs a Decision-log line saying why.

Rolling hygiene (rule 12) runs inside every phase; it has no phase of its own
on purpose — cleanliness is a property of every commit, not a milestone.

---

## Phase R0 — Adjudicate the 2026-07-20 in-flight work [CLOSED — 2026-07-23]

All 2026-07-20 uncommitted work (F9/F10-shaped tests, CI job, scholarly edits)
was classified per hunk, verified against the canonical gate, and committed,
repaired, or reverted; the F8/F9/F10 checkboxes below reflect the verdicts.
Full inventory and task list archived verbatim in
`.claude/PLAN-archive-2026-07-27-r0-f20.md`.

## Phase R1 — Repository cleanse & record reconciliation [CLOSED — 2026-07-23]

Non-behavioral hygiene, all closed: TODO.md reconciled entry-by-entry, docs
truth audit + drift gates green, `.claude/` record hygiene done, tracked-junk
census executed (data-read-only rule preserved one orphan), CHANGELOG
structure verified, branch state recorded (merge is an operator decision —
see the R1 branch-disposition Binding decision). Full task list archived in
`.claude/PLAN-archive-2026-07-27-r0-f20.md`.

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

**Derivation↔finding pairing map (do the derivation before or with the
finding — it is the finding's oracle):** D2/D3 (envelope + LB admissibility)
before F27/F28/F29 (GPU LB correctness — you cannot judge an LB bug without
the admissibility proof); D7 (Soft-DTW negativity bound) before D13 and any
F10-successor sampling decision; D9 backs the standing DTW_I ≤ DTW_D hard
band; D15 before the solver edge-case lens; D17 before the precision,
cross-compiler, and numerical-extremes lenses (they are told "derive the band,
don't guess" — D17 is where the band comes from); D18 before any scores-related
finding. Unpaired derivations (D4, D5, D6, D8, D10, D11, D12, D14, D16) are
free-standing science — schedule them via the cadence rule.

Derivation targets — each is one checkbox, one file, one conformance pass:

- [x] **D1. DTW recurrence + Sakoe–Chiba band.** Optimal substructure; boundary
      conditions; band feasibility (`band ≥ |n−m|` for a nonempty path);
      monotonicity `DTW_band ≥ DTW_full` and monotone-in-band; L1 vs squared-L2
      local costs and what "distance" each yields (squared form is not a metric
      — say so). Conformance: `dtw_kernel.hpp`, `dtwBanded`. CPU conformance is
      **CONFIRMED** by `9f78212` and derivation commit `cf5b9d8`; CUDA geometry
      is **CONFIRMED** by `4583443`, while exact real-Metal parity remains
      **DISCREPANCY** F12.
- [x] **D2. Envelopes + LB_Keogh.** The derivation proves scalar L1/squared-L2
      admissibility, feasible unequal-length prefixes, and additive dependent
      and independent multivariate forms, with explicit units and assumptions.
      The exhaustive oracle covers 2,004 envelopes, 28,602 equal-length paths,
      17,712 unequal-length paths, and both nonempty CPU full-DTW call-site
      classes; the exact target passes 65 assertions/1 case in all three
      matrices. Canonical/llfio-OFF/Arrow-ON pass 123/123, 123/123, and 125/125
      with exact 6/9/8 skips; Arrow readers pass 390/11. F29's old premise is
      falsified, but its real-device gate and F27–F30/F46–F50 remain open.
      Evidence: `docs/derivations/02-envelopes-lb-keogh.md` and
      `.claude/baselines/2026-07-30-d2-lb-keogh.md`.
- [x] **D3. LB_Enhanced + local LB_Webb_NoLR plus tail cap.** Prove
      admissibility in the registered finite equal-length scalar L1/squared-L2
      domain (Tan SDM 2019; Webb & Petitjean PR 2021). Prove the local
      directional bound dominates matching-direction Keogh and its symmetric
      maximum dominates symmetric Keogh. Prove only the column-alignment tail
      cap (`idx=min(j+w,n-1)`) is no greater than exact-predicate NoLR; claim no
      ordering with full Algorithm 2. Enhanced dominates matching-direction
      Keogh at effective `V=1`; exact repository witnesses establish both
      strict directions at effective `V>=2`, so the live cascade takes max.
      **CLOSED 2026-08-09:** derivation committed (`7dce222`, 40 equation
      tags, cellwise Webb proof), fail-closed doc checker committed
      (`639e1c4`, six mutants reject), expected red CONFIRMED, product
      attempt 2 PASS (D3 115/115 + F57 24/24, ctest 2/2), F57 WSL-UBSan
      clean. Serial canonical/llfio-OFF/Arrow-ON matrices pass 125/125,
      125/125, and 127/127 with exact 6/9/8 skips; Arrow runtime subjects pass
      7/7 with reader 390/11 and all four real-CLI markers. Evidence:
      `.claude/baselines/2026-07-30-d3-lb-enhanced-webb.md`.
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

**Immediate next step (2026-08-09):** execute the GPU-LB correctness cluster
against the now-closed D2/D3 admissibility oracles: F27 (squared-L2 LB_Keogh),
F29 (feasible unequal-length prefix execution), F47 (squared-L2 LB_Kim), and
F50 (`INT_MAX` device window arithmetic). Run every CUDA half on the local RTX
and implement-and-retain matching Metal gates with `[BLOCKED-ENV]` runtime
evidence on this Windows host. Keep F28's real-Metal-only narrow-envelope
subject open. Then complete D4 before opening another finding, per the cadence
rule. F22 remains unchecked because its exhausted C++ mutation closure band is
FALSIFIED at 33/46; do not rerun or rescue-tune it.

**Suggested sequencing for the remaining findings (improvise freely, rule 8 —
this is a route, not a script).** Cluster by shared context so each cluster
amortizes its setup, and interleave R2 per the cadence rule:
(a) **bindings cluster** F23–F26 (Python checkpoint/device-error/view
semantics, assertion taxonomy) — pure local C++/Python work, no derivation
prerequisite; (b) **GPU-LB correctness cluster** F27–F29 (+F30/F31 loudness/
taxonomy) — do D2/D3 FIRST, then the CUDA halves on the local RTX; Metal
halves are `[BLOCKED-ENV]`, implement-and-retain their gates; (c) F32
multivariate rejection — pairs with D9; (d) **supply-chain cluster**
F34–F36 — F36 is a design replacement, not another lexical patch; (e)
**contract cluster** F37–F39; (f) **environment/MEX cluster** F40–F44 —
F42/F43 are crash root-causes with registered differentials, F41 stays
`[BLOCKED-ENV]` on this host. Proportionality (rule 11) applies: the GPU
correctness cluster deserves the full ceremony; loudness/warning findings
deserve focused gates.

Open findings first (status after R0 adjudication — update these boxes there):

- [x] **F8 — resident≡stream parity pinned by nothing.** CLOSED by `1df77fc`:
      SHA-pinned 8-series/4-row-group fixture + non-skippable real-CLI gate;
      nine resident/stream artifact pairs byte-identical across
      f64/f32/Soft-DTW. Hosted Ubuntu execution remains operator-owned.
- [x] **F9 — Parquet reader suite absent from the canonical gate.** CLOSED:
      a local Arrow-ON build runs `test_io_readers` for real (390 assertions /
      11 cases, skip message absent — a skip is a pass to ctest); the CI job
      is the secondary, operator-verified layer.
- [x] **F10 — signed/degenerate D-sampling half-pinned.** CLOSED in
      `20b894d`: direct seam plus seeded/unseeded signed/degenerate routes;
      selected and unselected non-finite inputs fail closed. NOTE: R2-D13 may
      CHANGE the sampling rule; if so, these tests pin the new rule and the
      old rule's tests are updated in the same commit.
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
      **CLOSED 2026-07-24** (`62c6f26`, `1eb8609`, `cb11c90`, `3784251`,
      `3783b12`): seven assignment bodies pinned — first-slot ties, ordered
      IEEE-754 objectives, exact sentinel translation, non-finite rejection,
      transactional Lloyd publication; nine mutation classes fail. Full scan
      consolidation remains R4-owned. Evidence:
      `.claude/baselines/2026-07-24-f13-medoid-assignment-contract.md`.
- [x] **F14 — four CSV emitters have no byte-parity contract.** **CLOSED
      2026-07-24** (`e5bfd20`): native ASCII bytes frozen (binary64 at
      `max_digits10`, comma/LF, locale independent, raw signed zero, NaN
      empty, infinities rejected; 83-byte SHA-pinned 3x3 oracle); all 21
      registered mutation executions fail. R4 retains formatter
      consolidation; F37 retains cross-language parity. Evidence:
      `.claude/baselines/2026-07-24-f14-csv-wire-format.md`.
- [x] **F15 — benchmark/test generators and CPU oracles are fragmented.**
      The historical eight-copy byte-identity claim stays FALSIFIED. **CLOSED
      2026-07-24** (`3061a31`): only the two exact generator families and the
      dense symmetric production-CPU traversal extracted; intentional variants
      preserved; attempt 1's Catch2 compile failure recorded; all 13
      mutations killed. Evidence:
      `.claude/baselines/2026-07-24-f15-test-support.md`.
- [ ] **F16 — CMake presets encode one developer machine and a stale floor.**
      `CMakePresets.json` hardcodes a Windows LLVM path and declares CMake 3.21
      while the root requires 3.26. First gate: portable clean configure probes
      plus a metadata check that rejects any future floor drift.
      **REPAIR RETAINED / CLOSURE FALSIFIED 2026-07-24:** `7aef30d` removes
      the developer path, aligns all active floors at 3.26, and makes preset
      visibility host-specific. Focused/configure/full gates pass, but
      registered M10 proves configure-time `string(JSON)` accepts trailing
      non-whitespace after a valid JSON value. The two-attempt cap leaves F16
      open as an evidence-state checkbox only: no further implementation is
      owned here. F38 uniquely owns fail-closed metadata architecture, and the
      binding active resume pointer is F17.
- [ ] **F17 — CLI resume replay. REPAIR RETAINED / CLOSURE FALSIFIED
      2026-07-24:** completed-result replay and 12/12 mutation gate pass;
      frozen manifest sub-band failed 28 vs 27. Attempts exhausted; F39 owns
      reconciliation. Verbatim registration/status archived in
      `.claude/PLAN-archive-2026-08-09-findings.md`; evidence:
      `.claude/baselines/2026-07-24-f17-cli-resume.md`.
- [x] **F18 — MATLAB estimator routing. CLOSED 2026-09-02:** `DTWClustering.fit`
      executes `Metric` (exact SquaredL2 matrix via `DTWClustering_compute_distance_matrix`)
      and `Device` (validated through `Env`, restored on error, forwarded to the
      Problem's strategy and `cuda_settings.device_id`); both red gates green on
      R2024b and R2025b (`.claude/reports/2026-09-02-matlab-parity.md`,
      `-adversarial-matlab.md`). GPU execution itself remains unverified here
      (no CUDA MEX built); F41/F42 keep that. Historical status: ATTEMPTS
      EXHAUSTED / FALSIFIED 2026-07-24: product rolled back after R2024b CUDA Auto crashed before
      the first kernel row; retained red-first gates are `625b5b7`. F40/F41/F42
      own the separated residuals, with F42 prerequisite to reopening. Verbatim
      registration/status archived in
      `.claude/PLAN-archive-2026-08-09-findings.md`; evidence:
      `.claude/baselines/2026-07-24-f18-matlab-routing.md`.
- [x] **F19 — the frozen `Problem` encapsulation/accessor cleanup was
      incomplete.** CLOSED 2026-07-24 (`3612b68`): ten fields privatized,
      canonical accessors added, redundant MATLAB writeback removed; 31/31
      public-header assertions, twelve permanent mutations, 24/24 MATLAB
      routes on both releases. Full registration prose archived verbatim in
      `.claude/PLAN-archive-2026-07-30-decisions.md`. Evidence:
      `.claude/baselines/2026-07-24-f19-problem-encapsulation.md`.
- [ ] **F20 — storage-policy routing. REPAIR RETAINED / CLOSURE FALSIFIED
      2026-07-24:** C++/Python and R2025b routes pass, but R2024b llfio-ON
      crashes in the first runtime mutex lock; 5/6 binding band, both attempts
      consumed. F43/F44 own the separated residuals. Verbatim
      registration/status archived in
      `.claude/PLAN-archive-2026-08-09-findings.md`; evidence:
      `.claude/baselines/2026-07-24-f20-storage-policy.md`.
- [x] **F21 — four frozen C++ snake_case entry points are absent.**
      CLOSED 2026-07-29 (`5e4a7b6`, `a48635b`, `36b9c99`): canonical
      `start_column`/`start_row`/`set_data_path`/`set_results_path` added with
      deprecated old-name forwarders; 12/12 signatures, 81 assertions/2 cases
      in all three builds, 12/12 mutants killed, all three full matrices
      green. Full registration prose archived verbatim in
      `.claude/PLAN-archive-2026-07-30-decisions.md`. Evidence:
      `.claude/baselines/2026-07-29-f21-cpp-renames.md`.
- [ ] **F22 — compatibility aliases failed the frozen deprecation
      policy.** At registered base `5352bc0`, C++ `maxIter`/`N_repetition`
      were unannotated, most Python aliases forwarded without
      `DeprecationWarning`, and MATLAB legacy properties/functions forwarded
      silently. Representative retained repairs now live at
      `dtwc/Problem.hpp:258-261`,
      `python/src/_dtwcpp_core.cpp:785-796,850-857,875-881,914-933,1183-1246`,
      and `bindings/matlab/+dtwc/Problem.m:96-134,193-199,362-396`; the
      complete inventory is pinned in the F22 baseline. First gate: table-drive
      every retained alias—C++ compile probes
      require a deprecation diagnostic, Python uses
      `pytest.warns(DeprecationWarning)` exactly once per call, and MATLAB
      captures one stable warning identifier/message—while asserting canonical
      names stay silent and results remain identical.
      **R5 STATUS 2026-07-29:** product repairs and all focused gates are
      retained. Python kills 31/31 registered mutants and MATLAB kills 33/33
      on both R2024b and R2025b (66/66 release kills, 140/140 MEX hash checks).
      The C++ mutation band is permanently FALSIFIED at 33/46 after both
      permitted attempts timed out on the first runtime mutant. Keep F22
      unchecked and do not rerun or rescue-tune the exhausted campaign. Evidence:
      `.claude/baselines/2026-07-29-f22-deprecation-policy.md`.
      **R6 STATUS 2026-07-30:** documentation and every registered final gate
      pass, including the expected F18/F39 reds, but do not reinterpret the
      exhausted C++ mutation criterion. F22 remains FALSIFIED and unchecked.
      Evidence: `.claude/baselines/2026-07-30-f22-final-gates.md`.
- [x] **F23 — Python binary result checkpoints. CLOSED 2026-07-30:** native
      binary-v1 bindings pass exact 3/3, parity 157/157, and registered full
      inventories; F56 owns path-formatting residuals. Verbatim live closure
      prose archived in `.claude/PLAN-archive-2026-08-09-findings.md`; evidence:
      `.claude/baselines/2026-07-30-f23-python-binary-checkpoint.md`.
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
      with `assert(!data.is_view())` (`dtwc/Problem.hpp:300-324`), contrary to
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
      calls the C++ view setter (`python/src/_dtwcpp_core.cpp:901-908`); true
      non-owning spans remain C++/CLARA-internal. First gate: bind a contiguous
      ndarray through the public Python method, mutate a non-degenerate element
      in the source, and require `Problem.series()`/a recomputed distance to
      observe that mutation while the Problem keeps the Python owner alive.
      Non-contiguous, readonly, dtype, and lifetime cases must be explicit and
      typed. The inherited binding must fail the aliasing assertion.
- [ ] **F27 — GPU LB_Keogh uses L1 excess under squared-L2 DTW.** CUDA and
      Metal always sum raw envelope excess (`dtwc/cuda/cuda_dtw.cu:766-811`,
      `dtwc/metal/metal_dtw.mm:946-954`) even when the distance kernel uses
      squared local costs, so threshold pruning can discard a pair whose true
      squared-DTW cost is below threshold. First gate on each real backend:
      series `{0}` and `{0.5}`, band 0, squared L2, LB enabled, threshold 0.3;
      the pair must survive and equal 0.25 rather than be pruned by the current
      L1 bound 0.5. Force Metal Wavefront so its LB stage actually runs.
- [ ] **F28 — Metal permits a narrow LB envelope for full DTW.** When DTW is
      unbanded and `lb_envelope_band` is unset, Metal chooses roughly
      `max_L/10`; it also accepts an explicitly narrower window
      (`dtwc/metal/metal_dtw.mm:1510-1515`). Such a bound is not admissible for
      the larger/full warping window. First real-Metal gate (forced
      Wavefront): `x={0,0,0,0,1,1,1,1,1,1}`,
      `y={0,0,0,0,0,0,1,1,1,1}`, full DTW, envelope band 1, threshold 0.5.
      True DTW is zero, so the pair must not be pruned. Reject or widen any LB
      envelope that does not cover the actual DTW window before dispatch.
- [ ] **F29 — unequal-length GPU LB_Keogh prefixes lacked a proof and an
      executable gate.** Both kernels compare only the first `min(Li,Lj)`
      rows. D2's path-row proof now shows that this truncation is admissible
      under the repaired fixed window `|i-j|<=band` whenever a path exists and
      the envelope covers that window; its exhaustive independent arbiter
      passes 17,712 feasible cases. The inherited fixture
      `a={0,0.25}`, `b={0,0,0.25}`, band 0 is **FALSIFIED** as a test of that
      claim: fixed-band DTW has no path because `|2-3|=1`. Replacement CUDA
      gate: at band 1 require the corrected zero-warp fixture to survive a
      0.1 threshold at exact distance 0, and require `{0,0}` versus
      `{1,1,1}` to return prefix LB 2 and exact L1 DTW 3 (survives threshold
      3.5). The same real-Metal gate or a recorded `[BLOCKED-ENV]` result is
      required before closure; no product repair is justified unless an
      executable disagrees with the derived values.
- [ ] **F30 — explicit GPU option requests silently degrade.** Metal disables
      requested LB on regtile/banded-row and after LB-buffer allocation failure
      (`dtwc/metal/metal_dtw.mm:1493-1504,1536-1550`); CUDA ignores LB when
      `band<0` (`dtwc/cuda/cuda_dtw.cu:1473-1480`); unsupported kernel
      overrides silently select Auto (`dtwc/enums/KernelOverride.hpp:8-10`);
      Metal FP64 quietly becomes FP32 unless verbose
      (`dtwc/metal/metal_dtw.mm:1263-1264`). The existing Metal LB test
      explicitly expects the banded-row no-op. First gate: table-drive every
      explicit option across supported/unsupported paths and the injected
      allocation seam. Each request must execute, raise a typed error, or return
      universally inspectable fallback metadata—never an unobservable no-op.
- [ ] **F31 — operational GPU failures escape the public device-error
      taxonomy.** `Problem` raises `DeviceError` for unavailable/uncompiled and
      empty-result cases, but Metal allocation/launch failures throw
      `std::runtime_error` and pass through `Problem::fill_distance_matrix`
      (`dtwc/Problem.cpp:895-924,944-996`,
      `dtwc/metal/metal_dtw.mm:1274-1295,1577-1582,1607-1612,1636-1641,
      1665-1672,1736-1751`).
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
      fixed Sakoe–Chiba window.** CLOSED by `9f78212` (shared CPU kernel plus
      public singleton/MV/AROW bypasses); independent full-matrix DP and
      exhaustive path enumeration agree. Evidence:
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
      (`docs/api-contract-2.0.md:201-208,779-787`), but Python delegates the
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
- [ ] **F38 — the CMake preset metadata guard is not fail-closed.** F16's
      registered trailing-content mutation is rejected by
      `cmake --list-presets=all` but accepted by every configure-time
      `string(JSON)` query, so successful field lookup does not prove complete
      document consumption. Independent review also found that a future
      top-level `toolchainFile` containing a UNC path can evade the current
      cache-variable and drive-letter guards. First gate: retain both exact
      mutations, require the real preset parser and permanent metadata subject
      to reject them, and use complete-input parsing plus schema-aware
      top-level field checks. Do not add a third F16 lexical sentinel.
- [ ] **F39 — F17's permanent real-CLI driver changed the tracked CMake
      manifest inventory without preregistered reconciliation.** Product commit
      `fb853eb` legitimately adds
      `tests/integration/test_cli_resume_state.cmake`; the index-owned
      supply-chain subject observes 28 manifests against F17's frozen 27 and
      fails one of 63 focused tests. Do not hide executable CMake behind another
      extension. First gate: register the exact 28-file tracked inventory,
      prove the only delta from F17's base is the named driver, prove it contains
      no remote acquisition, and explicitly authorize the checker/test update.
      Preserve the exact 39 workflow-action, 7 archive, and 1 Arrow identities,
      all F17 behavior/mutations, and the 120/120, 120/120, 122/122 full-suite
      floors.
- [x] **F40 — CLOSED 2026-09-02:** `dtwc.cluster` is now argument parsing plus
      one MEX call into C++ `dtwc::cluster`, so device, method set, `k <= N`,
      `max_iter`, `skip_*` and output naming are the C++ ones by construction
      (`.claude/reports/2026-09-02-matlab-parity.md`). The CUDA-MEX profiling gate
      below was NOT run (no CUDA MEX here); GPU routing is inherited from C++
      `configure_device`, not measured. Historical text: functional MATLAB
      `dtwc.cluster(...,'device',...)` validated Env
      but still computed through a default CPU `Problem`. The F18 audit found
      this separate Tier-1 function changes or reads the global device, records
      that name in `Result`, then creates an Auto `Problem`; FastPAM, CLARA,
      MIP, and hierarchical routes never consume the device. It exposes no
      metric parameter (`bindings/matlab/+dtwc/cluster.m:36-89`: device set/read
      36-39, unconditional local materialisation/default `Problem` 41-64, method
      routes 66-82, misleading Result device 89). F18 owns only the estimator
      named in its contract and must not claim this function fixed. First gate:
      profile the public functional PAM route in an isolated CUDA-MEX process,
      require a named DTWC kernel and exact result, then register which
      matrix-free/method routes are unsupported and must fail rather than
      silently execute CPU.
- [ ] **F41 — real Metal reachability for the MATLAB `DTWClustering` estimator
      is environment-blocked.** F18 can compile both optional branches only on
      their owning platforms and locally mutation-pin the guarded Metal
      producer, normalized metric, ordinal-zero rule, result validation, and
      no-CPU-fallback structure. This Windows host reports
      `Microsoft Windows NT 10.0.26200.0` and
      `DTWC_ENABLE_METAL:BOOL=OFF`; F12's Apple gate covers core fixed-band
      geometry, not the estimator/MEX route. **[BLOCKED-ENV 2026-07-24]** First
      gate on an Apple host with MATLAB and a fresh Metal-ON MEX: profile public
      active-global GPU-L1 and explicit GPU-SquaredL2 fits at `NInit=2`, require
      exact F18 oracle results and one Metal DTW launch per fit, then profile
      explicit CPU as a no-Metal control.
- [ ] **F42 — CUDA `CUDAPrecision::Auto` access-violates inside the Windows
      MATLAB MEX while both explicit precisions succeed.** F18's fresh
      R2024b-built CUDA MEX exits `0xc0000005` on the all-distinct four-by-two
      fixture before any Nsight kernel row. The pre-existing public
      `Problem`+CUDA route reproduces the crash without Nsight; forcing FP32 or
      FP64 instead returns exact matrix entry `D(1,2)=10`, and
      `dtwc.test.gpu()` remains green because it forces FP64. The only
      Auto-specific source step is `resolve_fp32()` calling the header-inline
      `query_gpu_config()` function-local mutex/cache
      (`dtwc/cuda/cuda_dtw.cu:75-84`,
      `dtwc/cuda/gpu_config.cuh:42-92`), but that precise cause remains
      `[inferred]`. First gate: run Auto, forced FP32, and forced FP64 in three
      isolated real-MEX children on the exact fixture, require all three exact
      matrices and zero crashes, add a direct executing configuration-query
      seam to localise the fault, then run the Auto child under
      compute-sanitizer. F42 is a prerequisite for reopening F18; substituting
      an explicit precision in F18 would not close the frozen Auto route.
- [ ] **F43 — R2024b MATLAB + llfio MEX crashes in `std::mutex` before any
      I/O.** The clean optimized llfio-ON MEX (SHA-256
      `4AE0FE630BE2F5F83A54B4BF34C81ABDE12A0C473BFE15E4325968C51F493846`)
      crashes under R2024b and passes under R2025b. PDB/import/disassembly
      evidence localises the fault to LLFIO's first Windows initialization
      `std::mutex` lock: VS 14.50 headers emit the new constexpr mutex
      representation; R2024b's private MSVCP140 14.36 dereferences the absent
      legacy vptr, while R2025b's private 14.40 runtime accepts the SRW
      representation. The crash occurs before file I/O returns or `Problem`
      publishes state [confirmed: F20 baseline].
      `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` is the registered next
      differential but remains [inferred] until an optimized R2024b/R2025b
      pair confirms it. Do not conflate with F42 — common mutex-looking
      symptoms do not prove a common source statement. First gate: reproduce
      the R2024b crash on the frozen MEX, then a single-define rebuild must
      pass both R2024b and R2025b focused F20 binding profiles 6/6 with a
      digit-identical no-product-behavior-change oracle.
- [ ] **F44 — mmap cache-path discovery can escape the typed-error
      taxonomy.** F20's review found default cache-path discovery can throw a
      raw `std::filesystem::filesystem_error` before the series router's
      `IOError` translation. First gate: force the discovery failure
      (invalid/unwritable base) in a real process and require a typed
      `dtwc::` error with an actionable message, never a raw filesystem
      exception; the inherited code must fail this gate.
- [x] **F45 — canonical LLFIO-ON public headers suppress all downstream Clang
      deprecation diagnostics.** quickcpplib's `ringbuffer_log.hpp`, included
      by LLFIO, installs a bare ignore with no push/pop. The unmasked F22
      driver reports canonical 0/33 versus llfio-OFF 24/33 [confirmed].
      First gate: bracket the dependency at both mmap public-header boundaries,
      compile a deprecated sentinel after each header, kill three registered
      mutations, and require LLFIO-ON and OFF to agree at 24/33 before F22
      alias product work. Product attempt 1 (`392d3ed`) passes all four compiler
      profiles, 3/3 mutations, exact ON/OFF 24/33 parity, and both serial
      122-test matrices. Evidence:
      `.claude/baselines/2026-07-29-f45-llfio-diagnostic-state.md`.
- [ ] **F46 — the public envelope API cannot represent or enforce its
      admissibility contract.** `compute_envelopes(..., band<0)` silently
      constructs a radius-zero envelope although negative DTW bands mean full
      DTW; bare `Envelope` stores mutable arrays without source length/window
      provenance; vector/span Keogh overloads either index to query length or
      silently truncate while ignoring the lower-array length; pointer outputs
      permit destructive aliasing; TADPole also narrows `size()` to `int`
      without a range check (`dtwc/algorithms/tadpole.cpp:149-160`). D2's exact
      fixture confirms bound 2 against full DTW 0. First gate: table-drive
      negative/full mode, too-short and
      unequal upper/lower arrays, too-narrow valid-shaped envelopes, and both
      alias classes. The replacement must expose an explicit radius/full
      descriptor, validate coverage and shape before reading, keep unchecked
      pointer kernels internal, and raise typed errors rather than coerce or
      truncate.
- [ ] **F47 — squared-L2 LB_Kim is advertised but computed in L1 units.**
      `lb_kim_valid<SquaredL2Metric>` is true and the metrics page claims
      support, while both implementations return raw absolute feature
      differences. For `{0}` versus `{0.5}`, the advertised bound is 0.5 but
      squared DTW is 0.25; `{0,0}` versus `{0.25,0.25}` gives 0.25 versus
      0.125. Current matrix consumers avoid squared Kim, so this is a public
      primitive/trait defect rather than proven live matrix corruption. First
      gate: metric-dispatch both fixtures plus exhaustive small-series
      admissibility; implement squared feature costs (or reject the trait)
      and synchronize the public documentation.
- [ ] **F48 — TADPole's empty-series upper bound classifies no-path pairs as
      neighbours.** Empty series are accepted and exact DTW returns the
      no-path maximum, but empty envelopes and the diagonal L1 upper bound both
      return zero. On `{{0},{},{}}`, band 0, `dc=1`, `k=1`, brute TADPole
      selects centre 0 while pruned TADPole selects centre 1. First gate:
      assert that exact no-path value and the inherited medoid disagreement;
      repair by rejecting empty series or bypassing every bound shortcut under
      one documented empty-distance policy, then require prune/brute identity.
- [ ] **F49 — direct pruned-matrix fill can cache a distance under the wrong
      band provenance.** `fill_distance_matrix_pruned(Problem&, int band,...)`
      accepts a radius independent of `Problem::band` and publishes the result
      into that Problem's current dense cache. Configure band 1 for
      `{0,0,1}`/`{0,1,1}` (distance 0), then direct-fill band 0 with bounds
      disabled: inherited code caches distance 1 as if current. First gate:
      require typed rejection before any matrix write and unchanged cache
      state; remove the redundant argument or enforce exact equality.
- [ ] **F50 — GPU envelope window arithmetic overflows at `INT_MAX`.** Both
      device kernels form signed `k+w+1`; host launchers accept any nonnegative
      `int` radius. A valid full-covering `INT_MAX` band can therefore wrap
      even though CPU envelope construction and repaired DTW band geometry are
      safe. Source defect is confirmed; the expected numeric CUDA/Metal
      result remains inferred until execution. First real-backend gate:
      standalone LB on `{{0,10},{10,10}}` at `INT_MAX` must equal the CPU
      global symmetric LB 10, then the thresholded matrix must preserve exact
      DTW 10. Clamp the host envelope radius to `max_L-1` with checked
      conversions; run CUDA locally and record Metal `[BLOCKED-ENV]` if no
      Apple executor exists.
- [x] **F51 — binary-v1 result checkpoints trust noncanonical wire state before
      proving its size.** The reader allocates from signed `k`/`N`, ignores
      reserved/padding bytes, accepts any nonzero convergence byte and trailing
      payload, and decodes native representations despite the documented
      little-endian format. F23 would expose this parser directly to Python.
      First gate: preserve the registered 72-byte oracle while all 85 fixed
      corruptions return false, throw zero exceptions, and leave the destination
      unchanged; a separate bounded `k=257` allocation probe must prove exact
      size before count-derived allocation. Require explicit LE
      integer/binary64 codecs, save-load-save byte identity, and all seven F17
      semantic-invalid files still reach the CLI's contextual validator. These
      cases seed rather than replace the later randomized checkpoint/config
      robustness lens.
- [ ] **F52 — MATLAB binary-checkpoint load failures bypass the frozen I/O
      taxonomy.** `cmd_load_binary_checkpoint` throws `std::runtime_error` when
      the production reader returns false, so the MEX catch ladder publishes
      `dtwc:runtime` instead of contract-required `dtwc:ioError`
      (`bindings/matlab/dtwc_mex.cpp:965-972,1510-1519`). First gate: drive
      missing and structurally malformed files through the public
      `dtwc.load_binary_checkpoint` wrapper under both installed MATLAB
      releases; each must raise exact identifier `dtwc:ioError`, retain an
      actionable path-bearing message, and never crash or return a default
      result.
- [ ] **F53 — MATLAB binary-checkpoint result conversion accepts incomplete or
      non-integral state and its parity test masks field/order loss.**
      `mx_to_clustering_result` requires only labels/medoids, defaults three
      missing fields, narrows arbitrary doubles to `int`, and treats every
      nonzero convergence value as true; the live test compares only label
      count and sorted medoids (`dtwc_mex.cpp:908-940`;
      `test_contract_parity.m:527-537`). First gate: table-drive each missing
      field, fractional/non-finite/out-of-int32 indices and iterations,
      noncanonical convergence, and nonscalar fields; reject before filesystem
      effects, then round-trip a non-degenerate result with all five fields and
      medoid order exact on R2024b and R2025b.
- [x] **F54 — the live Enhanced pruning cascade does not take the documented
      maximum with LB_Keogh.** `LowerBoundStrategy::Enhanced` constructs a
      Keogh envelope but evaluates only symmetric LB_Enhanced. For the
      registered `{C,A,B}` ordering at radius 1, the first two exact distances
      are zero, while the final pair has Kim 0, Enhanced 0, Keogh 10, and DTW
      20. The inherited route therefore performs all three full evaluations.
      First gate: preserve the exact matrix while both the direct and public
      `Problem` routes report exactly 3 pairs, 1 envelope prune, 1 early
      abandon, and 2 full evaluations under serial OpenMP. Repair by taking
      `max(Enhanced,Keogh)`; do not claim saved exact-matrix work because the
      abandoned pair is recomputed for publication.
      **Repair retained 2026-07-30:** `d09cf9c` activates both bounds and takes
      their maximum; `53506a9` corrected the test's output literal (attempt 1
      conservatively consumed on that literal alone). Focused attempt 2 PASS
      115/115 with the exact registered counters. **CLOSED 2026-08-09:** all
      three serial matrices and the Arrow runtime subjects pass; evidence is
      the D3 baseline.
- [x] **F55 — the local Webb implementation and its MinLR omission are
      misidentified.** `lb_webb` is the paper's all-index `LB_Webb_NoLR`
      bridge/correction formula plus a conservative tail-flag cap, not full
      Algorithm 2 with `MinLRPaths`. The source and changelog claim that the
      omission can only loosen the bound, but Webb and Petitjean report Wafer
      tightness 0.96904 for NoLR versus 0.96891 for full Webb, disproving that
      ordering. First gate: compare the live formula with an independent
      direct-predicate NoLR oracle, reach all four correction branches, prove
      the tail-cap inequality separately, and correct every provenance and
      ordering claim. Do not attribute the repository's Enhanced/Keogh
      counterexamples to the paper.
      **Corrections retained 2026-07-30:** `6abff20` fixes provenance across
      source contracts, public enum/metric page, changelog, lessons, legacy
      test commentary, and the 2026-07-08 run-log corrigendum;
      `check_docs_contract.py` passes. **CLOSED 2026-08-09:** the exhaustive
      direct-predicate/branch/tail ledger and all three serial matrices pass;
      evidence is the D3 baseline.
- [ ] **F56 — Python binary-checkpoint failure formatting is not total over
      accepted filesystem paths.** The binding converts `std::filesystem::path`
      with `u8string()` and the exception translator later uses
      `PyErr_SetString`; POSIX surrogateescaped non-UTF-8 filenames can yield
      invalid UTF-8, while Windows lone surrogates can fail before native I/O.
      First gate: on POSIX, construct a missing path from raw undecodable bytes
      through `os.fsdecode`; on Windows, probe an unpaired surrogate where the
      runtime permits it. Every reachable case must raise exact
      `dtwcpp.IOError` without a secondary Unicode/conversion exception and
      retain an unambiguous reversible path representation. Record
      platform-impossible cases as `[BLOCKED-ENV]`, not as coverage.
- [x] **F57 — CPU LB_Webb window arithmetic has signed overflow at a valid
      full-covering `INT_MAX` radius.** `twoW=2*w` and signed free counters can
      overflow even though a radius larger than `n-1` is geometrically
      equivalent to `n-1`. On `A={-2,-2}`, `B={0,0}`, inherited wrapped flags
      can double-count and return inadmissible L1/squared values 8/16 instead
      of global DTW 4/8. First gate: require exact
      `F57_LB_WEBB_INTMAX` results, global-radius parity and admissibility in a
      non-skippable target, then repeat under WSL UBSan. Normalize the effective
      radius to `min(max(band,0),n-1)` and use overflow-safe unsigned
      geometry/counters. F46 retains public envelope shape/provenance,
      negative-mode, aliasing, and unrepresentable-length ownership; F50
      retains device arithmetic.
      **Repair retained 2026-07-30:** `29f9103` saturates the CPU Webb/Enhanced
      radius and makes the doubled radius and free counters unsigned and
      saturating; `13cd4f6` added the analytic nonconstant discriminator.
      Focused 24/24 passes normally and under WSL UBSan
      (`halt_on_error=1`) with zero diagnostics. **CLOSED 2026-08-09:** the
      non-skippable 24-assertion target passes in all three serial matrices and
      the Arrow runtime selection; evidence is the D3 baseline.

Remaining lenses (verbatim from 8.2 — each is one round-item; run all, round
after round, to the exit band):

- [ ] **Property/metamorphic fuzz harness** (new `tests/fuzz/` or Catch2 generators, seeds committed): classify each generated case against the subject's documented contract before applying an invariant; out-of-contract NaN/Inf, empty, mixed-length, metric, or mode cases require a typed rejection or an explicitly documented behavior, not a borrowed theorem. Within each valid domain check symmetry `d(x,y)=d(y,x)`; `d(x,x)=0`; `DTW_band ≥ DTW_full` and monotonicity in band; `DTW_I ≤ DTW_D`; `dtwFull_eap == dtwFull_L`; prune==no-prune digit identity (TADPole, pruned matrix); MSM/TWE triangle inequality; and checkpoint save→load→identical state. Apply the D3 LB invariants only to finite nonempty equal-length scalar L1/unrooted squared-L2 inputs with a shared saturated window and valid envelope provenance: every bound is `≤ DTW`, the local directional NoLR-plus-tail-cap result is `≥` matching-direction Keogh, and the symmetric maximum is `≥` symmetric Keogh. Every violation is a bug or a documented, justified exclusion.
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

## Binding decisions (digest — append NEW decisions here; verbatim prose is in the dated `.claude/PLAN-archive-*` files, most recently the two 2026-08-09 archives)

- Decisions through F23's 2026-07-30 path residual remain binding and are
  archived verbatim in `.claude/PLAN-archive-2026-08-09-decisions.md`. In
  particular, `.claude/MISSING.md` / `READ.md` were retired by `0449f7c`;
  do not recreate them. F11/F16/F17/F18/F20/F22 retain their registered
  falsification and attempt caps; local-only/no-publication restrictions and
  every M-series authorization remain unchanged.
- 2026-07-30 (D3 registration): Treat the production `lb_webb` as
  `LB_Webb_NoLR` plus a separately proved conservative tail cap; no ordering is
  claimed between it and full Algorithm 2. D3 owns exact-arithmetic
  admissibility and ordering for finite equal-length scalar L1 and unrooted
  squared-L2 inputs with a shared saturated window. F54, F55, and F57 are
  closure prerequisites discovered by the pre-execution audits. Two isolated
  non-skippable targets bind the exact D3 and F57 markers, serial live-cascade
  reachability, `INT_MAX` behavior, and a two-product-attempt cap before any
  decisive execution. F46/F50 and D17 retain their separate public-envelope,
  device, and floating-threshold scopes. Evidence:
  `.claude/baselines/2026-07-30-d3-lb-enhanced-webb.md`.
- 2026-07-30 (F55 provenance correction): Preserve the public `Webb` API name
  for compatibility, but identify its implementation as all-index
  `LB_Webb_NoLR` plus a separately proved tail cap. Only
  `production <= exact-predicate NoLR` has a loosening direction; the paper's
  Wafer result rejects a universal NoLR/full-Algorithm-2 order. Enhanced
  dominates matching-direction Keogh at effective `V=1`; the no-ordering
  statement at `V>=2` comes from the D3 exact witnesses, not Tan et al. This
  decision explicitly supersedes the stale Algorithm-2, blanket-Enhanced, and
  universal-LB-fuzz wording preserved in
  `.claude/PLAN-archive-2026-07-20-phases0-9.md` and
  `.claude/PLAN-archive-2026-07-27-r0-f20.md`; those verbatim archives are not
  rewritten. F55 closes only with the D3 derivation and integration gates.
  Evidence: `.claude/baselines/2026-07-30-d3-lb-enhanced-webb.md`.
- 2026-08-09 (D3/F54/F55/F57 closure): Promote the preregistered post-D3
  floors after serial runtime adjudication, not inventory inference:
  canonical 125/125 with six exact skips, llfio-OFF 125/125 with nine, and
  Arrow-ON 127/127 with eight. The separate Arrow selection executed D3
  115/1, F57 24/1, reader 390/11, and all four real-CLI markers. The exhaustive
  direct-predicate ledger confirms the most-at-risk clipped-tail claim without
  moving its band. Close D3 and its F54/F55/F57 prerequisites; retain F46/F50
  and D17 ownership of public provenance, device arithmetic, and floating
  thresholds. Evidence:
  `.claude/baselines/2026-07-30-d3-lb-enhanced-webb.md`.
- 2026-09-02 (quality/correctness campaign, Claude orchestrating Opus agents):
  four read-only reviews + four adversarial re-reviews + fixer rounds closed
  silent-wrong-answer bugs (MATLAB MIP ignored k; Benders incumbent-as-LB and
  heuristic-as-complete; MV+Interpolate/SoftDTW channel flattening; TADPole f32
  UB; degenerate silhouette/DB; CUDA zeros with no device; read_distance_matrix
  false success; metric-less checkpoint fingerprint) and lock-free violations
  (unnamed `omp critical` ×4, per-call GPU-config mutex). New rule from Volkan:
  design is always lock-free and high-performance. Python `Problem` is
  single-thread-per-instance with the GIL released consistently. rapidcsv
  dependency removed; dead `medoid_utils` helpers and `core::*Dist` functors
  deleted with byte-identical objects. Serial gates: 128/128, 128/128, 130/130
  (6/9/8 skips), CUDA 4/4 ran, Python 1045/0. PR #32 (Kasper Westman) ported
  hunk-by-hunk with a Windows-correct adaptation of its path pin, then merged
  for attribution. `[BLOCKED-ENV]`: HiGHS-enabled MEX crashes MATLAB on any MIP
  solve. Evidence: `.claude/baselines/2026-09-02-quality-campaign.md`,
  `.claude/reports/2026-09-02-*.md`.
- 2026-09-02 (second pass — parity / checkpoint / portability): MEX HiGHS crash
  root-caused (MATLAB private msvcp140 14.36 vs constexpr `std::mutex`; tree-wide
  `_DISABLE_CONSTEXPR_MUTEX_CONSTRUCTOR` for MATLAB builds), MATLAB MIP/PDLP run
  on R2024b+R2025b, `matlab_suite` in CTest (123 passed, floor 121);
  `[BLOCKED-ENV]` retired. Mid-fill interval checkpoint implemented
  (`Problem::checkpoint`, rows between saves, one retained generation) and the
  latent resume bug fixed (brute-force `resize(N)` wiped restored matrices).
  F15 generator made portable (`genrand_res53`, one fingerprint). YAML CLI
  config removed. CasADi parity: `skip_rows`/`lr_max_nodes` everywhere; Python
  and MATLAB Tier-1 re-routed through C++ `load`/`cluster` (F18/F40 closed;
  7 Python + 9 MATLAB drifts); Tier-1 routes made side-effect-free (Lloyd wrote
  CWD-relative `./results`); UTF-8 names end to end. Serial gates: 130/130,
  130/130, 132/132 (6/9/8 skips), CUDA 4/4, Python 1112/16/0, MATLAB
  124/123/0/1 both releases. Evidence: `.claude/reports/2026-09-02-*.md`,
  `.claude/summaries/handoff-2026-09-02-parity-checkpoint.md`.

## Progress log (append-only; older entries in the archive)

- Entries from 2026-07-23 (PLAN v2.0 adoption, R0/R1) through 2026-07-24
  (F11-F20 registrations, attempts, and verdicts) are archived verbatim in
  `.claude/PLAN-archive-2026-07-27-r0-f20.md`; the F20 verdict, F21
  registration/closure, and record-hygiene-restoration entries are archived
  verbatim in `.claude/PLAN-archive-2026-07-29-decisions.md`.
- Detailed F22/F45 registration, attempt, mutation, documentation, and final
  gate entries are archived verbatim in
  `.claude/PLAN-archive-2026-07-29-decisions.md`; the live F22 task and final
  binding decision preserve its permanent 33/46 falsification and attempt cap.
- Reconciliation/progress entries from 2026-07-29 through the 2026-07-30 D3
  in-flight handoff and v2.2 slimming are archived verbatim in
  `.claude/PLAN-archive-2026-08-09-decisions.md`.
- 2026-08-09 (D3/F54/F55/F57 CLOSED): fail-closed adjudicator self-test
  rejected 25/25 transcript mutations; focused canonical subjects passed 5/5;
  serial canonical/llfio-OFF/Arrow-ON matrices passed 125/125, 125/125, and
  127/127 with exact 6/9/8 skips; the Arrow executable selection passed 7/7
  with D3 115/1, F57 24/1, reader 390/11, and four exact real-CLI markers.
  AGENTS floors, derivation index, and CHANGELOG were promoted only after
  execution. Cursor: GPU-LB CUDA cluster F27/F29/F47/F50, then D4. Evidence:
  `.claude/baselines/2026-07-30-d3-lb-enhanced-webb.md`.
- 2026-08-09 (rule-12 slimming): F17/F18/F20/F23 live bodies and older
  Binding-decision/Progress digests moved verbatim to the two 2026-08-09 PLAN
  archives; exact line-for-line comparison against `d2e8834` passes for all
  six blocks. The live PLAN is again below its ~1,200-line threshold and keeps
  current status, attempt caps, record-retirement markers, and evidence links.
- 2026-09-02: quality campaign landed (see Binding decisions); floors promoted
  to 128/128, 128/128, 130/130. Cursor unchanged: GPU-LB CUDA cluster, then D4.
