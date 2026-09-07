# Handoff — 2026-09-07 — DTWC++ 2.0 quality & design campaign (spec stage)

Base: branch `Claude`, HEAD `a31956e` (Windows-CI test fixes, committed this session).
Campaign branch `design-2.0` NOT yet created (created at W0 start, after spec approval).

## Accomplished

- Windows unit CI: three failures root-caused and fixed (bitwise barycenter fingerprint;
  F22 probes blind on MSVC — address-taking not diagnosed, `C4996` wording, llfio STL4047;
  CRLF checkout vs pinned SHA-256). MSVC Debug 131/131. Commit `a31956e`.
- Scope decisions taken by Volkan (see spec header and memory
  `feedback_design_campaign_method.md`).
- As-is maps (read-only Opus agents, all `[confirmed file:line]`):
  `.claude/reports/2026-09-07-asis-{core,orchestration,algorithms-mip,io-cli-build,backends}.md`
  and `…-tests-taxonomy-{unit-flat,unit-subdirs,adversarial-integration}.md`; mechanical
  arbiters `…-include-graph.tsv`, `…-dupscan.txt`.
- Ideal design written BEFORE reading the maps, then diffed:
  `.claude/specs/2026-09-07-design-2.0-campaign.md` (Parts I–IV) and the row-level
  `.claude/specs/2026-09-07-diff-ledger.md` (129 rows: C/O/A/B/T/G).

## Decisions pending Volkan (spec III.10)

C-05 Auto=BruteForce · C-09/C-19 MV LBs and L2 token · C-11 move foundation headers ·
C-13 custom distance vs mmap cache · O-09 drop `Problem_IO` legacy filenames ·
B-14 option prefixes · B-19 keep shims, collapse F22 apparatus · A-05 keep CLARANS ·
G-07 defer MPI wiring · T-* test merges/deletes as a class · T-04/T-12 slow tests.

## Two findings that are gates, not cleanups (go first, W0)

- `tests/cpp_conformance.cpp:212` regenerates the reference then compares against it.
- 114/125 registered tests have no pass floor (`SKIP_RETURN_CODE 4` blanket).

## Next steps

1. Volkan reviews the spec; DECIDE rows resolved; spec status → APPROVED.
2. `superpowers:writing-plans` for W0 only (T-15, T-16, build hygiene, layer report,
   benchmarks, IPO check, conformance snapshot). Then per wave.
3. Create branch `design-2.0`; PLAN.md decision-log digest entry; AGENTS.md floors
   updated when W0 changes the inventory.

## Open questions

- Latency- vs memory-bound: both regimes true; PMU task registered in W8 docs.
- Whether IPO inlines `dist_by_ind` into algorithm TUs (W0 measures).
- `.dtws` crash-consistency parity (deferred, PLAN).

## W0 plan (written 2026-09-07, after the spec, before approval)

`.claude/plans/2026-09-07-w0-baseline-tooling.md` — 14 tasks, TDD steps, exact
code: baseline logs -> `dtwc_add_test` + generated floors -> conformance
read-only/hash-pinned + regen tool -> build hygiene -> preset twins + table F16
-> header file set -> fast-math contract -> vendored CPM -> CI legs (bare core,
dev-warnings ratchet, lint) -> layer report -> allocation guard -> two
benchmarks -> IPO report -> exit gate/records. W0 does not depend on any III.10
decision; B-14 (option prefixes) is the only W0-adjacent DECIDE and is left out.

## Resume point

Nothing implemented. Spec DRAFT v0 complete and self-reviewed; awaiting review.
W0 plan complete and self-reviewed; execution starts only after the spec is
approved (choose subagent-driven or inline execution then).
Uncommitted: the two spec files, nine reports, two arbiter files, the W0 plan,
this handoff.
