# Handoff — 2026-07-29 — F21 C++ snake_case entry points

## Accomplishments

- Re-read `AGENTS.md`, live `PLAN.md`, the complete `LESSONS.md`, the current
  TODO/design records, F20 handoff, killed ideas, both relevant archives, the
  frozen API contract, source, tests, and rename history.
- Confirmed the tree is clean at
  `ab41930b35e1cce25af2b785c154e92628822fb1`.
- Confirmed F21 is the binding campaign cursor and has no killed-idea conflict.
- Captured a clean canonical baseline: 122/122, zero failed, with the exact six
  expected capability skips.
- Registered the expected-red, exact-signature, runtime-state, deprecation,
  mutation, docs, and full-gate bands in
  `.claude/baselines/2026-07-29-f21-cpp-renames.md`.
- Ran the corrected inherited-tree public-header probe. Canonical compilation
  exits 1 with all four names diagnosed (including the private `start_row`
  collision); the otherwise-identical legacy control exits 0. The first
  ad-hoc command's missing RapidCSV include is retained as invalid harness
  evidence, not misreported as an F21 result.
- Added the permanent public-header fixture red-first in `a48635b`.
- Implemented the four canonical-owned entry points and deprecated forwarding
  aliases in product attempt 1 (`5e4a7b6`), migrated ordinary repository calls,
  updated the changelog and generated contract pages, and hardened the
  documentation drift checker.
- Confirmed the exact 12/12 signature surface and runtime state ledger in both
  canonical and llfio-OFF builds. Each printed the registered marker and
  `All tests passed (81 assertions in 2 test cases)`.
- Confirmed canonical-only compilation is warning-clean and the legacy-only
  probe emits all four exact `use <canonical>` diagnostics.
- Added the permanent 12-case mutation harness in `36b9c99`. Its decisive run
  killed 8/8 compile mutants and 4/4 runtime mutants, passed both controls,
  reported zero survivors, emitted no stderr, and restored exact source bytes.

## Decisions

- Implement exactly the frozen setter surface; do not add canonical no-argument
  loader getters.
- Mirror both existing path overloads (`fs::path` and C-string).
- Rename both coupled private loader fields with trailing underscores because
  the inherited `start_row` field otherwise makes `start_row(int)` ill-formed.
- Canonical functions own behaviour; the four legacy names are deprecated
  inline forwarders. F22 retains the exhaustive cross-language warning audit.
- Reuse the existing `unit_test_DataLoader` target so CTest and tracked-CMake
  inventories do not move.
- Migrate ordinary repository-owned call sites to canonical setters; retain
  old-name execution only in the focused compatibility fixture.

## Exact resume point

Commit the R5 evidence, then run R6 from the immutable product commit: full
canonical, llfio-OFF, and Arrow-ON rebuild/CTest matrices; assert the Arrow
reader executed; rerun documentation, deprecation/call-site hygiene, supply
chain, and inventory checks. If all registered bands pass, close F21 in a
separate bookkeeping commit and advance the campaign cursor to F22. Product
attempts consumed: 1/2.

## Open risks

- A test that calls canonical then legacy path setters without poisoning both
  globals before each call can false-green a no-op alias.
- A canonical `fs::path` overload alone accepts string literals by conversion
  but does not preserve the frozen exact `const char*` function-pointer surface.
- The most likely claim to be wrong is that the existing-target assertion
  ledger will be exactly 79; the registered acceptance is a floor, and the
  exact post-build count must be recorded rather than inferred.
