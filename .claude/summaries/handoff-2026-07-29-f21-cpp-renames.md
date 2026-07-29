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

Commit this preregistration, add the focused public-header fixture without
product changes, and run the registered inherited-tree red/legacy-control
probe. Product attempts consumed: 0/2.

## Open risks

- A test that calls canonical then legacy path setters without poisoning both
  globals before each call can false-green a no-op alias.
- A canonical `fs::path` overload alone accepts string literals by conversion
  but does not preserve the frozen exact `const char*` function-pointer surface.
- The most likely claim to be wrong is that the existing-target assertion
  ledger will be exactly 79; the registered acceptance is a floor, and the
  exact post-build count must be recorded rather than inferred.
