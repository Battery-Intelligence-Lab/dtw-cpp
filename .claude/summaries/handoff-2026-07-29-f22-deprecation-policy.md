# Handoff — 2026-07-29 — F22 deprecation policy

## Accomplishments

- Re-read the live PLAN, complete working rules, frozen API contract, killed
  ideas, relevant F19/F21 decisions, current handoff, and implementation
  history before touching product code.
- Confirmed a clean base at
  `5352bc0b0e459011ae1dbe8ef38b38262306459d`.
- Rebuilt and ran the canonical baseline: 122/122, zero failed, with exactly
  the six registered capability skips.
- Independently audited C++, Python, and MATLAB alias inventories and ordinary
  call sites.
- Registered the inherited-red, exact diagnostics, behavior identity,
  canonical silence, fresh binding, two-release MATLAB, mutation, docs, and
  three-build full-gate bands in
  `.claude/baselines/2026-07-29-f22-deprecation-policy.md`.

## Decisions

- Retain the F19 raw `int` field shape; annotate the two fields directly and
  move canonical accessors out of line under narrow suppression.
- Invert the seven C++ I/O names so canonical functions own behavior.
- Include Python `Problem.cluster_size` and both distance aliases in F22.
- Preserve exact `ClusterResult is Result` identity through module
  `__getattr__`; do not use a subclass, proxy, or factory.
- Treat MATLAB `set_distance_matrix` as canonical and silent; only the getter
  is an alias.
- Warn on assignment through the four PascalCase MATLAB configuration
  properties. Their reads remain functional because no canonical read
  replacements were frozen.
- Preserve the two named retained-red F18 MATLAB tests; F22 full-suite success
  is 85/82/2/3 after adding its one case, not a false zero-failure claim.

## Exact resume point

Commit this registration as its own `docs:` commit. Then add the permanent C++,
Python, and MATLAB fixtures red-first without product edits; capture the exact
inherited verdicts against R1 and commit the retained evidence before beginning
product attempt 1/2.

## Open risks

- Inline canonical C++ accessors that touch deprecated fields contaminate every
  canonical consumer unless the suppression is isolated out of line.
- Warning inside a shared MATLAB MEX command contaminates canonical wrappers.
- A stale Python extension can false-green the pure-Python aliases; the final
  gate requires a fresh native discriminator and matching hashes.
- The most likely decision to need later revision is write-only warning
  semantics for the four readable PascalCase MATLAB config properties.
