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
- Committed the three-language red-first fixtures in `b70a259`.
- Reproduced Python's exact 18 collected / 16 failed / 2 passed inherited
  ledger and MATLAB's exact 15/15 behavior-but-0/15-warning marker on both
  installed releases.
- Exposed an additional public-header defect instead of masking it: canonical
  LLFIO-ON compilation reports 0/33 legacy diagnostics because quickcpplib
  leaks a Clang diagnostic ignore; llfio-OFF reports the expected 24/33.
- Localized and registered that prerequisite as F45 in
  `.claude/baselines/2026-07-29-f45-llfio-diagnostic-state.md`.
- Closed F45 in product commit `392d3ed`: one product-owned LLFIO wrapper
  restores Clang diagnostic state at both public-header boundaries.
- Passed the four-profile actual ON/OFF compiler gate, killed all 3/3
  registered mutants with exact source restoration, and proved both F22
  drivers now report the identical 24/33 expected-red ledger.
- Rebuilt and passed canonical 122/122 with six exact capability skips and
  llfio-OFF 122/122 with nine. A concurrent-matrix run was rejected as harness
  evidence after shared source-root `CSV` collisions; the two affected tests
  passed 2/2 in each unchanged build before both serial full passes.
- Hardened the F45 raw-include inventory in `f7f91d6` so trailing line/block
  comments cannot hide an include and the audit proves its own controls.
- Committed the exhaustive F22 C++ behavior fixture and combined
  diagnostic/behavior launcher in `c210504` without changing the 122-test
  CTest inventory.
- Forced the generator-independent compiler-probe fallback: it reproduced the
  exact 24/33 inherited diagnostic ledger and exact nine silent entities.
- Ran the canonical and llfio-OFF behavior executables. Both passed 229
  assertions in five cases and printed the exact 33/33 behavior, 4/4 field,
  7/7 I/O, 6/6 file, and 2/2 stdout marker with zero skips.
- Ran the preferred compiler driver and combined CTest entry in both builds.
  Both drivers reproduced 24/33; both CTest entries failed only because the
  registered 33/33 product diagnostic marker is still absent.

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
- Do not filter LLFIO out of the F22 compiler driver. F45 must contain the
  third-party diagnostic state at both product include boundaries, after which
  canonical LLFIO-ON must match llfio-OFF at 24/33 before F22 resumes.
- Use the real `build/nollfio` compile database for the F45 OFF profile; a
  canonical-context flag filter is not configuration evidence.
- Run full configured CTest matrices serially while their generated metadata
  gives every test the repository source root as its working directory.
- Keep one CTest entry by using a tests-only Python 3.9 launcher. On generators
  without `compile_commands.json`, force three marked, build-local
  `EXCLUDE_FROM_ALL` object probes instead of weakening the compile context or
  requiring a newer-than-3.26 CMake launcher property.

## Exact resume point

Begin F22 product attempt 1/2 from clean commit `c210504`. Implement the C++
canonical-owned I/O and deprecated fields/functions, native and pure-Python
warning paths, and MATLAB property/function warnings in disjoint file scopes.
Then run the registered focused gates before creating any mutation runner.

## Open risks

- Inline canonical C++ accessors that touch deprecated fields contaminate every
  canonical consumer unless the suppression is isolated out of line.
- Warning inside a shared MATLAB MEX command contaminates canonical wrappers.
- A stale Python extension can false-green the pure-Python aliases; the final
  gate requires a fresh native discriminator and matching hashes.
- CPython 3.13 calls module `__getattr__` twice for `from ... import` and star
  import. Suppress only importlib's `_handle_fromlist` preflight; never cache
  `ClusterResult`, or later lookups become silently non-deprecated.
- The most likely decision to need later revision is write-only warning
  semantics for the four readable PascalCase MATLAB config properties.
