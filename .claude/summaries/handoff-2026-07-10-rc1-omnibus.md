# Handoff 2026-07-10 — rc1 omnibus reconstruction and Phase 8 entry

## Goal

Reconstruct the paper trail omitted by commit `8debf1d` without retroactively
turning unrecorded claims into evidence. This handoff distinguishes what the
commit changed, what the 2026-07-10 adversarial review independently reproduced,
and what remains open in Phase 8.

## What `8debf1d` shipped

`8debf1d` (`Added more algorithms and tests.`) was one 103-file omnibus commit
on top of `3f827c2`. Its material scope included:

- **Task 5.8:** OneBatchPAM C++ implementation and API surface, deterministic
  tests, and matrix-free Python method dispatch.
- **Task 5.9:** DBA, an SSG-style DTW averaging path, soft-DTW barycenters, and
  barycenter k-means, with C++ tests and Python bindings.
- **Task 5.10:** the scikit-learn-compatible `DTWCKMedoids` estimator and tests.
  The separate conda-forge feedstock remained an explicitly deferred external
  2.1 item.
- **Task 5.11:** an OpenMP schedule sweep and hot-kernel timing record. The
  existing adaptive schedule was retained; privileged PMU/cache collection was
  unavailable, so no SIMD layer was authorized. The advisory record now lives
  at `.claude/baselines/2026-07-10-openmp-profile.md`.
- **Phase 6 / rc1:** `VERSION` moved to `2.0.0rc1`; CMake, wheel/sdist, MATLAB,
  release-archive, attribution, and release-workflow surfaces were changed; a
  wheel smoke path and release notes were added. Production tag and PyPI upload
  were not authorized by this work and remain explicit operator actions.
- **Phase 7:** a Hugo documentation site, Tier-1/Tier-2 API pages, guides,
  migration and release pages, quickstarts, LR-core math documentation, UCR
  benchmark documentation, generated-source drift checks, and internal-link
  validation tooling.
- **Tier-1 API:** owning C++ `device` / `load` / `cluster` / `Result` interfaces,
  binding support, and a C++ quickstart were added alongside the Python and
  MATLAB surfaces.

The commit also absorbed fixes and contract changes discovered while integrating
those tasks, including residual Metal pair-offset widening, stricter requested-
device and solver failure behavior, and a distance-function rebind race fix.
Those behavior changes did not all receive matching changelog or migration
entries; Phase 8 tracks the omissions explicitly.

## Evidence boundary

At commit time, the plan/changelog described local floors and packaging checks,
but no corresponding Phase 5.8–5.11 / Phase 6 / Phase 7 run-log was committed.
Those descriptions are historical claims, not durable gate evidence.

The subsequent adversarial review independently reproduced only these full
suite results:

- C++: `ctest` **99/99, 0 failed** in `build/highs-1151`.
- Python: **407 passed, 11 skipped**, exit 0. The recorded **409 passed** claim
  was stale by two tests and must not be repeated as the reproduced floor.

The review did **not** produce a committed artifact establishing the claimed
MATLAB 61/61, TestPyPI dry-run, installed-wheel smoke, sdist, archive, or docs
gate results. Task 8.0 subsequently reran the applicable local checks and
captured their provenance and decisive output in
`.claude/baselines/2026-07-10-phase67-floors.md`; the log preserves every
disagreement rather than normalizing it away.

No build/runtime test suite, benchmark, or simulation was run while
reconstructing this documentation; validation was limited to static consistency
and byte-preservation checks.

## Adversarial review outcome

Three review agents and one gate agent found no test weakening and no confirmed
core mathematical error after re-derivation. In particular, they found the
Metal int64 widening complete at the reviewed residual sites and the LR-core
derivation sound. The runtime-loudness tests were strengthened to match the
hard-error contract. That positive result does not erase the following confirmed
defects, coverage gaps, and governance failures.

| ID | Confirmed finding recorded for Phase 8 |
|---|---|
| H1 | Python matrix-free methods silently drop the requested `band`. |
| H2 | `DTWCKMedoids.__sklearn_tags__` omits the precomputed-matrix `pairwise` tag used by scikit-learn 1.6+. |
| H3 | The soft-DTW adjoint recursion has no non-trivial gradient test; the existing 1×1 case bypasses it. |
| H4 | The FROZEN API contract was changed non-additively without the required decision entry, and its governance clause was removed. |
| M1–M5 | Device-alias/ordinal parity, matrix-free migration docs, unsupported barycenter settings, the cited SSG multiplicity, and two divergent soft-DTW cost implementations need repair or an explicit decision. |
| M6–M8 | The hidden OneBatchPAM oracle is degenerate, reference-code-derived weighting is not yet paper-verified, and three behavior changes lack changelog/migration coverage. |
| M10–M13 | The Metal offset test is tautological; TADPole can still allocate dense N² storage after bypassing mmap; `auto` can select a GPU-incompatible method; Tier-1 default seeds diverge across languages. |
| L1–L7 | Mutable workflow/archive inputs, a degraded LR-core docs drift gate, silent OneBatchPAM batch-size inflation, lost serial-MATLAB coverage, deprecated/inconsistent seeding, barycenter hot-path allocations, and ambiguous test-floor records remain open. |

Process findings were independently actionable: `8debf1d` replaced the required
per-task commits with one vague omnibus commit, omitted the session handoff and
lessons, asserted gate results without committed run-logs, and edited a FROZEN
contract without a decision entry. The new lessons record why each pattern is
unsafe rather than treating this as a naming-only problem.

## Current Phase 8 state

Commit `b160269` opened Phase 8 and made a CLEAN Phase 8 exit a prerequisite for
the final `2.0.0` tag. At Task 8.0 close:

- **8.0:** complete. This handoff, three lessons, verified bibliography, OpenMP
  baseline relocation, and the durable Phase 6/7 floor log are committed
  together. The reconciled local results are native 99 registered / zero failed
  with LLFIO ON (six skips) and OFF (eight skips), Python 407 passed / 11 skipped
  from a fresh isolated wheel, MATLAB 61/61 with 24 OpenMP threads, and green
  local wheel/sdist/publish-command/quickstart/docs/archive checks. The hidden
  50k OneBatchPAM simulation is explicitly not part of that normal floor and
  remains owned by Task 8.1-M6 after its fixture is repaired.
- **8.1:** all H/M/L review findings above remain governed by the itemized plan.
  This handoff claims none of them fixed.
- **8.2:** the systematic sanitizer, fuzz, differential, determinism, overflow,
  error-path, compiler-warning, CUDA, MEX, and solver-edge sweep has not yet
  established its two consecutive clean rounds.
- **8.3:** the no-op fingerprint oracle and behavior-neutral simplification pass
  have not yet been completed.
- **8.4:** the full configuration matrix, fresh language floors, CI wiring,
  rc2 changelog, and final adversarial CLEAN verdict remain release blockers.

External operator gates also remain external: hosted cross-platform runs,
quiet-host spot benchmarking, Oxford ARC, Metal runtime verification, and any
production tag/TestPyPI/PyPI action. Do not relabel local inspection or advisory
shared-host timing as those gates.

## Recommended next step

Execute Task 8.1 one finding and one conventional commit at a time, preserving
the fresh configuration-specific floors above, before entering the two-round
bugfinding sweep.
