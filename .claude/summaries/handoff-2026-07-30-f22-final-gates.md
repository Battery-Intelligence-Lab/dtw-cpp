# Handoff — 2026-07-30 — F22 final gates

## Accomplishments

- Re-read `AGENTS.md`, the complete live `PLAN.md`, `LESSONS.md`,
  `CITATIONS.md`, `design.md`, `TODO.md`, the named historical plan archive,
  the latest F22 handoff, and the complete F22 evidence record.
- Confirmed a clean start at
  `e0df96ea09d5f7c7a6dad620c399ec12edbcbb32`.
- Preserved the exhausted F22 C++ mutation verdict as FALSIFIED 33/46; a third
  run remains prohibited.
- Registered the unchanged final native/Python/MATLAB bands and completed the
  adversarial pre-run inventory, configuration, marker-reachability, and
  artifact-provenance checks in
  `.claude/baselines/2026-07-30-f22-final-gates.md`.
- Verified locally that Claude Code 2.1.220 accepts explicit
  `--model fable --effort medium`; the final whole-campaign review will use
  that exact model/effort with read-only tools and will not accept completion
  without a structured CLEAN verdict.
- Clean-first rebuilt the canonical HiGHS/llfio matrix, settled it to
  `ninja: no work to do`, and passed 122/122 serially with the exact six
  capability skips.
- Re-ran the hardened F22 entry verbosely: diagnostics 33/33, behavior 33/33,
  229 assertions / 5 cases, zero skips.
- Reconfigured and clean-first rebuilt the llfio-OFF matrix, settled it to
  `ninja: no work to do`, and passed 122/122 serially with the exact nine
  registered capability skips.
- Re-ran the llfio-OFF F22 entry verbosely: diagnostics 33/33, behavior 33/33,
  229 assertions / 5 cases, zero skips.
- Reconfigured and clean-first rebuilt the PyArrow 23.0.1 Arrow-ON matrix,
  settled it to `ninja: no work to do`, and passed 124/124 serially with the
  exact eight registered capability skips.
- Proved through CTest's JSON test model that the reader receives all three
  registered runtime-path modifications; then executed its 390 assertions /
  11 cases with no skip.
- Re-ran the Arrow-ON F22 entry verbosely: diagnostics 33/33, behavior 33/33,
  229 assertions / 5 cases, zero skips.

## Decisions

- Run all three native matrices and both MATLAB releases serially because
  configured tests share source-root-relative artifacts.
- Treat the timed-out combined Python discovery wrapper as a harness failure;
  it receives no test credit. Use one bounded process per decisive subject.
- Force native binding rebuilds even though current artifact timestamps and
  hashes are internally coherent; timestamp freshness alone is not evidence.
- Force clean-first rebuilds of all three native matrices. The earlier mutation
  runner restored source bytes, but a clean object-file state was not separately
  evidenced after its timeout.

## Exact resume point

Force-clean the Python extension, install the exact new binary plus `libomp`,
prove hashes/provenance/new-symbol discrimination, then run focused/full
Python. Follow with forced-fresh OpenMP MATLAB focused/full on R2024b then
R2025b. Append exact outputs to the run-log after each gate and commit each
completed evidence step immediately.

## Open risks

- All three native dry builds currently need CMake regeneration; generated
  inventory drift must be checked again after each rebuild.
- Python full acceptance intentionally includes the known F39 supply-chain
  inventory failure unless F39 has independently closed.
- MATLAB full acceptance intentionally includes two retained F18 failures;
  any different failed/incomplete name is a regression.
- The claim most likely to fail is that a forced-fresh binding build reproduces
  the exact registered inventory without exposing a stale generated manifest
  or runtime-path dependency.
- CTest's mutable `LastTestsFailed.log` retained a historical F22 failure after
  two current green runs; only the current complete transcript may adjudicate
  a gate.
