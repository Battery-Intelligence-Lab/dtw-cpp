# Handoff - 2026-07-30 - F51 binary checkpoint wire

## Accomplishments

- Two independent F23 preflight audits confirmed that the binary-v1 reader
  allocates from unchecked signed counts, accepts noncanonical structural
  bytes/trailing data, and implements documented little endian with native
  object representations.
- Standardized the stronger coherent 72-byte oracle at SHA-256
  `DC832EDBD214FD847B7EC8BC57884881F1CEA196139FAC7DD5B939D5E7CD1A98`.
- Registered 85 safe deterministic corruptions: 72 truncations, negative k/N,
  each reserved/padding byte, two convergence values, trailing data, bad
  magic/version, and a safely wrong-endian count.
- Added a separate bounded k=257/72-byte allocation-order discriminator using
  the existing global-new probe pattern: inherited one 1028-byte allocation,
  repaired zero, without changing the 85-case corpus.
- Registered exact false/no-throw/unchanged, valid-byte, resave, F17 semantic
  compatibility, real-CLI, native-matrix, documentation, hygiene, and
  two-attempt bands in
  `.claude/baselines/2026-07-30-f51-binary-checkpoint-wire.md`.
- Captured the untouched canonical target baseline: test #82 passed with
  9 assertions / 1 case; CTest names the F17 real-CLI sibling as test #123.
- Captured the untouched F17 real-CLI baseline: exact 12/12 marker, zero skips,
  and 1/1 CTest pass; its production writer/reader preflight covers all seven
  semantic-invalid modes before the CLI rejects them contextually.
- Committed the permanent test/metadata as `6f30665` and executed the inherited
  expected red. It matched the preregistration exactly:
  `false=74 accepted=8 threw=3 unchanged=77/85`,
  `allocations_1028=1`, and the writer raised a non-`dtwc::IOError`.
  Catch2 reported 298 assertions / 2 failed cases; CTest #82 failed because the
  green marker was absent.

## Decisions

- F51 is a safety prerequisite to F23 and preserves binary version 1.
- It owns deterministic wire canonicality only: explicit little-endian codecs,
  count/size-before-allocation, canonical reserved/padding/convergence, exact
  EOF, and strong unchanged-on-false behavior.
- F17's contextual semantic validation stays in the real CLI.
- The later checkpoint/config robustness lens consumes the 85 cases as fixed
  seeds and retains randomized fuzzing, semantic/provenance/authentication,
  atomicity/durability, and CLI/config combinations.
- F52 separately owns MATLAB's false-load error-taxonomy mismatch; F53 owns its
  incomplete/narrowing result conversion and weak field/order parity test.
- Product attempts are capped at two.

## Exact resume point

Implement product attempt 1 in `dtwc/checkpoint.cpp`/`.hpp`, then rebuild and
execute target #82 against the immutable marker before any wider gate. Product
attempts consumed: `0 / 2`.

Rollback is the eventual local F51 commits in reverse order. No remote,
published, data, or operator state changed. The claim most likely to be wrong
is the inherited `74 false / 8 accepted / 3 throws` split; the repaired 85/85
band is fixed.
