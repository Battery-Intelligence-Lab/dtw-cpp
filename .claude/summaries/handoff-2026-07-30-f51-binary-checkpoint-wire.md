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
- Registered exact false/no-throw/unchanged, valid-byte, resave, F17 semantic
  compatibility, real-CLI, native-matrix, documentation, hygiene, and
  two-attempt bands in
  `.claude/baselines/2026-07-30-f51-binary-checkpoint-wire.md`.

## Decisions

- F51 is a safety prerequisite to F23 and preserves binary version 1.
- It owns deterministic wire canonicality only: explicit little-endian codecs,
  count/size-before-allocation, canonical reserved/padding/convergence, exact
  EOF, and strong unchanged-on-false behavior.
- F17's contextual semantic validation stays in the real CLI.
- The later checkpoint/config robustness lens consumes the 85 cases as fixed
  seeds and retains randomized fuzzing, semantic/provenance/authentication,
  atomicity/durability, and CLI/config combinations.
- Product attempts are capped at two.

## Exact resume point

Commit this preregistration, replace the inherited binary checkpoint test with
the permanent deterministic red-first gate, and execute the expected red before
any product change. Product attempts consumed: `0 / 2`.

Rollback is the eventual local F51 commits in reverse order. No remote,
published, data, or operator state changed. The claim most likely to be wrong
is the inherited `74 false / 8 accepted / 3 throws` split; the repaired 85/85
band is fixed.
