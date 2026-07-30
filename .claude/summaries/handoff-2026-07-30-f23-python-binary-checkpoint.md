# Handoff - 2026-07-30 - F23 Python binary checkpoint

## Accomplishments

- Started from clean D2 closure commit `ab08ac1`.
- Read the F23 task, frozen checkpoint/error contract, killed ideas, F17
  decisions and residuals, relevant `LESSONS.md`, live C++ serializer/reader,
  MATLAB binding, Python extension/package surface, parity test, build wiring,
  and current documentation.
- Confirmed the built and installed extension are byte-identical at
  `0E6FCE5C3C312B916E845F1A1D30F2E86F37B663D30809EAE131160562932BDF`
  and neither core nor public package exposes either binary function.
- Independently derived the 72-byte non-degenerate fixture:
  SHA-256
  `DC832EDBD214FD847B7EC8BC57884881F1CEA196139FAC7DD5B939D5E7CD1A98`.
- Recounted the live inventory: contract parity 155, Python 1,041,
  conformance 2, combined 1,043.
- Registered the red-first import/parity failure, exact API and error boundary,
  3-test focused marker, 157 parity, 1,048 full inventory, fresh-extension
  provenance, native matrices, documentation, hygiene, two-attempt cap, and
  rollback in
  `.claude/baselines/2026-07-30-f23-python-binary-checkpoint.md`.
- Closed F51 on product attempt 1. Its exact 85-input gate passed at 298/2,
  F17 retained 12/12, native matrices retained 123/123, 123/123, and 125/125
  with exact 6/9/8 skips, and Arrow readers executed 390/11. F23 may now expose
  the codec.
- A read-only binding audit localized the implementation to two extension
  lambdas, unconditional package exports, one new three-test module, and the
  existing parity inventory. It also identified a mutable-result race unless
  save snapshots the native result before releasing the GIL.

## Decisions

- Python directly binds the frozen C++ binary writer and reader. It does not
  implement a second serializer.
- `load_binary_checkpoint(path)` returns a new `ClusteringResult`; failure raises
  `dtwcpp.IOError`. Directory `load_checkpoint(prob, path) -> bool` is unchanged.
- Both operations release the GIL for filesystem work, and native write failures
  are translated to the frozen typed I/O error.
- Save copies the bound `ClusteringResult` while the GIL is held, then releases
  the GIL around native I/O; load keeps only native state in the release scope
  and throws after the GIL is restored. The failed-load path text is prepared
  as UTF-8 before release.
- F23 preserves binary v1 exactly and performs no contextual N/k or semantic
  validation. The preflight found a narrow unsafe-reader prerequisite now owned
  by F51; F23 product work waits for its deterministic wire-canonicality gate.
  The later checkpoint/config robustness lens retains randomized fuzz,
  semantic/provenance/authentication, and CLI/config combinations.
- Separate source-confirmed MATLAB issues are numbered rather than absorbed:
  F52 owns false-load `dtwc:runtime` versus `dtwc:ioError`; F53 owns incomplete
  or narrowing result conversion and the weak sorted-medoid/two-field test.
- Product attempts are capped at two.

## Exact resume point

Add the permanent three-test Python module and the two parity nodes, commit
them, then execute the inherited stale-extension import/parity red before
touching F23 product code. After repair, clean-first rebuild the extension,
rebuild the sibling CLI, prove built/installed hashes and both new symbols in a
fresh process, and run the focused/parity/full registered gates.
Product attempts consumed: `0 / 2`.

Rollback is the eventual local F23 commits in reverse order. No remote or
operator state has changed. The claim most likely to be wrong is the exact
1,048-node post-F23 full inventory; adjudicate any delta from named collected
nodes rather than changing the band after execution.
