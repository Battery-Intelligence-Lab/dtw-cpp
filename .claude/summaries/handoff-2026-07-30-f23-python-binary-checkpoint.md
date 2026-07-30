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
- Commit `90fa58d` adds the permanent red-first test module and the two parity
  nodes. Against the unchanged stale extension, the focused module failed
  during import with zero subject tests and no marker; the full parity gate was
  exactly 155 passed / 2 expected missing-symbol failures. The registered red
  is therefore confirmed without consuming a product attempt.
- Product attempt 1 compiled and its fresh-artifact provenance passed, including
  the intended `_hpc` CLI route, but the focused pytest command errored before
  the wire subject because the parent of its nested `--basetemp` did not exist.
  Exact outcome: 2 passed / 1 setup error / no marker. The unchanged product
  proceeds to attempt 2 only after explicitly creating and verifying that
  parent; the registered band is unchanged.
- Attempt 2 ran the unchanged product after verifying the basetemp parent and
  passed exactly 3/3 with the sole exact `F23_PYTHON_CHECKPOINT` marker. Commit
  `5bf517f` contains the two GIL-safe native lambdas, unconditional exports, and
  Unreleased changelog entry.

## Exact resume point

Run parity and the combined inventory with both CLI environment routes pinned
to the already rebuilt
`build/cfg-gate-normal/bin/dtwc_cl.exe`.
Product attempts executed: `2 / 2`; attempt 2 passed.

Rollback is the eventual local F23 commits in reverse order. No remote or
operator state has changed. The claim most likely to be wrong is the exact
1,048-node post-F23 full inventory; adjudicate any delta from named collected
nodes rather than changing the band after execution.
