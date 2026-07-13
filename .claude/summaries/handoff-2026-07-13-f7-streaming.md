# Handoff — 2026-07-13 — F7 streaming, Soft-DTW init, and what is still unpinned

Branch `Claude`, three commits ahead of `main`, working tree clean.

| commit | subject |
|---|---|
| `602a3a9` | `fix: translate signed distances before D-sampling` |
| `84693d4` | `feat!: plan Parquet loads from metadata before payload I/O` |
| `57613ef` | `docs: record the streaming contract and its two breaking changes` |

## What happened

Codex ran the 8.2/F7 task and died at the commit step — 8 PIDs, flat CPU, last
write 2026-07-12 07:24. Its work was entirely on disk, so nothing was lost by
killing it. This session audited that work, fixed what was wrong with it, and
landed it.

**F7, as designed, is sound.** `--ram-limit` used to be parsed, printed, and then
applied *after* the whole Parquet file had been decoded — the allocation the cap
exists to prevent had already happened by the time the cap was read. The CLI now
decides resident-vs-streaming from schema plus row-group metadata alone, before
any payload I/O, and the route matrix fails closed.

**Codex also found a second, unrelated bug and fixed it correctly:** raw Soft-DTW
dissimilarities are finite but can be negative, which `std::discrete_distribution`
rejects, so every D-sampling initialiser was broken on a Soft-DTW matrix
containing a negative entry.

## Where its own review was wrong

Codex's reviewer returned PASS. That PASS did not survive an independent pass.

**D1 — fixed here.** `--ram-limit` was left *silently ignored* on every
non-Parquet input. The old "only effective with Parquet" warning was deleted and
nothing replaced it, while the verbose path still printed the cap. `dtwc_cl -i
huge.csv --ram-limit 1G` loaded the entire file and reported a limit it never
applied — F7's own bug class, re-entering through the door F7 had just closed.
The CLI now rejects the flag outright there.

**D2 — documented, not a bug.** `--device cuda` is now rejected for *every*
non-full-sample FastCLARA, including `--method auto` above 5000 series. The
rejection is correct (that schedule is matrix-free after F6, so the GPU matrix was
built and thrown away), but the CHANGELOG had scoped the break to over-budget runs.
It fires on default flags. Now recorded as breaking.

## The lesson worth carrying (already in LESSONS.md)

**My first fix for D1 was itself defective, and the full gate did not catch it.**
I put the guard inside `#ifdef DTWC_HAS_PARQUET`. The canonical gate builds
`DTWC_ENABLE_ARROW=OFF`, so that code does not exist in the tested binary. The new
unit test passed (the test TU defines the macro by including `dtwc_cl.cpp`), the
113-test gate passed, and the real CLI *still* ran CSV + `--ram-limit 1G` to
completion with exit 0. Only driving the actual binary exposed it.

A green unit test on a helper proves the helper works. It never proves the helper
is reachable. Optional-dependency guards hide whole branches from the gate — put
guards that must fire on a build *without* the dependency outside the `#ifdef`.

## Verification (bands registered before each run)

- Canonical gate `build/highs-1151`, re-run on the committed tree:
  **113/113 passed, 0 failed**, 6 capability skips.
- Real `dtwc_cl.exe`, driven directly:
  - CSV + `--ram-limit 1G` → `exit=1`, `Error: --ram-limit caps Parquet series
    materialisation and cannot be honoured for this input; ...`
  - CSV without the flag → `exit=0`, emits `dtwc_labels.csv`, `dtwc_medoids.csv`,
    `dtwc_distance_matrix.csv`, `dtwc_checkpoint.bin`.
- `DTWC_HAS_PARQUET` branch syntax-checked against PyArrow 23.0.1 headers (the
  Arrow-off gate cannot see that code at all).

## The claim most likely to be wrong

**That the streaming path is correct in the configurations I could not execute.**
The Arrow-off build I verified by *running*. The Arrow-on build I verified only by
*compiling*. Codex's resident-vs-streaming parity numbers come from a hand-linked
exe I did not rebuild. F8 and F9 exist precisely to close that hole; until they
are closed, treat streaming parity as **[inferred]**, not confirmed.

## Open, tracked in PLAN.md

- **F8** — resident ≡ streaming byte-identity is pinned by *nothing*. The evidence
  is a one-off manual simulation with no committed test.
- **F9** — `test_io_readers` (348 assertions, all the Parquet hardening) is
  **skipped in the canonical gate**, because that gate is Arrow-off. This is the
  structural weakness that hid D1. Fixing F9 is the highest-value next move: it
  converts a whole untested subsystem into a gated one.
- **F10** — the seeded half of the zero-weight k-means++ fallback is untested, and
  `core::distance_sampling_weights` has no direct unit test.

## Next

Phase 8 still gates the 2.0.0 tag. 8.2's lens list is where the remaining work is —
roughly 21 items barely touched (fuzz harness, aeon cross-oracle, CUDA
compute-sanitizer, mutation sweep, libFuzzer on the parsers, revert-probe, flaky
detector), then 8.3 simplification, 8.5 perf fences, 8.4 exit gate. Phase 9 (WASM
playground) does not gate 2.0.0.

Start with **F9**. Everything else in 8.2 is testing code that at least runs in CI;
F9 is testing code that currently does not.
