# Handoff — 2026-10-11/12 (Mac) — the Mac after the Windows units: gates, Arrow 26, sanitizers, FM's Metal path, timing

## Base

- `design-2.0` local only, ahead of `origin/design-2.0` 6ae7824f by this session's commits (1d96c83c, 292defe3,
  13d23225, c6fb8ab3 and this file's); tree clean. `pb/FM-metal` dcc847be (local) on `origin/pb/FM` 3fe34165.
  Push both (Volkan's). This session's worktrees are removed; the agents' trees stay in the session scratchpad.
- Kept `handoff-2026-10-10-sw-fm-ms.md` (Windows); the 10-09 one deleted.

## Done

- The pull deleted `.claude/CHARTER.md` and `.claude/baselines/` (034d8994 untracked them): restored from `034d8994^`
  (the 10-09 versions). The Windows records after that (`tmp/FM`, `tmp/MS`, `tmp/MM`) are only in
  `C:/D/git/wt/tmp/handoff-mac/records-2026-10-11.zip`, not on the Mac.
- Records (on disk): `baselines/2026-10-11-mac-gates-fm-metal.md` (every result; the pyarrow shim beside it) and
  `2026-10-11-mac-bands.md` (§A, §B, §E registered 10-11 23:20 and D1 10-12 05:00, each before its runs).
- 292defe3: the parity test's float32 stream limit fits libc++'s 24-byte `std::string`. 13d23225: the codegen gate is
  not registered under a sanitizer. dcc847be (pb/FM-metal): FM's Metal float32 fill pinned, in RAM and mapped.
- PLAN: the macOS row ☑, FM's Mac evidence, BF's candidate line; MAP §4 counts, §7's gate note; a lesson.

## Verified by me (verbatim lines in the record)

- 6ae7824f: build 0 warnings; ctest 95 (CUDA the MAY_SKIP); codegen 61 loops 0 calls; Metal 2168 + 169 assertions;
  docs gates PASS; conformance D-19's ulp only; pytest 904/11/0; matlab_suite 141/141 (R2026a). + pb/FM: 0 warnings,
  ctest 95, pytest 907/11/0, matlab_suite 142/142; the CLI's Metal float32 matrix = float64 narrowed (160,000 entries).
- After 292defe3 and 13d23225: Release ctest 95/95, the Arrow 26 tree 98/98 (CUDA skipped), "F8_PARITY …
  parity=9/9"; an ASan tree lists 94 tests. dcc847be's tests re-run: 21972 and 10072 assertions.
- Timing E, every run at load1 < 1.5, outputs identical in every run (head 6ae7824f / base 263e0e24 speed): E1 equal
  fill, 18 threads, 0.944 and 0.945 — FALSIFIED (band ≥ 0.95); E2 ragged fill 3.888; E3a one thread 0.995; E3b 4.686;
  E4 FasterPAM 2.057; E5 FastCLARA 3.072: PASS. E6 NOT MEASURED (no silhouette time). D1: BF at 32 chunks per thread
  against head 1.038 — inconclusive (registered: ≥ 1.04 holds, < 1.02 not supported).
- Correction to the 10-10 handoff: W4d, W13a, W10 and GC merged before 263e0e24, whose Mac ctest ran the Metal tests.

## Reported by agents, unverified by me

- A (pyarrow 26 shim): `test_io_readers` 67761 assertions; the stream's peak footprint ≤ its limit, 4/4, streamed
  outputs byte-identical 8/8. B: ASan + UBSan, 0 reports in two serial runs, markers shown to bite. C: the mutations
  that fail dcc847be's tests. E: the kit, flags, data and smoke.

## Decisions

- Volkan this session: "Do if there is anything you could do on a mac if not then handoff".
- Awaiting him: BF on the Mac (E1 −5.5 % at 18 threads: a unit for chunking or largest-first order, timed on both
  machines?); where FM is timed; MM's finding 7 (cluster() through C++ `run` in Python and MATLAB?); the records zip.

## Next steps

1. Volkan: push `design-2.0` and `pb/FM-metal`.
2. Windows: `test_fast_clara_parquet_parity` in the Arrow 23 tree (MS STL's float32 window is inferred); FM's timing,
   then merge pb/FM and pb/FM-metal; a CUDA counterpart of the float32 GPU test (none exists).
3. MM round 4 (PLAN MM), with the zip; HG after MM.
4. Candidates: BF's chunks (PLAN); one repo script for the pyarrow CMake shim (both machines hand-make one); CMake
   4.4's CMP0219 in `test_cli_config_formats.cmake:131` (passes either way); a silhouette time in `-v`.

## Open questions

- From 10-10: compacting `ext4.vhdx`; the GitHub history purge; RA's notice; which `.claude` files to untrack.
- The Mac's charter is the 10-09 copy; the Windows one may have later entries (DECISIONS §3 has the rulings).

## Status honesty

- Mac only. Timing ran 23:49–05:41 on a machine idling near the gate (Sophos, Tanium; AC, then battery for E2–E5 and
  E1's retry); medians of five interleaved pairs; wall-clock, no counters. Not run: Windows, Linux, CI, Arrow 23,
  CUDA, a second D1 attempt, largest-first order, BF's change on Windows.
